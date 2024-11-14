import os
import gc
from pathlib import Path
import wfdb
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io
from sklearn.model_selection import train_test_split
import random
import ecg_plot
import numpy as np
from scipy.signal import decimate
from tqdm import tqdm

from io import BytesIO

#from concurrent.futures import ThreadPoolExecutor
#from threading import Lock
#plot_lock = Lock()
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
Python data loader that retrieves waveform files downloaded from
https://physionet.org/content/mimic-iv-ecg-demo/0.1/ and converts them
into png images. It also generates text-based reports for each waveform
record based on diagnostic data from the MIMIC-IV-ECG and MIMIC-IV
clinical databases.

If a different database is used in future replace self.mimicDataPath with the new path.

Other databases that can be used https://physionet.org/content/mimic-iv-ecg/1.0/files/
"""

def collate_fn(batch):
    images = batch['waveform']
    reports = batch['text']
    images = torch.stack(images, dim=0)  # Stack all images in a batch
    return {'waveform': images, 'text': reports}


mimiciv2ptbxl_signal_adjustments = {
    "offset": np.array([-119, -55, 64, 86, -91, 4, -69, -31, 0, -26, -39, -79]),
    "gain": 5
}

default_ecg_config= {"type": "image",
                        "dimensions": (300,300),
                        "signal_adjustments": {},
                        "ecg_sampling_rate": 500 #2500
                    }

class ECGDataLoader(Dataset):
    def __init__(self, split_subjects: list,
                 dynamic_loading: bool = True,
                 ecg_config: dict = default_ecg_config,
                 batch_size: int = 16,
                 shuffle: bool = True,
                 ecg_subset: str = "mimic-iv-ecg_complete"):
        
        self.split_subjects = split_subjects
        self.dynamic_loading = dynamic_loading
        self.ecg_config = ecg_config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ecg_subset = ecg_subset
       
        # Initialize file paths
        self.init_paths()
        print("Paths initialized")

        #check if dynamic/static loading is possible or not
        self.sanity_checks()
        print("Sanity Checked")

        self.ecg_sampling_rate = self.ecg_config["ecg_sampling_rate"]
        self.gain = None
        self.offset = None
        if "signal_adjustments" in self.ecg_config:
            if "offset" in self.ecg_config["signal_adjustments"]: self.offset = self.ecg_config["signal_adjustments"]["offset"]
            if "gain" in self.ecg_config["signal_adjustments"]: self.gain = self.ecg_config["signal_adjustments"]["gain"]

        self.machine_reports = pd.read_csv(
            self.ecg_meta_path / "machine_measurements.csv",
            na_filter=False,
            parse_dates=["ecg_time"]
        )

        self.compile_record_list()
        print("Record list compiled")
        
        #self.under_sample_for_testing(under_sample_to=self.batch_size*2)
        
        self.admissions = pd.read_csv(
            self.hosp_path / "admissions.csv",
            usecols=[
                "subject_id", "hadm_id", "admittime",
                "dischtime", "edregtime", "edouttime"
            ],
            parse_dates=["admittime", "dischtime", "edregtime", "edouttime"]
        )
        self.diagnoses_icd = pd.read_csv(
            self.hosp_path / "diagnoses_icd.csv",
            usecols=["subject_id", "hadm_id", "icd_code"]
        )
        self.d_icd_diagnoses = pd.read_csv(
            self.hosp_path / "d_icd_diagnoses.csv",
            usecols=["icd_code", "long_title"]
        )
        
        # Define transform to convert matplotlib plot to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize((512, 512))
        ])

    def init_paths(self):
        self.mimiciv_root_dir = Path.cwd().joinpath("MIMIC-IV-data")
        
        self.mimic_ecg_matched_path = self.mimiciv_root_dir / "mimic-iv-ecg-matched-subset"
        self.ecg_meta_path = self.mimic_ecg_matched_path / "meta_files"
        
        self.ecg_path = self.mimic_ecg_matched_path / self.ecg_subset
        self.ecg_waveforms_path = self.ecg_path / "files"        
        
        self.clinical_path = self.mimiciv_root_dir / "mimic-iv-2.2"
        self.hosp_path = self.clinical_path / "hosp"

    def sanity_checks(self):
        file_types = []
        self.can_do_dynamic_loading = False
        self.can_do_static_loading = False
        file_acquired = False
        for gr in self.ecg_waveforms_path.iterdir():
            if gr.is_file():continue
            for sb in gr.iterdir():
                if sb.is_file():continue
                for st in sb.iterdir():
                    if st.is_file():continue
                    for dfile in st.iterdir():
                        file_types.append(dfile.suffix)
                    file_acquired = True
                    break
                if file_acquired: break
            if file_acquired: break

                
        if ".hea" in file_types or ".dat" in file_types: 
            self.can_do_dynamic_loading = True
        if ".png" in file_types or ".jpg" in file_types or ".npy" in file_types:
            self.can_do_static_loading = True
        
        if self.dynamic_loading and not self.can_do_dynamic_loading:
            raise RuntimeError(f"ECG subset in {self.ecg_subset} is not meant for dynamic loading")
        if not self.dynamic_loading and not self.can_do_static_loading:
            raise RuntimeError(f"ECG subset in {self.ecg_subset} is not meant for static loading")

    def compile_record_list(self):
        #match machine_reports with the file paths in record_list.csv
        if self.ecg_subset != "mimic-iv-ecg_complete":
            #for subsets other than mimic-iv-ecg_complete you have to make your own record_list.csv
            if (self.ecg_path / "record_list.csv").exists():
                self.record_list = pd.read_csv(self.ecg_path / "record_list.csv",na_filter=False)
            else: #if you don't have a record list for your ecg_subset bob will build one
                print(f"Building record list because {self.ecg_path} hasn't got one")
                from BuildRecordList import BuildRecordList
                bob = BuildRecordList(self.ecg_subset)
                bob.build()
                self.record_list = pd.read_csv(self.ecg_path / "record_list.csv",na_filter=False)
                print(f"Finished building records list")
            #self.machine_reports = self.machine_reports[self.machine_reports['study_id'].isin(self.record_list['study_id'])]
        else: #there shouldn't be a record_list in mimic-iv-ecg_complete since we just use the official record list in meta_files
            self.record_list = pd.read_csv(
                self.ecg_meta_path / "record_list.csv",
                na_filter=False,
            )
        #remove any missing studies from the record
        missing_studies_file = self.ecg_path / "missing_studies.txt"
        if missing_studies_file.exists():
            print("Detected missing_studies.txt file, now filtering out missing studies")
            with open(missing_studies_file, 'r') as f:
                file_paths_set = set(line.strip() for line in f)
            self.record_list = self.record_list[~self.record_list['path'].isin(file_paths_set)].reset_index(drop=True)
        self.record_list['path'] = self.record_list['path'].apply(Path)
        
        #only get subjects in current split
        self.record_list = self.record_list[self.record_list['subject_id'].isin(self.split_subjects)]
        self.machine_reports = self.machine_reports[self.machine_reports['subject_id'].isin(self.record_list['subject_id'])].reset_index(drop=True)
        

    def under_sample_for_testing(self, under_sample_to):
        self.record_list = self.record_list[:under_sample_to]
        self.machine_reports = self.machine_reports[self.machine_reports['study_id'].isin(self.record_list['study_id'])]

    #this is just used to generate ecg dataset with images, not part of the dataloader used during training
    def load_and_save_slow(self):
        #save a new directory in mimic-iv-ecg-matched-subset for the ecg images
        missing_studies = set()
        with open(self.ecg_path / "missing_studies.txt", "r") as f:
            for line in f:
                missing_studies.add(line.strip())
        total_groups = len(list(self.ecg_waveforms_path.iterdir()))
        with tqdm(total=total_groups, desc="Groups", leave=True) as pbar_groups:
            for gr in self.ecg_waveforms_path.iterdir():
                if gr.is_file(): continue
                total_subjects = len(list(gr.iterdir()))
                for sb in gr.iterdir():
                    if sb.is_file():continue
                    for st in sb.iterdir():
                        if st.is_file(): continue
                        inner_name = str(st.name[1:])
                        if str(Path(inner_name).parts[-5:]) in missing_studies: continue
                        dat_file = st / (inner_name+".dat")
                        hea_file = st / (inner_name+".hea")
                        if not (dat_file.exists() and hea_file.exists()): continue 
                        typeless_file = st / inner_name
                        _, ecg_image = self.get_standard_ecg_image(typeless_file)
                        new_ecg_subset_dir = self.ecg_subset+"_300x300_images"
                        save_file = str(typeless_file).replace(self.ecg_subset, new_ecg_subset_dir)+".png"
                        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
                        ecg_image.save(save_file)
                pbar_groups.update(1)                    


    def process_study(self, st, new_ecg_subset_dir):
        #used for threading
        if st.is_file(): return
        inner_name = str(st.name[1:])
        dat_file = st / (inner_name + ".dat")
        hea_file = st / (inner_name + ".hea")
        typeless_file = st / inner_name
        ecg_image = self.get_standard_ecg_image_fast(typeless_file)
        
        # Define the save path and create directories if necessary
        save_file = str(typeless_file).replace(self.ecg_subset, new_ecg_subset_dir) + ".png"
        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        ecg_image.save(save_file)
            
    def load_and_save_threading_fast(self):
        #save a new directory in mimic-iv-ecg-matched-subset for the ecg images
        # Load the missing studies list
        missing_studies = set()
        with open(self.ecg_path / "missing_studies.txt", "r") as f:
            for line in f:
                missing_studies.add(line.strip())
        
        # Define the new subset directory name
        new_ecg_subset_dir = self.ecg_subset + "_300x300_images"
        total_groups = len(list(self.ecg_waveforms_path.iterdir()))
        
        # Use a single ProcessPoolExecutor for the entire function to reduce overhead
        with ProcessPoolExecutor(max_workers=20) as executor:
            with tqdm(total=total_groups, desc="Groups", leave=True) as pbar_groups:
                # Iterate over each group
                for gr in self.ecg_waveforms_path.iterdir():
                    if gr.is_file():continue
                    # Process each subject within the group
                    for sb in gr.iterdir():
                        if sb.is_file():continue

                        # Gather all valid studies within this subject
                        studies = [
                            st for st in sb.iterdir()
                            if st.is_dir() and str(Path(*((st / st.name[1:]).parts[-5:]))) not in missing_studies
                        ]

                        # Submit all studies for processing in a batch
                        futures = [executor.submit(self.process_study, st, new_ecg_subset_dir) for st in studies]

                        # Use as_completed to handle the futures as they complete
                        for future in as_completed(futures):
                            # Retrieve results if needed or handle exceptions
                            try:
                                future.result()
                            except Exception as e:
                                print(f"An error occurred: {e}")

                    # Update progress for each group
                    pbar_groups.update(1)

    def load_and_save_threading(self):
        #save a new directory in mimic-iv-ecg-matched-subset for the ecg images
        missing_studies = set()
        with open(self.ecg_path / "missing_studies.txt", "r") as f:
            for line in f:
                missing_studies.add(line.strip())
        # Define the new subset directory name
        new_ecg_subset_dir = self.ecg_subset + "_300x300_images"
        total_groups = len(list(self.ecg_waveforms_path.iterdir()))
        with tqdm(total=total_groups, desc="Groups", leave=True) as pbar_groups:
            for gr in self.ecg_waveforms_path.iterdir():
                if gr.is_file(): continue
                # Process each subject within a group
                for sb in gr.iterdir():
                    if sb.is_file(): continue
                    studies = [st for st in sb.iterdir() if st.is_dir() and str(Path(*((st / st.name[1:]).parts[-5:]))) not in missing_studies]
                    with ProcessPoolExecutor(max_workers=12) as executor:
                        futures = [executor.submit(self.process_study, st, new_ecg_subset_dir) for st in studies]
                        for future in futures:
                            future.result()
                    # with ThreadPoolExecutor(max_workers=12) as executor:
                    #     futures = [
                    #         executor.submit(self.process_study, st, new_ecg_subset_dir)
                    #         for st in studies
                    #     ]
                    #     for future in futures:
                    #         future.result()
                pbar_groups.update(1)
    
    def pregen_reports(self):
        #this will only be called for mimic-iv-ecg_complete
        report_records = []
        with tqdm(total=len(self.machine_reports), desc="Reports", leave=True) as pbar_groups:
            for _, record in self.machine_reports.iterrows():
                try:
                    record_row = self.record_list[self.record_list['study_id'] == record['study_id']]
                    path_stem = record_row.iloc[0]['path']
                    report = self.get_report(record)
                    report_records.append({"study_path": path_stem, "report": report})
                    pbar_groups.update(1)
                except: 
                    cur_report_df = pd.DataFrame(report_records)
                    cur_report_df.to_csv("all_reports.csv", index=False)
        cur_report_df = pd.DataFrame(report_records)
        cur_report_df.to_csv("all_reports.csv", index=False)
                
    def __len__(self):
        # Dataset size based on the number of records in machine reports
        return len(self.machine_reports)
    
    def __getitem__(self, idx):
        while True:
            try:
                record = self.machine_reports.iloc[idx]
                study_id = record["study_id"]
                record_row = self.record_list[self.record_list['study_id'] == study_id]
                path_stem = record_row.iloc[0]['path']
                if self.dynamic_loading:
                    file_path = self.ecg_path / path_stem
                    ecg_image, _ = self.get_standard_ecg_image(file_path)
                else:
                    image_path = Path(str(self.ecg_path / path_stem)+".png")
                    image = Image.open(image_path).convert("RGB") 
                    ecg_image = self.transform(image)
                break
            except Exception as e:
                print(f"Error loading sample at {idx}: {e}")
                idx = random.randint(0, len(self) - 1)
        report = self.get_report(record)    
        sample = {
            "waveform": ecg_image,
            "text": report
        }
        return sample

    def get_dataloader(self) -> DataLoader:
        return DataLoader(self, 
                          batch_size=self.batch_size, 
                          shuffle=self.shuffle, 
                          collate_fn=None)

    def get_ecg_signal(self, file_path):
        try:
            signal = wfdb.rdsamp(file_path)[0] #numpy (5000, 12)
        except Exception as e:
            print(f"Error loading ECG signal: {file_path}")
            return None
        if self.gain is not None:
            signal = signal * 5
        if self.offset is not None:
            signal = signal - self.offset
        return signal

    def expand_to_square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def get_standard_ecg_image_fast(self, file_path):
        signal = self.get_ecg_signal(file_path)
        net_input_size = 300
        buffer = BytesIO()
        fig, ax = plt.subplots()
        ecg_plot.plot(np.transpose(signal), sample_rate = self.ecg_sampling_rate, title = ' ', style = 'bw')
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close('all')
        del fig, ax
        buffer.seek(0)
        image = Image.open(buffer)
        ecg_image = image.resize((300, 300))
        return ecg_image


    def get_standard_ecg_image(self, file_path):
        signal = self.get_ecg_signal(file_path)
        net_input_size = 300
        
        ecg_plot.plot(np.transpose(signal), sample_rate = self.ecg_sampling_rate, title = ' ', style = 'bw')
        '''
        if str(file_path.name) == "48446569":
            #sr = 500
            sr = 100
        else:
            #sr = 2500
            sr=500
        ecg_plot.plot(np.transpose(signal), sample_rate = sr, title = ' ', style = 'bw')
        secs  = len(np.transpose(signal)[0])/sr
        x_max = 2*secs
        print(f"x_max: {x_max}")
        '''
        #ecg_plot.show()
        
        ecg_plot.save_as_png('temp', './')
        image = Image.open('./' + 'temp.png')
        
        #fig, axes = plt.subplots(1)
        #axes.imshow(image)
        #squared_image = self.expand_to_square(image, (0,0,0))
        #new_image = squared_image.resize((net_input_size, net_input_size))
        #print(str(file_path.name))
        #image.save(str(file_path.name) + '.png')
        squared_image = image.resize((300,300))
        #squared_image.save(str(file_path.name) + ".png")
        
        return self.transform(squared_image), squared_image

    def get_nonstandard_ecg_image(self, file_path):
        """Load and return the ECG waveform as a tensor."""
        #Performs dynamic loading, saves the image to a buffer buf
        dpi = 300
        # width_pixels = 2400
        # height_pixels = 1200
        # fig_width = width_pixels/dpi
        # fig_height = height_pixels/dpi
        try:
            rd_record = wfdb.rdrecord(file_path)
            fig = wfdb.plot_wfdb(record=rd_record, figsize=(24, 18), title=None, ecg_grids='all', return_fig=True)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
            return self.transform(image)
        except Exception as e:
            print(f"Error loading ECG image: {e}")
            return None

    def get_report(self, record):
        """Generate report from the diagnostic data for a single record."""
        report = []
        report.append(self.get_measurements(record))
        report.append(self.get_machine_reports(record))
        report.append(self.get_diagnoses(record, self.admissions, self.diagnoses_icd, self.d_icd_diagnoses))
        return " ".join(report)

    def get_measurements(self, rec):
        """
        Returns a string of compiled summary measurements from a
        specific ECG record. The measurements describe the average
        RR interval, as well as the average onset time, end time, and
        electrial axis for the P-wave, QRS complex, and T-wave.
        """
        return (
            f"This waveform record has the following summary measurements: "
            f"The average RR interval is {rec['rr_interval']}msec. "
            f"On average, "
            f"the onset of the P-wave occurs at {rec['p_onset']}msec "
            f"and the end of the P-wave at {rec['p_end']}msec; "
            f"the onset of the QRS complex occurs at {rec['qrs_onset']}msec "
            f"and the end of the QRS complex at {rec['qrs_end']}msec; "
            f"and the end of the T-wave occurs at {rec['t_end']}msec. "
            f"The electrical axis is "
            f"{rec['p_axis']} degrees for the P-wave, "
            f"{rec['qrs_axis']} degrees for the QRS complex, and "
            f"{rec['qrs_axis']} degrees for the T-wave."
        )

    def get_machine_reports(self, record):
        """
        Returns a string of compiled machine-generated reports from a
        specific ECG record.
        """
        # Extract all machine-generated reports for the given record
        reports = [x for x in record.loc["report_0":"report_17"] if x != ""]
        # TODO: filter out reports with invalid or missing data?

        if (len(reports) == 0):
            return "No ECG reports were recorded."

        return (
            "The following machine-generated ECG "
            f"report{' was' if len(reports) == 1 else 's were'} recorded: "
            f"{'; '.join(reports).rstrip('.')}.")

    def get_diagnoses(self, record, adm, diagnoses_icd, d_icd_diagnoses):
        """
        Returns a string of arrhythmia diagnoses recorded for a specific
        patient and hospital admission. A diagnosis is represented by an
        ICD-9 or ICD-10 code and its corresponding title/description.
        """
        empty_str = "No diagnosis was recorded for this hospital admission."
        
        # Get patient admission that overlaps with time of ECG recording
        hadm_id_cell = adm.loc[
            (adm["subject_id"] == record["subject_id"])
            & (
                ((adm["admittime"] <= record["ecg_time"]) & (record["ecg_time"] <= adm["dischtime"]))
                | ((adm["edregtime"] <= record["ecg_time"]) & (record["ecg_time"] <= adm["edouttime"]))
            ),
            "hadm_id"]

        if hadm_id_cell.empty:
            return empty_str

        # Get list of all ICD codes related to an arrhythmia diagnosis
        #   ICD-9: all 427n codes except 4275; 7850
        #   ICD-10: all I47n and I48n codes; all I49n codes except I495
        #   (where n is 0 or more digits)
        hadm_id = hadm_id_cell.item()
        substr = r"^427(?:[^5]|\Z)|^7850|^I47|^I48|^I49(?:[^5]|\Z)"
        ecg_icds = diagnoses_icd.loc[
            (diagnoses_icd["subject_id"] == record["subject_id"])
            & (diagnoses_icd["hadm_id"] == hadm_id)
            & diagnoses_icd["icd_code"].str.contains(substr),
            "icd_code"
        ].values.flatten().tolist()

        if len(ecg_icds) == 0:
            return empty_str

        # Match ICD codes with their human-readable titles/descriptions
        ecg_diagnoses = []
        for icd_code in ecg_icds:
            icd_name = d_icd_diagnoses.loc[
                d_icd_diagnoses["icd_code"] == icd_code, "long_title"
            ].item()
            version = "10" if icd_code[0] == "I" else "9"
            ecg_diagnoses.append(f"{icd_name} (ICD-{version} {icd_code})")
        
        return (
            "During this hospital admission, the patient received the "
            f"following ICD diagnos{'i' if len(ecg_icds) == 1 else 'e'}s: "
            f"{'; '.join(ecg_diagnoses)}.")

class TrainValSplit:
    def __init__(self, ecg_subset: str = "mimic-iv-ecg_complete"):
        self.ecg_subset = ecg_subset
        # Initialize file paths and parameters
        self.mimiciv_root_dir = Path.cwd().joinpath("MIMIC-IV-data")
        
        self.mimic_ecg_matched_path = self.mimiciv_root_dir / "mimic-iv-ecg-matched-subset"
        self.ecg_meta_path = self.mimic_ecg_matched_path / "meta_files"
        
        self.ecg_path = self.mimic_ecg_matched_path / self.ecg_subset
        
        #match machine_reports with the file paths in record_list.csv
        if (self.ecg_path / "record_list.csv").exists():
            #for subsets other than mimic-iv-ecg_complete you have to make your own record_list.csv
            if self.ecg_path / "record_list.csv".exists():
                self.record_list = pd.read_csv(self.ecg_path / "record_list.csv",na_filter=False)
            else: #if you don't have a record list for ecg_subset bob will build one
                from BuildRecordList import BuildRecordList
                bob = BuildRecordList(self.ecg_subset)
                bob.build()
                self.record_list = pd.read_csv(self.ecg_path / "record_list.csv",na_filter=False)
            self.machine_reports = self.machine_reports[self.machine_reports['study_id'].isin(self.record_list['study_id'])]
        else: #there shouldn't be a record_list in mimic-iv-ecg_complete since we just use the official record list in meta_files
            self.record_list = pd.read_csv(
                self.ecg_meta_path / "record_list.csv",
                na_filter=False,
            )
        
        self.subject_ids = self.record_list['subject_id'].unique()
    
    def train_val_split(self, val_ratio: float = 0.2, random_seed: int = 42):
        train_subjects, val_subjects = train_test_split(
            self.subject_ids,
            test_size=val_ratio,
            random_state=random_seed
        )
        return train_subjects, val_subjects
    


def show_ecg_batch(batch_images, batch_size=4, figsize=(15, 15)):
    """
    Display a batch of ECG images.

    Args:
        batch_images (torch.Tensor): A batch of images from the DataLoader.
        batch_size (int): Number of images to display in one row/column of the grid.
        figsize (tuple): Size of the matplotlib figure.
    """
    # Convert tensor to numpy array and remove any channel dimension
    batch_images = batch_images.detach().cpu()
    
    # Set up plot grid
    fig, axes = plt.subplots(batch_size, figsize=figsize)
    for i in range(batch_size):
        image = batch_images[i].permute(1, 2, 0).numpy()
        axes.imshow(image)  # Display image in grayscale
        axes.axis("off")  # Hide axes for clarity

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    #tvs = TrainValSplit()
    #tr, vl = tvs.train_val_split()
    tr = [10000032, 10000117, 10000285]
    data_loader = ECGDataLoader(split_subjects=tr, dynamic_loading=True)# ecg_subset="mimic-iv-ecg_complete_300x300_narrow_images")
    print(len(data_loader))
    #data_loader.load_and_save_threading_faster()
    data_loader.pregen_reports()
    #dl = data_loader.get_dataloader()
    #for b in dl:
        #show_ecg_batch(b['waveform'], batch_size=1)


'''
def load_and_save(self):
    total_groups = len(list(self.ecg_waveforms_path.iterdir()))
    with tqdm(total=total_groups, desc="Groups", leave=True) as pbar_groups:
        for gr in self.ecg_waveforms_path.iterdir():
            if gr.is_file(): continue
            total_subjects = len(list(gr.iterdir()))
            with tqdm(total=total_subjects, desc="Subjects", leave=False) as pbar_subjects:
                for sb in gr.iterdir():
                    if sb.is_file():continue
                    total_studies = len(list(sb.iterdir()))
                    with tqdm(total=total_studies, desc="Studies", leave=False) as pbar_studies:
                        for st in sb.iterdir():
                            if st.is_file(): continue
                            inner_name = str(st.name[1:])
                            dat_file = st / (inner_name+".dat")
                            hea_file = st / (inner_name+".hea")
                            if not (dat_file.exists() and hea_file.exists()): continue 
                            typeless_file = st / inner_name
                            
                            signal = wfdb.rdsamp(typeless_file)[0] #numpy (5000, 12)
                            net_input_size = 300
                            ecg_plot.plot(np.transpose(signal), sample_rate = self.ecg_sampling_rate, title = ' ', style = 'bw')
                            ecg_plot.save_as_png('temp', './')
                            image = Image.open('./' + 'temp.png')
                            ecg_image = image.resize((300,300))

                            new_ecg_subset_dir = self.ecg_subset+"_300x300_narrow_images"
                            save_file = str(typeless_file).replace(self.ecg_subset, new_ecg_subset_dir)+".png"
                            Path(save_file).parent.mkdir(parents=True, exist_ok=True)
                            ecg_image.save(save_file)
                            pbar_studies.update(1)
                    pbar_subjects.update(1)
            pbar_groups.update(1) 
def ecg_plot(
        ecg, 
        sample_rate    = 500, 
        title          = 'ECG 12', 
        lead_index     = lead_index, 
        lead_order     = None,
        style          = None,
        columns        = 2,
        row_height     = 6,
        show_lead_name = True,
        show_grid      = True,
        show_separate_line  = True,
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on 
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order 
        columns    : display columns, defaults to 2
        style      : display style, defaults to None, can be 'bw' which means black white
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
    """

    if not lead_order:
        lead_order = list(range(0,len(ecg)))
    secs  = len(ecg[0])/sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    # display_factor = 2.5
    display_factor = 1
    line_width = 0.5
    fig, ax = plt.subplots(figsize=(secs*columns * display_factor, rows * row_height / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0,
        left   = 0,  # the left side of the subplots of the figure
        right  = 1,  # the right side of the subplots of the figure
        bottom = 0,  # the bottom of the subplots of the figure
        top    = 1
        )

    fig.suptitle(title)

    x_min = 0
    x_max = columns*secs
    y_min = row_height/4 - (rows/2)*row_height
    y_max = row_height/4

    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    else:
        color_major = (1,0,0)
        color_minor = (1, 0.7, 0.7)
        color_line  = (0,0,0.7)

    if(show_grid):
        ax.set_xticks(np.arange(x_min,x_max,0.2))    
        ax.set_yticks(np.arange(y_min,y_max,0.5))

        ax.minorticks_on()
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)


    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height/2) * ceil(i%rows)
                # if (y_offset < -5):
                #     y_offset = y_offset + 0.25

                x_offset = 0
                if(c > 0):
                    x_offset = secs * c
                    if(show_separate_line):
                        ax.plot([x_offset, x_offset], [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3], linewidth=line_width * display_factor, color=color_line)

         
                t_lead = lead_order[c * rows + i]
         
                step = 1.0/sample_rate
                if(show_lead_name):
                    ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor)
                ax.plot(
                    np.arange(0, len(ecg[t_lead])*step, step) + x_offset, 
                    ecg[t_lead] + y_offset,
                    linewidth=line_width * display_factor, 
                    color=color_line
                    )
'''