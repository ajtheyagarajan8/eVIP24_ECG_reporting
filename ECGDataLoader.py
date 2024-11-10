import os
import gc
from pathlib import Path
import wfdb
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io
from sklearn.model_selection import train_test_split
import random



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

class ECGDataLoader(Dataset):
    def __init__(self, split_subjects: list,
                 batch_size: int = 16,
                 shuffle: bool = True,
                 ecg_subset: str = "mimic-iv-ecg_complete"):
        self.split_subjects = split_subjects
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ecg_subset = ecg_subset
        # Initialize file paths and parameters
        self.mimiciv_root_dir = Path.cwd().joinpath("MIMIC-IV-data")
        
        self.mimic_ecg_matched_path = self.mimiciv_root_dir / "mimic-iv-ecg-matched-subset"
        self.ecg_meta_path = self.mimic_ecg_matched_path / "meta_files"
        
        self.ecg_path = self.mimic_ecg_matched_path / self.ecg_subset
        self.ecg_waveforms_path = self.ecg_path / "files"        
        
        self.clinical_path = self.mimiciv_root_dir / "mimic-iv-2.2"
        self.hosp_path = self.clinical_path / "hosp"

        self.machine_reports = pd.read_csv(
            self.ecg_meta_path / "machine_measurements.csv",
            na_filter=False,
            parse_dates=["ecg_time"]
        )
        
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
        
        self.record_list['path'] = self.record_list['path'].apply(Path)
        
        #print(f"len(record_list): {len(self.record_list)}, len(machine_reports): {len(self.machine_reports)}")
        self.record_list = self.record_list[self.record_list['subject_id'].isin(self.split_subjects)]
        self.machine_reports = self.machine_reports[self.machine_reports['subject_id'].isin(self.split_subjects)]
        #print(f"len(record_list): {len(self.record_list)}, len(machine_reports): {len(self.machine_reports)}")
        
        '''
        under_sample_factor = self.batch_size*2 #for testing
        self.record_list = self.record_list[:under_sample_factor]
        self.machine_reports = self.machine_reports[self.machine_reports['study_id'].isin(self.record_list['study_id'])]
        #print(f"len(record_list): {len(self.record_list)}, len(machine_reports): {len(self.machine_reports)}")
        '''
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
            transforms.Resize((512, 512))  # Resize all images to 256x256
        ])

    def __len__(self):
        # Dataset size based on the number of records in machine reports
        return len(self.machine_reports)
    
    def __getitem__(self, idx):
        # Fetch the specific row from machine_reports
        while True:
            try:
                record = self.machine_reports.iloc[idx]
                #print(f"record: {record}")
                # Generate ECG image and report on demand
                study_id = record["study_id"]
                #print(f"study_id: {study_id}")
                row = self.record_list[self.record_list['study_id'] == study_id]
                #print(f"row: {row}")
                path_stem = row.iloc[0]['path']
                #More work must be done to convers study_id to file path
                file_path = self.ecg_path / path_stem
                ecg_image = self.get_ecg_image(file_path)
                if ecg_image is None:
                    raise RuntimeError(f"Failed to load ECG image for {path_stem}")
                report = self.get_report(record)
                break
            except Exception as e:
                print(f"Error loading sample at {idx}: {e}")
                idx = random.randint(0, len(self) - 1)

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


    def get_ecg_image(self, file_path):
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
    tvs = TrainValSplit()
    tr, vl = tvs.train_val_split()
    
    
    data_loader = ECGDataLoader(tr)
    print(len(data_loader))
    dl = data_loader.get_dataloader()
    lim = 1
    cur = 1
    for b in dl:
        show_ecg_batch(b['waveform'], batch_size=1)
        if cur > lim:
            break
        cur+=1