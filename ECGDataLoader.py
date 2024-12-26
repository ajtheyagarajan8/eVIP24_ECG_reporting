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
import traceback
from io import BytesIO
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import threading

#USE CASES
#UC1.1 Pregenerate ECG images
## Instantiate ECGDataBase
## Instantiate ECGDataPreparation, passing it ECGDataBase 
#UC1.2 Pregenerate ECG report
## Instantiate ECGDataBase
## Instantiate ECGDataPreparation, passing it ECGDataBase 
#In future we may add more features to ECGPreparation to be more flexible when we prepare a subset of the data

#UC2.1 Train-val split 
## Instantiate ECGDataBase

#UC3.1 Build a train/val dataloader
## Instantiate ECGDataBase
## Instantiate ECGDataLoader, passing it ECGDataBase and the subjects in the current split



#For report generation we used ECGDataLoader2.pregen_reports_threading, version saved on this computer
#For image generation we used ___, version saved on smaller dell


"""
Python data loader that retrieves waveform files downloaded from
https://physionet.org/content/mimic-iv-ecg-demo/0.1/ and converts them
into png images. It also generates synthetic text-based reports for each waveform
record based on diagnostic data from the MIMIC-IV-ECG and MIMIC-IV
clinical databases.

If a different database is used in future replace self.mimicDataPath with the new path.

Other databases that can be used https://physionet.org/content/mimic-iv-ecg/1.0/files/
"""

MIMIC_IV_DIR_MINI_SUBSET = "MIMIC-IV-data-mini-subset"
MIMIC_IV_DIR_COMPLETE = "MIMIC-IV-data"
MIMIC_IV_DIR = MIMIC_IV_DIR_MINI_SUBSET
#UNCOMMENT if you have access to the complete dataset:
#MIMIC_IV_DIR = MIMIC_IV_DIR_COMPLETE
MIMIC_IV_ECG_DIR = "mimic-iv-ecg-matched-subset"
MIMIC_IV_ECG_DATA_SRC_DIR = "mimic-iv-ecg_complete" 
MIMIC_IV_ECG_META_DIR = "meta_files"
MIMIC_IV_EMR_DIR = "mimic-iv-2.2"

MIMICIV2PTBXL_SIGNAL_ADJUSTMENTS = {
    "offset": np.array([-119, -55, 64, 86, -91, 4, -69, -31, 0, -26, -39, -79]),
    "gain": 5
}

DEFAULT_ECG_CONFIG = {
    "type": "image",
    "dimensions": (300,300),
    "signal_adjustments": {},
    #"signal_adjustments": MIMICIV2PTBXL_SIGNAL_ADJUSTMENTS,
    "ecg_sampling_rate": 500 #2500
}

class ECGDataBase:
    #TODO: improve docummentation for this class
    """
    ECGDataBase class defines the database for an ECGReporting experiment by compiling a list of records to use. 
    This class also acts as a helper class for ECGDataLoader and ECGDataPreparation. An instance of this class is passed to any instance of ECGDataLoader and ECGDataPreparation

    Attributes:
        
    """
    #WARNING: avoid implementing any methods in ECGDataBase which rely on reading data from self.record_list or self.machine_reports because this will not be synchronized with instances of ECGDataLoader
    def __init__(self, ecg_config: dict = DEFAULT_ECG_CONFIG,
                 ecg_data_dir: str = MIMIC_IV_ECG_DATA_SRC_DIR):
        self.ecg_config = ecg_config
        self.ecg_data_dir = ecg_data_dir
        self.emr_loaded = False
        self.admissions = None
        self.diagnoses_icd = None
        self.d_icd_diagnoses = None
        
        print("Instansiating ECGDataBase")
        
        self.init_paths()
        print("Paths initialized")

        self.base_sanity_check()
        print("ECGDataBase sanity check passed")

        self.unpack_ecg_config()
        print("ECG config unpacked")

        self.compile_record_list()
        print("Record list compiled")

        self.all_subject_ids = self.record_list['subject_id'].unique()

    def init_paths(self):
        """
        Initialise all of the MIMIC-IV dataset paths. ECGDataLoader is dependent on these.
        """
        self.mimiciv_root_dir = Path.cwd().joinpath(MIMIC_IV_DIR)
        
        self.mimic_ecg_matched_path = self.mimiciv_root_dir / MIMIC_IV_ECG_DIR
        self.ecg_meta_path = self.mimic_ecg_matched_path / MIMIC_IV_ECG_META_DIR
        
        self.ecg_path = self.mimic_ecg_matched_path / self.ecg_data_dir
        self.ecg_waveforms_path = self.ecg_path / "files"        
        
        self.clinical_path = self.mimiciv_root_dir / MIMIC_IV_EMR_DIR
        self.hosp_path = self.clinical_path / "hosp"
    
    def base_sanity_check(self):
        #TODO
        return

    def unpack_ecg_config(self):
        """
        Unpack ecg configurations for pre-processing the DICOM and generating images. ECGDataLoader is not dependent on these ECG configuration attributes.
        """
        self.ecg_sampling_rate = self.ecg_config["ecg_sampling_rate"]
        self.gain = None
        self.offset = None
        self.ecg_image_dimensions = None
        if "signal_adjustments" in self.ecg_config:
            #signal adjustments may make MIMIC-IV ECG more compatible with another dataset
            if "offset" in self.ecg_config["signal_adjustments"]: self.offset = self.ecg_config["signal_adjustments"]["offset"]
            if "gain" in self.ecg_config["signal_adjustments"]: self.gain = self.ecg_config["signal_adjustments"]["gain"]
        if self.ecg_config['type'] == "image":
            self.ecg_image_dimensions = self.ecg_config['dimensions']
    
    def compile_record_list(self):
        """
        Compile list of all mimic-iv-ecg-matched-subset records and store as a dataframe in self.record_list. 
        Load the machine_measurements.csv and align it with record_list
        """
        self.machine_reports = pd.read_csv(
            self.ecg_meta_path / "machine_measurements.csv",
            na_filter=False,
            parse_dates=["ecg_time"]
        )
        #load record_list.csv
        if self.ecg_data_dir != "mimic-iv-ecg_complete":
            #for ecg_data_dir subsets other than "mimic-iv-ecg_complete" we have to make our own record_list.csv
            if (self.ecg_path / "record_list.csv").exists():
                self.record_list = pd.read_csv(self.ecg_path / "record_list.csv",na_filter=False)
            else: #if you don't have a record list for your ecg_data_dir subset bob will build one:
                print(f"Building record list because {self.ecg_path} hasn't got one")
                from BuildRecordList import BuildRecordList
                bob = BuildRecordList(self.ecg_data_dir)
                bob.build()
                self.record_list = pd.read_csv(self.ecg_path / "record_list.csv",na_filter=False)
                print(f"Finished building records list")
        else: #there doesn't need to be a record_list stored in 'mimic-iv-ecg_complete' since we just use the official full record list in 'meta_files'
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
        #filter machine_reports to match record_list
        self.machine_reports = self.machine_reports[self.machine_reports['study_id'].isin(self.record_list['study_id'])].reset_index(drop=True)
        
        self.record_list['path'] = self.record_list['path'].apply(Path)
    
    def load_emr(self):
        """
        Load EMR data to be used for synthetic text report generation
        """
        if self.emr_loaded:
            print("EMR already loaded: admissions.csv, diagnoses_icd.csv, d_icd_diagnoses.csv")
            return
        else:
            print("Loading EMR: admissions.csv, diagnoses_icd.csv, d_icd_diagnoses.csv")
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
        self.emr_loaded = True
    
    def get_standard_ecg_image(self, file_path):
        """
        The slow way to convert a DICOM formatted ECG study into a standardized ECG image representation.
        """
        #load ECG signal as a numpy array
        signal = self.get_ecg_signal(file_path)
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
        squared_image = image.resize(self.ecg_image_dimensions)
        #squared_image.save(str(file_path.name) + ".png")
        
        return squared_image

    def get_standard_ecg_image_fast(self, file_path):
        """
        The fast way to convert a DICOM formatted ECG study into a standardized ECG image representation.
        """
        signal = self.get_ecg_signal(file_path)
        buffer = BytesIO()
        fig, ax = plt.subplots()
        ecg_plot.plot(np.transpose(signal), sample_rate = self.ecg_sampling_rate, title = ' ', style = 'bw')
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close('all')
        del fig, ax
        buffer.seek(0)
        image = Image.open(buffer)
        ecg_image = image.resize(self.ecg_image_dimensions)
        return ecg_image

    def get_ecg_signal(self, file_path):
        """
        Load DICOM ECG study to numpy array
        """
        try:
            signal = wfdb.rdsamp(file_path)[0] #numpy (5000, 12)
        except Exception as e:
            print(f"Error loading ECG signal: {file_path} - {e}")
            return None
        if self.gain is not None:
            signal = signal * 5
        if self.offset is not None:
            signal = signal - self.offset
        return signal

    def compile_report(self, record):
        """
        Compile synthetic free text report from EMR for a single study record belonging to machine_measurements
        """
        report = []
        report.append(self.get_measurements(record))
        report.append(self.get_machine_reports(record))
        report.append(self.get_diagnoses(record, self.admissions, self.diagnoses_icd, self.d_icd_diagnoses))
        return " ".join(report)
    
    def get_measurements(self, rec):
        """
        Returns a string of compiled summary measurements from a
        specific ECG record from machine_measurements. The measurements describe the average
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
            f"{rec['t_axis']} degrees for the T-wave."
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
        #print(f"hadm_id_cell: {hadm_id_cell}")
        #print(f"hadm_id: {hadm_id}")
        ecg_icds = []
        substr = r"^427(?:[^5]|\Z)|^7850|^I47|^I48|^I49(?:[^5]|\Z)"
        # ecg_icds = diagnoses_icd.loc[
        #     (diagnoses_icd["subject_id"] == record["subject_id"])
        #     & (diagnoses_icd["hadm_id"] == hadm_id)
        #     & diagnoses_icd["icd_code"].str.contains(substr),
        #     "icd_code"
        # ].values.flatten().tolist()

        for hadm_id in hadm_id_cell:
            icds_for_hadm_id = diagnoses_icd.loc[
                (diagnoses_icd["subject_id"] == record["subject_id"])
                & (diagnoses_icd["hadm_id"] == hadm_id)
                & diagnoses_icd["icd_code"].str.contains(substr),
                "icd_code"
            ].values.flatten().tolist()
            ecg_icds.extend(icds_for_hadm_id)

        # Remove duplicates if needed
        ecg_icds = list(set(ecg_icds))

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
        
    def train_val_split(self, val_ratio: float = 0.2, random_seed: int = 42):
        """
        Perform subject-wise train-val split and return subject_ids
        """
        train_subjects, val_subjects = train_test_split(
            self.all_subject_ids,
            test_size=val_ratio,
            random_state=random_seed
        )
        return train_subjects, val_subjects

    def random_undersample(self, undersample_to):
        """
        Randomly removes N-undersample_to records from the current database by removing them from record_list, machine_reports and all_subject_ids.
        Useful if you want to test the code quickly.
        """
        undersample_mask = random.sample(range(len(self.record_list)), undersample_to)
        self.record_list = self.record_list.iloc[undersample_mask]
        self.machine_reports = self.machine_reports[self.machine_reports['study_id'].isin(self.record_list['study_id'])]
        self.all_subject_ids = self.record_list['subject_id'].unique()

    def save_mini_subset(self, num_studies):
        """
        Code used to compile the MIMIC-IV-data-mini-subset from the original MIMIC-IV-data
        """
        self.random_undersample(undersample_to=num_studies)

        #save machine_reports
        mini_mimiciv_root_dir = Path.cwd().joinpath("MIMIC-IV-data-mini-subset")
        
        mini_mimic_ecg_matched_path = mini_mimiciv_root_dir / MIMIC_IV_ECG_DIR
        mini_ecg_meta_path = mini_mimic_ecg_matched_path / MIMIC_IV_ECG_META_DIR
        
        mini_ecg_path = mini_mimic_ecg_matched_path / "mimic-iv-ecg_complete"
        mini_ecg_img_path = mini_mimic_ecg_matched_path / "mimic-iv-ecg_complete_300x300_images"
        mini_ecg_waveforms_path = mini_ecg_path / "files"
        
        mini_clinical_path = mini_mimiciv_root_dir / MIMIC_IV_EMR_DIR
        mini_hosp_path = mini_clinical_path / "hosp"

        def copy_data(x):
            src_dat = self.mimic_ecg_matched_path / "mimic-iv-ecg_complete" / Path(str(x)+".dat")
            src_hea = self.mimic_ecg_matched_path / "mimic-iv-ecg_complete" / Path(str(x)+".hea")
            src_png = self.mimic_ecg_matched_path / "mimic-iv-ecg_complete_300x300_images_to_p1067" / Path(str(x)+".png")
            
            tgt_dat = mini_ecg_path / Path(str(x)+".dat")
            tgt_hea = mini_ecg_path / Path(str(x)+".hea")
            tgt_png = mini_ecg_img_path / Path(str(x)+".png")

            # Copy files if they exist
            for src, tgt in [(src_dat, tgt_dat), (src_hea, tgt_hea), (src_png, tgt_png)]:
                if src.exists():
                    tgt.parent.mkdir(parents=True, exist_ok=True)  # Ensure the target directory exists
                    shutil.copy(src, tgt)  # Copy file
                else:
                    print(f"Source file not found: {src}")
            return x

        self.record_list['path'].apply(copy_data)

        self.machine_reports.to_csv(mini_ecg_meta_path/"machine_measurements.csv", index=False)          
        self.record_list.to_csv(mini_ecg_meta_path/"record_list.csv", index=False)                   
        
        wlink = pd.read_csv(self.ecg_meta_path / "waveform_note_links.csv", na_filter=False)
        wlink = wlink[wlink['study_id'].isin(self.record_list['study_id'])]
        wlink.to_csv(mini_ecg_meta_path/"waveform_note_links.csv", index=False)          
        
        admi = pd.read_csv(self.hosp_path / "admissions.csv", na_filter=False)
        admi = admi[admi['subject_id'].isin(self.record_list['subject_id'])]
        admi.to_csv(mini_hosp_path/"admissions.csv", index=False)          
        
        #just copy the whole original file because it is not that big
        #dicd = pd.read_csv(self.hosp_path / "d_icd_diagnoses.csv", na_filter=False)
        # how to filter out rows not in random undersampled mimi subset?
        #dicd.to_csv(mini_hosp_path/"d_icd_diagnoses.csv", index=False)          
        
        #this file is slightly too big, so filter out by subject id
        diag = pd.read_csv(self.hosp_path / "diagnoses_icd.csv", na_filter=False)
        diag = diag[diag['subject_id'].isin(self.record_list['subject_id'])]
        diag.to_csv(mini_hosp_path/"diagnoses_icd.csv", index=False)   

        reports_df = pd.read_csv(self.mimic_ecg_matched_path / "mimic-iv-ecg_complete_300x300_images_to_p1067" / "all_reports.csv")
        reports_df['study_path'] = reports_df['study_path'].apply(Path)
        reports_df = reports_df[reports_df['study_path'].isin(self.record_list['path'])]
        reports_df.to_csv(mini_ecg_path / "all_reports.csv", index=False)
      
class ECGDataPreparation:
    #TODO: improve docummentation for this class
    #TODO: test this class more extensively
    """
    ECGDataPreparation class is used for preparing the dataset for static loading. It performs to main functions:
        1. generating the synthetic free text reports from the EMR and ECG machine measurements
        2. converting DICOM representation of ECG data to images (signal arrays currently not supported)

    Attributes:
        
    """
    def __init__(self, dataset_base: ECGDataBase,
                 ecg_save_dir: str = ""):
        self.base = dataset_base
        self.ecg_save_dir = ecg_save_dir
        if self.ecg_save_dir == "":
            import datetime
            self.ecg_save_dir = str(datetime.datetime.now())
        self.ecg_save_path = self.base.mimic_ecg_matched_path / self.ecg_save_dir
        self.ecg_save_path.mkdir(exist_ok=True)
        self.reports_save_interval = 10
    
    def save_report_csv(self, report_records: list):
        """Saves the current text report records to CSV."""
        df = pd.DataFrame(report_records)
        df.to_csv(self.ecg_save_path / "all_reports.csv", index=False)
        print("Saved intermediate CSV")
    
    def process_record_report(self, record: pd.Series) -> dict:
        try:
            # Filter for the matching study_id
            record_row = self.base.record_list[self.base.record_list['study_id'] == record['study_id']]
            if len(record_row) == 0:
                return None
            path_stem = record_row.iloc[0]['path']
            report = self.base.compile_report(record)
            return {"study_path": path_stem, "report": report}
        except Exception as e:
            print(f"Error processing record {record['study_id']}: {e}")
            traceback.print_exc()
            return None
        
    def save_ecg_image(self, study_path):
        #(previously process_study)
        #used for threading
        if study_path.is_file(): 
            print(f"Error: study path is a file: {study_path}")
            return
        inner_name = str(study_path.name[1:])
        dat_file = study_path / (inner_name + ".dat")
        hea_file = study_path / (inner_name + ".hea")
        typeless_file = study_path / inner_name
        ecg_image = self.base.get_standard_ecg_image_fast(typeless_file)
        
        # Define the save path and create directories if necessary
        save_file = str(typeless_file).replace(self.base.ecg_data_dir, self.ecg_save_dir) + ".png"
        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        ecg_image.save(save_file)
        
    def pregen_reports_threading(self): #this is the most efficient way to pregenerate the synthetic text reports
        self.base.load_emr()
        #TODO: not efficient to save the entire df repeatedly, would be better to only write new entries to the file
        report_records = []
        report_lock = threading.Lock()  # Lock for thread-safe access to shared data
        pbar_groups = tqdm(total=len(self.base.machine_reports), desc="Reports", leave=True)
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_record_report, record) for _, record in self.base.machine_reports.iterrows()]
            for i, future in enumerate(as_completed(futures),1):
                result = future.result()
                if result:
                    with report_lock:  # Ensure that only one thread accesses report_records at a time
                        report_records.append(result)
                        pbar_groups.update(1)
                if i % self.reports_save_interval == 0:
                    with report_lock:
                        self.save_report_csv(report_records)
        pbar_groups.close()
        # Save the results to a DataFrame
        with report_lock:
            self.save_report_csv(report_records)
    
    def pregen_reports(self): #slow
        self.base.load_emr()
        #this will only be called for mimic-iv-ecg_complete
        # present_studies = set()
        # with open(self.ecg_path / "present_studies.txt", "r") as f:
        #     for line in f:
        #         present_studies.add(line.strip())
        report_records = []
        with tqdm(total=len(self.base.machine_reports), desc="Reports", leave=True) as pbar_groups:
            for i, record in self.base.machine_reports.iterrows():
                result = self.process_record_report(record)
                if result:
                    report_records.append(result)
                if i % self.reports_save_interval == 0:
                    self.save_report_csv(report_records)
        self.save_report_csv(report_records)

    def pregen_ecg_images_threading(self):
        #TODO: rather than iterating through the file system can't we iterate through record_list?
        #(previously load_and_save_threading_fast)
        #save a new directory in mimic-iv-ecg-matched-subset for the ecg images
        #load the missing studies list
        missing_studies = set()
        with open(self.base.ecg_path / "missing_studies.txt", "r") as f:
            for line in f:
                missing_studies.add(line.strip())
        
        # Define the new subset directory name
        total_groups = len(list(self.base.ecg_waveforms_path.iterdir()))
        
        # Use a single ProcessPoolExecutor for the entire function to reduce overhead
        with ProcessPoolExecutor(max_workers=20) as executor:
            with tqdm(total=total_groups, desc="Groups", leave=True) as pbar_groups:
                # Iterate over each group
                for gr in self.base.ecg_waveforms_path.iterdir():
                    if gr.is_file():continue
                    # Process each subject within the group
                    for sb in gr.iterdir():
                        if sb.is_file():continue
                        # Gather all valid studies for this subject
                        studies = [
                            st for st in sb.iterdir()
                            if st.is_dir() and str(Path(*((st / st.name[1:]).parts[-5:]))) not in missing_studies
                        ]
                        # Submit all studies for processing in a batch
                        futures = [executor.submit(self.save_ecg_image, st) for st in studies]
                        # Use as_completed to handle the futures as they complete
                        for future in as_completed(futures):
                            # Retrieve results if needed or handle exceptions
                            try:
                                future.result()
                            except Exception as e:
                                #TODO: better error handling
                                print(f"An error occurred: {e}")
                    # Update progress for each group
                    pbar_groups.update(1)

class ECGDataLoader(Dataset):
    """
    ECGDataLoader class defines the torch.utils.data.Dataset for one specific split (train or val) for an ECGReporting experiment. 
    The method get_dataloader returns a torch.utils.data.DataLoader for the current split.

    Attributes:

    """
    def __init__(self, database: ECGDataBase, 
                 split_subjects: list = [],
                 dynamic_loading: bool = True,
                 batch_size: int = 8,
                 shuffle: bool = True):
        self.base = database
        self.split_subjects = split_subjects
        self.dynamic_loading = dynamic_loading
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.dataloader_sanity_checks()
        print("Dataloader sanity checks passed")

        self.load_reports()
        print("Loaded clinical reports")

        #creates its own versions of record_list and machine_reports 
        self.filter_subject_split()
        print("Subject split filtered")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def dataloader_sanity_checks(self):
        file_types = []
        self.can_do_dynamic_loading = False
        self.can_do_static_loading = False
        file_acquired = False
        for gr in self.base.ecg_waveforms_path.iterdir():
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
        if ".png" in file_types and (self.base.ecg_path / "all_reports.csv").exists():
            self.can_do_static_loading = True
        
        if self.dynamic_loading and not self.can_do_dynamic_loading:
            raise RuntimeError(f"ECG subset in {self.base.ecg_subset} is not meant for dynamic loading")
        if not self.dynamic_loading and not self.can_do_static_loading:
            raise RuntimeError(f"ECG subset in {self.base.ecg_subset} is not meant for static loading")

    def load_reports(self):
        if self.dynamic_loading:
            self.base.load_emr()
            self.reports_df = None
        else:
            self.reports_df = pd.read_csv(self.base.ecg_path / "all_reports.csv")
            self.reports_df['study_id'] = self.reports_df['study_path'].apply(os.path.basename).astype(str)

    def filter_subject_split(self):
        #only get subjects in current split
        self.record_list = self.base.record_list[self.base.record_list['subject_id'].isin(self.split_subjects)]
        self.machine_reports = self.base.machine_reports[self.base.machine_reports['subject_id'].isin(self.record_list['subject_id'])].reset_index(drop=True)
        #TODO: filter split_subjects in self.admissions, self.diagnoses_icd, self.d_icd_diagnoses, self.reports_df to save memory

    def __len__(self):
        # Dataset size based on the number of records in machine reports
        return len(self.machine_reports)
    
    def __getitem__(self, idx):
        while True:
            #the reason we have a while True loop here is because some studies from MIMIC-IV ECG Matched Subset failed to download properly and can't be loaded
            #TODO: reattempt to download missing files
            #currently if idx indexes a missing study we just set idx to another random study in the except clause
            try:
                record = self.machine_reports.iloc[idx]
                study_id = record["study_id"]
                record_row = self.record_list[self.record_list['study_id'] == study_id]
                path_stem = record_row.iloc[0]['path']
                ecg_image = self.get_ecg_image(path_stem)
                ecg_image = self.transform(ecg_image)
                break
            except Exception as e:
                print(f"Error loading sample at {idx}: {e}")
                traceback.print_exc()
                idx = random.randint(0, len(self) - 1)
        report = self.get_report(record)    
        sample = {
            "waveform": ecg_image,
            "text": report
        }
        return sample

    def get_ecg_image(self, study_path_stem):
        if self.dynamic_loading:
            #load the DICOM to an image
            file_path = self.base.ecg_path / study_path_stem
            image = self.base.get_standard_ecg_image_fast(file_path)
        else:
            #load pregenerated ECG image
            image_path = Path(str(self.base.ecg_path / study_path_stem)+".png")
            image = Image.open(image_path).convert("RGB") 
        return image

    def get_report(self, record):
        """Get report from the diagnostic data for a single record."""
        if self.dynamic_loading:
            return self.base.compile_report(record)
        else:
            study_id = record["study_id"]
            report_row = self.reports_df[self.reports_df['study_id'] == str(study_id)]
            if len(report_row) == 0: 
                #TODO: when and why does this happen?
                print(f"No report for {study_id}")
                return " " 
            return report_row['report'].iloc[0]

    def get_dataloader(self) -> DataLoader:
        return DataLoader(self, 
                          batch_size=self.batch_size, 
                          shuffle=self.shuffle, 
                          collate_fn=None)

def show_ecg_batch(batch, figsize=(15, 15)):
    """
    Display a batch of ECG images.

    Args:
        batch: dict
        batch_size (int): Number of images to display in one row/column of the grid.
        figsize (tuple): Size of the matplotlib figure.
    """
    # Convert tensor to numpy array and remove any channel dimension
    batch_images = batch['waveform'].detach().cpu()
    batch_size = batch_images.shape[0]
    batch_reports = batch['text']
    
    # Set up plot grid
    fig, axes = plt.subplots(batch_size, figsize=figsize)
    for i in range(batch_size):
        image = batch_images[i].permute(1, 2, 0).numpy()
        #axes[i].imshow(image)  # Display image in grayscale
        #axes[i].axis("off")  # Hide axes for clarity
        #print(f"{batch_reports[i]}\n\n")

    #plt.tight_layout()
    #plt.show()

def test_dl(dl, nbatches):
    print(f"len(dl): {len(dl)}")
    for i, b in enumerate(dl):
        print(f"batch: {i}")
        if i >= nbatches:
            return
        show_ecg_batch(b) 

def test_ecg_data_preparation():
    edp = ECGDataPreparation(ecg_save_dir="test_prep")
    edp.basic_undersample_for_testing(100)
    #edp.pregen_reports_threading()
    #edp.pregen_reports()
    edp.pregen_ecg_images_threading()

def test_static_loading():
    edb = ECGDataBase(ecg_data_dir="mimic-iv-ecg_complete_300x300_images_to_p1067")
    #edb.random_undersample(100)
    tr_subjects, vl_subjects = edb.train_val_split()
    print(f"tr_subjects: {len(tr_subjects)}")
    print(f"vl_subjects: {len(vl_subjects)}")
    edl = ECGDataLoader(database=edb, split_subjects=tr_subjects, dynamic_loading=False, batch_size=8, shuffle=True)
    print(f"len(edl): {len(edl)}")
    dl = edl.get_dataloader()
    test_dl(dl, 2)

def test_dynamic_loading():
    edb = ECGDataBase()
    print(f"Pre undersample: edb.record_list: {len(edb.record_list)} edb.machine_reports: {len(edb.machine_reports)}")
    edb.random_undersample(100)
    print(f"Post undersample: edb.record_list: {len(edb.record_list)} edb.machine_reports: {len(edb.machine_reports)}")
    tr_subjects, vl_subjects = edb.train_val_split()
    print(f"tr_subjects: {len(tr_subjects)}")
    print(f"vl_subjects: {len(vl_subjects)}")
    edl = ECGDataLoader(database=edb, split_subjects=tr_subjects, dynamic_loading=True, batch_size=8, shuffle=True)
    print(f"len(edl): {len(edl)}")
    print(f"edl.record_list: {len(edl.record_list)} edl.machine_reports: {len(edl.machine_reports)}")
    dl = edl.get_dataloader()
    test_dl(dl, 2)


if __name__ == "__main__":
    #test_ecg_data_preparation()
    test_static_loading()
    #test_dynamic_loading()
    