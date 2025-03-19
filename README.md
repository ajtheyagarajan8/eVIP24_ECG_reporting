# eVIP ECG Reporting
This repository is an implementation of an LLM and contrastive learning approach for generating clinical free-text reports from ECG records in the MIMIC-IV dataset. 
* [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)
* [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)

## Environment Setup

1. Create new conda environment: 

      `conda env create -f environment.yml`

2. The required pytorch versions have a known bug in the foreign function interface (ffi) dynamic link libraries. To debug this you need to copy files `ffi.dll`, `ffi-7.dll`, and `ffi-8.dll` from `<Your Anaconda Path>/Library/bin` to `<Your Anaconda Path>/envs/ecg_reporting/Library/bin`.

3. Use pip to install pytorch packages:
    
    Option 1: if you have a GPU from NVIDIA (e.g., RTX, GTX, A-series, Tesla) and CUDA installed on your system
    
      `pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121`
    
    Option 2: if you have no dedicate GPU or only an Intel/AMD integrated GPU
    
      `pip install torch==2.1.2+cpu torchvision==0.16.2+cpu --index-url https://download.pytorch.org/whl/cpu`


## Quick Start

Make sure `MIMIC_IV_DIR = MIMIC_IV_DIR_MINI_SUBSET` in `ECGDataloader.py`. Run cells in `ECGReporting.ipynb` to train a model using the `MIMIC-IV-data-mini-subset` from this repository.

## MIMIC-IV Datasets
The MIMIC-IV datasets used in this project are >100 Gb and contain >3 million files. It is available for download once you become a credentialed user on Physionet. Please email [jonathan.williams@student.unsw.edu.au](mailto:jonathan.williams@student.unsw.edu.au) for instructions. Once downloaded please unzip and make sure it is stored in a folder named `MIMIC-IV-data`. 

This repository contains a synthetic subset of MIMIC-IV in `MIMIC-IV-data-mini-subset` which is only 65Mb and contains only 600 records. This can be used for testing and learning the code. The file structure in `MIMIC-IV-data-mini-subset` mostly resembles the file structure for the full dataset `MIMIC-IV-data`. 

### MIMIC-IV File Structure
- MIMIC-IV-data-mini-subset/
  - mimic-iv-2.2/
    - hosp/
      - admissions.csv
      - d_icd_diagnoses.csv
      - diagnoses_icd.csv
  - mimic-iv-ecg-matched-subset/
    - meta_files/
      - machine_measurements.csv
      - record_list.csv
    - mimic-iv-ecg_complete/
      - all_reports.csv
      - missing_studies.txt
      - files/
        - p{group_id}
          - p{subject_id}
            - s{study_id}
              - {study_id}.dat
              - {study_id}.hea
    - mimic-iv-ecg_complete_300x300_images/
      - all_reports.csv
      - missing_studies.txt
      - files/
        - p{group_id}
          - p{subject_id}
            - s{study_id}
              - {study_id}.png

* `mimic-iv-2.2` contains the hosp module from [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) which provide diagnostic EMR used for generating synthetic clinical text reports
* `mimic-iv-ecg-matched-subset` contains ECG data from [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/)
* `mimic-iv-ecg_complete` contains the ECG DICOM files (.dat and .hea). For the full dataset `MIMIC-IV-data`, `mimic-iv-ecg_complete` will contain DICOM files for ALL studies 
* `mimic-iv-ecg_complete_300x300_images` contains a file structure identical to `mimic-iv-ecg_complete` however the DICOM files are replaced by 300x300 png images of the ECG capture
* `meta_files` contain machine_measurements.csv and record_list.csv from [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/). For the MIMIC-IV-data-mini-subset these tables have been shortened down to 600 records. 

## ECGDataLoader.py
This file contains classes `ECGDataBase`, `ECGDataPreparation` and `ECGDataLoader` for preparing the dataset and building a `torch.utils.data.DataLoader`.

## MIMIC Files

* `missing_studies.txt`: WGET was used to download MIMIC-IV-ECG and it failed to download some files. These will be redownloaded soon. For now we use these `missing_studies.txt` to filter out studies with missing DICOM files from the `ECGDataBase`.
* `all_reports.csv`
* `record_list.csv` is the list of all records (studies) that should be present in this ecg subset (including the missing studies), it tells us the relative path for each study
* `machine_measurements.csv`

## Contact

For support or inquiries, please contact the repository manager:

- **Email**: [jonathan.williams@student.unsw.edu.au](mailto:jonathan.williams@student.unsw.edu.au)
- **GitHub**: [@Jon-bon-Jono](https://github.com/Jon-bon-Jono)
