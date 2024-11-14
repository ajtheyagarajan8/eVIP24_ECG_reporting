from pathlib import Path
import pandas as pd

class BuildRecordList:
    def __init__(self, subset_root):
        # Initialize file paths and parameters
        self.mimiciv_root_dir = Path.cwd().joinpath("MIMIC-IV-data")
        self.mimic_ecg_matched_path = self.mimiciv_root_dir / "mimic-iv-ecg-matched-subset"
        self.ecg_meta_path = self.mimic_ecg_matched_path / "meta_files"
        self.subset_root = subset_root
        self.ecg_path = self.mimic_ecg_matched_path / subset_root
        self.ecg_waveforms_path = self.ecg_path / "files"        

        self.parent_record_list = pd.read_csv(
                self.ecg_meta_path / "record_list.csv",
                na_filter=False,
        )
        #have to match these with the files that are in record_list.csv
        self.record_list = pd.DataFrame(columns=["subject_id", "study_id", "file_name", "path"])
        
    def build(self):
        for nests in self.ecg_waveforms_path.iterdir():
            if "index.html" in str(nests): continue
            for subject_path in nests.iterdir():
                if "RECORDS" in str(subject_path): continue
                if "index.html" in str(subject_path): continue
                for study_path in subject_path.iterdir():
                    if "index.html" in str(study_path): continue
                    relative_path_str = str(study_path/study_path.name[1:]).split(self.subset_root)[1].replace('\\', '/')
                    row = {"subject_id": subject_path.name[1:], "study_id": study_path.name[1:], "file_name": study_path.name[1:], "path": relative_path_str[1:]}
                    self.record_list = pd.concat([self.record_list, pd.DataFrame([row])], ignore_index=True)
        matched_subset = self.record_list.merge(self.parent_record_list[['path', 'ecg_time']], on='path', how='left')

        matched_subset.to_csv(self.ecg_path/"record_list.csv", index=False)


                    

if __name__ == "__main__":
    brl = BuildRecordList("mimiv-iv-ecg_p10000032_to_p10044189")
    brl.build()