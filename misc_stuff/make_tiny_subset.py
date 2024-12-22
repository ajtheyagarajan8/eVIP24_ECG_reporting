from ECGDataLoader import ECGDataLoader, ECGDataBase
import warnings
warnings.filterwarnings("ignore")

edb = ECGDataBase(ecg_data_dir="mimic-iv-ecg_complete_300x300_images_to_p1067")
edb.save_mini_subset(300)