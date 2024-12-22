from pathlib import Path
import wfdb
import matplotlib.pyplot as plt
import ecg_plot
from tqdm import tqdm
from io import BytesIO
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


ecg_waveforms_path = Path("D:\\MIMIC-IV-ECG\\physionet.org\\files\\mimic-iv-ecg\\1.0\\files")
new_ecg_subset_dir = Path("D:\\MIMIC-IV-ECG\\mimic-iv-ecg_complete_300x300_images")
#getting missing_studies.txt
#ecg_subset

def get_ecg_signal(file_path):
    try:
        signal = wfdb.rdsamp(file_path)[0] #numpy (5000, 12)
    except Exception as e:
        print(f"Error loading ECG signal: {file_path}")
        return None
    return signal

def get_standard_ecg_image_fast(file_path):
    signal = get_ecg_signal(file_path)
    net_input_size = 300
    buffer = BytesIO()
    fig, ax = plt.subplots()
    ecg_plot.plot(np.transpose(signal), sample_rate = 500, title = ' ', style = 'bw')
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close('all')
    del fig, ax
    buffer.seek(0)
    image = Image.open(buffer)
    ecg_image = image.resize((300, 300))
    return ecg_image

def process_study(st, new_ecg_subset_dir):
    if st.is_file(): return
    inner_name = str(st.name[1:])
    dat_file = st / (inner_name + ".dat")
    hea_file = st / (inner_name + ".hea")
    typeless_file = st / inner_name
    ecg_image = get_standard_ecg_image_fast(typeless_file)
    save_file = str(new_ecg_subset_dir / Path(*typeless_file.parts[-5:]))+".png"
    Path(save_file).parent.mkdir(parents=True, exist_ok=True)
    ecg_image.save(save_file)

def load_and_save_threading_fast():
    missing_studies = set()
    with open("missing_studies.txt", "r") as f:
        for line in f:
            missing_studies.add(line.strip())
        
    
    total_groups = len(list(ecg_waveforms_path.iterdir()))
    not_yet = True
    with ProcessPoolExecutor(max_workers=20) as executor:
        with tqdm(total=total_groups, desc="Groups", leave=True) as pbar_groups:
            for gr in ecg_waveforms_path.iterdir():
                if "p1068" in str(gr): not_yet = False
                if not_yet: continue
                if gr.is_file():continue
                for sb in gr.iterdir():
                    if sb.is_file():continue

                    studies = [
                        st for st in sb.iterdir()
                        if st.is_dir() and str(Path(*((st / st.name[1:]).parts[-5:]))) not in missing_studies
                    ]

                    futures = [executor.submit(process_study, st, new_ecg_subset_dir) for st in studies]

                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"An error occurred: {e}")

                pbar_groups.update(1)
if __name__ == "__main__":
    load_and_save_threading_fast()