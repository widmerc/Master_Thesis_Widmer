import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# -------- Settings --------
folder = r"D:\Masterarbeit\03_Model\Scripts\2_Feature_Geolocation\2_2_Model_Depth_Estimation\data\depth_processed"  # Folder with .npy files
use_float16 = True
delete_original = True

# --------------------------
npy_files = list(Path(folder).rglob("*.npy"))
print(f"Found {len(npy_files):,} .npy files to compress...")

for npy_path in tqdm(npy_files, desc="Compressing"):
    try:
        data = np.load(npy_path)
        if isinstance(data, np.lib.npyio.NpzFile):  # already compressed
            continue
        if use_float16:
            data = data.astype(np.float16)
        npz_path = npy_path.with_suffix(".npz")
        np.savez_compressed(npz_path, depth=data)
        if delete_original:
            os.remove(npy_path)
    except Exception as e:
        print(f"Error processing {npy_path.name}: {e}")
