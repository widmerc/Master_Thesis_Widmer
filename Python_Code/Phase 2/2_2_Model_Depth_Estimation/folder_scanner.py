"""
Folder Scanner for Image Validation
Date: 27.07.2025

This script scans a folder for JPG images and validates them using PIL.
It's designed as a preprocessing step for depth estimation to filter out
invalid or corrupted images that could cause processing errors.

Purpose: Before running depth estimation models, it's crucial to identify
and exclude invalid images that might crash the processing pipeline.
This script creates a clean list of valid images for reliable batch processing.

Main features:
- Recursively scans folders for .jpg files
- Validates images using PIL to ensure they can be opened
- Uses multiprocessing for parallel validation
- Outputs a clean list of valid image paths
- Provides progress tracking and statistics
"""

from pathlib import Path
from PIL import Image
import multiprocessing as mp
from tqdm import tqdm

# Step 1: Collect paths to all .jpg files
def stream_jpg_paths(folder: str, output_txt: str):
    count = 0
    with open(output_txt, "w", encoding="utf-8") as f_out:
        for path in Path(folder).iterdir():
            if path.suffix.lower() == ".jpg":
                f_out.write(str(path) + "\n")
                count += 1
                if count % 1_000_000 == 0:
                    print(f"{count:,} .jpg files found...")
    print(f"Done. Total of {count:,} .jpg files saved in '{output_txt}'.")

# Step 2: Validate images using PIL
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return str(path)
    except:
        return None

def validate_images_parallel(jpg_list_file: str, output_valid_file: str, num_workers: int = None):
    with open(jpg_list_file, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]

    print(f"Validating {len(paths):,} images with PIL (parallel)...")

    valid_paths = []
    with mp.Pool(processes=num_workers or mp.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(is_valid_image, paths), total=len(paths), desc="Image validation"):
            if result:
                valid_paths.append(result)

    with open(output_valid_file, "w", encoding="utf-8") as f_out:
        for p in valid_paths:
            f_out.write(p + "\n")

    print(f"{len(valid_paths):,} valid images saved in: {output_valid_file}")
    print(f"{len(paths) - len(valid_paths):,} invalid images detected.")

# ▶ Main program
if __name__ == "__main__":
    input_folder = r"D:\Mapillary_Data"
    jpg_list_txt = r"D:\jpg_paths.txt"
    valid_list_txt = r"D:\jpg_valid.txt"

    # Step 1: Collect all paths
    stream_jpg_paths(input_folder, jpg_list_txt)

    # Step 2: Save only valid images
    validate_images_parallel(jpg_list_txt, valid_list_txt, num_workers=None)  # or None for auto
