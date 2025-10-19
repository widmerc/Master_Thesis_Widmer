"""
Image Downscaling Utility  
Date: 27.07.2025  
Adapted: 29.07.2025

This script provides functionality to downscale images by half their original size.
It processes a list of valid image paths and creates downscaled versions using
multiprocessing for efficient batch processing.

Purpose: This script was added because depth estimation was taking too long with 
full-size images. By reducing image dimensions by 50%, the depth estimation process
becomes significantly faster while maintaining reasonable accuracy.

Update (29.07.2025):
→ If a resized image already exists in the target folder, it will be skipped.
"""

from PIL import Image
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import os

def downscale_image_task(args):
    path, src_root, dst_root = args
    try:
        rel_path = Path(path).relative_to(src_root)
        dst_path = Path(dst_root) / rel_path

        # Skip if already exists
        if dst_path.exists():
            return True

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(path) as img:
            img = img.convert("RGB")
            img_small = img.resize((img.width // 2, img.height // 2), Image.BILINEAR)
            img_small.save(dst_path)
        return True
    except Exception:
        return False

def downscale_and_save_valid_images(valid_list_txt, src_root, dst_root, num_workers=None):
    with open(valid_list_txt, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]

    print(f"Downscaling {len(paths):,} images to half size (skipping existing)...")

    args = [(path, Path(src_root), Path(dst_root)) for path in paths]
    failed = 0
    skipped = 0

    with mp.Pool(processes=num_workers or mp.cpu_count()) as pool:
        for success in tqdm(pool.imap_unordered(downscale_image_task, args), total=len(paths), desc="Resizing"):
            if success is False:
                failed += 1

    print(f"✅ Done. {len(paths) - failed:,} processed. {failed:,} failed (others were skipped).")
