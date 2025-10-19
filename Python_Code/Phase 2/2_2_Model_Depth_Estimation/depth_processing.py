"""
Depth Estimation Processing Pipeline  
Date: 12.06.2025  
Adjusted: 13.07.2025 (batch processing and downscaling)  
Adjusted: 16.07.2025 (added .png and histogram output support)  
Adjusted: 27.07.2025 (model switching + skipping existing .npy)  
Adjusted: 27.07.2025 (real batching for efficient GPU usage and fixed memory issues)

This script processes images through a depth estimation pipeline using a selectable
depth model (e.g. MiDaS-small or depth-anything). It supports batch processing of
millions of images with inversion, .npy export, and optional visualization.
"""


from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import torch
from transformers import pipeline, AutoConfig
from pathlib import Path
import matplotlib.pyplot as plt
import gc
import torch

# CHANGE MODEL HERE:
MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
config = AutoConfig.from_pretrained(MODEL_NAME)

# Load model once
pipe_v3 = pipeline("depth-estimation", model=MODEL_NAME, config=config,
                   device=0 if torch.cuda.is_available() else -1)

# Options
use_float16 = True         # << reduces size by 50%
use_compressed_npz = True  # << saves as .npz instead of .npy

def load_paths_from_txt(txt_file: str):
    with open(txt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def compute_depth_batch(image_paths):
    results = pipe_v3(image_paths)
    if isinstance(results, dict):
        results = [results]
    return [r["predicted_depth"].squeeze().cpu().numpy() for r in results]

def invert_depth_map(depth_map: np.ndarray) -> np.ndarray:
    depth_map = np.maximum(depth_map, 0)
    depth_map[depth_map < 1] = 9999
    valid = depth_map != 9999
    if not valid.any():
        return None
    d_min, d_max = depth_map[valid].min(), depth_map[valid].max()
    inverted = np.copy(depth_map)
    inverted[valid] = d_max - (depth_map[valid] - d_min)
    return inverted

def save_depth_array(depth_map: np.ndarray, path_base: str):
    if use_float16:
        depth_map = depth_map.astype(np.float16)

    if use_compressed_npz:
        np.savez_compressed(path_base + ".npz", depth=depth_map)
    else:
        np.save(path_base + ".npy", depth_map)

def save_depth_image_and_histogram(depth_map: np.ndarray, output_path_base: str):
    valid_mask = depth_map != 9999.0
    valid_values = depth_map[valid_mask]
    if valid_values.size == 0:
        return

    normalized = ((valid_values - valid_values.min()) / (valid_values.max() - valid_values.min()) * 255).astype(np.uint8)
    visual_array = np.zeros_like(depth_map, dtype=np.uint8)
    visual_array[valid_mask] = normalized

    img = Image.fromarray(visual_array).convert("RGB")
    img.save(output_path_base + ".png")

    plt.figure(figsize=(6, 4))
    plt.hist(valid_values, bins=256, color='gray', edgecolor='black')
    plt.title("Histogram of Valid Depth Values")
    plt.xlabel("Depth Value")
    plt.ylabel("Pixel Count")
    plt.tight_layout()
    plt.savefig(output_path_base + "_hist.png")
    plt.close()

def process_from_valid_list(valid_list_file: str, original_root: str, reduced_root: str, output_folder: str,
                             batch_size: int = 128, save_images: bool = False):
    os.makedirs(output_folder, exist_ok=True)
    original_paths = load_paths_from_txt(valid_list_file)

    original_root = Path(original_root)
    reduced_root = Path(reduced_root)
    reduced_paths = [str(reduced_root / Path(p).relative_to(original_root)) for p in original_paths]

    print(f"Processing {len(reduced_paths):,} images using model: {MODEL_NAME}")
    if not reduced_paths:
        print("No valid images found.")
        return

    ext = ".npz" if use_compressed_npz else ".npy"

    with tqdm(total=len(reduced_paths), desc="Depth Estimation") as pbar:
        for i in range(0, len(reduced_paths), batch_size):
            batch_paths = reduced_paths[i:i + batch_size]
            image_names = []
            batch_out_paths = []

            valid_batch_paths = []
            for path in batch_paths:
                name = Path(path).stem
                out_path = os.path.join(output_folder, name + ext)
                if os.path.exists(out_path):
                    continue
                valid_batch_paths.append(path)
                image_names.append(name)
                batch_out_paths.append(os.path.join(output_folder, name))

            if not valid_batch_paths:
                pbar.update(len(batch_paths))
                continue

            try:
                depth_maps = compute_depth_batch(valid_batch_paths)
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                pbar.update(len(valid_batch_paths))
                continue

            saved = 0
            for depth_map, name, path_base in zip(depth_maps, image_names, batch_out_paths):
                inverted = invert_depth_map(depth_map)
                if inverted is None:
                    continue
                save_depth_array(inverted, path_base)
                if save_images:
                    save_depth_image_and_histogram(inverted, path_base)
                saved += 1

            pbar.update(len(batch_paths))
            
            # 🧹 Speicher aufräumen
            del depth_maps, valid_batch_paths, image_names, batch_out_paths
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ---------- MAIN ----------
if __name__ == "__main__":
    original_folder = r"D:\Mapillary_Data"
    reduced_folder = r"D:\Mapillary_Data_red"
    output_folder = r"D:\Mapillary_Data_Depths"
    valid_txt_file = r"D:\Mapillary_Data_valid_list.txt"

    process_from_valid_list(
        valid_list_file=valid_txt_file,
        original_root=original_folder,
        reduced_root=reduced_folder,
        output_folder=output_folder,
        batch_size=128,
        save_images=True
    )
