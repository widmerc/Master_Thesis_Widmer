"""
Depth Estimation and Object Detection Processing Pipeline
Date: 12.07.2025
Adjusted: 13.07.2025 (batch processing and downscaling)
Adjusted: 16.07.2025 (added .png and histogram output support)
Adjusted: 27.07.2025 (model switching + skipping existing .npy)
Adjusted: 27.07.2025 (real batching for efficient GPU usage and fixed memory issues)
Adjusted: 29.07.2025 (added Numba for fast depth calculation)
Adjusted: 30.07.2025 (attempted GPU acceleration for multiprocessing)
Adjusted: 31.07.2025 (added Parquet output for x10 speed, Polars integration)

This script processes images through a depth estimation and object detection pipeline 
using selectable models (e.g., MiDaS-small, depth-anything, YOLO). It supports batch 
processing of millions of images, inversion, .npy/.parquet export, and optional visualization.
Major updates include Numba acceleration, GPU attempts, Parquet streaming, and Polars for ultra-fast
geometry joins.
"""

import os
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
from tqdm.notebook import tqdm
import logging
import pandas as pd
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np, os
from multiprocessing import Pool
from config import (
    IMAGE_FOLDER, SUITABLE_IMAGE_FOLDER, YOLO_OUTPUT_FOLDER, YOLO_MODEL_PATH, 
    DEPTH_OUTPUT_FOLDER, OUTPUT_JSON_FOLDER, PLOT_OUTPUT_FOLDER, CSV_PATH,
    BATCH_SIZE, NUM_IMAGES_TO_SAMPLE, NUM_WORKERS, CONFIDENCE_THRESHOLD,
    PLOT_OPTIONS, PLOT, DEPTH_METHOD, CATEGORY_METHOD, CATEGORY_THRESHOLDS, USE_MASK, GPKG_PATH, USE_NUMBA
)
import geopandas as gpd
from shapely.geometry import mapping
import json
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from numba import njit




logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Helper functions for depth calculation and categorization ---
@njit
def compute_depth_numba(region, mask, method_id):
    """
    Fast depth calculation using Numba for a given region and optional mask.
    Supports three methods: median, mean, minmax_avg.
    Ignores invalid pixels (value 9999) and masked-out pixels.
    Returns the computed depth value or -1.0 if no valid pixels are found.
    """
    h, w = region.shape
    values = []
    for y in range(h):
        for x in range(w):
            if (mask is None or mask[y, x] > 0) and region[y, x] != 9999:
                values.append(region[y, x])
    if len(values) == 0:
        return -1.0
    values.sort()
    n = len(values)
    if method_id == 0:  # median
        if n % 2 == 0:
            return (values[n//2 - 1] + values[n//2]) / 2.0
        else:
            return values[n//2]
    elif method_id == 1:  # mean
        return sum(values) / n
    elif method_id == 2:  # minmax_avg
        return (values[0] + values[-1]) / 2.0
    else:
        return -1.0

def _compute_depth(region, method="median", mask=None, use_numba=False):
    """
    Computes the depth value for a given region using the specified method ('median', 'mean', 'minmax_avg').
    Optionally uses a mask to exclude pixels and Numba for acceleration.
    Returns None if no valid pixels are found.
    """
    if mask is not None:
        mask = mask.astype(np.uint8)
    region = region.astype(np.float32)

    if use_numba:
        method_map = {"median": 0, "mean": 1, "minmax_avg": 2}
        method_id = method_map.get(method, 0)
        depth = compute_depth_numba(region, mask, method_id)
        return None if depth < 0 else float(depth)

    # Standard: NumPy variant
    if mask is not None:
        valid = region[(mask > 0) & (region != 9999)]
    else:
        valid = region[region != 9999]
    if valid.size == 0:
        return None
    if method == "median":
        return float(np.median(valid))
    elif method == "mean":
        return float(np.mean(valid))
    elif method == "minmax_avg":
        return float((np.min(valid) + np.max(valid)) / 2)
    else:
        return None

# --- Helper functions for category assignment ---
def _depth_categories(objects, method="quantile", thresholds=None, plot_distr=False, bins=20):
    """
    Assigns depth categories to detected objects based on their depth value.
    Supports three methods:
      - 'thirds': splits objects into three groups (near, medium, far)
      - 'quantile': uses quartiles to assign 'very near', 'near', 'medium', 'far'
      - 'thresholds': uses custom thresholds for 'near', 'medium', 'far'
    Optionally plots the depth distribution as a histogram and KDE.
    Returns the list of objects with added 'z_class' field.
    """

    if not objects:
        return objects
    
    if method == "thirds":
        objects = sorted(objects, key=lambda o: o["depth"])
        count = len(objects)
        for idx, obj in enumerate(objects):
            if idx < count / 3:
                obj["z_class"] = "near"
            elif idx < 2 * count / 3:
                obj["z_class"] = "medium"
            else:
                obj["z_class"] = "far"
    elif method == "quantile":
        depths = np.array([o["depth"] for o in objects])
        q1, q2, q3 = np.percentile(depths, [25, 50, 75])
        for obj in objects:
            if obj["depth"] <= q1:
                obj["z_class"] = "very near"
            elif obj["depth"] <= q2:
                obj["z_class"] = "near"
            elif obj["depth"] <= q3:
                obj["z_class"] = "medium"
            else:
                obj["z_class"] = "far"
    elif method == "thresholds":
        if thresholds is None:
            thresholds = [5, 15, 30]
        for obj in objects:
            if obj["depth"] <= thresholds[0]:
                obj["z_class"] = "near"
            elif obj["depth"] <= thresholds[1]:
                obj["z_class"] = "medium"
            else:
                obj["z_class"] = "far"
    else:
        raise ValueError(f"Unknown method: {method}")
    if plot_distr:
        import scipy.stats as stats
        depths = [o["depth"] for o in objects]
        plt.figure(figsize=(8,4))
        # Histogram
        plt.hist(depths, bins=bins, color='skyblue', edgecolor='k', alpha=0.7, density=True, label='Histogram')
        # KDE line
        kde = stats.gaussian_kde(depths)
        x = np.linspace(min(depths), max(depths), 200)
        plt.plot(x, kde(x), color='red', lw=2, label='KDE')
        plt.title('Distribution of Object Depths')
        plt.xlabel('Depth')
        plt.ylabel('Density')
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    return objects

def extract_result_data(result, output_dir):
    """
    Extracts detection results from a YOLO result object and prepares them for saving.
    Returns a tuple containing output path, bounding boxes, confidences, class IDs, masks, class names, and image shape.
    """
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy() if boxes else None
    conf = boxes.conf.cpu().numpy() if boxes else None
    cls = boxes.cls.cpu().numpy() if boxes else None
    masks = result.masks.data.cpu().numpy() if hasattr(result, 'masks') and result.masks else None
    names = result.names
    shape = result.orig_shape
    img_name = os.path.splitext(os.path.basename(result.path))[0]
    out_path = os.path.join(output_dir, f"{img_name}.npz")
    return (out_path, xyxy, conf, cls, masks, names, shape)

def save_npz(args):
    """
    Saves detection results to a compressed .npz file for later use.
    Args should be a tuple as returned by extract_result_data.
    """
    out_path, xyxy, conf, cls, masks, names, shape = args
    np.savez_compressed(out_path,
                        xyxy=xyxy,
                        conf=conf,
                        cls=cls,
                        masks=masks,
                        names=names,
                        shape=shape)

def _plot_objects(image, objects, title="Detected Objects with Depth Classes", min_conf=0.0, allowed_labels=None, show=True, save_path=None, multiple=False, dpi=100):
    """
    Visualizes detected objects on the image with bounding boxes and depth class labels.
    If multiple=True, plots separate images for each depth class side by side.
    Optionally saves the plot to disk and/or displays it.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    # Colors for classes
    colors = {
        "very near": (0, 255, 128),
        "near": (0, 255, 0),
        "medium": (0, 255, 255),
        "far": (255, 0, 0),
        "very far": (255, 0, 128)
    }
    if not multiple:
        image_vis = image.copy()
        for obj in objects:
            if obj["confidence"] < min_conf:
                continue
            if allowed_labels is not None and obj["label"] not in allowed_labels:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            color_bgr = colors.get(obj["z_class"], (0, 255, 0))
            text = f"{obj['names']} ({obj['z_class']})"
            cv2.rectangle(image_vis, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(image_vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)
        image_vis_rgb = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
        if show:
            plt.figure(figsize=(10, 8), dpi=dpi)
            plt.imshow(image_vis_rgb)
            plt.title(title)
            plt.axis("off")
            # Legend: Convert colors from BGR to RGB
            from matplotlib.patches import Patch
            patches = [
            Patch(color=np.array(color)[::-1]/255, label=class_name)
            for class_name, color in colors.items()
            if any(o["z_class"] == class_name for o in objects)
            ]
            plt.legend(handles=patches, title="Depth Class", loc="lower right")
            plt.tight_layout()
            plt.show()
        if save_path is not None:
            plt.figure(figsize=(10, 8), dpi=dpi)
            plt.imshow(image_vis_rgb)
            plt.title(title)
            plt.axis("off")
            from matplotlib.patches import Patch
            patches = [
            Patch(color=np.array(color)[::-1]/255, label=class_name)
            for class_name, color in colors.items()
            if any(o["z_class"] == class_name for o in objects)
            ]
            plt.legend(handles=patches, title="Depth Class", loc="lower right")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            plt.close()
    else:
        # Group objects by class
        classes = ["very near", "near", "medium", "far", "very far"]
        objects_by_class = defaultdict(list)
        for obj in objects:
            if obj["confidence"] < min_conf:
                continue
            if allowed_labels is not None and obj["label"] not in allowed_labels:
                continue
            objects_by_class[obj["z_class"]].append(obj)
        n = sum(1 for k in classes if objects_by_class[k])

        if n == 0:
            logging.warning("Keine Objekte mit gültiger Tiefenklasse zum Plotten vorhanden – überspringe Bild.")
            return

        fig, axs = plt.subplots(1, n, figsize=(3*n, 3), gridspec_kw={
            'wspace': 0.05, 'left': 0.01, 'right': 0.99, 'top': 0.8, 'bottom': 0.15
        }, dpi=dpi)
        if n == 1:
            axs = [axs]
        for ax, class_name in zip(axs, [k for k in classes if objects_by_class[k]]):
            image_vis = image.copy()
            for obj in objects_by_class[class_name]:
                x1, y1, x2, y2 = obj["bbox"]
                color = colors.get(obj["z_class"], (0, 255, 0))
                text = f"{obj['names']} ({obj['z_class']})"
                cv2.rectangle(image_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            image_vis_rgb = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
            ax.imshow(image_vis_rgb)
            ax.set_title(class_name)
            ax.axis("off")
        fig.suptitle(title)
        # Legend: Convert colors from BGR to RGB
        from matplotlib.patches import Patch
        patches = [
            Patch(color=np.array(color)[::-1]/255, label=class_name)
            for class_name, color in colors.items()
            if any(o["z_class"] == class_name for o in objects)
        ]
        fig.legend(handles=patches, title="Depth Class", loc="lower center", ncol=n, bbox_to_anchor=(0.5, 0))
        if show:
            plt.show()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
            plt.close(fig)

import os
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

def yolo_batch_inference_folder(model, image_folder, yolo_output_folder, batch_size=None, num_threads=8):
    os.makedirs(yolo_output_folder, exist_ok=True)

    image_paths = [os.path.join(image_folder, f)
                   for f in os.listdir(image_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if len(image_paths) == 0:
        print(f"Keine validen Bilder in {image_folder}")
        return

    # Queue für Ergebnisse → Verarbeitung in separatem Thread
    result_queue = queue.Queue()

    def save_worker():
        while True:
            item = result_queue.get()
            if item is None:
                break
            save_npz(item)  # z.B. {'filename':..., 'boxes':..., ...}
            result_queue.task_done()

    # Starte Threads zum Speichern
    for _ in range(num_threads):
        threading.Thread(target=save_worker, daemon=True).start()

    # GPU-Batch-Inferenz → keine Stream-Verarbeitung!
    results = model.predict(
        source=image_paths,
        batch=batch_size,
        stream=False,
        save=False,
        verbose=False
    )

    # Resultate in die Queue legen
    for res in results:
        data = extract_result_data(res, yolo_output_folder)
        result_queue.put(data)

    # Queue schließen
    result_queue.join()
    for _ in range(num_threads):
        result_queue.put(None)


def process_single_image(args):
    """
    Processes a single image by combining YOLO detection results and depth map.
    Computes depth for each detected object, assigns depth categories, and optionally saves a visualization.
    Returns a status dictionary with object data and processing status.
    Used in multiprocessing for batch processing.
    """
    import json
    import cv2
    import numpy as np
    
    (img_name, yolo_output_folder, depth_output_folder, image_folder, 
     output_json_folder, depth_method, category_method, category_thresholds, 
     min_conf, plot, plot_output_folder, plot_kwargs, use_numba) = args
    
    base = os.path.splitext(img_name)[0]
    yolo_path = os.path.join(yolo_output_folder, f"{base}.npz")
    depth_path = os.path.join(depth_output_folder, f"{base}.npz")
    
    # Status for return
    status = {"missing_yolo": 0, "missing_depth": 0, "total_objects": 0, "success": False}
    
    if not os.path.exists(yolo_path):
        status["missing_yolo"] = 1
        return status
    if not os.path.exists(depth_path):
        status["missing_depth"] = 1
        return status

    try:
        with np.load(yolo_path, allow_pickle=True) as yolo_data:
            xyxy = yolo_data["xyxy"]
            conf = yolo_data["conf"]
            cls = yolo_data["cls"]
            masks = yolo_data["masks"] if "masks" in yolo_data else None
            names = yolo_data["names"].item()
    except Exception as e:
        return status

    try:
        with np.load(depth_path) as depth_data:
            if "depth" in depth_data:
                depth_map = depth_data["depth"]
            else:
                return status
    except Exception as e:
        return status

    image_path = os.path.join(image_folder, img_name)
    image = cv2.imread(image_path)
    if image is None:
        return status

    H_img, W_img = image.shape[:2]
    H_depth, W_depth = depth_map.shape[:2]

    # Scaling if necessary
    if (H_img != H_depth) or (W_img != W_depth):
        depth_map = cv2.resize(depth_map, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

    objects = []

    if isinstance(xyxy, np.ndarray) and xyxy.ndim > 1 and conf is not None and cls is not None:
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i])
            confidence = float(conf[i])
            class_id = int(cls[i])
            name = names[class_id] if class_id in names else "Unknown"
            if confidence < min_conf:
                continue
            x1 = max(0, min(x1, W_img - 1))
            x2 = max(0, min(x2, W_img))
            y1 = max(0, min(y1, H_img - 1))
            y2 = max(0, min(y2, H_img))
            
            region = depth_map[y1:y2, x1:x2]
            
            # Use mask if available
            object_mask = None
            if masks is not None and i < len(masks):
                full_mask = masks[i]
                if full_mask.shape != (H_img, W_img):
                    full_mask = cv2.resize(full_mask.astype(np.uint8), (W_img, H_img), interpolation=cv2.INTER_NEAREST)
                object_mask = full_mask[y1:y2, x1:x2]

            depth = _compute_depth(region, method=depth_method, mask=object_mask, use_numba=use_numba)
            if depth is None:
                continue
            objects.append({
                "label": class_id,
                "names": name,
                "confidence": confidence,
                "depth": float(depth),
                "bbox": [x1, y1, x2, y2]
            })

    status["total_objects"] = len(objects)

    status["success"] = True
    objects = _depth_categories(
                objects,
                method=category_method,
                thresholds=category_thresholds,
                plot_distr=False
            )
    status["objects"] = objects
    status["image"] = img_name


    # Save plot if requested
    if plot and plot_output_folder is not None:
        plot_path = os.path.join(plot_output_folder, f"{base}.jpg")
        kwargs = plot_kwargs.copy() if plot_kwargs else {}
        kwargs.setdefault("title", f"Objects with Depth: {img_name}")
        kwargs.setdefault("save_path", plot_path)
        kwargs.setdefault("show", False)
        _plot_objects(image, objects, **kwargs)

    status["success"] = True
    return status

def combine_yolo_depth(
    yolo_output_folder,
    depth_output_folder,
    image_folder,
    output_parquet_path,
    depth_method="median",
    image_list=None,
    category_method="quantile",
    category_thresholds=None,
    min_conf=0.0,
    plot=False,
    plot_output_folder=None,
    plot_kwargs=None,
    num_workers=None,
    use_numba=False
):
    """
    Combines YOLO detection results and depth maps for a set of images using multiprocessing.
    For each image, matches detections with depth, computes per-object depth, assigns categories, and streams results to a Parquet file.
    Optionally generates and saves visualizations for each image.
    Designed for high-throughput processing of large datasets.
    """
    import multiprocessing as mp
    import pyarrow as pa
    import pyarrow.parquet as pq

    start_time = time.time()

    if image_list is not None:
        image_files_all = [os.path.basename(p) for p in image_list]
    else:
        image_files_all = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]

    already_processed = set()
    if os.path.exists(output_parquet_path):
        try:
            df_existing = pd.read_parquet(output_parquet_path, columns=["image"])
            already_processed = set(df_existing["image"].unique())
            logging.info(f"{len(already_processed)} Bilder bereits verarbeitet – werden übersprungen.")
        except Exception as e:
            logging.warning(f"Fehler beim Laden des bestehenden Parquet-Files: {e}")

    # Schritt 3: Filter anwenden
    image_files = [f for f in image_files_all if f not in already_processed]

    logging.info(f"🔄 Starting combination of {len(image_files)} images with YOLO + depth information...")

    if plot and plot_output_folder is not None:
        os.makedirs(plot_output_folder, exist_ok=True)

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(image_files))

    logging.info(f"🚀 Using {num_workers} workers for multiprocessing")

    args_list = [
        (img_name, yolo_output_folder, depth_output_folder, image_folder,
         "", depth_method, category_method, category_thresholds,
         min_conf, plot, plot_output_folder, plot_kwargs, use_numba)
        for img_name in image_files
    ]

    missing_depth = 0
    missing_yolo = 0
    total_objects = 0
    successful_images = 0

    writer = None
    try:
        with mp.Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap(process_single_image, args_list), total=len(args_list), desc="Processing Images"):
                missing_yolo += result["missing_yolo"]
                missing_depth += result["missing_depth"]
                total_objects += result["total_objects"]
                if result["success"]:
                    successful_images += 1
                    objs = result.get("objects", [])
                    for obj in objs:
                        obj["image"] = result["image"]
                    if objs:
                        df = pd.json_normalize(objs)
                        table = pa.Table.from_pandas(df)
                        if writer is None:
                            writer = pq.ParquetWriter(output_parquet_path, table.schema, compression='zstd')
                        writer.write_table(table)
    finally:
        if writer:
            writer.close()

    duration = time.time() - start_time
    logging.info("✅ Combination finished")
    logging.info(f"  ➤ Processed images: {successful_images}/{len(image_files)}")
    logging.info(f"  ➤ Detected objects: {total_objects}")
    logging.info(f"  ➤ Missing YOLO files: {missing_yolo}")
    logging.info(f"  ➤ Missing depth files: {missing_depth}")
    logging.info(f"  ➤ Parquet file saved at: {os.path.abspath(output_parquet_path)}")
    logging.info(f"  ➤ Runtime: {duration:.2f} seconds")
    if successful_images > 0:
        logging.info(f"  ➤ Performance: {successful_images/duration:.2f} images/second")




def add_geometry_to_parquet(parquet_path, gpkg_path, output_path=None):
    """
    Adds geometry coordinates (x, y) from a GeoPackage to a Parquet file containing object data.
    Merges geometry by image ID and writes a new Parquet file with added columns 'geometry.x', 'geometry.y'.
    Useful for spatial analysis and mapping of detected objects.
    """
    import pandas as pd
    import geopandas as gpd
    import time
    import os

    start_time = time.time()

    if output_path is None:
        output_path = parquet_path.replace(".parquet", "_with_geometry.parquet")

    logging.info("📦 Loading object data from Parquet...")
    df = pd.read_parquet(parquet_path)

    logging.info("🌍 Loading GPKG geometries...")
    gdf = gpd.read_file(gpkg_path)[["id", "geometry"]]
    gdf["id"] = gdf["id"].astype(str)
    gdf["geometry.x"] = gdf.geometry.x
    gdf["geometry.y"] = gdf.geometry.y
    gdf = gdf.drop(columns="geometry")

    logging.info("🔗 Linking geometries with objects...")
    df["image_id"] = df["image"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    df = df.merge(gdf, left_on="image_id", right_on="id", how="left")
    missing = df["geometry.x"].isna().sum()

    df.to_parquet(output_path, index=False)

    duration = time.time() - start_time
    logging.info("✅ Geometries added and saved")
    logging.info(f"  ➤ Saved file: {os.path.abspath(output_path)}")
    logging.info(f"  ➤ Number of objects: {len(df)}")
    logging.info(f"  ➤ Missing geometries (no match): {missing}")
    logging.info(f"  ➤ Runtime: {duration:.2f} seconds")
    if len(df) > 0:
        logging.info(f"  ➤ Performance: {len(df)/duration:.2f} rows/second")




def add_geometry_to_parquet_fast(parquet_path, gpkg_path, output_path=None, temp_geom_parquet_path=None):
    """
    Adds geometry coordinates (x, y) to a Parquet file using ultra-fast Polars for efficient joining.
    Converts the GeoPackage to Parquet if needed for speed.
    Recommended for very large datasets where performance is critical.
    """
    import os, time

    start_time = time.time()

    if output_path is None:
        output_path = parquet_path.replace(".parquet", "_with_geometry.parquet")
    if temp_geom_parquet_path is None:
        temp_geom_parquet_path = gpkg_path.replace(".gpkg", "_geometry.parquet")

    # Check if geometry.parquet already exists
    if not os.path.exists(temp_geom_parquet_path):
        # GPKG → Parquet
        logging.info("🌍 Converting GPKG → Parquet...")
        gdf = gpd.read_file(gpkg_path)[["id", "geometry"]]
        gdf["id"] = gdf["id"].astype(str)
        gdf["geometry.x"] = gdf.geometry.x
        gdf["geometry.y"] = gdf.geometry.y
        gdf.drop(columns="geometry", inplace=True)
        gdf.to_parquet(temp_geom_parquet_path, index=False)
        logging.info(f"✅ Geometry Parquet saved: {temp_geom_parquet_path}")
    else:
        logging.info(f"📦 Using existing file: {temp_geom_parquet_path}")

    # Objekte & Geometrie laden
    df = pl.read_parquet(parquet_path)
    gdf = pl.read_parquet(temp_geom_parquet_path)

    # Extract image_id
    df = df.with_columns(
        pl.col("image").map_elements(lambda p: os.path.splitext(os.path.basename(p))[0]).alias("image_id")
    )

    # Join
    df_joined = df.join(gdf, left_on="image_id", right_on="id", how="left")
    missing = df_joined["geometry.x"].is_null().sum()
    df_joined.write_parquet(output_path)

    duration = time.time() - start_time
    logging.info("✅ Geometries added and saved")
    logging.info(f"  ➤ Saved file: {os.path.abspath(output_path)}")
    logging.info(f"  ➤ Number of objects: {df_joined.shape[0]}")
    logging.info(f"  ➤ Missing geometries (no match): {missing}")
    logging.info(f"  ➤ Runtime: {duration:.2f} seconds")





# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

#     # Step 1: Load CSV and filter already processed images
#     df_valid = pd.read_csv(CSV_PATH).sample(n=NUM_IMAGES_TO_SAMPLE, random_state=42)
#     df_valid_yolo = df_valid[
#         ~df_valid["image_path"].apply(
#             lambda p: os.path.exists(
#                 os.path.join(YOLO_OUTPUT_FOLDER, os.path.splitext(os.path.basename(p))[0] + ".npz")
#             )
#         )
#     ].reset_index(drop=True)

#     logging.info(f"{len(df_valid) - len(df_valid_yolo)} images already processed – will be skipped.")

#     # Step 2: Run YOLO inference in batches
#     for i in range(0, len(df_valid_yolo), BATCH_SIZE):
#         if os.path.exists(SUITABLE_IMAGE_FOLDER):
#             shutil.rmtree(SUITABLE_IMAGE_FOLDER)
#         os.makedirs(SUITABLE_IMAGE_FOLDER, exist_ok=True)

#         df_batch = df_valid_yolo.iloc[i:i+BATCH_SIZE]
#         image_batch = df_batch["image_path"].tolist()

#         success_count = 0
#         t0 = time.time()

#         for img_rel_path in image_batch:          
#             src = os.path.join(IMAGE_FOLDER, img_rel_path)
#             dst = os.path.join(SUITABLE_IMAGE_FOLDER, os.path.basename(img_rel_path))
#             try:
#                 shutil.copy2(src, dst)
#                 success_count += 1
#             except Exception as e:
#                 logging.error(f"Error copying {src}: {e}")

#         # YOLO inference
#         model = YOLO(YOLO_MODEL_PATH)
#         yolo_batch_inference_folder(model, SUITABLE_IMAGE_FOLDER, YOLO_OUTPUT_FOLDER)

#         t1 = time.time()
#         if (i // BATCH_SIZE + 1) % 5 == 0:
#             batches_done = (i // BATCH_SIZE) + 1
#             avg_speed = ((batches_done * BATCH_SIZE) / ((t1 - t0) * batches_done)) if batches_done > 0 else 0
#             batches_left = (len(df_valid_yolo) - (batches_done * BATCH_SIZE)) // BATCH_SIZE
#             est_time_left = (batches_left * BATCH_SIZE) / avg_speed if avg_speed > 0 else 0
#             logging.info(f"Estimated remaining time for {batches_left} batches: {est_time_left/60/60:.2f} hours")

#     # Step 3: Combine YOLO + Depth
#     combine_yolo_depth(
#         yolo_output_folder=YOLO_OUTPUT_FOLDER,
#         depth_output_folder=DEPTH_OUTPUT_FOLDER,
#         image_folder=IMAGE_FOLDER,
#         output_parquet_path=os.path.join(OUTPUT_JSON_FOLDER, "combined_output.parquet"),  # new target here
#         image_list=[os.path.basename(p) for p in df_valid["image_path"]],
#         depth_method=DEPTH_METHOD,
#         category_method=CATEGORY_METHOD,
#         category_thresholds=CATEGORY_THRESHOLDS,
#         min_conf=CONFIDENCE_THRESHOLD,
#         plot=PLOT,
#         plot_output_folder=PLOT_OUTPUT_FOLDER,
#         plot_kwargs=PLOT_OPTIONS,
#         num_workers=NUM_WORKERS,
#         use_numba=USE_NUMBA
#     )

#     # Step 4: Add geometry to JSON (Multiprocessing)
#     add_geometry_to_parquet_fast(
#         parquet_path=os.path.join(OUTPUT_JSON_FOLDER, "combined_output.parquet"),
#         gpkg_path=GPKG_PATH
#     )
