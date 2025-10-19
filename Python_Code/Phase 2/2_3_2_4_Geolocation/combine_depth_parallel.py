"""
Depth-Based Object Categorization and Visualization Pipeline
Date: 13.07.2025
Adjusted: 15.07.2025 (added quantile-based depth classification)
Adjusted: 18.07.2025 (added visualization of depth categories with legend)
Adjusted: 21.07.2025 (improved CSV handling and error messages)
Adjusted: 25.07.2025 (added minmax_avg and center depth calculation methods)
Adjusted: 27.07.2025 (enhanced plotting layout with color-coded classes)
Adjusted: 29.07.2025 (added multiprocessing structure for scalability)
Adjusted: 30.07.2025 (added skip logic for missing or blurry images)
Adjusted: 31.07.2025 (improved logging output and tqdm integration)

This script processes image–depth–label triplets from a CSV file to compute per-object
depth statistics, categorize objects into depth classes (e.g. "near", "medium", "far"),
and visualize the results. It supports:
- configurable depth computation (median, mean, minmax_avg, center)
- flexible depth categorization methods (quantile, thresholds, thirds)
- per-image processing with optional plotting and saving
- multiprocessing-ready structure for large datasets
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from matplotlib.patches import Patch
from collections import defaultdict
from scipy.stats import gaussian_kde
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- Helper function: Compute depth ---
def _compute_depth(region, method="median", mask=None):
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
    elif method == "center":
        h, w = region.shape
        if mask is not None:
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                return None
            center_idx = len(ys) // 2
            return float(region[ys[center_idx], xs[center_idx]])
        else:
            return float(region[h // 2, w // 2])
    else:
        raise ValueError(f"Unbekannte Methode: {method}")

# --- Helper function: Categorize objects by depth ---
def _depth_categories(objects, method="quantile", thresholds=None, plot_distr=False, bins=20):
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
        raise ValueError(f"Unbekannte Methode: {method}")
    return objects

# --- Visualization ---
def _plot_objects(image, objects, title="Detected Objects with Depth Classes", min_conf=0.0, allowed_labels=None, show=True, save_path=None, multiple=False, dpi=100):
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
            color = colors.get(obj["z_class"], (0, 255, 0))
            text = f"{obj['label']} ({obj['z_class']})"
            cv2.rectangle(image_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        image_rgb = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8), dpi=dpi)
        plt.imshow(image_rgb)
        plt.title(title)
        plt.axis("off")
        patches = [
            Patch(color=np.array(color)[::-1]/255, label=class_name)
            for class_name, color in colors.items()
            if any(o["z_class"] == class_name for o in objects)
        ]
        plt.legend(handles=patches, title="Depth Class", loc="lower right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            plt.close()
        elif show:
            plt.show()
    else:
        classes = ["very near", "near", "medium", "far", "very far"]
        objects_by_class = defaultdict(list)
        for obj in objects:
            if obj["confidence"] < min_conf:
                continue
            if allowed_labels is not None and obj["label"] not in allowed_labels:
                continue
            objects_by_class[obj["z_class"]].append(obj)
        n = sum(1 for k in classes if objects_by_class[k])
        fig, axs = plt.subplots(1, n, figsize=(3 * n, 3), dpi=dpi)
        if n == 1:
            axs = [axs]
        for ax, class_name in zip(axs, [k for k in classes if objects_by_class[k]]):
            image_vis = image.copy()
            for obj in objects_by_class[class_name]:
                x1, y1, x2, y2 = obj["bbox"]
                color = colors.get(obj["z_class"], (0, 255, 0))
                text = f"{obj['label']} ({obj['z_class']})"
                cv2.rectangle(image_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            image_rgb = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
            ax.imshow(image_rgb)
            ax.set_title(class_name)
            ax.axis("off")
        fig.suptitle(title)
        patches = [
            Patch(color=np.array(color)[::-1]/255, label=class_name)
            for class_name, color in colors.items()
            if any(o["z_class"] == class_name for o in objects)
        ]
        fig.legend(handles=patches, title="Depth Class", loc="lower center", ncol=n, bbox_to_anchor=(0.5, 0))
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
            plt.close(fig)
        elif show:
            plt.show()

# --- Hauptfunktion: Kombination auf Basis CSV ---
def process_row(row, depth_method="median", category_method="quantile", category_thresholds=None, min_conf=0.0, plot=False, plot_folder=None, plot_kwargs=None):
    try:
        image_path = row["image_path"]
        depth_path = row["depth_path"]
        label_path = row["yolo_label_path"] if pd.notna(row["yolo_label_path"]) else None
        img_name = os.path.basename(image_path)

        print(f"[START] {img_name}")

        if not (row["has_image"] and row["has_depth"] and not row["is_blury"]):
            print(f"[SKIP] {img_name}: kein Bild oder Depth oder unscharf.")
            return None

        if not os.path.exists(image_path):
            print(f"[SKIP] {img_name}: Bildpfad fehlt.")
            return None
        if not os.path.exists(depth_path):
            print(f"[SKIP] {img_name}: Depth-Datei fehlt.")
            return None

        image = cv2.imread(image_path)
        original_shape = image.shape[:2]
        depth_map = np.load(depth_path)["depth"]
        H, W = depth_map.shape[:2]

        if image.shape[:2] != (H, W):
            print(f"[INFO] {img_name}: Originalgrösse {original_shape}, Depthgrösse {(H, W)} – resizing.")
            image = cv2.resize(image, (W, H))
        else:
            print(f"[INFO] {img_name}: Bildgrösse = Depthgrösse {(H, W)}")

        if not label_path or not os.path.exists(label_path):
            print(f"[SKIP] {img_name}: Keine Label-Datei.")
            return None

        with open(label_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            print(f"[SKIP] {img_name}: Leere Label-Datei.")
            return None

        total_boxes = 0
        used_boxes = 0
        skipped_boxes = 0
        objects = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[WARN] {img_name}: Ungültige Label-Zeile: {line.strip()}")
                continue

            class_id, cx, cy, w, h = map(float, parts)
            x1 = int((cx - w / 2) * W)
            y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W)
            y2 = int((cy + h / 2) * H)
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H))

            total_boxes += 1
            region = depth_map[y1:y2, x1:x2]
            depth = _compute_depth(region, method=depth_method)

            if depth is None:
                skipped_boxes += 1
                print(f"[SKIP] {img_name}: Leere Tiefe bei Box {x1,y1,x2,y2}")
                continue

            used_boxes += 1
            objects.append({
                "label": int(class_id),
                "confidence": 1.0,
                "depth": float(depth),
                "bbox": [x1, y1, x2, y2]
            })

        if not objects:
            print(f"[SKIP] {img_name}: Alle Objekte verworfen ({total_boxes} Boxen, {skipped_boxes} ungültig).")
            return None

        print(f"[DONE] {img_name}: {used_boxes}/{total_boxes} Objekte behalten.")

        objects = _depth_categories(objects, method=category_method, thresholds=category_thresholds)
        result = {"image": img_name, "objects": objects}

        if plot and plot_folder:
            base = os.path.splitext(img_name)[0]
            plot_path = os.path.join(plot_folder, f"{base}.jpg")
            kwargs = plot_kwargs.copy() if plot_kwargs else {}
            kwargs.setdefault("title", f"Objects with Depth: {img_name}")
            kwargs.setdefault("save_path", plot_path)
            kwargs.setdefault("show", False)
            _plot_objects(image, objects, **kwargs)

        return result

    except Exception as e:
        print(f"[ERROR] {img_name}: {e}")
        return None



# --- Main function: Combine from CSV ---
def combine_from_csv_single(
    csv_path,
    output_json_path,
    depth_method="median",
    category_method="quantile",
    category_thresholds=None,
    min_conf=0.0,
    plot=False,
    plot_output_folder=None,
    plot_kwargs=None,
    require_yolo_labels=False,
):
    df = pd.read_csv(csv_path)
    df = df[(df["has_image"]) & (df["has_depth"]) & (~df["is_blury"])]
    if require_yolo_labels:
        df = df[~df["yolo_label_path"].isna()]
    df = df.reset_index(drop=True)

    print(f"[INFO] Starte Verarbeitung von {len(df)} Bildern im Einzelmodus...")

    results = []
    for i in tqdm(range(len(df))):
        row = df.loc[i].to_dict()
        result = process_row(
            row,
            depth_method=depth_method,
            category_method=category_method,
            category_thresholds=category_thresholds,
            min_conf=min_conf,
            plot=plot,
            plot_folder=plot_output_folder,
            plot_kwargs=plot_kwargs,
        )
        if result is not None:
            results.append(result)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Fertig! {len(results)} gültige Ergebnisse gespeichert in {output_json_path}")


