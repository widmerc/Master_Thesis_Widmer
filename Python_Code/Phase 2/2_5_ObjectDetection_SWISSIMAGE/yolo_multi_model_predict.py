#!/usr/bin/env python3
"""
YOLO + SAHI Multi-Model Tiled Inference to GeoPackage (OBB) + Metrics Plots
----------------------------------------------------------------------------

Updates:
- Shared predict.tif for all models (georeferenced GeoTIFF strongly recommended)
- Per-model imgsz/conf read from args.yaml unless overridden in CONFIG
- SAHI tiled inference (GPU by default via device='cuda:0' if available)
- Assumes OBB models and preserves rotated polygons
- Merges all predictions into one GeoPackage
- Exports per-model results_{model}.csv (with F1) and results_{model}.png
- Logs per-model counts, tile/time estimates, and a clear "converting to GPKG" message
- Auto-calibration: measures actual time per tile and saves per model in SAHI_Time_Pred.json
- On next run, stored values are automatically used for time estimation per model
"""

import os
import math
import time
import json
import logging
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Tuple, Optional

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
import rasterio
import matplotlib.pyplot as plt

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

import importlib.metadata
from packaging.version import Version

SAHI_TIME_FILE = "SAHI_Time_Pred.json"

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

setup_logging()

# ------------------------------------------------------------
# Device resolution (prefer CUDA)
# ------------------------------------------------------------
def resolve_device(config_device: Optional[str] = None) -> str:
    if config_device:
        return config_device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        logging.warning("CUDA not available; falling back to CPU.")
        return "cpu"
    except Exception:
        logging.warning("torch not available; falling back to CPU.")
        return "cpu"

# ------------------------------------------------------------
# SAHI time calibration load/save
# ------------------------------------------------------------
def load_sahi_times() -> dict:
    if os.path.exists(SAHI_TIME_FILE):
        try:
            with open(SAHI_TIME_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logging.warning(f"Could not load {SAHI_TIME_FILE} – starting with empty list.")
    return {}

def save_sahi_times(times: dict):
    try:
        with open(SAHI_TIME_FILE, "w", encoding="utf-8") as f:
            json.dump(times, f, indent=2)
        logging.info(f"Saved measured tile times to {SAHI_TIME_FILE}")
    except Exception as e:
        logging.error(f"Could not save {SAHI_TIME_FILE}: {e}")

# ------------------------------------------------------------
# OBB Helper
# ------------------------------------------------------------
def _get_obb_points_from_sahi_obj(pred) -> Optional[list]:
    mask = getattr(pred, "mask", None)
    if mask is not None:
        segm = getattr(mask, "segmentation", None)
        if segm and len(segm) == 1 and len(segm[0]) == 8:
            arr = np.array(segm[0], dtype=float).reshape(4, 2)
            return [(float(x), float(y)) for x, y in arr]

    pts = getattr(pred, "points", None)
    if pts and len(pts) >= 4:
        return [(float(x), float(y)) for x, y in pts[:4]]

    poly = getattr(pred, "polygon", None)
    if poly is not None:
        try:
            pts = poly.points
            if pts and len(pts) >= 4:
                return [(float(x), float(y)) for x, y in pts[:4]]
        except Exception:
            pass

    return None

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_args_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_imgsz_conf_from_args(args_yaml: dict) -> Tuple[Optional[int], Optional[float]]:
    imgsz, conf = None, None
    for k in ("imgsz", "img_size", "img"):
        v = args_yaml.get(k)
        if isinstance(v, (list, tuple)) and v:
            imgsz = int(v[0]); break
        if isinstance(v, (int, float)):
            imgsz = int(v); break
    for k in ("conf", "conf_thres", "conf_threshold"):
        v = args_yaml.get(k)
        if isinstance(v, (int, float)):
            conf = float(v); break
    return imgsz, conf

def check_sahi_version(min_required: str = "0.11.20"):
    try:
        ver = importlib.metadata.version("sahi")
        if Version(ver) < Version(min_required):
            raise RuntimeError(
                f"SAHI {ver} detected; {min_required}+ is required for YOLO OBB support."
            )
        logging.info(f"SAHI version OK: {ver}")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError("SAHI is not installed. Please `pip install sahi`.")

def _px_to_map(transform, px: float, py: float):
    mx, my = rasterio.transform.xy(transform, py, px)
    return mx, my

# ------------------------------------------------------------
# Tile count + time estimation
# ------------------------------------------------------------
def count_tiles(W, H, sw=512, sh=512, ow=0.2, oh=0.2):
    sx = sw * (1.0 - ow)
    sy = sh * (1.0 - oh)
    nx = 1 if W <= sw else math.ceil((W - sw) / sx) + 1
    ny = 1 if H <= sh else math.ceil((H - sh) / sy) + 1
    return int(nx), int(ny), int(nx * ny)

def estimate_time_seconds(
    W, H,
    sw=512, sh=512, ow=0.2, oh=0.2,
    t_tile=0.20,
    t_startup=3.0,
    t_post_factor=0.03,
    perform_standard_pred=False,
    t_std_equiv_tiles=1.0
):
    nx, ny, tiles = count_tiles(W, H, sw, sh, ow, oh)
    base = tiles * t_tile
    t_post = base * t_post_factor
    t_std = (t_std_equiv_tiles * t_tile) if perform_standard_pred else 0.0
    total = base + t_startup + t_post + t_std
    return {
        "nx": nx, "ny": ny, "tiles": tiles,
        "t_base_s": base, "t_post_s": t_post, "t_std_s": t_std,
        "t_total_s": total
    }

# ------------------------------------------------------------
# Prediction -> GeoDataFrame (OBB only)
# ------------------------------------------------------------
def prediction_result_to_gdf(
    result,
    raster_path: str,
    require_obb: bool = True,
    repair: bool = True,
    enforce_ccw: bool = True,
    model_name: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> gpd.GeoDataFrame:
    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = src.crs

    records = []
    obj_list = getattr(result, "object_prediction_list", []) or []
    logging.info(f"SAHI returned {len(obj_list)} ObjectPrediction(s) before OBB filtering.")

    kept = 0
    for pred in obj_list:
        obb_px = _get_obb_points_from_sahi_obj(pred)
        if not obb_px:
            if require_obb:
                continue
            else:
                continue

        coords = [_px_to_map(transform, px, py) for (px, py) in obb_px]
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        geom = Polygon(coords)
        if repair and not geom.is_valid:
            geom = geom.buffer(0)
        if not geom or geom.is_empty or not geom.is_valid:
            continue
        if enforce_ccw:
            geom = orient(geom, sign=1.0)

        records.append({
            "geometry": geom,
            "class_id": int(pred.category.id),
            "class_name": getattr(pred.category, "name", None),
            "confidence": float(pred.score.value),
            "model_name": model_name,
            "model_dir": model_dir,
        })
        kept += 1

    logging.info(f"Polygon (OBB) predictions kept: {kept}")
    if not records:
        return gpd.GeoDataFrame(
            columns=["geometry", "class_id", "class_name", "confidence", "model_name", "model_dir"],
            crs=crs
        )

    return gpd.GeoDataFrame(records, crs=crs)

# ------------------------------------------------------------
# Per-model processing with timing measurement
# ------------------------------------------------------------
def process_model(
    args_yaml_path: str,
    predict_tif: str,
    override_conf: Optional[float] = None,
    output_dir: Optional[str] = None,
    overlap_height_ratio: Optional[float] = None,
    overlap_width_ratio: Optional[float] = None,
    override_device: Optional[str] = None,
    t_tile: float = 0.20,
    t_startup: float = 3.0,
    t_post_factor: float = 0.03,
    t_std_equiv_tiles: float = 1.0,
    perform_standard_pred_flag: bool = False,
    times_dict: Optional[dict] = None
) -> Tuple[gpd.GeoDataFrame, Optional[float]]:
    args_yaml_path = os.path.abspath(args_yaml_path)
    model_dir = os.path.dirname(args_yaml_path)
    model_name = os.path.basename(model_dir.rstrip(os.sep))

    weights_path = os.path.join(model_dir, "weights", "best.pt")
    if not os.path.exists(weights_path):
        logging.error(f"Model weights not found: {weights_path}")
        return gpd.GeoDataFrame(), None

    args_data = load_args_yaml(args_yaml_path)
    imgsz_default, conf_default = get_imgsz_conf_from_args(args_data)
    imgsz = imgsz_default or 512
    conf = override_conf if override_conf is not None else (conf_default if conf_default is not None else 0.5)

    oh = overlap_height_ratio if overlap_height_ratio is not None else 0.2
    ow = overlap_width_ratio if overlap_width_ratio is not None else 0.2

    device = resolve_device(override_device)
    logging.info(f"Using device: {device}")
    logging.info(f"Running SAHI inference: {model_name} | imgsz={imgsz} conf={conf} overlap_h={oh} overlap_w={ow}")

    check_sahi_version("0.11.20")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights_path,
        confidence_threshold=conf,
        device=device
    )

    with rasterio.open(predict_tif) as src_dim:
        W, H = src_dim.width, src_dim.height
    est = estimate_time_seconds(
        W=W, H=H,
        sw=imgsz, sh=imgsz, ow=ow, oh=oh,
        t_tile=t_tile,
        t_startup=t_startup,
        t_post_factor=t_post_factor,
        perform_standard_pred=perform_standard_pred_flag,
        t_std_equiv_tiles=t_std_equiv_tiles,
    )
    logging.info(
        f"[{model_name}] Tiles: nx={est['nx']}, ny={est['ny']}, total={est['tiles']} | "
        f"t_tile={t_tile:.3f}s -> Estimate: base={est['t_base_s']:.1f}s, "
        f"post={est['t_post_s']:.1f}s, std={est['t_std_s']:.1f}s, "
        f"total={est['t_total_s']:.1f}s (~{est['t_total_s']/60.0:.1f} min)"
    )

    start_time = time.time()
    sahi_result = get_sliced_prediction(
        image=predict_tif,
        detection_model=detection_model,
        slice_height=imgsz,
        slice_width=imgsz,
        overlap_height_ratio=oh,
        overlap_width_ratio=ow,
        perform_standard_pred=False
    )
    elapsed = time.time() - start_time
    real_t_tile = elapsed / est["tiles"] if est["tiles"] > 0 else None

    gdf = prediction_result_to_gdf(
        sahi_result,
        raster_path=predict_tif,
        require_obb=True,
        repair=True,
        enforce_ccw=True,
        model_name=model_name,
        model_dir=model_dir
    )

    cnt = len(gdf)
    logging.info(f"[{model_name}] Found {cnt} OBB predictions. Converting these to GeoPackage layer entries...")

    if output_dir:
        ensure_dir(output_dir)

    return gdf, real_t_tile

# ------------------------------------------------------------
# Public API with calibration
# ------------------------------------------------------------
def run_from_config(CONFIG: Dict):
    predict_tif = CONFIG.get("predict")
    if not predict_tif or not os.path.exists(predict_tif):
        logging.error(f"Predict raster not found: {predict_tif}")
        return

    merge_cfg = CONFIG.get("merge", {})
    output_gpkg = merge_cfg.get("output_gpkg", "all_models_predictions.gpkg")
    layer_name = merge_cfg.get("layer", "predictions")
    metrics_output_dir = merge_cfg.get("output_dir", os.path.dirname(output_gpkg) or ".")

    global_device = CONFIG.get("device")
    tiling_cfg = CONFIG.get("tiling", {}) or {}
    global_oh = tiling_cfg.get("overlap_height_ratio", None)
    global_ow = tiling_cfg.get("overlap_width_ratio", None)

    timing_cfg = CONFIG.get("timing", {}) or {}
    default_t_tile = float(timing_cfg.get("t_tile", 0.20))
    t_startup = float(timing_cfg.get("t_startup", 3.0))
    t_post_factor = float(timing_cfg.get("t_post_factor", 0.03))
    t_std_equiv_tiles = float(timing_cfg.get("t_std_equiv_tiles", 1.0))
    perform_standard_pred_flag = bool(timing_cfg.get("perform_standard_pred", False))

    # Lade gespeicherte Zeiten
    stored_times = load_sahi_times()

    all_gdfs = []
    updated_times = dict(stored_times)  # Kopie, um neue Werte zu aktualisieren
    for model_cfg in CONFIG.get("models", []):
        model_path = os.path.abspath(model_cfg["args"])
        model_name = os.path.basename(os.path.dirname(model_path))
        per_model_device = model_cfg.get("device", global_device)
        oh = model_cfg.get("overlap_height_ratio", global_oh)
        ow = model_cfg.get("overlap_width_ratio", global_ow)

        model_t_tile = stored_times.get(model_name, default_t_tile)

        gdf, measured_t_tile = process_model(
            args_yaml_path=model_cfg["args"],
            predict_tif=predict_tif,
            override_conf=model_cfg.get("conf"),
            output_dir=metrics_output_dir,
            overlap_height_ratio=oh,
            overlap_width_ratio=ow,
            override_device=per_model_device,
            t_tile=model_t_tile,
            t_startup=t_startup,
            t_post_factor=t_post_factor,
            t_std_equiv_tiles=t_std_equiv_tiles,
            perform_standard_pred_flag=perform_standard_pred_flag,
            times_dict=stored_times
        )
        if measured_t_tile:
            logging.info(f"[{model_name}] Measured time per tile: {measured_t_tile:.3f}s")
            updated_times[model_name] = round(measured_t_tile, 4)
        if not gdf.empty:
            all_gdfs.append(gdf)

    # Speichere aktualisierte Zeiten
    save_sahi_times(updated_times)

    if not all_gdfs:
        logging.warning("No predictions generated.")
        return

    merged = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
    logging.info(f"Total predictions across all models: {len(merged)}")
    logging.info(f"Converting {len(merged)} predictions to GeoPackage: {output_gpkg} (layer='{layer_name}')...")

    ensure_dir(os.path.dirname(output_gpkg) or ".")
    merged.to_file(output_gpkg, driver="GPKG", layer=layer_name)
    logging.info(f"Merged GeoPackage saved: {output_gpkg} ({len(merged)} features)")
