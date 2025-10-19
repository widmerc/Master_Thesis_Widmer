#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mapillary Object Geolocation Pipeline (GeoParquet, LV95 CRS)
Date: 2025-08-14
Adjusted: 2025-08-16 (added parallel coordinate adjustment)
Adjusted: 2025-08-29 (improved multiprocessing stability and output schema)
Adjusted: 2025-09-02 (refactored code structure and default filters)

This script combines YOLO detection results with Mapillary image metadata
to compute georeferenced object positions (in LV95 / EPSG:2056). The pipeline
produces two GeoParquet outputs:

1. **joined_dataset.parquet** – original object positions joined with Mapillary
   metadata (LV95 geometry)
2. **adjusted_dataset.parquet** – adjusted object positions accounting for
   estimated depth-based offsets and camera field of view

Workflow:
- Step 1: Join YOLO (Parquet) and Mapillary (GPKG) data
- Step 2: Adjust object positions in parallel using compass angle and depth class
- Step 3: Save joined and adjusted datasets as GeoParquet (EPSG:2056)

Supports efficient multiprocessing, pandas–geopandas integration, and
robust geometry parsing (JSON, WKT, or literal structures).
"""

import ast
import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*GeoDataFrame.swapaxes.*")

# =====================================
# Defaults
# =====================================
DEFAULT_Z_CLASS_OFFSETS = {
    "very near": 5,
    "near": 7.5,
    "medium": 20.0,
    "far": 50.0,
    "very far": 100.0,
}

DEFAULT_EXCLUDE_LABELS = [
    0, 1, 59, 60, 61, 62, 63, 65, 66, 69, 73, 76, 78, 79, 83, 85, 88, 104, 106, 109, 117, 118, 119, 120, 121, 122
]

DEFAULT_EXCLUDE_Z = ["far", "very far"]

REQUIRED_GPKG_COLS = [
    "id", "is_pano", "sequence_id", "computed_altitude",
    "computed_compass_angle", "computed_geometry", "computed_rotation", "is_blurry"
]

# =====================================
# Helpers
# =====================================
def parse_geometry(val):
    if pd.isna(val):
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        try:
            return json.loads(val)
        except Exception:
            return None

def to_shape(gdict):
    return shape(gdict) if gdict is not None else None

def bbox_center_x(bbox):
    try:
        return (bbox[0] + bbox[2]) / 2.0
    except Exception:
        return None

def safe_bbox_to_list(b):
    if b is None or (isinstance(b, float) and math.isnan(b)):
        return None
    if isinstance(b, (list, tuple)):
        return list(b)
    if isinstance(b, np.ndarray):
        return b.tolist()
    if isinstance(b, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                out = parser(b)
                if isinstance(out, (list, tuple, np.ndarray)):
                    return list(out)
            except Exception:
                pass
    return None

def move_point_lv95(point, angle_deg, distance_m):
    if point is None or distance_m == 0 or pd.isna(angle_deg):
        return point
    rad = math.radians(float(angle_deg))
    dx = distance_m * math.sin(rad)
    dy = distance_m * math.cos(rad)
    return Point(point.x + dx, point.y + dy)

# =====================================
# Step 1: Build joined dataset (GeoParquet)
# =====================================
def build_joined_dataset(yolo_path, gpkg_path, joined_file, sample=None):
    logging.info(f"Reading Parquet (YOLO): {yolo_path}")
    df_yolo = pd.read_parquet(yolo_path)

    if sample is not None:
        df_yolo = df_yolo.sample(sample)

    logging.info(f"Reading GPKG (Mapillary): {gpkg_path}")
    df_gpkg = gpd.read_file(gpkg_path, columns=REQUIRED_GPKG_COLS)

    # Filters
    df_yolo = df_yolo[~df_yolo["label"].isin(DEFAULT_EXCLUDE_LABELS)]
    df_yolo = df_yolo[~df_yolo["z_class"].isin(DEFAULT_EXCLUDE_Z)]

    # Join
    df_yolo["image_id"] = df_yolo["image_id"].astype(str).str.strip()
    df_gpkg["id"] = df_gpkg["id"].astype(str).str.strip()

    df_joined = df_yolo.merge(
        df_gpkg,
        left_on="image_id",
        right_on="id",
        how="left",
        validate="many_to_one"
    )

    # Create geometry in EPSG:2056
    geoms = df_joined["computed_geometry"].map(parse_geometry).map(to_shape)
    gdf = gpd.GeoDataFrame(
        df_joined,
        geometry=geoms,
        crs="EPSG:4326"
    ).to_crs(epsg=2056)

    os.makedirs(os.path.dirname(joined_file), exist_ok=True)
    gdf.to_parquet(joined_file, engine="pyarrow", index=False)
    logging.info(f"Joined dataset saved: {joined_file} ({len(gdf)} rows)")

# =====================================
# Step 2: Adjust coordinates in a chunk
# =====================================
def adjust_chunk(pdf, image_width, fov):
    gdf = gpd.GeoDataFrame(
        pdf,
        geometry=pdf.geometry,
        crs="EPSG:2056"
    )

    x_adj, y_adj = [], []
    for row in gdf.itertuples():
        if row.is_pano:
            pt = row.geometry
        else:
            angle = row.computed_compass_angle
            bbox_val = safe_bbox_to_list(row.bbox)
            if bbox_val and len(bbox_val) >= 4:
                cx = bbox_center_x(bbox_val)
                if cx is not None:
                    offset = (cx / float(image_width) - 0.5) * fov
                    if angle is not None and not pd.isna(angle):
                        angle = float(angle) + offset
            dist = DEFAULT_Z_CLASS_OFFSETS.get(row.z_class, 0.0)
            pt = move_point_lv95(row.geometry, angle, dist)

        x_adj.append(pt.x if pt else None)
        y_adj.append(pt.y if pt else None)

    gdf["x_adj"] = x_adj
    gdf["y_adj"] = y_adj
    gdf["geometry"] = [Point(x, y) if x is not None and y is not None else None
                       for x, y in zip(x_adj, y_adj)]
    return gdf

# =====================================
# Step 3: Process dataset in parallel (GeoParquet)
# =====================================
def process_dataset(in_parquet, out_file, image_width, fov, max_workers=8):
    gdf = gpd.read_parquet(in_parquet)

    chunks = np.array_split(gdf, max_workers or os.cpu_count())
    logging.info(f"Processing {len(gdf)} rows in {len(chunks)} chunks...")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(adjust_chunk, chunks, [image_width]*len(chunks), [fov]*len(chunks)))

    gdf_out = pd.concat(results, ignore_index=True)
    gdf_out.to_parquet(out_file, engine="pyarrow", index=False)
    logging.info(f"Adjusted dataset saved: {out_file} ({len(gdf_out)} rows)")

# =====================================
# Main
# =====================================
def main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    joined_file = os.path.join(args.out_dir, "joined_dataset.parquet")
    adjusted_file = os.path.join(args.out_dir, "adjusted_dataset.parquet")

    build_joined_dataset(args.parquet, args.gpkg, joined_file, sample=getattr(args, "sample", None))
    process_dataset(joined_file, adjusted_file, args.image_width, args.fov, max_workers=args.workers)
