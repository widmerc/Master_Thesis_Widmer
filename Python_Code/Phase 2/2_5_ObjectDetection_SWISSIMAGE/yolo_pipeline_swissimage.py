
"""
yolo_pipeline_swissimage.py
---------------------------
End-to-end pipeline for SwissImage: Download -> Mosaic -> Clip -> Tiles -> Dataset -> Training -> Prediction -> Export.

Contains:
- Parallel, robust downloader (A)
- Clip step for training extent and optionally separate inference extent (B)
- Improved TileExtractor with 2D shifts, NoData skip, clean IO (C)
- Various bug fixes (detect/obb val path, missing polygon import, 16bit->8bit export, CRS handling)
- Simple config-based pipeline class
"""

import os
import re
import gc
import io
import math
import time
import random
import shutil
import yaml
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import random

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.windows import Window
from rasterio.transform import rowcol
from shapely.geometry import box as shapely_box, Polygon, box
from shapely.ops import unary_union


from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from multiprocessing import Pool
from functools import partial


try:
    from ultralytics import YOLO
    _HAVE_ULTRALYTICS = True
except Exception:
    _HAVE_ULTRALYTICS = False

# =============================================================================
# Logging
# =============================================================================

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

setup_logging()

# =============================================================================
# Utilities
# =============================================================================

def make_requests_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def to_uint8_rgb(arr: np.ndarray, nodata=None) -> np.ndarray:
    """
    Expects raster as (bands, H, W). Takes the first 3 bands.
    - uint8: pass through
    - uint16: simple 16->8 scaling (/256)
    - float: per-channel stretch (2..98 percentiles), ignoring 0/nodata
    """
    if arr.ndim != 3 or arr.shape[0] < 3:
        raise ValueError("Expect array (bands, H, W) with at least 3 bands.")

    x = arr[:3]

    # Mask for nodata (optional)
    mask = None
    if nodata is not None:
        mask = (x[0] == nodata) & (x[1] == nodata) & (x[2] == nodata)

    if x.dtype == np.uint8:
        rgb = np.transpose(x, (1, 2, 0))
        return rgb.copy()

    x = x.astype(np.float32)

    if x.dtype == np.uint16 or arr.dtype == np.uint16:
        x = x / 256.0  # simple 16->8 scaling
    else:
        # percentile stretch per channel
        for b in range(3):
            band = x[b]
            v = band[~mask] if mask is not None else band
            v = v[v > 0]  # avoid 0 background
            if v.size:
                p2, p98 = np.percentile(v, [2, 98])
                if p98 > p2:
                    band = (band - p2) / (p98 - p2) * 255.0
                    x[b] = band

    x = np.clip(x, 0, 255).astype(np.uint8)
    return np.transpose(x, (1, 2, 0))


import re

def parse_resolution_to_int(resolution_value):
    """ '10 cm' -> 10, '25cm' -> 25, 10 -> 10 """
    if resolution_value is None:
        return None
    s = str(resolution_value)
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else None

def extract_year(datenstand_val):
    """ '2024' / '2024-05-01' / '20240501' -> '2024' """
    if not datenstand_val:
        return None
    m = re.search(r'(20\d{2})', str(datenstand_val))
    return m.group(1) if m else None

def normalize_tile_id_strict(tile_id: str):
    """
    Enforces exactly four digits, dash, four digits.
    Accepts underscore/space as separators and converts to '-'.
    Examples:
      '2647_1236' -> '2647-1236'
      '2647-1236' -> '2647-1236'
      '2647 1236' -> '2647-1236'
    """
    if tile_id is None:
        return None
    s = str(tile_id).strip().replace('_', '-').replace(' ', '-')
    m = re.fullmatch(r'(\d{4})-(\d{4})', s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    if re.fullmatch(r'\d{8}', s):
        return f"{s[:4]}-{s[4:]}"
    return None

def build_swissimage_url(datenstand, tile_id, resolution, epsg=2056):
    year = extract_year(datenstand)
    res  = parse_resolution_to_int(resolution)
    kid  = normalize_tile_id_strict(tile_id)
    if not (year and res and kid):
        return None
    base   = f"https://data.geo.admin.ch/ch.swisstopo.swissimage-dop{res}"
    folder = f"swissimage-dop{res}_{year}_{kid}"
    fname  = f"{folder}_0.1_{epsg}.tif"
    return f"{base}/{folder}/{fname}"

# =============================================================================
# A) Downloader + Mosaic builder
# =============================================================================

class SwissImageMosaikDownloader:
    def __init__(self, raster_path: str, fussgaenger_path: str, output_dir: str):
        self.raster_path = raster_path
        self.fussgaenger_path = fussgaenger_path
        self.output_dir = output_dir
        self.download_links: List[str] = []
        self.raster: Optional[gpd.GeoDataFrame] = None
        self.fussgaenger: Optional[gpd.GeoDataFrame] = None
        self.raster_relevant: Optional[gpd.GeoDataFrame] = None

    def load_data(self):
        self.raster = gpd.read_file(self.raster_path)
        self.fussgaenger = gpd.read_file(self.fussgaenger_path)
        if self.raster.crs != self.fussgaenger.crs:
            self.fussgaenger = self.fussgaenger.to_crs(self.raster.crs)
        logging.info("Raster features: %d | Fussgaenger features: %d", len(self.raster), len(self.fussgaenger))

    def identify_relevant_tiles(self):
        fuss_bounds = self.fussgaenger.unary_union.envelope
        raster_bbox = self.raster[self.raster.intersects(fuss_bounds)]
        logging.info("Raster filtered with bbox: %d -> %d", len(self.raster), len(raster_bbox))
        join = gpd.sjoin(self.fussgaenger, raster_bbox, how='inner', predicate='intersects')
        relevante_kacheln = join['id'].unique()
        self.raster_relevant = raster_bbox[
            (raster_bbox.get('resolution') == '10 cm') &
            (raster_bbox['id'].isin(relevante_kacheln))
        ]
        logging.info("Relevant tiles (10cm): %d", len(self.raster_relevant))

    def generate_download_links(self):
        self.download_links.clear()
        for _, row in self.raster_relevant.iterrows():
            url = build_swissimage_url(
                datenstand=row.get('datenstand'),
                tile_id=row.get('id'),
                resolution=row.get('resolution')
            )
            if url:
                self.download_links.append(url)
            else:
                logging.warning(
                    "Invalid metadata for URL: id=%r datenstand=%r resolution=%r",
                    row.get('id'), row.get('datenstand'), row.get('resolution')
                )
        logging.info("Download links generated: %d", len(self.download_links))

    def _download_one(self, sess: requests.Session, url: str, outpath: str) -> bool:
        try:
            if os.path.exists(outpath):
                return True
            h = sess.head(url, timeout=30)
            if h.status_code != 200:
                logging.warning("HEAD != 200 fuer %s (%s)", url, h.status_code)
                logging.warning("HEAD != 200 for %s (%s)", url, h.status_code)
                return False
            r = sess.get(url, stream=True, timeout=120)
            if r.status_code != 200:
                logging.warning("GET != 200 fuer %s (%s)", url, r.status_code)
                logging.warning("GET != 200 for %s (%s)", url, r.status_code)
                return False
            with open(outpath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1<<15):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            logging.exception("Fehler beim Download %s: %s", url, e)
            logging.exception("Error downloading %s: %s", url, e)
            return False

    def download_tiles(self, max_workers: int = 8) -> int:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        ensure_dir(self.output_dir)
        ok = 0
        sess = make_requests_session()
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for url in self.download_links:
                out = os.path.join(self.output_dir, os.path.basename(url))
                tasks.append(ex.submit(self._download_one, sess, url, out))
            for t in as_completed(tasks):
                if t.result():
                    ok += 1
        logging.info("Successful downloads: %d / %d", ok, len(self.download_links))
        return ok

    def create_mosaik(self, output_mosaik_path: str):
        ensure_dir(os.path.dirname(output_mosaik_path) or ".")
        tiffs = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f.lower().endswith(".tif")]
        if not tiffs:
            logging.warning("No TIFF files found for mosaic in %s.", self.output_dir)
            return
        logging.info("Loading %d rasters for mosaic ...", len(tiffs))
        mosaik = None
        raster_objs = []
        try:
            for fp in tiffs:
                raster_objs.append(rasterio.open(fp))
            mosaik, out_transform = merge(raster_objs)
            meta = raster_objs[0].meta.copy()
            meta.update({
                "driver": "GTiff",
                "count": mosaik.shape[0],
                "height": mosaik.shape[1],
                "width": mosaik.shape[2],
                "transform": out_transform,
                "tiled": True,
                "compress": "LZW",
                "BIGTIFF": "IF_SAFER",
                "blockxsize": 512,
                "blockysize": 512
            })
            with rasterio.open(output_mosaik_path, 'w', **meta) as dst:
                dst.write(mosaik)
            logging.info("Mosaic saved: %s", output_mosaik_path)
        finally:
            for src in raster_objs:
                try:
                    src.close()
                except Exception:
                    pass
            if mosaik is not None:
                del mosaik
            gc.collect()

    def run_all(self, output_mosaik_path: str) -> str:
        self.load_data()
        self.identify_relevant_tiles()
        self.generate_download_links()
        self.download_tiles()
        self.create_mosaik(output_mosaik_path)
        return output_mosaik_path


# =============================================================================
# B) Clipping
# =============================================================================

def clip_mosaic_to_extent(mosaic_path: str, extent_gpkg: str, output_path: str) -> str:
    extent = gpd.read_file(extent_gpkg)
    with rasterio.open(mosaic_path) as src:
        if extent.crs != src.crs:
            extent = extent.to_crs(src.crs)
        out_image, out_transform = mask(src, extent.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width":  out_image.shape[2],
            "transform": out_transform,
            "tiled": True,
            "compress": "LZW",
            "BIGTIFF": "IF_SAFER",
            "blockxsize": 512,
            "blockysize": 512
        })
        if "nodata" not in out_meta or out_meta["nodata"] is None:
            out_meta["nodata"] = 0
    ensure_dir(os.path.dirname(output_path) or ".")
    with rasterio.open(output_path, 'w', **out_meta) as dest:
        dest.write(out_image)
    logging.info("Mosaic clipped: %s", output_path)
    return output_path

# =============================================================================
# Bounding Boxes
# =============================================================================

class BoundingBoxGenerator:
    def __init__(self, input_fp: str, output_path: str, box_width: float = 5.0, use_oriented_boxes: bool = False, buffer_config: Optional[Dict[str, float]] = None):
        self.input_fp = input_fp
        self.output_path = output_path
        self.box_width = box_width
        self.use_oriented_boxes = use_oriented_boxes
        self.buffer_config = buffer_config or {}

    def create_aligned_box(self, center, width):
        dx = width / 2
        dy = width / 2
        return box(center.x - dx, center.y - dy, center.x + dx, center.y + dy)

    def generate_boxes(self):
        data_gdf = gpd.read_file(self.input_fp).to_crs(epsg=2056)
        if "Art" not in data_gdf.columns:
            raise ValueError("Column 'Art' is missing in the input file.")
        if not self.buffer_config:
            unique_arten = data_gdf["Art"].unique()
            self.buffer_config = {art: self.box_width for art in unique_arten}

        all_boxes = []
        for art, width in self.buffer_config.items():
            subset = data_gdf[data_gdf["Art"] == art].copy()
            if subset.empty:
                logging.info("Skipping '%s' (no geometries)", art)
                continue
            logging.info("Processing '%s' with buffer = %.2fm (oriented=%s)", art, width, self.use_oriented_boxes)
            shape_type = subset.geometry.iloc[0].geom_type

            if self.use_oriented_boxes:
                # Erwartet LineStrings -> Puffer als Rechteck (cap_style=2 = flat)
                if shape_type != 'LineString':
                    logging.warning("%s: Expected LineString, got: %s. Using envelope as fallback.", art, shape_type)
                    subset["bbox_geom"] = subset.geometry.envelope.buffer(0)
                else:
                    subset["bbox_geom"] = subset.geometry.buffer(width / 2, cap_style=2, join_style=2)
            else:
                # AABB aus Points/Lines/Polygone
                if shape_type == 'Point':
                    subset["bbox_geom"] = subset.geometry.apply(lambda g: self.create_aligned_box(g, width))
                else:
                    subset["bbox_geom"] = subset.geometry.envelope.buffer(0)

            boxes_gdf = gpd.GeoDataFrame({
                "original_id": subset.index,
                "Art": art,
                "class": art,  # fuer DatasetPreparer
                "geometry": subset["bbox_geom"]
            }, crs=subset.crs)
            all_boxes.append(boxes_gdf)

        if all_boxes:
            final_gdf = gpd.GeoDataFrame(pd.concat(all_boxes, ignore_index=True), crs=data_gdf.crs)
            ensure_dir(os.path.dirname(self.output_path) or ".")
            final_gdf.to_file(self.output_path, driver="GPKG", layer="bounding_boxes")
            logging.info("Export finished: %s", self.output_path)
        else:
            logging.warning("No valid geometries found.")

# =============================================================================
# C) Tile Extraction
# =============================================================================

class TileExtractor:
    def __init__(self, mosaik_path: str, boxes_path: Optional[str] = None,
                 output_dir: str = './tiles', tile_size: int = 512,
                 shift_fractions: Optional[Sequence[float]] = None,
                 num_workers: int = os.cpu_count()):
        self.mosaik_path = mosaik_path
        self.boxes_path = boxes_path
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.shift_fractions = list(shift_fractions) if shift_fractions else [0.0]
        self.filter_with_boxes = boxes_path is not None
        self.boxes_gdf: Optional[gpd.GeoDataFrame] = None
        self.num_workers = num_workers
        ensure_dir(self.output_dir)

    def load_boxes(self, crs):
        """Loads bounding boxes if present and ensures CRS alignment."""
        if self.filter_with_boxes:
            self.boxes_gdf = gpd.read_file(self.boxes_path)
            if self.boxes_gdf.crs != crs:
                self.boxes_gdf = self.boxes_gdf.to_crs(crs)
            logging.info("%d bounding boxes loaded", len(self.boxes_gdf))

    @staticmethod
    def _process_tile(mosaik_path, x_off, y_off, tile_size, sx, sy,
                      boxes_serialized, filter_with_boxes, output_dir):
        """Worker function to extract a single tile."""
        import rasterio
        from shapely.geometry import shape as shapely_shape, box as shapely_box
        import pickle

        boxes_gdf = pickle.loads(boxes_serialized) if boxes_serialized else None

        with rasterio.open(mosaik_path) as src:
            window = Window(x_off, y_off, tile_size, tile_size)
            w_transform = src.window_transform(window)

            # Tile geometry
            minx, miny = w_transform * (0, tile_size)
            maxx, maxy = w_transform * (tile_size, 0)
            tile_geom = shapely_box(minx, miny, maxx, maxy)

            if filter_with_boxes:
                if not boxes_gdf.intersects(tile_geom).any():
                    return None

            # Read tile & check for NoData/emptiness
            tile = src.read(window=window)
            nodata = src.nodata
            if nodata is not None and np.all(tile == nodata):
                return None
            if np.all(tile == 0):
                return None

            meta = src.meta.copy()
            meta.update({
                "height": tile_size,
                "width": tile_size,
                "transform": w_transform
            })

            tile_filename = f"tile_{y_off}_{x_off}_sy{int(sy*100)}_sx{int(sx*100)}.tif"
            tile_path = os.path.join(output_dir, tile_filename)
            with rasterio.open(tile_path, 'w', **meta) as dst:
                dst.write(tile)

        return tile_path

    def extract_tiles(self) -> int:
        """Extracts all tiles using multiprocessing."""
        import pickle

        tasks = []
        with rasterio.open(self.mosaik_path) as src:
            width, height = src.width, src.height
            self.load_boxes(src.crs)

            # Pickle boxes for workers
            boxes_serialized = pickle.dumps(self.boxes_gdf) if self.filter_with_boxes else None

            for y in range(0, height, self.tile_size):
                for x in range(0, width, self.tile_size):
                    for sy in self.shift_fractions:
                        for sx in self.shift_fractions:
                            x_off = x + int(self.tile_size * sx)
                            y_off = y + int(self.tile_size * sy)
                            if x_off + self.tile_size > width or y_off + self.tile_size > height:
                                continue
                            tasks.append((
                                self.mosaik_path, x_off, y_off, self.tile_size,
                                sx, sy, boxes_serialized, self.filter_with_boxes, self.output_dir
                            ))

        logging.info("Starting extraction of %d tiles with %d worker processes",
                     len(tasks), self.num_workers)

        with Pool(processes=self.num_workers) as pool:
            results = list(pool.starmap(self._process_tile, tasks))

        # Count successful tiles
        results = [r for r in results if r]
        logging.info("%d tiles saved in: %s", len(results), self.output_dir)
        return len(results)

# =============================================================================
# Dataset Builder
# =============================================================================

class DatasetPreparer:
    def __init__(self, tile_dir: str, boxes_path: str, base_output: str, use_oriented_boxes: bool, val_split: float = 0.2, class_column: str = "class"):
        self.tile_dir = os.path.abspath(tile_dir)
        self.boxes_path = boxes_path
        self.base_output = os.path.abspath(base_output)
        self.val_split = val_split
        self.train_images_dir = os.path.join(self.base_output, 'train', 'images')
        self.train_labels_dir = os.path.join(self.base_output, 'train', 'labels')
        self.val_images_dir = os.path.join(self.base_output, 'valid', 'images')
        self.val_labels_dir = os.path.join(self.base_output, 'valid', 'labels')
        self.use_oriented_boxes = use_oriented_boxes
        self.class_column = class_column
        self.class_names: List[str] = []
        self.class_to_id: Dict[str, int] = {}

        for d in [self.train_images_dir, self.train_labels_dir, self.val_images_dir, self.val_labels_dir]:
            ensure_dir(d)

    def create_dataset(self):
        boxes_gdf = gpd.read_file(self.boxes_path)
        tile_files = [os.path.join(self.tile_dir, f) for f in os.listdir(self.tile_dir) if f.lower().endswith(".tif")]
        logging.info("Found tiles: %d", len(tile_files))

        if self.class_column not in boxes_gdf.columns:
            raise ValueError(f"Column '{self.class_column}' not present in geodata.")

        unique_classes = sorted(boxes_gdf[self.class_column].dropna().unique().tolist())
        self.class_names = unique_classes
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        logging.info("Found classes (%d): %s", len(self.class_names), ", ".join(self.class_names))

        # Transform CRS once to tile CRS (taken from first tile)
        if not tile_files:
            logging.warning("No tiles found, aborting.")
            return
        with rasterio.open(tile_files[0]) as src0:
            tile_crs = src0.crs
        if boxes_gdf.crs != tile_crs:
            boxes_gdf = boxes_gdf.to_crs(tile_crs)

        saved_tile_count = 0
        for tile_path in tile_files:
            with rasterio.open(tile_path) as src:
                transform = src.transform
                tile_w, tile_h = src.width, src.height
                tile = src.read()
                tile_rgb = to_uint8_rgb(tile, nodata=src.nodata)

                # Skip if entire tile is empty (only zeros)
                if tile_rgb.max() == 0:
                    continue

                # Tile geometry
                minx, miny = transform * (0, tile_h)
                maxx, maxy = transform * (tile_w, 0)
                tile_geom = shapely_box(minx, miny, maxx, maxy)

                intersecting = boxes_gdf[boxes_gdf.intersects(tile_geom)].copy()
                if intersecting.empty:
                    continue

                label_lines = []
                for _, row in intersecting.iterrows():
                    geom = row.geometry.intersection(tile_geom)
                    if geom.is_empty:
                        continue

                    class_id = self.class_to_id.get(row[self.class_column], None)
                    if class_id is None:
                        continue

                    if self.use_oriented_boxes:
                        if geom.geom_type != "Polygon":
                            continue
                        coords = list(geom.exterior.coords)[:4]
                        if len(coords) != 4:
                            continue
                        norm = []
                        for x_abs, y_abs in coords:
                            py, px = rowcol(transform, x_abs, y_abs)
                            norm_x = px / tile_w
                            norm_y = py / tile_h
                            norm.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
                        if len(norm) == 8:
                            label_lines.append(f"{class_id} " + " ".join(norm))
                    else:
                        xmin, ymin, xmax, ymax = geom.bounds
                        py_min, px_min = rowcol(transform, xmin, ymax)
                        py_max, px_max = rowcol(transform, xmax, ymin)
                        x_center = (px_min + px_max) / 2 / tile_w
                        y_center = (py_min + py_max) / 2 / tile_h
                        w = abs(px_max - px_min) / tile_w
                        h = abs(py_max - py_min) / tile_h
                        label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                if label_lines:
                    base_name = os.path.splitext(os.path.basename(tile_path))[0]
                    img_out = os.path.join(self.train_images_dir, f"{base_name}.jpg")
                    lbl_out = os.path.join(self.train_labels_dir, f"{base_name}.txt")
                    Image.fromarray(tile_rgb).save(img_out, quality=95, subsampling=0)
                    with open(lbl_out, "w", encoding="utf-8") as f:
                        f.write("\n".join(label_lines))
                    saved_tile_count += 1

        logging.info("%d training images with labels created.", saved_tile_count)


        train_imgs = [f for f in os.listdir(self.train_images_dir) if f.lower().endswith(".jpg")]
        if train_imgs:
            sample_img_name = random.choice(train_imgs)
            sample_img_path = os.path.join(self.train_images_dir, sample_img_name)
            sample_lbl_path = os.path.join(self.train_labels_dir, sample_img_name.replace(".jpg", ".txt"))

            img = np.array(Image.open(sample_img_path))
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img)

            if os.path.exists(sample_lbl_path):
                with open(sample_lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_center, y_center, w, h = map(float, parts[1:5])
                            # Convert bounding box normalized coordinates back to pixels
                            x_center *= img.shape[1]
                            y_center *= img.shape[0]
                            w *= img.shape[1]
                            h *= img.shape[0]
                            rect_x = x_center - w / 2
                            rect_y = y_center - h / 2
                            ax.add_patch(plt.Rectangle(
                                (rect_x, rect_y), w, h,
                                fill=False, color="red", linewidth=2
                            ))
                            ax.text(rect_x, rect_y - 5, f"{cls_id}", color="yellow", fontsize=10)
            plt.title(f"Sample: {sample_img_name}")
            plt.show()


        self.split_validation()
        self.create_yaml()

    def split_validation(self):
        image_files = [os.path.join(self.train_images_dir, f) for f in os.listdir(self.train_images_dir) if f.lower().endswith(".jpg")]
        if not image_files:
            logging.warning("No training images found for split.")
            return
        val_count = int(len(image_files) * self.val_split)
        val_images = set(random.sample(image_files, max(1, val_count)))
        for img_path in list(image_files):
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(self.train_labels_dir, f"{base}.txt")
            if img_path in val_images:
                shutil.move(img_path, os.path.join(self.val_images_dir, os.path.basename(img_path)))
                if os.path.exists(lbl_path):
                    shutil.move(lbl_path, os.path.join(self.val_labels_dir, os.path.basename(lbl_path)))
        logging.info("%d files moved to 'valid'.", len(val_images))

    def create_yaml(self):
        data_yaml = {
            'path': str(self.base_output),
            'train': 'train/images',
            'val': 'valid/images',
            'names': {idx: name for idx, name in enumerate(self.class_names)}
        }
        yaml_path = os.path.join(os.path.dirname(self.base_output), 'data.yaml')
        with open(yaml_path, 'w', encoding="utf-8") as f:
            yaml.dump(data_yaml, f, sort_keys=False, allow_unicode=True)
        logging.info("data.yaml saved at: %s", yaml_path)

# =============================================================================
# YOLO Training / Prediction / Export
# =============================================================================

class YOLOTraining:
    def __init__(self, base_output_dir: str, use_oriented_bounding_box: bool, model_dir: str = 'Model_Fussgaenger_NO_TUNE'):
        self.base_output_dir = os.path.abspath(base_output_dir)
        self.model_dir = model_dir
        self.prediction_run_dir: Optional[str] = None
        self.use_oriented_bounding_box = use_oriented_bounding_box
        self.bounding_mosaik: Optional[gpd.GeoDataFrame] = None

        if not _HAVE_ULTRALYTICS:
            logging.warning("Ultralytics not available. Install with: pip install ultralytics")

    def load_class_names(self, yaml_path) -> Dict[int, str]:
        yaml_path = os.path.abspath(yaml_path)
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")
        with open(yaml_path, 'r', encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # keys may be int or str
        names = {}
        for k, v in data['names'].items():
            try:
                names[int(k)] = v
            except Exception:
                # falls liste statt dict
                pass
        if not names and isinstance(data['names'], list):
            names = {i: n for i, n in enumerate(data['names'])}
        return names

    def train_model(self, model_path: str, use_tuning: bool, imgsz: int, batch_size: int, final_epochs: int, tune_epochs: int = 30, tune_iterations: int = 20):
        if not _HAVE_ULTRALYTICS:
            raise RuntimeError("Ultralytics not available.")

        data_yaml = os.path.abspath(os.path.join(self.base_output_dir, 'data.yaml'))
        model = YOLO(model_path)

        if use_tuning:
            logging.info("Starting hyperparameter tuning...")
            model.tune(
                data=data_yaml, imgsz=imgsz,
                epochs=tune_epochs, iterations=tune_iterations,
                plots=True, val=True, device=0
            )
            best_hyp_path = './runs/detect/tune/best_hyperparameters.yaml'
            with open(best_hyp_path, 'r', encoding="utf-8") as file:
                best_hyp = yaml.safe_load(file)

            model.train(
                data=data_yaml, imgsz=imgsz,
                epochs=final_epochs, device=0, batch=batch_size,
                name='Model_Fussgaenger_TUNED',
                **best_hyp
            )
            self.model_dir = 'Model_Fussgaenger_TUNED'
        else:
            model.train(
                data=data_yaml, imgsz=imgsz,
                epochs=final_epochs, device=0, batch=batch_size,
                name=self.model_dir, patience=10
            )

        # Correct validation path depending on task
        if self.use_oriented_bounding_box:
            model.val(
                model=f"runs/obb/{self.model_dir}/weights/best.pt",
                data=data_yaml, imgsz=imgsz
            )
        else:
            model.val(
                model=f"runs/detect/{self.model_dir}/weights/best.pt",
                data=data_yaml, imgsz=imgsz
            )
        return model, self.model_dir

    def create_prediction_tiles(self, mosaik_path: str, output_dir: str, tile_size: int):
        logging.info("Creating prediction tiles from mosaic ...")
        ensure_dir(output_dir)
        tile_id = 0
        with rasterio.open(mosaik_path) as src:
            n_cols = math.ceil(src.width / tile_size)
            n_rows = math.ceil(src.height / tile_size)
            logging.info("Expected number of tiles: %d (%d x %d)", n_cols * n_rows, n_cols, n_rows)

            bounds = src.bounds
            crs = src.crs
            mosaik_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            self.bounding_mosaik = gpd.GeoDataFrame([{"geometry": mosaik_geom}], crs=crs)

            for y in range(0, src.height, tile_size):
                for x in range(0, src.width, tile_size):
                    ww = min(tile_size, src.width - x)
                    wh = min(tile_size, src.height - y)
                    window = Window(x, y, ww, wh)
                    w_transform = src.window_transform(window)

                    tile_data = src.read(window=window)
                    meta = src.meta.copy()
                    meta.update({
                        "height": wh,
                        "width": ww,
                        "transform": w_transform
                    })
                    tile_filename = f"tile_{tile_id}.tif"
                    tile_path = os.path.join(output_dir, tile_filename)
                    with rasterio.open(tile_path, 'w', **meta) as dst:
                        dst.write(tile_data)
                    tile_id += 1
        logging.info("Prediction tiles saved: %s (n=%d)", output_dir, tile_id)

    def predict_on_directory(self, model, source_dir: str, imgsz: int = 512, conf: float = 0.4, save_images: bool = True):
        if not _HAVE_ULTRALYTICS:
            raise RuntimeError("Ultralytics not available.")
        logging.info("Starting inference in %s ...", source_dir)
        results = model.predict(
            source=source_dir,
            imgsz=imgsz,
            device=0,
            conf=conf,
            save=save_images,
            save_txt=True,
            save_conf=True,
            name=f"{self.model_dir}_PRED"
        )
        if self.use_oriented_bounding_box:
            self.prediction_run_dir = os.path.join("runs", "obb", f"{self.model_dir}_PRED")
        else:
            self.prediction_run_dir = os.path.join("runs", "detect", f"{self.model_dir}_PRED")
        logging.info("Inference complete. Output: %s", self.prediction_run_dir)
        return results

    def convert_predictions_to_geopackage(self, image_dir: str, output_path: str):
        if not self.prediction_run_dir:
            raise ValueError("Inference not yet run or path not set.")
        label_dir = os.path.join(self.prediction_run_dir, "labels")
        geoms = []
        raster_crs = None

        for label_file in os.listdir(label_dir):
            if not label_file.endswith(".txt"):
                continue
            label_path = os.path.join(label_dir, label_file)
            image_name = label_file.replace(".txt", ".tif")
            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):
                continue

            with rasterio.open(image_path) as src:
                width, height = src.width, src.height
                transform = src.transform
                raster_crs = raster_crs or src.crs

                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        # Axis-aligned
                        if len(parts) == 6:
                            cls_id, x_center, y_center, w, h, conf = map(float, parts)
                            x_min = (x_center - w / 2) * width
                            x_max = (x_center + w / 2) * width
                            y_min = (y_center - h / 2) * height
                            y_max = (y_center + h / 2) * height
                            topleft = rasterio.transform.xy(transform, y_min, x_min, offset='ul')
                            bottomright = rasterio.transform.xy(transform, y_max, x_max, offset='lr')
                            geom = box(topleft[0], bottomright[1], bottomright[0], topleft[1])
                            geoms.append({
                                "geometry": geom,
                                "class_id": int(cls_id),
                                "confidence": float(conf),
                                "source_image": image_name
                            })
                        # OBB: x1 y1 x2 y2 x3 y3 x4 y4 + conf
                        elif len(parts) == 10:
                            cls_id = int(parts[0])
                            coords = list(map(float, parts[1:9]))
                            conf = float(parts[9])
                            px_coords = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
                            geo_coords = [rasterio.transform.xy(transform, y * height, x * width) for x, y in px_coords]
                            geom = Polygon(geo_coords)
                            geoms.append({
                                "geometry": geom,
                                "class_id": cls_id,
                                "confidence": conf,
                                "source_image": image_name
                            })

        if geoms:
            gdf = gpd.GeoDataFrame(geoms, crs=raster_crs)
            # add class_name
            data_yaml_path = os.path.abspath(os.path.join(self.base_output_dir, 'data.yaml'))
            class_names = self.load_class_names(data_yaml_path)
            gdf['class_name'] = gdf['class_id'].map(class_names)
            ensure_dir(os.path.dirname(output_path) or ".")
            gdf.to_file(output_path, driver="GPKG", layer="predictions")
            logging.info("Exported: %s (n=%d)", output_path, len(gdf))

            # Optional: Mosaik-Bounding speichern
            if self.bounding_mosaik is not None:
                gpkg_path = os.path.join(os.path.dirname(output_path), "mosaik_extent.gpkg")
                self.bounding_mosaik.to_file(gpkg_path, driver="GPKG")
                logging.info("Mosaic extent saved: %s", gpkg_path)
        else:
            logging.warning("No prediction geometries found, nothing exported.")

# =============================================================================
# Config-based pipeline
# =============================================================================

@dataclass
class PipelineConfig:
    chdir: str
    fussgaenger_path: str
    raster_path: str
    output_base: str
    mosaic_name: str = "swissimage_raster_mosaik.tif"
    tiles_dir: str = "tiles_512"
    dataset_dir: str = "ultralytics/dataset"
    cropped_dir: str = "pred_tiles"
    predictions_gpkg: str = "predictions.gpkg"
    model_name: str = "yolov8n.pt"
    model_dir: str = "Model_NO_TUNE"
    img_size: int = 512
    batch_size: int = 16
    epochs: int = 100
    use_tuning: bool = False
    box_width: float = 5.0
    use_oriented_boxes: bool = False
    shift_fractions: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
    val_split: float = 0.2
    preview_count: int = 4
    conf_threshold: float = 0.5
    class_column: str = "class"
    buffer_config: Dict[str, float] = field(default_factory=dict)
    extent_train_set: Optional[str] = None
    extent_infer_set: Optional[str] = None  # optional abweichende Extent fuer Inferenz
    train_on_custom_dataset: Optional[str] = None  # falls statt Mosaik ein fixes Raster fuer Training/Prediction genutzt werden soll

class YOLOPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.paths = self._resolve_paths()
        self.model = None
        self.model_dir = None
        self.trainer: Optional[YOLOTraining] = None

    def _resolve_paths(self) -> Dict[str, str]:
        base = self.config.output_base
        return {
            "mosaic": os.path.join(base, self.config.mosaic_name),
            "download": os.path.join(base, "swissimage_raster_download"),
            "bbox": os.path.join(base, "bounding_boxes.gpkg"),
            "tiles": os.path.join(base, self.config.tiles_dir),
            "dataset": os.path.join(base, self.config.dataset_dir),
            "cropped": os.path.join(base, self.config.cropped_dir),
        }

    def _log_header(self, title: str):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"\n{'=' * 70}\n🧩 {title}  [{now}]\n{'=' * 70}"
        print(header)
        self._step_start_time = time.time()

    def _log_duration(self):
        duration = time.time() - getattr(self, '_step_start_time', time.time())
        print(f"⏱ Duration: {duration:.2f} seconds")

    def run(self):
        start_time = time.time()
        self._log_header("Starting YOLO pipeline")
        os.chdir(self.config.chdir)

        self.download_and_mosaic()

        # Clipping for training (and optional inference extent)
        if self.config.extent_train_set:
            self.clip_mosaic_train_extent()
        else:
            logging.info("No extent_train_set set – skipping clip.")

        self.generate_bounding_boxes()
        self.extract_tiles()
        self.prepare_dataset()
        self.train_model()
        self.create_prediction_tiles()
        self.predict()
        self.export_predictions()

        self._log_header("✅ Pipeline completed")
        total_time = time.time() - start_time
        gpkg_path = self._pred_gpkg_path()
        print("\n📦 Pipeline summary:")
        print(f"🗂  Predictions GeoPackage: {gpkg_path}")
        print(f"⏳ Total duration: {total_time:.2f} seconds")

        if os.path.exists(gpkg_path):
            try:
                gdf = gpd.read_file(gpkg_path)
                if "class_name" in gdf.columns:
                    print("\n🔎 Class distribution:")
                    print(gdf['class_name'].value_counts())
            except Exception as e:
                print(f"⚠️ Error reading prediction file: {e}")

    # ---- Steps ----

    def download_and_mosaic(self):
        self._log_header("1. Downloading and mosaicking raster tiles")
        print(f"📌 raster_path: {self.config.raster_path}")
        print(f"📌 fussgaenger_path: {self.config.fussgaenger_path}")
        print(f"📌 output_dir: {self.paths['download']}")

        if os.path.exists(self.paths["mosaic"]):
            logging.info("Mosaic already exists: %s – skipping step.", self.paths["mosaic"])
            return
    
        downloader = SwissImageMosaikDownloader(
            raster_path=self.config.raster_path,
            fussgaenger_path=self.config.fussgaenger_path,
            output_dir=self.paths["download"]
        )
        downloader.run_all(self.paths["mosaic"])
        self._log_duration()

    def clip_mosaic_train_extent(self):
        self._log_header("1b. Clipping mosaic to training extent")
        clipped_path = self.paths["mosaic"].replace(".tif", "_trainclip.tif")

        if os.path.exists(clipped_path):
            logging.info("Train clip already exists: %s – skipping step.", clipped_path)
            self.paths["mosaic"] = clipped_path
            return

        clip_mosaic_to_extent(self.paths["mosaic"], self.config.extent_train_set, clipped_path)
        self.paths["mosaic"] = clipped_path  # for subsequent steps
        self._log_duration()

        if self.config.extent_infer_set:
            infer_clip = self.paths["mosaic"].replace("_trainclip.tif", "_inferclip.tif")
            clip_mosaic_to_extent(self.paths["mosaic"], self.config.extent_infer_set, infer_clip)
            # store path for later create_prediction_tiles
            self._infer_clip_path = infer_clip
        else:
            self._infer_clip_path = None

    def generate_bounding_boxes(self):
        self._log_header("2. Generating bounding boxes")
        print(f"📌 input_fp: {self.config.fussgaenger_path}")
        print(f"📌 output_path: {self.paths['bbox']}")
        print(f"📌 box_width: {self.config.box_width}")
        print(f"📌 use_oriented_boxes: {self.config.use_oriented_boxes}")
        bbox_generator = BoundingBoxGenerator(
            input_fp=self.config.fussgaenger_path,
            output_path=self.paths["bbox"],
            box_width=self.config.box_width,
            use_oriented_boxes=self.config.use_oriented_boxes,
            buffer_config=self.config.buffer_config
        )
        print(f"📌 buffer_config: {self.config.buffer_config}")
        bbox_generator.generate_boxes()
        self._log_duration()

    def extract_tiles(self):
        self._log_header("3. Extracting training tiles")
        print(f"📌 mosaik_path: {self.paths['mosaic']}")
        print(f"📌 bbox_path: {self.paths['bbox']}")
        print(f"📌 output_dir: {self.paths['tiles']}")
        print(f"📌 tile_size: {self.config.img_size}")
        print(f"📌 shift_fractions: {self.config.shift_fractions}")
        extractor = TileExtractor(
            mosaik_path=self.paths["mosaic"],
            boxes_path=self.paths["bbox"],
            output_dir=self.paths["tiles"],
            tile_size=self.config.img_size,
            shift_fractions=self.config.shift_fractions
        )
        extractor.extract_tiles()
        self._log_duration()

    def prepare_dataset(self):
        self._log_header("4. Preparing dataset")
        print(f"📌 tile_dir: {self.paths['tiles']}")
        print(f"📌 boxes_path: {self.paths['bbox']}")
        print(f"📌 base_output: {self.paths['dataset']}")
        print(f"📌 val_split: {self.config.val_split}")
        print(f"📌 use_oriented_boxes: {self.config.use_oriented_boxes}")
        preparer = DatasetPreparer(
            tile_dir=self.paths["tiles"],
            boxes_path=self.paths["bbox"],
            base_output=self.paths["dataset"],
            val_split=self.config.val_split,
            use_oriented_boxes=self.config.use_oriented_boxes,
            class_column=self.config.class_column
        )
        preparer.create_dataset()
        self._log_duration()

    def train_model(self):
        self._log_header("5. Training YOLO model")
        print(f"📌 model_path: {self.config.model_name}")
        print(f"📌 use_tuning: {self.config.use_tuning}")
        print(f"📌 imgsz: {self.config.img_size}")
        print(f"📌 batch_size: {self.config.batch_size}")
        print(f"📌 epochs: {self.config.epochs}")
        trainer = YOLOTraining(
            base_output_dir=os.path.dirname(self.paths["dataset"]),
            model_dir=self.config.model_dir,
            use_oriented_bounding_box=self.config.use_oriented_boxes
        )
        if not _HAVE_ULTRALYTICS:
            raise RuntimeError("Ultralytics not available, training cannot start.")
        self.model, self.model_dir = trainer.train_model(
            model_path=self.config.model_name,
            use_tuning=self.config.use_tuning,
            imgsz=self.config.img_size,
            batch_size=self.config.batch_size,
            final_epochs=self.config.epochs
        )
        self.trainer = trainer
        self._log_duration()

    def create_prediction_tiles(self):
        self._log_header("6. Creating tiles for inference")
        input_path = self.config.train_on_custom_dataset
        if not input_path:
            if hasattr(self, "_infer_clip_path") and self._infer_clip_path:
                input_path = self._infer_clip_path
        if not input_path or not os.path.isfile(input_path):
            input_path = self.paths["mosaic"]
        print(f"📌 input_path: {input_path}")
        print(f"📌 output_dir: {self.paths['cropped']}")
        print(f"📌 tile_size: {self.config.img_size}")
        self.trainer.create_prediction_tiles(
            mosaik_path=input_path,
            output_dir=self.paths["cropped"],
            tile_size=self.config.img_size
        )
        self._log_duration()

    def predict(self):
        self._log_header("7. Running inference")
        print(f"📌 source_dir: {self.paths['cropped']}")
        print(f"📌 imgsz: {self.config.img_size}")
        print(f"📌 conf_threshold: {self.config.conf_threshold}")
        if not _HAVE_ULTRALYTICS:
            raise RuntimeError("Ultralytics not available, inference cannot start.")
        self.trainer.predict_on_directory(
            model=self.model,
            source_dir=self.paths["cropped"],
            imgsz=self.config.img_size,
            conf=self.config.conf_threshold,
            save_images=True
        )
        self._log_duration()

    def _pred_gpkg_path(self) -> str:
        base = os.path.join(self.config.chdir, self.config.output_base)
        run_dir = os.path.join("runs", "obb" if self.config.use_oriented_boxes else "detect", f"{self.config.model_dir}_PRED")
        return os.path.join(self.config.chdir, self.config.output_base, run_dir, "predictions.gpkg").replace("\\", "/")

    def export_predictions(self):
        self._log_header("8. Exporting predictions to GeoPackage")
        out_path = self._pred_gpkg_path()
        print(f"📌 output_path: {out_path}")
        # Ultralytics stores runs under CWD. Could switch to ultralytics root (optional):
        # os.chdir(os.path.join(self.config.chdir, self.config.output_base, "ultralytics"))
        self.trainer.convert_predictions_to_geopackage(
            image_dir=self.paths["cropped"],
            output_path=out_path
        )
        self._log_duration()

    # Hilfsmethoden
    def delete_generated_data(self):
        self._log_header("🧹 Cleaning up generated data")
        output_dir = self.config.output_base
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"🗑 Deleted: {output_dir}")
        else:
            print("⚠️ Nothing to delete.")
        self._log_duration()

    def update_config(self, new_config: Dict):
        """
        Updates the pipeline configuration with new values provided in the dictionary.
        
        Args:
            new_config (Dict): Dictionary containing configuration keys and their new values.
        """
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"⚠️ Unknown config key: {key}")
        self.paths = self._resolve_paths()
        print("🔧 Configuration updated.")
