from pathlib import Path

# config.py
ACCESS_TOKEN = '...'
ACCESS_TOKEN2 = '...'
ACCESS_TOKEN3 = '...'
ACCESS_TOKEN4 = '...'
ACCESS_TOKEN5 = '...'
ACCESS_TOKEN6 = '...'


from pathlib import Path

# Thresholds
LAPLACIAN_THRESHOLD = 100.0
USE_GPU = True
TEST_LIMIT = None

# Project paths
ROOT_PATH = Path(".")
DATA_PATH = ROOT_PATH / "data"
IMAGE_DIR = r"D:\Mapillary_Data"

# File paths
GPKG_PATH = DATA_PATH / "images_bbox_fullmeta.gpkg"
OUTPUT_GPKG = GPKG_PATH.with_name(GPKG_PATH.stem + "_with_blur.gpkg")
FAILED_DOWNLOADS_PATH = DATA_PATH / f"{GPKG_PATH.stem}_failed_downloads.txt"
TMP_BLUR_PATH = ROOT_PATH / "tmp_blur"

# GPKG column names
IMAGE_COLUMN = "thumb_1024_url"
BLURRY_COL = "blur_value"

