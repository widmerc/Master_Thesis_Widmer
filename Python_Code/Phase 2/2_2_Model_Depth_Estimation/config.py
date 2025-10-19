from pathlib import Path

# Projektpfade
ROOT_PATH = Path(".")
DATA_PATH = ROOT_PATH / "data"
IMAGE_DIR = r"D:\Mapillary_Data"
MODEL_PATH = Path(".").resolve().parent / "2_1_Model_Mapillary" / "data" / "Mapillary_Vistas_Yolo" / "yolo11n_seg_train" / "weights" / "best.pt"

# Dateipfade
GPKG_PATH = DATA_PATH / "images_bbox_fullmeta.gpkg"
OUTPUT_GPKG = GPKG_PATH.with_name(GPKG_PATH.stem + "_with_blur.gpkg")

