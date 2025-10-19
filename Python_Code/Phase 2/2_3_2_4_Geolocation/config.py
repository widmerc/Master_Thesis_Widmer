import os

# --- General Configuration ---
USE_NUMBA = False  # True to use Numba for depth calculation
USE_MASK = True  # True to use segmentation mask for depth estimation
NUM_WORKERS = None  # None = automatic (number of CPU cores)
BATCH_SIZE = 80
NUM_IMAGES_TO_SAMPLE = 0  # 0 = all images, >0 = sample this many images from the CSV
CONFIDENCE_THRESHOLD = 0.2

# --- Depth Calculation and Classification ---
DEPTH_METHOD = "median"  # "median", "mean", "center", "minmax_avg"
CATEGORY_METHOD = "quantile"  # "thirds", "quantile", "thresholds"
CATEGORY_THRESHOLDS = None  # e.g. [5, 15, 30] for method="thresholds"

# --- Plot Configuration ---
PLOT = False  # True to create plots
PLOT_DPI = 300
PLOT_TITLE = "Detected Objects with Depth Classes"
PLOT_OPTIONS = {
    "min_conf": CONFIDENCE_THRESHOLD,
    "multiple": True,
    "title": PLOT_TITLE,
    "dpi": PLOT_DPI
}

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)

# Input data
IMAGE_FOLDER = r"D:\Mapillary_Data"
CSV_PATH = r"D:\Masterarbeit\03_Model\Scripts\2_Feature_Geolocation\2_3_Geolocation\data\valid_images_with_yolo_labels.csv"
SUITABLE_IMAGE_FOLDER = r"D:\Mapillary_Suitable"
GPKG_PATH = r"D:\Masterarbeit\03_Model\Scripts\2_Feature_Geolocation\2_0_Fetch_Mapillary\data\images_bbox_basic.gpkg"

# YOLO
YOLO_OUTPUT_FOLDER = r"D:\Masterarbeit\03_Model\Scripts\2_Feature_Geolocation\2_3_Geolocation\data\yolo_results_new"
YOLO_MODEL_PATH = r"D:\Masterarbeit\03_Model\Scripts\2_Feature_Geolocation\2_1_Model_Mapillary\data\Mapillary_Vistas_Yolo_staged_ultrasmall\runs\yolo11m_segment_staged_finetune_2\weights\best.pt"

# Depth
DEPTH_OUTPUT_FOLDER = r"D:\Masterarbeit\03_Model\Scripts\2_Feature_Geolocation\2_2_Model_Depth_Estimation\data\depth_processed"

# Output
OUTPUT_JSON_FOLDER = os.path.join(BASE_DIR, "data", "combined_results_json")
PLOT_OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "combined_plots")
