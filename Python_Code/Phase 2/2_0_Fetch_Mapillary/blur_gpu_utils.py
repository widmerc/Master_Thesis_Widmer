"""
Revisions:
24.06.2025:
- Refactored the code to improve clarity and efficiency. Added multithreading for faster processing of images.

26.06.2025:
- sobel_variance_blur & laplacian_variance_blur combined into one function: improvement in speed by ~20%
- CV_32F used instead of CV_64F for performance optimization
- Optional cuPy support prepared for GPU acceleration
- On large batches, cuPy (GPU) performs up to 5–10× faster than NumPy (CPU), depending on image size and hardware. This is because cuPy leverages thousands of parallel CUDA cores on the GPU for array operations (e.g., convolution, variance), whereas NumPy relies on CPU-based execution with limited parallelism, making cuPy significantly more efficient for high-volume image analysis.

25.07.2025:
- Build Paths before and not by each worker
- kernel load before
"""


import os
import cv2
import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import convolve as cp_convolve
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

kernel = cp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=cp.float32)


def compute_laplacian_variance(image_path, use_gpu=True, kernel=kernel):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None

    try:
        if use_gpu:
            image_array = cp.asarray(image, dtype=cp.float32)
            lap = cp_convolve(image_array, kernel)
            var = float(lap.var().get())
        else:
            lap = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F)
            var = float(lap.var())
        return var
    except Exception as e:
        print(f"⚠️ Fehler bei Verarbeitung {image_path}: {e}")
        return None

def batch_compute_blur(image_tuples, use_gpu=True, max_workers=8, kernel=kernel):
    def worker(image_id, path):
        blur_val = compute_laplacian_variance(path, use_gpu, kernel)
        return (image_id, blur_val)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(lambda args: worker(*args), image_tuples),
            total=len(image_tuples),
            desc="📸 Blur Detection (parallel)"
        ))

    return [r for r in results if r[1] is not None]

