"""
Preprocessing pipeline for Vietnamese License Plate OCR.
Shared between training, offline evaluation, and OpenCV DNN deployment.
"""

from typing import Tuple, Optional, List
import cv2
import numpy as np


def crop_plate_gt(img: np.ndarray, bbox: List[int], pad_ratio: float = 0.03) -> np.ndarray:
    """
    Crops plate using ground-truth bbox [x1, y1, x2, y2] with small padding clamped to image boundary.
    
    Args:
        img: Input image (BGR).
        bbox: [x1, y1, x2, y2] coordinates.
        pad_ratio: Fractional padding around bbox (default 0.03, i.e. 3%).
        
    Returns:
        Cropped plate image.
    """
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    pad_w = int(round(bw * pad_ratio))
    pad_h = int(round(bh * pad_ratio))

    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(iw, x2 + pad_w)
    cy2 = min(ih, y2 + pad_h)

    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return img[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
    return crop


def find_horizontal_split(gray_crop: np.ndarray) -> int:
    """
    Finds horizontal split valley between top row and bottom row
    using horizontal projection of pixel gradients in the middle 35%-60% zone.
    
    Args:
        gray_crop: Grayscale two-row plate crop.
        
    Returns:
        Optimal split y-coordinate.
    """
    h, w = gray_crop.shape[:2]
    min_y = int(h * 0.35)
    max_y = int(h * 0.60)

    if max_y <= min_y or h <= 10:
        return int(h * 0.48)

    # Use vertical Sobel derivative to find text gap (gap has minimal horizontal edges)
    grad_y = cv2.Sobel(gray_crop, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = np.sum(np.abs(grad_y), axis=1)

    search_zone = edge_energy[min_y:max_y]
    best_offset = int(np.argmin(search_zone))
    split_y = min_y + best_offset

    # Sanity check: must remain within reasonable bounds
    if split_y < int(h * 0.35) or split_y > int(h * 0.60):
        split_y = int(h * 0.48)

    return split_y


def rectify_two_row_plate(crop_bgr: np.ndarray, target_h: int = 48) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits two-row plate into top row and bottom row,
    then concatenates them horizontally: [top_row, 6px spacer, bottom_row].
    
    Args:
        crop_bgr: BGR crop of two-row plate.
        target_h: Target height for horizontal concatenation.
        
    Returns:
        (rectified_bgr, top_crop, bot_crop)
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        empty = np.full((target_h, target_h * 3, 3), 255, dtype=np.uint8)
        return empty, crop_bgr, crop_bgr

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    split_y = find_horizontal_split(gray)

    top_crop = crop_bgr[:split_y, :]
    bot_crop = crop_bgr[split_y:, :]

    th, tw = top_crop.shape[:2]
    bh, bw = bot_crop.shape[:2]

    # Resize both rows to target_h while maintaining individual aspect ratios
    scale_top = target_h / float(th) if th > 0 else 1.0
    new_tw = max(1, int(round(tw * scale_top)))
    top_resized = cv2.resize(top_crop, (new_tw, target_h), interpolation=cv2.INTER_LINEAR)

    scale_bot = target_h / float(bh) if bh > 0 else 1.0
    new_bw = max(1, int(round(bw * scale_bot)))
    bot_resized = cv2.resize(bot_crop, (new_bw, target_h), interpolation=cv2.INTER_LINEAR)

    # 6px white spacer between top and bottom lines
    spacer = np.full((target_h, 6, 3), 255, dtype=np.uint8)
    rectified = np.hstack([top_resized, spacer, bot_resized])

    return rectified, top_crop, bot_crop


def letterbox_ocr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (160, 48),
    pad_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Standardizes image aspect ratio and pads to target_size (width, height) without stretching.
    
    Args:
        img_bgr: BGR image.
        target_size: (width, height), default (160, 48).
        pad_color: (B, G, R) color for letterbox padding (default white 255, 255, 255).
        
    Returns:
        Letterboxed image with shape (height, width, 3).
    """
    tw, th = target_size
    h, w = img_bgr.shape[:2]

    if h == 0 or w == 0:
        return np.full((th, tw, 3), pad_color, dtype=np.uint8)

    scale = min(tw / float(w), th / float(h))
    new_w = max(1, min(tw, int(round(w * scale))))
    new_h = max(1, min(th, int(round(h * scale))))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = tw - new_w
    pad_h = th - new_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=pad_color
    )
    return padded


def preprocess_plate_for_ocr(
    img_bgr: np.ndarray,
    layout: str = "one_row",
    bbox: Optional[List[int]] = None,
    target_size: Tuple[int, int] = (160, 48)
) -> np.ndarray:
    """
    Complete end-to-end preprocessing for a raw plate crop or full frame with bbox:
    1. If bbox is provided, crops plate using ground-truth bbox.
    2. If layout is 'two_rows' or 'two_rows_label_xe_may', rectifies into single row.
    3. Letterbox pads to target_size (160, 48) while preserving aspect ratio.
    
    Args:
        img_bgr: Raw input image (BGR).
        layout: 'one_row', 'two_rows', or 'two_rows_label_xe_may'.
        bbox: Optional [x1, y1, x2, y2].
        target_size: (width, height), default (160, 48).
        
    Returns:
        Standardized BGR image of shape (48, 160, 3).
    """
    if bbox is not None:
        crop = crop_plate_gt(img_bgr, bbox)
    else:
        crop = img_bgr

    if layout in ("two_rows", "two_rows_label_xe_may"):
        crop, _, _ = rectify_two_row_plate(crop, target_h=target_size[1])

    letterboxed = letterbox_ocr(crop, target_size=target_size)
    return letterboxed


def normalize_image_for_model(img_bgr: np.ndarray) -> np.ndarray:
    """
    Converts BGR uint8 [0, 255] to RGB float32 in range [-1.0, 1.0].
    Formula: (pixel / 127.5) - 1.0
    Compatible with cv2.dnn.blobFromImage(..., scalefactor=1/127.5, mean=(127.5, 127.5, 127.5), swapRB=True).
    
    Args:
        img_bgr: BGR uint8 image of shape (H, W, 3).
        
    Returns:
        RGB float32 image of shape (H, W, 3).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return (img_rgb.astype(np.float32) / 127.5) - 1.0


# --- Legacy functions for Tesseract baseline compatibility ---
PREPROCESS_STRATEGIES = ["gray", "otsu", "otsu_inv", "clahe_otsu", "adaptive", "resize"]


def resize_preserve_aspect(image: np.ndarray, target_height: int = 80) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image
    scale = target_height / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)


def to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def preprocess_plate(
    image: np.ndarray,
    strategy: str = "clahe_otsu",
    target_height: int = 80,
    denoise: bool = True,
    border_pad: int = 8,
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Invalid empty image passed to preprocess_plate")

    resized = resize_preserve_aspect(image, target_height=target_height)
    gray = to_gray(resized)
    if denoise:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if strategy == "gray" or strategy == "resize":
        result = gray
    elif strategy == "otsu":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif strategy == "otsu_inv":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif strategy == "clahe_otsu":
        enhanced = apply_clahe(gray, clip_limit=2.0)
        _, result = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif strategy == "adaptive":
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        enhanced = apply_clahe(gray, clip_limit=2.0)
        _, result = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if border_pad > 0:
        result = cv2.copyMakeBorder(
            result, border_pad, border_pad, border_pad, border_pad,
            cv2.BORDER_CONSTANT, value=255
        )
    return result

