"""
Modular LPRNet-Lite OCR Engine for Raspberry Pi 5.
Runs CPU inference using OpenCV DNN and pure NumPy CTC decoding.
Zero dependencies on PyTorch, Ultralytics, or ONNX Runtime.
"""

import os
import re
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import cv2
import numpy as np

# Standard 36 alphanumeric characters + 1 CTC blank (index 36) = 37 classes
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_INDEX = len(CHARSET)  # 36
NUM_CLASSES = len(CHARSET) + 1  # 37


def normalize_plate_text(text: str) -> str:
    """Normalizes plate string to uppercase alphanumeric only."""
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.strip().upper())


def estimate_plate_layout(crop: np.ndarray) -> str:
    """
    Determines plate layout (one_row vs two_rows) using aspect ratio:
    - Vietnamese standard long plate aspect ratio is ~4.3:1 (threshold >= 2.0)
    - Vietnamese square car / motorcycle plate aspect ratio is ~1.4:1 (threshold < 2.0)
    """
    h, w = crop.shape[:2]
    if h == 0:
        return "one_row"
    aspect_ratio = float(w) / float(h)
    return "one_row" if aspect_ratio >= 2.0 else "two_rows"


def find_horizontal_split(gray_crop: np.ndarray) -> int:
    """Finds vertical gradient energy valley between lines in [0.35H, 0.60H]."""
    h, w = gray_crop.shape[:2]
    min_y = int(h * 0.35)
    max_y = int(h * 0.60)
    if max_y <= min_y or h <= 10:
        return int(h * 0.48)

    grad_y = cv2.Sobel(gray_crop, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = np.sum(np.abs(grad_y), axis=1)

    search_zone = edge_energy[min_y:max_y]
    best_offset = int(np.argmin(search_zone))
    split_y = min_y + best_offset

    if split_y < int(h * 0.35) or split_y > int(h * 0.60):
        split_y = int(h * 0.48)
    return split_y


def rectify_two_row_plate(crop_bgr: np.ndarray, target_h: int = 48) -> np.ndarray:
    """Splits two-row plate and concatenates horizontally with 6px white spacer."""
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_h * 3, 3), 255, dtype=np.uint8)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    split_y = find_horizontal_split(gray)

    top_crop = crop_bgr[:split_y, :]
    bot_crop = crop_bgr[split_y:, :]

    th, tw = top_crop.shape[:2]
    bh, bw = bot_crop.shape[:2]

    scale_top = target_h / float(th) if th > 0 else 1.0
    new_tw = max(1, int(round(tw * scale_top)))
    top_resized = cv2.resize(top_crop, (new_tw, target_h), interpolation=cv2.INTER_LINEAR)

    scale_bot = target_h / float(bh) if bh > 0 else 1.0
    new_bw = max(1, int(round(bw * scale_bot)))
    bot_resized = cv2.resize(bot_crop, (new_bw, target_h), interpolation=cv2.INTER_LINEAR)

    spacer = np.full((target_h, 6, 3), 255, dtype=np.uint8)
    rectified = np.hstack([top_resized, spacer, bot_resized])
    return rectified


def letterbox_ocr(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (160, 48),
    pad_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Pads image to target_size (width, height) without stretching."""
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


def ctc_greedy_decode(logits: np.ndarray, blank_index: int = BLANK_INDEX) -> str:
    """Pure NumPy CTC greedy decoder."""
    arr = np.squeeze(logits)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [C, T] or [T, C], got shape {logits.shape}")

    if arr.shape[0] == (blank_index + 1) or arr.shape[0] < arr.shape[1]:
        pred_indices = np.argmax(arr, axis=0).tolist()
    else:
        pred_indices = np.argmax(arr, axis=1).tolist()

    collapsed = []
    prev_idx = None
    for idx in pred_indices:
        if idx != prev_idx:
            if idx != blank_index:
                collapsed.append(idx)
            prev_idx = idx

    chars = [CHARSET[idx] for idx in collapsed if idx < len(CHARSET)]
    return "".join(chars)


class PlateOCR:
    """
    OpenCV DNN License Plate Recognizer using LPRNet-Lite ONNX.
    """
    def __init__(self, model_path: str):
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"OCR model not found: {self.model_path}")

        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        if hasattr(cv2.dnn, "DNN_BACKEND_OPENCV"):
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        if hasattr(cv2.dnn, "DNN_TARGET_CPU"):
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def recognize(self, plate_image: np.ndarray, layout: str = "auto") -> Dict[str, Any]:
        """
        Runs complete OCR recognition on a cropped plate image.
        
        Args:
            plate_image: BGR crop of license plate.
            layout: 'auto', 'one_row', 'two_rows', or 'motorcycle'.
            
        Returns:
            {
                "text": str,
                "layout": str,
                "inference_ms": float,
                "timings": dict,
                "rectified_image": Optional[np.ndarray]
            }
        """
        if plate_image is None or plate_image.size == 0:
            return {
                "text": "UNKNOWN",
                "layout": "unknown",
                "inference_ms": 0.0,
                "timings": {"prep_ms": 0.0, "forward_ms": 0.0, "decode_ms": 0.0, "total_ms": 0.0},
                "rectified_image": None
            }

        t0 = time.perf_counter()

        # 1. Layout determination
        if layout == "auto":
            resolved_layout = estimate_plate_layout(plate_image)
        elif layout in ("two_rows", "two_rows_label_xe_may", "motorcycle"):
            resolved_layout = "two_rows"
        else:
            resolved_layout = "one_row"

        # 2. Rectification
        rectified_img = None
        if resolved_layout == "two_rows":
            rectified_img = rectify_two_row_plate(plate_image, target_h=48)
            norm_input = rectified_img
        else:
            norm_input = plate_image

        letterboxed = letterbox_ocr(norm_input, target_size=(160, 48))
        t_pre = time.perf_counter()

        # 3. OpenCV DNN Forward Pass
        blob = cv2.dnn.blobFromImage(
            letterboxed,
            scalefactor=1.0 / 127.5,
            size=(160, 48),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False
        )
        self.net.setInput(blob)
        logits = self.net.forward()
        t_fwd = time.perf_counter()

        # 4. CTC Decode
        raw_text = ctc_greedy_decode(logits, blank_index=BLANK_INDEX)
        clean_text = normalize_plate_text(raw_text)
        if not clean_text:
            clean_text = "UNKNOWN"
        t_dec = time.perf_counter()

        timings = {
            "prep_ms": (t_pre - t0) * 1000.0,
            "forward_ms": (t_fwd - t_pre) * 1000.0,
            "decode_ms": (t_dec - t_fwd) * 1000.0,
            "total_ms": (t_dec - t0) * 1000.0,
        }

        return {
            "text": clean_text,
            "layout": resolved_layout,
            "inference_ms": timings["forward_ms"],
            "timings": timings,
            "rectified_image": rectified_img
        }
