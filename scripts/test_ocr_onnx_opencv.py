"""
OpenCV DNN Vietnamese License Plate OCR Inference Script.

Strict requirements:
- ONLY uses cv2 and numpy (NO torch, ultralytics, or onnxruntime in inference path).
- Performs identical preprocessing as training:
  - Ground-truth / detection crop
  - Two-row rectification (top + spacer + bottom) if layout is two-row
  - Aspect-preserving letterbox padding to 160x48
  - Normalization via cv2.dnn.blobFromImage
- Executes forward pass with cv2.dnn
- Decodes output logits with pure NumPy CTC greedy decoder

Usage:
    python scripts/test_ocr_onnx_opencv.py --model weights/ocr/best_lprnet_vn.onnx --image path/to/plate.jpg [--layout two_rows]
"""

import argparse
import time
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np

# Standard 36 alphanumeric characters + blank (index 36)
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_INDEX = len(CHARSET)  # 36
NUM_CLASSES = len(CHARSET) + 1  # 37


def find_horizontal_split(gray_crop: np.ndarray) -> int:
    """Finds horizontal split valley between top row and bottom row using vertical Sobel derivative."""
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
    """Splits two-row plate into top row and bottom row, then concatenates horizontally with 6px spacer."""
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
    """Standardizes image aspect ratio and pads to target_size (width, height) without stretching."""
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


def ctc_greedy_decode_numpy(logits: np.ndarray, blank_index: int = BLANK_INDEX) -> str:
    """NumPy-only CTC greedy decoder."""
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


class OpenCVPlateOCR:
    """OpenCV DNN inference engine for LPRNet-Lite ONNX model."""
    def __init__(self, model_path: str, backend: int = cv2.dnn.DNN_BACKEND_OPENCV, target: int = cv2.dnn.DNN_TARGET_CPU):
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)

    def predict(self, plate_bgr: np.ndarray, layout: str = "auto") -> Tuple[str, float]:
        """
        Runs complete OCR inference on a plate image crop.
        
        Args:
            plate_bgr: BGR crop of the license plate.
            layout: 'auto', 'one_row', 'two_rows', or 'motorcycle'.
            
        Returns:
            (predicted_plate_text, inference_latency_ms)
        """
        h, w = plate_bgr.shape[:2]

        # Auto-detect two-row if aspect ratio is roughly square (w/h < 2.0)
        is_two_row = False
        if layout in ("two_rows", "two_rows_label_xe_may", "motorcycle"):
            is_two_row = True
        elif layout == "auto":
            is_two_row = (w / float(h)) < 2.0

        t0 = time.perf_counter()

        if is_two_row:
            processed = rectify_two_row_plate(plate_bgr, target_h=48)
        else:
            processed = plate_bgr

        letterboxed = letterbox_ocr(processed, target_size=(160, 48))

        # Normalization: scalefactor=1/127.5, mean=(127.5, 127.5, 127.5), swapRB=True
        # Exactly matches: (pixel_rgb / 127.5) - 1.0
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

        pred_text = ctc_greedy_decode_numpy(logits, blank_index=BLANK_INDEX)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return pred_text, latency_ms


def main():
    parser = argparse.ArgumentParser(description="Test OCR ONNX model with OpenCV DNN")
    parser.add_argument("--model", type=str, default="weights/ocr/best_lprnet_vn.onnx", help="Path to ONNX model")
    parser.add_argument("--image", type=str, required=True, help="Path to plate image crop")
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "one_row", "two_rows", "motorcycle"], help="Plate layout")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Could not load image: {args.image}")

    ocr = OpenCVPlateOCR(model_path=args.model)

    # Warmup
    for _ in range(3):
        ocr.predict(img, layout=args.layout)

    pred, latency = ocr.predict(img, layout=args.layout)
    print(f"Prediction: {pred}")
    print(f"Inference:  {latency:.2f} ms")


if __name__ == "__main__":
    main()
