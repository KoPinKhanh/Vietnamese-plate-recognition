"""
Test Fine-Tuned PARSeq-Tiny VNLP Model on Vietnamese License Plate Crops.
Model: weights/ocr/teachers/parseq_tiny_vn/best.pth

Supports:
  --image PATH      : Single image inference
  --folder PATH     : Folder batch inference
  --checkpoint PATH : Path to fine-tuned checkpoint
  --device cuda     : Device (cuda/cpu)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.preprocess import rectify_two_row_plate
from ocr.plate_text_utils import normalize_plate_text
from teacher_adapters.parseq_vn import parseq_predict_fn


def load_parseq_vn(checkpoint_path: str, device: torch.device):
    """Loads fine-tuned PARSeq Tiny model from checkpoint."""
    print(f"Loading fine-tuned PARSeq-Tiny from {checkpoint_path} on {device}...")
    model = torch.hub.load("baudm/parseq", "parseq_tiny", pretrained=False, trust_repo=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def get_parseq_vn_transform(target_size: Tuple[int, int] = (128, 32)):
    """Preprocessing consistent with PARSeq VNLP training: resize to (128, 32), [-1.0, 1.0]."""
    tw, th = target_size
    def _transform(crop_bgr: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(crop_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - 0.5) / 0.5
        return torch.from_numpy(normalized).permute(2, 0, 1).float()
    return _transform


def preprocess_plate_crop(img_bgr: np.ndarray, layout: Optional[str] = None) -> np.ndarray:
    """
    Applies project-consistent plate preprocessing:
    - If layout is two_rows / motorcycle OR aspect ratio W/H < 1.9 -> rectify
    - Otherwise keep single row
    """
    h, w = img_bgr.shape[:2]
    is_two_rows = False
    if layout in ("two_rows", "two_rows_label_xe_may"):
        is_two_rows = True
    elif layout is None:
        if (w / float(max(1, h))) < 1.9:
            is_two_rows = True

    if is_two_rows:
        rectified, _, _ = rectify_two_row_plate(img_bgr, target_h=48)
        return rectified
    return img_bgr


def predict_single_image(
    model,
    img_bgr: np.ndarray,
    transform,
    device: torch.device,
    layout: Optional[str] = None,
) -> Tuple[str, str, float, float]:
    """
    Inference on a single image.
    Returns: (raw_pred, normalized_pred, confidence, time_ms)
    """
    crop = preprocess_plate_crop(img_bgr, layout=layout)
    tensor = transform(crop).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        preds, confs = parseq_predict_fn(model, tensor)
    t1 = time.perf_counter()
    time_ms = (t1 - t0) * 1000.0

    raw_pred = preds[0] if preds else ""
    norm_pred = normalize_plate_text(raw_pred)
    conf = confs[0] if confs else 0.95

    return raw_pred, norm_pred, conf, time_ms


def main():
    parser = argparse.ArgumentParser(description="Test Fine-Tuned PARSeq-Tiny VNLP")
    parser.add_argument("--image", type=str, default=None, help="Path to single license plate crop")
    parser.add_argument("--folder", type=str, default=None, help="Path to folder of plate crops")
    parser.add_argument("--checkpoint", type=str, default="weights/ocr/teachers/parseq_tiny_vn/best.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Must provide either --image or --folder")

    device = torch.device(args.device)
    model = load_parseq_vn(args.checkpoint, device)
    transform = get_parseq_vn_transform((128, 32))

    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.folder:
        folder_path = Path(args.folder)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(sorted(folder_path.glob(ext)))

    print(f"\nProcessing {len(image_paths)} image(s)...")
    print("=" * 70)

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"IMAGE:   {p}\nERROR:   Could not read image.\n" + "-" * 70)
            continue

        raw_pred, norm_pred, conf, time_ms = predict_single_image(model, img, transform, device)

        print(f"IMAGE:    {p}")
        print(f"GT:       N/A")
        print(f"PRED:     {norm_pred}")
        print(f"CONF:     {conf:.4f}")
        print(f"TIME_MS:  {time_ms:.2f} ms")
        print("-" * 70)


if __name__ == "__main__":
    main()
