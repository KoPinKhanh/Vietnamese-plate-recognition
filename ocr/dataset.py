"""
PyTorch Dataset and DataLoader for Vietnamese License Plate OCR.
Loads annotations on-the-fly from VNLP processed manifests.
Performs ground-truth cropping, two-row rectification, OCR-safe augmentations,
letterboxing to (160, 48), and normalizes for CTC training.
"""

import ast
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ocr.charset import encode_text
from ocr.preprocess import (
    crop_plate_gt,
    rectify_two_row_plate,
    letterbox_ocr,
    normalize_image_for_model,
)


class OCRAugmentor:
    """
    Lightweight, OCR-safe image augmentations.
    Preserves text readability and prevents spatial distortion or character mirroring.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img_bgr

        img = img_bgr.copy()
        h, w = img.shape[:2]

        # 1. Slight brightness / contrast adjustment
        if random.random() < 0.4:
            alpha = random.uniform(0.85, 1.15)  # Contrast
            beta = random.uniform(-20, 20)      # Brightness
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # 2. Slight rotation (+-3 degrees)
        if random.random() < 0.3:
            angle = random.uniform(-3.5, 3.5)
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )

        # 3. Slight Gaussian blur or motion blur
        if random.random() < 0.25:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 4. Slight Gaussian noise
        if random.random() < 0.2:
            noise = np.random.normal(0, random.uniform(4, 10), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img


class VNLPOCRDataset(Dataset):
    """
    VNLP OCR Dataset reading from manifest CSV.
    Loads images on-the-fly, crops GT bbox, rectifies two-row plates, letterboxes to 160x48.
    """
    def __init__(
        self,
        manifest_csv: str,
        target_size: Tuple[int, int] = (160, 48),
        is_training: bool = False,
        augment_p: float = 0.5,
        max_samples: Optional[int] = None,
    ):
        self.manifest_path = Path(manifest_csv)
        self.target_size = target_size
        self.is_training = is_training
        self.augmentor = OCRAugmentor(p=augment_p) if is_training else None

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

        self.records: List[Dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse bbox: support both [x1, y1, x2, y2] columns and bbox column
                bbox = None
                if "x1" in row and row["x1"] and row["x1"].strip():
                    try:
                        bbox = [int(float(row["x1"])), int(float(row["y1"])), int(float(row["x2"])), int(float(row["y2"]))]
                    except Exception:
                        bbox = None
                elif "bbox" in row and row["bbox"] and row["bbox"].strip():
                    try:
                        bbox = ast.literal_eval(row["bbox"])
                    except Exception:
                        bbox = None

                self.records.append({
                    "image_path": row["image_path"],
                    "plate_text": row["plate_text"],
                    "layout": row.get("subset", "one_row"),
                    "bbox": bbox,
                })

        if max_samples is not None and max_samples < len(self.records):
            # Deterministic subset selection
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        img_path = item["image_path"]
        raw_text = item["plate_text"]
        layout = item["layout"]
        bbox = item["bbox"]

        # Read image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Fallback black image in case of corrupted file
            img_bgr = np.full((100, 300, 3), 255, dtype=np.uint8)

        # 1. Crop ground truth plate
        if bbox is not None and len(bbox) == 4:
            crop = crop_plate_gt(img_bgr, bbox, pad_ratio=0.03)
        else:
            crop = img_bgr

        # 2. Rectify two-row plates if applicable
        if layout in ("two_rows", "two_rows_label_xe_may"):
            crop, _, _ = rectify_two_row_plate(crop, target_h=self.target_size[1])

        # 3. Augmentation (training only, applied to rectified crop)
        if self.augmentor is not None:
            crop = self.augmentor(crop)

        # 4. Standardize to target size (160, 48) with letterbox padding
        letterboxed = letterbox_ocr(crop, target_size=self.target_size, pad_color=(255, 255, 255))

        # 5. Normalize to RGB float32 [-1.0, 1.0]
        normalized = normalize_image_for_model(letterboxed)

        # 6. Convert to NCHW PyTorch tensor: (48, 160, 3) -> (3, 48, 160)
        img_tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()

        # 7. Encode text to class indices
        encoded = encode_text(raw_text)

        return {
            "image": img_tensor,
            "label": torch.tensor(encoded, dtype=torch.long),
            "label_len": len(encoded),
            "raw_text": raw_text,
            "layout": layout,
            "image_path": img_path,
        }


def ocr_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to batch variable-length OCR samples for CTCLoss.
    - Stacks images into [B, 3, 48, 160]
    - Concatenates labels into 1D targets tensor [sum(label_lens)]
    - Generates target_lengths [B]
    - Generates input_lengths [B] = 40 (fixed timesteps for width 160)
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = [item["label"] for item in batch]
    targets = torch.cat(labels, dim=0) if len(labels) > 0 else torch.empty(0, dtype=torch.long)
    target_lengths = torch.tensor([item["label_len"] for item in batch], dtype=torch.long)

    # Input timesteps T = 40 for LPRNetLite with input_w = 160
    batch_size = images.shape[0]
    input_lengths = torch.full((batch_size,), 40, dtype=torch.long)

    raw_texts = [item["raw_text"] for item in batch]
    layouts = [item["layout"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "input_lengths": input_lengths,
        "raw_texts": raw_texts,
        "layouts": layouts,
        "image_paths": image_paths,
    }


def create_ocr_dataloader(
    manifest_csv: str,
    batch_size: int = 64,
    is_training: bool = False,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (160, 48),
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Creates a configured DataLoader for VNLP OCR."""
    dataset = VNLPOCRDataset(
        manifest_csv=manifest_csv,
        target_size=target_size,
        is_training=is_training,
        max_samples=max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        collate_fn=ocr_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    return loader
