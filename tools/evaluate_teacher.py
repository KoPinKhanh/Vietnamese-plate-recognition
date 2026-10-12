"""
Common Teacher Model Evaluator for Vietnamese License Plate OCR.
Evaluates Teacher models on standard manifests (val.csv, val_unseen_text_v2.csv, test.csv).
Ensures preprocessing consistency:
  - Ground truth crop (pad_ratio=0.03)
  - Two-row Sobel vertical gradient valley rectification
  - Native Teacher resizing and normalization
Outputs:
  - Exact Accuracy
  - Character Accuracy
  - CER
  - Average Edit Distance
  - Layout breakdowns (one_row, two_rows, motorcycle)
"""

import ast
import csv
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.plate_text_utils import normalize_plate_text, calculate_cer, levenshtein_distance
from ocr.preprocess import crop_plate_gt, rectify_two_row_plate


class TeacherEvaluationDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        target_size: Tuple[int, int] = (128, 32), # (W, H)
        mean: float = 0.5,
        std: float = 0.5,
    ):
        self.manifest_path = Path(manifest_csv)
        self.target_size = target_size
        self.mean = mean
        self.std = std

        self.records: List[Dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
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

                layout = row.get("subset", row.get("layout", "one_row"))
                self.records.append({
                    "image_path": row["image_path"],
                    "plate_text": row["plate_text"],
                    "layout": layout,
                    "bbox": bbox,
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        img_bgr = cv2.imread(item["image_path"])
        if img_bgr is None:
            img_bgr = np.full((100, 300, 3), 255, dtype=np.uint8)

        # 1. Ground truth crop
        if item["bbox"] is not None and len(item["bbox"]) == 4:
            crop = crop_plate_gt(img_bgr, item["bbox"], pad_ratio=0.03)
        else:
            crop = img_bgr

        # 2. Rectify two-row plates if applicable
        if item["layout"] in ("two_rows", "two_rows_label_xe_may"):
            crop, _, _ = rectify_two_row_plate(crop, target_h=48)

        # 3. Resize to Teacher native size (W, H)
        tw, th = self.target_size
        resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)

        # 4. Normalize to float32 [C, H, W]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()

        return {
            "image": tensor,
            "raw_text": item["plate_text"],
            "layout": item["layout"],
            "image_path": item["image_path"],
        }


def teacher_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    raw_texts = [item["raw_text"] for item in batch]
    layouts = [item["layout"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    return {
        "images": images,
        "raw_texts": raw_texts,
        "layouts": layouts,
        "image_paths": image_paths,
    }


def evaluate_teacher(
    model: Any,
    manifest_csv: str,
    predict_fn: Callable[[Any, torch.Tensor], Tuple[List[str], List[float]]],
    device: torch.device,
    target_size: Tuple[int, int] = (128, 32), # (W, H)
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates a Teacher model on a given manifest CSV.
    Args:
        model: Teacher PyTorch model.
        manifest_csv: Path to manifest CSV.
        predict_fn: Function mapping (model, images_tensor) -> (list_of_pred_strings, list_of_confidences).
        device: Torch device.
    """
    dataset = TeacherEvaluationDataset(manifest_csv, target_size=target_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=teacher_collate_fn,
        pin_memory=True,
    )

    total_samples = 0
    total_exact = 0
    total_cer = 0.0
    total_edit_dist = 0.0

    layout_stats: Dict[str, Dict[str, Any]] = {
        "one_row": {"total": 0, "exact": 0, "cer_sum": 0.0, "edit_dist_sum": 0.0},
        "two_rows": {"total": 0, "exact": 0, "cer_sum": 0.0, "edit_dist_sum": 0.0},
        "two_rows_label_xe_may": {"total": 0, "exact": 0, "cer_sum": 0.0, "edit_dist_sum": 0.0},
    }

    all_records = []

    if hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            raw_texts = batch["raw_texts"]
            layouts = batch["layouts"]
            image_paths = batch["image_paths"]

            pred_strings, confidences = predict_fn(model, images)

            for gt, pred, conf, layout, img_path in zip(raw_texts, pred_strings, confidences, layouts, image_paths):
                gt_norm = normalize_plate_text(gt)
                pred_norm = normalize_plate_text(pred)
                exact = (gt_norm == pred_norm)
                cer = calculate_cer(gt_norm, pred_norm)
                ed = levenshtein_distance(gt_norm, pred_norm)

                total_samples += 1
                if exact:
                    total_exact += 1
                total_cer += cer
                total_edit_dist += ed

                lay_key = layout if layout in layout_stats else ("two_rows" if "two" in layout else "one_row")
                layout_stats[lay_key]["total"] += 1
                if exact:
                    layout_stats[lay_key]["exact"] += 1
                layout_stats[lay_key]["cer_sum"] += cer
                layout_stats[lay_key]["edit_dist_sum"] += ed

                all_records.append({
                    "image_path": img_path,
                    "layout": lay_key,
                    "ground_truth": gt_norm,
                    "prediction": pred_norm,
                    "confidence": conf,
                    "exact": exact,
                    "cer": cer,
                    "edit_distance": ed,
                })

    exact_acc = (total_exact / total_samples) * 100.0 if total_samples > 0 else 0.0
    avg_cer = total_cer / total_samples if total_samples > 0 else 0.0
    char_acc = max(0.0, (1.0 - avg_cer) * 100.0)
    avg_ed = total_edit_dist / total_samples if total_samples > 0 else 0.0

    layout_breakdown = {}
    for k, v in layout_stats.items():
        tot = v["total"]
        ex = v["exact"]
        acc = (ex / tot) * 100.0 if tot > 0 else 0.0
        c = v["cer_sum"] / tot if tot > 0 else 0.0
        ca = max(0.0, (1.0 - c) * 100.0)
        ed = v["edit_dist_sum"] / tot if tot > 0 else 0.0
        layout_breakdown[k] = {
            "total_samples": tot,
            "correct": ex,
            "exact_accuracy": acc,
            "char_accuracy": ca,
            "cer": c,
            "avg_edit_distance": ed,
        }

    metrics = {
        "manifest": str(manifest_csv),
        "total_samples": total_samples,
        "correct": total_exact,
        "exact_accuracy": exact_acc,
        "cer": avg_cer,
        "char_accuracy": char_acc,
        "avg_edit_distance": avg_ed,
        "layout_breakdown": layout_breakdown,
    }

    return metrics, all_records
