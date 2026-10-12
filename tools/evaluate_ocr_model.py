"""
Standard PyTorch OCR Evaluator for Vietnamese License Plate Recognition (LPRNet-Lite).
Used across Phase 3 for all official evaluations and model selections.

Supports CLI:
  python tools/evaluate_ocr_model.py --weights <path_to_pth> --manifest <path_to_csv>
"""

import argparse
import ast
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.charset import CHARSET, NUM_CLASSES, BLANK_INDEX
from ocr.models.lprnet_lite import LPRNetLite
from ocr.dataset import create_ocr_dataloader
from ocr.ctc_decoder import ctc_greedy_decode_torch
from ocr.plate_text_utils import normalize_plate_text, calculate_cer, levenshtein_distance
from ocr.preprocess import preprocess_plate_for_ocr


def evaluate_manifest(
    model: torch.nn.Module,
    manifest_csv: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates PyTorch model on a given manifest CSV.
    Uses exact production/training preprocessing:
      - Ground truth crop (pad_ratio=0.03)
      - Two-row Sobel vertical gradient valley rectification
      - Letterbox padding to (160, 48) with (255, 255, 255)
      - Normalization to [-1.0, 1.0] RGB
      - CTC greedy decoding
    """
    loader = create_ocr_dataloader(
        manifest_csv=manifest_csv,
        batch_size=batch_size,
        is_training=False,
        num_workers=num_workers,
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

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            raw_texts = batch["raw_texts"]
            layouts = batch["layouts"]
            image_paths = batch["image_paths"]

            logits = model(images)
            preds = ctc_greedy_decode_torch(logits, blank_index=BLANK_INDEX)

            for gt, pred, layout, img_path in zip(raw_texts, preds, layouts, image_paths):
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
                    "exact": exact,
                    "cer": cer,
                    "edit_distance": ed,
                })

    exact_acc = (total_exact / total_samples) * 100.0 if total_samples > 0 else 0.0
    avg_cer = total_cer / total_samples if total_samples > 0 else 0.0
    char_acc = max(0.0, (1.0 - avg_cer) * 100.0)
    avg_ed = total_edit_dist / total_samples if total_samples > 0 else 0.0

    metrics = {
        "manifest": str(manifest_csv),
        "total_samples": total_samples,
        "correct": total_exact,
        "exact_accuracy": exact_acc,
        "cer": avg_cer,
        "char_accuracy": char_acc,
        "avg_edit_distance": avg_ed,
        "layout_breakdown": {},
    }

    for lay, stats in layout_stats.items():
        n = stats["total"]
        if n > 0:
            ex = (stats["exact"] / n) * 100.0
            c = stats["cer_sum"] / n
            e = stats["edit_dist_sum"] / n
            metrics["layout_breakdown"][lay] = {
                "total": n,
                "correct": stats["exact"],
                "exact_accuracy": ex,
                "cer": c,
                "char_accuracy": max(0.0, (1.0 - c) * 100.0),
                "avg_edit_distance": e,
            }

    return metrics, all_records


def print_evaluation_summary(metrics: Dict[str, Any]):
    manifest_name = Path(metrics["manifest"]).name
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY: {manifest_name}")
    print("=" * 70)
    print(f"Total Samples:          {metrics['total_samples']}")
    print(f"Correct:                {metrics['correct']} / {metrics['total_samples']}")
    print(f"Exact Plate Accuracy:   {metrics['exact_accuracy']:.2f}%")
    print(f"Character Accuracy:     {metrics['char_accuracy']:.2f}%")
    print(f"Character Error Rate:   {metrics['cer']:.4f}")
    print(f"Average Edit Distance:  {metrics['avg_edit_distance']:.4f}")
    print("-" * 70)
    print(f"{'Layout':<24} | {'Samples':<8} | {'Exact Acc':<12} | {'Char Acc':<10} | {'CER':<8} | {'Avg ED':<8}")
    print("-" * 70)
    for lay, st in metrics["layout_breakdown"].items():
        lay_name = "1 Dòng (one_row)" if lay == "one_row" else ("2 Dòng Ô tô" if lay == "two_rows" else "2 Dòng Xe Máy")
        print(f"{lay_name:<24} | {st['total']:<8} | {st['exact_accuracy']:>8.2f}%    | {st['char_accuracy']:>7.2f}%   | {st['cer']:>6.4f} | {st['avg_edit_distance']:>6.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Standard PyTorch OCR Evaluator")
    parser.add_argument("--weights", "--checkpoint", dest="weights", type=str, default="weights/ocr/best_lprnet_vn.pth",
                        help="Path to PyTorch .pth checkpoint")
    parser.add_argument("--manifest", "--test_csv", dest="manifest", type=str, default=None,
                        help="Path to manifest CSV to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    ckpt_path = Path(args.weights)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path} on {device}")
    model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        metrics, _ = evaluate_manifest(model, str(manifest_path), device, batch_size=args.batch_size, num_workers=args.num_workers)
        print_evaluation_summary(metrics)
    else:
        # Default behavior: run on test.csv and test_unseen_text.csv
        test_path = Path("datasets/VNLP_processed/test.csv")
        unseen_path = Path("datasets/VNLP_processed/test_unseen_text.csv")
        m_test, _ = evaluate_manifest(model, str(test_path), device, batch_size=args.batch_size, num_workers=args.num_workers)
        print_evaluation_summary(m_test)
        m_unseen, _ = evaluate_manifest(model, str(unseen_path), device, batch_size=args.batch_size, num_workers=args.num_workers)
        print_evaluation_summary(m_unseen)


if __name__ == "__main__":
    main()
