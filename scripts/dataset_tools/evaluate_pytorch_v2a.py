"""
Evaluates PyTorch LPRNet-Lite V2-A checkpoint (best_lprnet_v2a.pth) on:
- VAL (datasets/VNLP_processed/val.csv)
- TEST_FINAL (datasets/VNLP_processed/test.csv)
- TEST_CLEAN (datasets/VNLP_processed/test_clean.csv)
- TEST_UNSEEN_TEXT (datasets/VNLP_processed/test_unseen_text.csv)

Breakdown:
- one_row
- two_rows
- motorcycle (two_rows_label_xe_may)

Generates:
- reports/v2a_pytorch_evaluation.md
"""

import argparse
import ast
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.charset import CHARSET, NUM_CLASSES, BLANK_INDEX
from ocr.models.lprnet_lite import LPRNetLite
from ocr.dataset import create_ocr_dataloader
from ocr.ctc_decoder import ctc_greedy_decode_torch
from ocr.plate_text_utils import normalize_plate_text, calculate_cer, levenshtein_distance


def evaluate_manifest_pytorch(
    model: torch.nn.Module,
    manifest_csv: Path,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[str, Any]:
    loader = create_ocr_dataloader(
        manifest_csv=str(manifest_csv),
        batch_size=batch_size,
        is_training=False,
        num_workers=num_workers,
    )

    total_samples = 0
    total_exact = 0
    total_cer = 0.0
    total_edit_dist = 0.0

    layout_stats = {
        "one_row": {"total": 0, "exact": 0, "cer_sum": 0.0, "edit_dist_sum": 0.0},
        "two_rows": {"total": 0, "exact": 0, "cer_sum": 0.0, "edit_dist_sum": 0.0},
        "two_rows_label_xe_may": {"total": 0, "exact": 0, "cer_sum": 0.0, "edit_dist_sum": 0.0},
    }

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            raw_texts = batch["raw_texts"]
            layouts = batch["layouts"]

            logits = model(images)
            preds = ctc_greedy_decode_torch(logits, blank_index=BLANK_INDEX)

            for gt, pred, layout in zip(raw_texts, preds, layouts):
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

    exact_acc = (total_exact / total_samples) * 100.0 if total_samples > 0 else 0.0
    avg_cer = total_cer / total_samples if total_samples > 0 else 0.0
    char_acc = max(0.0, (1.0 - avg_cer) * 100.0)
    avg_edit_dist = total_edit_dist / total_samples if total_samples > 0 else 0.0

    res = {
        "total_samples": total_samples,
        "correct": total_exact,
        "exact_accuracy": exact_acc,
        "character_accuracy": char_acc,
        "cer": avg_cer,
        "avg_edit_distance": avg_edit_dist,
        "layout_breakdown": {},
    }

    for lay, stats in layout_stats.items():
        n = stats["total"]
        if n > 0:
            ex = (stats["exact"] / n) * 100.0
            c = stats["cer_sum"] / n
            e = stats["edit_dist_sum"] / n
            res["layout_breakdown"][lay] = {
                "total": n,
                "correct": stats["exact"],
                "exact_accuracy": ex,
                "cer": c,
                "char_accuracy": max(0.0, (1.0 - c) * 100.0),
                "avg_edit_distance": e,
            }

    print(f"  [{manifest_csv.name}] Total: {total_samples} | Exact: {total_exact}/{total_samples} ({exact_acc:.2f}%) | CharAcc: {char_acc:.2f}% | CER: {avg_cer:.4f} | AvgEditDist: {avg_edit_dist:.4f}")
    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch V2-A Model")
    parser.add_argument("--checkpoint", type=str, default="weights/ocr/experiments/v2a/best_lprnet_v2a.pth")
    parser.add_argument("--report", type=str, default="reports/v2a_pytorch_evaluation.md")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {ckpt_path}")

    model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    best_epoch = ckpt.get("epoch", "N/A")
    val_exact = ckpt.get("best_val_exact", "N/A")
    print(f"Loaded checkpoint saved from epoch {best_epoch} with val_exact = {val_exact}")

    manifests = {
        "VAL": Path("datasets/VNLP_processed/val.csv"),
        "TEST_FINAL": Path("datasets/VNLP_processed/test.csv"),
        "TEST_CLEAN": Path("datasets/VNLP_processed/test_clean.csv"),
        "TEST_UNSEEN_TEXT": Path("datasets/VNLP_processed/test_unseen_text.csv"),
    }

    results = {}
    for name, path in manifests.items():
        if path.exists():
            print(f"\nEvaluating PyTorch model on {name} ({path})...")
            results[name] = evaluate_manifest_pytorch(model, path, device=device)
        else:
            print(f"Warning: {path} not found!")

    # Generate Markdown Report
    lines = [
        "# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH PYTORCH LPRNET V2-A (V2-A PYTORCH EVALUATION)",
        "",
        "**Thời gian thực hiện:** 2026-09-24",
        f"**File Checkpoint:** `{ckpt_path}`",
        f"**Epoch tốt nhất:** `{best_epoch}`",
        f"**Thiết bị thực thi:** `{device}`",
        "",
        "---",
        "",
        "## 1. Kết Quả Tổng Thể trên Các Tập Đánh Giá",
        "| Tập dữ liệu | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name, r in results.items():
        lines.append(
            f"| **{name}** | {r['total_samples']} | {r['correct']}/{r['total_samples']} | "
            f"**{r['exact_accuracy']:.2f}%** | {r['character_accuracy']:.2f}% | "
            f"{r['cer']:.4f} | {r['avg_edit_distance']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Phân Tích Chi Tiết Theo Kiểu Biển (Layout Breakdown)",
    ])

    for name, r in results.items():
        lines.extend([
            f"### Tập dữ liệu: {name}",
            "| Kiểu biển (Layout) | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for lay, st in r["layout_breakdown"].items():
            lay_name = "1 Dòng (one_row)" if lay == "one_row" else ("2 Dòng Ô tô (two_rows)" if lay == "two_rows" else "2 Dòng Xe Máy (motorcycle)")
            lines.append(
                f"| {lay_name} | {st['total']} | {st['correct']}/{st['total']} | "
                f"**{st['exact_accuracy']:.2f}%** | {st['char_accuracy']:.2f}% | "
                f"{st['cer']:.4f} | {st['avg_edit_distance']:.4f} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Nhận Xét Khái Quát về Checkpoint PyTorch V2-A",
        f"- Checkpoint PyTorch tốt nhất đạt Exact Accuracy trên TEST_FINAL là **{results['TEST_FINAL']['exact_accuracy']:.2f}%** (so với V1 baseline 85.27%).",
        f"- Khả năng tổng quát hóa trên TEST_UNSEEN_TEXT đạt **{results['TEST_UNSEEN_TEXT']['exact_accuracy']:.2f}%** (so với V1 baseline 72.11%).",
        f"- Tập loại bỏ rò rỉ hình ảnh TEST_CLEAN đạt **{results['TEST_CLEAN']['exact_accuracy']:.2f}%**.",
        ""
    ])

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))

    print(f"\nPyTorch evaluation report saved to {report_path}")


if __name__ == "__main__":
    main()
