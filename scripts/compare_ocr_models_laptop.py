"""
Comprehensive Comparison of 3 OCR Models on Windows Laptop:
1. Official Pretrained PARSeq-Tiny (baudm/parseq)
2. Fine-Tuned PARSeq-Tiny VNLP (weights/ocr/teachers/parseq_tiny_vn/best.pth)
3. Baseline LPRNetLite (weights/ocr/experiments/phase4/control_seed42/best.pth)

CLI:
  python scripts/compare_ocr_models_laptop.py --folder TEST_FOLDER
  python scripts/compare_ocr_models_laptop.py --manifest MANIFEST.csv
"""

import argparse
import ast
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.charset import CHARSET, NUM_CLASSES, BLANK_INDEX
from ocr.ctc_decoder import ctc_greedy_decode_torch
from ocr.models.lprnet_lite import LPRNetLite
from ocr.plate_text_utils import normalize_plate_text, calculate_cer, levenshtein_distance
from ocr.preprocess import crop_plate_gt, rectify_two_row_plate, letterbox_ocr, normalize_image_for_model
from teacher_adapters.parseq_vn import parseq_predict_fn


def load_all_models(
    pretrained_repo: str = "baudm/parseq",
    vn_ckpt_path: str = "weights/ocr/teachers/parseq_tiny_vn/best.pth",
    lprnet_ckpt_path: str = "weights/ocr/experiments/phase4/control_seed42/best.pth",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    print("=" * 75)
    print(f"LOADING ALL 3 OCR MODELS ON DEVICE: {device}")
    print("=" * 75)

    # 1. PARSeq Official Pretrained
    print("1/3 Loading Official Pretrained PARSeq-Tiny...")
    parseq_pre = torch.hub.load(pretrained_repo, "parseq_tiny", pretrained=True, trust_repo=True)
    parseq_pre = parseq_pre.to(device).eval()

    # 2. PARSeq VNLP Fine-Tuned
    print(f"2/3 Loading Fine-Tuned PARSeq-Tiny from {vn_ckpt_path}...")
    parseq_vn = torch.hub.load(pretrained_repo, "parseq_tiny", pretrained=False, trust_repo=True)
    ckpt_vn = torch.load(vn_ckpt_path, map_location=device, weights_only=False)
    state_vn = ckpt_vn["model_state_dict"] if "model_state_dict" in ckpt_vn else ckpt_vn
    parseq_vn.load_state_dict(state_vn)
    parseq_vn = parseq_vn.to(device).eval()

    # 3. LPRNetLite Baseline
    print(f"3/3 Loading LPRNetLite Baseline from {lprnet_ckpt_path}...")
    lprnet = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
    ckpt_lpr = torch.load(lprnet_ckpt_path, map_location=device, weights_only=False)
    state_lpr = ckpt_lpr["model_state_dict"] if "model_state_dict" in ckpt_lpr else ckpt_lpr
    lprnet.load_state_dict(state_lpr)
    lprnet = lprnet.to(device).eval()

    print("All 3 models loaded successfully into eval mode.\n")
    return parseq_pre, parseq_vn, lprnet


def parse_sample_metadata(img_path: Path, manifest_row: Optional[Dict[str, Any]] = None) -> Tuple[str, str, Optional[List[int]]]:
    """Extracts (ground_truth, layout, bbox) from manifest row or filename."""
    if manifest_row:
        gt = manifest_row.get("plate_text", "")
        layout = manifest_row.get("subset", manifest_row.get("layout", "one_row"))
        bbox = None
        if "x1" in manifest_row and manifest_row["x1"] and manifest_row["x1"].strip():
            try:
                bbox = [int(float(manifest_row["x1"])), int(float(manifest_row["y1"])), int(float(manifest_row["x2"])), int(float(manifest_row["y2"]))]
            except Exception:
                bbox = None
        elif "bbox" in manifest_row and manifest_row["bbox"] and manifest_row["bbox"].strip():
            try:
                bbox = ast.literal_eval(manifest_row["bbox"])
            except Exception:
                bbox = None
        return gt, layout, bbox

    # From filename heuristic: crop_00_one_row_54S9449.jpg
    stem = img_path.stem
    gt = "N/A"
    layout = "one_row"
    if "two_rows" in stem:
        layout = "two_rows"
    elif "motorcycle" in stem:
        layout = "two_rows_label_xe_may"
    elif "one_row" in stem:
        layout = "one_row"

    parts = stem.split("_")
    if len(parts) >= 4:
        cand_gt = parts[-1].strip().upper()
        if any(c.isdigit() for c in cand_gt) and any(c.isalpha() for c in cand_gt):
            gt = cand_gt

    return gt, layout, None


def run_comparison(
    folder_path: Optional[str] = None,
    manifest_csv: Optional[str] = None,
    output_csv: str = "outputs/laptop_ocr_model_comparison.csv",
    max_samples: Optional[int] = None,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    warmup_n: int = 10,
):
    device = torch.device(device_str)
    parseq_pre, parseq_vn, lprnet = load_all_models(device=device)

    # Transforms
    parseq_pre_transform = T.Compose([
        T.Resize((32, 128), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])

    # Discover samples
    samples = []
    if manifest_csv:
        man_p = Path(manifest_csv)
        base_dir = man_p.parent
        with open(man_p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = Path(row["image_path"])
                if not img_path.is_absolute() and not img_path.exists():
                    alt = base_dir / img_path
                    if alt.exists():
                        img_path = alt
                gt, layout, bbox = parse_sample_metadata(img_path, manifest_row=row)
                samples.append({
                    "image_path": str(img_path),
                    "gt": gt,
                    "layout": layout,
                    "bbox": bbox,
                })
    elif folder_path:
        f_p = Path(folder_path)
        # Check if sibling test_plate_crops.csv exists in same folder
        sib_csv = f_p / "test_plate_crops.csv"
        sib_dict = {}
        if sib_csv.exists():
            with open(sib_csv, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    sib_dict[r["filename"]] = (r.get("plate_text", ""), r.get("layout", "one_row"))

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for p in sorted(f_p.glob(ext)):
                if p.name in sib_dict:
                    gt, layout = sib_dict[p.name]
                    bbox = None
                else:
                    gt, layout, bbox = parse_sample_metadata(p, None)
                samples.append({
                    "image_path": str(p),
                    "gt": gt,
                    "layout": layout,
                    "bbox": bbox,
                })

    if max_samples is not None and max_samples < len(samples):
        samples = samples[:max_samples]

    print(f"Total samples to benchmark: {len(samples)}")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Warmup
    if warmup_n > 0:
        print(f"Warming up GPU with {warmup_n} dummy passes...")
        for _ in range(warmup_n):
            d1 = torch.zeros(1, 3, 32, 128, device=device)
            d2 = torch.zeros(1, 3, 48, 160, device=device)
            with torch.no_grad():
                _ = parseq_pre(d1)
                _ = parseq_vn(d1)
                _ = lprnet(d2)
        if device.type == "cuda":
            torch.cuda.synchronize()

    results = []
    latencies_pre = []
    latencies_vn = []
    latencies_lpr = []

    for idx, s in enumerate(samples, 1):
        img_bgr = cv2.imread(s["image_path"])
        if img_bgr is None:
            continue

        # 1. Ground truth crop if bbox is present
        if s["bbox"] is not None and len(s["bbox"]) == 4:
            crop = crop_plate_gt(img_bgr, s["bbox"], pad_ratio=0.03)
        else:
            crop = img_bgr

        # 2. Layout classification & rectification
        h, w = crop.shape[:2]
        is_two_rows = False
        if s["layout"] in ("two_rows", "two_rows_label_xe_may"):
            is_two_rows = True
        elif (w / float(max(1, h))) < 1.9:
            is_two_rows = True

        if is_two_rows:
            rectified, _, _ = rectify_two_row_plate(crop, target_h=48)
        else:
            rectified = crop

        # --- MODEL 1: PARSeq Official Pretrained ---
        pil_pre = Image.fromarray(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
        pre_tensor = parseq_pre_transform(pil_pre).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pre_logits = parseq_pre(pre_tensor)
            pre_probs = pre_logits.softmax(-1)
            pre_preds, pre_confs = parseq_pre.tokenizer.decode(pre_probs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pre_ms = (time.perf_counter() - t0) * 1000.0
        latencies_pre.append(pre_ms)
        raw_pre = pre_preds[0] if pre_preds else ""
        norm_pre = normalize_plate_text(raw_pre)
        conf_pre = float(pre_confs[0].mean().item()) if hasattr(pre_confs[0], "mean") else 0.95

        # --- MODEL 2: PARSeq VNLP Fine-Tuned ---
        vn_resized = cv2.resize(rectified, (128, 32), interpolation=cv2.INTER_LINEAR)
        vn_rgb = cv2.cvtColor(vn_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        vn_tensor = torch.from_numpy((vn_rgb - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0).float().to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            vn_preds, vn_confs = parseq_predict_fn(parseq_vn, vn_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        vn_ms = (time.perf_counter() - t0) * 1000.0
        latencies_vn.append(vn_ms)
        pred_vn = vn_preds[0] if vn_preds else ""
        conf_vn = vn_confs[0] if vn_confs else 0.95

        # --- MODEL 3: LPRNetLite Baseline ---
        lpr_letterbox = letterbox_ocr(rectified, target_size=(160, 48), pad_color=(255, 255, 255))
        lpr_norm = normalize_image_for_model(lpr_letterbox)
        lpr_tensor = torch.from_numpy(lpr_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            lpr_out = lprnet(lpr_tensor)
            lpr_preds = ctc_greedy_decode_torch(lpr_out, blank_index=BLANK_INDEX)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lpr_ms = (time.perf_counter() - t0) * 1000.0
        latencies_lpr.append(lpr_ms)
        pred_lpr = normalize_plate_text(lpr_preds[0] if lpr_preds else "")

        gt_norm = normalize_plate_text(s["gt"]) if s["gt"] != "N/A" else "N/A"
        has_gt = (gt_norm != "N/A" and len(gt_norm) > 0)

        pre_corr = (norm_pre == gt_norm) if has_gt else ""
        vn_corr = (pred_vn == gt_norm) if has_gt else ""
        lpr_corr = (pred_lpr == gt_norm) if has_gt else ""

        row = {
            "image_path": s["image_path"],
            "layout": s["layout"],
            "ground_truth": s["gt"],
            "parseq_pretrained_raw": raw_pre,
            "parseq_pretrained_normalized": norm_pre,
            "parseq_pretrained_correct": pre_corr,
            "parseq_pretrained_confidence": round(conf_pre, 4),
            "parseq_pretrained_ms": round(pre_ms, 2),
            "parseq_vn_prediction": pred_vn,
            "parseq_vn_correct": vn_corr,
            "parseq_vn_confidence": round(conf_vn, 4),
            "parseq_vn_ms": round(vn_ms, 2),
            "lprnet_prediction": pred_lpr,
            "lprnet_correct": lpr_corr,
            "lprnet_ms": round(lpr_ms, 2),
        }
        results.append(row)

        if idx <= 10 or idx % 50 == 0 or idx == len(samples):
            print(f"[{idx:03d}/{len(samples):03d}] GT: {gt_norm:<10} | Pre: {norm_pre:<10} ({pre_corr}) | VN: {pred_vn:<10} ({vn_corr}) | LPR: {pred_lpr:<10} ({lpr_corr})")

    # Write output CSV
    fieldnames = [
        "image_path", "layout", "ground_truth",
        "parseq_pretrained_raw", "parseq_pretrained_normalized", "parseq_pretrained_correct", "parseq_pretrained_confidence", "parseq_pretrained_ms",
        "parseq_vn_prediction", "parseq_vn_correct", "parseq_vn_confidence", "parseq_vn_ms",
        "lprnet_prediction", "lprnet_correct", "lprnet_ms",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved comparison results to: {output_csv}")

    # Compute metrics if GT available
    gt_rows = [r for r in results if r["ground_truth"] != "N/A" and r["ground_truth"].strip()]
    if gt_rows:
        n_gt = len(gt_rows)
        pre_exact = sum(1 for r in gt_rows if r["parseq_pretrained_correct"] is True) / n_gt * 100.0
        vn_exact = sum(1 for r in gt_rows if r["parseq_vn_correct"] is True) / n_gt * 100.0
        lpr_exact = sum(1 for r in gt_rows if r["lprnet_correct"] is True) / n_gt * 100.0

        pre_cer = np.mean([calculate_cer(r["ground_truth"], r["parseq_pretrained_normalized"]) for r in gt_rows])
        vn_cer = np.mean([calculate_cer(r["ground_truth"], r["parseq_vn_prediction"]) for r in gt_rows])
        lpr_cer = np.mean([calculate_cer(r["ground_truth"], r["lprnet_prediction"]) for r in gt_rows])

        print("\n" + "=" * 75)
        print(f"ACCURACY BENCHMARK ON {n_gt} SAMPLES WITH GROUND TRUTH:")
        print(f"  Model                   | Exact Acc (%) | Char Acc (%) | CER    | Mean Latency")
        print(f"  ----------------------- | ------------- | ------------ | ------ | ------------")
        print(f"  PARSeq Pretrained (Off) | {pre_exact:11.2f}% | {100-pre_cer*100:10.2f}% | {pre_cer:6.4f} | {np.mean(latencies_pre):6.2f} ms")
        print(f"  PARSeq VNLP Fine-Tuned  | {vn_exact:11.2f}% | {100-vn_cer*100:10.2f}% | {vn_cer:6.4f} | {np.mean(latencies_vn):6.2f} ms")
        print(f"  LPRNetLite Baseline     | {lpr_exact:11.2f}% | {100-lpr_cer*100:10.2f}% | {lpr_cer:6.4f} | {np.mean(latencies_lpr):6.2f} ms")
        print("=" * 75)

    return results, {
        "pre_latencies": latencies_pre,
        "vn_latencies": latencies_vn,
        "lpr_latencies": latencies_lpr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare OCR Models on Windows Laptop")
    parser.add_argument("--folder", type=str, default=None, help="Folder containing plate crops")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV")
    parser.add_argument("--output_csv", type=str, default="outputs/laptop_ocr_model_comparison.csv")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.folder and not args.manifest:
        parser.error("Must provide either --folder or --manifest")

    run_comparison(
        folder_path=args.folder,
        manifest_csv=args.manifest,
        output_csv=args.output_csv,
        max_samples=args.max_samples,
        device_str=args.device,
    )
