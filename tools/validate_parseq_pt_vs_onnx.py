"""
Validate PyTorch PARSeq-Tiny VNLP vs ONNX Runtime CPU PARSeq on 100 real plate crops.
Models:
  - PyTorch: weights/ocr/teachers/parseq_tiny_vn/best.pth
  - ONNX: weights/deployment/pi_cpu/parseq_tiny_vn_fp32.onnx (CPUExecutionProvider)
Manifest: reports/fresh_holdout_500_manifest.csv (first 100 samples)
Evaluates:
  - Preprocessing identity (tensor generated ONCE)
  - Logits shape match [1, 26, 95]
  - Mean absolute error & Max absolute error
  - Argmax token agreement across all sequence steps
  - Decoded text agreement (PyTorch text == ONNX text)
  - Disagreement logging if any
Outputs:
  - Detailed console report
  - Appends to reports/pi_cpu_onnx_equivalence_results.csv
"""

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ocr.parseq_vn_ocr import ParseqVNRecognizer
from scripts.laptop_parseq_recognize import crop_with_padding


def main():
    print("=" * 80)
    print("PARSeq-TINY VNLP OCR PYTORCH VS ONNX RUNTIME CPU EQUIVALENCE VALIDATION")
    print("=" * 80)

    pt_model_path = PROJECT_ROOT / "weights/ocr/teachers/parseq_tiny_vn/best.pth"
    onnx_model_path = PROJECT_ROOT / "weights/deployment/pi_cpu/parseq_tiny_vn_fp32.onnx"
    manifest_path = PROJECT_ROOT / "reports/fresh_holdout_500_manifest.csv"

    if not pt_model_path.exists():
        raise FileNotFoundError(f"PyTorch model not found: {pt_model_path}")
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Load 100 samples
    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))[:100]

    layout_counts = {}
    for s in samples:
        lay = s.get("gt_layout", "unknown")
        layout_counts[lay] = layout_counts.get(lay, 0) + 1
    print(f"Loaded {len(samples)} evaluation samples from {manifest_path.name}")
    print(f"Layout distribution: {layout_counts}")

    # Initialize PyTorch Recognizer on CPU
    print(f"\nInitializing PyTorch PARSeq Recognizer on CPU: {pt_model_path}")
    rec = ParseqVNRecognizer(str(pt_model_path), device="cpu")
    tokenizer = rec.model.tokenizer

    # Initialize ONNX Runtime Session on CPU
    print(f"Initializing ONNX Runtime CPU session: {onnx_model_path}")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_model_path), sess_options, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    mean_abs_errors = []
    max_abs_errors = []
    token_agreements = []
    text_matches = []
    disagreements = []
    results_records = []

    for idx, sample in enumerate(samples, 1):
        sample_id = sample["sample_id"]
        img_path = sample["image_path"]
        gt_text = sample["ground_truth"].strip().upper()
        gt_box = [int(float(sample[k])) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")]
        gt_layout = sample["gt_layout"]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # Crop plate with identical 5% padding
        crop, _ = crop_with_padding(img, gt_box, pad_ratio=0.05)

        # Generate preprocessed tensor ONCE
        tensor, layout_pred, _, _ = rec.prepare_input(crop)
        # Shape [1, 3, 32, 128]
        inp_tensor = tensor.unsqueeze(0)
        inp_np = inp_tensor.numpy()

        # 1. PyTorch CPU Inference
        t0_pt = time.perf_counter()
        with torch.no_grad():
            pt_logits = rec.model(inp_tensor)  # [1, 26, 95]
        t_pt_ms = (time.perf_counter() - t0_pt) * 1000.0

        pt_probs = torch.softmax(pt_logits, -1)
        pt_preds, pt_confs = tokenizer.decode(pt_probs)
        pt_str = pt_preds[0]
        if pt_confs and len(pt_confs) > 0:
            c = pt_confs[0]
            pt_conf = float(c.mean().item()) if hasattr(c, "mean") else float(np.mean(c))
        else:
            pt_conf = 0.95

        # 2. ONNX Runtime CPU Inference
        t0_onnx = time.perf_counter()
        onnx_logits = sess.run(None, {input_name: inp_np})[0]  # [1, 26, 95]
        t_onnx_ms = (time.perf_counter() - t0_onnx) * 1000.0

        onnx_probs = torch.softmax(torch.from_numpy(onnx_logits), -1)
        onnx_preds, onnx_confs = tokenizer.decode(onnx_probs)
        onnx_str = onnx_preds[0]
        if onnx_confs and len(onnx_confs) > 0:
            c = onnx_confs[0]
            onnx_conf = float(c.mean().item()) if hasattr(c, "mean") else float(np.mean(c))
        else:
            onnx_conf = 0.95

        # Numerical diffs
        abs_diff = np.abs(pt_logits.numpy() - onnx_logits)
        mean_diff = float(np.mean(abs_diff))
        max_diff = float(np.max(abs_diff))
        mean_abs_errors.append(mean_diff)
        max_abs_errors.append(max_diff)

        # Argmax token agreement across all 26 positions
        pt_tokens = pt_logits.argmax(dim=-1).numpy()[0]
        onnx_tokens = onnx_logits.argmax(axis=-1)[0]
        tok_agree = float(np.mean(pt_tokens == onnx_tokens))
        token_agreements.append(tok_agree)

        # Decoded text equality
        match = (pt_str == onnx_str)
        text_matches.append(match)

        if not match:
            disagreements.append({
                "sample_id": sample_id,
                "gt": gt_text,
                "pt": pt_str,
                "onnx": onnx_str,
                "pt_conf": pt_conf,
                "onnx_conf": onnx_conf,
                "max_diff": max_diff,
            })

        results_records.append({
            "sample_id": sample_id,
            "model": "parseq_tiny_vn",
            "image_path": img_path,
            "ground_truth": gt_text,
            "pytorch_prediction": pt_str,
            "onnx_prediction": onnx_str,
            "prediction_match": str(match),
            "mean_abs_diff": f"{mean_diff:.6e}",
            "max_abs_diff": f"{max_diff:.6e}",
            "bbox_iou_pt_vs_onnx": "1.0000",  # Exact same crop fed
            "confidence_diff": f"{abs(pt_conf - onnx_conf):.6e}",
            "latency_ms": f"{t_onnx_ms:.2f}",
        })

    # Summary
    total_samples = len(samples)
    match_count = sum(text_matches)
    match_rate = (match_count / total_samples) * 100.0

    print("\n" + "=" * 80)
    print("PARSeq-TINY VNLP EQUIVALENCE SUMMARY (100 REAL PLATE CROPS)")
    print("=" * 80)
    print(f"Total Plate Crops Evaluated       : {total_samples}")
    print(f"Layouts Covered                   : {layout_counts}")
    print(f"Decoded Text Exact Matches        : {match_count} / {total_samples} ({match_rate:.2f}%)")
    print(f"Mean Token Argmax Agreement       : {np.mean(token_agreements)*100:.2f}%")
    print(f"Logits Mean Absolute Error        : {np.mean(mean_abs_errors):.6e}")
    print(f"Logits Max Absolute Error         : {np.max(max_abs_errors):.6e}")
    print("-" * 80)

    if disagreements:
        print(f"WARNING: {len(disagreements)} DISAGREEMENTS DETECTED:")
        for d in disagreements:
            print(f"  Sample {d['sample_id']}: GT={d['gt']} | PT={d['pt']} ({d['pt_conf']:.3f}) | ONNX={d['onnx']} ({d['onnx_conf']:.3f}) | MaxDiff={d['max_diff']:.6e}")
    else:
        print("PERFECT 100.00% DECODED TEXT EQUIVALENCE ACHIEVED! ZERO DISAGREEMENTS.")

    print("=" * 80)

    # Append to existing CSV
    csv_path = PROJECT_ROOT / "reports/pi_cpu_onnx_equivalence_results.csv"
    fieldnames = [
        "sample_id", "model", "image_path", "ground_truth",
        "pytorch_prediction", "onnx_prediction", "prediction_match",
        "mean_abs_diff", "max_abs_diff", "bbox_iou_pt_vs_onnx",
        "confidence_diff", "latency_ms"
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(results_records)

    print(f"Appended PARSeq results to: {csv_path}")


if __name__ == "__main__":
    main()
