"""
Raspberry Pi 5 CPU-only Vietnamese License Plate OCR Deployment Script.
Pure OpenCV DNN + NumPy pipeline (NO PyTorch, NO Ultralytics, NO ONNX Runtime).

Features:
- Fixed input: 1 x 3 x 48 x 160
- Sobel-based two-row plate rectification (matches training exactly)
- Letterbox padding preserving aspect ratio
- OpenCV DNN inference with DNN_BACKEND_OPENCV and DNN_TARGET_CPU
- NumPy CTC greedy decoding
- Single image inference, batch evaluation, and latency benchmarking
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np

# Standard 36 alphanumeric characters + 1 blank (index 36) = 37 classes
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_INDEX = len(CHARSET)  # 36
NUM_CLASSES = len(CHARSET) + 1  # 37


def normalize_plate_text(text: str) -> str:
    """Normalizes plate string to uppercase alphanumeric only."""
    if not text:
        return ""
    import re
    return re.sub(r"[^A-Z0-9]", "", text.strip().upper())


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def calculate_cer(gt: str, pred: str) -> float:
    """Calculates Character Error Rate (CER)."""
    gt_c = normalize_plate_text(gt)
    pred_c = normalize_plate_text(pred)
    if len(gt_c) == 0:
        return 0.0 if len(pred_c) == 0 else 1.0
    return levenshtein_distance(gt_c, pred_c) / float(len(gt_c))


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
    """Splits two-row plate and concatenates horizontally with 6px spacer."""
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

    # Determine axis
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


class PiPlateOCR:
    """OpenCV DNN inference engine for LPRNet-Lite ONNX on Raspberry Pi."""
    def __init__(self, model_path: str):
        self.model_path = str(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def preprocess(self, img_bgr: np.ndarray, layout: str = "auto") -> np.ndarray:
        """Runs rectification and letterboxing."""
        h, w = img_bgr.shape[:2]
        is_two_row = False
        if layout in ("two_rows", "two_rows_label_xe_may", "motorcycle"):
            is_two_row = True
        elif layout == "auto":
            is_two_row = (w / float(h)) < 2.0

        if is_two_row:
            rectified = rectify_two_row_plate(img_bgr, target_h=48)
        else:
            rectified = img_bgr

        letterboxed = letterbox_ocr(rectified, target_size=(160, 48))
        return letterboxed

    def predict(self, plate_bgr: np.ndarray, layout: str = "auto") -> Tuple[str, Dict[str, float]]:
        """Runs full prediction and returns timings."""
        t0 = time.perf_counter()
        letterboxed = self.preprocess(plate_bgr, layout=layout)
        t_pre = time.perf_counter()

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

        pred_text = ctc_greedy_decode(logits, blank_index=BLANK_INDEX)
        t_dec = time.perf_counter()

        timings = {
            "preprocess_ms": (t_pre - t0) * 1000.0,
            "forward_ms": (t_fwd - t_pre) * 1000.0,
            "decode_ms": (t_dec - t_fwd) * 1000.0,
            "total_ms": (t_dec - t0) * 1000.0,
        }
        return pred_text, timings


def run_benchmark(ocr: PiPlateOCR, img_bgr: np.ndarray, layout: str = "auto", runs: int = 100):
    print(f"\n--- Running OCR Benchmark ({runs} iterations) ---")
    # Warmup 5 runs
    for _ in range(5):
        ocr.predict(img_bgr, layout=layout)

    pre_times = []
    fwd_times = []
    dec_times = []
    tot_times = []

    for _ in range(runs):
        _, t = ocr.predict(img_bgr, layout=layout)
        pre_times.append(t["preprocess_ms"])
        fwd_times.append(t["forward_ms"])
        dec_times.append(t["decode_ms"])
        tot_times.append(t["total_ms"])

    avg_pre = float(np.mean(pre_times))
    avg_fwd = float(np.mean(fwd_times))
    avg_dec = float(np.mean(dec_times))
    avg_tot = float(np.mean(tot_times))
    min_tot = float(np.min(tot_times))
    max_tot = float(np.max(tot_times))
    plates_per_sec = 1000.0 / avg_tot if avg_tot > 0 else 0.0

    print("PERFORMANCE RESULTS:")
    print(f"  preprocess average ms: {avg_pre:.2f}")
    print(f"  inference average ms:  {avg_fwd:.2f}")
    print(f"  decode average ms:     {avg_dec:.2f}")
    print(f"  total average ms:      {avg_tot:.2f}")
    print(f"  min total ms:          {min_tot:.2f}")
    print(f"  max total ms:          {max_tot:.2f}")
    print(f"  plates/sec:            {plates_per_sec:.2f}")
    return {
        "preprocess_avg_ms": avg_pre,
        "inference_avg_ms": avg_fwd,
        "decode_avg_ms": avg_dec,
        "total_avg_ms": avg_tot,
        "min_ms": min_tot,
        "max_ms": max_tot,
        "plates_per_sec": plates_per_sec,
    }


def run_batch_evaluation(ocr: PiPlateOCR, csv_path: str, crop_dir: Optional[str] = None):
    print(f"\n--- Running Batch Evaluation from {csv_path} ---")
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)

    total_samples = 0
    total_exact = 0
    total_cer = 0.0

    layout_stats = {
        "one_row": {"total": 0, "exact": 0, "cer_sum": 0.0},
        "two_rows": {"total": 0, "exact": 0, "cer_sum": 0.0},
        "motorcycle": {"total": 0, "exact": 0, "cer_sum": 0.0},
    }

    results = []
    base_dir = Path(crop_dir) if crop_dir else Path(csv_path).parent

    for r in records:
        fname = r.get("filename", "")
        gt = r.get("plate_text", "")
        layout = r.get("layout", "auto")

        img_path = base_dir / fname if not os.path.isabs(fname) else Path(fname)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: could not read {img_path}")
            continue

        pred, timings = ocr.predict(img, layout=layout)
        gt_c = normalize_plate_text(gt)
        pred_c = normalize_plate_text(pred)

        exact = (gt_c == pred_c)
        cer = calculate_cer(gt_c, pred_c)

        total_samples += 1
        if exact:
            total_exact += 1
        total_cer += cer

        l_key = "motorcycle" if layout in ("motorcycle", "two_rows_label_xe_may") else ("two_rows" if layout == "two_rows" else "one_row")
        if l_key not in layout_stats:
            layout_stats[l_key] = {"total": 0, "exact": 0, "cer_sum": 0.0}
        layout_stats[l_key]["total"] += 1
        if exact:
            layout_stats[l_key]["exact"] += 1
        layout_stats[l_key]["cer_sum"] += cer

        results.append({
            "filename": fname,
            "ground_truth": gt_c,
            "prediction": pred_c,
            "exact": exact,
            "cer": cer,
            "layout": l_key,
            "total_ms": timings["total_ms"]
        })

    overall_exact_acc = (total_exact / total_samples) * 100.0 if total_samples > 0 else 0.0
    overall_cer = total_cer / total_samples if total_samples > 0 else 0.0
    overall_char_acc = max(0.0, (1.0 - overall_cer) * 100.0)

    print("\nACCURACY RESULTS ON PI:")
    print(f"  samples:            {total_samples}")
    print(f"  Overall Exact:      {overall_exact_acc:.2f}%")
    print(f"  Character Accuracy: {overall_char_acc:.2f}%")
    print(f"  CER:                {overall_cer:.4f}")

    for l_key, stats in layout_stats.items():
        n = stats["total"]
        if n > 0:
            ex = (stats["exact"] / n) * 100.0
            c = stats["cer_sum"] / n
            print(f"  {l_key.upper()} ({n} samples): Exact={ex:.2f}% | CER={c:.4f}")

    return {
        "samples": total_samples,
        "exact_accuracy": overall_exact_acc,
        "char_accuracy": overall_char_acc,
        "cer": overall_cer,
        "layout_stats": layout_stats,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 OCR Plate Recognition")
    parser.add_argument("--model", type=str, default="models/best_lprnet_vn.onnx", help="Path to ONNX model")
    parser.add_argument("--image", type=str, default="", help="Single image path")
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "one_row", "two_rows", "motorcycle"], help="Plate layout")
    parser.add_argument("--ground_truth", type=str, default="", help="Expected plate text")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark run count")
    parser.add_argument("--csv", type=str, default="", help="Batch evaluation CSV path")
    parser.add_argument("--crop_dir", type=str, default="", help="Directory containing test crops")
    args = parser.parse_args()

    # Print Environment info
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Model path:     {args.model}")

    ocr = PiPlateOCR(model_path=args.model)
    print("OpenCV DNN load: PASS (DNN_BACKEND_OPENCV, DNN_TARGET_CPU)")

    if args.csv:
        run_batch_evaluation(ocr, args.csv, crop_dir=args.crop_dir)
    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {args.image}")

        if args.benchmark:
            run_benchmark(ocr, img, layout=args.layout, runs=args.runs)
        else:
            pred, timings = ocr.predict(img, layout=args.layout)
            print(f"\nImage:       {args.image}")
            print(f"Prediction:  {pred}")
            print(f"Inference:   {timings['forward_ms']:.2f} ms")
            print(f"Total:       {timings['total_ms']:.2f} ms")

            if args.ground_truth:
                gt_c = normalize_plate_text(args.ground_truth)
                pred_c = normalize_plate_text(pred)
                is_match = (gt_c == pred_c)
                print(f"Ground truth: {gt_c}")
                print(f"Exact match:  {'YES' if is_match else 'NO'}")
    else:
        print("Please provide --image or --csv.")


if __name__ == "__main__":
    main()
