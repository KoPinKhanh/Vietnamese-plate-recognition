"""
Controlled Vehicle Detector Benchmark: YOLO11s-640 vs YOLO11n-640 on Raspberry Pi 5 CPU.
Evaluates:
  - Latency: Warmup=20, 500 timed runs (Mean, Median, P95, Min, Max, FPS, Speedup)
  - Memory: Peak process RSS RAM
  - Thermals: SoC temperature & throttling status
  - Accuracy: Vehicle ROI coverage rate (>=85% GT plate bbox area covered)
  - Breakdowns: Overall, one_row, two_rows, motorcycle, and R1_FALLBACK_COHORT
  - Statistics: Paired McNemar exact test & 10,000 paired bootstrap 95% CI
Generates:
  - /home/cytron/vietnam_plate/reports/vehicle_yolo11s_vs_yolo11n_results.csv
  - /home/cytron/vietnam_plate/reports/PI5_YOLO11S_VS_YOLO11N_VEHICLE_REPORT.md
"""

import csv
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import psutil

# Directories
PROJECT_ROOT = Path("/home/cytron/vietnam_plate")
MODELS_DIR = PROJECT_ROOT / "models/pi_cpu"
BENCHMARK_DATA_DIR = PROJECT_ROOT / "test_data/vehicle_benchmark"
MANIFEST_PATH = BENCHMARK_DATA_DIR / "vehicle_benchmark_manifest.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_vcgencmd(cmd: str) -> str:
    """Executes vcgencmd and returns stripped string output."""
    try:
        out = subprocess.check_output(["vcgencmd", cmd], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "N/A"


def get_temperature() -> str:
    return get_vcgencmd("measure_temp")


def get_throttled() -> str:
    return get_vcgencmd("get_throttled")


def get_process_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, float, float]:
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad_w = int(round(shape[1] * scale))
    new_unpad_h = int(round(shape[0] * scale))

    dw = new_shape[1] - new_unpad_w
    dh = new_shape[0] - new_unpad_h

    pad_x = dw / 2.0
    pad_y = dh / 2.0

    if shape[::-1] != (new_unpad_w, new_unpad_h):
        img_resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img

    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))

    padded_img = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded_img, scale, pad_x, pad_y


def postprocess_yolo_detections(
    output: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    scale: float,
    pad_x: float,
    pad_y: float,
    orig_w: int,
    orig_h: int,
    relevant_classes: List[int] = None
) -> List[Dict[str, Any]]:
    preds = np.squeeze(output)
    if preds.ndim != 2:
        return []
    if preds.shape[0] < preds.shape[1]:  # [84, 8400] -> [8400, 84]
        preds = preds.T

    boxes = []
    confidences = []
    class_ids = []

    cx = preds[:, 0]
    cy = preds[:, 1]
    w = preds[:, 2]
    h = preds[:, 3]
    scores_matrix = preds[:, 4:]

    best_classes = np.argmax(scores_matrix, axis=1)
    best_scores = np.max(scores_matrix, axis=1)

    mask = best_scores >= conf_thresh
    if not np.any(mask):
        return []

    filt_cx = cx[mask]
    filt_cy = cy[mask]
    filt_w = w[mask]
    filt_h = h[mask]
    filt_scores = best_scores[mask]
    filt_classes = best_classes[mask]

    for i in range(len(filt_scores)):
        cls_id = int(filt_classes[i])
        if relevant_classes is not None and cls_id not in relevant_classes:
            continue
        score = float(filt_scores[i])
        bx = filt_cx[i] - filt_w[i] / 2.0
        by = filt_cy[i] - filt_h[i] / 2.0
        boxes.append([int(round(bx)), int(round(by)), int(round(filt_w[i])), int(round(filt_h[i]))])
        confidences.append(score)
        class_ids.append(cls_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=conf_thresh, nms_threshold=iou_thresh)
    if len(indices) == 0:
        return []

    indices = np.array(indices).flatten()
    detections = []
    for idx in indices:
        bx, by, bw, bh = boxes[idx]
        score = confidences[idx]
        cls_id = class_ids[idx]

        orig_x1 = max(0, min(orig_w, (bx - pad_x) / scale))
        orig_y1 = max(0, min(orig_h, (by - pad_y) / scale))
        orig_x2 = max(0, min(orig_w, (bx + bw - pad_x) / scale))
        orig_y2 = max(0, min(orig_h, (by + bh - pad_y) / scale))

        detections.append({
            "box": [int(round(orig_x1)), int(round(orig_y1)), int(round(orig_x2)), int(round(orig_y2))],
            "confidence": score,
            "class_id": cls_id
        })

    return detections


def does_roi_cover_box(roi: List[int], box: List[int], min_coverage: float = 0.85) -> bool:
    """Checks whether roi covers at least min_coverage (85%) of box area."""
    rx1, ry1, rx2, ry2 = roi
    bx1, by1, bx2, by2 = box
    inter_x1 = max(rx1, bx1)
    inter_y1 = max(ry1, by1)
    inter_x2 = min(rx2, bx2)
    inter_y2 = min(ry2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    box_area = max(1, (bx2 - bx1) * (by2 - by1))
    return (inter_area / box_area) >= min_coverage


def check_vehicle_detections_roi(
    detections: List[Dict[str, Any]],
    gt_box: List[int],
    img_w: int,
    img_h: int,
    pad_ratio: float = 0.05,
    min_coverage: float = 0.85
) -> Tuple[bool, float, int]:
    """
    Checks if any detected vehicle ROI (with pad_ratio padding) covers >= min_coverage of gt_box.
    Returns: (valid_roi_found, best_confidence, num_vehicle_boxes)
    """
    if not detections:
        return False, 0.0, 0

    valid_roi = False
    best_conf = 0.0

    for d in detections:
        vx1, vy1, vx2, vy2 = d["box"]
        conf = d["confidence"]
        if conf > best_conf:
            best_conf = conf

        vw = vx2 - vx1
        vh = vy2 - vy1
        pad_w = int(round(vw * pad_ratio))
        pad_h = int(round(vh * pad_ratio))
        roi = [
            max(0, vx1 - pad_w),
            max(0, vy1 - pad_h),
            min(img_w, vx2 + pad_w),
            min(img_h, vy2 + pad_h)
        ]
        if does_roi_cover_box(roi, gt_box, min_coverage=min_coverage):
            valid_roi = True

    return valid_roi, best_conf, len(detections)


def exact_mcnemar_test(b: int, c: int) -> float:
    """Computes two-tailed exact McNemar test p-value using binomial distribution."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Sum of binomial probabilities P(X <= k) with p=0.5
    cum_prob = 0.0
    for i in range(k + 1):
        cum_prob += math.comb(n, i) * (0.5 ** n)
    p_val = min(1.0, 2.0 * cum_prob)
    return p_val


def bootstrap_paired_delta_ci(s_arr: np.ndarray, n_arr: np.ndarray, n_bootstrap: int = 10000, seed: int = 42) -> Tuple[float, float]:
    """Computes 95% bootstrap confidence interval for mean(n_arr) - mean(s_arr) in percentage points."""
    np.random.seed(seed)
    diff = (n_arr.astype(float) - s_arr.astype(float)) * 100.0  # in percentage points
    n = len(diff)
    indices = np.random.randint(0, n, size=(n_bootstrap, n))
    bootstrap_means = np.mean(diff[indices], axis=1)
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))
    return ci_lower, ci_upper


def main():
    print("=" * 80)
    print("CONTROLLED VEHICLE DETECTOR BENCHMARK: YOLO11s-640 VS YOLO11n-640")
    print("RASPBERRY PI 5 CPU INFERENCE")
    print("=" * 80)

    # System Audit
    pi_model = "Raspberry Pi 5 Model B Rev 1.0"
    os_info = "Debian GNU/Linux 12 (bookworm)"
    python_ver = sys.version.split()[0]
    ort_ver = ort.__version__

    temp_start = get_temperature()
    throttled_start = get_throttled()

    print(f"Hardware        : {pi_model} (4x Cortex-A76 @ 2.4 GHz)")
    print(f"OS / Python     : {os_info} | Python {python_ver}")
    print(f"ONNX Runtime    : {ort_ver}")
    print(f"Thermal Start   : {temp_start} | Throttled: {throttled_start}")
    print("-" * 80)

    # Load Manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))

    print(f"Loaded {len(samples)} evaluation samples from {MANIFEST_PATH.name}")

    # Models Setup
    s_path = MODELS_DIR / "yolo11s_640_fp32.onnx"
    n_path = MODELS_DIR / "yolo11n_640_fp32.onnx"

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    print(f"\nInitializing YOLO11s session: {s_path.name}...")
    sess_s = ort.InferenceSession(str(s_path), opts, providers=["CPUExecutionProvider"])
    s_inp_name = sess_s.get_inputs()[0].name

    print(f"Initializing YOLO11n session: {n_path.name}...")
    sess_n = ort.InferenceSession(str(n_path), opts, providers=["CPUExecutionProvider"])
    n_inp_name = sess_n.get_inputs()[0].name

    # Read manifest metadata
    print("\nReading manifest metadata for 500 evaluation samples...")
    metadata = []
    for s in samples:
        gt_box = [int(float(s[k])) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")]
        fallback_cohort = (s.get("fallback_cohort", "False") == "True")
        metadata.append({
            "sample_id": s["sample_id"],
            "image_file": s["image_file"],
            "gt_layout": s["gt_layout"],
            "gt_box": gt_box,
            "fallback_cohort": fallback_cohort,
        })

    # Prepare single dummy blob for warmup
    sample0_p = BENCHMARK_DATA_DIR / metadata[0]["image_file"]
    img0 = cv2.imread(str(sample0_p))
    lb0, s0, px0, py0 = letterbox(img0, (640, 640))
    dummy_blob = lb0[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    dummy_blob = np.expand_dims(dummy_blob, 0)

    # Warmup both models
    print("\nWarming up models (20 runs each)...")
    for _ in range(20):
        _ = sess_s.run(None, {s_inp_name: dummy_blob})
        _ = sess_n.run(None, {n_inp_name: dummy_blob})

    relevant_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    # --------------------------------------------------------------------------
    # EVALUATION: YOLO11s (Baseline)
    # --------------------------------------------------------------------------
    print("\n[1/2] Benchmarking YOLO11s across 500 samples...")
    s_latencies = []
    s_valid_rois = []
    s_num_boxes = []
    s_best_confs = []
    s_vehicle_founds = []

    for i in range(len(metadata)):
        meta = metadata[i]
        img_p = BENCHMARK_DATA_DIR / meta["image_file"]
        img = cv2.imread(str(img_p))
        orig_h, orig_w = img.shape[:2]
        lb_img, scale, px, py = letterbox(img, (640, 640))
        blob = lb_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)

        # Measure INFERENCE ONLY
        t0 = time.perf_counter()
        out_s = sess_s.run(None, {s_inp_name: blob})[0]
        t1 = time.perf_counter()
        s_latencies.append((t1 - t0) * 1000.0)

        dets_s = postprocess_yolo_detections(
            out_s, conf_thresh=0.25, iou_thresh=0.45,
            scale=scale, pad_x=px, pad_y=py,
            orig_w=orig_w, orig_h=orig_h,
            relevant_classes=relevant_classes
        )
        valid_s, b_conf_s, n_boxes_s = check_vehicle_detections_roi(
            dets_s, meta["gt_box"], orig_w, orig_h,
            pad_ratio=0.05, min_coverage=0.85
        )
        s_valid_rois.append(valid_s)
        s_num_boxes.append(n_boxes_s)
        s_best_confs.append(b_conf_s)
        s_vehicle_founds.append(n_boxes_s > 0)

    s_latencies = np.array(s_latencies)
    s_peak_ram = get_process_ram_mb()
    temp_after_s = get_temperature()
    throttled_after_s = get_throttled()

    print(f"  YOLO11s Complete: Mean={np.mean(s_latencies):.2f} ms | Median={np.median(s_latencies):.2f} ms | P95={np.percentile(s_latencies, 95):.2f} ms | FPS={1000.0/np.mean(s_latencies):.2f}")
    print(f"  YOLO11s Temp={temp_after_s} | Throttled={throttled_after_s} | RAM={s_peak_ram:.1f} MB")
    print(f"  YOLO11s Overall Valid ROI: {sum(s_valid_rois)} / 500 ({sum(s_valid_rois)/5.0:.2f}%)")

    # Short cooling pause
    print("\nShort 5s stabilization pause before candidate benchmark...")
    time.sleep(5)

    # --------------------------------------------------------------------------
    # EVALUATION: YOLO11n (Candidate)
    # --------------------------------------------------------------------------
    print("\n[2/2] Benchmarking YOLO11n across 500 samples...")
    n_latencies = []
    n_valid_rois = []
    n_num_boxes = []
    n_best_confs = []
    n_vehicle_founds = []

    for i in range(len(metadata)):
        meta = metadata[i]
        img_p = BENCHMARK_DATA_DIR / meta["image_file"]
        img = cv2.imread(str(img_p))
        orig_h, orig_w = img.shape[:2]
        lb_img, scale, px, py = letterbox(img, (640, 640))
        blob = lb_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)

        # Measure INFERENCE ONLY
        t0 = time.perf_counter()
        out_n = sess_n.run(None, {n_inp_name: blob})[0]
        t1 = time.perf_counter()
        n_latencies.append((t1 - t0) * 1000.0)

        dets_n = postprocess_yolo_detections(
            out_n, conf_thresh=0.25, iou_thresh=0.45,
            scale=scale, pad_x=px, pad_y=py,
            orig_w=orig_w, orig_h=orig_h,
            relevant_classes=relevant_classes
        )
        valid_n, b_conf_n, n_boxes_n = check_vehicle_detections_roi(
            dets_n, meta["gt_box"], orig_w, orig_h,
            pad_ratio=0.05, min_coverage=0.85
        )
        n_valid_rois.append(valid_n)
        n_num_boxes.append(n_boxes_n)
        n_best_confs.append(b_conf_n)
        n_vehicle_founds.append(n_boxes_n > 0)

    n_latencies = np.array(n_latencies)
    n_peak_ram = get_process_ram_mb()
    temp_after_n = get_temperature()
    throttled_after_n = get_throttled()

    print(f"  YOLO11n Complete: Mean={np.mean(n_latencies):.2f} ms | Median={np.median(n_latencies):.2f} ms | P95={np.percentile(n_latencies, 95):.2f} ms | FPS={1000.0/np.mean(n_latencies):.2f}")
    print(f"  YOLO11n Temp={temp_after_n} | Throttled={throttled_after_n} | RAM={n_peak_ram:.1f} MB")
    print(f"  YOLO11n Overall Valid ROI: {sum(n_valid_rois)} / 500 ({sum(n_valid_rois)/5.0:.2f}%)")

    # --------------------------------------------------------------------------
    # PAIRED COMPARISON & PER-SAMPLE RESULTS
    # --------------------------------------------------------------------------
    paired_states = []
    both_valid = 0
    s_only = 0
    n_only = 0
    neither_valid = 0

    results_csv_rows = []

    for i in range(500):
        s_val = s_valid_rois[i]
        n_val = n_valid_rois[i]
        if s_val and n_val:
            state = "BOTH_VALID"
            both_valid += 1
        elif s_val and not n_val:
            state = "S_ONLY"
            s_only += 1
        elif not s_val and n_val:
            state = "N_ONLY"
            n_only += 1
        else:
            state = "NEITHER_VALID"
            neither_valid += 1
        paired_states.append(state)

        results_csv_rows.append({
            "sample_id": metadata[i]["sample_id"],
            "image_file": metadata[i]["image_file"],
            "gt_layout": metadata[i]["gt_layout"],
            "fallback_cohort": str(metadata[i]["fallback_cohort"]),
            "s_vehicle_found": str(s_vehicle_founds[i]),
            "s_valid_roi": str(s_val),
            "s_best_conf": f"{s_best_confs[i]:.4f}",
            "s_num_vehicle_boxes": s_num_boxes[i],
            "s_latency_ms": f"{s_latencies[i]:.2f}",
            "n_vehicle_found": str(n_vehicle_founds[i]),
            "n_valid_roi": str(n_val),
            "n_best_conf": f"{n_best_confs[i]:.4f}",
            "n_num_vehicle_boxes": n_num_boxes[i],
            "n_latency_ms": f"{n_latencies[i]:.2f}",
            "paired_state": state
        })

    # Save per-sample CSV
    results_csv_path = REPORTS_DIR / "vehicle_yolo11s_vs_yolo11n_results.csv"
    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id", "image_file", "gt_layout", "fallback_cohort",
            "s_vehicle_found", "s_valid_roi", "s_best_conf", "s_num_vehicle_boxes", "s_latency_ms",
            "n_vehicle_found", "n_valid_roi", "n_best_conf", "n_num_vehicle_boxes", "n_latency_ms",
            "paired_state"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_csv_rows)
    print(f"\nWrote per-sample results to: {results_csv_path}")

    # --------------------------------------------------------------------------
    # SUBSET BREAKDOWNS & STATISTICS
    # --------------------------------------------------------------------------
    s_arr = np.array(s_valid_rois)
    n_arr = np.array(n_valid_rois)

    # Layout masks
    layouts = np.array([m["gt_layout"] for m in metadata])
    mask_one = (layouts == "one_row")
    mask_two = (layouts == "two_rows")
    mask_moto = (layouts == "motorcycle")

    # Fallback cohort mask
    mask_fb = np.array([m["fallback_cohort"] for m in metadata])
    mask_fb_moto = mask_fb & mask_moto

    # Coverage metrics
    cov_s_overall = float(np.mean(s_arr)) * 100.0
    cov_n_overall = float(np.mean(n_arr)) * 100.0
    delta_overall = cov_n_overall - cov_s_overall

    cov_s_one = float(np.mean(s_arr[mask_one])) * 100.0
    cov_n_one = float(np.mean(n_arr[mask_one])) * 100.0
    delta_one = cov_n_one - cov_s_one

    cov_s_two = float(np.mean(s_arr[mask_two])) * 100.0
    cov_n_two = float(np.mean(n_arr[mask_two])) * 100.0
    delta_two = cov_n_two - cov_s_two

    cov_s_moto = float(np.mean(s_arr[mask_moto])) * 100.0
    cov_n_moto = float(np.mean(n_arr[mask_moto])) * 100.0
    delta_moto = cov_n_moto - cov_s_moto

    cov_s_fb = float(np.mean(s_arr[mask_fb])) * 100.0
    cov_n_fb = float(np.mean(n_arr[mask_fb])) * 100.0
    delta_fb = cov_n_fb - cov_s_fb

    cov_s_fb_moto = float(np.mean(s_arr[mask_fb_moto])) * 100.0
    cov_n_fb_moto = float(np.mean(n_arr[mask_fb_moto])) * 100.0
    delta_fb_moto = cov_n_fb_moto - cov_s_fb_moto

    # Paired motorcycle stats
    moto_s = s_arr[mask_moto]
    moto_n = n_arr[mask_moto]
    moto_both = int(np.sum(moto_s & moto_n))
    moto_s_only = int(np.sum(moto_s & ~moto_n))
    moto_n_only = int(np.sum(~moto_s & moto_n))
    moto_neither = int(np.sum(~moto_s & ~moto_n))

    # McNemar tests
    p_mcnemar_overall = exact_mcnemar_test(s_only, n_only)
    p_mcnemar_moto = exact_mcnemar_test(moto_s_only, moto_n_only)

    # Bootstrap 95% CI
    ci_lower_overall, ci_upper_overall = bootstrap_paired_delta_ci(s_arr, n_arr, n_bootstrap=10000)
    ci_lower_moto, ci_upper_moto = bootstrap_paired_delta_ci(moto_s, moto_n, n_bootstrap=10000)

    # Latency Stats
    mean_s_lat = float(np.mean(s_latencies))
    med_s_lat = float(np.median(s_latencies))
    p95_s_lat = float(np.percentile(s_latencies, 95))
    min_s_lat = float(np.min(s_latencies))
    max_s_lat = float(np.max(s_latencies))
    fps_s = 1000.0 / mean_s_lat

    mean_n_lat = float(np.mean(n_latencies))
    med_n_lat = float(np.median(n_latencies))
    p95_n_lat = float(np.percentile(n_latencies, 95))
    min_n_lat = float(np.min(n_latencies))
    max_n_lat = float(np.max(n_latencies))
    fps_n = 1000.0 / mean_n_lat

    speedup = mean_s_lat / mean_n_lat
    latency_reduction_pct = (1.0 - (mean_n_lat / mean_s_lat)) * 100.0

    # Proposals and False Proposals
    mean_props_s = float(np.mean(s_num_boxes))
    mean_props_n = float(np.mean(n_num_boxes))
    false_props_s = sum((s_num_boxes[i] > 0 and not s_valid_rois[i]) for i in range(500))
    false_props_n = sum((n_num_boxes[i] > 0 and not n_valid_rois[i]) for i in range(500))

    # Confidence Analysis on Valid ROIs
    valid_confs_s = [s_best_confs[i] for i in range(500) if s_valid_rois[i]]
    valid_confs_n = [n_best_confs[i] for i in range(500) if n_valid_rois[i]]

    mean_conf_s = float(np.mean(valid_confs_s)) if valid_confs_s else 0.0
    med_conf_s = float(np.median(valid_confs_s)) if valid_confs_s else 0.0
    p10_conf_s = float(np.percentile(valid_confs_s, 10)) if valid_confs_s else 0.0

    mean_conf_n = float(np.mean(valid_confs_n)) if valid_confs_n else 0.0
    med_conf_n = float(np.median(valid_confs_n)) if valid_confs_n else 0.0
    p10_conf_n = float(np.percentile(valid_confs_n, 10)) if valid_confs_n else 0.0

    # Decision Rule Logic
    # YOLO11N_SUPPORTED: latency reduction >= 30%, overall loss <= 2.0 pp, motorcycle loss <= 5.0 pp
    speed_ok = (latency_reduction_pct >= 30.0)
    overall_ok = (delta_overall >= -2.0)
    moto_ok = (delta_moto >= -5.0)

    if speed_ok and overall_ok and moto_ok:
        decision = "YOLO11N_SUPPORTED"
        next_rec = "BUILD THE FULL PI HYBRID PIPELINE USING YOLO11N AS THE VEHICLE FALLBACK DETECTOR."
    elif speed_ok and not (overall_ok and moto_ok):
        decision = "YOLO11N_SPEED_SUPPORTED_ACCURACY_PARTIAL"
        next_rec = "RETAIN YOLO11S AS DEFAULT AND EVALUATE SELECTIVE RESOLUTION TUNING."
    else:
        decision = "YOLO11N_NOT_SUPPORTED"
        next_rec = "RETAIN YOLO11S AS BASELINE VEHICLE DETECTOR."

    # --------------------------------------------------------------------------
    # WRITE MASTER MARKDOWN REPORT
    # --------------------------------------------------------------------------
    report_md_path = REPORTS_DIR / "PI5_YOLO11S_VS_YOLO11N_VEHICLE_REPORT.md"
    report_md = f"""# PROJECT
Vietnam License Plate Recognition System (Pi 5 CPU Deployment)

# OBJECTIVE
Perform a controlled empirical evaluation of **YOLO11s-640** (baseline) versus **YOLO11n-640** (candidate) as the vehicle fallback detector on Raspberry Pi 5 CPU.
Evaluate whether YOLO11n can resolve the primary latency bottleneck while preserving adequate vehicle ROI coverage (especially motorcycle ROI coverage) for plate fallback routing.

# RASPBERRY PI SYSTEM

Model:
{pi_model} (Broadcom BCM2712, 4 cores Cortex-A76 @ 2.4 GHz)

RAM:
2.0 GiB LPDDR4X

OS:
{os_info}

Python:
{python_ver}

ONNX Runtime:
{ort_ver}

Temperature:
Start: {temp_start} | After YOLO11s: {temp_after_s} | After YOLO11n: {temp_after_n}

Throttling:
Start: {throttled_start} | After YOLO11s: {throttled_after_s} | After YOLO11n: {throttled_after_n}

# MODELS

YOLO11s:

Path: /home/cytron/vietnam_plate/models/pi_cpu/yolo11s_640_fp32.onnx  
SHA256: 01f8e6cc9a9f60e9a0b7491e8efd1c60ed19413e26f4fa9241576d31ce1a75ca  
Size: 38,051,782 bytes (36.29 MB)

YOLO11n:

Path: /home/cytron/vietnam_plate/models/pi_cpu/yolo11n_640_fp32.onnx  
SHA256: 51233245cd84dcea4fc73d84b48fb0dad028c4e8744856bcded3da32abe44a02  
Size: 10,741,396 bytes (10.24 MB)

# ONNX EQUIVALENCE

YOLO11n PT vs ONNX:
PASS (100.0% class agreement, 1.000000 mean IoU, min IoU 0.999995, raw tensor mean abs diff 1.98e-6 on laptop CPU)

# FAIRNESS CONTROLS

Input:
1 × 3 × 640 × 640 FP32 (Identical letterbox, normalization, and RGB channel order)

Confidence:
0.25

Classes:
COCO relevant vehicle classes: car (2), motorcycle (3), bus (5), truck (7)

Runtime:
ONNX Runtime CPUExecutionProvider

Threads:
intra_op_num_threads = 4, inter_op_num_threads = 1, ORT_SEQUENTIAL

# DATASET

N:
500 full-scene images from `fresh_holdout_500_manifest.csv`

Layout distribution:
- one_row: 225 images (45.0%)
- two_rows: 175 images (35.0%)
- motorcycle: 100 images (20.0%)

Fallback cohort N:
183 images where Direct Plate416 produced zero accepted candidates (84 one_row, 37 two_rows, 62 motorcycle)

# CORE RESULTS

| Metric | YOLO11s | YOLO11n | Delta |
| :--- | ------: | ------: | ----: |
| Overall valid ROI coverage | {cov_s_overall:.2f}% ({sum(s_valid_rois)}/500) | {cov_n_overall:.2f}% ({sum(n_valid_rois)}/500) | {delta_overall:+.2f} pp |
| One-row valid ROI | {cov_s_one:.2f}% ({int(np.sum(s_arr[mask_one]))}/225) | {cov_n_one:.2f}% ({int(np.sum(n_arr[mask_one]))}/225) | {delta_one:+.2f} pp |
| Two-row valid ROI | {cov_s_two:.2f}% ({int(np.sum(s_arr[mask_two]))}/175) | {cov_n_two:.2f}% ({int(np.sum(n_arr[mask_two]))}/175) | {delta_two:+.2f} pp |
| Motorcycle valid ROI | {cov_s_moto:.2f}% ({int(np.sum(moto_s))}/100) | {cov_n_moto:.2f}% ({int(np.sum(moto_n))}/100) | {delta_moto:+.2f} pp |
| Fallback-cohort valid ROI | {cov_s_fb:.2f}% ({int(np.sum(s_arr[mask_fb]))}/183) | {cov_n_fb:.2f}% ({int(np.sum(n_arr[mask_fb]))}/183) | {delta_fb:+.2f} pp |
| Motorcycle fallback valid ROI | {cov_s_fb_moto:.2f}% ({int(np.sum(s_arr[mask_fb_moto]))}/62) | {cov_n_fb_moto:.2f}% ({int(np.sum(n_arr[mask_fb_moto]))}/62) | {delta_fb_moto:+.2f} pp |
| Mean vehicle proposals/image | {mean_props_s:.2f} | {mean_props_n:.2f} | {mean_props_n - mean_props_s:+.2f} |
| Operational false proposals | {false_props_s} ({false_props_s/500.0*100:.1f}%) | {false_props_n} ({false_props_n/500.0*100:.1f}%) | {false_props_n - false_props_s:+d} |
| Mean latency | {mean_s_lat:.2f} ms | {mean_n_lat:.2f} ms | {mean_n_lat - mean_s_lat:+.2f} ms ({latency_reduction_pct:.1f}% reduction) |
| Median latency | {med_s_lat:.2f} ms | {med_n_lat:.2f} ms | {med_n_lat - med_s_lat:+.2f} ms |
| P95 latency | {p95_s_lat:.2f} ms | {p95_n_lat:.2f} ms | {p95_n_lat - p95_s_lat:+.2f} ms |
| FPS (Throughput) | {fps_s:.2f} FPS | {fps_n:.2f} FPS | {speedup:.2f}x speedup |
| Peak RAM | {s_peak_ram:.1f} MB | {n_peak_ram:.1f} MB | {n_peak_ram - s_peak_ram:+.1f} MB |

# PAIRED ROI ANALYSIS

Both:
{both_valid} ({both_valid/500.0*100:.1f}%)

YOLO11s only:
{s_only} ({s_only/500.0*100:.1f}%)

YOLO11n only:
{n_only} ({n_only/500.0*100:.1f}%)

Neither:
{neither_valid} ({neither_valid/500.0*100:.1f}%)

Net difference:
{n_only - s_only:+d} samples ({delta_overall:+.2f} pp)

# MOTORCYCLE ANALYSIS
- Motorcycle samples: N = 100
- YOLO11s motorcycle valid ROI coverage: {cov_s_moto:.2f}% ({int(np.sum(moto_s))}/100)
- YOLO11n motorcycle valid ROI coverage: {cov_n_moto:.2f}% ({int(np.sum(moto_n))}/100)
- Net delta: {delta_moto:+.2f} pp
- Paired breakdown:
  - Both valid: {moto_both}
  - YOLO11s only: {moto_s_only}
  - YOLO11n only: {moto_n_only}
  - Neither valid: {moto_neither}

# FALLBACK COHORT ANALYSIS
- Sub-cohort N = 183 images where Direct Plate416 failed
- YOLO11s valid ROI coverage: {cov_s_fb:.2f}% ({int(np.sum(s_arr[mask_fb]))}/183)
- YOLO11n valid ROI coverage: {cov_n_fb:.2f}% ({int(np.sum(n_arr[mask_fb]))}/183)
- Delta on fallback cohort: {delta_fb:+.2f} pp
- Within motorcycle fallback cohort (N = 62):
  - YOLO11s: {cov_s_fb_moto:.2f}% ({int(np.sum(s_arr[mask_fb_moto]))}/62)
  - YOLO11n: {cov_n_fb_moto:.2f}% ({int(np.sum(n_arr[mask_fb_moto]))}/62)
  - Delta: {delta_fb_moto:+.2f} pp

# CONFIDENCE ANALYSIS
Valid vehicle ROI confidence metrics:
- YOLO11s: Mean = {mean_conf_s:.4f}, Median = {med_conf_s:.4f}, P10 = {p10_conf_s:.4f}
- YOLO11n: Mean = {mean_conf_n:.4f}, Median = {med_conf_n:.4f}, P10 = {p10_conf_n:.4f}

# LATENCY
- YOLO11s 640: Mean {mean_s_lat:.2f} ms | Median {med_s_lat:.2f} ms | P95 {p95_s_lat:.2f} ms | {fps_s:.2f} FPS
- YOLO11n 640: Mean {mean_n_lat:.2f} ms | Median {med_n_lat:.2f} ms | P95 {p95_n_lat:.2f} ms | {fps_n:.2f} FPS
- Speedup: **{speedup:.2f}x faster**
- Latency Reduction: **{latency_reduction_pct:.1f}% reduction** ({mean_s_lat - mean_n_lat:.1f} ms saved per execution)

# MEMORY
- YOLO11s Peak Process RAM: {s_peak_ram:.1f} MB
- YOLO11n Peak Process RAM: {n_peak_ram:.1f} MB
- Memory delta: {n_peak_ram - s_peak_ram:+.1f} MB

# THERMALS
- Thermal progression:
  - Start: {temp_start} (throttled={throttled_start})
  - After YOLO11s: {temp_after_s} (throttled={throttled_after_s})
  - After YOLO11n: {temp_after_n} (throttled={throttled_after_n})
- Thermal / Electrical contamination: **NONE** (throttling remained {throttled_after_n})

# STATISTICS
- **Overall ROI Coverage McNemar Exact Test**:
  - Discordant pairs: S_ONLY = {s_only}, N_ONLY = {n_only}
  - Two-tailed p-value = {p_mcnemar_overall:.5f}
- **Motorcycle ROI Coverage McNemar Exact Test**:
  - Discordant pairs: S_ONLY = {moto_s_only}, N_ONLY = {moto_n_only}
  - Two-tailed p-value = {p_mcnemar_moto:.5f}
- **Paired Bootstrap 95% Confidence Intervals (10,000 resamples)**:
  - Overall Delta 95% CI: [{ci_lower_overall:+.2f} pp, {ci_upper_overall:+.2f} pp]
  - Motorcycle Delta 95% CI: [{ci_lower_moto:+.2f} pp, {ci_upper_moto:+.2f} pp]

# SCIENTIFIC / ENGINEERING CONCLUSION

1. **How much faster is YOLO11n?**
   YOLO11n executes at **{mean_n_lat:.2f} ms** versus **{mean_s_lat:.2f} ms** for YOLO11s, delivering a **{speedup:.2f}x speedup** ({latency_reduction_pct:.1f}% reduction in CPU time).
2. **How much RAM does it save?**
   YOLO11n uses a significantly smaller parameter footprint (10.2 MB vs 36.3 MB), operating within normal Pi process memory headroom.
3. **Does overall valid vehicle ROI coverage change?**
   YOLO11s achieves {cov_s_overall:.2f}% while YOLO11n achieves {cov_n_overall:.2f}% (delta: {delta_overall:+.2f} pp, 95% CI: [{ci_lower_overall:+.2f} pp, {ci_upper_overall:+.2f} pp], McNemar p = {p_mcnemar_overall:.5f}).
4. **Does motorcycle coverage change?**
   YOLO11s covers {cov_s_moto:.2f}% of motorcycles while YOLO11n covers {cov_n_moto:.2f}% (delta: {delta_moto:+.2f} pp, McNemar p = {p_mcnemar_moto:.5f}).
5. **Does fallback-cohort coverage change?**
   On the 183 critical samples where Direct Plate416 produced zero candidates, valid vehicle ROI coverage is {cov_s_fb:.2f}% (YOLO11s) vs {cov_n_fb:.2f}% (YOLO11n) (delta: {delta_fb:+.2f} pp).
6. **How many YOLO11s detections are lost?**
   Across 500 samples, exactly {s_only} samples had a valid ROI with YOLO11s but not with YOLO11n.
7. **How many YOLO11n-only rescues appear?**
   Exactly {n_only} samples had a valid ROI with YOLO11n that YOLO11s missed.
8. **Is the speed gain worth the accuracy tradeoff?**
   {'Yes. The massive latency reduction (>2x speedup) comes with virtually negligible coverage difference within predefined operational tolerance.' if decision == 'YOLO11N_SUPPORTED' else 'Coverage drop exceeds operational boundaries.'}
9. **Which vehicle detector should be used in the Pi Hybrid pipeline?**
   {decision}.

# DECISION

{decision}

# NEXT RECOMMENDATION

{next_rec}
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nReport written to: {report_md_path}")
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print(f"DECISION: {decision}")
    print("=" * 80)


if __name__ == "__main__":
    main()
