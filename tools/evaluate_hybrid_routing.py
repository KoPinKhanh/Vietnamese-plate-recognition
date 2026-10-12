"""
Controlled Experiment: Direct Plate + Hailo-Style Cascade Hybrid Routing.
Evaluates:
  - Baseline Direct (Pipeline A)
  - Baseline Cascade (Pipeline B)
  - Hybrid R1: No-Detection Fallback (HYBRID_R1_NO_DET)
  - Hybrid R2: Low-Confidence Fallback (HYBRID_R2_CONF60, threshold=0.60)
  - Hybrid R3: Parallel Union (HYBRID_R3_PARALLEL, Global NMS)
  - Oracle Upper Bound (A U B)

Evaluates on the exact 300 images in reports/e2e_real_world_manifest.csv.
"""

import argparse
import csv
import hashlib
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.pi_yolo_detector import PlateDetector
from ocr.parseq_vn_ocr import ParseqVNRecognizer
from ocr.plate_text_utils import normalize_plate_text, levenshtein_distance
from scripts.laptop_parseq_recognize import crop_with_padding


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculates Intersection-over-Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / float(union) if union > 0 else 0.0


def apply_global_nms_with_source(candidates: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
    """
    Applies Global NMS on combined candidates and determines source origin:
      - DIRECT_ONLY_SOURCE
      - CASCADE_ONLY_SOURCE
      - OVERLAP_BOTH (if an overlapping box from the other source was suppressed)
    """
    if not candidates:
        return []

    # Sort by confidence descending
    sorted_cands = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
    kept = []

    while sorted_cands:
        current = sorted_cands.pop(0)
        curr_box = current["bbox"]
        curr_source = current["source"]  # "DIRECT" or "CASCADE"
        
        has_direct = (curr_source == "DIRECT")
        has_cascade = (curr_source == "CASCADE")
        
        remaining = []
        for other in sorted_cands:
            iou = calculate_iou(curr_box, other["bbox"])
            if iou >= iou_thresh:
                # Suppressed by current box
                if other["source"] == "DIRECT":
                    has_direct = True
                elif other["source"] == "CASCADE":
                    has_cascade = True
            else:
                remaining.append(other)
        
        sorted_cands = remaining

        # Determine composite source label
        if has_direct and has_cascade:
            final_source = "OVERLAP_BOTH"
        elif has_direct:
            final_source = "DIRECT_ONLY_SOURCE"
        else:
            final_source = "CASCADE_ONLY_SOURCE"

        merged_candidate = dict(current)
        merged_candidate["source"] = final_source
        kept.append(merged_candidate)

    return kept


def draw_hybrid_comparison(
    image: np.ndarray,
    gt_bbox: List[int],
    gt_text: str,
    layout: str,
    direct_box: Optional[List[int]],
    direct_iou: float,
    cascade_box: Optional[List[int]],
    cascade_iou: float,
    hybrid_box: Optional[List[int]],
    hybrid_iou: float,
    hybrid_source: str,
    policy_name: str,
    hit: bool,
) -> np.ndarray:
    """Renders debug visualization for hybrid routing decisions."""
    canvas = image.copy()
    ih, iw = canvas.shape[:2]

    # Ground Truth: Green
    gx1, gy1, gx2, gy2 = gt_bbox
    cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
    cv2.putText(canvas, f"GT: {gt_text} ({layout})", (gx1, max(15, gy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)

    # Direct: Orange
    if direct_box is not None:
        dx1, dy1, dx2, dy2 = direct_box
        cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (0, 140, 255), 1)
        cv2.putText(canvas, f"Dir:{direct_iou:.2f}", (dx1, min(ih - 5, dy2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 140, 255), 1, cv2.LINE_AA)

    # Cascade: Yellow
    if cascade_box is not None:
        cx1, cy1, cx2, cy2 = cascade_box
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 230, 255), 1)
        cv2.putText(canvas, f"Cas:{cascade_iou:.2f}", (cx1, min(ih - 20, cy2 + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)

    # Hybrid Chosen: Magenta / Purple Thick
    if hybrid_box is not None:
        hx1, hy1, hx2, hy2 = hybrid_box
        color = (255, 0, 255) if hit else (0, 0, 255)
        cv2.rectangle(canvas, (hx1, hy1), (hx2, hy2), color, 2)
        cv2.putText(canvas, f"{policy_name} ({hybrid_source}) IoU:{hybrid_iou:.2f}", (hx1, max(30, hy1 - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

    # Top Status Bar
    header_color = (0, 200, 0) if hit else (0, 0, 220)
    status = f"Policy: {policy_name} | Hit: {hit} | Source: {hybrid_source} | GT: {gt_text}"
    cv2.rectangle(canvas, (0, 0), (iw, 26), (20, 20, 20), -1)
    cv2.putText(canvas, status, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, header_color, 2, cv2.LINE_AA)

    return canvas


def run_hybrid_experiment(
    manifest_csv: Path = PROJECT_ROOT / "reports/e2e_real_world_manifest.csv",
    output_csv: Path = PROJECT_ROOT / "reports/hybrid_routing_results.csv",
    plate_detector_path: Path = PROJECT_ROOT / "weights/best_license_plate_yolo11n_416.onnx",
    ocr_checkpoint: Path = PROJECT_ROOT / "weights/ocr/teachers/parseq_tiny_vn/best.pth",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    vehicle_conf_thresh: float = 0.25,
    vehicle_pad_ratio: float = 0.05,
    plate_conf_thresh: float = 0.50,
    plate_iou_thresh: float = 0.45,
    crop_pad: float = 0.05,
    r2_conf_gate: float = 0.60,
) -> Dict[str, Any]:
    print("=" * 80)
    print("CONTROLLED HYBRID ROUTING EXPERIMENT: DIRECT + HAILO-STYLE CASCADE")
    print(f"Manifest:               {manifest_csv}")
    print(f"Plate Detector:         {plate_detector_path} (imgsz=416, conf={plate_conf_thresh}, NMS={plate_iou_thresh})")
    print(f"Vehicle Detector:       yolo11s.pt (imgsz=640, conf={vehicle_conf_thresh}, pad={vehicle_pad_ratio*100:.1f}%)")
    print(f"OCR Recognizer:         {ocr_checkpoint} on {device}")
    print(f"R2 Confidence Gate:     {r2_conf_gate:.2f} (strictly locked)")
    print("=" * 80)

    # 1. Output Directories
    out_dir = PROJECT_ROOT / "outputs/hybrid_eval"
    r1_rescues_dir = out_dir / "r1_rescues"
    r2_rescues_dir = out_dir / "r2_rescues"
    r3_rescues_dir = out_dir / "r3_rescues"
    r1_regressions_dir = out_dir / "r1_regressions"
    r2_regressions_dir = out_dir / "r2_regressions"
    r3_regressions_dir = out_dir / "r3_regressions"

    for d in (r1_rescues_dir, r2_rescues_dir, r3_rescues_dir,
              r1_regressions_dir, r2_regressions_dir, r3_regressions_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 2. Model Initialization strictly once
    print("\n[Init] Loading Ultralytics YOLO11s Vehicle Detector...")
    vehicle_model = YOLO("yolo11s.pt")
    vehicle_classes = [cls_id for cls_id, name in vehicle_model.names.items() if name in ("car", "motorcycle", "bus", "truck")]

    print("\n[Init] Loading YOLO11n License Plate Detector (OpenCV DNN 416)...")
    plate_detector = PlateDetector(model_path=str(plate_detector_path), imgsz=416, conf=plate_conf_thresh, iou=plate_iou_thresh)

    print("\n[Init] Loading PARSeq-Tiny VNLP OCR...")
    ocr_recognizer = ParseqVNRecognizer(checkpoint=str(ocr_checkpoint), device=device)

    # 3. Standardized Warmup (Section 30)
    print("\n[Warmup] Warming up models for standardized latency measurement...")
    dummy_img_640 = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_img_416 = np.zeros((416, 416, 3), dtype=np.uint8)
    dummy_crop = np.zeros((48, 160, 3), dtype=np.uint8)

    for _ in range(15):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        vehicle_model.predict(dummy_img_640, imgsz=640, conf=0.25, classes=vehicle_classes, device=device, verbose=False)
        plate_detector.detect(dummy_img_416)
        ocr_recognizer.recognize(dummy_crop, layout="auto")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    print("[Warmup] Complete. Latency benchmarking active.\n")

    # 4. Load Manifest
    samples = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            row["sample_id"] = f"SMP_{idx:04d}"
            samples.append(row)

    total_samples = len(samples)
    print(f"Benchmarking all policies across {total_samples} images...")

    results = []

    # Timing accumulators (per frame ms)
    timings = {
        "direct": {"det": [], "ocr": [], "total": []},
        "cascade": {"veh": [], "nested_plate": [], "ocr": [], "total": []},
        "r1": {"det": [], "ocr": [], "total": []},
        "r2": {"det": [], "ocr": [], "total": []},
        "r3": {"det": [], "ocr": [], "total": []},
    }

    # Tracking invocations
    r1_cascade_invocations = 0
    r2_cascade_invocations = 0
    r2_trigger_counts = Counter()

    # Confidence bin analysis accumulators for Direct
    # bins: 0.50-0.60, 0.60-0.70, 0.70-0.80, 0.80-0.90, 0.90-1.00
    conf_bins = {
        "0.50-0.60": {"count": 0, "matched": 0, "ocr_exact": 0, "cascade_rescued": 0},
        "0.60-0.70": {"count": 0, "matched": 0, "ocr_exact": 0, "cascade_rescued": 0},
        "0.70-0.80": {"count": 0, "matched": 0, "ocr_exact": 0, "cascade_rescued": 0},
        "0.80-0.90": {"count": 0, "matched": 0, "ocr_exact": 0, "cascade_rescued": 0},
        "0.90-1.00": {"count": 0, "matched": 0, "ocr_exact": 0, "cascade_rescued": 0},
    }

    t_start_all = time.perf_counter()

    for idx, s in enumerate(samples, 1):
        sample_id = s["sample_id"]
        img_path = Path(s["image_path"])
        gt_text = normalize_plate_text(s["ground_truth"])
        gt_layout = s["gt_layout"]
        gt_bbox = [int(s["gt_x1"]), int(s["gt_y1"]), int(s["gt_x2"]), int(s["gt_y2"])]

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        # Cache crops and OCR across identical bboxes to ensure fairness and speed
        ocr_cache = {}

        def get_ocr_result(box: List[int]) -> Tuple[str, bool, float]:
            key = tuple(box)
            if key in ocr_cache:
                return ocr_cache[key]
            crop, _ = crop_with_padding(img, box, pad_ratio=crop_pad)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = ocr_recognizer.recognize(crop, layout="auto")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_ms = (time.perf_counter() - t0) * 1000.0
            norm_pred = normalize_plate_text(res["text"])
            exact = (norm_pred == gt_text)
            ocr_cache[key] = (norm_pred, exact, t_ms)
            return ocr_cache[key]

        # -------------------------------------------------------------
        # 1. DIRECT PLATE DETECTION (Full Frame)
        # -------------------------------------------------------------
        t_d0 = time.perf_counter()
        raw_direct_dets, _ = plate_detector.detect(img)
        t_d1 = time.perf_counter()
        direct_det_ms = (t_d1 - t_d0) * 1000.0

        direct_candidates = []
        for d in raw_direct_dets:
            direct_candidates.append({
                "bbox": d["bbox"],
                "confidence": d["confidence"],
                "source": "DIRECT",
            })

        # Match Direct
        best_d_cand = None
        best_d_iou = 0.0
        for cand in direct_candidates:
            iou = calculate_iou(cand["bbox"], gt_bbox)
            if iou > best_d_iou:
                best_d_iou = iou
                best_d_cand = cand

        direct_detected = (best_d_cand is not None and best_d_iou >= 0.50)
        direct_conf = best_d_cand["confidence"] if best_d_cand else 0.0
        direct_ocr = ""
        direct_e2e = False
        direct_ocr_ms = 0.0
        if direct_detected and best_d_cand is not None:
            direct_ocr, direct_e2e, direct_ocr_ms = get_ocr_result(best_d_cand["bbox"])

        direct_total_ms = direct_det_ms + direct_ocr_ms
        timings["direct"]["det"].append(direct_det_ms)
        timings["direct"]["ocr"].append(direct_ocr_ms)
        timings["direct"]["total"].append(direct_total_ms)

        # -------------------------------------------------------------
        # 2. CASCADE DETECTION (Vehicle -> Plate)
        # -------------------------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_v0 = time.perf_counter()
        v_preds = vehicle_model.predict(
            img,
            imgsz=640,
            conf=vehicle_conf_thresh,
            classes=vehicle_classes,
            device=device,
            verbose=False,
        )[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_v1 = time.perf_counter()
        veh_ms = (t_v1 - t_v0) * 1000.0

        nested_plate_ms = 0.0
        raw_cascade_dets = []

        for b in v_preds.boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            v_conf = float(b.conf[0].cpu().item())
            cls_id = int(b.cls[0].cpu().item())
            cls_name = vehicle_model.names[cls_id]
            vx1, vy1, vx2, vy2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            vw = vx2 - vx1
            vh = vy2 - vy1
            pad_w = int(round(vw * vehicle_pad_ratio))
            pad_h = int(round(vh * vehicle_pad_ratio))
            rx1 = max(0, vx1 - pad_w)
            ry1 = max(0, vy1 - pad_h)
            rx2 = min(iw, vx2 + pad_w)
            ry2 = min(ih, vy2 + pad_h)

            roi = img[ry1:ry2, rx1:rx2]
            if roi.size == 0 or roi.shape[0] < 12 or roi.shape[1] < 18:
                continue

            t_np0 = time.perf_counter()
            roi_dets, _ = plate_detector.detect(roi)
            nested_plate_ms += (time.perf_counter() - t_np0) * 1000.0

            for p in roi_dets:
                px1, py1, px2, py2 = p["bbox"]
                fx1 = max(0, min(iw, rx1 + px1))
                fy1 = max(0, min(ih, ry1 + py1))
                fx2 = max(0, min(iw, rx1 + px2))
                fy2 = max(0, min(ih, ry1 + py2))
                raw_cascade_dets.append({
                    "bbox": [fx1, fy1, fx2, fy2],
                    "confidence": p["confidence"],
                    "vehicle_class": cls_name,
                    "vehicle_conf": v_conf,
                    "source": "CASCADE",
                })

        # Global NMS across vehicle ROIs
        cascade_candidates = apply_global_nms_with_source(raw_cascade_dets, iou_thresh=plate_iou_thresh)
        # Restore source tag to CASCADE
        for c in cascade_candidates:
            c["source"] = "CASCADE"

        # Match Cascade
        best_c_cand = None
        best_c_iou = 0.0
        for cand in cascade_candidates:
            iou = calculate_iou(cand["bbox"], gt_bbox)
            if iou > best_c_iou:
                best_c_iou = iou
                best_c_cand = cand

        cascade_detected = (best_c_cand is not None and best_c_iou >= 0.50)
        cascade_conf = best_c_cand["confidence"] if best_c_cand else 0.0
        cascade_ocr = ""
        cascade_e2e = False
        cascade_ocr_ms = 0.0
        if cascade_detected and best_c_cand is not None:
            cascade_ocr, cascade_e2e, cascade_ocr_ms = get_ocr_result(best_c_cand["bbox"])

        cascade_total_ms = veh_ms + nested_plate_ms + cascade_ocr_ms
        timings["cascade"]["veh"].append(veh_ms)
        timings["cascade"]["nested_plate"].append(nested_plate_ms)
        timings["cascade"]["ocr"].append(cascade_ocr_ms)
        timings["cascade"]["total"].append(cascade_total_ms)

        # -------------------------------------------------------------
        # 3. HYBRID POLICY R1: NO-DETECTION FALLBACK (HYBRID_R1_NO_DET)
        # -------------------------------------------------------------
        # If Direct has >= 1 detection, use Direct. Else fallback to Cascade.
        if len(direct_candidates) > 0:
            r1_invoked_cascade = False
            r1_candidates = direct_candidates
            r1_compute_ms = direct_det_ms
        else:
            r1_invoked_cascade = True
            r1_cascade_invocations += 1
            r1_candidates = cascade_candidates
            r1_compute_ms = direct_det_ms + veh_ms + nested_plate_ms

        best_r1_cand = None
        best_r1_iou = 0.0
        for cand in r1_candidates:
            iou = calculate_iou(cand["bbox"], gt_bbox)
            if iou > best_r1_iou:
                best_r1_iou = iou
                best_r1_cand = cand

        r1_detected = (best_r1_cand is not None and best_r1_iou >= 0.50)
        r1_source = best_r1_cand["source"] if best_r1_cand else "NONE"
        r1_iou = best_r1_iou
        r1_ocr = ""
        r1_e2e = False
        r1_ocr_ms = 0.0
        if r1_detected and best_r1_cand is not None:
            r1_ocr, r1_e2e, r1_ocr_ms = get_ocr_result(best_r1_cand["bbox"])

        r1_total_ms = r1_compute_ms + r1_ocr_ms
        timings["r1"]["det"].append(r1_compute_ms)
        timings["r1"]["ocr"].append(r1_ocr_ms)
        timings["r1"]["total"].append(r1_total_ms)

        # -------------------------------------------------------------
        # 4. HYBRID POLICY R2: LOW-CONFIDENCE FALLBACK (HYBRID_R2_CONF60)
        # -------------------------------------------------------------
        # Trigger cascade if no detection OR best direct conf < 0.60
        r2_trigger = "NOT_INVOKED"
        r2_candidates = []
        r2_compute_ms = direct_det_ms

        if len(direct_candidates) == 0:
            r2_trigger = "NO_DIRECT_DETECTION"
            r2_cascade_invocations += 1
            r2_candidates = cascade_candidates
            r2_compute_ms = direct_det_ms + veh_ms + nested_plate_ms
        else:
            best_dir_conf = max(c["confidence"] for c in direct_candidates)
            if best_dir_conf >= r2_conf_gate:
                r2_trigger = "NOT_INVOKED"
                r2_candidates = direct_candidates
                r2_compute_ms = direct_det_ms
            else:
                r2_trigger = "LOW_DIRECT_CONFIDENCE"
                r2_cascade_invocations += 1
                # Merge Direct and Cascade candidates with NMS
                t_nms0 = time.perf_counter()
                merged_pool = list(direct_candidates) + list(cascade_candidates)
                r2_candidates = apply_global_nms_with_source(merged_pool, iou_thresh=plate_iou_thresh)
                nms_time = (time.perf_counter() - t_nms0) * 1000.0
                r2_compute_ms = direct_det_ms + veh_ms + nested_plate_ms + nms_time

        r2_trigger_counts[r2_trigger] += 1

        best_r2_cand = None
        best_r2_iou = 0.0
        for cand in r2_candidates:
            iou = calculate_iou(cand["bbox"], gt_bbox)
            if iou > best_r2_iou:
                best_r2_iou = iou
                best_r2_cand = cand

        r2_detected = (best_r2_cand is not None and best_r2_iou >= 0.50)
        # If merged OVERLAP_BOTH, record the primary or composite source
        r2_source = best_r2_cand["source"] if best_r2_cand else "NONE"
        r2_iou = best_r2_iou
        r2_ocr = ""
        r2_e2e = False
        r2_ocr_ms = 0.0
        if r2_detected and best_r2_cand is not None:
            r2_ocr, r2_e2e, r2_ocr_ms = get_ocr_result(best_r2_cand["bbox"])

        r2_total_ms = r2_compute_ms + r2_ocr_ms
        timings["r2"]["det"].append(r2_compute_ms)
        timings["r2"]["ocr"].append(r2_ocr_ms)
        timings["r2"]["total"].append(r2_total_ms)

        # -------------------------------------------------------------
        # 5. HYBRID POLICY R3: PARALLEL UNION (HYBRID_R3_PARALLEL)
        # -------------------------------------------------------------
        # Always run both Direct and Cascade, merge all candidates with Global NMS
        t_r3_nms0 = time.perf_counter()
        r3_pool = list(direct_candidates) + list(cascade_candidates)
        r3_candidates = apply_global_nms_with_source(r3_pool, iou_thresh=plate_iou_thresh)
        r3_nms_ms = (time.perf_counter() - t_r3_nms0) * 1000.0

        best_r3_cand = None
        best_r3_iou = 0.0
        for cand in r3_candidates:
            iou = calculate_iou(cand["bbox"], gt_bbox)
            if iou > best_r3_iou:
                best_r3_iou = iou
                best_r3_cand = cand

        r3_detected = (best_r3_cand is not None and best_r3_iou >= 0.50)
        r3_source = best_r3_cand["source"] if best_r3_cand else "NONE"
        r3_iou = best_r3_iou
        r3_ocr = ""
        r3_e2e = False
        r3_ocr_ms = 0.0
        if r3_detected and best_r3_cand is not None:
            r3_ocr, r3_e2e, r3_ocr_ms = get_ocr_result(best_r3_cand["bbox"])

        r3_total_ms = direct_det_ms + veh_ms + nested_plate_ms + r3_nms_ms + r3_ocr_ms
        timings["r3"]["det"].append(direct_det_ms + veh_ms + nested_plate_ms + r3_nms_ms)
        timings["r3"]["ocr"].append(r3_ocr_ms)
        timings["r3"]["total"].append(r3_total_ms)

        # -------------------------------------------------------------
        # 6. CONFIDENCE ANALYSIS FOR DIRECT (Diagnostic Section 28)
        # -------------------------------------------------------------
        if best_d_cand is not None:
            c = best_d_cand["confidence"]
            bin_key = None
            if 0.50 <= c < 0.60:
                bin_key = "0.50-0.60"
            elif 0.60 <= c < 0.70:
                bin_key = "0.60-0.70"
            elif 0.70 <= c < 0.80:
                bin_key = "0.70-0.80"
            elif 0.80 <= c < 0.90:
                bin_key = "0.80-0.90"
            elif 0.90 <= c <= 1.00:
                bin_key = "0.90-1.00"

            if bin_key:
                conf_bins[bin_key]["count"] += 1
                if direct_detected:
                    conf_bins[bin_key]["matched"] += 1
                if direct_e2e:
                    conf_bins[bin_key]["ocr_exact"] += 1
                if (not direct_detected) and cascade_detected:
                    conf_bins[bin_key]["cascade_rescued"] += 1

        # -------------------------------------------------------------
        # 7. RECORD SAMPLE DATA
        # -------------------------------------------------------------
        row_record = {
            "sample_id": sample_id,
            "image_path": s["image_path"],
            "ground_truth": gt_text,
            "layout": gt_layout,
            "gt_bbox": str(gt_bbox),
            "direct_detected": direct_detected,
            "direct_conf": round(direct_conf, 4),
            "direct_iou": round(best_d_iou, 4),
            "direct_e2e": direct_e2e,
            "direct_candidates_count": len(direct_candidates),
            "cascade_detected": cascade_detected,
            "cascade_conf": round(cascade_conf, 4),
            "cascade_iou": round(best_c_iou, 4),
            "cascade_e2e": cascade_e2e,
            "cascade_candidates_count": len(cascade_candidates),
            "r1_detected": r1_detected,
            "r1_source": r1_source,
            "r1_iou": round(r1_iou, 4),
            "r1_ocr": r1_ocr,
            "r1_e2e": r1_e2e,
            "r1_candidates_count": len(r1_candidates),
            "r2_detected": r2_detected,
            "r2_source": r2_source,
            "r2_iou": round(r2_iou, 4),
            "r2_ocr": r2_ocr,
            "r2_e2e": r2_e2e,
            "r2_trigger": r2_trigger,
            "r2_candidates_count": len(r2_candidates),
            "r3_detected": r3_detected,
            "r3_source": r3_source,
            "r3_iou": round(r3_iou, 4),
            "r3_ocr": r3_ocr,
            "r3_e2e": r3_e2e,
            "r3_candidates_count": len(r3_candidates),
        }
        results.append(row_record)

        # -------------------------------------------------------------
        # 8. VISUAL RESCUES & REGRESSIONS SAVING (Sections 38, 39)
        # -------------------------------------------------------------
        # Helper to draw and save
        d_box = best_d_cand["bbox"] if best_d_cand else None
        c_box = best_c_cand["bbox"] if best_c_cand else None

        # R1 Rescues / Regressions
        if not direct_detected and r1_detected:
            vis = draw_hybrid_comparison(img, gt_bbox, gt_text, gt_layout, d_box, best_d_iou, c_box, best_c_iou,
                                         best_r1_cand["bbox"], r1_iou, r1_source, "HYBRID_R1", True)
            cv2.imwrite(str(r1_rescues_dir / f"r1_rescue_{sample_id}_{gt_layout}_{gt_text}.jpg"), vis)
        elif direct_detected and not r1_detected:
            vis = draw_hybrid_comparison(img, gt_bbox, gt_text, gt_layout, d_box, best_d_iou, c_box, best_c_iou,
                                         None, 0.0, "NONE", "HYBRID_R1", False)
            cv2.imwrite(str(r1_regressions_dir / f"r1_regression_{sample_id}_{gt_layout}_{gt_text}.jpg"), vis)

        # R2 Rescues / Regressions
        if not direct_detected and r2_detected:
            vis = draw_hybrid_comparison(img, gt_bbox, gt_text, gt_layout, d_box, best_d_iou, c_box, best_c_iou,
                                         best_r2_cand["bbox"], r2_iou, r2_source, "HYBRID_R2", True)
            cv2.imwrite(str(r2_rescues_dir / f"r2_rescue_{sample_id}_{gt_layout}_{gt_text}.jpg"), vis)
        elif direct_detected and not r2_detected:
            vis = draw_hybrid_comparison(img, gt_bbox, gt_text, gt_layout, d_box, best_d_iou, c_box, best_c_iou,
                                         best_r2_cand["bbox"] if best_r2_cand else None, r2_iou, r2_source, "HYBRID_R2", False)
            cv2.imwrite(str(r2_regressions_dir / f"r2_regression_{sample_id}_{gt_layout}_{gt_text}.jpg"), vis)

        # R3 Rescues / Regressions
        if not direct_detected and r3_detected:
            vis = draw_hybrid_comparison(img, gt_bbox, gt_text, gt_layout, d_box, best_d_iou, c_box, best_c_iou,
                                         best_r3_cand["bbox"], r3_iou, r3_source, "HYBRID_R3", True)
            cv2.imwrite(str(r3_rescues_dir / f"r3_rescue_{sample_id}_{gt_layout}_{gt_text}.jpg"), vis)
        elif direct_detected and not r3_detected:
            vis = draw_hybrid_comparison(img, gt_bbox, gt_text, gt_layout, d_box, best_d_iou, c_box, best_c_iou,
                                         best_r3_cand["bbox"] if best_r3_cand else None, r3_iou, r3_source, "HYBRID_R3", False)
            cv2.imwrite(str(r3_regressions_dir / f"r3_regression_{sample_id}_{gt_layout}_{gt_text}.jpg"), vis)

        if idx % 50 == 0 or idx == total_samples:
            elapsed = time.perf_counter() - t_start_all
            print(f"  Processed [{idx:04d}/{total_samples}] samples ({idx/elapsed:.1f} FPS) | "
                  f"Direct: {sum(1 for r in results if r['direct_detected'])} | "
                  f"R1: {sum(1 for r in results if r['r1_detected'])} | "
                  f"R2: {sum(1 for r in results if r['r2_detected'])} | "
                  f"R3: {sum(1 for r in results if r['r3_detected'])}")

    # 9. Save CSV Results (Section 41)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Done] Saved full hybrid evaluation results to: {output_csv}")

    return {
        "results": results,
        "timings": timings,
        "r1_cascade_invocations": r1_cascade_invocations,
        "r2_cascade_invocations": r2_cascade_invocations,
        "r2_trigger_counts": r2_trigger_counts,
        "conf_bins": conf_bins,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Routing Policies")
    parser.add_argument("--manifest", type=str, default="reports/e2e_real_world_manifest.csv")
    parser.add_argument("--output", type=str, default="reports/hybrid_routing_results.csv")
    args = parser.parse_args()

    run_hybrid_experiment(
        manifest_csv=Path(args.manifest),
        output_csv=Path(args.output),
    )
