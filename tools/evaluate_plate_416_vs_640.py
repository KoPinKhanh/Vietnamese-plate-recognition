"""
Controlled Plate Detector Resolution Experiment: 416 vs 640 with Locked Hybrid R1 Architecture.
Evaluates System 416 vs System 640 (+ Diagnostic Mixed Arm) on fresh_plate_resolution_500_manifest.csv (N=500).
Models:
- Plate Detector 416: weights/best_license_plate_yolo11n_416.onnx (OpenCV DNN CPU)
- Plate Detector 640: weights/best_license_plate_yolo11n_640.onnx (OpenCV DNN CPU)
- Vehicle Detector: yolo11s.pt (Ultralytics GPU/CPU)
- OCR: weights/ocr/teachers/parseq_tiny_vn/best.pth (PyTorch PARSeq-Tiny VNLP)
"""

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.pi_yolo_detector import PlateDetector
from ocr.parseq_vn_ocr import ParseqVNRecognizer
from ocr.plate_text_utils import normalize_plate_text, levenshtein_distance
from scripts.laptop_parseq_recognize import crop_with_padding


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculates IoU between two [x1, y1, x2, y2] bounding boxes."""
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


def does_roi_cover_box(roi: List[int], box: List[int], min_coverage: float = 0.85) -> bool:
    """Checks whether roi covers at least min_coverage of box area."""
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


def apply_global_plate_nms(detections: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
    """Applies NMS over candidate plate bounding boxes."""
    if not detections:
        return []

    boxes = []
    scores = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(d["confidence"])

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.50, nms_threshold=iou_thresh)
    if len(indices) == 0:
        return []

    kept = []
    for idx in np.array(indices).flatten():
        kept.append(detections[idx])
    return kept


def run_controlled_experiment(
    manifest_path: Path = PROJECT_ROOT / "reports/fresh_plate_resolution_500_manifest.csv",
    output_csv_path: Path = PROJECT_ROOT / "reports/plate416_vs_640_results.csv",
    vehicle_model_path: Path = PROJECT_ROOT / "yolo11s.pt",
    ocr_model_path: Path = PROJECT_ROOT / "weights/ocr/teachers/parseq_tiny_vn/best.pth",
):
    print("=" * 80)
    print("CONTROLLED PLATE DETECTOR RESOLUTION EXPERIMENT: 416 vs 640")
    print("=" * 80)

    # 1. Initialize Detectors with OpenCV DNN CPU
    print("[1/4] Initializing Plate Detectors (OpenCV DNN CPU)...")
    plate_det_416 = PlateDetector(
        str(PROJECT_ROOT / "weights/best_license_plate_yolo11n_416.onnx"),
        imgsz=416,
        conf=0.50,
        iou=0.45,
    )
    plate_det_640 = PlateDetector(
        str(PROJECT_ROOT / "weights/best_license_plate_yolo11n_640.onnx"),
        imgsz=640,
        conf=0.50,
        iou=0.45,
    )

    # 2. Initialize Vehicle Detector
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[2/4] Initializing YOLO11s Vehicle Detector on {device}...")
    vehicle_model = YOLO(str(vehicle_model_path))
    vehicle_classes = [cls_id for cls_id, name in vehicle_model.names.items() if name in ("car", "motorcycle", "bus", "truck")]
    print(f"      Vehicle tracking target classes: {vehicle_classes}")

    # 3. Initialize PARSeq-Tiny VNLP OCR
    print(f"[3/4] Initializing PARSeq-Tiny VNLP OCR on {device}...")
    ocr_recognizer = ParseqVNRecognizer(
        checkpoint=str(ocr_model_path),
        device=device,
    )

    # 4. Load Manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))

    print(f"[4/4] Loaded {len(samples)} frozen test samples from {manifest_path.name}")

    # Warmup
    print("Warming up models...")
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = plate_det_416.detect(dummy_bgr)
        _ = plate_det_640.detect(dummy_bgr)
        _ = vehicle_model(dummy_bgr, imgsz=640, conf=0.25, classes=vehicle_classes, verbose=False)
        _ = ocr_recognizer.recognize(dummy_bgr)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    process = psutil.Process()
    results = []

    print("\nStarting inference across 500 samples...")
    t_start = time.perf_counter()

    for idx, s in enumerate(samples, 1):
        sample_id = s["sample_id"]
        img_p = Path(s["image_path"])
        gt_text = s["ground_truth"].strip().upper()
        gt_layout = s["gt_layout"]

        # Parse gt_bbox
        cleaned_bbox = s["gt_bbox"].strip().strip("[]()").replace(" ", "")
        gt_box = [int(float(v)) for v in cleaned_bbox.split(",")]
        gt_area = max(1, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))

        frame = cv2.imread(str(img_p))
        if frame is None:
            print(f"ERROR: Cannot read {img_p}")
            continue
        h, w = frame.shape[:2]

        # -------------------------------------------------------------
        # COMPONENT-LEVEL TIMING AND EXECUTION
        # -------------------------------------------------------------
        # Direct 416
        t0_d416 = time.perf_counter()
        d416_cands, _ = plate_det_416.detect(frame)
        t_d416_ms = (time.perf_counter() - t0_d416) * 1000.0
        d416_cnt = len(d416_cands)

        # Direct 640
        t0_d640 = time.perf_counter()
        d640_cands, _ = plate_det_640.detect(frame)
        t_d640_ms = (time.perf_counter() - t0_d640) * 1000.0
        d640_cnt = len(d640_cands)

        # Match Direct 416
        direct416_detected = False
        direct416_conf = 0.0
        direct416_iou = 0.0
        direct416_box = []
        if d416_cnt > 0:
            best_d416 = max(d416_cands, key=lambda c: c["confidence"])
            direct416_box = best_d416["bbox"]
            direct416_conf = best_d416["confidence"]
            direct416_iou = calculate_iou(direct416_box, gt_box)
            if direct416_iou >= 0.50:
                direct416_detected = True

        # Match Direct 640
        direct640_detected = False
        direct640_conf = 0.0
        direct640_iou = 0.0
        direct640_box = []
        if d640_cnt > 0:
            best_d640 = max(d640_cands, key=lambda c: c["confidence"])
            direct640_box = best_d640["bbox"]
            direct640_conf = best_d640["confidence"]
            direct640_iou = calculate_iou(direct640_box, gt_box)
            if direct640_iou >= 0.50:
                direct640_detected = True

        # Vehicle Detection (if either 416 or 640 invokes fallback)
        cascade416_invoked = (d416_cnt == 0)
        cascade640_invoked = (d640_cnt == 0)

        t_veh_ms = 0.0
        vehicle_found = False
        vehicle_roi_covers_gt = False
        v_boxes = []

        if cascade416_invoked or cascade640_invoked:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0_veh = time.perf_counter()
            v_results = vehicle_model(frame, imgsz=640, conf=0.25, classes=vehicle_classes, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_veh_ms = (time.perf_counter() - t0_veh) * 1000.0

            if len(v_results) > 0 and v_results[0].boxes is not None and len(v_results[0].boxes) > 0:
                vehicle_found = True
                for b in v_results[0].boxes:
                    coords = b.xyxy[0].cpu().numpy()
                    vx1, vy1, vx2, vy2 = [int(round(x)) for x in coords]
                    vw, vh = vx2 - vx1, vy2 - vy1
                    pad_w, pad_h = int(round(vw * 0.05)), int(round(vh * 0.05))
                    vx1_pad = max(0, vx1 - pad_w)
                    vy1_pad = max(0, vy1 - pad_h)
                    vx2_pad = min(w, vx2 + pad_w)
                    vy2_pad = min(h, vy2 + pad_h)
                    v_boxes.append([vx1_pad, vy1_pad, vx2_pad, vy2_pad])
                    if does_roi_cover_box([vx1_pad, vy1_pad, vx2_pad, vy2_pad], gt_box):
                        vehicle_roi_covers_gt = True

        # System 416 Nested Evaluation
        t_nested416_ms = 0.0
        r1_416_source = "NONE"
        r1_416_box = []
        r1_416_conf = 0.0
        r1_416_iou = 0.0
        r1_416_detected = False
        nested416_detected = False
        valid_roi_plate416_miss = False

        if not cascade416_invoked:
            r1_416_source = "DIRECT"
            r1_416_box = direct416_box
            r1_416_conf = direct416_conf
            r1_416_iou = direct416_iou
            r1_416_detected = direct416_detected
        else:
            if vehicle_found:
                t0_n416 = time.perf_counter()
                nested_cands_416 = []
                for v_box in v_boxes:
                    vx1, vy1, vx2, vy2 = v_box
                    if vx2 > vx1 and vy2 > vy1:
                        v_roi = frame[vy1:vy2, vx1:vx2]
                        if v_roi.size > 0:
                            p_dets, _ = plate_det_416.detect(v_roi)
                            for pd in p_dets:
                                px1, py1, px2, py2 = pd["bbox"]
                                nested_cands_416.append({
                                    "bbox": [px1 + vx1, py1 + vy1, px2 + vx1, py2 + vy1],
                                    "confidence": pd["confidence"],
                                })
                filtered_416 = apply_global_plate_nms(nested_cands_416, iou_thresh=0.45)
                t_nested416_ms = (time.perf_counter() - t0_n416) * 1000.0

                if filtered_416:
                    best_n416 = max(filtered_416, key=lambda c: c["confidence"])
                    r1_416_source = "CASCADE"
                    r1_416_box = best_n416["bbox"]
                    r1_416_conf = best_n416["confidence"]
                    r1_416_iou = calculate_iou(r1_416_box, gt_box)
                    if r1_416_iou >= 0.50:
                        r1_416_detected = True
                        nested416_detected = True

                if vehicle_roi_covers_gt and not nested416_detected:
                    valid_roi_plate416_miss = True

        # System 640 Nested Evaluation
        t_nested640_ms = 0.0
        r1_640_source = "NONE"
        r1_640_box = []
        r1_640_conf = 0.0
        r1_640_iou = 0.0
        r1_640_detected = False
        nested640_detected = False

        if not cascade640_invoked:
            r1_640_source = "DIRECT"
            r1_640_box = direct640_box
            r1_640_conf = direct640_conf
            r1_640_iou = direct640_iou
            r1_640_detected = direct640_detected
        else:
            if vehicle_found:
                t0_n640 = time.perf_counter()
                nested_cands_640 = []
                for v_box in v_boxes:
                    vx1, vy1, vx2, vy2 = v_box
                    if vx2 > vx1 and vy2 > vy1:
                        v_roi = frame[vy1:vy2, vx1:vx2]
                        if v_roi.size > 0:
                            p_dets, _ = plate_det_640.detect(v_roi)
                            for pd in p_dets:
                                px1, py1, px2, py2 = pd["bbox"]
                                nested_cands_640.append({
                                    "bbox": [px1 + vx1, py1 + vy1, px2 + vx1, py2 + vy1],
                                    "confidence": pd["confidence"],
                                })
                filtered_640 = apply_global_plate_nms(nested_cands_640, iou_thresh=0.45)
                t_nested640_ms = (time.perf_counter() - t0_n640) * 1000.0

                if filtered_640:
                    best_n640 = max(filtered_640, key=lambda c: c["confidence"])
                    r1_640_source = "CASCADE"
                    r1_640_box = best_n640["bbox"]
                    r1_640_conf = best_n640["confidence"]
                    r1_640_iou = calculate_iou(r1_640_box, gt_box)
                    if r1_640_iou >= 0.50:
                        r1_640_detected = True
                        nested640_detected = True

        # OCR 416
        t_ocr416_ms = 0.0
        r1_416_ocr = ""
        r1_416_ocr_conf = 0.0
        r1_416_e2e = False
        if r1_416_box:
            crop_416, _ = crop_with_padding(frame, r1_416_box, pad_ratio=0.05)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0_ocr = time.perf_counter()
            ocr_res_416 = ocr_recognizer.recognize(crop_416)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_ocr416_ms = (time.perf_counter() - t0_ocr) * 1000.0
            r1_416_ocr = normalize_plate_text(ocr_res_416["text"])
            r1_416_ocr_conf = ocr_res_416["confidence"]
            if r1_416_detected and r1_416_ocr == normalize_plate_text(gt_text):
                r1_416_e2e = True

        # OCR 640
        t_ocr640_ms = 0.0
        r1_640_ocr = ""
        r1_640_ocr_conf = 0.0
        r1_640_e2e = False
        if r1_640_box:
            crop_640, _ = crop_with_padding(frame, r1_640_box, pad_ratio=0.05)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0_ocr = time.perf_counter()
            ocr_res_640 = ocr_recognizer.recognize(crop_640)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_ocr640_ms = (time.perf_counter() - t0_ocr) * 1000.0
            r1_640_ocr = normalize_plate_text(ocr_res_640["text"])
            r1_640_ocr_conf = ocr_res_640["confidence"]
            if r1_640_detected and r1_640_ocr == normalize_plate_text(gt_text):
                r1_640_e2e = True

        # Total Latency Calculations
        latency416 = t_d416_ms + (t_veh_ms + t_nested416_ms if cascade416_invoked else 0.0) + t_ocr416_ms
        latency640 = t_d640_ms + (t_veh_ms + t_nested640_ms if cascade640_invoked else 0.0) + t_ocr640_ms

        # -------------------------------------------------------------
        # OPTIONAL DIAGNOSTIC THIRD ARM: SYSTEM MIXED
        # Direct 416 + Nested 640
        # -------------------------------------------------------------
        mixed_detected = False
        mixed_e2e = False
        mixed_source = "NONE"
        if d416_cnt > 0:
            mixed_detected = direct416_detected
            mixed_e2e = r1_416_e2e
            mixed_source = "DIRECT_416"
        else:
            if nested640_detected:
                mixed_detected = True
                mixed_source = "CASCADE_640"
                mixed_e2e = r1_640_e2e

        # Attribution classification
        failure_stage_416 = "NONE"
        if not r1_416_detected:
            if d416_cnt > 0:
                failure_stage_416 = "DIRECT_POOR_LOCALIZATION_OR_FP"
            elif not vehicle_found:
                failure_stage_416 = "VEHICLE_STAGE_MISS"
            elif not vehicle_roi_covers_gt:
                failure_stage_416 = "VEHICLE_ROI_MISSES_GT"
            elif not nested416_detected:
                failure_stage_416 = "PLATE_IN_ROI_MISS"
            else:
                failure_stage_416 = "LOCALIZATION_FAIL"

        failure_stage_640 = "NONE"
        if not r1_640_detected:
            if d640_cnt > 0:
                failure_stage_640 = "DIRECT_POOR_LOCALIZATION_OR_FP"
            elif not vehicle_found:
                failure_stage_640 = "VEHICLE_STAGE_MISS"
            elif not vehicle_roi_covers_gt:
                failure_stage_640 = "VEHICLE_ROI_MISSES_GT"
            elif not nested640_detected:
                failure_stage_640 = "PLATE_IN_ROI_MISS"
            else:
                failure_stage_640 = "LOCALIZATION_FAIL"

        results.append({
            "sample_id": sample_id,
            "ground_truth": gt_text,
            "layout": gt_layout,
            "plate_area": gt_area,
            "gt_bbox": f"[{gt_box[0]},{gt_box[1]},{gt_box[2]},{gt_box[3]}]",
            "image_path": str(img_p).replace("\\", "/"),
            # Direct 416
            "direct416_detected": direct416_detected,
            "direct416_conf": round(direct416_conf, 4),
            "direct416_iou": round(direct416_iou, 4),
            "direct416_candidates_count": d416_cnt,
            "direct416_bbox": f"[{direct416_box[0]},{direct416_box[1]},{direct416_box[2]},{direct416_box[3]}]" if direct416_box else "",
            # R1 416
            "r1_416_detected": r1_416_detected,
            "r1_416_source": r1_416_source,
            "r1_416_iou": round(r1_416_iou, 4),
            "r1_416_ocr": r1_416_ocr,
            "r1_416_e2e": r1_416_e2e,
            "cascade416_invoked": cascade416_invoked,
            "failure_stage_416": failure_stage_416,
            # Direct 640
            "direct640_detected": direct640_detected,
            "direct640_conf": round(direct640_conf, 4),
            "direct640_iou": round(direct640_iou, 4),
            "direct640_candidates_count": d640_cnt,
            "direct640_bbox": f"[{direct640_box[0]},{direct640_box[1]},{direct640_box[2]},{direct640_box[3]}]" if direct640_box else "",
            # R1 640
            "r1_640_detected": r1_640_detected,
            "r1_640_source": r1_640_source,
            "r1_640_iou": round(r1_640_iou, 4),
            "r1_640_ocr": r1_640_ocr,
            "r1_640_e2e": r1_640_e2e,
            "cascade640_invoked": cascade640_invoked,
            "failure_stage_640": failure_stage_640,
            # Vehicle & Nested
            "vehicle_found": vehicle_found,
            "vehicle_roi_covers_gt": vehicle_roi_covers_gt,
            "nested416_detected": nested416_detected,
            "nested640_detected": nested640_detected,
            "valid_roi_plate416_miss": valid_roi_plate416_miss,
            # Diagnostic Mixed Arm
            "mixed_detected": mixed_detected,
            "mixed_e2e": mixed_e2e,
            "mixed_source": mixed_source,
            # Standardized Latency
            "latency416": round(latency416, 2),
            "latency640": round(latency640, 2),
        })

        if idx % 50 == 0 or idx == len(samples):
            elapsed = time.perf_counter() - t_start
            fps = idx / elapsed
            print(f"  Processed {idx:>3}/500 | 416 Det: {sum(r['r1_416_detected'] for r in results):>3} | 640 Det: {sum(r['r1_640_detected'] for r in results):>3} | Rate: {fps:.1f} FPS")

    # Peak memory tracking
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    peak_cpu_mb = process.memory_info().rss / (1024 * 1024)
    print(f"\nResource Usage: Peak GPU Allocated = {peak_gpu_mb:.1f} MB | Peak CPU RSS = {peak_cpu_mb:.1f} MB")

    # Save results CSV
    fieldnames = list(results[0].keys())
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults successfully written to: {output_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_controlled_experiment()
