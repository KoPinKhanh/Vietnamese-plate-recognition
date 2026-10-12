"""
Controlled Experiment: Direct Plate Detection vs Hailo-Inspired Vehicle -> Plate Cascade.
Evaluates both Pipeline A (Direct) and Pipeline B (Cascade) on the exact same 300 full-scene images
from reports/e2e_real_world_manifest.csv.

Outputs:
  - reports/direct_vs_cascade_results.csv
  - outputs/cascade_eval/debug/
  - outputs/cascade_eval/rescued_by_cascade/
  - outputs/cascade_eval/direct_only/
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

# Ensure project root is in sys.path
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


def mcnemar_exact_test(b: int, c: int) -> float:
    """Computes exact two-tailed McNemar test p-value using Binomial distribution."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2.0 * prob)


def paired_bootstrap_delta(direct_hits: List[bool], cascade_hits: List[bool], n_boot: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    """Computes paired bootstrap delta recall and 95% confidence interval."""
    np.random.seed(seed)
    n = len(direct_hits)
    d_arr = np.array(direct_hits, dtype=np.float32)
    c_arr = np.array(cascade_hits, dtype=np.float32)
    obs_delta = (c_arr.mean() - d_arr.mean()) * 100.0

    indices = np.random.randint(0, n, size=(n_boot, n))
    d_boot = d_arr[indices].mean(axis=1) * 100.0
    c_boot = c_arr[indices].mean(axis=1) * 100.0
    delta_boot = c_boot - d_boot

    ci_low = float(np.percentile(delta_boot, 2.5))
    ci_high = float(np.percentile(delta_boot, 97.5))
    return obs_delta, ci_low, ci_high


def apply_global_plate_nms(detections: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
    """Eliminates duplicate plate detections across multiple overlapping vehicle ROIs."""
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


def draw_cascade_comparison_image(
    image: np.ndarray,
    gt_bbox: List[int],
    gt_text: str,
    layout: str,
    direct_bbox: Optional[List[int]],
    direct_iou: float,
    direct_conf: float,
    vehicle_boxes: List[Dict[str, Any]],
    cascade_bbox: Optional[List[int]],
    cascade_iou: float,
    cascade_conf: float,
    paired_state: str,
) -> np.ndarray:
    """Renders visual debug comparison showing Vehicle ROIs, Direct plate, and Cascade plate."""
    canvas = image.copy()
    ih, iw = canvas.shape[:2]

    # 1. Draw Vehicle ROIs (Purple / Magenta)
    for v in vehicle_boxes:
        vx1, vy1, vx2, vy2 = v["bbox"]
        v_label = f"{v['class']} {v['conf']:.2f}"
        cv2.rectangle(canvas, (vx1, vy1), (vx2, vy2), (220, 50, 220), 1)
        cv2.putText(canvas, v_label, (vx1 + 4, max(12, vy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 50, 220), 1, cv2.LINE_AA)

    # 2. Draw Ground Truth BBox (Bright Green)
    gx1, gy1, gx2, gy2 = gt_bbox
    cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
    cv2.putText(canvas, f"GT: {gt_text} ({layout})", (gx1, max(15, gy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)

    # 3. Draw Direct Plate BBox (Orange)
    if direct_bbox is not None:
        dx1, dy1, dx2, dy2 = direct_bbox
        cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (0, 140, 255), 2)
        cv2.putText(canvas, f"Direct IoU:{direct_iou:.2f}", (dx1, min(ih - 5, dy2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1, cv2.LINE_AA)

    # 4. Draw Cascade Plate BBox (Cyan)
    if cascade_bbox is not None:
        cx1, cy1, cx2, cy2 = cascade_bbox
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (255, 220, 0), 2)
        cv2.putText(canvas, f"Cascade IoU:{cascade_iou:.2f}", (cx1, min(ih - 20, cy2 + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 0), 1, cv2.LINE_AA)

    # 5. Top Status Header
    header_color = (0, 220, 0) if paired_state in ("BOTH_DETECT", "CASCADE_ONLY") else (0, 0, 220)
    status_text = f"State: {paired_state} | Direct IoU: {direct_iou:.2f} | Cascade IoU: {cascade_iou:.2f} | GT: {gt_text}"
    cv2.rectangle(canvas, (0, 0), (iw, 28), (25, 25, 25), -1)
    cv2.putText(canvas, status_text, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.50, header_color, 2, cv2.LINE_AA)

    return canvas


def run_direct_vs_cascade_experiment(
    manifest_csv: Path = PROJECT_ROOT / "reports/e2e_real_world_manifest.csv",
    output_csv: Path = PROJECT_ROOT / "reports/direct_vs_cascade_results.csv",
    plate_detector_path: Path = PROJECT_ROOT / "weights/best_license_plate_yolo11n_416.onnx",
    ocr_checkpoint: Path = PROJECT_ROOT / "weights/ocr/teachers/parseq_tiny_vn/best.pth",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    vehicle_conf_thresh: float = 0.25,
    vehicle_pad_ratio: float = 0.05,
    plate_conf_thresh: float = 0.50,
    plate_iou_thresh: float = 0.45,
    crop_pad: float = 0.05,
) -> Dict[str, Any]:
    print("=" * 80)
    print("CONTROLLED EXPERIMENT: DIRECT PLATE DETECTION VS VEHICLE -> PLATE CASCADE")
    print(f"Manifest:               {manifest_csv}")
    print(f"Vehicle Detector:       yolo11s.pt (imgsz=640, conf={vehicle_conf_thresh}, pad={vehicle_pad_ratio*100:.1f}%)")
    print(f"Plate Detector:         {plate_detector_path} (imgsz=416, conf={plate_conf_thresh}, NMS={plate_iou_thresh})")
    print(f"OCR Recognizer:         {ocr_checkpoint} on {device}")
    print("=" * 80)

    # 1. Output Folders
    eval_dir = PROJECT_ROOT / "outputs/cascade_eval"
    debug_dir = eval_dir / "debug"
    rescued_dir = eval_dir / "rescued_by_cascade"
    direct_only_dir = eval_dir / "direct_only"

    for d in (debug_dir, rescued_dir, direct_only_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load Models Strictly Once
    print("\n[Init] Loading Ultralytics YOLO11s Vehicle Detector...")
    vehicle_model = YOLO("yolo11s.pt")
    vehicle_classes = [cls_id for cls_id, name in vehicle_model.names.items() if name in ("car", "motorcycle", "bus", "truck")]
    print(f"  -> Monitored vehicle classes: {[vehicle_model.names[c] for c in vehicle_classes]}")

    print("\n[Init] Loading YOLO11n License Plate Detector (OpenCV DNN 416)...")
    plate_detector = PlateDetector(model_path=str(plate_detector_path), imgsz=416, conf=plate_conf_thresh, iou=plate_iou_thresh)

    print("\n[Init] Loading PARSeq-Tiny VNLP OCR...")
    ocr_recognizer = ParseqVNRecognizer(checkpoint=str(ocr_checkpoint), device=device)
    ocr_recognizer.warmup(15)
    print("All models initialized successfully.\n")

    # 3. Read Manifest
    samples = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            row["sample_id"] = f"SMP_{idx:04d}"
            samples.append(row)

    total_samples = len(samples)
    print(f"Evaluating {total_samples} full-scene images...")

    results_records = []
    direct_hits = []
    cascade_hits = []

    t_start_all = time.perf_counter()

    for idx, s in enumerate(samples, 1):
        sample_id = s["sample_id"]
        img_path = Path(s["image_path"])
        gt_text = normalize_plate_text(s["ground_truth"])
        gt_layout = s["gt_layout"]
        gt_bbox = [int(s["gt_x1"]), int(s["gt_y1"]), int(s["gt_x2"]), int(s["gt_y2"])]
        gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2.0
        gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2.0

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        # =========================================================
        # PIPELINE A: DIRECT FULL-FRAME PLATE DETECTION
        # =========================================================
        t_d_start = time.perf_counter()
        direct_dets, d_timings = plate_detector.detect(img)
        t_d_end = time.perf_counter()
        direct_total_ms = (t_d_end - t_d_start) * 1000.0

        # Match Direct Plate
        best_d_det = None
        best_d_iou = 0.0
        for d in direct_dets:
            iou = calculate_iou(d["bbox"], gt_bbox)
            if iou > best_d_iou:
                best_d_iou = iou
                best_d_det = d

        direct_detected = (best_d_det is not None and best_d_iou >= 0.50)
        direct_bbox = best_d_det["bbox"] if best_d_det else None
        direct_plate_conf = best_d_det["confidence"] if best_d_det else 0.0

        # Direct OCR
        direct_ocr = ""
        direct_ocr_correct = False
        direct_e2e_correct = False
        if direct_detected and direct_bbox is not None:
            t_ocr0 = time.perf_counter()
            d_crop, _ = crop_with_padding(img, direct_bbox, pad_ratio=crop_pad)
            d_res = ocr_recognizer.recognize(d_crop, layout="auto")
            direct_ocr = d_res["text"]
            direct_ocr_correct = (direct_ocr == gt_text)
            direct_e2e_correct = direct_ocr_correct
            direct_total_ms += (time.perf_counter() - t_ocr0) * 1000.0

        # =========================================================
        # PIPELINE B: HAILO-INSPIRED VEHICLE -> PLATE CASCADE
        # =========================================================
        t_c_start = time.perf_counter()
        v_preds = vehicle_model.predict(
            img,
            imgsz=640,
            conf=vehicle_conf_thresh,
            classes=vehicle_classes,
            device=device,
            verbose=False,
        )[0]
        t_v_end = time.perf_counter()
        veh_ms = (t_v_end - t_c_start) * 1000.0

        vehicle_boxes = []
        vehicle_found = len(v_preds.boxes) > 0
        best_vehicle_conf = 0.0
        best_vehicle_class = ""
        best_vehicle_bbox = []
        vehicle_roi_available = False

        cascade_raw_plate_dets = []
        nested_plate_total_ms = 0.0

        for b in v_preds.boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            v_conf = float(b.conf[0].cpu().item())
            cls_id = int(b.cls[0].cpu().item())
            cls_name = vehicle_model.names[cls_id]
            vx1, vy1, vx2, vy2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            if v_conf > best_vehicle_conf:
                best_vehicle_conf = v_conf
                best_vehicle_class = cls_name
                best_vehicle_bbox = [vx1, vy1, vx2, vy2]

            # 5% vehicle padding
            vw = vx2 - vx1
            vh = vy2 - vy1
            pad_w = int(round(vw * vehicle_pad_ratio))
            pad_h = int(round(vh * vehicle_pad_ratio))
            rx1 = max(0, vx1 - pad_w)
            ry1 = max(0, vy1 - pad_h)
            rx2 = min(iw, vx2 + pad_w)
            ry2 = min(ih, vy2 + pad_h)

            vehicle_boxes.append({
                "bbox": [vx1, vy1, vx2, vy2],
                "roi_bbox": [rx1, ry1, rx2, ry2],
                "class": cls_name,
                "conf": v_conf,
            })

            # Check if this ROI contains GT plate center
            if rx1 <= gt_cx <= rx2 and ry1 <= gt_cy <= ry2:
                vehicle_roi_available = True

            # Crop ROI and run plate detector
            roi = img[ry1:ry2, rx1:rx2]
            if roi.size == 0 or roi.shape[0] < 12 or roi.shape[1] < 18:
                continue

            t_p0 = time.perf_counter()
            roi_plate_dets, _ = plate_detector.detect(roi)
            nested_plate_total_ms += (time.perf_counter() - t_p0) * 1000.0

            # Map coordinates back to full image
            for p_det in roi_plate_dets:
                px1, py1, px2, py2 = p_det["bbox"]
                fx1 = max(0, min(iw, rx1 + px1))
                fy1 = max(0, min(ih, ry1 + py1))
                fx2 = max(0, min(iw, rx1 + px2))
                fy2 = max(0, min(ih, ry1 + py2))
                cascade_raw_plate_dets.append({
                    "bbox": [fx1, fy1, fx2, fy2],
                    "confidence": p_det["confidence"],
                    "vehicle_class": cls_name,
                })

        # Apply Global Plate NMS across all vehicle ROIs
        cascade_filtered_dets = apply_global_plate_nms(cascade_raw_plate_dets, iou_thresh=plate_iou_thresh)

        # Match Cascade Plate
        best_c_det = None
        best_c_iou = 0.0
        for cd in cascade_filtered_dets:
            iou = calculate_iou(cd["bbox"], gt_bbox)
            if iou > best_c_iou:
                best_c_iou = iou
                best_c_det = cd

        cascade_detected = (best_c_det is not None and best_c_iou >= 0.50)
        cascade_bbox = best_c_det["bbox"] if best_c_det else None
        cascade_plate_conf = best_c_det["confidence"] if best_c_det else 0.0

        # Cascade OCR
        cascade_ocr = ""
        cascade_ocr_correct = False
        cascade_e2e_correct = False
        c_ocr_ms = 0.0
        if cascade_detected and cascade_bbox is not None:
            t_c_ocr0 = time.perf_counter()
            c_crop, _ = crop_with_padding(img, cascade_bbox, pad_ratio=crop_pad)
            c_res = ocr_recognizer.recognize(c_crop, layout="auto")
            cascade_ocr = c_res["text"]
            cascade_ocr_correct = (cascade_ocr == gt_text)
            cascade_e2e_correct = cascade_ocr_correct
            c_ocr_ms = (time.perf_counter() - t_c_ocr0) * 1000.0

        cascade_total_ms = veh_ms + nested_plate_total_ms + c_ocr_ms

        # =========================================================
        # PAIRED COMPARISON & CASCADE MISS ATTRIBUTION
        # =========================================================
        if direct_detected and cascade_detected:
            paired_state = "BOTH_DETECT"
        elif direct_detected and not cascade_detected:
            paired_state = "DIRECT_ONLY"
        elif not direct_detected and cascade_detected:
            paired_state = "CASCADE_ONLY"
        else:
            paired_state = "NEITHER"

        # Cascade Miss Stage
        if cascade_detected:
            cascade_miss_stage = "NONE"
        elif not vehicle_found or not vehicle_roi_available:
            cascade_miss_stage = "VEHICLE_STAGE_MISS"
        elif best_c_det is None:
            cascade_miss_stage = "PLATE_IN_ROI_MISS"
        elif best_c_iou < 0.50:
            cascade_miss_stage = "PLATE_LOCALIZATION_FAIL"
        else:
            cascade_miss_stage = "OTHER"

        direct_hits.append(direct_detected)
        cascade_hits.append(cascade_detected)

        record = {
            "sample_id": sample_id,
            "image_path": s["image_path"],
            "ground_truth": gt_text,
            "layout": gt_layout,
            "gt_bbox": str(gt_bbox),
            "direct_detected": direct_detected,
            "direct_plate_conf": round(direct_plate_conf, 4),
            "direct_bbox": str(direct_bbox) if direct_bbox else "",
            "direct_iou": round(best_d_iou, 4),
            "direct_ocr": direct_ocr,
            "direct_ocr_correct": direct_ocr_correct,
            "direct_e2e_correct": direct_e2e_correct,
            "vehicle_found": vehicle_found,
            "vehicle_class": best_vehicle_class,
            "vehicle_conf": round(best_vehicle_conf, 4),
            "vehicle_bbox": str(best_vehicle_bbox) if best_vehicle_bbox else "",
            "cascade_detected": cascade_detected,
            "cascade_plate_conf": round(cascade_plate_conf, 4),
            "cascade_bbox": str(cascade_bbox) if cascade_bbox else "",
            "cascade_iou": round(best_c_iou, 4),
            "cascade_ocr": cascade_ocr,
            "cascade_ocr_correct": cascade_ocr_correct,
            "cascade_e2e_correct": cascade_e2e_correct,
            "paired_detection_state": paired_state,
            "cascade_miss_stage": cascade_miss_stage,
            "direct_total_ms": round(direct_total_ms, 2),
            "cascade_total_ms": round(cascade_total_ms, 2),
        }
        results_records.append(record)

        # Visual Debug Image
        debug_vis = draw_cascade_comparison_image(
            image=img,
            gt_bbox=gt_bbox,
            gt_text=gt_text,
            layout=gt_layout,
            direct_bbox=direct_bbox,
            direct_iou=best_d_iou,
            direct_conf=direct_plate_conf,
            vehicle_boxes=vehicle_boxes,
            cascade_bbox=cascade_bbox,
            cascade_iou=best_c_iou,
            cascade_conf=cascade_plate_conf,
            paired_state=paired_state,
        )

        # Save representative visual cases
        if idx <= 30 or paired_state in ("CASCADE_ONLY", "DIRECT_ONLY"):
            dbg_name = f"dbg_{sample_id}_{paired_state}_{gt_text}.jpg"
            cv2.imwrite(str(debug_dir / dbg_name), debug_vis)

        if paired_state == "CASCADE_ONLY":
            res_name = f"rescued_{sample_id}_{gt_layout}_{gt_text}.jpg"
            cv2.imwrite(str(rescued_dir / res_name), debug_vis)
        elif paired_state == "DIRECT_ONLY":
            dir_name = f"direct_only_{sample_id}_{gt_layout}_{gt_text}.jpg"
            cv2.imwrite(str(direct_only_dir / dir_name), debug_vis)

        if idx % 50 == 0 or idx == total_samples:
            elapsed = time.perf_counter() - t_start_all
            print(f"  Processed [{idx:04d}/{total_samples}] samples ({idx/elapsed:.1f} FPS) | "
                  f"Direct Hits: {sum(direct_hits)} | Cascade Hits: {sum(cascade_hits)} | "
                  f"Rescued: {sum(1 for r in results_records if r['paired_detection_state'] == 'CASCADE_ONLY')} | "
                  f"Direct-Only: {sum(1 for r in results_records if r['paired_detection_state'] == 'DIRECT_ONLY')}")

    total_eval_time = time.perf_counter() - t_start_all

    # 4. Save Results CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_records[0].keys()))
        writer.writeheader()
        writer.writerows(results_records)
    print(f"\nSaved detailed comparison results to: {output_csv}")

    return {
        "records": results_records,
        "direct_hits": direct_hits,
        "cascade_hits": cascade_hits,
        "total_time_s": total_eval_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Direct vs Hailo Cascade")
    parser.add_argument("--manifest", type=str, default="reports/e2e_real_world_manifest.csv")
    parser.add_argument("--output", type=str, default="reports/direct_vs_cascade_results.csv")
    args = parser.parse_args()

    run_direct_vs_cascade_experiment(
        manifest_csv=Path(args.manifest),
        output_csv=Path(args.output),
    )
