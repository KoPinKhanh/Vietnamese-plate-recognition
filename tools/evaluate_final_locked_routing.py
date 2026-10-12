"""
Final Locked Routing Benchmark: Direct Plate Detection vs Locked Hybrid R1
Evaluates on the Canonical Clean Test Partition: datasets/VNLP_processed/test_clean.csv (N=3,727 full-scene images).

Outputs:
  - reports/final_locked_routing_results.csv
  - outputs/final_routing_test/false_positives/
  - outputs/final_routing_test/neither_detected/
"""

import argparse
import csv
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


def run_final_benchmark(
    manifest_csv: Path = PROJECT_ROOT / "datasets/VNLP_processed/test_clean.csv",
    output_csv: Path = PROJECT_ROOT / "reports/final_locked_routing_results.csv",
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
    print("FINAL LOCKED ROUTING BENCHMARK: DIRECT PLATE VS HYBRID R1")
    print(f"Manifest:               {manifest_csv}")
    print(f"Plate Detector:         {plate_detector_path} (416x416, conf={plate_conf_thresh}, NMS={plate_iou_thresh})")
    print(f"Vehicle Detector:       yolo11s.pt (640x640, conf={vehicle_conf_thresh}, pad={vehicle_pad_ratio*100:.1f}%)")
    print(f"OCR Engine:             {ocr_checkpoint} on {device}")
    print("=" * 80)

    # 1. Output Folders
    eval_dir = PROJECT_ROOT / "outputs/final_routing_test"
    fp_dir = eval_dir / "false_positives"
    neither_dir = eval_dir / "neither_detected"

    for d in (fp_dir, neither_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 2. Model Initialization strictly once
    print("\n[Init] Loading Ultralytics YOLO11s Vehicle Detector...")
    vehicle_model = YOLO("yolo11s.pt")
    vehicle_classes = [cls_id for cls_id, name in vehicle_model.names.items() if name in ("car", "motorcycle", "bus", "truck")]

    print("\n[Init] Loading YOLO11n License Plate Detector (OpenCV DNN 416)...")
    plate_detector = PlateDetector(model_path=str(plate_detector_path), imgsz=416, conf=plate_conf_thresh, iou=plate_iou_thresh)

    print("\n[Init] Loading PARSeq-Tiny VNLP OCR...")
    ocr_recognizer = ParseqVNRecognizer(checkpoint=str(ocr_checkpoint), device=device)

    # 3. Standardized Warmup (Section 24)
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
    print("[Warmup] Complete. Benchmark running on full canonical test partition.\n")

    # 4. Load Manifest
    samples = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader, 1):
            r["sample_id"] = f"TEST_{idx:04d}"
            # Resolve layout
            sub = r.get("subset", "")
            if sub == "two_rows_label_xe_may":
                r["gt_layout"] = "motorcycle"
            elif sub == "two_rows" or r.get("layout") == "two_rows":
                r["gt_layout"] = "two_rows"
            else:
                r["gt_layout"] = "one_row"
            samples.append(r)

    total_samples = len(samples)
    print(f"Loaded {total_samples} samples from canonical test partition.")

    results = []
    direct_hits = 0
    r1_hits = 0
    direct_e2e_hits = 0
    r1_e2e_hits = 0
    cascade_invocations = 0

    t_start_all = time.perf_counter()

    for idx, s in enumerate(samples, 1):
        sample_id = s["sample_id"]
        img_path = Path(s["image_path"])
        gt_text = normalize_plate_text(s["plate_text"])
        gt_layout = s["gt_layout"]
        gt_bbox = [int(float(s["x1"])), int(float(s["y1"])), int(float(s["x2"])), int(float(s["y2"]))]
        gt_w = gt_bbox[2] - gt_bbox[0]
        gt_h = gt_bbox[3] - gt_bbox[1]
        gt_area = gt_w * gt_h
        gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2.0
        gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2.0

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        ocr_cache = {}

        def get_ocr_result(box: List[int]) -> Tuple[str, float, bool, float]:
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
            conf = float(res.get("confidence", 0.0))
            exact = (norm_pred == gt_text)
            ocr_cache[key] = (norm_pred, conf, exact, t_ms)
            return ocr_cache[key]

        # -------------------------------------------------------------
        # SYSTEM A: DIRECT PLATE DETECTION
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
        direct_bbox = best_d_cand["bbox"] if best_d_cand else None
        direct_ocr = ""
        direct_ocr_conf = 0.0
        direct_e2e = False
        direct_ocr_ms = 0.0

        if direct_detected and direct_bbox is not None:
            direct_ocr, direct_ocr_conf, direct_e2e, direct_ocr_ms = get_ocr_result(direct_bbox)

        direct_total_ms = direct_det_ms + direct_ocr_ms

        # -------------------------------------------------------------
        # SYSTEM B: LOCKED HYBRID R1 (NO-DETECTION FALLBACK)
        # -------------------------------------------------------------
        vehicle_found = False
        best_vehicle_class = ""
        best_vehicle_conf = 0.0
        best_vehicle_bbox = []
        cascade_invoked = False
        failure_stage = "NONE"

        if len(direct_candidates) > 0:
            cascade_invoked = False
            r1_candidates = direct_candidates
            r1_det_ms = direct_det_ms
            failure_stage = "NOT_INVOKED"
        else:
            cascade_invoked = True
            cascade_invocations += 1

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

            vehicle_found = len(v_preds.boxes) > 0
            vehicle_roi_available = False
            nested_plate_ms = 0.0
            raw_cascade_dets = []

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

                vw = vx2 - vx1
                vh = vy2 - vy1
                pad_w = int(round(vw * vehicle_pad_ratio))
                pad_h = int(round(vh * vehicle_pad_ratio))
                rx1 = max(0, vx1 - pad_w)
                ry1 = max(0, vy1 - pad_h)
                rx2 = min(iw, vx2 + pad_w)
                ry2 = min(ih, vy2 + pad_h)

                if rx1 <= gt_cx <= rx2 and ry1 <= gt_cy <= ry2:
                    vehicle_roi_available = True

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
                        "source": "CASCADE",
                    })

            # Global NMS across vehicle ROIs
            cascade_candidates = apply_global_plate_nms(raw_cascade_dets, iou_thresh=plate_iou_thresh)
            for c in cascade_candidates:
                c["source"] = "CASCADE"

            r1_candidates = cascade_candidates
            r1_det_ms = direct_det_ms + veh_ms + nested_plate_ms

            if len(cascade_candidates) > 0:
                failure_stage = "NONE"
            elif not vehicle_found or not vehicle_roi_available:
                failure_stage = "VEHICLE_STAGE_MISS"
            else:
                failure_stage = "PLATE_IN_ROI_MISS"

        # Match R1 candidates with GT
        best_r1_cand = None
        best_r1_iou = 0.0
        for cand in r1_candidates:
            iou = calculate_iou(cand["bbox"], gt_bbox)
            if iou > best_r1_iou:
                best_r1_iou = iou
                best_r1_cand = cand

        r1_detected = (best_r1_cand is not None and best_r1_iou >= 0.50)
        r1_source = best_r1_cand["source"] if best_r1_cand else "NONE"
        r1_conf = best_r1_cand["confidence"] if best_r1_cand else 0.0
        r1_bbox = best_r1_cand["bbox"] if best_r1_cand else None
        r1_ocr = ""
        r1_ocr_conf = 0.0
        r1_e2e = False
        r1_ocr_ms = 0.0

        if r1_detected and r1_bbox is not None:
            r1_ocr, r1_ocr_conf, r1_e2e, r1_ocr_ms = get_ocr_result(r1_bbox)

        r1_total_ms = r1_det_ms + r1_ocr_ms

        if cascade_invoked and not r1_detected and failure_stage == "NONE":
            failure_stage = "PLATE_LOCALIZATION_FAIL"

        if direct_detected: direct_hits += 1
        if r1_detected: r1_hits += 1
        if direct_e2e: direct_e2e_hits += 1
        if r1_e2e: r1_e2e_hits += 1

        record = {
            "sample_id": sample_id,
            "image_path": s["image_path"],
            "ground_truth": gt_text,
            "gt_layout": gt_layout,
            "gt_bbox": str(gt_bbox),
            "direct_detected": direct_detected,
            "direct_conf": round(direct_conf, 4),
            "direct_bbox": str(direct_bbox) if direct_bbox else "",
            "direct_iou": round(best_d_iou, 4),
            "direct_ocr": direct_ocr,
            "direct_ocr_conf": round(direct_ocr_conf, 4),
            "direct_e2e": direct_e2e,
            "direct_candidates_count": len(direct_candidates),
            "cascade_invoked": cascade_invoked,
            "vehicle_found": vehicle_found,
            "vehicle_class": best_vehicle_class,
            "vehicle_conf": round(best_vehicle_conf, 4),
            "vehicle_bbox": str(best_vehicle_bbox) if best_vehicle_bbox else "",
            "r1_source": r1_source,
            "r1_detected": r1_detected,
            "r1_conf": round(r1_conf, 4),
            "r1_bbox": str(r1_bbox) if r1_bbox else "",
            "r1_iou": round(best_r1_iou, 4),
            "r1_ocr": r1_ocr,
            "r1_ocr_conf": round(r1_ocr_conf, 4),
            "r1_e2e": r1_e2e,
            "r1_candidates_count": len(r1_candidates),
            "failure_stage": failure_stage,
            "plate_area": gt_area,
            "direct_ms": round(direct_total_ms, 2),
            "r1_ms": round(r1_total_ms, 2),
        }
        results.append(record)

        # Save representative false positives and neither detected
        fp_count = len(r1_candidates) - (1 if r1_detected else 0)
        if fp_count > 0 and len(list(fp_dir.glob("*.jpg"))) < 50:
            cv2.imwrite(str(fp_dir / f"fp_{sample_id}_{gt_layout}_{gt_text}.jpg"), img)
        if not direct_detected and not r1_detected and len(list(neither_dir.glob("*.jpg"))) < 50:
            cv2.imwrite(str(neither_dir / f"neither_{sample_id}_{gt_layout}_{gt_text}.jpg"), img)

        if idx % 200 == 0 or idx == total_samples:
            elapsed = time.perf_counter() - t_start_all
            print(f"  Processed [{idx:04d}/{total_samples}] samples ({idx/elapsed:.1f} FPS) | "
                  f"Direct: {direct_hits} ({direct_hits/idx*100:.1f}%) | "
                  f"R1: {r1_hits} ({r1_hits/idx*100:.1f}%) | "
                  f"Direct E2E: {direct_e2e_hits} ({direct_e2e_hits/idx*100:.1f}%) | "
                  f"R1 E2E: {r1_e2e_hits} ({r1_e2e_hits/idx*100:.1f}%) | "
                  f"Cascade Invocations: {cascade_invocations}")

    # Write Output CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    total_time = time.perf_counter() - t_start_all
    print(f"\n[Done] Benchmark complete in {total_time:.1f}s ({total_samples/total_time:.1f} FPS).")
    print(f"Results saved to: {output_csv}")

    return {
        "results": results,
        "direct_hits": direct_hits,
        "r1_hits": r1_hits,
        "direct_e2e": direct_e2e_hits,
        "r1_e2e": r1_e2e_hits,
        "total_samples": total_samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Locked Routing Benchmark")
    parser.add_argument("--manifest", type=str, default="datasets/VNLP_processed/test_clean.csv")
    parser.add_argument("--output", type=str, default="reports/final_locked_routing_results.csv")
    args = parser.parse_args()

    run_final_benchmark(
        manifest_csv=Path(args.manifest),
        output_csv=Path(args.output),
    )
