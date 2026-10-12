"""
Real-World End-to-End Evaluation Suite for YOLO11n + PARSeq-Tiny VNLP.
Evaluates detection, localization crop, layout classification, two-row rectification,
and OCR stages systematically with full error attribution.

Outputs:
  - reports/e2e_real_world_results.csv
  - reports/e2e_crop_margin_sensitivity.csv
  - reports/e2e_character_confusions.csv
  - outputs/e2e_eval/crops/gt/
  - outputs/e2e_eval/crops/yolo/
  - outputs/e2e_eval/rectification/
  - outputs/e2e_eval/debug/
  - outputs/e2e_eval/errors/detection/
  - outputs/e2e_eval/errors/crop/
  - outputs/e2e_eval/errors/layout/
  - outputs/e2e_eval/errors/rectification/
  - outputs/e2e_eval/errors/ocr/
  - outputs/e2e_eval/errors/unknown/
"""

import argparse
import csv
import difflib
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

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.pi_yolo_detector import PlateDetector
from ocr.parseq_vn_ocr import ParseqVNRecognizer
from ocr.preprocess import rectify_two_row_plate, find_horizontal_split
from ocr.plate_text_utils import normalize_plate_text, calculate_cer, levenshtein_distance
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


def align_characters_for_confusion(gt: str, pred: str) -> List[Tuple[str, str]]:
    """Aligns GT and predicted string to extract character-level substitution pairs."""
    matcher = difflib.SequenceMatcher(None, gt, pred)
    confusions = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            g_sub = gt[i1:i2]
            p_sub = pred[j1:j2]
            min_l = min(len(g_sub), len(p_sub))
            for k in range(min_l):
                confusions.append((g_sub[k], p_sub[k]))
            if len(g_sub) > min_l:
                for k in range(min_l, len(g_sub)):
                    confusions.append((g_sub[k], "<DEL>"))
            elif len(p_sub) > min_l:
                for k in range(min_l, len(p_sub)):
                    confusions.append(("<INS>", p_sub[k]))
        elif tag == "delete":
            for k in range(i1, i2):
                confusions.append((gt[k], "<DEL>"))
        elif tag == "insert":
            for k in range(j1, j2):
                confusions.append(("<INS>", pred[k]))
    return confusions


def draw_eval_debug_image(
    image: np.ndarray,
    gt_bbox: List[int],
    pred_bbox: Optional[List[int]],
    gt_text: str,
    pred_text: str,
    iou: float,
    ocr_conf: float,
    gt_layout: str,
    pred_layout: str,
    failure_stage: str,
) -> np.ndarray:
    """Renders comprehensive debug annotation on full image."""
    canvas = image.copy()
    ih, iw = canvas.shape[:2]

    # Draw Ground Truth BBox (Bright Green)
    gx1, gy1, gx2, gy2 = gt_bbox
    cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
    cv2.putText(canvas, f"GT: {gt_text} ({gt_layout})", (gx1, max(15, gy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw Predicted BBox (Bright Blue / Red based on correctness)
    if pred_bbox is not None:
        px1, py1, px2, py2 = pred_bbox
        p_color = (255, 200, 0) if failure_stage == "NONE" else (0, 0, 255)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), p_color, 2)
        cv2.putText(canvas, f"Pred: {pred_text} | Conf: {ocr_conf:.2f} | IoU: {iou:.2f}",
                    (px1, min(ih - 8, py2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, p_color, 2, cv2.LINE_AA)

    # Status Banner Top Left
    status_color = (0, 200, 0) if failure_stage == "NONE" else (0, 0, 220)
    banner_text = f"Status: {failure_stage} | IoU: {iou:.2f} | Pred: {pred_text} | GT: {gt_text}"
    cv2.rectangle(canvas, (0, 0), (iw, 30), (30, 30, 30), -1)
    cv2.putText(canvas, banner_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)

    return canvas


def run_evaluation(
    manifest_csv: Path = PROJECT_ROOT / "reports/e2e_real_world_manifest.csv",
    results_csv: Path = PROJECT_ROOT / "reports/e2e_real_world_results.csv",
    detector_path: Path = PROJECT_ROOT / "weights/best_license_plate_yolo11n_416.onnx",
    ocr_checkpoint: Path = PROJECT_ROOT / "weights/ocr/teachers/parseq_tiny_vn/best.pth",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    det_conf: float = 0.50,
    det_iou: float = 0.45,
    crop_pad: float = 0.05,
    run_ab_lprnet: bool = True,
) -> Dict[str, Any]:
    print("=" * 80)
    print("STARTING REAL-WORLD END-TO-END EVALUATION SUITE")
    print(f"Manifest:       {manifest_csv}")
    print(f"Detector Model: {detector_path} (conf={det_conf}, iou={det_iou}, imgsz=416)")
    print(f"OCR Model:      {ocr_checkpoint} on {device}")
    print(f"Crop Padding:   {crop_pad * 100:.1f}%")
    print(f"A/B LPRNet:     {run_ab_lprnet}")
    print("=" * 80)

    # 1. Setup Output Folders
    eval_dir = PROJECT_ROOT / "outputs/e2e_eval"
    crops_gt_dir = eval_dir / "crops/gt"
    crops_yolo_dir = eval_dir / "crops/yolo"
    rect_dir = eval_dir / "rectification"
    debug_dir = eval_dir / "debug"

    err_dirs = {
        "DETECTION_MISS": eval_dir / "errors/detection",
        "LOCALIZATION_CROP_ERROR": eval_dir / "errors/crop",
        "LAYOUT_ERROR": eval_dir / "errors/layout",
        "RECTIFICATION_ERROR": eval_dir / "errors/rectification",
        "OCR_ERROR": eval_dir / "errors/ocr",
        "UNKNOWN": eval_dir / "errors/unknown",
    }

    for d in [crops_gt_dir, crops_yolo_dir, rect_dir, debug_dir] + list(err_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load Models Strictly Once
    detector = PlateDetector(model_path=str(detector_path), imgsz=416, conf=det_conf, iou=det_iou)
    ocr_recognizer = ParseqVNRecognizer(checkpoint=str(ocr_checkpoint), device=device)
    ocr_recognizer.warmup(15)

    lprnet_ocr = None
    if run_ab_lprnet:
        try:
            from ocr.pi_lprnet_ocr import PlateOCR
            lpr_path = PROJECT_ROOT / "weights/ocr/best_lprnet_vn.onnx"
            if lpr_path.exists():
                lprnet_ocr = PlateOCR(model_path=str(lpr_path))
                print(f"[A/B Mode] Loaded Reference LPRNet from {lpr_path}")
        except Exception as e:
            print(f"[A/B Mode Notice] LPRNet could not be loaded: {e}. Skipping A/B.")

    # 3. Read Manifest
    samples = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            row["sample_id"] = f"SMP_{idx:04d}"
            samples.append(row)

    total_samples = len(samples)
    print(f"Loaded {total_samples} samples for evaluation.")

    # Tracking metrics
    results_records = []
    margin_sensitivity_data = {0.0: 0, 0.03: 0, 0.05: 0, 0.08: 0, 0.10: 0}
    margin_total_tested = 0
    all_confusions = Counter()

    t_eval_start = time.perf_counter()

    for idx, s in enumerate(samples, 1):
        sample_id = s["sample_id"]
        img_path = Path(s["image_path"])
        gt_text = normalize_plate_text(s["ground_truth"])
        gt_layout = s["gt_layout"]  # 'one_row', 'two_rows', 'motorcycle'
        gt_bbox = [int(s["gt_x1"]), int(s["gt_y1"]), int(s["gt_x2"]), int(s["gt_y2"])]

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [Warning] Cannot read image: {img_path}")
            continue

        ih, iw = img.shape[:2]

        # ---------------------------------------------------------
        # STAGE 1: DETECTOR EVALUATION
        # ---------------------------------------------------------
        detections, det_timings = detector.detect(img)
        detector_ms = det_timings["total_ms"]

        # Match detection with highest IoU to GT BBox
        best_det = None
        best_iou = 0.0
        for d in detections:
            iou = calculate_iou(d["bbox"], gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_det = d

        is_detected = (best_det is not None and best_iou >= 0.50)
        det_conf_val = best_det["confidence"] if best_det else 0.0
        pred_bbox = best_det["bbox"] if best_det else None

        # ---------------------------------------------------------
        # STAGE 2: GT CROP INFERENCE (Ideal Localization Baseline)
        # ---------------------------------------------------------
        gt_crop, _ = crop_with_padding(img, gt_bbox, pad_ratio=crop_pad)
        gt_ocr_res = ocr_recognizer.recognize(gt_crop, layout="auto")
        gt_pred = gt_ocr_res["text"]
        gt_conf = gt_ocr_res["confidence"]
        gt_correct = (gt_pred == gt_text)

        # Save GT crop
        cv2.imwrite(str(crops_gt_dir / f"{sample_id}_{gt_text}_gt.jpg"), gt_crop)

        # ---------------------------------------------------------
        # STAGE 3: YOLO CROP & PIPELINE INFERENCE (Real E2E)
        # ---------------------------------------------------------
        yolo_pred = ""
        yolo_conf = 0.0
        yolo_correct = False
        pred_layout = "unknown"
        crop_w, crop_h = 0, 0
        aspect_ratio = 0.0
        ocr_ms = 0.0
        prep_ms = 0.0
        is_suspect_rect = False
        notes = []

        if is_detected and pred_bbox is not None:
            # Crop using YOLO bbox + 5% pad
            yolo_crop, _ = crop_with_padding(img, pred_bbox, pad_ratio=crop_pad)
            crop_h, crop_w = yolo_crop.shape[:2]
            aspect_ratio = float(crop_w) / float(max(1, crop_h))

            # Run OCR on YOLO crop
            yolo_ocr_res = ocr_recognizer.recognize(yolo_crop, layout="auto")
            yolo_pred = yolo_ocr_res["text"]
            yolo_conf = yolo_ocr_res["confidence"]
            pred_layout = yolo_ocr_res["layout"]  # 'one_row' or 'two_rows'
            ocr_ms = yolo_ocr_res["ocr_ms"]
            prep_ms = yolo_ocr_res["timings"]["prep_ms"]
            yolo_correct = (yolo_pred == gt_text)

            # Save YOLO crop
            cv2.imwrite(str(crops_yolo_dir / f"{sample_id}_{gt_text}_yolo.jpg"), yolo_crop)

            # Save Two-Row Rectification Stages for two-row plates
            if pred_layout == "two_rows" or gt_layout in ("two_rows", "motorcycle"):
                gray_c = cv2.cvtColor(yolo_crop, cv2.COLOR_BGR2GRAY)
                split_y = find_horizontal_split(gray_c)
                rect_bgr, top_c, bot_c = rectify_two_row_plate(yolo_crop, target_h=48)

                # Diagnostic suspect check
                if split_y < int(crop_h * 0.35) or split_y > int(crop_h * 0.60):
                    is_suspect_rect = True
                    notes.append("split_y_out_of_bounds")
                if top_c.shape[0] < 8 or bot_c.shape[0] < 8:
                    is_suspect_rect = True
                    notes.append("row_too_thin")

                # Save rectification composite for review
                rect_vis_name = f"{sample_id}_{gt_text}_{yolo_pred}_rect.jpg"
                cv2.imwrite(str(rect_dir / rect_vis_name), rect_bgr)

            # Margin Sensitivity Diagnostic
            margin_total_tested += 1
            for m_ratio in [0.0, 0.03, 0.05, 0.08, 0.10]:
                m_crop, _ = crop_with_padding(img, pred_bbox, pad_ratio=m_ratio)
                m_res = ocr_recognizer.recognize(m_crop, layout="auto")
                if m_res["text"] == gt_text:
                    margin_sensitivity_data[m_ratio] += 1

        else:
            # Detection miss
            notes.append("detection_miss" if best_det is None else "low_iou_detection")

        # Layout correctness check
        expected_binary_layout = "one_row" if gt_layout == "one_row" else "two_rows"
        layout_correct = (pred_layout == expected_binary_layout) if is_detected else False

        # Character Confusions Analysis for mismatched predictions
        if is_detected and not yolo_correct and yolo_pred:
            pairs = align_characters_for_confusion(gt_text, yolo_pred)
            for p1, p2 in pairs:
                if p1 != p2:
                    all_confusions[(p1, p2)] += 1

        # ---------------------------------------------------------
        # STAGE 4: STRICT ERROR TAXONOMY ATTRIBUTION
        # ---------------------------------------------------------
        failure_stage = "NONE"
        if yolo_correct:
            failure_stage = "NONE"
        elif not is_detected:
            failure_stage = "DETECTION_MISS"
        elif gt_correct and not yolo_correct:
            # Localization / Crop error: GT bbox OCR succeeded, but YOLO bbox OCR failed
            if not layout_correct:
                failure_stage = "LAYOUT_ERROR"
            else:
                failure_stage = "LOCALIZATION_CROP_ERROR"
        elif not gt_correct and not yolo_correct:
            # Both failed
            if not layout_correct:
                failure_stage = "LAYOUT_ERROR"
            elif is_suspect_rect:
                failure_stage = "RECTIFICATION_ERROR"
            else:
                failure_stage = "OCR_ERROR"
        else:
            failure_stage = "UNKNOWN"

        ed = levenshtein_distance(gt_text, yolo_pred) if yolo_pred else len(gt_text)
        total_frame_ms = detector_ms + prep_ms + ocr_ms

        record = {
            "sample_id": sample_id,
            "image_path": s["image_path"],
            "ground_truth": gt_text,
            "gt_layout": gt_layout,
            "gt_bbox": str(gt_bbox),
            "pred_bbox": str(pred_bbox) if pred_bbox else "",
            "detected": is_detected,
            "det_conf": round(det_conf_val, 4),
            "iou": round(best_iou, 4),
            "crop_width": crop_w,
            "crop_height": crop_h,
            "aspect_ratio": round(aspect_ratio, 3),
            "pred_layout": pred_layout,
            "layout_correct": layout_correct,
            "gt_crop_prediction": gt_pred,
            "gt_crop_conf": round(gt_conf, 4),
            "gt_crop_correct": gt_correct,
            "yolo_crop_prediction": yolo_pred,
            "yolo_crop_conf": round(yolo_conf, 4),
            "yolo_crop_correct": yolo_correct,
            "edit_distance": ed,
            "failure_stage": failure_stage,
            "detector_ms": round(detector_ms, 2),
            "preprocess_ms": round(prep_ms, 2),
            "ocr_ms": round(ocr_ms, 2),
            "total_ms": round(total_frame_ms, 2),
            "notes": ";".join(notes),
        }
        results_records.append(record)

        # Save Visual Debug Image & Error Examples
        debug_img = draw_eval_debug_image(
            image=img,
            gt_bbox=gt_bbox,
            pred_bbox=pred_bbox,
            gt_text=gt_text,
            pred_text=yolo_pred,
            iou=best_iou,
            ocr_conf=yolo_conf,
            gt_layout=gt_layout,
            pred_layout=pred_layout,
            failure_stage=failure_stage,
        )

        if failure_stage != "NONE":
            # Save into failure category folder
            err_folder = err_dirs.get(failure_stage, err_dirs["UNKNOWN"])
            err_name = f"{sample_id}_{failure_stage}_{gt_text}_{yolo_pred or 'MISS'}.jpg"
            cv2.imwrite(str(err_folder / err_name), debug_img)

        # Save sample debug images (first 25 always, plus any error)
        if idx <= 25 or failure_stage != "NONE":
            dbg_name = f"dbg_{sample_id}_{failure_stage}.jpg"
            cv2.imwrite(str(debug_dir / dbg_name), debug_img)

        if idx % 50 == 0 or idx == total_samples:
            elapsed = time.perf_counter() - t_eval_start
            cur_fps = idx / elapsed
            print(f"  Processed [{idx:04d}/{total_samples}] samples ({cur_fps:.1f} FPS) | "
                  f"Detected: {sum(1 for r in results_records if r['detected'])} | "
                  f"YOLO Correct: {sum(1 for r in results_records if r['yolo_crop_correct'])}")

    total_eval_time = time.perf_counter() - t_eval_start

    # ---------------------------------------------------------
    # WRITE OUTPUT CSVs
    # ---------------------------------------------------------
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results_records[0].keys())
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_records)
    print(f"\nSaved per-sample evaluation results to: {results_csv}")

    # Margin Sensitivity CSV
    margin_csv = PROJECT_ROOT / "reports/e2e_crop_margin_sensitivity.csv"
    with open(margin_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["margin_ratio", "margin_percent", "correct_samples", "total_tested", "exact_accuracy"])
        for m_ratio, corr in margin_sensitivity_data.items():
            acc = (corr / margin_total_tested) * 100.0 if margin_total_tested > 0 else 0.0
            writer.writerow([m_ratio, f"{int(m_ratio*100)}%", corr, margin_total_tested, round(acc, 2)])
    print(f"Saved margin sensitivity diagnostic to: {margin_csv}")

    # Character Confusions CSV
    conf_csv = PROJECT_ROOT / "reports/e2e_character_confusions.csv"
    with open(conf_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gt_char", "pred_char", "count"])
        for (g_c, p_c), count in all_confusions.most_common(50):
            writer.writerow([g_c, p_c, count])
    print(f"Saved character confusion analysis to: {conf_csv}")

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE IN {total_eval_time:.2f}s ({total_samples / total_eval_time:.1f} FPS)")
    print("=" * 80)

    return {
        "total_samples": total_samples,
        "results_records": results_records,
        "eval_time_s": total_eval_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Real-World End-to-End Evaluation")
    parser.add_argument("--manifest", type=str, default="reports/e2e_real_world_manifest.csv")
    parser.add_argument("--output", type=str, default="reports/e2e_real_world_results.csv")
    parser.add_argument("--det-conf", type=float, default=0.50)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--crop-pad", type=float, default=0.05)
    args = parser.parse_args()

    run_evaluation(
        manifest_csv=Path(args.manifest),
        results_csv=Path(args.output),
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        crop_pad=args.crop_pad,
    )
