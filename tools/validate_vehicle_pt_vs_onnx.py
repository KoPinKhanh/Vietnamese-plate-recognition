"""
Validate PyTorch YOLO11s vs ONNX Runtime CPU YOLO11s on 100 full-scene images.
Input models:
  - PyTorch: yolo11s.pt
  - ONNX: weights/deployment/pi_cpu/yolo11s_640_fp32.onnx (CPUExecutionProvider)
Manifest: reports/fresh_holdout_500_manifest.csv (first 100 samples)
Evaluates:
  - Raw tensor max/mean abs diff
  - Detection count match
  - Class agreement
  - BBox IoU (PT vs ONNX)
  - Confidence absolute differences
Outputs:
  - Detailed console report
  - Appends to reports/pi_cpu_onnx_equivalence_results.csv
"""

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms, ops

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def calculate_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def main():
    print("=" * 80)
    print("VEHICLE DETECTOR PYTORCH VS ONNX RUNTIME CPU EQUIVALENCE VALIDATION")
    print("=" * 80)

    pt_model_path = PROJECT_ROOT / "yolo11s.pt"
    onnx_model_path = PROJECT_ROOT / "weights/deployment/pi_cpu/yolo11s_640_fp32.onnx"
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

    print(f"Loaded {len(samples)} evaluation samples from {manifest_path.name}")

    # Load PyTorch model on CPU
    print(f"Loading PyTorch YOLO11s on CPU: {pt_model_path}")
    pt_yolo = YOLO(str(pt_model_path))
    pt_model = pt_yolo.model.cpu().eval()

    # Relevant vehicle classes in COCO: car=2, motorcycle=3, bus=5, truck=7
    coco_names = pt_yolo.names
    relevant_classes = {cls_id for cls_id, name in coco_names.items() if name in ("car", "motorcycle", "bus", "truck")}
    print(f"Relevant vehicle class IDs: {relevant_classes} ({[coco_names[c] for c in sorted(relevant_classes)]})")

    # Load ONNX Runtime session on CPU
    print(f"Initializing ONNX Runtime CPU session: {onnx_model_path}")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_model_path), sess_options, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    letterbox = LetterBox(640, auto=False, stride=32)

    raw_mean_diffs = []
    raw_max_diffs = []

    matched_ious_all = []
    matched_conf_diffs_all = []
    matched_class_agreements_all = []

    matched_ious_rel = []
    matched_conf_diffs_rel = []
    matched_class_agreements_rel = []

    identical_all_count = 0
    identical_relevant_count = 0

    results_records = []

    for idx, sample in enumerate(samples, 1):
        sample_id = sample["sample_id"]
        img_path = sample["image_path"]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # Preprocessing (Letterbox 640x640, BGR->RGB, float32 0..1)
        img_lb = letterbox(image=img)
        blob = img_lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)
        pt_input = torch.from_numpy(blob)

        # 1. PyTorch CPU forward
        t0_pt = time.perf_counter()
        with torch.no_grad():
            pt_raw = pt_model(pt_input)[0]  # shape [1, 84, 8400]
        t_pt_ms = (time.perf_counter() - t0_pt) * 1000.0

        # 2. ONNX CPU forward
        t0_onnx = time.perf_counter()
        onnx_raw = sess.run(None, {input_name: blob})[0]  # shape [1, 84, 8400]
        t_onnx_ms = (time.perf_counter() - t0_onnx) * 1000.0

        # Raw tensor diff
        diff = np.abs(pt_raw.numpy() - onnx_raw)
        m_diff = float(diff.mean())
        max_d = float(diff.max())
        raw_mean_diffs.append(m_diff)
        raw_max_diffs.append(max_d)

        # 3. Detection Postprocessing (NMS conf=0.25, iou=0.45)
        pt_dets = nms.non_max_suppression(pt_raw, conf_thres=0.25, iou_thres=0.45)[0]
        onnx_dets = nms.non_max_suppression(torch.from_numpy(onnx_raw), conf_thres=0.25, iou_thres=0.45)[0]

        pt_boxes = ops.scale_boxes(img_lb.shape[:2], pt_dets[:, :4], img.shape[:2]).cpu().numpy() if len(pt_dets) > 0 else np.empty((0, 4))
        onnx_boxes = ops.scale_boxes(img_lb.shape[:2], onnx_dets[:, :4], img.shape[:2]).cpu().numpy() if len(onnx_dets) > 0 else np.empty((0, 4))

        pt_confs = pt_dets[:, 4].cpu().numpy() if len(pt_dets) > 0 else np.empty(0)
        onnx_confs = onnx_dets[:, 4].cpu().numpy() if len(onnx_dets) > 0 else np.empty(0)

        pt_classes = pt_dets[:, 5].cpu().numpy().astype(int) if len(pt_dets) > 0 else np.empty(0, dtype=int)
        onnx_classes = onnx_dets[:, 5].cpu().numpy().astype(int) if len(onnx_dets) > 0 else np.empty(0, dtype=int)

        # Count relevant
        pt_rel_mask = [c in relevant_classes for c in pt_classes]
        onnx_rel_mask = [c in relevant_classes for c in onnx_classes]

        pt_rel_cnt = int(np.sum(pt_rel_mask))
        onnx_rel_cnt = int(np.sum(onnx_rel_mask))

        if len(pt_dets) == len(onnx_dets):
            identical_all_count += 1
        if pt_rel_cnt == onnx_rel_cnt:
            identical_relevant_count += 1

        # Match predictions greedily between PT and ONNX
        # For each PT detection, find closest ONNX detection
        unmatched_onnx = set(range(len(onnx_dets)))
        sample_ious = []
        sample_conf_diffs = []
        sample_class_agrees = []

        for p_i in range(len(pt_dets)):
            p_box = pt_boxes[p_i]
            p_conf = pt_confs[p_i]
            p_cls = pt_classes[p_i]

            best_iou = -1.0
            best_o = -1
            for o_i in unmatched_onnx:
                iou = calculate_box_iou(p_box, onnx_boxes[o_i])
                if iou > best_iou:
                    best_iou = iou
                    best_o = o_i

            if best_o >= 0 and best_iou >= 0.5:
                unmatched_onnx.remove(best_o)
                o_box = onnx_boxes[best_o]
                o_conf = onnx_confs[best_o]
                o_cls = onnx_classes[best_o]

                c_agree = (p_cls == o_cls)
                c_diff = abs(float(p_conf - o_conf))

                matched_ious_all.append(best_iou)
                matched_conf_diffs_all.append(c_diff)
                matched_class_agreements_all.append(c_agree)

                sample_ious.append(best_iou)
                sample_conf_diffs.append(c_diff)
                sample_class_agrees.append(c_agree)

                if p_cls in relevant_classes or o_cls in relevant_classes:
                    matched_ious_rel.append(best_iou)
                    matched_conf_diffs_rel.append(c_diff)
                    matched_class_agreements_rel.append(c_agree)

        # Summary for this sample
        avg_iou = float(np.mean(sample_ious)) if sample_ious else 1.0
        avg_conf_d = float(np.mean(sample_conf_diffs)) if sample_conf_diffs else 0.0
        all_class_ok = all(sample_class_agrees) if sample_class_agrees else True
        det_cnt_ok = (len(pt_dets) == len(onnx_dets))
        pred_match = det_cnt_ok and all_class_ok and (avg_iou >= 0.99)

        results_records.append({
            "sample_id": sample_id,
            "model": "yolo11s_vehicle",
            "image_path": img_path,
            "ground_truth": f"N/A (Full scene vehicles: {pt_rel_cnt})",
            "pytorch_prediction": f"dets={len(pt_dets)} rel={pt_rel_cnt}",
            "onnx_prediction": f"dets={len(onnx_dets)} rel={onnx_rel_cnt}",
            "prediction_match": str(pred_match),
            "mean_abs_diff": f"{m_diff:.6e}",
            "max_abs_diff": f"{max_d:.6e}",
            "bbox_iou_pt_vs_onnx": f"{avg_iou:.4f}",
            "confidence_diff": f"{avg_conf_d:.6e}",
            "latency_ms": f"{t_onnx_ms:.2f}",
        })

    # Summary Statistics
    total_samples = len(samples)
    print("\n" + "=" * 80)
    print("VEHICLE DETECTOR EQUIVALENCE SUMMARY (100 FULL-SCENE SAMPLES)")
    print("=" * 80)
    print(f"Total Images Evaluated                 : {total_samples}")
    print(f"Images with Identical ALL Detection Cnt : {identical_all_count} / {total_samples} ({identical_all_count/total_samples*100:.1f}%)")
    print(f"Images with Identical RELEVANT Cnt     : {identical_relevant_count} / {total_samples} ({identical_relevant_count/total_samples*100:.1f}%)")
    print(f"Raw Output Tensor Mean Abs Diff         : {np.mean(raw_mean_diffs):.6e}")
    print(f"Raw Output Tensor Max Abs Diff          : {np.max(raw_max_diffs):.6e}")
    print("-" * 80)
    print("ALL DETECTIONS (COCO 80 classes):")
    print(f"  Total Matched Detections             : {len(matched_ious_all)}")
    print(f"  Class Agreement Rate                 : {np.mean(matched_class_agreements_all)*100:.2f}% ({sum(matched_class_agreements_all)}/{len(matched_class_agreements_all)})")
    print(f"  Mean BBox IoU (PT vs ONNX)           : {np.mean(matched_ious_all):.6f}")
    print(f"  Min BBox IoU (PT vs ONNX)            : {np.min(matched_ious_all):.6f}")
    print(f"  Mean Confidence Abs Diff             : {np.mean(matched_conf_diffs_all):.6e}")
    print(f"  Max Confidence Abs Diff              : {np.max(matched_conf_diffs_all):.6e}")
    print("-" * 80)
    print("RELEVANT VEHICLES (car, motorcycle, bus, truck):")
    print(f"  Total Matched Vehicle Detections     : {len(matched_ious_rel)}")
    if matched_ious_rel:
        print(f"  Vehicle Class Agreement Rate         : {np.mean(matched_class_agreements_rel)*100:.2f}% ({sum(matched_class_agreements_rel)}/{len(matched_class_agreements_rel)})")
        print(f"  Mean Vehicle BBox IoU                : {np.mean(matched_ious_rel):.6f}")
        print(f"  Min Vehicle BBox IoU                 : {np.min(matched_ious_rel):.6f}")
        print(f"  Mean Vehicle Conf Abs Diff           : {np.mean(matched_conf_diffs_rel):.6e}")
        print(f"  Max Vehicle Conf Abs Diff            : {np.max(matched_conf_diffs_rel):.6e}")
    print("=" * 80)

    # Save to CSV
    csv_path = PROJECT_ROOT / "reports/pi_cpu_onnx_equivalence_results.csv"
    fieldnames = [
        "sample_id", "model", "image_path", "ground_truth",
        "pytorch_prediction", "onnx_prediction", "prediction_match",
        "mean_abs_diff", "max_abs_diff", "bbox_iou_pt_vs_onnx",
        "confidence_diff", "latency_ms"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_records)

    print(f"Wrote vehicle results to: {csv_path}")


if __name__ == "__main__":
    main()
