"""
Accuracy, Small Plate, and Crop Quality Evaluation for YOLO11n ONNX models
across resolutions: 640, 512, 416, 320 using OpenCV DNN.
Evaluates on 100 test images with ground-truth YOLO labels.
"""

import sys
import os
import json
from pathlib import Path
import cv2
import numpy as np

# Import detection functions from pi_detect_image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from pi_detect_image import load_network, detect_image


def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def load_gt_boxes(label_path, orig_w, orig_h):
    """
    Reads YOLO format label file (class cx cy w h in normalized coordinates).
    Returns list of [x1, y1, x2, y2] in pixel coordinates.
    """
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # class_id, cx, cy, w, h
                cx = float(parts[1]) * orig_w
                cy = float(parts[2]) * orig_h
                bw = float(parts[3]) * orig_w
                bh = float(parts[4]) * orig_h
                x1 = max(0, int(round(cx - bw / 2.0)))
                y1 = max(0, int(round(cy - bh / 2.0)))
                x2 = min(orig_w, int(round(cx + bw / 2.0)))
                y2 = min(orig_h, int(round(cy + bh / 2.0)))
                boxes.append([x1, y1, x2, y2])
    return boxes


def is_small_plate(box):
    """
    Defines small plate: width <= 75 px or height <= 35 px or area <= 2200 px^2.
    Matches small plates like 28x18, 29x15, 43x23 etc.
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (w <= 75 or h <= 35 or (w * h) <= 2200)


def evaluate_dataset(
    test_img_dir: str,
    test_lbl_dir: str,
    models: list,
    num_images: int = 100,
    main_conf: float = 0.5,
    main_iou: float = 0.45,
    save_crops_limit: int = 25,
):
    img_dir = Path(test_img_dir)
    lbl_dir = Path(test_lbl_dir)

    all_images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    selected_images = all_images[:num_images]
    print(f"Loaded {len(selected_images)} test images for evaluation.")

    # Preload GT boxes
    gt_data = []
    total_gt = 0
    total_small_gt = 0

    for p in selected_images:
        lbl_p = lbl_dir / f"{p.stem}.txt"
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = load_gt_boxes(str(lbl_p), w, h)
        gt_data.append({"path": p, "w": w, "h": h, "gt_boxes": boxes})
        total_gt += len(boxes)
        for b in boxes:
            if is_small_plate(b):
                total_small_gt += 1

    print(f"Total Ground-Truth plates: {total_gt}")
    print(f"Total Small Ground-Truth plates: {total_small_gt}")

    summary_results = []
    small_plate_results = []

    # Also test 416 & 320 at conf=0.4 as required in step 10
    eval_configs = []
    for sz, m_path in models:
        eval_configs.append((sz, m_path, main_conf, f"benchmark_{sz}"))

    # Extra test for conf=0.4
    for sz, m_path in models:
        if sz in [416, 320]:
            eval_configs.append((sz, m_path, 0.4, f"benchmark_{sz}_conf04"))

    for sz, model_path, conf_t, save_folder in eval_configs:
        print(f"\n--- Evaluating {sz}x{sz} (conf={conf_t}) ---")
        net = load_network(model_path)

        crop_out_dir = Path("outputs") / save_folder / "plate_crops"
        det_out_dir = Path("outputs") / save_folder / "detections"
        crop_out_dir.mkdir(parents=True, exist_ok=True)
        det_out_dir.mkdir(parents=True, exist_ok=True)

        tp = 0
        fp = 0
        fn = 0
        total_dets = 0
        conf_list = []

        small_gt_count = 0
        small_detected_count = 0

        saved_crops = 0

        for item in gt_data:
            img_p = item["path"]
            gt_boxes = item["gt_boxes"]

            detections, annotated, crops, _ = detect_image(
                str(img_p),
                net=net,
                conf_thresh=conf_t,
                iou_thresh=main_iou,
                imgsz=sz,
            )

            total_dets += len(detections)
            for d in detections:
                conf_list.append(d["conf"])

            # Save 20-30 visual samples
            if saved_crops < save_crops_limit:
                cv2.imwrite(str(det_out_dir / f"{img_p.stem}_detected.jpg"), annotated)
                for idx, c in enumerate(crops, 1):
                    cv2.imwrite(str(crop_out_dir / f"{img_p.stem}_plate_{idx:02d}.jpg"), c)
                saved_crops += 1

            # Match detections with GT boxes
            matched_gt = set()
            for d in detections:
                d_box = d["box"]
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g_box in enumerate(gt_boxes):
                    if g_idx in matched_gt:
                        continue
                    iou = compute_iou(d_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= 0.5 and best_gt_idx >= 0:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn += (len(gt_boxes) - len(matched_gt))

            # Small plate tracking
            for g_idx, g_box in enumerate(gt_boxes):
                if is_small_plate(g_box):
                    small_gt_count += 1
                    if g_idx in matched_gt:
                        small_detected_count += 1

        recall = (tp / max(1, total_gt)) * 100
        avg_conf = float(np.mean(conf_list)) if conf_list else 0.0

        small_missed = small_gt_count - small_detected_count
        small_recall = (small_detected_count / max(1, small_gt_count)) * 100

        res_entry = {
            "imgsz": sz,
            "conf": conf_t,
            "folder": save_folder,
            "images": len(gt_data),
            "gt_plates": total_gt,
            "detections": total_dets,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "recall": round(recall, 2),
            "avg_conf": round(avg_conf, 4),
        }
        summary_results.append(res_entry)

        small_entry = {
            "imgsz": sz,
            "conf": conf_t,
            "small_gt": small_gt_count,
            "detected": small_detected_count,
            "missed": small_missed,
            "recall": round(small_recall, 2),
        }
        small_plate_results.append(small_entry)

        print(f"Results for {sz}x{sz} (conf={conf_t}):")
        print(f"  GT: {total_gt} | Det: {total_dets} | TP: {tp} | FN: {fn} | FP: {fp} | Recall: {recall:.2f}% | Avg Conf: {avg_conf:.4f}")
        print(f"  Small Plates: GT: {small_gt_count} | Detected: {small_detected_count} | Missed: {small_missed} | Recall: {small_recall:.2f}%")

    # Output formatted tables
    print("\n" + "=" * 70)
    print("DETECTION QUALITY (Main Benchmark at conf=0.5):")
    print(f"{'imgsz':>5} | {'GT plates':>9} | {'TP':>4} | {'FN':>4} | {'FP':>4} | {'Recall':>7} | {'Avg Conf':>8}")
    print("-" * 60)
    for r in summary_results:
        if r["conf"] == 0.5:
            print(f"{r['imgsz']:>5} | {r['gt_plates']:>9} | {r['tp']:>4} | {r['fn']:>4} | {r['fp']:>4} | {r['recall']:>6.2f}% | {r['avg_conf']:>8.4f}")

    print("\n" + "=" * 70)
    print("SMALL PLATE QUALITY (Main Benchmark at conf=0.5):")
    print(f"{'imgsz':>5} | {'Small GT':>8} | {'Detected':>8} | {'Missed':>6} | {'Recall':>7}")
    print("-" * 50)
    for s in small_plate_results:
        if s["conf"] == 0.5:
            print(f"{s['imgsz']:>5} | {s['small_gt']:>8} | {s['detected']:>8} | {s['missed']:>6} | {s['recall']:>6.2f}%")

    print("\n" + "=" * 70)
    print("CONFIDENCE THRESHOLD COMPARISON (conf=0.5 vs conf=0.4 for 416 & 320):")
    for r in summary_results:
        if r["imgsz"] in [416, 320]:
            print(f"Size {r['imgsz']} (conf={r['conf']}): Recall={r['recall']:.2f}%, FP={r['fp']}, TP={r['tp']}")

    # Save to JSON
    with open("outputs/detection_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary_results, "small_plate": small_plate_results}, f, indent=2)


if __name__ == "__main__":
    test_img = "vietnamese license plate.v1i.yolov11/test/images"
    test_lbl = "vietnamese license plate.v1i.yolov11/test/labels"
    models_list = [
        (640, "weights/best_license_plate_yolo11n_640.onnx"),
        (512, "weights/best_license_plate_yolo11n_512.onnx"),
        (416, "weights/best_license_plate_yolo11n_416.onnx"),
        (320, "weights/best_license_plate_yolo11n_320.onnx"),
    ]
    evaluate_dataset(test_img, test_lbl, models_list, num_images=100)
