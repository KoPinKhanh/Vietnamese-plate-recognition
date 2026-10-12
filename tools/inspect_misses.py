"""
Inspect root causes of detection misses.
"""
import csv
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detector.pi_yolo_detector import PlateDetector


def inspect_misses():
    det_low = PlateDetector(str(PROJECT_ROOT / 'weights/best_license_plate_yolo11n_416.onnx'), imgsz=416, conf=0.10, iou=0.45)
    with open(PROJECT_ROOT / 'reports/e2e_real_world_results.csv', 'r', encoding='utf-8') as f:
        misses = [r for r in csv.DictReader(f) if r['failure_stage'] == 'DETECTION_MISS'][:10]

    for m in misses:
        img = cv2.imread(m['image_path'])
        dets, _ = det_low.detect(img)
        gt_bbox = eval(m['gt_bbox'])
        gt_w = gt_bbox[2] - gt_bbox[0]
        gt_h = gt_bbox[3] - gt_bbox[1]
        max_c = max([d['confidence'] for d in dets], default=0.0)
        print(f"{m['sample_id']} | Layout: {m['gt_layout']} | GT Size: {gt_w}x{gt_h} | LowConf Dets: {len(dets)} | Max Conf: {max_c:.3f}")


if __name__ == "__main__":
    inspect_misses()
