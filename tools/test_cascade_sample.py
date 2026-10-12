"""
Test script for vehicle detector + nested plate detector on a single image.
"""
import sys
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detector.pi_yolo_detector import PlateDetector
from tools.evaluate_e2e_pipeline import calculate_iou

def test_single_cascade():
    v_model = YOLO('yolo11s.pt')
    vehicle_classes = [cls_id for cls_id, name in v_model.names.items() if name in ('car', 'motorcycle', 'bus', 'truck')]

    p_detector = PlateDetector('weights/best_license_plate_yolo11n_416.onnx', imgsz=416, conf=0.50, iou=0.45)

    img_path = 'datasets/VNLP_detection/detection/one_row/5_3285_1_36C04519_108_207_527_293.jpg'
    img = cv2.imread(img_path)
    ih, iw = img.shape[:2]
    gt_bbox = [108, 207, 527, 293]

    v_res = v_model.predict(img, imgsz=640, conf=0.25, classes=vehicle_classes, device='cuda', verbose=False)[0]
    print(f"Detected {len(v_res.boxes)} vehicle(s)")

    for i, box in enumerate(v_res.boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().item())
        cls_id = int(box.cls[0].cpu().item())
        cls_name = v_model.names[cls_id]
        vx1, vy1, vx2, vy2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        print(f"Vehicle {i+1}: {cls_name} (conf={conf:.2f}), box=[{vx1}, {vy1}, {vx2}, {vy2}]")

        vw = vx2 - vx1
        vh = vy2 - vy1
        pad_w = int(round(vw * 0.05))
        pad_h = int(round(vh * 0.05))
        rx1 = max(0, vx1 - pad_w)
        ry1 = max(0, vy1 - pad_h)
        rx2 = min(iw, vx2 + pad_w)
        ry2 = min(ih, vy2 + pad_h)

        roi = img[ry1:ry2, rx1:rx2]
        p_dets, _ = p_detector.detect(roi)
        print(f"  Nested plate detections inside ROI: {len(p_dets)}")
        for p_d in p_dets:
            px1, py1, px2, py2 = p_d['bbox']
            fx1 = rx1 + px1
            fy1 = ry1 + py1
            fx2 = rx1 + px2
            fy2 = ry1 + py2
            iou = calculate_iou([fx1, fy1, fx2, fy2], gt_bbox)
            print(f"  Mapped full bbox: [{fx1}, {fy1}, {fx2}, {fy2}] (conf={p_d['confidence']:.2f}, IoU={iou:.4f})")

if __name__ == "__main__":
    test_single_cascade()
