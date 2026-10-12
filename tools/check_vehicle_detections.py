"""
Check vehicle detection on first 20 samples of e2e_real_world_manifest.csv
"""
import csv
from pathlib import Path
import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def check_first_20():
    model = YOLO('yolo11s.pt')
    vehicle_classes = [cls_id for cls_id, name in model.names.items() if name in ('car', 'motorcycle', 'bus', 'truck')]

    with open(PROJECT_ROOT / 'reports/e2e_real_world_manifest.csv', 'r', encoding='utf-8') as f:
        samples = list(csv.DictReader(f))[:20]

    detected_v = 0
    for s in samples:
        img = cv2.imread(s['image_path'])
        res = model.predict(img, imgsz=640, conf=0.25, classes=vehicle_classes, device='cuda', verbose=False)[0]
        v_count = len(res.boxes)
        if v_count > 0:
            detected_v += 1
        max_c = max([float(b.conf[0].item()) for b in res.boxes], default=0.0)
        c_names = [model.names[int(b.cls[0].item())] for b in res.boxes]
        print(f"{s['image_path'].split('/')[-1][:30]} | {s['gt_layout']:<10} | Veh: {v_count} | Classes: {c_names} | MaxConf: {max_c:.3f}")

    print(f"\nTotal with vehicle detected at conf=0.25: {detected_v} / 20 ({detected_v/20*100:.1f}%)")

if __name__ == "__main__":
    check_first_20()
