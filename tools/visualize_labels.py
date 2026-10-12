"""
visualize_labels.py - Vẽ bounding box lên ảnh để kiểm tra annotation bằng mắt.

Sử dụng:
    python tools/visualize_labels.py

Chọn ngẫu nhiên ~20 ảnh từ dataset, vẽ bounding box và text "license_plate",
lưu kết quả vào outputs/dataset_preview/. Không thay đổi ảnh gốc.
"""

import os
import sys
import random
from pathlib import Path

import cv2
import numpy as np

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_ROOT = PROJECT_ROOT / "vietnamese license plate.v1i.yolov11"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_preview"
NUM_SAMPLES = 20
RANDOM_SEED = 42

# Màu sắc bounding box (BGR)
BOX_COLOR = (0, 255, 0)  # Xanh lá
TEXT_COLOR = (255, 255, 255)  # Trắng
TEXT_BG_COLOR = (0, 180, 0)  # Xanh lá đậm


def draw_boxes_on_image(img, boxes, img_w, img_h):
    """Vẽ bounding box YOLO lên ảnh."""
    for box in boxes:
        cls_id, x_c, y_c, w, h = box

        # Chuyển từ normalized sang pixel
        x1 = int((x_c - w / 2) * img_w)
        y1 = int((y_c - h / 2) * img_h)
        x2 = int((x_c + w / 2) * img_w)
        y2 = int((y_c + h / 2) * img_h)

        # Vẽ bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Vẽ text "license_plate"
        label = "license_plate"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Background cho text
        cv2.rectangle(img, (x1, y1 - text_h - baseline - 4), (x1 + text_w + 4, y1), TEXT_BG_COLOR, -1)
        cv2.putText(img, label, (x1 + 2, y1 - baseline - 2), font, font_scale, TEXT_COLOR, thickness)

    return img


def parse_yolo_label(lbl_path):
    """Đọc file label YOLO."""
    boxes = []
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    boxes.append((cls_id, x_c, y_c, w, h))
    except Exception as e:
        print(f"  [LỖI] Không đọc được label: {lbl_path} — {e}")
    return boxes


def main():
    print("=" * 60)
    print("  VISUALIZE LABELS — Vietnamese License Plate Dataset")
    print("=" * 60)

    # Tạo thư mục output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Thu thập tất cả cặp ảnh-label
    all_pairs = []
    for split in ["train", "valid", "test"]:
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if lbl_file.exists():
                    all_pairs.append((split, img_file, lbl_file))

    print(f"\nTổng số cặp ảnh-label: {len(all_pairs)}")

    # Chọn ngẫu nhiên
    random.seed(RANDOM_SEED)
    samples = random.sample(all_pairs, min(NUM_SAMPLES, len(all_pairs)))
    print(f"Chọn {len(samples)} mẫu ngẫu nhiên để visualize.\n")

    for i, (split, img_path, lbl_path) in enumerate(samples):
        # Đọc ảnh
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{i + 1}] Không đọc được: {img_path.name}")
            continue

        img_h, img_w = img.shape[:2]

        # Đọc label
        boxes = parse_yolo_label(lbl_path)

        # Vẽ bounding box
        img_annotated = draw_boxes_on_image(img.copy(), boxes, img_w, img_h)

        # Thêm thông tin split và tên file
        info_text = f"[{split}] {img_path.name} | {len(boxes)} box(es)"
        cv2.putText(img_annotated, info_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Lưu kết quả
        out_name = f"{split}_{img_path.stem}_preview.jpg"
        out_path = OUTPUT_DIR / out_name
        cv2.imwrite(str(out_path), img_annotated)
        print(f"  [{i + 1:2d}/{len(samples)}] {split}/{img_path.name} → {len(boxes)} box(es) → {out_name}")

    print(f"\n✓ Đã lưu {len(samples)} ảnh preview vào: {OUTPUT_DIR}")
    print("  Hãy mở thư mục trên để kiểm tra annotation bằng mắt.")
    print("=" * 60)


if __name__ == "__main__":
    main()
