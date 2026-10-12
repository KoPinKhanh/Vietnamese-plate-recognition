"""
check_dataset.py - Kiểm tra toàn diện dataset YOLO trước khi training.

Sử dụng:
    python tools/check_dataset.py

Script kiểm tra:
- Cặp ảnh-label (thiếu ảnh hoặc thiếu label)
- Label rỗng
- Format YOLO (5 trường: class_id x_center y_center width height)
- Class ID hợp lệ
- Tọa độ normalized [0, 1]
- Width/Height > 0
- Bounding box vượt biên
- Ảnh bị lỗi (corrupt)
- Thống kê kích thước ảnh và bounding box

Không tự động xóa dữ liệu. Chỉ báo cáo.
"""

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Tìm dataset root
DATASET_ROOT = PROJECT_ROOT / "vietnamese license plate.v1i.yolov11"
SPLITS = ["train", "valid", "test"]
VALID_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_NC = 1  # Số class mong đợi


def find_image_label_pairs(split_dir):
    """Tìm tất cả cặp ảnh-label trong một split."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        return {}, {}, img_dir, lbl_dir

    # Lấy danh sách file
    img_files = {}
    for f in img_dir.iterdir():
        if f.suffix.lower() in VALID_IMG_EXTENSIONS:
            img_files[f.stem] = f

    lbl_files = {}
    for f in lbl_dir.iterdir():
        if f.suffix.lower() == ".txt":
            lbl_files[f.stem] = f

    return img_files, lbl_files, img_dir, lbl_dir


def check_image_integrity(img_path):
    """Kiểm tra ảnh có đọc được không."""
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            im.verify()
        # Đọc lại để lấy thông tin
        with Image.open(img_path) as im:
            return True, im.size, im.format
    except Exception as e:
        return False, None, str(e)


def check_label_file(lbl_path, expected_nc):
    """Kiểm tra nội dung file label YOLO."""
    errors = []
    boxes = []

    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        errors.append(f"Không đọc được file: {e}")
        return errors, boxes, True

    # File rỗng
    stripped_lines = [line.strip() for line in lines if line.strip()]
    if len(stripped_lines) == 0:
        return errors, boxes, True  # empty=True

    for line_idx, line in enumerate(stripped_lines):
        parts = line.split()

        # Kiểm tra số trường
        if len(parts) != 5:
            errors.append(
                f"Dòng {line_idx + 1}: Cần 5 trường, có {len(parts)} — '{line}'"
            )
            continue

        # Parse giá trị
        try:
            cls_id = int(parts[0])
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            errors.append(f"Dòng {line_idx + 1}: Lỗi parse số — '{line}'")
            continue

        # Kiểm tra class ID
        if cls_id < 0 or cls_id >= expected_nc:
            errors.append(
                f"Dòng {line_idx + 1}: Class ID {cls_id} ngoài [0, {expected_nc - 1}]"
            )

        # Kiểm tra width/height <= 0
        if w <= 0:
            errors.append(f"Dòng {line_idx + 1}: width={w} <= 0")
        if h <= 0:
            errors.append(f"Dòng {line_idx + 1}: height={h} <= 0")

        # Kiểm tra tọa độ ngoài [0, 1]
        for name, val in [("x_center", x_c), ("y_center", y_c), ("width", w), ("height", h)]:
            if val < 0 or val > 1:
                errors.append(
                    f"Dòng {line_idx + 1}: {name}={val:.6f} ngoài [0, 1]"
                )

        # Kiểm tra bounding box vượt biên
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        if x1 < -0.001 or y1 < -0.001 or x2 > 1.001 or y2 > 1.001:
            errors.append(
                f"Dòng {line_idx + 1}: Box vượt biên — "
                f"x1={x1:.4f}, y1={y1:.4f}, x2={x2:.4f}, y2={y2:.4f}"
            )

        boxes.append({
            "class_id": cls_id,
            "x_center": x_c,
            "y_center": y_c,
            "width": w,
            "height": h,
        })

    return errors, boxes, False


def main():
    print("=" * 70)
    print("  KIỂM TRA DATASET YOLO — Vietnamese License Plate Detection")
    print("=" * 70)

    # Kiểm tra dataset tồn tại
    if not DATASET_ROOT.exists():
        print(f"\n[LỖI] Không tìm thấy dataset tại: {DATASET_ROOT}")
        sys.exit(1)

    print(f"\nDataset root: {DATASET_ROOT}")

    # Kiểm tra data.yaml gốc
    original_yaml = DATASET_ROOT / "data.yaml"
    if original_yaml.exists():
        print(f"data.yaml gốc: {original_yaml}")
        with open(original_yaml, "r", encoding="utf-8") as f:
            print("  Nội dung:")
            for line in f:
                print(f"    {line.rstrip()}")

    # Thống kê tổng
    total_images = 0
    total_labels = 0
    total_boxes = 0
    total_errors = 0
    all_classes = Counter()
    all_box_widths = []
    all_box_heights = []
    all_box_areas = []
    all_img_widths = []
    all_img_heights = []

    for split in SPLITS:
        split_dir = DATASET_ROOT / split
        print(f"\n{'─' * 60}")
        print(f"  SPLIT: {split.upper()}")
        print(f"{'─' * 60}")

        if not split_dir.exists():
            print(f"  [CẢNH BÁO] Thư mục '{split}' không tồn tại — bỏ qua.")
            continue

        img_files, lbl_files, img_dir, lbl_dir = find_image_label_pairs(split_dir)

        if not img_dir.exists():
            print(f"  [LỖI] Thư mục images không tồn tại: {img_dir}")
            continue
        if not lbl_dir.exists():
            print(f"  [LỖI] Thư mục labels không tồn tại: {lbl_dir}")
            continue

        num_images = len(img_files)
        num_labels = len(lbl_files)
        total_images += num_images
        total_labels += num_labels

        print(f"  Số ảnh:  {num_images}")
        print(f"  Số label: {num_labels}")

        # Ảnh không có label
        imgs_no_lbl = sorted(set(img_files.keys()) - set(lbl_files.keys()))
        if imgs_no_lbl:
            print(f"\n  [CẢNH BÁO] {len(imgs_no_lbl)} ảnh KHÔNG có label:")
            for name in imgs_no_lbl[:10]:
                print(f"    - {img_files[name].name}")
            if len(imgs_no_lbl) > 10:
                print(f"    ... và {len(imgs_no_lbl) - 10} ảnh khác")
            total_errors += len(imgs_no_lbl)
        else:
            print(f"  ✓ Tất cả ảnh đều có label")

        # Label không có ảnh
        lbls_no_img = sorted(set(lbl_files.keys()) - set(img_files.keys()))
        if lbls_no_img:
            print(f"\n  [CẢNH BÁO] {len(lbls_no_img)} label KHÔNG có ảnh:")
            for name in lbls_no_img[:10]:
                print(f"    - {lbl_files[name].name}")
            if len(lbls_no_img) > 10:
                print(f"    ... và {len(lbls_no_img) - 10} label khác")
            total_errors += len(lbls_no_img)
        else:
            print(f"  ✓ Tất cả label đều có ảnh tương ứng")

        # Kiểm tra ảnh corrupt
        corrupt_count = 0
        split_img_widths = []
        split_img_heights = []
        print(f"\n  Đang kiểm tra {num_images} ảnh...")
        for stem, img_path in img_files.items():
            ok, size, fmt = check_image_integrity(img_path)
            if not ok:
                corrupt_count += 1
                print(f"    [LỖI] Ảnh corrupt: {img_path.name} — {fmt}")
            else:
                split_img_widths.append(size[0])
                split_img_heights.append(size[1])
                all_img_widths.append(size[0])
                all_img_heights.append(size[1])

        if corrupt_count > 0:
            print(f"  [LỖI] {corrupt_count} ảnh bị corrupt!")
            total_errors += corrupt_count
        else:
            print(f"  ✓ Không có ảnh bị corrupt")

        # Thống kê kích thước ảnh
        if split_img_widths:
            print(f"\n  Kích thước ảnh:")
            print(f"    Width:  min={min(split_img_widths)}, max={max(split_img_widths)}, "
                  f"avg={sum(split_img_widths) / len(split_img_widths):.0f}")
            print(f"    Height: min={min(split_img_heights)}, max={max(split_img_heights)}, "
                  f"avg={sum(split_img_heights) / len(split_img_heights):.0f}")

        # Kiểm tra label
        empty_labels = 0
        split_boxes = 0
        split_label_errors = 0
        split_classes = Counter()
        split_box_widths = []
        split_box_heights = []

        for stem, lbl_path in lbl_files.items():
            errors, boxes, is_empty = check_label_file(lbl_path, EXPECTED_NC)

            if is_empty:
                empty_labels += 1

            if errors:
                split_label_errors += len(errors)
                for err in errors[:3]:  # In tối đa 3 lỗi mỗi file
                    print(f"    [LỖI LABEL] {lbl_path.name}: {err}")

            for box in boxes:
                split_classes[box["class_id"]] += 1
                split_box_widths.append(box["width"])
                split_box_heights.append(box["height"])
                all_box_widths.append(box["width"])
                all_box_heights.append(box["height"])
                all_box_areas.append(box["width"] * box["height"])

            split_boxes += len(boxes)
            all_classes.update({box["class_id"]: 1 for box in boxes} if boxes else {})

        for box_cls_id in split_classes:
            all_classes[box_cls_id] += split_classes[box_cls_id]
        # Sửa lại: đếm tổng chính xác
        all_classes = Counter()  # Reset, sẽ tính lại ở cuối

        total_boxes += split_boxes
        total_errors += split_label_errors

        print(f"\n  Label rỗng (background): {empty_labels}")
        print(f"  Lỗi trong label: {split_label_errors}")
        print(f"  Tổng bounding box: {split_boxes}")
        print(f"  Trung bình box/ảnh: {split_boxes / max(1, num_images):.2f}")
        print(f"  Phân bố class: {dict(split_classes)}")

        # Thống kê bounding box
        if split_box_widths:
            print(f"\n  Phân bố bounding box (normalized):")
            print(f"    Width:  min={min(split_box_widths):.4f}, max={max(split_box_widths):.4f}, "
                  f"avg={sum(split_box_widths) / len(split_box_widths):.4f}")
            print(f"    Height: min={min(split_box_heights):.4f}, max={max(split_box_heights):.4f}, "
                  f"avg={sum(split_box_heights) / len(split_box_heights):.4f}")
            print(f"    Area:   min={min(w * h for w, h in zip(split_box_widths, split_box_heights)):.6f}, "
                  f"max={max(w * h for w, h in zip(split_box_widths, split_box_heights)):.4f}, "
                  f"avg={sum(w * h for w, h in zip(split_box_widths, split_box_heights)) / len(split_box_widths):.4f}")

    # Tổng kết
    print(f"\n{'=' * 70}")
    print(f"  TỔNG KẾT")
    print(f"{'=' * 70}")
    print(f"  Dataset root:     {DATASET_ROOT}")
    print(f"  Tổng số ảnh:      {total_images}")
    print(f"  Tổng số label:    {total_labels}")
    print(f"  Tổng bounding box: {total_boxes}")
    print(f"  Tổng lỗi:         {total_errors}")
    print(f"  Format:           YOLO (class_id x_center y_center width height)")

    if all_img_widths:
        print(f"\n  Kích thước ảnh tổng hợp:")
        print(f"    Width:  min={min(all_img_widths)}, max={max(all_img_widths)}")
        print(f"    Height: min={min(all_img_heights)}, max={max(all_img_heights)}")

    if all_box_widths:
        print(f"\n  Phân bố bounding box tổng hợp (normalized):")
        print(f"    Width:  min={min(all_box_widths):.4f}, max={max(all_box_widths):.4f}, "
              f"avg={sum(all_box_widths) / len(all_box_widths):.4f}")
        print(f"    Height: min={min(all_box_heights):.4f}, max={max(all_box_heights):.4f}, "
              f"avg={sum(all_box_heights) / len(all_box_heights):.4f}")
        print(f"    Area:   min={min(all_box_areas):.6f}, max={max(all_box_areas):.4f}, "
              f"avg={sum(all_box_areas) / len(all_box_areas):.4f}")

    if total_errors == 0:
        print(f"\n  ✓ Dataset SẠCH — sẵn sàng training!")
    else:
        print(f"\n  ⚠ Có {total_errors} vấn đề cần xem xét.")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
