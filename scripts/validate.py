"""
validate.py - Validate model YOLO đã train trên tập validation.

Sử dụng:
    python scripts/validate.py
    python scripts/validate.py --weights path/to/best.pt
    python scripts/validate.py --split test

In ra: Precision, Recall, mAP50, mAP50-95.
"""

import argparse
import sys
from pathlib import Path

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WEIGHTS = PROJECT_ROOT / "outputs" / "train" / "vietnam_plate_yolo11n" / "weights" / "best.pt"
DEFAULT_DATA = PROJECT_ROOT / "configs" / "license_plate.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLO model")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                        help=f"Đường dẫn model (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                        help="Dataset YAML path")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Kích thước ảnh (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 0 (GPU), cpu")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Split để validate (default: val)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("[LỖI] Chưa cài ultralytics hoặc torch.")
        sys.exit(1)

    # Kiểm tra weights
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[LỖI] Không tìm thấy model: {weights_path}")
        print(f"  Hãy chạy training trước: python scripts/train.py")
        sys.exit(1)

    print("=" * 60)
    print("  VALIDATE — Vietnamese License Plate Detection")
    print("=" * 60)
    print(f"  Model:   {weights_path}")
    print(f"  Data:    {args.data}")
    print(f"  Split:   {args.split}")
    print(f"  ImgSz:   {args.imgsz}")
    print(f"  Batch:   {args.batch}")

    # Device
    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"  Device:  {device}")
    print("=" * 60)

    # Load model và validate
    model = YOLO(str(weights_path))
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        split=args.split,
        verbose=True,
    )

    # In kết quả
    print("\n" + "=" * 60)
    print("  KẾT QUẢ VALIDATION")
    print("=" * 60)

    # Lấy metrics
    precision = results.box.mp  # Mean Precision
    recall = results.box.mr  # Mean Recall
    map50 = results.box.map50  # mAP@0.5
    map50_95 = results.box.map  # mAP@0.5:0.95

    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  mAP50:       {map50:.4f}")
    print(f"  mAP50-95:    {map50_95:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
