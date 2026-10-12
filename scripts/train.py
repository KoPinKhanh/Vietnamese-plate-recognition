"""
train.py - Train YOLO11n cho bài toán phát hiện biển số xe Việt Nam.

Sử dụng:
    python scripts/train.py                    # Train mặc định 100 epochs
    python scripts/train.py --epochs 3         # Smoke test
    python scripts/train.py --resume           # Resume training từ last.pt

Model: yolo11n.pt (pretrained)
Dataset: configs/license_plate.yaml
"""

import argparse
import sys
from pathlib import Path

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11n - Vietnamese License Plate Detection")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Pretrained model (default: yolo11n.pt)")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "configs" / "license_plate.yaml"),
                        help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Số epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Kích thước ảnh input (default: 640)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size. -1 = auto (default: -1)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 0 (GPU), cpu. Mặc định: tự động chọn")
    parser.add_argument("--workers", type=int, default=4,
                        help="Số dataloader workers (default: 4)")
    parser.add_argument("--project", type=str, default=str(PROJECT_ROOT / "outputs" / "train"),
                        help="Thư mục lưu output")
    parser.add_argument("--name", type=str, default="vietnam_plate_yolo11n",
                        help="Tên thí nghiệm")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training từ last.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    # Import ultralytics
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("[LỖI] Chưa cài ultralytics hoặc torch.")
        print("  Chạy: pip install ultralytics torch torchvision")
        sys.exit(1)

    # Hiển thị thông tin hệ thống
    print("=" * 60)
    print("  TRAIN YOLO11n — Vietnamese License Plate Detection")
    print("=" * 60)
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  CUDA:      {'Có' if torch.cuda.is_available() else 'Không'}")
    if torch.cuda.is_available():
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  CUDA ver:  {torch.version.cuda}")
    print(f"  Device:    {args.device if args.device else 'auto'}")
    print(f"  Model:     {args.model}")
    print(f"  Data:      {args.data}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch} {'(auto)' if args.batch == -1 else ''}")
    print(f"  ImgSz:     {args.imgsz}")
    print(f"  Patience:  {args.patience}")
    print(f"  Workers:   {args.workers}")
    print(f"  Output:    {args.project}/{args.name}")
    print("=" * 60)

    # Xác định device
    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Resume training
    if args.resume:
        last_pt = Path(args.project) / args.name / "weights" / "last.pt"
        if last_pt.exists():
            print(f"\n→ Resume training từ: {last_pt}")
            model = YOLO(str(last_pt))
            results = model.train(resume=True)
        else:
            print(f"\n[LỖI] Không tìm thấy last.pt tại: {last_pt}")
            sys.exit(1)
    else:
        # Load pretrained model
        model = YOLO(args.model)

        # Train
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            device=device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=True,
            pretrained=True,
            verbose=True,
        )

    # Kiểm tra best.pt
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n✓ Training hoàn thành!")
        print(f"  best.pt: {best_pt}")
        print(f"  Kích thước: {best_pt.stat().st_size / 1024**2:.1f} MB")
    else:
        print(f"\n⚠ Không tìm thấy best.pt tại: {best_pt}")

    print("=" * 60)


if __name__ == "__main__":
    main()
