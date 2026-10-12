"""
predict_video.py - Phát hiện biển số xe trên video file.

Sử dụng:
    python scripts/predict_video.py --source path/to/video.mp4
    python scripts/predict_video.py --source path/to/video.mp4 --conf 0.5

Kết quả lưu vào: outputs/predictions/
"""

import argparse
import sys
from pathlib import Path

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WEIGHTS = PROJECT_ROOT / "outputs" / "train" / "vietnam_plate_yolo11n" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"


def parse_args():
    parser = argparse.ArgumentParser(description="Detect license plates on video")
    parser.add_argument("--source", type=str, required=True,
                        help="Đường dẫn video")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                        help=f"Đường dẫn model (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Kích thước ảnh (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Ngưỡng confidence (default: 0.25)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 0 (GPU), cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("[LỖI] Thiếu package. Chạy: pip install ultralytics opencv-python")
        sys.exit(1)

    # Kiểm tra weights
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[LỖI] Không tìm thấy model: {weights_path}")
        print(f"  Hãy chạy training trước: python scripts/train.py")
        sys.exit(1)

    # Kiểm tra source
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[LỖI] Không tìm thấy video: {args.source}")
        sys.exit(1)

    # Tạo output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PREDICT VIDEO — Vietnamese License Plate Detection")
    print("=" * 60)
    print(f"  Model:       {weights_path}")
    print(f"  Source:       {args.source}")
    print(f"  Confidence:  {args.conf}")
    print(f"  Output:      {OUTPUT_DIR}")
    print("=" * 60)

    # Device
    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model
    model = YOLO(str(weights_path))

    # Predict trên video
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        save=True,
        project=str(OUTPUT_DIR),
        name="videos",
        exist_ok=True,
        verbose=True,
        stream=True,  # Stream mode cho video lớn
    )

    # Xử lý stream results
    frame_count = 0
    total_detections = 0
    for result in results:
        frame_count += 1
        num_boxes = len(result.boxes)
        total_detections += num_boxes
        if frame_count % 100 == 0:
            print(f"  Frame {frame_count}: {num_boxes} biển số phát hiện")

    print(f"\n✓ Xử lý hoàn thành!")
    print(f"  Tổng frames: {frame_count}")
    print(f"  Tổng phát hiện: {total_detections}")
    print(f"  Trung bình: {total_detections / max(1, frame_count):.2f} biển số/frame")
    print(f"  Kết quả lưu tại: {OUTPUT_DIR / 'videos'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
