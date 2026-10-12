"""
predict_camera.py - Phát hiện biển số xe real-time từ USB Camera.

Sử dụng:
    python scripts/predict_camera.py
    python scripts/predict_camera.py --camera 0
    python scripts/predict_camera.py --conf 0.5

Nhấn 'q' để thoát.

Mục tiêu: chạy trên Raspberry Pi 5 + USB Camera.
Ở bước này chưa có OCR.
"""

import argparse
import sys
import time
from pathlib import Path

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WEIGHTS = PROJECT_ROOT / "outputs" / "train" / "vietnam_plate_yolo11n" / "weights" / "best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time license plate detection from USB camera")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
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
        import cv2
    except ImportError:
        print("[LỖI] Thiếu package. Chạy: pip install ultralytics opencv-python")
        sys.exit(1)

    # Kiểm tra weights
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[LỖI] Không tìm thấy model: {weights_path}")
        print(f"  Hãy chạy training trước: python scripts/train.py")
        sys.exit(1)

    print("=" * 60)
    print("  USB CAMERA — Vietnamese License Plate Detection")
    print("=" * 60)
    print(f"  Model:      {weights_path}")
    print(f"  Camera:     index {args.camera}")
    print(f"  Confidence: {args.conf}")
    print(f"  Nhấn 'q' để thoát")
    print("=" * 60)

    # Device
    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model
    model = YOLO(str(weights_path))

    # Mở camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[LỖI] Không mở được camera index {args.camera}")
        print("  Kiểm tra USB Camera đã kết nối chưa.")
        sys.exit(1)

    print(f"\n→ Camera đã mở. Đang detect biển số...")
    print(f"  Nhấn 'q' trên cửa sổ hiển thị để thoát.\n")

    frame_count = 0
    fps_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CẢNH BÁO] Không đọc được frame từ camera.")
                break

            frame_count += 1

            # Predict
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                verbose=False,
            )

            # Vẽ kết quả lên frame
            annotated_frame = results[0].plot()

            # Tính FPS
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Thêm thông tin FPS và số detection
            num_detections = len(results[0].boxes)
            info_text = f"FPS: {fps:.1f} | Plates: {num_detections}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Hiển thị confidence cho từng detection
            for box in results[0].boxes:
                conf = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = f"license_plate {conf:.2f}"
                cv2.putText(annotated_frame, label,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Hiển thị
            cv2.imshow("License Plate Detection - Press 'q' to quit", annotated_frame)

            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n→ Nhấn 'q' — đang thoát...")
                break

    except KeyboardInterrupt:
        print("\n→ Ctrl+C — đang thoát...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Đã xử lý {frame_count} frames")
        print(f"  FPS trung bình: {frame_count / max(0.001, time.time() - fps_start):.1f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
