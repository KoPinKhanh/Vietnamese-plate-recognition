"""
Raspberry Pi 5 Test & Benchmark Script for YOLO11n ONNX using OpenCV DNN.
Designed to run on the Pi without PyTorch or Ultralytics dependencies.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def parse_args():
    parser = argparse.ArgumentParser(description="Raspberry Pi OpenCV DNN ONNX Benchmark.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_license_plate_yolo11n_640.onnx",
        help="Path to ONNX model file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input test image.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of benchmark iterations (default: 20).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)

    print("=" * 55)
    print("      RASPBERRY PI 5 - OPENCV DNN ONNX BENCHMARK      ")
    print("=" * 55)
    print(f"OpenCV version   : {cv2.__version__}")
    print(f"Model path       : {model_path}")
    print(f"Input image      : {image_path}")

    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found!")
        sys.exit(1)

    if not image_path.exists():
        print(f"Error: Image file '{image_path}' not found!")
        sys.exit(1)

    # 1. Load model with OpenCV DNN
    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
        print("Model loaded successfully: YES")
    except Exception as e:
        print(f"Model loaded successfully: NO ({e})")
        sys.exit(1)

    # 2. Read and prepare input image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Unable to decode image '{image_path}'!")
        sys.exit(1)

    padded_img, ratio, (pad_w, pad_h) = letterbox(img, new_shape=(640, 640))
    blob = cv2.dnn.blobFromImage(
        padded_img, scalefactor=1.0 / 255.0, size=(640, 640), swapRB=True, crop=False
    )
    print(f"Blob shape       : {blob.shape}")

    # 3. Initial forward pass
    net.setInput(blob)
    t_start = time.time()
    out = net.forward()
    first_latency_ms = (time.time() - t_start) * 1000.0
    first_fps = 1000.0 / max(first_latency_ms, 1e-5)

    print(f"Output shape     : {out.shape}")
    print(f"Initial latency  : {first_latency_ms:.2f} ms")
    print(f"Initial raw FPS  : {first_fps:.2f}")

    # 4. Warm-up runs
    print(f"\nRunning {args.warmup} warmup iterations...")
    for _ in range(args.warmup):
        net.setInput(blob)
        _ = net.forward()

    # 5. Benchmark runs
    print(f"Running {args.runs} benchmark iterations...")
    latencies = []
    for _ in range(args.runs):
        net.setInput(blob)
        t0 = time.time()
        _ = net.forward()
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)

    avg_latency = float(np.mean(latencies))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))
    raw_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("=" * 55)
    print("BENCHMARK RESULTS:")
    print(f"Warmup           : {args.warmup}")
    print(f"Runs             : {args.runs}")
    print(f"Average inference: {avg_latency:.2f} ms")
    print(f"Min              : {min_latency:.2f} ms")
    print(f"Max              : {max_latency:.2f} ms")
    print(f"Raw inference FPS: {raw_fps:.2f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
