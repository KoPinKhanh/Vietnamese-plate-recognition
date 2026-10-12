"""
Test YOLO11n ONNX model with OpenCV DNN on laptop.
Confirms cv2.dnn compatibility without PyTorch inference.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def parse_args():
    parser = argparse.ArgumentParser(description="Test ONNX model with OpenCV DNN.")
    parser.add_argument(
        "--model",
        type=str,
        default="weights/best_license_plate_yolo11n_640.onnx",
        help="Path to ONNX model file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="vietnamese license plate.v1i.yolov11/test/images/CarLongPlate857_jpg.rf.37f06c27051958c8ba23251aca65a87e.jpg",
        help="Path to test image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)

    print("=" * 50)
    print("      OPENCV DNN ONNX TEST (LAPTOP)      ")
    print("=" * 50)
    print(f"OpenCV version   : {cv2.__version__}")
    print(f"Model path       : {model_path}")
    print(f"Image path       : {image_path}")

    if not model_path.exists():
        print(f"Error: Model file '{model_path}' not found.")
        print("Load status      : FAIL")
        sys.exit(1)

    # 1. Load model with OpenCV DNN
    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
        print("Load status      : PASS")
    except Exception as e:
        print(f"Load status      : FAIL - {e}")
        sys.exit(1)

    # 2. Read and prepare test image
    if not image_path.exists():
        # Fallback to creating a dummy test image if specific test image is missing
        print(f"Warning: Image '{image_path}' not found, creating dummy 640x640 frame.")
        img = np.zeros((640, 640, 3), dtype=np.uint8)
    else:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Unable to read image '{image_path}'.")
            sys.exit(1)

    orig_h, orig_w = img.shape[:2]
    padded_img, ratio, (pad_w, pad_h) = letterbox(img, new_shape=(640, 640))

    # Convert to blob: 1 x 3 x 640 x 640, scaled by 1/255.0, swapRB (BGR->RGB)
    blob = cv2.dnn.blobFromImage(
        padded_img, scalefactor=1.0 / 255.0, size=(640, 640), swapRB=True, crop=False
    )
    print(f"Input shape      : {blob.shape}")

    # 3. Inference
    try:
        net.setInput(blob)
        t0 = time.time()
        output = net.forward()
        latency_ms = (time.time() - t0) * 1000.0
        print(f"Output shape     : {output.shape}")
        print(f"Inference latency: {latency_ms:.2f} ms")
        print("Inference status : PASS")

        # Basic sanity check on output tensor
        print(f"Output min/max   : {output.min():.4f} / {output.max():.4f}")
        print("=" * 50)
        print("OpenCV DNN successfully loaded and executed the ONNX model!")
    except Exception as e:
        print(f"Inference status : FAIL - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
