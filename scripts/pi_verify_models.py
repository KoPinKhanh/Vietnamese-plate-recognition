"""
Verify that YOLO11n-416 and LPRNet ONNX models load cleanly into cv2.dnn on Raspberry Pi 5.
"""
import sys
import cv2

def verify():
    print(f"OpenCV Version: {cv2.__version__}")
    det_path = "/home/cytron/vietnam_plate/models/best_license_plate_yolo11n_416.onnx"
    ocr_path = "/home/cytron/vietnam_plate/models/best_lprnet_vn.onnx"

    print(f"Loading Detector from: {det_path}")
    try:
        net_det = cv2.dnn.readNetFromONNX(det_path)
        net_det.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net_det.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("  -> Detector loaded successfully! (Backend: OPENCV, Target: CPU)")
    except Exception as e:
        print(f"  -> FAIL: Could not load detector: {e}")
        return False

    print(f"Loading OCR from: {ocr_path}")
    try:
        net_ocr = cv2.dnn.readNetFromONNX(ocr_path)
        net_ocr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net_ocr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("  -> OCR loaded successfully! (Backend: OPENCV, Target: CPU)")
    except Exception as e:
        print(f"  -> FAIL: Could not load OCR: {e}")
        return False

    print("ALL MODELS VERIFIED AND READY.")
    return True

if __name__ == "__main__":
    ok = verify()
    if not ok:
        sys.exit(1)
