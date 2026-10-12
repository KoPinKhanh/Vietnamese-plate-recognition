"""
Smoke test script for Real Laptop Pipeline (YOLO11n + PARSeq-Tiny VNLP).
Runs end-to-end benchmark on 5 diverse full images (one_row and two_rows).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.laptop_parseq_recognize import RealLaptopPipeline


def run_smoke_test():
    pipeline = RealLaptopPipeline(
        detector_path="weights/best_license_plate_yolo11n_416.onnx",
        ocr_model="parseq",
        ocr_checkpoint="weights/ocr/teachers/parseq_tiny_vn/best.pth",
        device="cuda",
        enable_stabilizer=False,
        save_crops=True,
        log_low_confidence=True,
    )

    test_images = [
        "datasets/VNLP_detection/detection/one_row/1_1010_1_36A38587_102_344_227_413.jpg",
        "datasets/VNLP_detection/detection/one_row/1_1017_1_36A38600_96_293_200_365.jpg",
        "datasets/VNLP_detection/detection/one_row/5_3285_1_36C04519_108_207_527_293.jpg",
        "datasets/VNLP_detection/detection/two_rows/1_1000_0_36A38569_94_293_144_357.jpg",
        "datasets/VNLP_detection/detection/two_rows/1_1003_0_36A38576_478_271_539_345.jpg",
    ]

    print("\n" + "=" * 125)
    print(f"| {'Image':<38} | {'BBox':<22} | {'Layout':<10} | {'DetConf':<8} | {'Prediction':<12} | {'OCRConf':<8} | {'Det(ms)':<8} | {'OCR(ms)':<8} |")
    print("=" * 125)

    results = []
    for p_str in test_images:
        p = Path(p_str)
        res = pipeline.run_image(p, save=True, show_debug=True, no_gui=True)
        plates = res["plates"]
        timings = res["timings"]
        det_ms = timings["detector_ms"]

        if plates:
            for pl in plates:
                print(f"| {p.name:<38} | {str(pl['bbox']):<22} | {pl['layout']:<10} | {pl['confidence']:<8.2f} | {pl['plate_text']:<12} | {pl['ocr_conf']:<8.4f} | {det_ms:<8.1f} | {pl['ocr_ms']:<8.1f} |")
                results.append({
                    "image": p.name,
                    "bbox": pl["bbox"],
                    "layout": pl["layout"],
                    "det_conf": pl["confidence"],
                    "pred": pl["plate_text"],
                    "ocr_conf": pl["ocr_conf"],
                    "det_ms": det_ms,
                    "ocr_ms": pl["ocr_ms"],
                })
        else:
            print(f"| {p.name:<38} | {'NONE':<22} | {'-':<10} | {'-':<8} | {'-':<12} | {'-':<8} | {det_ms:<8.1f} | {0.0:<8.1f} |")

    print("=" * 125)
    return results


if __name__ == "__main__":
    run_smoke_test()
