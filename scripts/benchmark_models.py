"""
Benchmark all 4 ONNX models (640, 512, 416, 320) on Raspberry Pi 5.
Runs warmup 5, benchmark 30 on the exact same test image.
Measures preprocess, inference, postprocess, drawing, and total latency.
"""

import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np

# Add scripts directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi_detect_image import load_network, detect_image

IMAGE_PATH = "/home/cytron/vietnam_plate/test_images/boderngoaigiao1_jpg.rf.8fd405fb48280fbc7fc1389d35acc5d0.jpg"
MODELS_INFO = [
    (640, "/home/cytron/vietnam_plate/models/best_license_plate_yolo11n_640.onnx"),
    (512, "/home/cytron/vietnam_plate/models/best_license_plate_yolo11n_512.onnx"),
    (416, "/home/cytron/vietnam_plate/models/best_license_plate_yolo11n_416.onnx"),
    (320, "/home/cytron/vietnam_plate/models/best_license_plate_yolo11n_320.onnx"),
]

WARMUP_RUNS = 5
BENCH_RUNS = 30


def main():
    print("=" * 65)
    print("     RASPBERRY PI 5 - MULTI-MODEL PERFORMANCE BENCHMARK     ")
    print("=" * 65)
    print(f"Test Image  : {Path(IMAGE_PATH).name}")
    print(f"Warmup runs : {WARMUP_RUNS}")
    print(f"Bench runs  : {BENCH_RUNS}")
    print("-" * 65)

    results = []

    for sz, model_path in MODELS_INFO:
        m_p = Path(model_path)
        if not m_p.exists():
            print(f"Error: {model_path} not found!")
            continue

        m_size_mb = m_p.stat().st_size / (1024 * 1024)
        net = load_network(str(m_p))

        # Warmup
        for _ in range(WARMUP_RUNS):
            detect_image(IMAGE_PATH, net=net, conf_thresh=0.5, iou_thresh=0.45, imgsz=sz)

        # Benchmark runs
        timings = {"prep": [], "inf": [], "post": [], "draw": [], "total": []}
        for _ in range(BENCH_RUNS):
            _, _, _, t = detect_image(IMAGE_PATH, net=net, conf_thresh=0.5, iou_thresh=0.45, imgsz=sz)
            timings["prep"].append(t["prep_ms"])
            timings["inf"].append(t["inf_ms"])
            timings["post"].append(t["post_ms"])
            timings["draw"].append(t["draw_ms"])
            timings["total"].append(t["total_ms"])

        avg_prep = float(np.mean(timings["prep"]))
        avg_inf = float(np.mean(timings["inf"]))
        avg_post = float(np.mean(timings["post"]))
        avg_draw = float(np.mean(timings["draw"]))
        avg_total = float(np.mean(timings["total"]))

        min_inf = float(np.min(timings["inf"]))
        max_inf = float(np.max(timings["inf"]))
        min_total = float(np.min(timings["total"]))
        max_total = float(np.max(timings["total"]))

        raw_fps = 1000.0 / avg_inf if avg_inf > 0 else 0
        e2e_fps = 1000.0 / avg_total if avg_total > 0 else 0

        res = {
            "imgsz": sz,
            "model_size_mb": round(m_size_mb, 2),
            "prep_ms": round(avg_prep, 2),
            "inf_ms": round(avg_inf, 2),
            "min_inf": round(min_inf, 2),
            "max_inf": round(max_inf, 2),
            "post_ms": round(avg_post, 2),
            "draw_ms": round(avg_draw, 2),
            "total_ms": round(avg_total, 2),
            "min_total": round(min_total, 2),
            "max_total": round(max_total, 2),
            "raw_fps": round(raw_fps, 2),
            "e2e_fps": round(e2e_fps, 2),
        }
        results.append(res)

        print(
            f"[{sz}x{sz}] Prep: {avg_prep:.2f}ms | Inf: {avg_inf:.2f}ms (Min: {min_inf:.2f}, Max: {max_inf:.2f}) | "
            f"Post: {avg_post:.2f}ms | Total: {avg_total:.2f}ms | Raw FPS: {raw_fps:.2f} | E2E FPS: {e2e_fps:.2f}"
        )

    print("\n" + "=" * 65)
    print("MODEL PERFORMANCE TABLE:")
    print(f"{'imgsz':>5} | {'model size':>10} | {'preprocess':>10} | {'inference':>10} | {'postprocess':>11} | {'total ms':>9} | {'Raw FPS':>7} | {'E2E FPS':>7}")
    print("-" * 88)
    for r in results:
        print(
            f"{r['imgsz']:>5} | {r['model_size_mb']:>7.2f} MB | {r['prep_ms']:>8.2f} ms | "
            f"{r['inf_ms']:>8.2f} ms | {r['post_ms']:>9.2f} ms | {r['total_ms']:>7.2f} ms | "
            f"{r['raw_fps']:>7.2f} | {r['e2e_fps']:>7.2f}"
        )
    print("=" * 65)

    print("\nJSON_RESULTS_START")
    print(json.dumps(results))
    print("JSON_RESULTS_END")


if __name__ == "__main__":
    main()
