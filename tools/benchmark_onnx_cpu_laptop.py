"""
Benchmark ONNX Runtime CPU latency on Laptop for deployed models:
1. Vehicle YOLO11s (1x3x640x640)
2. PARSeq-Tiny VNLP (1x3x32x128)
Warmup: >= 20 runs
Timed: >= 100 runs
Outputs: Mean, Median, P95, Min, Max latencies.
"""

import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def benchmark_session(
    onnx_path: Path,
    input_shape: tuple,
    input_name: str,
    warmup_runs: int = 25,
    timed_runs: int = 100,
    num_threads: int = 4,
) -> dict:
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = num_threads
    sess = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])

    dummy = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup_runs):
        _ = sess.run(None, {input_name: dummy})

    # Timed runs
    latencies = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies)
    return {
        "model": onnx_path.name,
        "input_shape": list(input_shape),
        "warmup": warmup_runs,
        "runs": timed_runs,
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
    }


def main():
    print("=" * 80)
    print("LAPTOP CPU ONNX RUNTIME LATENCY BENCHMARK")
    print("=" * 80)
    print("NOTE: Sanity baseline only. Do not use laptop CPU latency to predict Pi FPS.")

    yolo_path = PROJECT_ROOT / "weights/deployment/pi_cpu/yolo11s_640_fp32.onnx"
    parseq_path = PROJECT_ROOT / "weights/deployment/pi_cpu/parseq_tiny_vn_fp32.onnx"

    print("\n[1/2] Benchmarking Vehicle YOLO11s (640x640)...")
    res_yolo = benchmark_session(yolo_path, (1, 3, 640, 640), "images", warmup_runs=25, timed_runs=100)

    print("\n[2/2] Benchmarking PARSeq-Tiny VNLP (32x128)...")
    res_parseq = benchmark_session(parseq_path, (1, 3, 32, 128), "image", warmup_runs=25, timed_runs=100)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (CPUExecutionProvider on Laptop CPU)")
    print("=" * 80)
    print(f"{'Model':<28} | {'Shape':<16} | {'Mean (ms)':<10} | {'Median (ms)':<12} | {'P95 (ms)':<10}")
    print("-" * 80)
    print(f"{res_yolo['model']:<28} | {str(res_yolo['input_shape']):<16} | {res_yolo['mean_ms']:<10.2f} | {res_yolo['median_ms']:<12.2f} | {res_yolo['p95_ms']:<10.2f}")
    print(f"{res_parseq['model']:<28} | {str(res_parseq['input_shape']):<16} | {res_parseq['mean_ms']:<10.2f} | {res_parseq['median_ms']:<12.2f} | {res_parseq['p95_ms']:<10.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
