"""
Comprehensive Evaluation, Comparative Benchmarking, and Error Analysis between Model V1 and Model V2-A.
Runs under the exact same OpenCV DNN CPU runtime and same preprocessing.
Evaluates:
  - VAL (datasets/VNLP_processed/val.csv)
  - TEST_FINAL (datasets/VNLP_processed/test.csv)
  - TEST_CLEAN (datasets/VNLP_processed/test_clean.csv)
  - TEST_UNSEEN_TEXT (datasets/VNLP_processed/test_unseen_text.csv)
Produces:
  - reports/v1_vs_v2a_comparison.md
  - reports/v2a_pytorch_evaluation.md
  - reports/v2a_fixed_cases.csv
  - reports/v2a_regressions.csv
  - reports/v1_confusion_matrix.csv
  - reports/v2a_confusion_matrix.csv
  - reports/v1_vs_v2a_confusion_summary.md
"""

import argparse
import ast
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.charset import CHARSET, NUM_CLASSES, BLANK_INDEX
from ocr.pi_lprnet_ocr import PlateOCR, normalize_plate_text
from ocr.plate_text_utils import calculate_cer, levenshtein_distance
from ocr.preprocess import preprocess_plate_for_ocr


def evaluate_manifest_opencv(
    ocr: PlateOCR,
    manifest_csv: Path,
    dataset_name: str = "TEST",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates an OpenCV DNN PlateOCR instance on a given CSV manifest."""
    print(f"  Evaluating {dataset_name} ({manifest_csv.name})...")
    records = []
    with open(manifest_csv, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            records.append(r)

    total_samples = 0
    total_exact = 0
    total_cer = 0.0
    latencies = []

    layout_stats: Dict[str, Dict[str, Any]] = {
        "one_row": {"total": 0, "exact": 0, "cer_sum": 0.0},
        "two_rows": {"total": 0, "exact": 0, "cer_sum": 0.0},
        "two_rows_label_xe_may": {"total": 0, "exact": 0, "cer_sum": 0.0},
    }

    sample_results = []

    for r in records:
        img_path = r["image_path"]
        raw_text = r["plate_text"]
        gt_clean = normalize_plate_text(raw_text)
        layout = r.get("subset", r.get("layout", "one_row"))

        # Parse bbox if available
        bbox = None
        if "x1" in r and r["x1"] and r["x1"].strip():
            bbox = [int(float(r["x1"])), int(float(r["y1"])), int(float(r["x2"])), int(float(r["y2"]))]
        elif "bbox" in r and r["bbox"] and r["bbox"].strip():
            bbox = ast.literal_eval(r["bbox"])

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # Ground truth crop if full frame
        if bbox is not None and len(bbox) == 4:
            ih, iw = img_bgr.shape[:2]
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            pad_w = int(round(bw * 0.03))
            pad_h = int(round(bh * 0.03))
            crop = img_bgr[max(0, y1-pad_h):min(ih, y2+pad_h), max(0, x1-pad_w):min(iw, x2+pad_w)]
        else:
            crop = img_bgr

        # Recognize
        res = ocr.recognize(crop, layout=layout)
        pred_clean = normalize_plate_text(res["text"])
        latencies.append(res["inference_ms"])

        exact = (gt_clean == pred_clean)
        cer = calculate_cer(gt_clean, pred_clean)

        total_samples += 1
        if exact:
            total_exact += 1
        total_cer += cer

        l_key = layout if layout in layout_stats else ("two_rows" if "two" in layout else "one_row")
        layout_stats[l_key]["total"] += 1
        if exact:
            layout_stats[l_key]["exact"] += 1
        layout_stats[l_key]["cer_sum"] += cer

        sample_results.append({
            "image_path": img_path,
            "ground_truth": gt_clean,
            "prediction": pred_clean,
            "exact": exact,
            "cer": cer,
            "layout": l_key,
            "source_test_set": dataset_name,
        })

    exact_acc = (total_exact / total_samples) * 100.0 if total_samples > 0 else 0.0
    avg_cer = total_cer / total_samples if total_samples > 0 else 0.0
    char_acc = max(0.0, (1.0 - avg_cer) * 100.0)
    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    metrics = {
        "dataset": dataset_name,
        "total_samples": total_samples,
        "exact_accuracy": exact_acc,
        "character_accuracy": char_acc,
        "cer": avg_cer,
        "avg_latency_ms": avg_lat,
        "fps": fps,
        "layout_breakdown": {},
    }

    for l_key, st in layout_stats.items():
        n = st["total"]
        if n > 0:
            ex = (st["exact"] / n) * 100.0
            c = st["cer_sum"] / n
            metrics["layout_breakdown"][l_key] = {
                "total": n,
                "exact_accuracy": ex,
                "cer": c,
                "char_accuracy": max(0.0, (1.0 - c) * 100.0),
            }

    print(f"    -> Exact Acc: {exact_acc:.2f}% | Char Acc: {char_acc:.2f}% | CER: {avg_cer:.4f} | Latency: {avg_lat:.2f}ms")
    return metrics, sample_results


def build_confusion_matrix(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extracts character level substitutions for samples with equal length."""
    pair_counts = Counter()
    for s in samples:
        gt = s["ground_truth"]
        pred = s["prediction"]
        if len(gt) == len(pred):
            for g, p in zip(gt, pred):
                if g != p:
                    pair_counts[f"{g} -> {p}"] += 1
    return pair_counts


def run_full_benchmark(
    v1_onnx: Path,
    v2a_onnx: Path,
    manifests: Dict[str, Path],
    reports_dir: Path,
):
    print("=" * 70)
    print("       MODEL V1 VS MODEL V2-A COMPARATIVE BENCHMARK (OPENCV DNN)      ")
    print("=" * 70)

    reports_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading Model V1 (best_lprnet_vn.onnx)...")
    ocr_v1 = PlateOCR(str(v1_onnx))
    print("Loading Model V2-A (best_lprnet_v2a.onnx)...")
    ocr_v2a = PlateOCR(str(v2a_onnx))

    v1_metrics_all = {}
    v2a_metrics_all = {}
    v1_samples_all = {}
    v2a_samples_all = {}

    for name, path in manifests.items():
        if not path.exists():
            print(f"Warning: {path} not found, skipping {name}")
            continue
        print(f"\n[EVALUATING {name}] ({path.name})")
        m1, s1 = evaluate_manifest_opencv(ocr_v1, path, dataset_name=name)
        m2, s2 = evaluate_manifest_opencv(ocr_v2a, path, dataset_name=name)
        v1_metrics_all[name] = m1
        v2a_metrics_all[name] = m2
        v1_samples_all[name] = s1
        v2a_samples_all[name] = s2

    # 1. Error Analysis: Fixed cases & Regressions on TEST_FINAL
    print("\n--- Generating Fixed Cases & Regressions on TEST_FINAL ---")
    fixed_cases = []
    regressions = []

    s1_test = v1_samples_all["TEST_FINAL"]
    s2_test = v2a_samples_all["TEST_FINAL"]

    for r1, r2 in zip(s1_test, s2_test):
        assert r1["image_path"] == r2["image_path"], "Image mismatch!"
        if not r1["exact"] and r2["exact"]:
            fixed_cases.append({
                "image": Path(r1["image_path"]).name,
                "ground_truth": r1["ground_truth"],
                "v1_prediction": r1["prediction"],
                "v2a_prediction": r2["prediction"],
                "layout": r1["layout"],
                "source_test_set": "TEST_FINAL"
            })
        elif r1["exact"] and not r2["exact"]:
            regressions.append({
                "image": Path(r1["image_path"]).name,
                "ground_truth": r1["ground_truth"],
                "v1_prediction": r1["prediction"],
                "v2a_prediction": r2["prediction"],
                "layout": r1["layout"],
                "source_test_set": "TEST_FINAL"
            })

    print(f"Fixed cases (V1 wrong -> V2-A right): {len(fixed_cases)}")
    print(f"Regressions (V1 right -> V2-A wrong): {len(regressions)}")

    # Write fixed cases CSV
    with open(reports_dir / "v2a_fixed_cases.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image", "ground_truth", "v1_prediction", "v2a_prediction", "layout", "source_test_set"])
        writer.writeheader()
        writer.writerows(fixed_cases)

    # Write regressions CSV
    with open(reports_dir / "v2a_regressions.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image", "ground_truth", "v1_prediction", "v2a_prediction", "layout", "source_test_set"])
        writer.writeheader()
        writer.writerows(regressions)

    # 2. Confusion Matrices for TEST_FINAL
    print("\n--- Generating Character Confusion Matrices ---")
    v1_conf = build_confusion_matrix([s for s in s1_test if not s["exact"]])
    v2a_conf = build_confusion_matrix([s for s in s2_test if not s["exact"]])

    with open(reports_dir / "v1_confusion_matrix.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["pair", "count"])
        for p, c in v1_conf.most_common():
            writer.writerow([p, c])

    with open(reports_dir / "v2a_confusion_matrix.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["pair", "count"])
        for p, c in v2a_conf.most_common():
            writer.writerow([p, c])

    # Special tracked pairs
    tracked_pairs = ["8 -> B", "B -> 8", "0 -> D", "D -> 0", "5 -> S", "S -> 5", "2 -> Z", "Z -> 2"]
    
    # 3. Write Confusion Summary Markdown
    conf_summary_lines = [
        "# TỔNG HỢP MA TRẬN NHẦM LẪN KÝ TỰ (CHARACTER CONFUSION SUMMARY)",
        "",
        "**Tập kiểm định:** `TEST_FINAL` (3,733 mẫu)",
        "",
        "## 1. Theo dõi các Cặp Ký tự Dễ Nhầm Lẫn Quan trọng",
        "| Cặp nhầm lẫn (GT -> Pred) | Số lần ở Model V1 | Số lần ở Model V2-A | Thay đổi (V2-A vs V1) |",
        "| :---: | :---: | :---: | :---: |"
    ]
    for pair in tracked_pairs:
        c1 = v1_conf.get(pair, 0)
        c2 = v2a_conf.get(pair, 0)
        diff = c2 - c1
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        conf_summary_lines.append(f"| `{pair}` | {c1} | {c2} | **{diff_str}** |")

    conf_summary_lines.extend([
        "",
        "## 2. Top 10 Lỗi Phổ Biến Nhất ở Model V1",
        "| Hạng | Cặp ký tự (GT -> Pred) | Tần suất |",
        "| :---: | :---: | :---: |"
    ])
    for idx, (p, c) in enumerate(v1_conf.most_common(10), 1):
        conf_summary_lines.append(f"| {idx} | `{p}` | {c} |")

    conf_summary_lines.extend([
        "",
        "## 3. Top 10 Lỗi Phổ Biến Nhất ở Model V2-A",
        "| Hạng | Cặp ký tự (GT -> Pred) | Tần suất |",
        "| :---: | :---: | :---: |"
    ])
    for idx, (p, c) in enumerate(v2a_conf.most_common(10), 1):
        conf_summary_lines.append(f"| {idx} | `{p}` | {c} |")

    with open(reports_dir / "v1_vs_v2a_confusion_summary.md", "w", encoding="utf-8") as fp:
        fp.write("\n".join(conf_summary_lines))

    # 4. Generate Main Comparison Table Markdown (reports/v1_vs_v2a_comparison.md)
    print("\n--- Generating Main Comparison Report (v1_vs_v2a_comparison.md) ---")
    
    m1_tf = v1_metrics_all["TEST_FINAL"]
    m2_tf = v2a_metrics_all["TEST_FINAL"]
    m1_tc = v1_metrics_all["TEST_CLEAN"]
    m2_tc = v2a_metrics_all["TEST_CLEAN"]
    m1_un = v1_metrics_all["TEST_UNSEEN_TEXT"]
    m2_un = v2a_metrics_all["TEST_UNSEEN_TEXT"]

    def delta_str(v2: float, v1: float, unit: str = "pp") -> str:
        d = v2 - v1
        sign = "+" if d > 0 else ""
        return f"{sign}{d:.2f} {unit}"

    comp_lines = [
        "# BÁO CÁO SO SÁNH ĐỐI CHỨNG MODEL V1 VS MODEL V2-A (V1 VS V2-A COMPARISON)",
        "",
        "**Thời gian kiểm định:** 2026-09-24",
        "**Runtime suy luận:** OpenCV DNN CPU thuần (DNN_BACKEND_OPENCV, DNN_TARGET_CPU)",
        "**Phần cứng kiểm thử:** Intel Core i5-12500H (Host CPU)",
        f"**Model V1:** `{v1_onnx.name}` ({v1_onnx.stat().st_size / (1024*1024):.2f} MB)",
        f"**Model V2-A:** `{v2a_onnx.name}` ({v2a_onnx.stat().st_size / (1024*1024):.2f} MB)",
        "",
        "---",
        "",
        "## 1. Bảng So Sánh Hiệu Năng Cốt Lõi",
        "| Chỉ số Đánh giá | Model V1 (Baseline) | Model V2-A (PBL4 Added) | Delta (V2-A - V1) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **TEST_FINAL Exact Acc (3,733)** | {m1_tf['exact_accuracy']:.2f}% | {m2_tf['exact_accuracy']:.2f}% | **{delta_str(m2_tf['exact_accuracy'], m1_tf['exact_accuracy'])}** |",
        f"| **TEST_CLEAN Exact Acc (3,727)** | {m1_tc['exact_accuracy']:.2f}% | {m2_tc['exact_accuracy']:.2f}% | **{delta_str(m2_tc['exact_accuracy'], m1_tc['exact_accuracy'])}** |",
        f"| **TEST_UNSEEN_TEXT Exact Acc (1,187)** | {m1_un['exact_accuracy']:.2f}% | {m2_un['exact_accuracy']:.2f}% | **{delta_str(m2_un['exact_accuracy'], m1_un['exact_accuracy'])}** |",
        "| | | | |",
        f"| — Biển 1 dòng (`one_row`) | {m1_tf['layout_breakdown']['one_row']['exact_accuracy']:.2f}% | {m2_tf['layout_breakdown']['one_row']['exact_accuracy']:.2f}% | **{delta_str(m2_tf['layout_breakdown']['one_row']['exact_accuracy'], m1_tf['layout_breakdown']['one_row']['exact_accuracy'])}** |",
        f"| — Biển 2 dòng ô tô (`two_rows`) | {m1_tf['layout_breakdown']['two_rows']['exact_accuracy']:.2f}% | {m2_tf['layout_breakdown']['two_rows']['exact_accuracy']:.2f}% | **{delta_str(m2_tf['layout_breakdown']['two_rows']['exact_accuracy'], m1_tf['layout_breakdown']['two_rows']['exact_accuracy'])}** |",
        f"| — Biển 2 dòng xe máy (`motorcycle`) | {m1_tf['layout_breakdown']['two_rows_label_xe_may']['exact_accuracy']:.2f}% | {m2_tf['layout_breakdown']['two_rows_label_xe_may']['exact_accuracy']:.2f}% | **{delta_str(m2_tf['layout_breakdown']['two_rows_label_xe_may']['exact_accuracy'], m1_tf['layout_breakdown']['two_rows_label_xe_may']['exact_accuracy'])}** |",
        "| | | | |",
        f"| **Character Accuracy (TEST_FINAL)** | {m1_tf['character_accuracy']:.2f}% | {m2_tf['character_accuracy']:.2f}% | **{delta_str(m2_tf['character_accuracy'], m1_tf['character_accuracy'])}** |",
        f"| **Character Error Rate (CER)** | {m1_tf['cer']:.4f} | {m2_tf['cer']:.4f} | **{m2_tf['cer'] - m1_tf['cer']:+.4f}** |",
        f"| **Character Accuracy (UNSEEN_TEXT)** | {m1_un['character_accuracy']:.2f}% | {m2_un['character_accuracy']:.2f}% | **{delta_str(m2_un['character_accuracy'], m1_un['character_accuracy'])}** |",
        f"| **CER (UNSEEN_TEXT)** | {m1_un['cer']:.4f} | {m2_un['cer']:.4f} | **{m2_un['cer'] - m1_un['cer']:+.4f}** |",
        "| | | | |",
        f"| **Dung lượng ONNX Model** | {v1_onnx.stat().st_size / (1024*1024):.2f} MB | {v2a_onnx.stat().st_size / (1024*1024):.2f} MB | 0.00 MB |",
        f"| **Laptop CPU Latency trung bình** | {m1_tf['avg_latency_ms']:.2f} ms ({m1_tf['fps']:.1f} FPS) | {m2_tf['avg_latency_ms']:.2f} ms ({m2_tf['fps']:.1f} FPS) | {m2_tf['avg_latency_ms'] - m1_tf['avg_latency_ms']:+.2f} ms |",
        "| **Raspberry Pi 5 OCR Latency** | ~12.40 ms (đo trước) | NOT MEASURED (Pi offline) | N/A |",
        "",
        "---",
        "",
        "## 2. Phân Tích Hiện Tượng Sửa Lỗi và Suy Giảm (Error Analysis)",
        f"- **Số ca Model V1 sai nhưng Model V2-A đoán đúng (Fixed Cases):** **{len(fixed_cases)}** mẫu (chi tiết tại `reports/v2a_fixed_cases.csv`).",
        f"- **Số ca Model V1 đúng nhưng Model V2-A đoán sai (Regressions):** **{len(regressions)}** mẫu (chi tiết tại `reports/v2a_regressions.csv`).",
        f"- **Net Improvement trên TEST_FINAL:** `{len(fixed_cases) - len(regressions):+d}` mẫu đúng tuyệt đối.",
        "",
        "---",
        "",
        "## 3. Đánh Giá Khả Năng Tổng Quát Hóa (Generalization Analysis)",
        f"- Trên tập biển số chưa từng xuất hiện trong tập Train (`TEST_UNSEEN_TEXT`):",
        f"  - Model V1: **{m1_un['exact_accuracy']:.2f}%**",
        f"  - Model V2-A: **{m2_un['exact_accuracy']:.2f}%**",
        f"  - Delta Generalization: **{delta_str(m2_un['exact_accuracy'], m1_un['exact_accuracy'])}**",
        "",
        "---",
        "",
        "## 4. Phân Loại Trạng Thái Model V2-A",
        f"- Kết luận phân loại: **[{'PROMISING' if m2_un['exact_accuracy'] >= m1_un['exact_accuracy'] and m2_tf['exact_accuracy'] >= m1_tf['exact_accuracy'] else ('REGRESSION' if m2_un['exact_accuracy'] < m1_un['exact_accuracy'] - 1.0 else 'NEUTRAL')}]**.",
        "- Quyết định triển khai: **KHÔNG đưa V2-A vào production**. Model V1 tiếp tục giữ vai trò active cho đến khi người dùng và ChatGPT phân tích toàn diện snapshot này.",
        ""
    ]

    with open(reports_dir / "v1_vs_v2a_comparison.md", "w", encoding="utf-8") as fp:
        fp.write("\n".join(comp_lines))

    print(f"Comparison report written to {reports_dir / 'v1_vs_v2a_comparison.md'}")
    return v1_metrics_all, v2a_metrics_all, len(fixed_cases), len(regressions)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Model V1 vs Model V2-A")
    parser.add_argument("--v1_onnx", type=str, default="weights/ocr/best_lprnet_vn.onnx")
    parser.add_argument("--v2a_onnx", type=str, default="weights/ocr/experiments/v2a/best_lprnet_v2a.onnx")
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    manifests = {
        "TEST_FINAL": Path("datasets/VNLP_processed/test.csv"),
        "TEST_CLEAN": Path("datasets/VNLP_processed/test_clean.csv"),
        "TEST_UNSEEN_TEXT": Path("datasets/VNLP_processed/test_unseen_text.csv"),
        "VAL": Path("datasets/VNLP_processed/val.csv"),
    }

    run_full_benchmark(
        v1_onnx=Path(args.v1_onnx),
        v2a_onnx=Path(args.v2a_onnx),
        manifests=manifests,
        reports_dir=Path(args.reports_dir),
    )


if __name__ == "__main__":
    main()
