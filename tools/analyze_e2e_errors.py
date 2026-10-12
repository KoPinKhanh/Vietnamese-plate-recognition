"""
Comprehensive Analysis Tool for Real-World End-to-End Evaluation Results.
Reads reports/e2e_real_world_results.csv and computes:
  - Detection recall and IoU distributions
  - Conditional OCR accuracies (GT Crop vs YOLO Crop vs E2E)
  - Layout-wise breakdown (one_row, two_rows, motorcycle)
  - Failure attribution hierarchy
  - Accuracy loss waterfall
  - Confidence calibration bins
  - Plate size proxy analysis
  - Prefix/suffix hallucination audit
  - Core comparison tables
"""

import argparse
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def analyze_results(results_csv_path: Path = PROJECT_ROOT / "reports/e2e_real_world_results.csv") -> Dict[str, Any]:
    if not results_csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv_path}")

    records = []
    with open(results_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Parse booleans and floats
            r["detected"] = (r["detected"].lower() == "true")
            r["gt_crop_correct"] = (r["gt_crop_correct"].lower() == "true")
            r["yolo_crop_correct"] = (r["yolo_crop_correct"].lower() == "true")
            r["layout_correct"] = (r["layout_correct"].lower() == "true")
            r["iou"] = float(r["iou"]) if r["iou"] else 0.0
            r["det_conf"] = float(r["det_conf"]) if r["det_conf"] else 0.0
            r["gt_crop_conf"] = float(r["gt_crop_conf"]) if r["gt_crop_conf"] else 0.0
            r["yolo_crop_conf"] = float(r["yolo_crop_conf"]) if r["yolo_crop_conf"] else 0.0
            r["crop_width"] = int(r["crop_width"]) if r["crop_width"] else 0
            r["crop_height"] = int(r["crop_height"]) if r["crop_height"] else 0
            r["aspect_ratio"] = float(r["aspect_ratio"]) if r["aspect_ratio"] else 0.0
            r["edit_distance"] = int(r["edit_distance"]) if r["edit_distance"] else 0
            r["detector_ms"] = float(r["detector_ms"]) if r["detector_ms"] else 0.0
            r["preprocess_ms"] = float(r["preprocess_ms"]) if r["preprocess_ms"] else 0.0
            r["ocr_ms"] = float(r["ocr_ms"]) if r["ocr_ms"] else 0.0
            r["total_ms"] = float(r["total_ms"]) if r["total_ms"] else 0.0
            records.append(r)

    total_samples = len(records)
    print("=" * 90)
    print(f"REAL-WORLD END-TO-END ERROR ATTRIBUTION ANALYSIS ({total_samples} SAMPLES)")
    print("=" * 90)

    # 1. CORE COMPARISON TABLE (By Layout)
    layouts = ["Overall", "one_row", "two_rows", "motorcycle"]
    layout_subsets = {
        "Overall": records,
        "one_row": [r for r in records if r["gt_layout"] == "one_row"],
        "two_rows": [r for r in records if r["gt_layout"] == "two_rows"],
        "motorcycle": [r for r in records if r["gt_layout"] == "motorcycle"],
    }

    core_table = []
    for lay in layouts:
        subset = layout_subsets[lay]
        n = len(subset)
        if n == 0:
            continue
        det_count = sum(1 for r in subset if r["detected"])
        det_recall = (det_count / n) * 100.0

        matched_ious = [r["iou"] for r in subset if r["detected"]]
        mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

        gt_corr = sum(1 for r in subset if r["gt_crop_correct"])
        gt_acc = (gt_corr / n) * 100.0

        # YOLO crop accuracy on matched detections
        yolo_matched_corr = sum(1 for r in subset if r["detected"] and r["yolo_crop_correct"])
        yolo_matched_acc = (yolo_matched_corr / det_count) * 100.0 if det_count > 0 else 0.0

        # End-to-end exact accuracy on all samples in subset
        e2e_corr = sum(1 for r in subset if r["detected"] and r["yolo_crop_correct"])
        e2e_acc = (e2e_corr / n) * 100.0

        core_table.append({
            "Layout": lay,
            "N": n,
            "Detector Recall": f"{det_recall:.2f}%",
            "Mean IoU": f"{mean_iou:.4f}",
            "GT Crop OCR": f"{gt_acc:.2f}%",
            "YOLO Crop OCR": f"{yolo_matched_acc:.2f}%",
            "E2E Exact": f"{e2e_acc:.2f}%",
            "raw": {
                "n": n,
                "det_recall": det_recall,
                "mean_iou": mean_iou,
                "gt_acc": gt_acc,
                "yolo_matched_acc": yolo_matched_acc,
                "e2e_acc": e2e_acc,
            }
        })

    print("\n--- 1. CORE COMPARISON TABLE ---")
    print(f"| {'Layout':<12} | {'N':>5} | {'Det Recall':>11} | {'Mean IoU':>10} | {'GT Crop OCR':>12} | {'YOLO Crop OCR':>14} | {'E2E Exact':>10} |")
    print("|" + "-"*14 + "|" + "-"*7 + "|" + "-"*13 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*16 + "|" + "-"*12 + "|")
    for row in core_table:
        print(f"| {row['Layout']:<12} | {row['N']:>5} | {row['Detector Recall']:>11} | {row['Mean IoU']:>10} | {row['GT Crop OCR']:>12} | {row['YOLO Crop OCR']:>14} | {row['E2E Exact']:>10} |")

    # 2. ACCURACY LOSS WATERFALL
    print("\n--- 2. ACCURACY LOSS WATERFALL ---")
    step_0_total = total_samples
    step_1_det = sum(1 for r in records if r["detected"])
    # After detector + crop: plates where detector matched and localization allowed correct OCR
    step_2_crop = sum(1 for r in records if r["detected"] and r["gt_crop_correct"])
    # After layout: plates where layout was also classified correctly
    step_3_layout = sum(1 for r in records if r["detected"] and r["layout_correct"])
    # After rectification & final OCR: final exact correct E2E
    step_4_final = sum(1 for r in records if r["yolo_crop_correct"])

    print(f"1. Total Full-Scene Images:      {step_0_total:>4} (100.0%)")
    print(f"2. Correct Plate Detections:     {step_1_det:>4} ({step_1_det/step_0_total*100:.1f}%)  [Loss: -{step_0_total - step_1_det}]")
    print(f"3. Potential Correct with GT:    {step_2_crop:>4} ({step_2_crop/step_0_total*100:.1f}%)")
    print(f"4. Correct Layout Class:         {step_3_layout:>4} ({step_3_layout/step_0_total*100:.1f}%)")
    print(f"5. Final Correct E2E Predictions:{step_4_final:>4} ({step_4_final/step_0_total*100:.1f}%)  [Overall Loss: -{step_0_total - step_4_final}]")

    # 3. FAILURE ATTRIBUTION TAXONOMY
    fail_counts = Counter(r["failure_stage"] for r in records)
    total_failures = total_samples - step_4_final

    print("\n--- 3. FAILURE ATTRIBUTION BREAKDOWN ---")
    print(f"Total Failures: {total_failures} / {total_samples} ({total_failures/total_samples*100:.2f}%)")
    print(f"| {'Failure Stage':<26} | {'Count':>6} | {'% of All':>10} | {'% of Failures':>14} |")
    print("|" + "-"*28 + "|" + "-"*8 + "|" + "-"*12 + "|" + "-"*16 + "|")
    stages = ["DETECTION_MISS", "LOCALIZATION_CROP_ERROR", "LAYOUT_ERROR", "RECTIFICATION_ERROR", "OCR_ERROR", "UNKNOWN"]
    for st in stages:
        c = fail_counts.get(st, 0)
        pct_all = (c / total_samples) * 100.0
        pct_fail = (c / total_failures) * 100.0 if total_failures > 0 else 0.0
        print(f"| {st:<26} | {c:>6} | {pct_all:>9.2f}% | {pct_fail:>13.2f}% |")

    # 4. CHARACTER METRICS (CharAcc & CER)
    total_chars = sum(len(r["ground_truth"]) for r in records)
    total_ed = sum(r["edit_distance"] for r in records)
    all_cers = [r["edit_distance"] / max(1, len(r["ground_truth"])) for r in records]
    macro_cer = sum(all_cers) / total_samples
    micro_cer = total_ed / total_chars if total_chars > 0 else 0.0
    char_acc = max(0.0, (1.0 - micro_cer) * 100.0)
    avg_ed = total_ed / total_samples

    print("\n--- 4. CHARACTER ERROR METRICS ---")
    print(f"Total Target Characters:  {total_chars}")
    print(f"Total Edit Distance (ED): {total_ed}")
    print(f"Average Edit Distance:    {avg_ed:.4f} chars / plate")
    print(f"Micro CER:                {micro_cer:.4f}")
    print(f"Macro CER:                {macro_cer:.4f}")
    print(f"Character Accuracy:       {char_acc:.2f}%")

    # 5. CONFIDENCE CALIBRATION BINS
    conf_bins = [
        ("0.00 - 0.50", 0.00, 0.50),
        ("0.50 - 0.70", 0.50, 0.70),
        ("0.70 - 0.80", 0.70, 0.80),
        ("0.80 - 0.90", 0.80, 0.90),
        ("0.90 - 0.95", 0.90, 0.95),
        ("0.95 - 1.00", 0.95, 1.0001),
    ]

    print("\n--- 5. OCR CONFIDENCE CALIBRATION BINS ---")
    print(f"| {'Confidence Bin':<15} | {'Count':>6} | {'Correct':>8} | {'Accuracy':>10} |")
    print("|" + "-"*17 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*12 + "|")
    conf_bin_data = []
    for label, low, high in conf_bins:
        b_recs = [r for r in records if r["detected"] and (low <= r["yolo_crop_conf"] < high)]
        b_count = len(b_recs)
        b_corr = sum(1 for r in b_recs if r["yolo_crop_correct"])
        b_acc = (b_corr / b_count) * 100.0 if b_count > 0 else 0.0
        print(f"| {label:<15} | {b_count:>6} | {b_corr:>8} | {b_acc:>9.2f}% |")
        conf_bin_data.append({"bin": label, "count": b_count, "correct": b_corr, "accuracy": b_acc})

    # 6. PLATE SIZE / DISTANCE PROXY ANALYSIS
    areas = [r["crop_width"] * r["crop_height"] for r in records if r["detected"]]
    if areas:
        q33 = statistics.quantiles(areas, n=3)[0]
        q66 = statistics.quantiles(areas, n=3)[1]
    else:
        q33, q66 = 5000, 15000

    print("\n--- 6. PLATE PIXEL SIZE PROXY ANALYSIS ---")
    print(f"Size Thresholds: Small <= {int(q33)} px^2 < Medium <= {int(q66)} px^2 < Large")
    print(f"| {'Size Category':<15} | {'Area Range (px^2)':<20} | {'Count':>6} | {'E2E Correct':>12} | {'E2E Accuracy':>14} |")
    print("|" + "-"*17 + "|" + "-"*22 + "|" + "-"*8 + "|" + "-"*14 + "|" + "-"*16 + "|")

    size_bins = [
        ("Small (Distant)", f"<= {int(q33)}", [r for r in records if r["detected"] and (r["crop_width"] * r["crop_height"]) <= q33]),
        ("Medium (Mid)", f"{int(q33)} - {int(q66)}", [r for r in records if r["detected"] and q33 < (r["crop_width"] * r["crop_height"]) <= q66]),
        ("Large (Close)", f"> {int(q66)}", [r for r in records if r["detected"] and (r["crop_width"] * r["crop_height"]) > q66]),
    ]
    size_bin_data = []
    for name, rng, s_recs in size_bins:
        c = len(s_recs)
        corr = sum(1 for r in s_recs if r["yolo_crop_correct"])
        acc = (corr / c) * 100.0 if c > 0 else 0.0
        print(f"| {name:<15} | {rng:<20} | {c:>6} | {corr:>12} | {acc:>13.2f}% |")
        size_bin_data.append({"category": name, "range": rng, "count": c, "correct": corr, "accuracy": acc})

    # 7. PREFIX / SUFFIX HALLUCINATION ANALYSIS
    extra_leading = 0
    missing_leading = 0
    extra_trailing = 0
    missing_trailing = 0
    for r in records:
        if r["detected"] and not r["yolo_crop_correct"] and r["yolo_crop_prediction"]:
            gt = r["ground_truth"]
            p = r["yolo_crop_prediction"]
            if len(p) > len(gt) and p.endswith(gt):
                extra_leading += 1
            if len(p) < len(gt) and gt.endswith(p):
                missing_leading += 1
            if len(p) > len(gt) and p.startswith(gt):
                extra_trailing += 1
            if len(p) < len(gt) and gt.startswith(p):
                missing_trailing += 1

    print("\n--- 7. PREFIX / SUFFIX HALLUCINATION AUDIT ---")
    print(f"Extra Leading Characters:   {extra_leading}")
    print(f"Missing Leading Characters: {missing_leading}")
    print(f"Extra Trailing Characters:  {extra_trailing}")
    print(f"Missing Trailing Characters:{missing_trailing}")

    # 8. LATENCY SUMMARY
    det_times = [r["detector_ms"] for r in records]
    prep_times = [r["preprocess_ms"] for r in records if r["detected"]]
    ocr_times = [r["ocr_ms"] for r in records if r["detected"]]
    tot_times = [r["total_ms"] for r in records]

    print("\n--- 8. LATENCY PROFILE ---")
    print(f"Detector Mean Latency:   {sum(det_times)/len(det_times):.2f} ms")
    print(f"Preprocess Mean Latency: {sum(prep_times)/len(prep_times):.2f} ms" if prep_times else "Preprocess: N/A")
    print(f"PARSeq OCR Mean Latency: {sum(ocr_times)/len(ocr_times):.2f} ms / plate" if ocr_times else "OCR: N/A")
    print(f"End-to-End Mean Latency: {sum(tot_times)/len(tot_times):.2f} ms / frame")
    print(f"End-to-End Throughput:   {1000.0 / (sum(tot_times)/len(tot_times)):.2f} FPS")
    print("=" * 90)

    return {
        "total_samples": total_samples,
        "core_table": core_table,
        "fail_counts": fail_counts,
        "conf_bin_data": conf_bin_data,
        "size_bin_data": size_bin_data,
        "char_acc": char_acc,
        "micro_cer": micro_cer,
        "avg_ed": avg_ed,
        "latencies": {
            "det": sum(det_times) / len(det_times),
            "prep": sum(prep_times) / len(prep_times) if prep_times else 0.0,
            "ocr": sum(ocr_times) / len(ocr_times) if ocr_times else 0.0,
            "tot": sum(tot_times) / len(tot_times),
            "fps": 1000.0 / (sum(tot_times) / len(tot_times)),
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Real-World E2E Evaluation Results")
    parser.add_argument("--results", type=str, default="reports/e2e_real_world_results.csv")
    args = parser.parse_args()
    analyze_results(Path(args.results))
