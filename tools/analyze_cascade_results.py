"""
Statistical and Diagnostic Analysis for Direct vs Hailo Cascade Experiment.
Reads reports/direct_vs_cascade_results.csv and outputs:
  - Core Report Table with Deltas
  - Paired Detection Table with McNemar Test and Bootstrap 95% CI
  - Motorcycle Focused Rescue Analysis
  - Cascade Internal Miss Attribution Table
  - Oracle Complementarity Union Recall
  - Decision Gate Evaluation
"""

import argparse
import csv
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def mcnemar_exact_test(b: int, c: int) -> float:
    """Computes exact two-tailed McNemar test p-value using Binomial distribution."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2.0 * prob)


def paired_bootstrap_delta(direct_hits: List[bool], cascade_hits: List[bool], n_boot: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    """Computes paired bootstrap delta recall and 95% confidence interval."""
    np.random.seed(seed)
    n = len(direct_hits)
    d_arr = np.array(direct_hits, dtype=np.float32)
    c_arr = np.array(cascade_hits, dtype=np.float32)
    obs_delta = (c_arr.mean() - d_arr.mean()) * 100.0

    indices = np.random.randint(0, n, size=(n_boot, n))
    d_boot = d_arr[indices].mean(axis=1) * 100.0
    c_boot = c_arr[indices].mean(axis=1) * 100.0
    delta_boot = c_boot - d_boot

    ci_low = float(np.percentile(delta_boot, 2.5))
    ci_high = float(np.percentile(delta_boot, 97.5))
    return obs_delta, ci_low, ci_high


def analyze_cascade_results(csv_path: Path = PROJECT_ROOT / "reports/direct_vs_cascade_results.csv") -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["direct_detected"] = (r["direct_detected"].lower() == "true")
            r["direct_ocr_correct"] = (r["direct_ocr_correct"].lower() == "true")
            r["direct_e2e_correct"] = (r["direct_e2e_correct"].lower() == "true")
            r["cascade_detected"] = (r["cascade_detected"].lower() == "true")
            r["cascade_ocr_correct"] = (r["cascade_ocr_correct"].lower() == "true")
            r["cascade_e2e_correct"] = (r["cascade_e2e_correct"].lower() == "true")
            r["vehicle_found"] = (r["vehicle_found"].lower() == "true")
            r["direct_iou"] = float(r["direct_iou"]) if r["direct_iou"] else 0.0
            r["cascade_iou"] = float(r["cascade_iou"]) if r["cascade_iou"] else 0.0
            r["direct_total_ms"] = float(r["direct_total_ms"]) if r["direct_total_ms"] else 0.0
            r["cascade_total_ms"] = float(r["cascade_total_ms"]) if r["cascade_total_ms"] else 0.0
            records.append(r)

    total_samples = len(records)
    print("=" * 95)
    print(f"CONTROLLED EXPERIMENT ANALYSIS: DIRECT VS HAILO CASCADE ({total_samples} SAMPLES)")
    print("=" * 95)

    # 1. CORE COMPARISON TABLE
    def calc_metrics(subset):
        n = len(subset)
        if n == 0:
            return {}
        d_det = sum(1 for r in subset if r["direct_detected"])
        c_det = sum(1 for r in subset if r["cascade_detected"])
        d_rec = (d_det / n) * 100.0
        c_rec = (c_det / n) * 100.0

        d_ious = [r["direct_iou"] for r in subset if r["direct_detected"]]
        c_ious = [r["cascade_iou"] for r in subset if r["cascade_detected"]]
        d_mean_iou = statistics.mean(d_ious) if d_ious else 0.0
        c_mean_iou = statistics.mean(c_ious) if c_ious else 0.0

        d_ocr_cond = (sum(1 for r in subset if r["direct_detected"] and r["direct_ocr_correct"]) / d_det * 100.0) if d_det > 0 else 0.0
        c_ocr_cond = (sum(1 for r in subset if r["cascade_detected"] and r["cascade_ocr_correct"]) / c_det * 100.0) if c_det > 0 else 0.0

        d_e2e = (sum(1 for r in subset if r["direct_e2e_correct"]) / n) * 100.0
        c_e2e = (sum(1 for r in subset if r["cascade_e2e_correct"]) / n) * 100.0

        return {
            "n": n,
            "d_det": d_det, "c_det": c_det,
            "d_rec": d_rec, "c_rec": c_rec, "delta_rec": c_rec - d_rec,
            "d_mean_iou": d_mean_iou, "c_mean_iou": c_mean_iou, "delta_iou": c_mean_iou - d_mean_iou,
            "d_ocr_cond": d_ocr_cond, "c_ocr_cond": c_ocr_cond, "delta_ocr_cond": c_ocr_cond - d_ocr_cond,
            "d_e2e": d_e2e, "c_e2e": c_e2e, "delta_e2e": c_e2e - d_e2e,
        }

    subsets = {
        "Overall": records,
        "one_row": [r for r in records if r["layout"] == "one_row"],
        "two_rows": [r for r in records if r["layout"] == "two_rows"],
        "motorcycle": [r for r in records if r["layout"] == "motorcycle"],
    }
    m_data = {k: calc_metrics(v) for k, v in subsets.items()}

    d_times = [r["direct_total_ms"] for r in records]
    c_times = [r["cascade_total_ms"] for r in records]
    d_fps = 1000.0 / statistics.mean(d_times) if d_times else 0.0
    c_fps = 1000.0 / statistics.mean(c_times) if c_times else 0.0

    print("\n--- 1. CORE COMPARISON TABLE ---")
    headers = [
        ("Detection Recall Overall", f"{m_data['Overall']['d_rec']:.2f}%", f"{m_data['Overall']['c_rec']:.2f}%", f"{m_data['Overall']['delta_rec']:+.2f} pp"),
        ("One-row Recall", f"{m_data['one_row']['d_rec']:.2f}%", f"{m_data['one_row']['c_rec']:.2f}%", f"{m_data['one_row']['delta_rec']:+.2f} pp"),
        ("Two-row Recall", f"{m_data['two_rows']['d_rec']:.2f}%", f"{m_data['two_rows']['c_rec']:.2f}%", f"{m_data['two_rows']['delta_rec']:+.2f} pp"),
        ("Motorcycle Recall", f"{m_data['motorcycle']['d_rec']:.2f}%", f"{m_data['motorcycle']['c_rec']:.2f}%", f"{m_data['motorcycle']['delta_rec']:+.2f} pp"),
        ("Mean IoU (Matched)", f"{m_data['Overall']['d_mean_iou']:.4f}", f"{m_data['Overall']['c_mean_iou']:.4f}", f"{m_data['Overall']['delta_iou']:+.4f}"),
        ("OCR Given Detection", f"{m_data['Overall']['d_ocr_cond']:.2f}%", f"{m_data['Overall']['c_ocr_cond']:.2f}%", f"{m_data['Overall']['delta_ocr_cond']:+.2f} pp"),
        ("E2E Exact Overall", f"{m_data['Overall']['d_e2e']:.2f}%", f"{m_data['Overall']['c_e2e']:.2f}%", f"{m_data['Overall']['delta_e2e']:+.2f} pp"),
        ("E2E One-row", f"{m_data['one_row']['d_e2e']:.2f}%", f"{m_data['one_row']['c_e2e']:.2f}%", f"{m_data['one_row']['delta_e2e']:+.2f} pp"),
        ("E2E Two-row", f"{m_data['two_rows']['d_e2e']:.2f}%", f"{m_data['two_rows']['c_e2e']:.2f}%", f"{m_data['two_rows']['delta_e2e']:+.2f} pp"),
        ("E2E Motorcycle", f"{m_data['motorcycle']['d_e2e']:.2f}%", f"{m_data['motorcycle']['c_e2e']:.2f}%", f"{m_data['motorcycle']['delta_e2e']:+.2f} pp"),
        ("Mean Latency", f"{statistics.mean(d_times):.2f} ms", f"{statistics.mean(c_times):.2f} ms", f"{statistics.mean(c_times) - statistics.mean(d_times):+.2f} ms"),
        ("FPS", f"{d_fps:.1f}", f"{c_fps:.1f}", f"{c_fps - d_fps:+.1f}"),
    ]

    print(f"| {'Metric':<26} | {'Direct Plate 416':>18} | {'Vehicle->Plate Cascade':>24} | {'Delta':>12} |")
    print("|" + "-"*28 + "|" + "-"*20 + "|" + "-"*26 + "|" + "-"*14 + "|")
    for m, d_val, c_val, delta_val in headers:
        print(f"| {m:<26} | {d_val:>18} | {c_val:>24} | {delta_val:>12} |")

    # 2. PAIRED DETECTION TABLE
    p_states = Counter(r["paired_detection_state"] for r in records)
    both_c = p_states["BOTH_DETECT"]
    direct_only_c = p_states["DIRECT_ONLY"]
    cascade_only_c = p_states["CASCADE_ONLY"]
    neither_c = p_states["NEITHER"]
    net_rescue = cascade_only_c - direct_only_c

    direct_bools = [r["direct_detected"] for r in records]
    cascade_bools = [r["cascade_detected"] for r in records]
    mcnemar_p = mcnemar_exact_test(cascade_only_c, direct_only_c)
    obs_delta, ci_low, ci_high = paired_bootstrap_delta(direct_bools, cascade_bools)

    print("\n--- 2. PAIRED DETECTION TABLE ---")
    print(f"Both detect:           {both_c:>4} ({both_c/total_samples*100:.1f}%)")
    print(f"Direct only:           {direct_only_c:>4} ({direct_only_c/total_samples*100:.1f}%) [Lost by Cascade]")
    print(f"Cascade only:          {cascade_only_c:>4} ({cascade_only_c/total_samples*100:.1f}%) [Rescued by Cascade]")
    print(f"Neither:               {neither_c:>4} ({neither_c/total_samples*100:.1f}%)")
    print(f"Net Rescue:            {net_rescue:+d} plates")
    print(f"McNemar Exact p-value: {mcnemar_p:.4e}")
    print(f"Bootstrap Delta:       {obs_delta:+.2f} pp (95% CI: [{ci_low:+.2f} pp, {ci_high:+.2f} pp])")

    # 3. MOTORCYCLE FOCUSED ANALYSIS
    m_recs = subsets["motorcycle"]
    m_states = Counter(r["paired_detection_state"] for r in m_recs)
    m_d_bools = [r["direct_detected"] for r in m_recs]
    m_c_bools = [r["cascade_detected"] for r in m_recs]
    m_mcnemar_p = mcnemar_exact_test(m_states["CASCADE_ONLY"], m_states["DIRECT_ONLY"])
    m_obs_delta, m_ci_low, m_ci_high = paired_bootstrap_delta(m_d_bools, m_c_bools)

    print("\n--- 3. MOTORCYCLE FOCUSED RESCUE ANALYSIS (N=51) ---")
    print(f"Direct Detected:                   {m_data['motorcycle']['d_det']} / 51 ({m_data['motorcycle']['d_rec']:.2f}%)")
    print(f"Cascade Detected:                  {m_data['motorcycle']['c_det']} / 51 ({m_data['motorcycle']['c_rec']:.2f}%)")
    print(f"Cascade-only Motorcycle Rescues:   {m_states['CASCADE_ONLY']}")
    print(f"Direct-only Motorcycle Losses:     {m_states['DIRECT_ONLY']}")
    print(f"Both Motorcycle Detected:          {m_states['BOTH_DETECT']}")
    print(f"Neither Detected:                  {m_states['NEITHER']}")
    print(f"Motorcycle McNemar p-value:        {m_mcnemar_p:.4e}")
    print(f"Motorcycle Bootstrap Delta:        {m_obs_delta:+.2f} pp (95% CI: [{m_ci_low:+.2f} pp, {m_ci_high:+.2f} pp])")
    print(f"Direct E2E Motorcycle Exact:       {m_data['motorcycle']['d_e2e']:.2f}%")
    print(f"Cascade E2E Motorcycle Exact:      {m_data['motorcycle']['c_e2e']:.2f}%")

    # 4. CASCADE INTERNAL MISS ATTRIBUTION
    c_misses = [r for r in records if not r["cascade_detected"]]
    miss_stages = Counter(r["cascade_miss_stage"] for r in c_misses)

    print("\n--- 4. CASCADE INTERNAL FAILURE ATTRIBUTION ---")
    print(f"Total Cascade Misses:                          {len(c_misses)} / {total_samples} ({len(c_misses)/total_samples*100:.1f}%)")
    print(f"  1. Vehicle Stage Miss (No vehicle ROI):      {miss_stages.get('VEHICLE_STAGE_MISS', 0):>4} ({miss_stages.get('VEHICLE_STAGE_MISS', 0)/len(c_misses)*100:.1f}%)")
    print(f"  2. Plate-in-ROI Miss (ROI exists, no plate): {miss_stages.get('PLATE_IN_ROI_MISS', 0):>4} ({miss_stages.get('PLATE_IN_ROI_MISS', 0)/len(c_misses)*100:.1f}%)")
    print(f"  3. Localization IoU Failure (IoU < 0.50):    {miss_stages.get('PLATE_LOCALIZATION_FAIL', 0):>4} ({miss_stages.get('PLATE_LOCALIZATION_FAIL', 0)/len(c_misses)*100:.1f}%)")
    print(f"  4. Other:                                    {miss_stages.get('OTHER', 0):>4} ({miss_stages.get('OTHER', 0)/len(c_misses)*100:.1f}%)")

    # 5. ORACLE COMPLEMENTARITY UNION
    oracle_union_det = sum(1 for r in records if r["direct_detected"] or r["cascade_detected"])
    oracle_union_rec = (oracle_union_det / total_samples) * 100.0
    oracle_e2e_corr = sum(1 for r in records if r["direct_e2e_correct"] or r["cascade_e2e_correct"])
    oracle_e2e_acc = (oracle_e2e_corr / total_samples) * 100.0

    print("\n--- 5. ORACLE COMPLEMENTARITY (DIRECT OR CASCADE UNION) ---")
    print(f"Oracle Union Detection Recall: {oracle_union_det} / {total_samples} ({oracle_union_rec:.2f}%) [Delta vs Direct: {oracle_union_rec - m_data['Overall']['d_rec']:+.2f} pp]")
    print(f"Oracle Union E2E Exact:        {oracle_e2e_corr} / {total_samples} ({oracle_e2e_acc:.2f}%) [Delta vs Direct: {oracle_e2e_acc - m_data['Overall']['d_e2e']:+.2f} pp]")

    # 6. DECISION GATE DETERMINATION
    # Gates: CASCADE_SUPPORTED, CASCADE_COMPLEMENTARY, CASCADE_NOT_SUPPORTED, VEHICLE_DETECTOR_BOTTLENECK
    veh_miss_ratio = miss_stages.get('VEHICLE_STAGE_MISS', 0) / len(c_misses) if c_misses else 0.0

    if m_data['Overall']['delta_rec'] > 5.0 and mcnemar_p < 0.05:
        decision = "CASCADE_SUPPORTED"
    elif m_data['motorcycle']['delta_rec'] > 10.0 and m_data['Overall']['delta_rec'] <= 0.0:
        decision = "CASCADE_COMPLEMENTARY"
    elif veh_miss_ratio >= 0.50 and m_data['Overall']['delta_rec'] < 0.0:
        decision = "VEHICLE_DETECTOR_BOTTLENECK"
    elif m_data['Overall']['delta_rec'] <= 0.0 and m_data['motorcycle']['delta_rec'] <= 0.0:
        decision = "CASCADE_NOT_SUPPORTED"
    else:
        decision = "CASCADE_COMPLEMENTARY"

    print("\n" + "=" * 95)
    print(f"SCIENTIFIC DECISION: [{decision}]")
    print("=" * 95)

    return {
        "m_data": m_data,
        "paired": {
            "both": both_c,
            "direct_only": direct_only_c,
            "cascade_only": cascade_only_c,
            "neither": neither_c,
            "net_rescue": net_rescue,
            "mcnemar_p": mcnemar_p,
            "obs_delta": obs_delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
        "motorcycle": {
            "d_det": m_data['motorcycle']['d_det'],
            "c_det": m_data['motorcycle']['c_det'],
            "d_rec": m_data['motorcycle']['d_rec'],
            "c_rec": m_data['motorcycle']['c_rec'],
            "d_e2e": m_data['motorcycle']['d_e2e'],
            "c_e2e": m_data['motorcycle']['c_e2e'],
            "rescues": m_states["CASCADE_ONLY"],
            "losses": m_states["DIRECT_ONLY"],
            "mcnemar_p": m_mcnemar_p,
            "delta": m_obs_delta,
            "ci_low": m_ci_low,
            "ci_high": m_ci_high,
        },
        "miss_stages": miss_stages,
        "oracle_union": {
            "det_recall": oracle_union_rec,
            "e2e_exact": oracle_e2e_acc,
        },
        "decision": decision,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Direct vs Cascade Results")
    parser.add_argument("--results", type=str, default="reports/direct_vs_cascade_results.csv")
    args = parser.parse_args()
    analyze_cascade_results(Path(args.results))
