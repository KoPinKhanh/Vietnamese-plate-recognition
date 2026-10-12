"""
Statistical Analysis and Master Report Generator for Final Locked Routing Benchmark.
Reads reports/final_locked_routing_results.csv (N=3,727 canonical test samples).
"""

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr.plate_text_utils import levenshtein_distance


def mcnemar_exact_test(b: int, c: int) -> float:
    """Computes exact two-tailed McNemar test p-value using Binomial distribution."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2.0 * prob)


def paired_bootstrap_delta(hits_a: List[bool], hits_b: List[bool], n_boot: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    """Computes paired bootstrap delta (b - a) and 95% confidence interval."""
    np.random.seed(seed)
    n = len(hits_a)
    a_arr = np.array(hits_a, dtype=np.float32)
    b_arr = np.array(hits_b, dtype=np.float32)
    obs_delta = (b_arr.mean() - a_arr.mean()) * 100.0

    indices = np.random.randint(0, n, size=(n_boot, n))
    a_boot = a_arr[indices].mean(axis=1) * 100.0
    b_boot = b_arr[indices].mean(axis=1) * 100.0
    delta_boot = b_boot - a_boot

    ci_low = float(np.percentile(delta_boot, 2.5))
    ci_high = float(np.percentile(delta_boot, 97.5))
    return obs_delta, ci_low, ci_high


def analyze_final(csv_path: Path = PROJECT_ROOT / "reports/final_locked_routing_results.csv") -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ["direct_detected", "direct_e2e", "r1_detected", "r1_e2e", "cascade_invoked", "vehicle_found"]:
                r[k] = (r[k].lower() == "true")
            for k in ["direct_conf", "direct_iou", "direct_ocr_conf", "r1_conf", "r1_iou", "r1_ocr_conf",
                      "direct_ms", "r1_ms", "plate_area", "vehicle_conf"]:
                r[k] = float(r[k]) if r.get(k) else 0.0
            for k in ["direct_candidates_count", "r1_candidates_count"]:
                r[k] = int(r[k]) if r.get(k) else 0
            records.append(r)

    total_samples = len(records)
    layouts = ["one_row", "two_rows", "motorcycle"]

    # 1. Detection & E2E Counts
    d_hits = [r["direct_detected"] for r in records]
    r1_hits = [r["r1_detected"] for r in records]
    d_e2e = [r["direct_e2e"] for r in records]
    r1_e2e = [r["r1_e2e"] for r in records]

    d_det_count = sum(d_hits)
    r1_det_count = sum(r1_hits)
    d_e2e_count = sum(d_e2e)
    r1_e2e_count = sum(r1_e2e)

    d_recall = (d_det_count / total_samples) * 100.0
    r1_recall = (r1_det_count / total_samples) * 100.0
    d_e2e_acc = (d_e2e_count / total_samples) * 100.0
    r1_e2e_acc = (r1_e2e_count / total_samples) * 100.0

    d_ious = [r["direct_iou"] for r in records if r["direct_detected"]]
    r1_ious = [r["r1_iou"] for r in records if r["r1_detected"]]
    d_mean_iou = float(np.mean(d_ious)) if d_ious else 0.0
    r1_mean_iou = float(np.mean(r1_ious)) if r1_ious else 0.0
    d_med_iou = float(np.median(d_ious)) if d_ious else 0.0
    r1_med_iou = float(np.median(r1_ious)) if r1_ious else 0.0

    d_fps = [max(0, r["direct_candidates_count"] - (1 if r["direct_detected"] else 0)) for r in records]
    r1_fps = [max(0, r["r1_candidates_count"] - (1 if r["r1_detected"] else 0)) for r in records]
    d_mean_fp = float(np.mean(d_fps))
    r1_mean_fp = float(np.mean(r1_fps))
    d_total_fp = sum(d_fps)
    r1_total_fp = sum(r1_fps)

    d_total_cands = sum(r["direct_candidates_count"] for r in records)
    r1_total_cands = sum(r["r1_candidates_count"] for r in records)
    d_precision = (d_det_count / d_total_cands * 100.0) if d_total_cands > 0 else 0.0
    r1_precision = (r1_det_count / r1_total_cands * 100.0) if r1_total_cands > 0 else 0.0

    d_ocr_given_det = (d_e2e_count / d_det_count * 100.0) if d_det_count > 0 else 0.0
    r1_ocr_given_det = (r1_e2e_count / r1_det_count * 100.0) if r1_det_count > 0 else 0.0

    # Character metrics
    def get_char_metrics(preds, gts):
        total_chars = sum(len(g) for g in gts)
        total_ed = 0
        correct_chars = 0
        for p, g in zip(preds, gts):
            ed = levenshtein_distance(p, g)
            total_ed += ed
            correct_chars += max(0, len(g) - ed)
        cer = (total_ed / total_chars * 100.0) if total_chars > 0 else 0.0
        char_acc = (correct_chars / total_chars * 100.0) if total_chars > 0 else 0.0
        return char_acc, cer

    d_preds = [r["direct_ocr"] for r in records]
    r1_preds = [r["r1_ocr"] for r in records]
    gts = [r["ground_truth"] for r in records]

    d_char_acc, d_cer = get_char_metrics(d_preds, gts)
    r1_char_acc, r1_cer = get_char_metrics(r1_preds, gts)

    # 2. Paired Stats
    both_det = sum(1 for d, r in zip(d_hits, r1_hits) if d and r)
    dir_only_det = sum(1 for d, r in zip(d_hits, r1_hits) if d and not r)
    r1_only_det = sum(1 for d, r in zip(d_hits, r1_hits) if not d and r)
    neither_det = sum(1 for d, r in zip(d_hits, r1_hits) if not d and not r)
    mcnemar_det_p = mcnemar_exact_test(dir_only_det, r1_only_det)
    obs_delta_det, ci_low_det, ci_high_det = paired_bootstrap_delta(d_hits, r1_hits)

    both_e2e = sum(1 for d, r in zip(d_e2e, r1_e2e) if d and r)
    dir_only_e2e = sum(1 for d, r in zip(d_e2e, r1_e2e) if d and not r)
    r1_only_e2e = sum(1 for d, r in zip(d_e2e, r1_e2e) if not d and r)
    neither_e2e = sum(1 for d, r in zip(d_e2e, r1_e2e) if not d and not r)
    mcnemar_e2e_p = mcnemar_exact_test(dir_only_e2e, r1_only_e2e)
    obs_delta_e2e, ci_low_e2e, ci_high_e2e = paired_bootstrap_delta(d_e2e, r1_e2e)

    # 3. Layout breakdown
    layout_data = {}
    for lay in layouts:
        lay_recs = [r for r in records if r["gt_layout"] == lay]
        n_lay = len(lay_recs)
        lay_d_hits = [r["direct_detected"] for r in lay_recs]
        lay_r1_hits = [r["r1_detected"] for r in lay_recs]
        lay_d_e2e = [r["direct_e2e"] for r in lay_recs]
        lay_r1_e2e = [r["r1_e2e"] for r in lay_recs]

        lay_d_det = sum(lay_d_hits)
        lay_r1_det = sum(lay_r1_hits)
        lay_d_e = sum(lay_d_e2e)
        lay_r1_e = sum(lay_r1_e2e)

        both_l = sum(1 for d, r in zip(lay_d_hits, lay_r1_hits) if d and r)
        dir_only_l = sum(1 for d, r in zip(lay_d_hits, lay_r1_hits) if d and not r)
        r1_only_l = sum(1 for d, r in zip(lay_d_hits, lay_r1_hits) if not d and r)
        neither_l = sum(1 for d, r in zip(lay_d_hits, lay_r1_hits) if not d and not r)

        both_le = sum(1 for d, r in zip(lay_d_e2e, lay_r1_e2e) if d and r)
        dir_only_le = sum(1 for d, r in zip(lay_d_e2e, lay_r1_e2e) if d and not r)
        r1_only_le = sum(1 for d, r in zip(lay_d_e2e, lay_r1_e2e) if not d and r)
        neither_le = sum(1 for d, r in zip(lay_d_e2e, lay_r1_e2e) if not d and not r)

        mcn_p = mcnemar_exact_test(dir_only_l, r1_only_l)
        delta_l, ci_low_l, ci_high_l = paired_bootstrap_delta(lay_d_hits, lay_r1_hits)
        delta_e, ci_low_e, ci_high_e = paired_bootstrap_delta(lay_d_e2e, lay_r1_e2e)

        layout_data[lay] = {
            "n": n_lay,
            "d_det": lay_d_det,
            "d_rec": (lay_d_det / n_lay) * 100.0,
            "r1_det": lay_r1_det,
            "r1_rec": (lay_r1_det / n_lay) * 100.0,
            "d_e2e": lay_d_e,
            "d_e2e_acc": (lay_d_e / n_lay) * 100.0,
            "r1_e2e": lay_r1_e,
            "r1_e2e_acc": (lay_r1_e / n_lay) * 100.0,
            "both": both_l,
            "dir_only": dir_only_l,
            "r1_only": r1_only_l,
            "neither": neither_l,
            "net_rescue": r1_only_l - dir_only_l,
            "both_e2e": both_le,
            "dir_only_e2e": dir_only_le,
            "r1_only_e2e": r1_only_le,
            "neither_e2e": neither_le,
            "mcnemar_p": mcn_p,
            "delta_rec": delta_l,
            "ci_low_rec": ci_low_l,
            "ci_high_rec": ci_high_l,
            "delta_e2e": delta_e,
            "ci_low_e2e": ci_low_e,
            "ci_high_e2e": ci_high_e,
        }

    # 4. Plate size analysis (Tertiles of plate_area)
    areas = sorted([r["plate_area"] for r in records])
    t1 = np.percentile(areas, 33.33)
    t2 = np.percentile(areas, 66.67)

    size_bins = {"small": [], "medium": [], "large": []}
    for r in records:
        a = r["plate_area"]
        if a <= t1:
            size_bins["small"].append(r)
        elif a <= t2:
            size_bins["medium"].append(r)
        else:
            size_bins["large"].append(r)

    size_data = {}
    for s_name, s_recs in size_bins.items():
        n_s = len(s_recs)
        s_d_hits = [r["direct_detected"] for r in s_recs]
        s_r1_hits = [r["r1_detected"] for r in s_recs]
        s_d_det = sum(s_d_hits)
        s_r1_det = sum(s_r1_hits)
        s_rescues = sum(1 for d, r in zip(s_d_hits, s_r1_hits) if not d and r)
        s_losses = sum(1 for d, r in zip(s_d_hits, s_r1_hits) if d and not r)
        size_data[s_name] = {
            "n": n_s,
            "d_rec": (s_d_det / n_s) * 100.0,
            "r1_rec": (s_r1_det / n_s) * 100.0,
            "delta": ((s_r1_det - s_d_det) / n_s) * 100.0,
            "rescues": s_rescues,
            "losses": s_losses,
        }

    # 5. Fallback internal attribution
    invoked_recs = [r for r in records if r["cascade_invoked"]]
    total_invocations = len(invoked_recs)
    rescued_count = r1_only_det
    rescue_rate = (rescued_count / total_invocations * 100.0) if total_invocations > 0 else 0.0
    miss_stages = Counter(r["failure_stage"] for r in invoked_recs if not r["r1_detected"])

    # 6. Standardized Timing
    d_latencies = [r["direct_ms"] for r in records]
    r1_latencies = [r["r1_ms"] for r in records]
    d_mean_lat = float(np.mean(d_latencies))
    r1_mean_lat = float(np.mean(r1_latencies))
    d_med_lat = float(np.median(d_latencies))
    r1_med_lat = float(np.median(r1_latencies))
    d_p95_lat = float(np.percentile(d_latencies, 95))
    r1_p95_lat = float(np.percentile(r1_latencies, 95))
    d_fps = 1000.0 / d_mean_lat if d_mean_lat > 0 else 0.0
    r1_fps = 1000.0 / r1_mean_lat if r1_mean_lat > 0 else 0.0

    # 7. Decision
    if obs_delta_det > 3.0 and dir_only_det == 0 and mcnemar_det_p < 0.05:
        decision = "ROUTING_FINAL_CONFIRMED"
    elif layout_data["motorcycle"]["delta_rec"] > 5.0 and obs_delta_det >= 0.0:
        decision = "ROUTING_FINAL_PARTIAL"
    else:
        decision = "ROUTING_FINAL_NOT_CONFIRMED"

    return {
        "total_samples": total_samples,
        "direct": {
            "det": d_det_count, "rec": d_recall,
            "e2e": d_e2e_count, "e2e_acc": d_e2e_acc,
            "precision": d_precision, "mean_fp": d_mean_fp, "total_fp": d_total_fp,
            "mean_iou": d_mean_iou, "med_iou": d_med_iou,
            "char_acc": d_char_acc, "cer": d_cer,
            "ocr_given_det": d_ocr_given_det,
            "mean_lat": d_mean_lat, "med_lat": d_med_lat, "p95_lat": d_p95_lat, "fps": d_fps,
        },
        "r1": {
            "det": r1_det_count, "rec": r1_recall,
            "e2e": r1_e2e_count, "e2e_acc": r1_e2e_acc,
            "precision": r1_precision, "mean_fp": r1_mean_fp, "total_fp": r1_total_fp,
            "mean_iou": r1_mean_iou, "med_iou": r1_med_iou,
            "char_acc": r1_char_acc, "cer": r1_cer,
            "ocr_given_det": r1_ocr_given_det,
            "mean_lat": r1_mean_lat, "med_lat": r1_med_lat, "p95_lat": r1_p95_lat, "fps": r1_fps,
        },
        "paired": {
            "both_det": both_det, "dir_only_det": dir_only_det, "r1_only_det": r1_only_det, "neither_det": neither_det,
            "net_rescue_det": r1_only_det - dir_only_det,
            "mcnemar_det_p": mcnemar_det_p,
            "delta_rec": obs_delta_det, "ci_low_rec": ci_low_det, "ci_high_rec": ci_high_det,
            "both_e2e": both_e2e, "dir_only_e2e": dir_only_e2e, "r1_only_e2e": r1_only_e2e, "neither_e2e": neither_e2e,
            "net_rescue_e2e": r1_only_e2e - dir_only_e2e,
            "mcnemar_e2e_p": mcnemar_e2e_p,
            "delta_e2e": obs_delta_e2e, "ci_low_e2e": ci_low_e2e, "ci_high_e2e": ci_high_e2e,
        },
        "layout": layout_data,
        "size": size_data,
        "fallback": {
            "invocations": total_invocations,
            "invocations_pct": (total_invocations / total_samples) * 100.0,
            "rescues": rescued_count,
            "rescue_rate": rescue_rate,
            "miss_stages": miss_stages,
        },
        "decision": decision,
    }


def print_final_summary(data: Dict[str, Any]):
    d = data["direct"]
    r = data["r1"]
    p = data["paired"]
    l = data["layout"]
    f = data["fallback"]

    print("=" * 80)
    print(f"CANONICAL CLEAN TEST PARTITION (N={data['total_samples']}) SUMMARY")
    print("=" * 80)
    print(f"Direct Baseline : Recall={d['rec']:>6.2f}% ({d['det']:>4}/{data['total_samples']}) | E2E={d['e2e_acc']:>6.2f}% ({d['e2e']:>4}/{data['total_samples']}) | FP/img={d['mean_fp']:.4f} | Latency={d['mean_lat']:.1f}ms ({d['fps']:.1f} FPS)")
    print(f"Locked R1       : Recall={r['rec']:>6.2f}% ({r['det']:>4}/{data['total_samples']}) | E2E={r['e2e_acc']:>6.2f}% ({r['e2e']:>4}/{data['total_samples']}) | FP/img={r['mean_fp']:.4f} | Latency={r['mean_lat']:.1f}ms ({r['fps']:.1f} FPS)")
    print(f"Delta (R1-Direct): Recall={p['delta_rec']:>+6.2f} pp [95% CI: {p['ci_low_rec']:+.2f}, {p['ci_high_rec']:+.2f}] | E2E={p['delta_e2e']:>+6.2f} pp [95% CI: {p['ci_low_e2e']:+.2f}, {p['ci_high_e2e']:+.2f}]")
    print(f"Paired States   : Both={p['both_det']} | Regressions={p['dir_only_det']} | Rescues={p['r1_only_det']} | Neither={p['neither_det']} | McNemar p={p['mcnemar_det_p']:.4e}")
    print(f"Paired E2E      : Both={p['both_e2e']} | Regressions={p['dir_only_e2e']} | Rescues={p['r1_only_e2e']} | Neither={p['neither_e2e']} | McNemar p={p['mcnemar_e2e_p']:.4e}")
    print("-" * 80)
    print("Layout Breakdown:")
    for lay_name in ["one_row", "two_rows", "motorcycle"]:
        row = l[lay_name]
        print(f"  {lay_name:<12}: Direct={row['d_rec']:>6.2f}% ({row['d_det']:>4}/{row['n']}) | R1={row['r1_rec']:>6.2f}% ({row['r1_det']:>4}/{row['n']}) | Delta={row['delta_rec']:>+6.2f} pp [CI: {row['ci_low_rec']:+.2f}, {row['ci_high_rec']:+.2f}] | Rescues={row['r1_only']} | Losses={row['dir_only']}")
        print(f"  {' '*12}  E2E: Direct={row['d_e2e_acc']:>6.2f}% ({row['d_e2e']:>4}/{row['n']}) | R1={row['r1_e2e_acc']:>6.2f}% ({row['r1_e2e']:>4}/{row['n']}) | Delta={row['delta_e2e']:>+6.2f} pp [CI: {row['ci_low_e2e']:+.2f}, {row['ci_high_e2e']:+.2f}] | Rescues={row['r1_only_e2e']} | Losses={row['dir_only_e2e']}")
    print("-" * 80)
    print("Plate Size Analysis:")
    for s_name, s in data["size"].items():
        print(f"  {s_name:<8}: N={s['n']} | Direct={s['d_rec']:.2f}% | R1={s['r1_rec']:.2f}% | Delta={s['delta']:+.2f} pp | Rescues={s['rescues']}")
    print("-" * 80)
    print(f"Cascade Invocations : {f['invocations']} / {data['total_samples']} ({f['invocations_pct']:.1f}%) | Rescues={f['rescues']} | Rescue Rate={f['rescue_rate']:.1f}%")
    print(f"Miss Attribution    : {dict(f['miss_stages'])}")
    print(f"DECISION            : {data['decision']}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="reports/final_locked_routing_results.csv")
    args = parser.parse_args()

    data = analyze_final(Path(args.csv))
    print_final_summary(data)
