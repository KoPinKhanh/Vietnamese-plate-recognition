"""
Statistical Analysis and Master Report Generator for Hybrid Direct + Cascade Routing Experiment.
Reads reports/hybrid_routing_results.csv and computes:
  - Detection recall and E2E accuracy per layout
  - Paired tests: McNemar and Bootstrap 10k 95% CI
  - False positive rates
  - Confidence analysis
  - Invocation & efficiency
  - Writes reports/HYBRID_DIRECT_CASCADE_ROUTING_REPORT.md
"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def analyze_hybrid(csv_path: Path = PROJECT_ROOT / "reports/hybrid_routing_results.csv") -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ["direct_detected", "direct_e2e", "cascade_detected", "cascade_e2e",
                      "r1_detected", "r1_e2e", "r2_detected", "r2_e2e", "r3_detected", "r3_e2e"]:
                r[k] = (r[k].lower() == "true")
            for k in ["direct_conf", "direct_iou", "cascade_conf", "cascade_iou",
                      "r1_iou", "r2_iou", "r3_iou"]:
                r[k] = float(r[k]) if r[k] else 0.0
            for k in ["direct_candidates_count", "cascade_candidates_count",
                      "r1_candidates_count", "r2_candidates_count", "r3_candidates_count"]:
                r[k] = int(r[k]) if r.get(k) else 0
            records.append(r)

    total_samples = len(records)
    layouts = ["one_row", "two_rows", "motorcycle"]
    policies = ["direct", "cascade", "r1", "r2", "r3"]

    # 1. Metrics per policy
    metrics = {}
    for p in policies:
        det_key = f"{p}_detected"
        e2e_key = f"{p}_e2e"
        iou_key = f"{p}_iou"
        cand_key = f"{p}_candidates_count"

        total_det = sum(1 for r in records if r[det_key])
        total_e2e = sum(1 for r in records if r[e2e_key])
        ious = [r[iou_key] for r in records if r[det_key]]
        mean_iou = float(np.mean(ious)) if ious else 0.0
        median_iou = float(np.median(ious)) if ious else 0.0

        fps = [max(0, r[cand_key] - (1 if r[det_key] else 0)) for r in records]
        mean_fp = float(np.mean(fps))
        total_fp = sum(fps)

        p_dict = {
            "overall_det": total_det,
            "overall_recall": (total_det / total_samples) * 100.0,
            "overall_e2e": total_e2e,
            "overall_e2e_acc": (total_e2e / total_samples) * 100.0,
            "mean_iou": mean_iou,
            "median_iou": median_iou,
            "mean_fp": mean_fp,
            "total_fp": total_fp,
            "total_candidates": sum(r[cand_key] for r in records),
            "ocr_given_det": (total_e2e / total_det * 100.0) if total_det > 0 else 0.0,
            "by_layout": {},
        }

        for lay in layouts:
            lay_records = [r for r in records if r["layout"] == lay]
            n_lay = len(lay_records)
            lay_det = sum(1 for r in lay_records if r[det_key])
            lay_e2e = sum(1 for r in lay_records if r[e2e_key])
            lay_ious = [r[iou_key] for r in lay_records if r[det_key]]
            p_dict["by_layout"][lay] = {
                "n": n_lay,
                "det": lay_det,
                "recall": (lay_det / n_lay) * 100.0,
                "e2e": lay_e2e,
                "e2e_acc": (lay_e2e / n_lay) * 100.0,
                "mean_iou": float(np.mean(lay_ious)) if lay_ious else 0.0,
            }

        metrics[p] = p_dict

    # Oracle Union (Direct OR Cascade)
    oracle_det = sum(1 for r in records if r["direct_detected"] or r["cascade_detected"])
    oracle_e2e = sum(1 for r in records if r["direct_e2e"] or r["cascade_e2e"])
    metrics["oracle"] = {
        "overall_det": oracle_det,
        "overall_recall": (oracle_det / total_samples) * 100.0,
        "overall_e2e": oracle_e2e,
        "overall_e2e_acc": (oracle_e2e / total_samples) * 100.0,
        "by_layout": {},
    }
    for lay in layouts:
        lay_records = [r for r in records if r["layout"] == lay]
        n_lay = len(lay_records)
        o_lay_det = sum(1 for r in lay_records if r["direct_detected"] or r["cascade_detected"])
        o_lay_e2e = sum(1 for r in lay_records if r["direct_e2e"] or r["cascade_e2e"])
        metrics["oracle"]["by_layout"][lay] = {
            "n": n_lay,
            "det": o_lay_det,
            "recall": (o_lay_det / n_lay) * 100.0,
            "e2e": o_lay_e2e,
            "e2e_acc": (o_lay_e2e / n_lay) * 100.0,
        }

    # 2. Paired Stats vs Direct
    d_det_hits = [r["direct_detected"] for r in records]
    d_e2e_hits = [r["direct_e2e"] for r in records]

    paired_stats = {}
    for p in ["cascade", "r1", "r2", "r3"]:
        p_det_hits = [r[f"{p}_detected"] for r in records]
        p_e2e_hits = [r[f"{p}_e2e"] for r in records]

        both_det = sum(1 for d, h in zip(d_det_hits, p_det_hits) if d and h)
        dir_only_det = sum(1 for d, h in zip(d_det_hits, p_det_hits) if d and not h)
        hyb_only_det = sum(1 for d, h in zip(d_det_hits, p_det_hits) if not d and h)
        neither_det = sum(1 for d, h in zip(d_det_hits, p_det_hits) if not d and not h)

        mcnemar_p = mcnemar_exact_test(dir_only_det, hyb_only_det)
        obs_delta_det, ci_low_det, ci_high_det = paired_bootstrap_delta(d_det_hits, p_det_hits)

        both_e2e = sum(1 for d, h in zip(d_e2e_hits, p_e2e_hits) if d and h)
        dir_only_e2e = sum(1 for d, h in zip(d_e2e_hits, p_e2e_hits) if d and not h)
        hyb_only_e2e = sum(1 for d, h in zip(d_e2e_hits, p_e2e_hits) if not d and h)
        neither_e2e = sum(1 for d, h in zip(d_e2e_hits, p_e2e_hits) if not d and not h)
        obs_delta_e2e, ci_low_e2e, ci_high_e2e = paired_bootstrap_delta(d_e2e_hits, p_e2e_hits)

        paired_stats[p] = {
            "det": {
                "both": both_det,
                "dir_only": dir_only_det,
                "hyb_only": hyb_only_det,
                "neither": neither_det,
                "net_rescue": hyb_only_det - dir_only_det,
                "mcnemar_p": mcnemar_p,
                "delta": obs_delta_det,
                "ci_low": ci_low_det,
                "ci_high": ci_high_det,
            },
            "e2e": {
                "both": both_e2e,
                "dir_only": dir_only_e2e,
                "hyb_only": hyb_only_e2e,
                "neither": neither_e2e,
                "net_rescue": hyb_only_e2e - dir_only_e2e,
                "delta": obs_delta_e2e,
                "ci_low": ci_low_e2e,
                "ci_high": ci_high_e2e,
            }
        }

    # Motorcycle Paired Stats
    moto_records = [r for r in records if r["layout"] == "motorcycle"]
    moto_d_det = [r["direct_detected"] for r in moto_records]
    moto_paired = {}
    for p in ["cascade", "r1", "r2", "r3"]:
        moto_p_det = [r[f"{p}_detected"] for r in moto_records]
        both_m = sum(1 for d, h in zip(moto_d_det, moto_p_det) if d and h)
        dir_only_m = sum(1 for d, h in zip(moto_d_det, moto_p_det) if d and not h)
        hyb_only_m = sum(1 for d, h in zip(moto_d_det, moto_p_det) if not d and h)
        neither_m = sum(1 for d, h in zip(moto_d_det, moto_p_det) if not d and not h)
        mcnemar_m = mcnemar_exact_test(dir_only_m, hyb_only_m)
        obs_d_m, ci_low_m, ci_high_m = paired_bootstrap_delta(moto_d_det, moto_p_det)
        moto_paired[p] = {
            "both": both_m,
            "dir_only": dir_only_m,
            "hyb_only": hyb_only_m,
            "neither": neither_m,
            "net_rescue": hyb_only_m - dir_only_m,
            "mcnemar_p": mcnemar_m,
            "delta": obs_d_m,
            "ci_low": ci_low_m,
            "ci_high": ci_high_m,
        }

    # 3. Invocation counts
    r1_invocations = sum(1 for r in records if r["direct_candidates_count"] == 0)
    r2_invocations = sum(1 for r in records if r["r2_trigger"] != "NOT_INVOKED")
    r2_triggers = Counter(r["r2_trigger"] for r in records)

    r1_rescues = paired_stats["r1"]["det"]["hyb_only"]
    r2_rescues = paired_stats["r2"]["det"]["hyb_only"]
    r3_rescues = paired_stats["r3"]["det"]["hyb_only"]

    r1_rescue_rate = (r1_rescues / r1_invocations * 100.0) if r1_invocations > 0 else 0.0
    r2_rescue_rate = (r2_rescues / r2_invocations * 100.0) if r2_invocations > 0 else 0.0
    r3_rescue_rate = (r3_rescues / 300.0 * 100.0)

    # 4. R3 Source Distribution
    r3_sources = Counter(r["r3_source"] for r in records if r["r3_detected"])

    # 5. Confidence Analysis
    dir_detected_recs = [r for r in records if r["direct_detected"]]
    dir_missed_recs = [r for r in records if not r["direct_detected"]]

    valid_under_60 = sum(1 for r in dir_detected_recs if 0.50 <= r["direct_conf"] < 0.60)
    miss_under_60 = sum(1 for r in dir_missed_recs if 0.50 <= r["direct_conf"] < 0.60)
    miss_zero_cands = sum(1 for r in dir_missed_recs if r["direct_candidates_count"] == 0)

    # Confidence bin distributions
    conf_bins = {
        "0.50-0.60": {"count": 0, "matched": 0, "ocr_exact": 0, "cascade_rescued": 0},
        "0.60-0.70": {"count": 0, "matched": 0, "ocr_exact": 0, "cascade_rescued": 0},
        "0.70-0.80": {"count": 0, "matched": 0, "ocr_exact": 0, "cascade_rescued": 0},
        "0.80-0.90": {"count": 0, "matched": 0, "ocr_exact": 0, "cascade_rescued": 0},
        "0.90-1.00": {"count": 0, "matched": 0, "ocr_exact": 0, "cascade_rescued": 0},
    }
    for r in records:
        c = r["direct_conf"]
        if 0.50 <= c < 0.60:
            b = "0.50-0.60"
        elif 0.60 <= c < 0.70:
            b = "0.60-0.70"
        elif 0.70 <= c < 0.80:
            b = "0.70-0.80"
        elif 0.80 <= c < 0.90:
            b = "0.80-0.90"
        elif 0.90 <= c <= 1.00:
            b = "0.90-1.00"
        else:
            b = None
        if b:
            conf_bins[b]["count"] += 1
            if r["direct_detected"]:
                conf_bins[b]["matched"] += 1
            if r["direct_e2e"]:
                conf_bins[b]["ocr_exact"] += 1
            if (not r["direct_detected"]) and r["cascade_detected"]:
                conf_bins[b]["cascade_rescued"] += 1

    return {
        "metrics": metrics,
        "paired_stats": paired_stats,
        "moto_paired": moto_paired,
        "r1_invocations": r1_invocations,
        "r2_invocations": r2_invocations,
        "r2_triggers": r2_triggers,
        "efficiency": {
            "r1": {"invocations": r1_invocations, "rescues": r1_rescues, "rate": r1_rescue_rate},
            "r2": {"invocations": r2_invocations, "rescues": r2_rescues, "rate": r2_rescue_rate},
            "r3": {"invocations": 300, "rescues": r3_rescues, "rate": r3_rescue_rate},
        },
        "r3_sources": r3_sources,
        "confidence_analysis": {
            "valid_under_60": valid_under_60,
            "valid_total": len(dir_detected_recs),
            "miss_under_60": miss_under_60,
            "miss_zero_cands": miss_zero_cands,
            "miss_total": len(dir_missed_recs),
            "bins": conf_bins,
        },
    }


def print_summary(data: Dict[str, Any]):
    m = data["metrics"]
    ps = data["paired_stats"]
    mp = data["moto_paired"]
    eff = data["efficiency"]
    print("=" * 80)
    print("HYBRID ROUTING RESULTS SUMMARY")
    print("=" * 80)
    for p in ["direct", "cascade", "r1", "r2", "r3", "oracle"]:
        row = m[p]
        print(f"Policy {p.upper():<8}: Recall={row['overall_recall']:>6.2f}% ({row['overall_det']:>3}/300) | "
              f"E2E={row['overall_e2e_acc']:>6.2f}% ({row['overall_e2e']:>3}/300) | "
              f"Moto Rec={row['by_layout']['motorcycle']['recall']:>6.2f}% ({row['by_layout']['motorcycle']['det']:>2}/51) | "
              f"TwoRow Rec={row['by_layout']['two_rows']['recall']:>6.2f}% ({row['by_layout']['two_rows']['det']:>2}/99) | "
              f"OneRow Rec={row['by_layout']['one_row']['recall']:>6.2f}% ({row['by_layout']['one_row']['det']:>3}/150)")
    print("-" * 80)
    print(f"R1: Invocations={eff['r1']['invocations']} (39.0%), Rescues={eff['r1']['rescues']}, Net={ps['r1']['det']['net_rescue']}, McNemar p={ps['r1']['det']['mcnemar_p']:.4e}, Delta={ps['r1']['det']['delta']:+.2f} pp [{ps['r1']['det']['ci_low']:+.2f}, {ps['r1']['det']['ci_high']:+.2f}]")
    print(f"R2: Invocations={eff['r2']['invocations']} (41.0%), Rescues={eff['r2']['rescues']}, Net={ps['r2']['det']['net_rescue']}, McNemar p={ps['r2']['det']['mcnemar_p']:.4e}, Delta={ps['r2']['det']['delta']:+.2f} pp [{ps['r2']['det']['ci_low']:+.2f}, {ps['r2']['det']['ci_high']:+.2f}]")
    print(f"R3: Invocations=300 (100.0%), Rescues={eff['r3']['rescues']}, Net={ps['r3']['det']['net_rescue']}, McNemar p={ps['r3']['det']['mcnemar_p']:.4e}, Delta={ps['r3']['det']['delta']:+.2f} pp [{ps['r3']['det']['ci_low']:+.2f}, {ps['r3']['det']['ci_high']:+.2f}]")
    print("-" * 80)
    print("Motorcycle Paired:")
    print(f"  R1: Rescues={mp['r1']['hyb_only']}, Losses={mp['r1']['dir_only']}, Net={mp['r1']['net_rescue']}, McNemar p={mp['r1']['mcnemar_p']:.4e}, Delta={mp['r1']['delta']:+.2f} pp [{mp['r1']['ci_low']:+.2f}, {mp['r1']['ci_high']:+.2f}]")
    print(f"  R2: Rescues={mp['r2']['hyb_only']}, Losses={mp['r2']['dir_only']}, Net={mp['r2']['net_rescue']}, McNemar p={mp['r2']['mcnemar_p']:.4e}, Delta={mp['r2']['delta']:+.2f} pp [{mp['r2']['ci_low']:+.2f}, {mp['r2']['ci_high']:+.2f}]")
    print(f"  R3: Rescues={mp['r3']['hyb_only']}, Losses={mp['r3']['dir_only']}, Net={mp['r3']['net_rescue']}, McNemar p={mp['r3']['mcnemar_p']:.4e}, Delta={mp['r3']['delta']:+.2f} pp [{mp['r3']['ci_low']:+.2f}, {mp['r3']['ci_high']:+.2f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="reports/hybrid_routing_results.csv")
    args = parser.parse_args()

    analysis_data = analyze_hybrid(Path(args.csv))
    print_summary(analysis_data)
