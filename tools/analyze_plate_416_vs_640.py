"""
Comprehensive Analysis and Statistical Testing for Controlled Plate Detector Resolution Experiment (416 vs 640).
Reads reports/plate416_vs_640_results.csv (N=500).
Saves representative cases to outputs/plate640_eval/640_only/ and outputs/plate640_eval/416_only/.
"""

import argparse
import csv
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr.plate_text_utils import levenshtein_distance


def mcnemar_exact_test(b: int, c: int) -> float:
    """Computes exact two-tailed McNemar test p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2.0 * prob)


def paired_bootstrap_delta(hits_a: List[bool], hits_b: List[bool], n_boot: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    """Computes paired bootstrap delta (b - a) and 95% CI."""
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


def parse_box(b_str: str) -> List[float]:
    cleaned = b_str.strip().strip("[]()").replace(" ", "")
    return [float(x) for x in cleaned.split(",") if x]


def analyze_experiment(csv_path: Path = PROJECT_ROOT / "reports/plate416_vs_640_results.csv") -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in [
                "direct416_detected", "r1_416_detected", "r1_416_e2e", "cascade416_invoked",
                "direct640_detected", "r1_640_detected", "r1_640_e2e", "cascade640_invoked",
                "vehicle_found", "vehicle_roi_covers_gt", "nested416_detected", "nested640_detected",
                "valid_roi_plate416_miss", "mixed_detected", "mixed_e2e"
            ]:
                r[k] = (r[k].lower() == "true") if r.get(k) else False

            for k in ["direct416_conf", "direct416_iou", "r1_416_iou",
                      "direct640_conf", "direct640_iou", "r1_640_iou",
                      "plate_area", "latency416", "latency640"]:
                r[k] = float(r[k]) if r.get(k) else 0.0

            for k in ["direct416_candidates_count", "direct640_candidates_count"]:
                r[k] = int(r[k]) if r.get(k) else 0

            records.append(r)

    n_total = len(records)
    layouts = ["one_row", "two_rows", "motorcycle"]

    # 1. Detection & E2E Overall
    d416_hits = [r["direct416_detected"] for r in records]
    d640_hits = [r["direct640_detected"] for r in records]
    r416_hits = [r["r1_416_detected"] for r in records]
    r640_hits = [r["r1_640_detected"] for r in records]
    r416_e2e = [r["r1_416_e2e"] for r in records]
    r640_e2e = [r["r1_640_e2e"] for r in records]
    mixed_hits = [r["mixed_detected"] for r in records]
    mixed_e2e = [r["mixed_e2e"] for r in records]

    r416_rec = (sum(r416_hits) / n_total) * 100.0
    r640_rec = (sum(r640_hits) / n_total) * 100.0
    d416_rec = (sum(d416_hits) / n_total) * 100.0
    d640_rec = (sum(d640_hits) / n_total) * 100.0
    r416_acc = (sum(r416_e2e) / n_total) * 100.0
    r640_acc = (sum(r640_e2e) / n_total) * 100.0

    # 2. Paired Stats
    both_det = sum(1 for a, b in zip(r416_hits, r640_hits) if a and b)
    only416_det = sum(1 for a, b in zip(r416_hits, r640_hits) if a and not b)
    only640_det = sum(1 for a, b in zip(r416_hits, r640_hits) if not a and b)
    neither_det = sum(1 for a, b in zip(r416_hits, r640_hits) if not a and not b)
    mcn_det_p = mcnemar_exact_test(only416_det, only640_det)
    delta_rec, ci_l_rec, ci_h_rec = paired_bootstrap_delta(r416_hits, r640_hits)

    both_e2e = sum(1 for a, b in zip(r416_e2e, r640_e2e) if a and b)
    only416_e2e = sum(1 for a, b in zip(r416_e2e, r640_e2e) if a and not b)
    only640_e2e = sum(1 for a, b in zip(r416_e2e, r640_e2e) if not a and b)
    neither_e2e = sum(1 for a, b in zip(r416_e2e, r640_e2e) if not a and not b)
    mcn_e2e_p = mcnemar_exact_test(only416_e2e, only640_e2e)
    delta_e2e, ci_l_e2e, ci_h_e2e = paired_bootstrap_delta(r416_e2e, r640_e2e)

    # 3. Layout breakdown
    layout_data = {}
    for lay in layouts:
        l_recs = [r for r in records if r["layout"] == lay]
        n_l = len(l_recs)
        l_416 = [r["r1_416_detected"] for r in l_recs]
        l_640 = [r["r1_640_detected"] for r in l_recs]
        l_416_e = [r["r1_416_e2e"] for r in l_recs]
        l_640_e = [r["r1_640_e2e"] for r in l_recs]

        l_only416 = sum(1 for a, b in zip(l_416, l_640) if a and not b)
        l_only640 = sum(1 for a, b in zip(l_416, l_640) if not a and b)
        mcn_p = mcnemar_exact_test(l_only416, l_only640)
        d_rec, ci_l, ci_h = paired_bootstrap_delta(l_416, l_640)
        d_e, ci_le, ci_he = paired_bootstrap_delta(l_416_e, l_640_e)

        layout_data[lay] = {
            "n": n_l,
            "r416_det": sum(l_416), "r416_rec": (sum(l_416) / n_l) * 100.0,
            "r640_det": sum(l_640), "r640_rec": (sum(l_640) / n_l) * 100.0,
            "delta_rec": d_rec, "ci_l_rec": ci_l, "ci_h_rec": ci_h,
            "r416_e2e": sum(l_416_e), "r416_e2e_acc": (sum(l_416_e) / n_l) * 100.0,
            "r640_e2e": sum(l_640_e), "r640_e2e_acc": (sum(l_640_e) / n_l) * 100.0,
            "delta_e2e": d_e, "ci_l_e2e": ci_le, "ci_h_e2e": ci_he,
            "only416": l_only416, "only640": l_only640,
            "mcnemar_p": mcn_p,
        }

    # 4. Plate size tertiles
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
        s_416 = [r["r1_416_detected"] for r in s_recs]
        s_640 = [r["r1_640_detected"] for r in s_recs]
        s_only416 = sum(1 for a, b in zip(s_416, s_640) if a and not b)
        s_only640 = sum(1 for a, b in zip(s_416, s_640) if not a and b)
        mcn_p = mcnemar_exact_test(s_only416, s_only640)
        d_rec, ci_l, ci_h = paired_bootstrap_delta(s_416, s_640)

        size_data[s_name] = {
            "n": n_s,
            "r416_det": sum(s_416), "r416_rec": (sum(s_416) / n_s) * 100.0,
            "r640_det": sum(s_640), "r640_rec": (sum(s_640) / n_s) * 100.0,
            "delta_rec": d_rec, "ci_l_rec": ci_l, "ci_h_rec": ci_h,
            "only416": s_only416, "only640": s_only640,
            "mcnemar_p": mcn_p,
        }

    # 5. Direct-stage comparison
    direct_data = {
        "overall": {
            "d416_det": sum(d416_hits), "d416_rec": d416_rec,
            "d640_det": sum(d640_hits), "d640_rec": d640_rec,
            "delta": d640_rec - d416_rec,
        }
    }

    # 6. Fallback-stage analysis & Plate-in-ROI subset
    # Frames where cascade was invoked
    casc416_recs = [r for r in records if r["cascade416_invoked"]]
    casc640_recs = [r for r in records if r["cascade640_invoked"]]

    # Valid Vehicle ROI coverage among fallback frames
    valid_roi_416 = [r for r in casc416_recs if r["vehicle_roi_covers_gt"]]
    nested416_in_valid_roi = [r for r in valid_roi_416 if r["nested416_detected"]]

    valid_roi_640 = [r for r in casc640_recs if r["vehicle_roi_covers_gt"]]
    nested640_in_valid_roi = [r for r in valid_roi_640 if r["nested640_detected"]]

    # Section 22: Valid Vehicle ROI BUT Plate416 miss subset
    roi_plate416_miss_subset = [r for r in records if r["valid_roi_plate416_miss"]]
    plate640_rescues_in_subset = [r for r in roi_plate416_miss_subset if r["nested640_detected"]]
    still_misses_in_subset = len(roi_plate416_miss_subset) - len(plate640_rescues_in_subset)

    # 7. Localization: IoU on matched plates
    ious_416 = [r["r1_416_iou"] for r in records if r["r1_416_detected"]]
    ious_640 = [r["r1_640_iou"] for r in records if r["r1_640_detected"]]
    mean_iou_416 = float(np.mean(ious_416)) if ious_416 else 0.0
    mean_iou_640 = float(np.mean(ious_640)) if ious_640 else 0.0
    med_iou_416 = float(np.median(ious_416)) if ious_416 else 0.0
    med_iou_640 = float(np.median(ious_640)) if ious_640 else 0.0

    # 8. False positives
    fp_416 = sum(max(0, r["direct416_candidates_count"] - (1 if r["direct416_detected"] else 0)) for r in records)
    fp_640 = sum(max(0, r["direct640_candidates_count"] - (1 if r["direct640_detected"] else 0)) for r in records)
    tot_cands_416 = sum(r["direct416_candidates_count"] for r in records)
    tot_cands_640 = sum(r["direct640_candidates_count"] for r in records)
    prec_416 = (sum(d416_hits) / tot_cands_416 * 100.0) if tot_cands_416 > 0 else 0.0
    prec_640 = (sum(d640_hits) / tot_cands_640 * 100.0) if tot_cands_640 > 0 else 0.0

    # 9. OCR metrics
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

    gts = [r["ground_truth"] for r in records]
    preds_416 = [r["r1_416_ocr"] for r in records]
    preds_640 = [r["r1_640_ocr"] for r in records]
    char_acc_416, cer_416 = get_char_metrics(preds_416, gts)
    char_acc_640, cer_640 = get_char_metrics(preds_640, gts)

    ocr_given_det_416 = (sum(r416_e2e) / sum(r416_hits) * 100.0) if sum(r416_hits) > 0 else 0.0
    ocr_given_det_640 = (sum(r640_e2e) / sum(r640_hits) * 100.0) if sum(r640_hits) > 0 else 0.0

    # 10. Timing breakdown
    lats_416 = [r["latency416"] for r in records]
    lats_640 = [r["latency640"] for r in records]
    mean_lat_416 = float(np.mean(lats_416))
    mean_lat_640 = float(np.mean(lats_640))
    med_lat_416 = float(np.median(lats_416))
    med_lat_640 = float(np.median(lats_640))
    p95_lat_416 = float(np.percentile(lats_416, 95))
    p95_lat_640 = float(np.percentile(lats_640, 95))
    fps_416 = 1000.0 / mean_lat_416 if mean_lat_416 > 0 else 0.0
    fps_640 = 1000.0 / mean_lat_640 if mean_lat_640 > 0 else 0.0

    # 11. Diagnostic mixed arm
    mixed_rec = (sum(mixed_hits) / n_total) * 100.0
    mixed_acc = (sum(mixed_e2e) / n_total) * 100.0

    # 12. Save representative cases
    rescued_dir = PROJECT_ROOT / "outputs/plate640_eval/640_only"
    regression_dir = PROJECT_ROOT / "outputs/plate640_eval/416_only"
    rescued_dir.mkdir(parents=True, exist_ok=True)
    regression_dir.mkdir(parents=True, exist_ok=True)

    rescued_recs = [r for r in records if not r["r1_416_detected"] and r["r1_640_detected"]]
    regression_recs = [r for r in records if r["r1_416_detected"] and not r["r1_640_detected"]]

    # Save up to 5 representative rescues
    for i, r in enumerate(rescued_recs[:5]):
        p = Path(r["image_path"])
        if p.exists():
            img = Image.open(p).convert("RGB")
            draw = ImageDraw.Draw(img)
            gt_box = parse_box(r["gt_bbox"])
            draw.rectangle(gt_box, outline="green", width=2)
            if r.get("direct640_bbox"):
                box640 = parse_box(r["direct640_bbox"])
                draw.rectangle(box640, outline="cyan", width=2)
            s_id = r["sample_id"]
            img.save(rescued_dir / f"rescue_{i+1}_{r['layout']}_{s_id}.jpg")

    # Save up to 5 representative regressions
    for i, r in enumerate(regression_recs[:5]):
        p = Path(r["image_path"])
        if p.exists():
            img = Image.open(p).convert("RGB")
            draw = ImageDraw.Draw(img)
            gt_box = parse_box(r["gt_bbox"])
            draw.rectangle(gt_box, outline="green", width=2)
            if r.get("direct416_bbox"):
                box416 = parse_box(r["direct416_bbox"])
                draw.rectangle(box416, outline="red", width=2)
            s_id = r["sample_id"]
            img.save(regression_dir / f"regression_{i+1}_{r['layout']}_{s_id}.jpg")

    # 13. Decision Rule
    if only640_det > only416_det and delta_rec > 1.0 and mcn_det_p < 0.05 and (only416_det <= 5):
        decision = "PLATE640_SUPPORTED"
    elif only640_det > only416_det and (layout_data["motorcycle"]["delta_rec"] > 2.0 or size_data["small"]["delta_rec"] > 2.0):
        decision = "PLATE640_COMPLEMENTARY"
    elif delta_rec <= 0.0 or fps_640 < 5.0:
        decision = "PLATE640_NOT_SUPPORTED"
    else:
        decision = "PLATE640_SUPPORTED"

    return {
        "n_total": n_total,
        "plate416": {
            "det": sum(r416_hits), "rec": r416_rec,
            "e2e": sum(r416_e2e), "e2e_acc": r416_acc,
            "fp_count": fp_416, "fp_per_img": fp_416 / n_total, "precision": prec_416,
            "mean_iou": mean_iou_416, "med_iou": med_iou_416,
            "char_acc": char_acc_416, "cer": cer_416,
            "ocr_given_det": ocr_given_det_416,
            "cascade_invoked": len(casc416_recs),
            "cascade_pct": (len(casc416_recs) / n_total) * 100.0,
            "mean_lat": mean_lat_416, "med_lat": med_lat_416, "p95_lat": p95_lat_416, "fps": fps_416,
        },
        "plate640": {
            "det": sum(r640_hits), "rec": r640_rec,
            "e2e": sum(r640_e2e), "e2e_acc": r640_acc,
            "fp_count": fp_640, "fp_per_img": fp_640 / n_total, "precision": prec_640,
            "mean_iou": mean_iou_640, "med_iou": med_iou_640,
            "char_acc": char_acc_640, "cer": cer_640,
            "ocr_given_det": ocr_given_det_640,
            "cascade_invoked": len(casc640_recs),
            "cascade_pct": (len(casc640_recs) / n_total) * 100.0,
            "mean_lat": mean_lat_640, "med_lat": med_lat_640, "p95_lat": p95_lat_640, "fps": fps_640,
        },
        "mixed": {
            "det": sum(mixed_hits), "rec": mixed_rec,
            "e2e": sum(mixed_e2e), "e2e_acc": mixed_acc,
        },
        "paired": {
            "both_det": both_det, "only416_det": only416_det, "only640_det": only640_det, "neither_det": neither_det,
            "net_rescue_det": only640_det - only416_det, "mcnemar_det_p": mcn_det_p,
            "delta_rec": delta_rec, "ci_l_rec": ci_l_rec, "ci_h_rec": ci_h_rec,
            "both_e2e": both_e2e, "only416_e2e": only416_e2e, "only640_e2e": only640_e2e, "neither_e2e": neither_e2e,
            "net_rescue_e2e": only640_e2e - only416_e2e, "mcnemar_e2e_p": mcn_e2e_p,
            "delta_e2e": delta_e2e, "ci_l_e2e": ci_l_e2e, "ci_h_e2e": ci_h_e2e,
        },
        "layout": layout_data,
        "size": size_data,
        "direct_stage": direct_data,
        "fallback_stage": {
            "valid_roi_plate416_miss_count": len(roi_plate416_miss_subset),
            "plate640_rescued": len(plate640_rescues_in_subset),
            "still_misses": still_misses_in_subset,
        },
        "decision": decision,
    }


def print_summary(data: Dict[str, Any]):
    p4 = data["plate416"]
    p6 = data["plate640"]
    pr = data["paired"]
    fb = data["fallback_stage"]
    lay = data["layout"]
    sz = data["size"]

    print("=" * 80)
    print(f"CONTROLLED 416 vs 640 RESOLUTION EXPERIMENT SUMMARY (N={data['n_total']})")
    print("=" * 80)
    print(f"Plate416 R1 : Recall={p4['rec']:>6.2f}% ({p4['det']}/{data['n_total']}) | E2E={p4['e2e_acc']:>6.2f}% ({p4['e2e']}/{data['n_total']}) | FP/img={p4['fp_per_img']:.4f} | Latency={p4['mean_lat']:.1f}ms ({p4['fps']:.1f} FPS)")
    print(f"Plate640 R1 : Recall={p6['rec']:>6.2f}% ({p6['det']}/{data['n_total']}) | E2E={p6['e2e_acc']:>6.2f}% ({p6['e2e']}/{data['n_total']}) | FP/img={p6['fp_per_img']:.4f} | Latency={p6['mean_lat']:.1f}ms ({p6['fps']:.1f} FPS)")
    print(f"Delta       : Recall={pr['delta_rec']:>+6.2f} pp [95% CI: {pr['ci_l_rec']:+.2f}, {pr['ci_h_rec']:+.2f}] | E2E={pr['delta_e2e']:>+6.2f} pp [95% CI: {pr['ci_l_e2e']:+.2f}, {pr['ci_h_e2e']:+.2f}]")
    print(f"Paired Det  : Both={pr['both_det']} | 416_Only={pr['only416_det']} | 640_Only={pr['only640_det']} | Neither={pr['neither_det']} | McNemar p={pr['mcnemar_det_p']:.4e}")
    print("-" * 80)
    print("Sub-cohort Analysis:")
    print(f"  Motorcycle  : 416={lay['motorcycle']['r416_rec']:.2f}% | 640={lay['motorcycle']['r640_rec']:.2f}% | Delta={lay['motorcycle']['delta_rec']:+.2f} pp [CI: {lay['motorcycle']['ci_l_rec']:+.2f}, {lay['motorcycle']['ci_h_rec']:+.2f}] | p={lay['motorcycle']['mcnemar_p']:.4e}")
    print(f"  Two-Rows    : 416={lay['two_rows']['r416_rec']:.2f}% | 640={lay['two_rows']['r640_rec']:.2f}% | Delta={lay['two_rows']['delta_rec']:+.2f} pp [CI: {lay['two_rows']['ci_l_rec']:+.2f}, {lay['two_rows']['ci_h_rec']:+.2f}] | p={lay['two_rows']['mcnemar_p']:.4e}")
    print(f"  One-Row     : 416={lay['one_row']['r416_rec']:.2f}% | 640={lay['one_row']['r640_rec']:.2f}% | Delta={lay['one_row']['delta_rec']:+.2f} pp [CI: {lay['one_row']['ci_l_rec']:+.2f}, {lay['one_row']['ci_h_rec']:+.2f}] | p={lay['one_row']['mcnemar_p']:.4e}")
    print(f"  Small Plates: 416={sz['small']['r416_rec']:.2f}% | 640={sz['small']['r640_rec']:.2f}% | Delta={sz['small']['delta_rec']:+.2f} pp [CI: {sz['small']['ci_l_rec']:+.2f}, {sz['small']['ci_h_rec']:+.2f}]")
    print("-" * 80)
    print(f"Valid Vehicle ROI + Plate416 Miss Subset:")
    print(f"  N = {fb['valid_roi_plate416_miss_count']} | Rescued by 640 = {fb['plate640_rescued']} | Still Missed = {fb['still_misses']}")
    print("-" * 80)
    print(f"DECISION    : {data['decision']}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="reports/plate416_vs_640_results.csv")
    args = parser.parse_args()

    data = analyze_experiment(Path(args.csv))
    print_summary(data)
