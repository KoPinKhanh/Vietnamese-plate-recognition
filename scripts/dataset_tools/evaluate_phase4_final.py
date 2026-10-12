"""
Phase 4 Final Evaluation and Statistical Significance Analysis.
Evaluates:
  1. Baseline V1 (weights/ocr/best_lprnet_vn.pth)
  2. Phase 3 Candidate FT-B (weights/ocr/experiments/phase3_ft_b/best.pth)
  3. Phase 4 Selected Candidate (.pth specified by --cand_ckpt)

Evaluated Sets:
  - TEST_FINAL (datasets/VNLP_processed/test.csv)
  - TEST_CLEAN (datasets/VNLP_processed/test_clean.csv)
  - TEST_UNSEEN_TEXT (datasets/VNLP_processed/test_unseen_text.csv)

Outputs:
  - reports/phase4_fixed_cases.csv
  - reports/phase4_regressions.csv
  - McNemar exact test (two-sided binomial)
  - 10,000 paired bootstrap 95% Confidence Intervals
"""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, List, Any, Tuple

import numpy as np
import torch

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.charset import NUM_CLASSES
from ocr.models.lprnet_lite import LPRNetLite
from tools.evaluate_ocr_model import evaluate_manifest


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def mcnemar_exact_test(b: int, c: int) -> float:
    """Computes exact two-sided p-value for McNemar test using Binomial distribution."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Exact two-sided p-value under H0: p = 0.5
    prob_tails = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p_val = min(1.0, 2.0 * prob_tails)
    return p_val


def paired_bootstrap_ci(
    y1_correct: np.ndarray,
    y2_correct: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes 95% Paired Bootstrap Confidence Interval for (Accuracy2 - Accuracy1).
    Returns (delta_mean_pp, ci_lower_pp, ci_upper_pp).
    """
    rng = np.random.default_rng(seed)
    n = len(y1_correct)
    diffs = []
    diff_vector = y2_correct.astype(np.float64) - y1_correct.astype(np.float64)

    # Vectorized bootstrap resamples
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        sample_diff = np.mean(diff_vector[indices]) * 100.0
        diffs.append(sample_diff)

    diffs = np.sort(diffs)
    observed_delta = (np.mean(y2_correct) - np.mean(y1_correct)) * 100.0
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    return observed_delta, ci_lower, ci_upper


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Final Evaluation and Statistical Testing")
    parser.add_argument("--cand_ckpt", type=str, required=True, help="Path to selected Phase 4 candidate checkpoint")
    parser.add_argument("--cand_name", type=str, default="PHASE4_CANDIDATE", help="Display name for Phase 4 candidate")
    parser.add_argument("--v1_ckpt", type=str, default="weights/ocr/best_lprnet_vn.pth", help="Path to V1 baseline")
    parser.add_argument("--ftb_ckpt", type=str, default="weights/ocr/experiments/phase3_ft_b/best.pth", help="Path to Phase 3 FT-B")
    parser.add_argument("--out_json", type=str, default="reports/phase4_evaluation_results.json", help="Path to save output JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("PHASE 4 FINAL EVALUATION & STATISTICAL TESTING")
    print(f"Device: {device}")
    print(f"Baseline V1:   {args.v1_ckpt}")
    print(f"Phase 3 FT-B:  {args.ftb_ckpt}")
    print(f"Candidate:     {args.cand_ckpt} ({args.cand_name})")
    print("=" * 75)

    v1_model = load_model(Path(args.v1_ckpt), device)
    ftb_model = load_model(Path(args.ftb_ckpt), device)
    cand_model = load_model(Path(args.cand_ckpt), device)

    manifests = {
        "TEST_FINAL": Path("datasets/VNLP_processed/test.csv"),
        "TEST_CLEAN": Path("datasets/VNLP_processed/test_clean.csv"),
        "TEST_UNSEEN_TEXT": Path("datasets/VNLP_processed/test_unseen_text.csv"),
    }

    results = {
        "candidate_name": args.cand_name,
        "candidate_ckpt": args.cand_ckpt,
        "test_sets": {},
        "statistical_tests": {},
    }

    eval_data = {}

    for name, path in manifests.items():
        print(f"\nEvaluating on {name} ({path.name})...")
        m_v1, r_v1 = evaluate_manifest(v1_model, str(path), device=device, batch_size=128, num_workers=4)
        m_ftb, r_ftb = evaluate_manifest(ftb_model, str(path), device=device, batch_size=128, num_workers=4)
        m_cand, r_cand = evaluate_manifest(cand_model, str(path), device=device, batch_size=128, num_workers=4)

        eval_data[name] = {
            "v1": (m_v1, r_v1),
            "ftb": (m_ftb, r_ftb),
            "cand": (m_cand, r_cand),
        }

        results["test_sets"][name] = {
            "V1": {
                "exact": m_v1["exact_accuracy"],
                "char_acc": m_v1["char_accuracy"],
                "cer": m_v1["cer"],
                "avg_ed": m_v1["avg_edit_distance"],
                "one_row": m_v1["layout_breakdown"]["one_row"]["exact_accuracy"],
                "two_rows": m_v1["layout_breakdown"]["two_rows"]["exact_accuracy"],
                "motorcycle": m_v1["layout_breakdown"]["two_rows_label_xe_may"]["exact_accuracy"],
            },
            "FT_B": {
                "exact": m_ftb["exact_accuracy"],
                "char_acc": m_ftb["char_accuracy"],
                "cer": m_ftb["cer"],
                "avg_ed": m_ftb["avg_edit_distance"],
                "one_row": m_ftb["layout_breakdown"]["one_row"]["exact_accuracy"],
                "two_rows": m_ftb["layout_breakdown"]["two_rows"]["exact_accuracy"],
                "motorcycle": m_ftb["layout_breakdown"]["two_rows_label_xe_may"]["exact_accuracy"],
            },
            "CANDIDATE": {
                "exact": m_cand["exact_accuracy"],
                "char_acc": m_cand["char_accuracy"],
                "cer": m_cand["cer"],
                "avg_ed": m_cand["avg_edit_distance"],
                "one_row": m_cand["layout_breakdown"]["one_row"]["exact_accuracy"],
                "two_rows": m_cand["layout_breakdown"]["two_rows"]["exact_accuracy"],
                "motorcycle": m_cand["layout_breakdown"]["two_rows_label_xe_may"]["exact_accuracy"],
            },
        }

        print(f"  V1:        Exact={m_v1['exact_accuracy']:.2f}% | 1Row={m_v1['layout_breakdown']['one_row']['exact_accuracy']:.2f}% | 2Row={m_v1['layout_breakdown']['two_rows']['exact_accuracy']:.2f}% | Moto={m_v1['layout_breakdown']['two_rows_label_xe_may']['exact_accuracy']:.2f}% | CER={m_v1['cer']:.4f}")
        print(f"  FT-B:      Exact={m_ftb['exact_accuracy']:.2f}% | 1Row={m_ftb['layout_breakdown']['one_row']['exact_accuracy']:.2f}% | 2Row={m_ftb['layout_breakdown']['two_rows']['exact_accuracy']:.2f}% | Moto={m_ftb['layout_breakdown']['two_rows_label_xe_may']['exact_accuracy']:.2f}% | CER={m_ftb['cer']:.4f}")
        print(f"  Candidate: Exact={m_cand['exact_accuracy']:.2f}% | 1Row={m_cand['layout_breakdown']['one_row']['exact_accuracy']:.2f}% | 2Row={m_cand['layout_breakdown']['two_rows']['exact_accuracy']:.2f}% | Moto={m_cand['layout_breakdown']['two_rows_label_xe_may']['exact_accuracy']:.2f}% | CER={m_cand['cer']:.4f}")

    # Extract Fixed Cases and Regressions on TEST_FINAL
    print("\nExtracting Fixed Cases and Regressions between V1 and Candidate on TEST_FINAL...")
    _, r_v1_test = eval_data["TEST_FINAL"]["v1"]
    _, r_cand_test = eval_data["TEST_FINAL"]["cand"]

    fixed_cases = []
    regressions = []
    cand_correct_tf = []
    v1_correct_tf = []

    for item_v1, item_cand in zip(r_v1_test, r_cand_test):
        assert item_v1["image_path"] == item_cand["image_path"]
        img_name = Path(item_v1["image_path"]).name
        gt = item_v1["ground_truth"]
        p_v1 = item_v1["prediction"]
        p_cand = item_cand["prediction"]
        lay = item_v1["layout"]

        e_v1 = bool(item_v1["exact"])
        e_cand = bool(item_cand["exact"])
        v1_correct_tf.append(e_v1)
        cand_correct_tf.append(e_cand)

        if not e_v1 and e_cand:
            fixed_cases.append({
                "image": img_name,
                "ground_truth": gt,
                "v1_prediction": p_v1,
                "candidate_prediction": p_cand,
                "layout": lay,
                "source_test_set": "TEST_FINAL",
            })
        elif e_v1 and not e_cand:
            regressions.append({
                "image": img_name,
                "ground_truth": gt,
                "v1_prediction": p_v1,
                "candidate_prediction": p_cand,
                "layout": lay,
                "source_test_set": "TEST_FINAL",
            })

    # Save fixed cases & regressions
    fixed_path = Path("reports/phase4_fixed_cases.csv")
    with open(fixed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "v1_prediction", "candidate_prediction", "layout", "source_test_set"])
        writer.writeheader()
        writer.writerows(fixed_cases)
    print(f"Saved {fixed_path} ({len(fixed_cases)} fixed)")

    regress_path = Path("reports/phase4_regressions.csv")
    with open(regress_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "v1_prediction", "candidate_prediction", "layout", "source_test_set"])
        writer.writeheader()
        writer.writerows(regressions)
    print(f"Saved {regress_path} ({len(regressions)} regressions)")

    # Statistical Significance Testing: McNemar Exact Test
    b_cand = len(fixed_cases)
    c_cand = len(regressions)
    p_mcnemar_cand = mcnemar_exact_test(b_cand, c_cand)

    # For FT-B vs V1
    _, r_ftb_test = eval_data["TEST_FINAL"]["ftb"]
    b_ftb = sum(1 for v, f in zip(r_v1_test, r_ftb_test) if not v["exact"] and f["exact"])
    c_ftb = sum(1 for v, f in zip(r_v1_test, r_ftb_test) if v["exact"] and not f["exact"])
    p_mcnemar_ftb = mcnemar_exact_test(b_ftb, c_ftb)

    results["statistical_tests"]["mcnemar_phase3_ftb"] = {
        "b_fixed": b_ftb,
        "c_regressions": c_ftb,
        "net_gain": b_ftb - c_ftb,
        "p_value": p_mcnemar_ftb,
        "is_significant_05": p_mcnemar_ftb < 0.05,
    }

    results["statistical_tests"]["mcnemar_phase4_cand"] = {
        "b_fixed": b_cand,
        "c_regressions": c_cand,
        "net_gain": b_cand - c_cand,
        "p_value": p_mcnemar_cand,
        "is_significant_05": p_mcnemar_cand < 0.05,
    }

    # Bootstrap 95% Confidence Intervals for Candidate - V1
    for t_name in ["TEST_FINAL", "TEST_CLEAN", "TEST_UNSEEN_TEXT"]:
        _, rec_v1 = eval_data[t_name]["v1"]
        _, rec_cand = eval_data[t_name]["cand"]
        arr_v1 = np.array([r["exact"] for r in rec_v1], dtype=bool)
        arr_cand = np.array([r["exact"] for r in rec_cand], dtype=bool)
        delta, ci_lo, ci_up = paired_bootstrap_ci(arr_v1, arr_cand, n_resamples=10000, seed=42)
        results["statistical_tests"][f"bootstrap_{t_name.lower()}"] = {
            "delta_pp": round(delta, 2),
            "ci_lower_95": round(ci_lo, 2),
            "ci_upper_95": round(ci_up, 2),
            "spans_zero": (ci_lo <= 0.0 <= ci_up),
        }

    # Save output JSON
    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation summary to {out_json_path}")

    # Print summary
    print("\n" + "=" * 75)
    print("FINAL TEST SUMMARY")
    print(f"Candidate: {args.cand_name}")
    print(f"Fixed: {b_cand}, Regressions: {c_cand}, Net: {b_cand - c_cand:+d}")
    print(f"Candidate McNemar p-value: {p_mcnemar_cand:.4f} ({'Significant (p<0.05)' if p_mcnemar_cand < 0.05 else 'NOT statistically significant (p>=0.05)'})")
    print(f"Phase 3 FT-B McNemar p-value: {p_mcnemar_ftb:.4f} (fixed={b_ftb}, reg={c_ftb}, net={b_ftb-c_ftb:+d})")
    for t_name in ["TEST_FINAL", "TEST_CLEAN", "TEST_UNSEEN_TEXT"]:
        b_res = results["statistical_tests"][f"bootstrap_{t_name.lower()}"]
        print(f"Bootstrap 95% CI ({t_name}): {b_res['delta_pp']:+.2f} pp [ {b_res['ci_lower_95']:+.2f} pp, {b_res['ci_upper_95']:+.2f} pp ] (Spans zero: {b_res['spans_zero']})")
    print("=" * 75)


if __name__ == "__main__":
    main()
