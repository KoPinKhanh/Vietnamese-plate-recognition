"""
Audit B: Cross-Set Near-Duplicate and Exact Overlap Audit between
reports/e2e_real_world_manifest.csv (N=300 Development Benchmark)
and
reports/fresh_holdout_500_manifest.csv (N=500 Fresh Routing Holdout)

Outputs:
  - reports/dev300_vs_holdout500_near_duplicate_audit.csv
"""

import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> int:
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    val = 0
    for idx, v in enumerate(diff.flatten()):
        if v:
            val += 1 << idx
    return val


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def run_cross_set_audit():
    print("=" * 80)
    print("AUDIT B: DEV 300 VS HOLDOUT 500 CROSS-SET AUDIT")
    print("=" * 80)

    dev_manifest = PROJECT_ROOT / "reports/e2e_real_world_manifest.csv"
    holdout_manifest = PROJECT_ROOT / "reports/fresh_holdout_500_manifest.csv"
    output_audit_csv = PROJECT_ROOT / "reports/dev300_vs_holdout500_near_duplicate_audit.csv"

    # 1. Load Dev 300
    dev_samples = []
    with open(dev_manifest, "r", encoding="utf-8") as f:
        for idx, r in enumerate(csv.DictReader(f), 1):
            p = Path(r["image_path"])
            img = cv2.imread(str(p))
            sha = compute_sha256(p)
            dh = compute_dhash(img) if img is not None else 0
            dev_samples.append({
                "sample_id": f"DEV_{idx:03d}",
                "path": str(p),
                "filename": p.name,
                "sha256": sha,
                "dhash": dh,
                "gt": r["ground_truth"],
            })

    # 2. Load Holdout 500
    holdout_samples = []
    with open(holdout_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Path(r["image_path"])
            img = cv2.imread(str(p))
            dh = compute_dhash(img) if img is not None else 0
            holdout_samples.append({
                "sample_id": r["sample_id"],
                "path": str(p),
                "filename": p.name,
                "sha256": r["sha256"],
                "dhash": dh,
                "gt": r["ground_truth"],
            })

    print(f"Loaded {len(dev_samples)} Dev samples and {len(holdout_samples)} Holdout samples.")

    # 3. Exact SHA256 Overlap Check
    dev_sha_set = {s["sha256"] for s in dev_samples}
    dev_file_set = {s["filename"] for s in dev_samples}

    exact_sha_overlap = []
    exact_name_overlap = []

    for h in holdout_samples:
        if h["sha256"] in dev_sha_set:
            exact_sha_overlap.append(h)
        if h["filename"] in dev_file_set:
            exact_name_overlap.append(h)

    print(f"\nExact SHA256 Overlap:      {len(exact_sha_overlap)}")
    print(f"Exact Filename Overlap:    {len(exact_name_overlap)}")

    # 4. Cross-set perceptual near-duplicate detection (dHash distance <= 4)
    # Total comparisons: 300 * 500 = 150,000 pairs
    near_dup_records = []
    threshold = 4  # Hamming distance <= 4 is suspicious

    print(f"\nPerforming 150,000 pairwise cross-set dHash distance comparisons (Threshold <= {threshold})...")
    for d in dev_samples:
        for h in holdout_samples:
            dist = hamming_distance(d["dhash"], h["dhash"])
            if dist <= threshold:
                decision = "NEAR_DUPLICATE_CONTAMINATION" if dist <= 2 else "STRUCTURAL_SIMILARITY"
                near_dup_records.append({
                    "dev_sample": f"{d['sample_id']} ({d['filename']}, GT:{d['gt']})",
                    "holdout_sample": f"{h['sample_id']} ({h['filename']}, GT:{h['gt']})",
                    "distance": dist,
                    "decision": decision,
                })

    print(f"Near-duplicate candidates found (dist <= {threshold}): {len(near_dup_records)}")
    near_dups_strict = [r for r in near_dup_records if r["distance"] <= 2]
    print(f"Strict near duplicates (dist <= 2): {len(near_dups_strict)}")

    # 5. Write Audit CSV
    with open(output_audit_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dev_sample", "holdout_sample", "distance", "decision"])
        writer.writeheader()
        writer.writerows(near_dup_records)
    print(f"\nSaved cross-set near-duplicate audit to: {output_audit_csv}")

    print("\n" + "=" * 80)
    print("AUDIT B SUMMARY")
    print(f"EXACT OVERLAP:             {len(exact_sha_overlap)}")
    print(f"NEAR-DUPLICATE PAIRS:      {len(near_dups_strict)}")
    print(f"UNRESOLVED CONTAMINATION:  {len(exact_sha_overlap) + len(near_dups_strict)}")
    print("=" * 80)

    return len(exact_sha_overlap), len(near_dups_strict)


if __name__ == "__main__":
    run_cross_set_audit()
