"""
Deduplication and Leakage Verification Tool for License Plate Datasets.
Two-tier deduplication:
  Tier 1: SHA256 Exact Duplicate Check
  Tier 2: DCT-based Perceptual Hash (pHash) Near-Duplicate Check

Actions:
  - KEEP: Truly new, unique sample -> saved to dataset_staging/pbl4_unique/
  - QUARANTINE_EXACT: Exact bitwise match -> quarantined
  - QUARANTINE_NEAR: Near duplicate (pHash Hamming distance <= threshold) -> quarantined
  - TEST_LEAKAGE_RISK: Duplicate of any TEST_FINAL sample -> strict quarantine
  - REVIEW: Boundary match needing manual inspection
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import cv2
import numpy as np


def compute_sha256(filepath: Path) -> str:
    """Computes SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(img_bgr: np.ndarray) -> int:
    """
    Computes 64-bit Discrete Cosine Transform (DCT) Perceptual Hash (pHash).
    1. Grayscale conversion.
    2. Resize to 32x32.
    3. 2D DCT.
    4. Top-left 8x8 low-frequency block (excluding DC term [0, 0]).
    5. Median threshold -> 64-bit integer.
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))
    
    # 8x8 low-frequency components
    low_freq = dct[:8, :8]
    med = np.median(low_freq[1:, 1:]) if low_freq.size > 1 else np.median(low_freq)
    
    # 64-bit binary hash
    bits = (low_freq > med).flatten()
    hash_int = 0
    for b in bits:
        hash_int = (hash_int << 1) | int(b)
    return hash_int


def hamming_distance(h1: int, h2: int) -> int:
    """Computes Hamming distance between two 64-bit integers."""
    return (h1 ^ h2).bit_count()


def run_deduplication(
    new_dir: Path,
    vnlp_all_csv: Path,
    test_final_csv: Path,
    unique_out_dir: Path,
    quarantine_out_dir: Path,
    report_csv_path: Path,
    phash_near_thresh: int = 6,
    phash_review_thresh: int = 10,
):
    print("=" * 70)
    print("      DATASET DEDUPLICATION & TEST LEAKAGE VERIFICATION TOOL      ")
    print("=" * 70)
    
    unique_out_dir.mkdir(parents=True, exist_ok=True)
    quarantine_out_dir.mkdir(parents=True, exist_ok=True)
    report_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Preload VNLP existing dataset
    print("\n--- 1. Preloading Existing Dataset (VNLP) Hashes ---")
    t0 = time.time()
    
    vnlp_files = {} # sha256 -> record
    test_sha256_set = set()
    test_records = {} # plate_text -> list of test records

    if test_final_csv.exists():
        with open(test_final_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p = Path(r["image_path"])
                txt = r["plate_text"].upper()
                test_records.setdefault(txt, []).append(r)
                if p.exists():
                    sha = compute_sha256(p)
                    test_sha256_set.add(sha)
                    vnlp_files[sha] = {"path": str(p), "label": txt, "is_test": True}

    if vnlp_all_csv.exists():
        with open(vnlp_all_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p = Path(r["image_path"])
                txt = r["plate_text"].upper()
                if p.exists():
                    sha = compute_sha256(p)
                    if sha not in vnlp_files:
                        vnlp_files[sha] = {"path": str(p), "label": txt, "is_test": False}

    print(f"Preloaded {len(vnlp_files)} VNLP image hashes (including {len(test_sha256_set)} test images) in {time.time()-t0:.2f}s")

    # 2. Collect all new images (PBL4)
    print("\n--- 2. Scanning New Images ---")
    new_files = sorted(list(new_dir.rglob("*.jpg")) + list(new_dir.rglob("*.png")))
    print(f"Found {len(new_files)} new images to audit in {new_dir}")

    # 3. Deduplication execution
    print("\n--- 3. Running Two-Tier Deduplication & Leakage Detection ---")
    
    # Internal trackers for new dataset
    seen_sha256: Dict[str, Dict[str, Any]] = {}  # sha256 -> info
    seen_phashes: List[Dict[str, Any]] = []      # list of {phash, file, sha256, label}

    records_out = []
    
    stats = {
        "total": len(new_files),
        "keep": 0,
        "quarantine_exact_vnlp": 0,
        "quarantine_exact_internal": 0,
        "quarantine_near": 0,
        "test_leakage_risk": 0,
        "review": 0,
    }

    t_start = time.time()
    
    for idx, f in enumerate(new_files, 1):
        if idx % 2000 == 0 or idx == len(new_files):
            print(f"  Processed {idx}/{len(new_files)} images ({idx/len(new_files)*100:.1f}%)...")

        # Parse new label
        parts = f.stem.split("_")
        new_label = parts[0].upper()

        # Compute SHA256
        new_sha256 = compute_sha256(f)
        
        # Load image for pHash
        img = cv2.imread(str(f))
        new_phash = compute_phash(img) if img is not None else 0

        # Flag trackers
        action = "KEEP"
        dup_type = "NONE"
        existing_file = "NONE"
        sha_existing = "NONE"
        min_phash_dist = 64
        existing_label = "NONE"

        # Check A: Exact match against VNLP
        if new_sha256 in vnlp_files:
            match_vnlp = vnlp_files[new_sha256]
            existing_file = match_vnlp["path"]
            sha_existing = new_sha256
            existing_label = match_vnlp["label"]
            min_phash_dist = 0
            
            if match_vnlp["is_test"]:
                action = "TEST_LEAKAGE_RISK"
                dup_type = "EXACT_VNLP_TEST"
                stats["test_leakage_risk"] += 1
            else:
                action = "QUARANTINE_EXACT"
                dup_type = "EXACT_VNLP_TRAIN"
                stats["quarantine_exact_vnlp"] += 1

        # Check B: Exact match internally within new dataset
        elif new_sha256 in seen_sha256:
            prev = seen_sha256[new_sha256]
            existing_file = prev["file"]
            sha_existing = new_sha256
            existing_label = prev["label"]
            min_phash_dist = 0
            action = "QUARANTINE_EXACT"
            dup_type = "EXACT_INTERNAL"
            stats["quarantine_exact_internal"] += 1

        # Check C: Perceptual Hash near duplicate check (Tier 2)
        else:
            # Check against internal seen_phashes
            best_match = None
            closest_dist = 64
            
            for item in seen_phashes:
                # Fast XOR distance
                dist = hamming_distance(new_phash, item["phash"])
                if dist < closest_dist:
                    closest_dist = dist
                    best_match = item
                    if dist == 0:
                        break

            min_phash_dist = closest_dist
            if best_match is not None and closest_dist <= phash_near_thresh:
                existing_file = best_match["file"]
                sha_existing = best_match["sha256"]
                existing_label = best_match["label"]
                dup_type = f"NEAR_INTERNAL_PHASH_{closest_dist}"
                action = "QUARANTINE_NEAR"
                stats["quarantine_near"] += 1
            elif best_match is not None and closest_dist <= phash_review_thresh and best_match["label"] == new_label:
                existing_file = best_match["file"]
                sha_existing = best_match["sha256"]
                existing_label = best_match["label"]
                dup_type = f"NEAR_REVIEW_PHASH_{closest_dist}"
                action = "REVIEW"
                stats["review"] += 1
            else:
                action = "KEEP"
                stats["keep"] += 1

        # Process Action & File Staging
        rel_name = f"{f.parent.name}_{f.name}"
        
        if action == "KEEP":
            # Add to seen
            seen_sha256[new_sha256] = {"file": str(f), "label": new_label}
            seen_phashes.append({"phash": new_phash, "file": str(f), "sha256": new_sha256, "label": new_label})
            # Copy to staging pbl4_unique
            dest_unique = unique_out_dir / rel_name
            shutil.copy2(f, dest_unique)
        else:
            # Copy to quarantine
            dest_quarantine = quarantine_out_dir / rel_name
            shutil.copy2(f, dest_quarantine)

        records_out.append({
            "new_file": str(f).replace("\\", "/"),
            "existing_file": str(existing_file).replace("\\", "/"),
            "duplicate_type": dup_type,
            "sha256_new": new_sha256,
            "sha256_existing": sha_existing,
            "phash_distance": min_phash_dist,
            "new_label": new_label,
            "existing_label": existing_label,
            "action": action,
        })

    elapsed = time.time() - t_start
    print(f"\nDeduplication complete in {elapsed:.2f} seconds.")

    # 4. Export CSV Report
    print(f"\nWriting duplicates report to {report_csv_path}...")
    fieldnames = [
        "new_file", "existing_file", "duplicate_type",
        "sha256_new", "sha256_existing", "phash_distance",
        "new_label", "existing_label", "action"
    ]
    with open(report_csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records_out)

    print("\n" + "=" * 65)
    print("DEDUPLICATION AUDIT SUMMARY:")
    print("=" * 65)
    print(f"Total new images audited:           {stats['total']}")
    print(f"Truly Unique usable images (KEEP):  {stats['keep']} (copied to {unique_out_dir.name}/)")
    print(f"Exact internal duplicates:          {stats['quarantine_exact_internal']}")
    print(f"Exact VNLP duplicates:              {stats['quarantine_exact_vnlp']}")
    print(f"Near duplicates (pHash <= {phash_near_thresh}):       {stats['quarantine_near']}")
    print(f"Test leakage risk (vs TEST_FINAL):  {stats['test_leakage_risk']}")
    print(f"Review flagged (pHash {phash_near_thresh+1}-{phash_review_thresh}):       {stats['review']}")
    print(f"Total quarantined:                  {stats['total'] - stats['keep']} (copied to {quarantine_out_dir.name}/)")
    print("=" * 65)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Dataset Deduplication & Leakage Check")
    parser.add_argument("--new_dir", type=str, default="datasets_external/pbl4/PBL4_LPRNet/Dataset/BiensoxeVietNam")
    parser.add_argument("--vnlp_all", type=str, default="datasets/VNLP_processed/all.csv")
    parser.add_argument("--test_final", type=str, default="datasets/VNLP_processed/test.csv")
    parser.add_argument("--unique_out", type=str, default="dataset_staging/pbl4_unique")
    parser.add_argument("--quarantine_out", type=str, default="dataset_staging/quarantine_duplicates")
    parser.add_argument("--report_csv", type=str, default="reports/pbl4_vs_current_duplicates.csv")
    parser.add_argument("--phash_thresh", type=int, default=6)
    args = parser.parse_args()

    run_deduplication(
        new_dir=Path(args.new_dir),
        vnlp_all_csv=Path(args.vnlp_all),
        test_final_csv=Path(args.test_final),
        unique_out_dir=Path(args.unique_out),
        quarantine_out_dir=Path(args.quarantine_out),
        report_csv_path=Path(args.report_csv),
        phash_near_thresh=args.phash_thresh,
    )


if __name__ == "__main__":
    main()
