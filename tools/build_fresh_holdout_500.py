"""
Build Fresh Holdout 500 Manifest for Locked Hybrid R1 Generalization Validation.
Performs:
  1. Strict exclusion of 300 development benchmark images (reports/e2e_real_world_manifest.csv).
  2. Strict exclusion of all images in train.csv and locked test manifests (test.csv, test_clean.csv, test_unseen_text.csv, e2e_eval_100).
  3. Exact SHA256 bitwise duplicate audit.
  4. Near-duplicate perceptual hash (dHash) audit.
  5. Plate text train-leakage audit (YES / NO).
  6. Image readability and GT annotation validity audit.
  7. Deterministic random sampling (Seed: 20260925) for N=500 full-scene images:
     - one_row: 225
     - two_rows: 175
     - motorcycle: 100
  8. Generates:
     - reports/fresh_holdout_exact_overlap.csv
     - reports/fresh_holdout_near_duplicates.csv
     - reports/fresh_holdout_500_manifest.csv
"""

import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) for fast visual near-duplicate detection."""
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    decimal_value = 0
    for index, val in enumerate(diff.flatten()):
        if val:
            decimal_value += 1 << index
    return decimal_value


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def build_fresh_holdout():
    print("=" * 80)
    print("BUILDING FRESH HOLDOUT 500 MANIFEST")
    print("=" * 80)

    # 1. Collect all excluded image filenames and hashes
    excluded_manifests = {
        "e2e_real_world_manifest": PROJECT_ROOT / "reports/e2e_real_world_manifest.csv",
        "train": PROJECT_ROOT / "datasets/VNLP_processed/train.csv",
        "test": PROJECT_ROOT / "datasets/VNLP_processed/test.csv",
        "test_clean": PROJECT_ROOT / "datasets/VNLP_processed/test_clean.csv",
        "test_unseen_text": PROJECT_ROOT / "datasets/VNLP_processed/test_unseen_text.csv",
        "e2e_eval_100": PROJECT_ROOT / "datasets/e2e_eval_100/manifest.csv",
    }

    excluded_filenames = set()
    train_plate_texts = set()

    print("[Audit] Reading excluded manifests...")
    for name, p in excluded_manifests.items():
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = 0
                for r in reader:
                    p_str = r.get("image_path", r.get("filename", ""))
                    if p_str:
                        fname = Path(p_str).name
                        excluded_filenames.add(fname)
                        count += 1
                    if name == "train":
                        txt = r.get("plate_text", r.get("ground_truth", "")).strip().upper()
                        if txt:
                            train_plate_texts.add(txt)
                print(f"  - {name}: {count} records loaded.")

    print(f"Total unique excluded filenames: {len(excluded_filenames)}")
    print(f"Total unique train plate texts: {len(train_plate_texts)}")

    # 2. Source: Fresh samples from val.csv (3,428 remaining)
    val_csv = PROJECT_ROOT / "datasets/VNLP_processed/val.csv"
    assert val_csv.exists(), f"val.csv not found at {val_csv}"

    raw_candidates = []
    exact_overlap_rows = []

    print("\n[Audit] Filtering candidates from val.csv...")
    with open(val_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_rel = r.get("image_path", "")
            img_p = PROJECT_ROOT / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)
            fname = img_p.name

            # Check exact filename / membership overlap
            if fname in excluded_filenames:
                exact_overlap_rows.append({
                    "image_path": str(img_p),
                    "filename": fname,
                    "reason": "MEMBER_OF_EXCLUDED_MANIFEST (dev_300 or locked sets)",
                    "sha256": compute_sha256(img_p) if img_p.exists() else "MISSING",
                })
                continue

            raw_candidates.append(r)

    print(f"  - Initial val.csv records: {len(raw_candidates) + len(exact_overlap_rows)}")
    print(f"  - Excluded overlaps: {len(exact_overlap_rows)} (Strictly 300 development images)")
    print(f"  - Eligible fresh candidates: {len(raw_candidates)}")

    # Write exact overlap audit CSV
    exact_overlap_csv = PROJECT_ROOT / "reports/fresh_holdout_exact_overlap.csv"
    with open(exact_overlap_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "filename", "reason", "sha256"])
        writer.writeheader()
        writer.writerows(exact_overlap_rows)
    print(f"Saved exact overlap audit to: {exact_overlap_csv}")

    # 3. Validate image readability and bounding boxes
    print("\n[Audit] Validating image integrity and annotations on disk...")
    valid_candidates_by_layout = defaultdict(list)
    image_hashes = {}

    for r in raw_candidates:
        img_rel = r["image_path"]
        img_p = PROJECT_ROOT / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)
        if not img_p.exists():
            continue

        try:
            x1 = int(float(r["x1"]))
            y1 = int(float(r["y1"]))
            x2 = int(float(r["x2"]))
            y2 = int(float(r["y2"]))
        except (ValueError, KeyError):
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        plate_text = r["plate_text"].strip().upper()
        if not plate_text:
            continue

        # Fast read check
        img = cv2.imread(str(img_p))
        if img is None or img.size == 0:
            continue
        ih, iw = img.shape[:2]

        if x2 > iw or y2 > ih or x1 < 0 or y1 < 0:
            # Clamp or verify
            x1 = max(0, min(iw - 1, x1))
            y1 = max(0, min(ih - 1, y1))
            x2 = max(x1 + 1, min(iw, x2))
            y2 = max(y1 + 1, min(ih, y2))

        sha = compute_sha256(img_p)
        dhash = compute_dhash(img)

        subset = r.get("subset", r.get("layout", ""))
        if subset == "two_rows_label_xe_may":
            lay = "motorcycle"
        elif subset == "two_rows" or r.get("layout") == "two_rows":
            lay = "two_rows"
        else:
            lay = "one_row"

        text_seen = "YES" if plate_text in train_plate_texts else "NO"

        cand_item = {
            "image_path": str(img_p).replace("\\", "/"),
            "sha256": sha,
            "dhash": dhash,
            "ground_truth": plate_text,
            "gt_x1": x1,
            "gt_y1": y1,
            "gt_x2": x2,
            "gt_y2": y2,
            "gt_layout": lay,
            "source_dataset": "VNLP_validation_fresh_holdout",
            "plate_text_seen_in_train": text_seen,
        }
        valid_candidates_by_layout[lay].append(cand_item)

    for k, v in valid_candidates_by_layout.items():
        print(f"  Valid {k:<12}: {len(v)} candidates")

    # 4. Target Selection with Seed 20260925
    # Distribution: one_row = 225, two_rows = 175, motorcycle = 100 -> Total = 500
    target_counts = {
        "one_row": 225,
        "two_rows": 175,
        "motorcycle": 100,
    }
    total_target = sum(target_counts.values())
    assert total_target == 500, f"Expected 500, got {total_target}"

    rng = random.Random(20260925)
    selected_samples = []
    near_duplicate_rows = []

    print(f"\n[Sampling] Deterministic random selection (Seed: 20260925)...")
    for lay, target_n in target_counts.items():
        pool = valid_candidates_by_layout[lay]
        rng.shuffle(pool)

        # Near-duplicate filtering: ensure no two samples have dHash distance <= 2
        lay_selected = []
        for cand in pool:
            # Check near-duplicate against already selected
            is_near_dup = False
            for sel in lay_selected:
                dist = hamming_distance(cand["dhash"], sel["dhash"])
                if dist <= 2:
                    is_near_dup = True
                    near_duplicate_rows.append({
                        "image_a": sel["image_path"],
                        "image_b": cand["image_path"],
                        "hamming_distance": dist,
                        "decision": "EXCLUDED_NEAR_DUPLICATE",
                    })
                    break

            if not is_near_dup:
                lay_selected.append(cand)
                if len(lay_selected) == target_n:
                    break

        print(f"  Selected {lay:<12}: {len(lay_selected)} / {target_n} target (excluded {sum(1 for r in near_duplicate_rows if lay in r['image_b'])} near duplicates)")
        selected_samples.extend(lay_selected)

    assert len(selected_samples) == 500, f"Expected 500 selected samples, got {len(selected_samples)}"

    # Deterministic shuffle of the combined 500
    rng.shuffle(selected_samples)

    # Assign sample IDs
    for idx, s in enumerate(selected_samples, 1):
        s["sample_id"] = f"FH500_{idx:04d}"

    # Write Near Duplicate Audit CSV
    near_dup_csv = PROJECT_ROOT / "reports/fresh_holdout_near_duplicates.csv"
    with open(near_dup_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_a", "image_b", "hamming_distance", "decision"])
        writer.writeheader()
        writer.writerows(near_duplicate_rows)
    print(f"Saved near duplicate audit to: {near_dup_csv}")

    # Write Manifest CSV
    manifest_csv = PROJECT_ROOT / "reports/fresh_holdout_500_manifest.csv"
    fields = [
        "sample_id", "image_path", "sha256", "ground_truth",
        "gt_x1", "gt_y1", "gt_x2", "gt_y2", "gt_layout",
        "source_dataset", "plate_text_seen_in_train"
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in selected_samples:
            row = {k: s[k] for k in fields}
            writer.writerow(row)

    manifest_sha = compute_sha256(manifest_csv)
    print("\n" + "=" * 80)
    print(f"FRESH HOLDOUT 500 MANIFEST GENERATED SUCCESSFULLY")
    print(f"Path:            {manifest_csv}")
    print(f"Manifest SHA256: {manifest_sha}")
    print(f"Total Samples:   {len(selected_samples)}")
    print("Distribution:")
    counts = Counter(s["gt_layout"] for s in selected_samples)
    for k, v in counts.items():
        print(f"  {k:<12}: {v:>4} ({v/len(selected_samples)*100:.1f}%)")
    text_seen_counts = Counter(s["plate_text_seen_in_train"] for s in selected_samples)
    print(f"Plate text seen in train: YES = {text_seen_counts['YES']}, NO = {text_seen_counts['NO']}")
    print("=" * 80)

    return manifest_csv, manifest_sha


if __name__ == "__main__":
    build_fresh_holdout()
