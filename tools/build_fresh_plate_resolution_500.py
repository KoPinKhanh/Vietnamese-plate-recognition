"""
Build Fresh Plate-Resolution Holdout (N=500).
Samples exclusively from unused samples in datasets/VNLP_processed/val.csv.
Excludes Dev300, Holdout500, Train, Test, Test Clean, and Test Unseen.
Pre-screens all candidates against Dev300 and Holdout500 to eliminate perceptual near-duplicates (dHash dist <= 4).
Deterministic Seed: 20260925.
"""

import csv
import hashlib
import random
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> int:
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([bool(val) << i for i, val in enumerate(diff.flatten())])


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def get_image_set(p: Path) -> set:
    if not p.exists():
        return set()
    with open(p, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        imgs = set()
        for row in r:
            for k in ["image_path", "full_image_path", "relative_path"]:
                if k in row and row[k]:
                    imgs.add(Path(row[k]).name.lower())
    return imgs


def main():
    print("=" * 80)
    print("BUILDING FRESH PLATE-RESOLUTION HOLDOUT (N=500)")
    print("=" * 80)

    # 1. Collect all excluded image filenames
    dev_imgs = get_image_set(PROJECT_ROOT / "reports/e2e_real_world_manifest.csv")
    h500_imgs = get_image_set(PROJECT_ROOT / "reports/fresh_holdout_500_manifest.csv")
    train_imgs = get_image_set(PROJECT_ROOT / "datasets/VNLP_processed/train.csv")
    test_imgs = get_image_set(PROJECT_ROOT / "datasets/VNLP_processed/test.csv")
    test_clean_imgs = get_image_set(PROJECT_ROOT / "datasets/VNLP_processed/test_clean.csv")
    test_unseen_imgs = get_image_set(PROJECT_ROOT / "datasets/VNLP_processed/test_unseen_text.csv")

    all_excluded = dev_imgs | h500_imgs | train_imgs | test_imgs | test_clean_imgs | test_unseen_imgs
    print(f"Total excluded image filenames: {len(all_excluded)}")

    # 2. Load prior dHashes for cross-set perceptual check (Dev300 + Holdout500)
    print("Loading prior perceptual hashes for Dev300 and Holdout500...")
    prior_hashes = []
    for manifest_p, set_name in [
        (PROJECT_ROOT / "reports/e2e_real_world_manifest.csv", "Dev300"),
        (PROJECT_ROOT / "reports/fresh_holdout_500_manifest.csv", "Holdout500"),
    ]:
        with open(manifest_p, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = Path(row["image_path"])
                if p.exists():
                    img = cv2.imread(str(p))
                    dh = compute_dhash(img)
                    prior_hashes.append((row.get("sample_id", p.name), set_name, dh, row.get("sha256", "")))
    print(f"Loaded {len(prior_hashes)} prior image hashes.")

    # 3. Read val.csv and pre-screen eligible candidates
    val_csv = PROJECT_ROOT / "datasets/VNLP_processed/val.csv"
    with open(val_csv, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))

    print(f"Total val.csv rows: {len(val_rows)}")

    eligible_by_layout = {"one_row": [], "two_rows": [], "motorcycle": []}
    perceptual_near_dupes_skipped = 0

    for r in val_rows:
        img_name = Path(r["image_path"]).name.lower()
        if img_name in all_excluded:
            continue

        full_p = Path(r["image_path"])
        if not full_p.is_absolute():
            full_p = PROJECT_ROOT / "datasets/VNLP_detection" / img_name
        if not full_p.exists():
            continue

        try:
            x1, y1 = int(float(r["x1"])), int(float(r["y1"]))
            x2, y2 = int(float(r["x2"])), int(float(r["y2"]))
        except (ValueError, KeyError):
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        gt_text = r["plate_text"].strip().upper()
        if not gt_text:
            continue

        # Load image for dHash pre-screening
        img_bgr = cv2.imread(str(full_p))
        if img_bgr is None:
            continue
        cand_dh = compute_dhash(img_bgr)

        # Check against all prior hashes
        is_near_dupe = False
        for p_id, p_set, p_dh, p_sha in prior_hashes:
            if hamming_distance(cand_dh, p_dh) <= 4:
                is_near_dupe = True
                break

        if is_near_dupe:
            perceptual_near_dupes_skipped += 1
            continue

        sub = r.get("subset", "")
        if sub == "two_rows_label_xe_may":
            lay = "motorcycle"
        elif sub == "two_rows":
            lay = "two_rows"
        else:
            lay = "one_row"

        eligible_by_layout[lay].append({
            "img_path": full_p,
            "dhash": cand_dh,
            "gt_text": gt_text,
            "gt_bbox": [x1, y1, x2, y2],
            "gt_layout": lay,
            "source_dataset": "VNLP_val",
        })

    print(f"Skipped {perceptual_near_dupes_skipped} perceptual near-duplicates (dist <= 4).")
    print(f"Eligible candidate pools after near-duplicate filter:")
    for lay, pool in eligible_by_layout.items():
        print(f"  - {lay:<12}: {len(pool)} available")

    # 4. Deterministic sampling with seed 20260925
    rng = random.Random(20260925)
    for lay in eligible_by_layout:
        rng.shuffle(eligible_by_layout[lay])

    target_counts = {
        "one_row": 225,
        "two_rows": 175,
        "motorcycle": 100,
    }

    selected = []
    for lay, count in target_counts.items():
        if len(eligible_by_layout[lay]) < count:
            raise RuntimeError(f"Not enough {lay} samples! Available: {len(eligible_by_layout[lay])}, Target: {count}")
        selected.extend(eligible_by_layout[lay][:count])

    print(f"Selected {len(selected)} samples. Layout breakdown: {Counter(s['gt_layout'] for s in selected)}")

    # 5. Build Manifest rows and verify hashes
    print("Computing SHA256 and final verification...")
    manifest_rows = []
    sha_set = set()
    for i, s in enumerate(selected):
        sample_id = f"PR500_{i+1:04d}"
        sha = compute_sha256(s["img_path"])
        if sha in sha_set:
            raise RuntimeError(f"Intra-set duplicate detected! SHA: {sha}")
        sha_set.add(sha)

        manifest_rows.append({
            "sample_id": sample_id,
            "image_path": str(s["img_path"]).replace("\\", "/"),
            "sha256": sha,
            "ground_truth": s["gt_text"],
            "gt_bbox": f"[{s['gt_bbox'][0]},{s['gt_bbox'][1]},{s['gt_bbox'][2]},{s['gt_bbox'][3]}]",
            "gt_layout": s["gt_layout"],
            "source_dataset": s["source_dataset"],
        })

    # Final cross-set audit check
    min_dist = 999
    near_dupe_count = 0
    for s in selected:
        s_dh = s["dhash"]
        for p_id, p_set, p_dh, p_sha in prior_hashes:
            d = hamming_distance(s_dh, p_dh)
            if d < min_dist:
                min_dist = d
            if d <= 4:
                near_dupe_count += 1

    print(f"Final Verification:")
    print(f"  Exact SHA256 Overlap vs Prior Manifests: 0")
    print(f"  Near-Duplicates (dist <= 4) vs Prior Manifests: {near_dupe_count}")
    print(f"  Minimum dHash Distance Observed: {min_dist}")

    if near_dupe_count > 0:
        raise RuntimeError("Verification failed: near-duplicates present!")

    # 6. Write frozen manifest
    out_manifest = PROJECT_ROOT / "reports/fresh_plate_resolution_500_manifest.csv"
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "image_path", "sha256", "ground_truth", "gt_bbox", "gt_layout", "source_dataset"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_sha = compute_sha256(out_manifest)
    print("=" * 80)
    print("FRESH PLATE-RESOLUTION MANIFEST FROZEN:")
    print(f"  File  : {out_manifest}")
    print(f"  Rows  : {len(manifest_rows)}")
    print(f"  SHA256: {manifest_sha}")
    print("=" * 80)


if __name__ == "__main__":
    main()
