"""
Build Real-World End-to-End Evaluation Manifest for License Plate Pipeline.
Selects full-scene images from validation set with ground truth bbox and text,
verifies file accessibility on disk, guarantees 0% overlap with locked test manifests,
and records the dataset overlap audit log.

Outputs:
  - reports/e2e_real_world_manifest.csv
  - reports/e2e_overlap_audit.csv
"""

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path
from typing import Set, Dict, List, Any

import cv2

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_locked_test_filenames() -> Set[str]:
    """Loads all filenames in locked test manifests to strictly prevent data leakage."""
    test_manifests = [
        PROJECT_ROOT / "datasets/VNLP_processed/test.csv",
        PROJECT_ROOT / "datasets/VNLP_processed/test_clean.csv",
        PROJECT_ROOT / "datasets/VNLP_processed/test_unseen_text.csv",
    ]
    locked_filenames = set()
    for p in test_manifests:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    img_p = r.get("image_path", r.get("filename", ""))
                    if img_p:
                        locked_filenames.add(Path(img_p).name)
    return locked_filenames


def build_manifest(
    val_csv_path: Path = PROJECT_ROOT / "datasets/VNLP_processed/val.csv",
    output_manifest_path: Path = PROJECT_ROOT / "reports/e2e_real_world_manifest.csv",
    audit_csv_path: Path = PROJECT_ROOT / "reports/e2e_overlap_audit.csv",
    target_count: int = 300,
    seed: int = 42,
) -> Path:
    random.seed(seed)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BUILDING REAL-WORLD END-TO-END EVALUATION MANIFEST")
    print(f"Validation Source: {val_csv_path}")
    print(f"Target Count: {target_count} images (Seed: {seed})")
    print("=" * 70)

    # 1. Audit Locked Test Overlap
    locked_test_names = load_locked_test_filenames()
    print(f"Total locked test image filenames to protect: {len(locked_test_names)}")

    val_records = []
    overlap_records = []

    with open(val_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fname = Path(r["image_path"]).name
            if fname in locked_test_names:
                overlap_records.append(fname)
            else:
                val_records.append(r)

    print(f"Total validation candidate records: {len(val_records)}")
    print(f"Locked test overlaps detected: {len(overlap_records)}")

    # Write Overlap Audit CSV
    with open(audit_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_manifest", "total_records", "test_overlap_count", "leakage_status"])
        writer.writerow([str(val_csv_path), len(val_records) + len(overlap_records), len(overlap_records), "PASSED_ZERO_LEAKAGE" if len(overlap_records) == 0 else "LEAKAGE_DETECTED"])
    print(f"Saved overlap audit to: {audit_csv_path}")

    # 2. Partition by category to ensure balanced representation
    one_row_candidates = []
    two_rows_car_candidates = []
    motorcycle_candidates = []

    for r in val_records:
        img_p = Path(r["image_path"])
        if not img_p.is_absolute():
            img_p = PROJECT_ROOT / img_p
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

        item = {
            "image_path": str(img_p).replace("\\", "/"),
            "ground_truth": r["plate_text"].strip().upper(),
            "gt_x1": x1,
            "gt_y1": y1,
            "gt_x2": x2,
            "gt_y2": y2,
            "source_dataset": "VNLP_validation",
        }

        subset = r.get("subset", r.get("layout", ""))
        if subset == "two_rows_label_xe_may":
            item["gt_layout"] = "motorcycle"
            motorcycle_candidates.append(item)
        elif subset == "two_rows" or r.get("layout") == "two_rows":
            item["gt_layout"] = "two_rows"
            two_rows_car_candidates.append(item)
        else:
            item["gt_layout"] = "one_row"
            one_row_candidates.append(item)

    print(f"\nAvailable valid candidates on disk:")
    print(f"  - one_row:               {len(one_row_candidates)}")
    print(f"  - two_rows (car):        {len(two_rows_car_candidates)}")
    print(f"  - motorcycle (two_rows): {len(motorcycle_candidates)}")

    # Sample proportionally: ~50% one_row, ~33% two_rows, ~17% motorcycle
    n_one_row = int(target_count * 0.50)
    n_two_rows = int(target_count * 0.33)
    n_moto = target_count - (n_one_row + n_two_rows)

    random.shuffle(one_row_candidates)
    random.shuffle(two_rows_car_candidates)
    random.shuffle(motorcycle_candidates)

    selected = (
        one_row_candidates[:n_one_row] +
        two_rows_car_candidates[:n_two_rows] +
        motorcycle_candidates[:n_moto]
    )

    # Deterministic shuffle of final manifest
    random.shuffle(selected)

    # 3. Write Manifest CSV
    fields = ["image_path", "ground_truth", "gt_x1", "gt_y1", "gt_x2", "gt_y2", "gt_layout", "source_dataset"]
    with open(output_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    manifest_sha = compute_file_sha256(output_manifest_path)
    print("\n" + "=" * 70)
    print(f"MANIFEST GENERATION COMPLETE: {len(selected)} samples")
    print(f"Output File: {output_manifest_path}")
    print(f"SHA256:      {manifest_sha}")
    print("Distribution:")
    from collections import Counter
    counts = Counter(x["gt_layout"] for x in selected)
    for k, v in counts.items():
        print(f"  {k:<12}: {v:>4} ({v/len(selected)*100:.1f}%)")
    print("=" * 70)

    return output_manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Real-World E2E Evaluation Manifest")
    parser.add_argument("--count", type=int, default=300, help="Number of samples (200-500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    build_manifest(target_count=args.count, seed=args.seed)
