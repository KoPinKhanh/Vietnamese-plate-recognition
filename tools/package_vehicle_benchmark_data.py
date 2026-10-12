"""
Package 500 full-scene images and minimal manifest for Raspberry Pi vehicle detector benchmark.
"""

import csv
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

manifest_path = PROJECT_ROOT / "reports/fresh_holdout_500_manifest.csv"
results_path = PROJECT_ROOT / "reports/fresh_holdout_500_results.csv"

scratch_dir = PROJECT_ROOT / "scratch/vehicle_benchmark_pkg"
scratch_dir.mkdir(parents=True, exist_ok=True)

with open(manifest_path, "r", encoding="utf-8") as f:
    m_rows = list(csv.DictReader(f))

with open(results_path, "r", encoding="utf-8") as f:
    r_rows = list(csv.DictReader(f))

results_by_id = {r["sample_id"]: r for r in r_rows}

tar_path = scratch_dir / "vehicle_benchmark_500.tar"
manifest_out_path = scratch_dir / "vehicle_benchmark_manifest.csv"

out_manifest_rows = []

print(f"Creating tar archive of {len(m_rows)} images: {tar_path}...")
with tarfile.open(tar_path, "w") as tar:
    for r in m_rows:
        sid = r["sample_id"]
        img_p = Path(r["image_path"])
        if not img_p.exists():
            raise FileNotFoundError(f"Image not found: {img_p}")

        # Add to tar with flat filename
        tar.add(str(img_p), arcname=img_p.name)

        r_res = results_by_id[sid]
        fallback_cohort = (r_res.get("direct_candidates_count") == "0" or r_res.get("cascade_invoked") == "True")

        out_manifest_rows.append({
            "sample_id": sid,
            "image_file": img_p.name,
            "gt_layout": r["gt_layout"],
            "gt_x1": r["gt_x1"],
            "gt_y1": r["gt_y1"],
            "gt_x2": r["gt_x2"],
            "gt_y2": r["gt_y2"],
            "ground_truth": r["ground_truth"],
            "fallback_cohort": str(fallback_cohort),
        })

print(f"Tar created: {tar_path.stat().st_size / (1024*1024):.2f} MB")

with open(manifest_out_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["sample_id", "image_file", "gt_layout", "gt_x1", "gt_y1", "gt_x2", "gt_y2", "ground_truth", "fallback_cohort"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_manifest_rows)

print(f"Manifest created: {manifest_out_path} ({len(out_manifest_rows)} rows)")
