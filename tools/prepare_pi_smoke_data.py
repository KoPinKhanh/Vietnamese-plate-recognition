"""
Prepare 10 full-scene images and 10 real plate crops for Raspberry Pi smoke testing.
"""
import csv
import json
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.laptop_parseq_recognize import crop_with_padding


def main():
    manifest_path = PROJECT_ROOT / "reports/fresh_holdout_500_manifest.csv"
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:10]

    smoke_dir = PROJECT_ROOT / "scratch/pi_smoke"
    full_scene_dir = smoke_dir / "full_scene"
    crops_dir = smoke_dir / "plate_crops"
    full_scene_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    smoke_manifest = []

    for idx, r in enumerate(rows):
        img_path = r["image_path"]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        orig_name = Path(img_path).name
        scene_name = f"scene_{idx:02d}_{orig_name}"
        dest_scene = full_scene_dir / scene_name
        cv2.imwrite(str(dest_scene), img)

        gt_box = [int(float(r[k])) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")]
        crop, _ = crop_with_padding(img, gt_box, pad_ratio=0.05)
        crop_name = f"crop_{idx:02d}_{r['ground_truth']}_{r['gt_layout']}.jpg"
        dest_crop = crops_dir / crop_name
        cv2.imwrite(str(dest_crop), crop)

        smoke_manifest.append({
            "idx": idx,
            "scene_file": scene_name,
            "crop_file": crop_name,
            "ground_truth": r["ground_truth"],
            "gt_layout": r["gt_layout"],
            "gt_box": gt_box
        })

    with open(smoke_dir / "smoke_manifest.json", "w", encoding="utf-8") as f:
        json.dump(smoke_manifest, f, indent=2)

    print(f"Prepared {len(rows)} scenes and crops in {smoke_dir}")


if __name__ == "__main__":
    main()
