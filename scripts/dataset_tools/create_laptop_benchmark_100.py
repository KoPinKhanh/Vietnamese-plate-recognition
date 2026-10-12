"""
Create a balanced 100-sample test benchmark containing:
- 35 one_row
- 35 two_rows (car)
- 30 motorcycle (two_rows_label_xe_may)
From scratch/test_plate_crops and datasets/VNLP_processed/val.csv.
"""

import csv
from pathlib import Path

f60 = Path("scratch/test_plate_crops/test_plate_crops.csv")
val_csv = Path("datasets/VNLP_processed/val.csv")
out_csv = Path("scratch/benchmark_100.csv")

rows = []

# 1. Load the 60 pre-cropped plates from scratch/test_plate_crops
if f60.exists():
    with open(f60, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fname = r["filename"]
            img_path = f"scratch/test_plate_crops/{fname}"
            lay = r.get("layout", "one_row")
            if "motorcycle" in fname:
                lay = "two_rows_label_xe_may"
            elif "two_rows" in fname:
                lay = "two_rows"
            else:
                lay = "one_row"
            rows.append({
                "image_path": img_path,
                "plate_text": r["plate_text"],
                "layout": lay,
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
            })

print(f"Loaded {len(rows)} samples from {f60}")

# 2. Complete to 100 samples from val.csv
c_one = sum(1 for r in rows if r["layout"] == "one_row")
c_two = sum(1 for r in rows if r["layout"] == "two_rows")
c_moto = sum(1 for r in rows if r["layout"] == "two_rows_label_xe_may")

needed_one = max(0, 35 - c_one)
needed_two = max(0, 35 - c_two)
needed_moto = max(0, 30 - c_moto)

with open(val_csv, "r", encoding="utf-8") as f:
    val_reader = csv.DictReader(f)
    for r in val_reader:
        sub = r.get("subset", "one_row")
        if sub == "one_row" and needed_one > 0:
            rows.append({
                "image_path": r["image_path"],
                "plate_text": r["plate_text"],
                "layout": sub,
                "x1": r.get("x1", ""),
                "y1": r.get("y1", ""),
                "x2": r.get("x2", ""),
                "y2": r.get("y2", ""),
            })
            needed_one -= 1
        elif sub == "two_rows" and needed_two > 0:
            rows.append({
                "image_path": r["image_path"],
                "plate_text": r["plate_text"],
                "layout": sub,
                "x1": r.get("x1", ""),
                "y1": r.get("y1", ""),
                "x2": r.get("x2", ""),
                "y2": r.get("y2", ""),
            })
            needed_two -= 1
        elif sub == "two_rows_label_xe_may" and needed_moto > 0:
            rows.append({
                "image_path": r["image_path"],
                "plate_text": r["plate_text"],
                "layout": sub,
                "x1": r.get("x1", ""),
                "y1": r.get("y1", ""),
                "x2": r.get("x2", ""),
                "y2": r.get("y2", ""),
            })
            needed_moto -= 1

        if len(rows) >= 100:
            break

print(f"Total benchmark samples: {len(rows)}")
c_one = sum(1 for r in rows if r["layout"] == "one_row")
c_two = sum(1 for r in rows if r["layout"] == "two_rows")
c_moto = sum(1 for r in rows if r["layout"] == "two_rows_label_xe_may")
print(f"Layout composition: {c_one} one_row, {c_two} two_rows, {c_moto} motorcycle")

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "plate_text", "layout", "x1", "y1", "x2", "y2"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {out_csv}")
