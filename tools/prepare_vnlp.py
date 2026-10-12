"""
Prepares VNLP dataset for OCR evaluation and training.
Parses annotations from image filenames, creates balanced train/val/test splits (80/10/10, seed=42),
and exports manifest CSVs into datasets/VNLP_processed/.
"""

import os
import csv
import random
from pathlib import Path
from tqdm import tqdm


def parse_vnlp_filename(filename: str):
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 8:
        return None
    try:
        layout = "one_row" if parts[2] == "1" else "two_rows"
        x1 = int(parts[-4])
        y1 = int(parts[-3])
        x2 = int(parts[-2])
        y2 = int(parts[-1])
        label = "_".join(parts[3:-4]).upper()
        return {
            "layout": layout,
            "plate_text": label,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
    except (ValueError, IndexError):
        return None


def prepare_manifests(
    vnlp_root: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    random.seed(seed)
    root = Path(vnlp_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    subsets = ["one_row", "two_rows", "two_rows_label_xe_may"]
    all_records = []
    
    print(f"Scanning VNLP directories under: {root}")
    for subset in subsets:
        subdir = root / subset
        if not subdir.exists():
            print(f"Warning: {subdir} not found.")
            continue
            
        img_files = sorted(list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")))
        print(f"  Found {len(img_files)} images in {subset}")
        
        for p in img_files:
            parsed = parse_vnlp_filename(p.name)
            if parsed is None:
                continue
                
            all_records.append({
                "image_path": str(p.resolve()).replace("\\", "/"),
                "relative_path": f"{subset}/{p.name}",
                "plate_text": parsed["plate_text"],
                "layout": parsed["layout"],
                "subset": subset,
                "x1": parsed["x1"],
                "y1": parsed["y1"],
                "x2": parsed["x2"],
                "y2": parsed["y2"],
            })
            
    print(f"Total valid parsed records: {len(all_records)}")
    
    # Stratified shuffle by subset to keep one_row vs two_rows balanced in train/val/test
    subset_buckets = {}
    for r in all_records:
        sub = r["subset"]
        if sub not in subset_buckets:
            subset_buckets[sub] = []
        subset_buckets[sub].append(r)
        
    train_split, val_split, test_split = [], [], []
    
    for sub, records in subset_buckets.items():
        random.shuffle(records)
        n = len(records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_split.extend(records[:n_train])
        val_split.extend(records[n_train:n_train + n_val])
        test_split.extend(records[n_train + n_val:])
        
    random.shuffle(train_split)
    random.shuffle(val_split)
    random.shuffle(test_split)
    
    fieldnames = ["image_path", "relative_path", "plate_text", "layout", "subset", "x1", "y1", "x2", "y2"]
    
    def save_csv(filepath, data):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
            
    save_csv(out_dir / "train.csv", train_split)
    save_csv(out_dir / "val.csv", val_split)
    save_csv(out_dir / "test.csv", test_split)
    save_csv(out_dir / "all.csv", all_records)
    
    print("\nDataset Split Summary:")
    print(f"  Train: {len(train_split):6d} records ({(len(train_split)/len(all_records))*100:.1f}%)")
    print(f"  Val  : {len(val_split):6d} records ({(len(val_split)/len(all_records))*100:.1f}%)")
    print(f"  Test : {len(test_split):6d} records ({(len(test_split)/len(all_records))*100:.1f}%)")
    print(f"  Total: {len(all_records):6d} records")
    print(f"\nManifests successfully written to: {out_dir}")


if __name__ == "__main__":
    vnlp_dir = r"d:/Vietnam_Plate_Project/datasets/VNLP_detection/detection"
    processed_dir = r"d:/Vietnam_Plate_Project/datasets/VNLP_processed"
    prepare_manifests(vnlp_dir, processed_dir)
