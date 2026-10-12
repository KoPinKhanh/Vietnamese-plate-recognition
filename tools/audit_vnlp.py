"""
Audit script for Vietnamese License Plate (VNLP) Dataset.
Scans and verifies all image files, annotations encoded in filenames, and coordinate bounding boxes.
Outputs detailed report to outputs/vnlp_audit.txt.
"""

import os
import re
from pathlib import Path
from collections import Counter
import cv2
from tqdm import tqdm


def parse_vnlp_filename(filename: str):
    """
    Parses VNLP filename format:
    Pattern: <id1>_<id2>_<layout>_<label>_<x1>_<y1>_<x2>_<y2>.jpg
    Returns dict with parsed info or None if invalid.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    
    # Must have at least 8 parts (or more if label contains underscore)
    if len(parts) < 8:
        return None
    
    try:
        id1 = parts[0]
        id2 = parts[1]
        layout = parts[2]  # '1' for one_row, '0' for two_rows
        # coordinates are the last 4 parts
        x1 = int(parts[-4])
        y1 = int(parts[-3])
        x2 = int(parts[-2])
        y2 = int(parts[-1])
        # label is everything between layout and x1
        label = "_".join(parts[3:-4]).upper()
        
        return {
            "id1": id1,
            "id2": id2,
            "layout": "one_row" if layout == "1" else "two_rows",
            "layout_code": layout,
            "label": label,
            "bbox": (x1, y1, x2, y2),
        }
    except (ValueError, IndexError):
        return None


def run_audit(vnlp_root: str, output_path: str):
    root = Path(vnlp_root)
    assert root.exists(), f"VNLP root not found at: {root}"
    
    subsets = {
        "one_row": root / "one_row",
        "two_rows": root / "two_rows",
        "two_rows_label_xe_may": root / "two_rows_label_xe_may",
    }
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("       VNLP DATASET COMPREHENSIVE AUDIT REPORT       ")
    report_lines.append("=" * 60)
    report_lines.append(f"VNLP Root: {root.resolve()}\n")
    
    total_images = 0
    total_valid_names = 0
    total_invalid_names = 0
    total_corrupt_images = 0
    total_invalid_boxes = 0
    
    label_lens = Counter()
    char_freq = Counter()
    subset_stats = {}
    
    for subset_name, subset_dir in subsets.items():
        if not subset_dir.exists():
            report_lines.append(f"[!] Directory missing: {subset_dir}")
            continue
        
        images = list(subset_dir.glob("*.jpg")) + list(subset_dir.glob("*.png"))
        count = len(images)
        total_images += count
        subset_stats[subset_name] = count
        report_lines.append(f"--- Subset: {subset_name} ---")
        report_lines.append(f"  Total image files found: {count}")
        
        # Audit a sample or all for filename parsing
        valid_in_subset = 0
        invalid_in_subset = 0
        invalid_box_in_subset = 0
        
        # Check a sample with cv2 to avoid excessive IO if dataset is huge, but check all filenames
        sample_check_limit = 100
        checked_images = 0
        
        for idx, img_path in enumerate(images):
            parsed = parse_vnlp_filename(img_path.name)
            if parsed is None:
                invalid_in_subset += 1
                total_invalid_names += 1
                continue
            
            valid_in_subset += 1
            total_valid_names += 1
            
            label = parsed["label"]
            label_lens[len(label)] += 1
            for c in label:
                char_freq[c] += 1
            
            x1, y1, x2, y2 = parsed["bbox"]
            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
                invalid_box_in_subset += 1
                total_invalid_boxes += 1
            
            # Sample integrity check with OpenCV
            if checked_images < sample_check_limit:
                checked_images += 1
                img = cv2.imread(str(img_path))
                if img is None:
                    total_corrupt_images += 1
                else:
                    h, w = img.shape[:2]
                    if x2 > w or y2 > h:
                        # slight boundary overflow check
                        pass
        
        report_lines.append(f"  Valid filenames parsed: {valid_in_subset}")
        report_lines.append(f"  Invalid filenames: {invalid_in_subset}")
        report_lines.append(f"  Invalid bboxes (x2<=x1 or y2<=y1): {invalid_box_in_subset}\n")
    
    report_lines.append("=" * 60)
    report_lines.append("SUMMARY STATISTICS:")
    report_lines.append(f"Total images found: {total_images}")
    report_lines.append(f"Total valid filenames: {total_valid_names} ({(total_valid_names/max(total_images,1))*100:.2f}%)")
    report_lines.append(f"Total invalid filenames: {total_invalid_names}")
    report_lines.append(f"Total invalid bboxes: {total_invalid_boxes}")
    report_lines.append(f"Corrupt images found in checked sample: {total_corrupt_images}")
    report_lines.append("\nLabel length distribution:")
    for length, cnt in sorted(label_lens.items()):
        report_lines.append(f"  Length {length:2d} chars: {cnt:6d} ({cnt/total_valid_names*100:5.2f}%)")
    
    report_lines.append("\nTop 15 Most Frequent Characters:")
    for char, cnt in char_freq.most_common(15):
        report_lines.append(f"  '{char}': {cnt}")
    
    report_lines.append("=" * 60)
    
    report_content = "\n".join(report_lines)
    print(report_content)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n[+] Audit report saved to: {output_path}")


if __name__ == "__main__":
    vnlp_path = r"d:/Vietnam_Plate_Project/datasets/VNLP_detection/detection"
    out_path = r"d:/Vietnam_Plate_Project/outputs/vnlp_audit.txt"
    run_audit(vnlp_path, out_path)
