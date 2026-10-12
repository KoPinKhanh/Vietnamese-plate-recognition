"""
Evaluation script for License Plate OCR on VNLP test set.
Computes Exact Plate Accuracy, Character Error Rate (CER), Character Accuracy,
analyzes confusion patterns, and exports failed OCR cases for debugging.
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr.plate_text_utils import normalize_plate_text, calculate_cer, is_exact_match
from scripts.ocr_tesseract import run_tesseract


def evaluate(
    manifest_csv: str,
    output_errors_csv: str = "outputs/ocr_errors.csv",
    output_error_images_dir: str = "outputs/ocr_error_images",
    preprocess_strategy: str = "otsu",
    limit: int = None,
    save_error_limit: int = 50,
):
    manifest_path = Path(manifest_csv)
    assert manifest_path.exists(), f"Manifest file not found: {manifest_path}"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)
        
    if limit is not None and limit > 0:
        records = records[:limit]
        
    print(f"Evaluating OCR on {len(records)} test samples...")
    print(f"Preprocessing strategy: {preprocess_strategy}")
    
    errors_dir = Path(output_error_images_dir)
    errors_dir.mkdir(parents=True, exist_ok=True)
    
    total_samples = 0
    exact_matches = 0
    cer_list = []
    
    # Subsets metrics
    subset_stats = {}
    
    # Error records
    error_rows = []
    saved_error_images = 0
    
    for row in tqdm(records, desc="OCR Evaluation"):
        img_path = row["image_path"]
        gt_raw = row["plate_text"]
        gt = normalize_plate_text(gt_raw)
        layout = row.get("layout", "auto")
        subset = row.get("subset", "unknown")
        
        if subset not in subset_stats:
            subset_stats[subset] = {"total": 0, "exact": 0, "cers": []}
            
        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        ih, iw = img.shape[:2]
        crop = img[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
        if crop.size == 0:
            continue
            
        res = run_tesseract(crop, layout=layout, strategy=preprocess_strategy)
        pred = res["normalized_text"]
        
        match = is_exact_match(gt, pred)
        cer = calculate_cer(gt, pred)
        
        total_samples += 1
        cer_list.append(cer)
        if match:
            exact_matches += 1
            
        subset_stats[subset]["total"] += 1
        if match:
            subset_stats[subset]["exact"] += 1
        subset_stats[subset]["cers"].append(cer)
        
        if not match:
            error_info = {
                "image_path": img_path,
                "subset": subset,
                "layout": layout,
                "ground_truth": gt,
                "predicted": pred,
                "raw_pred": res["raw_text"],
                "cer": f"{cer:.4f}",
            }
            error_rows.append(error_info)
            
            # Save visual error image with comparison text
            if saved_error_images < save_error_limit:
                saved_error_images += 1
                # Annotate crop with GT and Pred
                ch, cw = crop.shape[:2]
                annotated = np.zeros((ch + 60, max(cw, 280), 3), dtype=np.uint8)
                annotated[:ch, :cw] = crop
                cv2.putText(
                    annotated,
                    f"GT  : {gt}",
                    (5, ch + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated,
                    f"Pred: {pred}",
                    (5, ch + 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )
                err_save_path = errors_dir / f"err_{saved_error_images:03d}_{gt}_vs_{pred}.jpg"
                cv2.imwrite(str(err_save_path), annotated)
                
    # Summary Report
    avg_cer = np.mean(cer_list) if cer_list else 0.0
    char_acc = max(0.0, 1.0 - avg_cer) * 100
    exact_acc = (exact_matches / max(total_samples, 1)) * 100
    
    report = []
    report.append("=" * 60)
    report.append("           OCR EVALUATION REPORT ON VNLP TEST SET          ")
    report.append("=" * 60)
    report.append(f"Samples Evaluated         : {total_samples}")
    report.append(f"Exact Plate Accuracy      : {exact_acc:.2f}% ({exact_matches}/{total_samples})")
    report.append(f"Average CER (Error Rate)  : {avg_cer:.4f}")
    report.append(f"Average Character Accuracy: {char_acc:.2f}%\n")
    
    report.append("Breakdown by Subset:")
    report.append("-" * 60)
    for sub, sdata in sorted(subset_stats.items()):
        sub_tot = sdata["total"]
        sub_exact = sdata["exact"]
        sub_acc = (sub_exact / max(sub_tot, 1)) * 100
        sub_cer = np.mean(sdata["cers"]) if sdata["cers"] else 0.0
        report.append(
            f"  {sub:<22} | Exact: {sub_acc:5.2f}% ({sub_exact:4d}/{sub_tot:4d}) | Mean CER: {sub_cer:.4f}"
        )
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Save errors CSV
    Path(output_errors_csv).parent.mkdir(parents=True, exist_ok=True)
    if error_rows:
        with open(output_errors_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image_path", "subset", "layout", "ground_truth", "predicted", "raw_pred", "cer"],
            )
            writer.writeheader()
            writer.writerows(error_rows)
            
    print(f"\n[+] Error details saved to: {output_errors_csv}")
    print(f"[+] Sample failure visualizations saved to: {output_error_images_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR on VNLP test manifest.")
    parser.add_argument("--manifest", type=str, default="datasets/VNLP_processed/test.csv")
    parser.add_argument("--preprocess", type=str, default="otsu")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for fast testing.")
    parser.add_argument("--errors-csv", type=str, default="outputs/ocr_errors.csv")
    parser.add_argument("--error-images", type=str, default="outputs/ocr_error_images")
    args = parser.parse_args()
    
    evaluate(
        manifest_csv=args.manifest,
        output_errors_csv=args.errors_csv,
        output_error_images_dir=args.error_images,
        preprocess_strategy=args.preprocess,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
