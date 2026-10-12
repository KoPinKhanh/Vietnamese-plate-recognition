"""
Quick A/B comparison: PARSeq vs LPRNet on the exact same 177 detected YOLO crops.
"""

import csv
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr.pi_lprnet_ocr import PlateOCR
from ocr.plate_text_utils import normalize_plate_text


def run_ab_comparison():
    lpr_path = PROJECT_ROOT / "weights/ocr/best_lprnet_vn.onnx"
    ocr = PlateOCR(model_path=str(lpr_path))

    records = []
    with open(PROJECT_ROOT / "reports/e2e_real_world_results.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["detected"].lower() == "true":
                records.append(r)

    total_detected = len(records)
    print(f"Total detected samples for A/B comparison: {total_detected}")

    lpr_correct = 0
    parseq_correct = 0

    for r in records:
        crop_path = PROJECT_ROOT / "outputs/e2e_eval/crops/yolo" / f"{r['sample_id']}_{r['ground_truth']}_yolo.jpg"
        if not crop_path.exists():
            continue
        img = cv2.imread(str(crop_path))
        res = ocr.recognize(img, layout="auto")
        pred = normalize_plate_text(res["text"])
        gt = r["ground_truth"]
        if pred == gt:
            lpr_correct += 1
        if r["yolo_crop_correct"].lower() == "true":
            parseq_correct += 1

    parseq_acc = (parseq_correct / total_detected) * 100.0
    lpr_acc = (lpr_correct / total_detected) * 100.0
    delta = parseq_acc - lpr_acc

    print("=" * 60)
    print("A/B COMPARISON ON MATCHED YOLO CROPS (N=177)")
    print(f"PARSeq-Tiny VNLP: {parseq_correct} / {total_detected} ({parseq_acc:.2f}%)")
    print(f"LPRNet-Lite ONNX: {lpr_correct} / {total_detected} ({lpr_acc:.2f}%)")
    print(f"Accuracy Delta:   {delta:+.2f} pp in favor of PARSeq")
    print(f"PARSeq Full E2E:  {parseq_correct} / 300 ({parseq_correct/300*100:.2f}%)")
    print(f"LPRNet Full E2E:  {lpr_correct} / 300 ({lpr_correct/300*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_ab_comparison()
