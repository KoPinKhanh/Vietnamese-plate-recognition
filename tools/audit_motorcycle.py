import csv
import sys
from pathlib import Path

csv_path = Path("reports/fresh_holdout_500_results.csv")
with open(csv_path, encoding="utf-8") as f:
    records = list(csv.DictReader(f))

moto = [r for r in records if r["gt_layout"] == "motorcycle"]
print(f"Total motorcycle records: {len(moto)}")

d_det = sum(1 for r in moto if r["direct_detected"].lower() == "true")
r1_det = sum(1 for r in moto if r["r1_detected"].lower() == "true")
d_e2e = sum(1 for r in moto if r["direct_e2e"].lower() == "true")
r1_e2e = sum(1 for r in moto if r["r1_e2e"].lower() == "true")

print(f"Direct Detected: {d_det} / {len(moto)}")
print(f"R1 Detected    : {r1_det} / {len(moto)}")
print(f"Direct E2E     : {d_e2e} / {len(moto)}")
print(f"R1 E2E         : {r1_e2e} / {len(moto)}")

both_det = sum(1 for r in moto if r["direct_detected"].lower() == "true" and r["r1_detected"].lower() == "true")
dir_only_det = sum(1 for r in moto if r["direct_detected"].lower() == "true" and r["r1_detected"].lower() != "true")
r1_only_det = sum(1 for r in moto if r["direct_detected"].lower() != "true" and r["r1_detected"].lower() == "true")
neither_det = sum(1 for r in moto if r["direct_detected"].lower() != "true" and r["r1_detected"].lower() != "true")

print(f"\nDetection States:")
print(f"  BOTH_DETECT      : {both_det}")
print(f"  DIRECT_ONLY_DETECT: {dir_only_det}")
print(f"  R1_ONLY_DETECT   : {r1_only_det}")
print(f"  NEITHER_DETECT   : {neither_det}")

both_e2e = sum(1 for r in moto if r["direct_e2e"].lower() == "true" and r["r1_e2e"].lower() == "true")
dir_only_e2e = sum(1 for r in moto if r["direct_e2e"].lower() == "true" and r["r1_e2e"].lower() != "true")
r1_only_e2e = sum(1 for r in moto if r["direct_e2e"].lower() != "true" and r["r1_e2e"].lower() == "true")
neither_e2e = sum(1 for r in moto if r["direct_e2e"].lower() != "true" and r["r1_e2e"].lower() != "true")

print(f"\nE2E States:")
print(f"  BOTH_E2E      : {both_e2e}")
print(f"  DIRECT_ONLY_E2E: {dir_only_e2e}")
print(f"  R1_ONLY_E2E   : {r1_only_e2e}")
print(f"  NEITHER_E2E   : {neither_e2e}")

print("\nSamples where Direct E2E != R1 E2E:")
for r in moto:
    de = r["direct_e2e"].lower() == "true"
    re = r["r1_e2e"].lower() == "true"
    dd = r["direct_detected"].lower() == "true"
    rd = r["r1_detected"].lower() == "true"
    if de != re:
        print(f"Sample: {r['sample_id']} | GT: {r['ground_truth']} | DirDet: {dd} | R1Det: {rd} | DirOCR: '{r['direct_ocr']}' (e2e={de}) | R1OCR: '{r['r1_ocr']}' (e2e={re}) | R1Source: {r['r1_source']} | CascadeInvoked: {r['cascade_invoked']} | DirCands: {r['direct_candidates_count']}")
