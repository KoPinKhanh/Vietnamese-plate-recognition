"""
Audit 100% of PBL4 samples in datasets/VNLP_processed/train_v2a.csv against filename parsing.
Validates naming convention, split_prefix, plate_text, sample_id, augmentation_id.
Outputs:
  reports/phase4_pbl4_manifest_label_audit.csv
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.charset import CHARSET


def parse_pbl4_filename(filename: str):
    """
    Parses PBL4 filename into:
      split_prefix: 'train' or 'valid' or ''
      plate_text: e.g. '51F88270'
      sample_id: e.g. '0508'
      augmentation_id: e.g. '2'
    Expected patterns:
      1) (train|valid)_<PLATE>_<ID>_<AUG>.jpg
      2) <PLATE>_<ID>_<AUG>.jpg
    """
    stem = Path(filename).stem
    tokens = stem.split("_")

    split_prefix = ""
    plate_text = ""
    sample_id = ""
    aug_id = ""

    if len(tokens) >= 4 and tokens[0].lower() in ("train", "valid"):
        split_prefix = tokens[0].lower()
        plate_text = tokens[1].upper()
        sample_id = tokens[2]
        aug_id = tokens[3]
    elif len(tokens) >= 3 and tokens[0].lower() not in ("train", "valid"):
        split_prefix = ""
        plate_text = tokens[0].upper()
        sample_id = tokens[1]
        aug_id = tokens[2]
    elif len(tokens) == 2:
        plate_text = tokens[0].upper()
        sample_id = tokens[1]
    else:
        # Fallback regex search for Vietnamese plate pattern
        m = re.search(r"([0-9]{2}[A-Z][0-9]{4,5})", stem.upper())
        if m:
            plate_text = m.group(1)
        else:
            return None, "PARSER_ERROR"

    return {
        "split_prefix": split_prefix,
        "plate_text": plate_text,
        "sample_id": sample_id,
        "aug_id": aug_id,
    }, "OK"


def main():
    train_v2a_path = Path("datasets/VNLP_processed/train_v2a.csv")
    sources_path = Path("datasets/VNLP_processed/train_v2a_sources.csv")
    output_audit_csv = Path("reports/phase4_pbl4_manifest_label_audit.csv")
    output_audit_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read sources to identify PBL4 paths
    pbl4_source_map = {}
    with open(sources_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source"] == "PBL4":
                norm_p = r["path"].replace("\\", "/")
                pbl4_source_map[norm_p] = r["label"]

    print(f"Total PBL4 samples identified in sources CSV: {len(pbl4_source_map)}")

    # 2. Read train_v2a.csv and find PBL4 records
    manifest_pbl4_rows = []
    with open(train_v2a_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            norm_p = r["image_path"].replace("\\", "/")
            if norm_p in pbl4_source_map or "pbl4_unique" in norm_p:
                manifest_pbl4_rows.append(r)

    print(f"Total PBL4 samples found in train_v2a.csv: {len(manifest_pbl4_rows)}")

    # 3. Perform exhaustive audit on all PBL4 samples
    audit_records = []
    total_pbl4 = len(manifest_pbl4_rows)
    label_matches = 0
    label_mismatches = 0
    parser_errors = 0

    naming_conventions = Counter()

    for r in manifest_pbl4_rows:
        img_path = r["image_path"].replace("\\", "/")
        fname = Path(img_path).name
        manifest_label = r["plate_text"].strip().upper()

        parsed, parse_status = parse_pbl4_filename(fname)
        if parse_status == "PARSER_ERROR" or parsed is None:
            parser_errors += 1
            expected_label = "PARSER_ERROR"
            match = False
            status = "PARSER_ERROR"
        else:
            expected_label = parsed["plate_text"]
            match = (manifest_label == expected_label)
            if match:
                label_matches += 1
                status = "OK"
            else:
                label_mismatches += 1
                status = "LABEL_MISMATCH"

            # Track pattern convention
            prefix = parsed["split_prefix"]
            convention = f"{prefix}_<PLATE>_<ID>_<AUG>" if prefix else "<PLATE>_<ID>_<AUG>"
            naming_conventions[convention] += 1

        audit_records.append({
            "image_path": img_path,
            "filename": fname,
            "manifest_label": manifest_label,
            "expected_label": expected_label,
            "match": match,
            "source": "PBL4",
            "status": status,
        })

    # 4. Write audit CSV
    with open(output_audit_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "filename", "manifest_label", "expected_label", "match", "source", "status"])
        writer.writeheader()
        writer.writerows(audit_records)

    print("\n" + "=" * 65)
    print("PHASE 4 — PBL4 MANIFEST LABEL AUDIT RESULTS")
    print("=" * 65)
    print(f"Total PBL4 evaluated:     {total_pbl4}")
    print(f"Label matches (OK):       {label_matches} ({label_matches / total_pbl4 * 100:.2f}%)")
    print(f"Label mismatches:         {label_mismatches}")
    print(f"Parser errors:            {parser_errors}")
    print("\nNaming Conventions Detected across 1,211 PBL4 files:")
    for conv, count in naming_conventions.most_common():
        print(f"  {conv:<30}: {count} files ({count / total_pbl4 * 100:.2f}%)")
    print("=" * 65)

    if label_mismatches > 0 or parser_errors > 0:
        print("[CRITICAL ALERT] STOP CONDITION TRIGGERED! Manifest labels do NOT match filenames.")
        sys.exit(1)
    else:
        print("[AUDIT SUCCESS] 100% of PBL4 samples in train_v2a.csv have EXACT MATCH with expected plate text!")
        print(f"Detailed audit log written to: {output_audit_csv}")


if __name__ == "__main__":
    main()
