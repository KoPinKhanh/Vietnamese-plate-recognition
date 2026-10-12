"""
Audit tool for OCR manifests: train.csv, val.csv, test.csv.
Checks image existence, label lengths, character frequencies, non-alphanumeric chars,
image duplicate hash leakage across splits, and plate text overlap.
Generates datasets/VNLP_processed/test_unseen_text.csv.
"""

import csv
import hashlib
import os
from collections import Counter
from pathlib import Path


def get_image_hash(filepath: str) -> str:
    """Computes MD5 hash of image content to detect identical images."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_manifests(data_dir: str):
    base_dir = Path(data_dir)
    splits = ["train", "val", "test"]
    split_data = {}

    print("=" * 65)
    print("           OCR DATASET MANIFEST AUDIT & LEAKAGE CHECK          ")
    print("=" * 65)

    all_chars = Counter()
    non_alphanumeric_chars = set()

    for s in splits:
        csv_path = base_dir / f"{s}.csv"
        assert csv_path.exists(), f"Missing {csv_path}"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        split_data[s] = records
        print(f"Loaded {s}.csv: {len(records)} records")

    # 1. Statistics
    print("\n--- 1. Subset Counts & Label Lengths ---")
    for s, records in split_data.items():
        subset_counts = Counter(r["subset"] for r in records)
        lengths = [len(r["plate_text"]) for r in records]
        min_l, max_l = min(lengths), max(lengths)
        print(f"[{s.upper()} - {len(records)} total]")
        print(f"  one_row: {subset_counts.get('one_row', 0)}")
        print(f"  two_rows: {subset_counts.get('two_rows', 0)}")
        print(f"  two_rows_label_xe_may: {subset_counts.get('two_rows_label_xe_may', 0)}")
        print(f"  Min label length: {min_l}, Max label length: {max_l}")

        for r in records:
            txt = r["plate_text"].upper()
            for ch in txt:
                all_chars[ch] += 1
                if not (("A" <= ch <= "Z") or ("0" <= ch <= "9")):
                    non_alphanumeric_chars.add(ch)

    # Character frequency & charset
    print("\n--- 2. Charset & Character Frequencies ---")
    unique_chars = sorted(all_chars.keys())
    print(f"Total unique characters found: {len(unique_chars)}")
    print("Charset:", "".join(unique_chars))
    print("Character counts:")
    for ch in sorted(unique_chars):
        print(f"  '{ch}': {all_chars[ch]:6d}")

    if non_alphanumeric_chars:
        print(f"[!] WARNING: Found characters outside [A-Z0-9]: {non_alphanumeric_chars}")
    else:
        print("[+] All characters strictly within [A-Z0-9]. No strange symbols!")

    # 3. Image existence check (sample or all)
    print("\n--- 3. Verifying Image File Existence ---")
    missing_images = 0
    checked = 0
    for s, records in split_data.items():
        for r in records:
            checked += 1
            if not os.path.exists(r["image_path"]):
                missing_images += 1
    print(f"Checked {checked} image paths. Missing images: {missing_images}")
    assert missing_images == 0, f"Found {missing_images} missing images!"

    # 4. Image duplicate check by hash
    print("\n--- 4. Checking Image Duplicate Leakage (Hash-based) ---")
    # To be fast, check a sample of 1000 from each split or check all
    # Since checking 37k files on SSD takes ~5s with 16KB reads, let's hash all
    train_hashes = {}
    print("Hashing train images...")
    for r in split_data["train"]:
        h = get_image_hash(r["image_path"])
        train_hashes[h] = r["image_path"]

    val_dup = 0
    print("Checking val duplicates against train...")
    for r in split_data["val"]:
        h = get_image_hash(r["image_path"])
        if h in train_hashes:
            val_dup += 1

    test_dup = 0
    print("Checking test duplicates against train...")
    for r in split_data["test"]:
        h = get_image_hash(r["image_path"])
        if h in train_hashes:
            test_dup += 1

    print(f"Duplicate images between Train and Val : {val_dup} / {len(split_data['val'])} ({val_dup/len(split_data['val'])*100:.2f}%)")
    print(f"Duplicate images between Train and Test: {test_dup} / {len(split_data['test'])} ({test_dup/len(split_data['test'])*100:.2f}%)")

    # 5. Text Overlap check
    print("\n--- 5. Checking Plate Text Overlap ---")
    train_texts = set(r["plate_text"].upper() for r in split_data["train"])
    val_texts = set(r["plate_text"].upper() for r in split_data["val"])
    test_texts = set(r["plate_text"].upper() for r in split_data["test"])

    print(f"Unique plate texts in Train: {len(train_texts)}")
    print(f"Unique plate texts in Val  : {len(val_texts)}")
    print(f"Unique plate texts in Test : {len(test_texts)}")

    val_overlap = sum(1 for r in split_data["val"] if r["plate_text"].upper() in train_texts)
    test_overlap = sum(1 for r in split_data["test"] if r["plate_text"].upper() in train_texts)

    print(f"Val plates appearing in Train : {val_overlap} / {len(split_data['val'])} ({val_overlap/len(split_data['val'])*100:.2f}%)")
    print(f"Test plates appearing in Train: {test_overlap} / {len(split_data['test'])} ({test_overlap/len(split_data['test'])*100:.2f}%)")

    # 6. Generate test_unseen_text.csv
    print("\n--- 6. Generating test_unseen_text.csv ---")
    unseen_test = [r for r in split_data["test"] if r["plate_text"].upper() not in train_texts]
    unseen_path = base_dir / "test_unseen_text.csv"

    fieldnames = list(split_data["test"][0].keys())
    with open(unseen_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unseen_test)

    print(f"Saved {len(unseen_test)} unseen test samples to: {unseen_path}")
    print(f"Percentage of test set with unseen text: {len(unseen_test)/len(split_data['test'])*100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    audit_manifests("datasets/VNLP_processed")
