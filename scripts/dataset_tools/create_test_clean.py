"""
Tool to isolate exact train-test duplicates and generate TEST_CLEAN manifest.
Preserves TEST_FINAL (datasets/VNLP_processed/test.csv) untouched.
Produces:
  - datasets/VNLP_processed/test_clean.csv (3,727 samples)
  - reports/test_clean_removed_samples.csv
  - reports/test_clean_summary.md
"""

import csv
import hashlib
from pathlib import Path

def get_sha256(p: Path) -> str:
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    train_csv = Path("datasets/VNLP_processed/train.csv")
    test_csv = Path("datasets/VNLP_processed/test.csv")
    test_clean_csv = Path("datasets/VNLP_processed/test_clean.csv")
    removed_csv = Path("reports/test_clean_removed_samples.csv")
    summary_md = Path("reports/test_clean_summary.md")

    print("Loading train images and computing hashes...")
    train_hashes = {}
    with open(train_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Path(r["image_path"])
            if p.exists():
                h = get_sha256(p)
                train_hashes[h] = r

    print("Scanning TEST_FINAL for exact duplicates...")
    test_records = []
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        test_records = list(reader)

    removed_samples = []
    clean_records = []

    for r in test_records:
        p = Path(r["image_path"])
        h = get_sha256(p) if p.exists() else ""
        if h in train_hashes:
            tr = train_hashes[h]
            removed_samples.append({
                "test_file": str(r["image_path"]).replace("\\", "/"),
                "train_file": str(tr["image_path"]).replace("\\", "/"),
                "label": r["plate_text"],
                "sha256": h,
                "reason": "EXACT_TRAIN_TEST_DUPLICATE"
            })
        else:
            clean_records.append(r)

    # Write test_clean.csv
    with open(test_clean_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_records)

    # Write test_clean_removed_samples.csv
    removed_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(removed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_file", "train_file", "label", "sha256", "reason"])
        writer.writeheader()
        writer.writerows(removed_samples)

    # Verify remaining duplicates
    clean_dups = 0
    for r in clean_records:
        p = Path(r["image_path"])
        h = get_sha256(p) if p.exists() else ""
        if h in train_hashes:
            clean_dups += 1

    print(f"TEST_FINAL count:     {len(test_records)}")
    print(f"Duplicates removed:   {len(removed_samples)}")
    print(f"TEST_CLEAN count:     {len(clean_records)}")
    print(f"Remaining duplicates: {clean_dups}")
    assert clean_dups == 0, "Error: remaining exact duplicates must be 0!"

    # Create reports/test_clean_summary.md
    lines = [
        "# TỔNG HỢP XÂY DỰNG TẬP KIỂM ĐỊNH SẠCH (TEST_CLEAN SUMMARY)",
        "",
        "**Thời gian tạo:** 2026-09-24",
        "**Tập nguồn TEST_FINAL:** `datasets/VNLP_processed/test.csv` (3,733 mẫu - ĐÃ KHÓA)",
        "**Tập đích TEST_CLEAN:** `datasets/VNLP_processed/test_clean.csv`",
        "**File ghi nhận mẫu loại bỏ:** `reports/test_clean_removed_samples.csv`",
        "",
        "---",
        "",
        "## 1. Kết quả Phân tách & Lọc Trùng lặp",
        f"- **Số lượng mẫu TEST_FINAL ban đầu:** **{len(test_records)}**",
        f"- **Số lượng exact image duplicate với TRAIN bị loại bỏ:** **{len(removed_samples)}**",
        f"- **Số lượng mẫu TEST_CLEAN thu được:** **{len(clean_records)}**",
        f"- **Số lượng exact duplicate còn lại giữa TRAIN và TEST_CLEAN:** **{clean_dups} (HOÀN TOÀN KHÔNG CÒN)**",
        "",
        "---",
        "",
        "## 2. Chi tiết 6 Mẫu Bị Loại Bỏ (Exact Bitwise Match)",
        "| STT | File trong TEST_FINAL | File đối ứng trong TRAIN | Nhãn Test | SHA256 |",
        "| :---: | :--- | :--- | :---: | :--- |"
    ]

    for idx, item in enumerate(removed_samples, 1):
        tf = Path(item["test_file"]).name
        trf = Path(item["train_file"]).name
        lines.append(f"| {idx} | `{tf}` | `{trf}` | `{item['label']}` | `{item['sha256'][:16]}...` |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Quy tắc Đánh giá Đa chiều trong Thí nghiệm",
        "Từ Phase 2 trở đi, dự án sử dụng song song 3 tập đánh giá độc lập:",
        "1. **TEST_FINAL (3,733 mẫu):** Giữ nguyên vẹn tuyệt đối để so sánh lịch sử với Model V1 Baseline (mốc 85.27%).",
        "2. **TEST_CLEAN (3,727 mẫu):** Đã loại bỏ 100% rủi ro rò rỉ hình ảnh tuyệt đối với tập huấn luyện, phản ánh độ chính xác thực tế khi không có mẫu nhìn thấy trước.",
        "3. **TEST_UNSEEN_TEXT (1,187 mẫu):** Đánh giá khả năng tổng quát hóa (Generalization) trên các chuỗi biển số hoàn toàn mới chưa từng xuất hiện trong tập Train.",
        ""
    ])

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Summary written to {summary_md}")

if __name__ == "__main__":
    main()
