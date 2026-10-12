"""
Creates datasets/VNLP_processed/val_unseen_text_v2.csv and reports/val_unseen_text_v2_summary.md
Definition: Keep VAL samples only if their plate text does not appear in TRAIN V2 (train_v2a.csv).
"""

import csv
from collections import Counter
from pathlib import Path


def main():
    train_v2_path = Path("datasets/VNLP_processed/train_v2a.csv")
    with open(train_v2_path, "r", encoding="utf-8") as f:
        train_texts = set(r["plate_text"].strip().upper() for r in csv.DictReader(f))
    print(f"Unique texts in TRAIN V2: {len(train_texts)}")

    val_path = Path("datasets/VNLP_processed/val.csv")
    with open(val_path, "r", encoding="utf-8") as f:
        val_reader = csv.DictReader(f)
        fieldnames = val_reader.fieldnames
        val_rows = list(val_reader)

    total_val = len(val_rows)
    print(f"Total VAL samples: {total_val}")

    unseen_val_rows = [r for r in val_rows if r["plate_text"].strip().upper() not in train_texts]
    unseen_count = len(unseen_val_rows)
    print(f"Unseen VAL count: {unseen_count} ({unseen_count / total_val * 100:.2f}%)")

    layouts = Counter(r.get("subset", r.get("layout", "unknown")) for r in unseen_val_rows)
    lengths = Counter(len(r["plate_text"].strip()) for r in unseen_val_rows)

    out_csv = Path("datasets/VNLP_processed/val_unseen_text_v2.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unseen_val_rows)
    print(f"Saved unseen VAL to {out_csv}")

    md_lines = [
        "# TỔNG HỢP TẬP VALIDATION CHƯA THẤY KÝ TỰ (VAL UNSEEN TEXT V2)",
        "",
        "**Thời gian tạo:** 2026-09-24",
        f"**Tập nguồn Train V2:** `{train_v2_path}` (31,047 mẫu: VNLP Train + PBL4 unique)",
        f"**Tập nguồn Validation:** `{val_path}` ({total_val} mẫu)",
        f"**Tập tạo mới:** `{out_csv}`",
        "",
        "---",
        "",
        "## 1. Thống Kê Tổng Quan",
        f"- **Tổng số mẫu trong VAL:** **{total_val}** mẫu",
        f"- **Số mẫu VAL có chuỗi ký tự chưa từng xuất hiện trong TRAIN V2:** **{unseen_count}** mẫu ({unseen_count / total_val * 100:.2f}%)",
        f"- **Số chuỗi ký tự VAL đã xuất hiện trong TRAIN V2 (Overlap):** **{total_val - unseen_count}** mẫu ({(total_val - unseen_count) / total_val * 100:.2f}%)",
        "",
        "---",
        "",
        "## 2. Phân Bố Theo Kiểu Biển (Layout Breakdown)",
        "| Kiểu biển (Layout) | Số mẫu | Tỷ lệ (%) |",
        "| :--- | :---: | :---: |",
        f"| 1 Dòng (`one_row`) | {layouts.get('one_row', 0)} | {layouts.get('one_row', 0) / unseen_count * 100:.2f}% |",
        f"| 2 Dòng ô tô (`two_rows`) | {layouts.get('two_rows', 0)} | {layouts.get('two_rows', 0) / unseen_count * 100:.2f}% |",
        f"| 2 Dòng xe máy (`two_rows_label_xe_may`) | {layouts.get('two_rows_label_xe_may', 0)} | {layouts.get('two_rows_label_xe_may', 0) / unseen_count * 100:.2f}% |",
        "",
        "---",
        "",
        "## 3. Phân Bố Độ Dài Biển Số (Label Length Distribution)",
        "| Độ dài (Ký tự) | Số mẫu | Tỷ lệ (%) |",
        "| :---: | :---: | :---: |",
    ]

    for length, count in sorted(lengths.items()):
        md_lines.append(f"| {length} | {count} | {count / unseen_count * 100:.2f}% |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Ý Nghĩa và Khuyến Nghị Sử Dụng",
        f"- Tập `val_unseen_text_v2.csv` có **{unseen_count}** mẫu ({unseen_count/total_val*100:.1f}% tập VAL).",
        "- Tập này hoàn toàn độc lập với TEST và cho phép quan sát sớm khả năng tổng quát hóa trong quá trình fine-tune.",
        "- **Lưu ý:** Metric chính để quyết định checkpoint candidate vẫn là **toàn bộ tập VAL (3,728 mẫu)** để đảm bảo độ tin cậy và kích thước mẫu tối ưu.",
        ""
    ])

    report_path = Path("reports/val_unseen_text_v2_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
