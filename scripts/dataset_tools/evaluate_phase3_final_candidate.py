"""
Final Phase 3 Evaluation of Selected Candidate (FT-B) vs Baseline V1 on Test Sets:
- TEST_FINAL (datasets/VNLP_processed/test.csv)
- TEST_CLEAN (datasets/VNLP_processed/test_clean.csv)
- TEST_UNSEEN_TEXT (datasets/VNLP_processed/test_unseen_text.csv)

Outputs:
- reports/phase3_v1_vs_candidate.md
- reports/phase3_fixed_cases.csv
- reports/phase3_regressions.csv
- reports/phase3_confusion_comparison.md
"""

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.charset import CHARSET, NUM_CLASSES
from ocr.models.lprnet_lite import LPRNetLite
from tools.evaluate_ocr_model import evaluate_manifest


def load_model(ckpt_path: Path, device: torch.device):
    model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def build_confusion_matrix(records: List[Dict[str, Any]]) -> Counter:
    pair_counts = Counter()
    for r in records:
        if not r["exact"]:
            gt = r["ground_truth"]
            pred = r["prediction"]
            if len(gt) == len(pred):
                for g, p in zip(gt, pred):
                    if g != p:
                        pair_counts[f"{g} -> {p}"] += 1
    return pair_counts


def delta_str(cand: float, v1: float, unit: str = "pp") -> str:
    d = cand - v1
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.2f} {unit}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v1_ckpt = Path("weights/ocr/best_lprnet_vn.pth")
    cand_ckpt = Path("weights/ocr/experiments/phase3_ft_b/best.pth")

    print(f"Loading Model V1: {v1_ckpt}...")
    model_v1 = load_model(v1_ckpt, device)
    print(f"Loading Candidate FT-B: {cand_ckpt}...")
    model_cand = load_model(cand_ckpt, device)

    manifests = {
        "TEST_FINAL": Path("datasets/VNLP_processed/test.csv"),
        "TEST_CLEAN": Path("datasets/VNLP_processed/test_clean.csv"),
        "TEST_UNSEEN_TEXT": Path("datasets/VNLP_processed/test_unseen_text.csv"),
    }

    v1_metrics = {}
    cand_metrics = {}
    v1_records = {}
    cand_records = {}

    for name, path in manifests.items():
        print(f"\n[EVALUATING {name}] ({path.name})...")
        m1, r1 = evaluate_manifest(model_v1, str(path), device=device, batch_size=64, num_workers=4)
        m2, r2 = evaluate_manifest(model_cand, str(path), device=device, batch_size=64, num_workers=4)
        v1_metrics[name] = m1
        cand_metrics[name] = m2
        v1_records[name] = r1
        cand_records[name] = r2

        print(f"  V1:        Exact={m1['exact_accuracy']:.2f}% | CharAcc={m1['char_accuracy']:.2f}% | CER={m1['cer']:.4f} | AvgED={m1['avg_edit_distance']:.4f}")
        print(f"  Candidate: Exact={m2['exact_accuracy']:.2f}% | CharAcc={m2['char_accuracy']:.2f}% | CER={m2['cer']:.4f} | AvgED={m2['avg_edit_distance']:.4f}")
        print(f"  Delta:     {delta_str(m2['exact_accuracy'], m1['exact_accuracy'])}")

    # 1. Error Analysis: Fixed Cases & Regressions on TEST_FINAL
    print("\nExtracting Fixed Cases and Regressions on TEST_FINAL...")
    r1_test = v1_records["TEST_FINAL"]
    r2_test = cand_records["TEST_FINAL"]

    fixed_cases = []
    regressions = []

    for item1, item2 in zip(r1_test, r2_test):
        assert item1["image_path"] == item2["image_path"], "Image mismatch!"
        img_name = Path(item1["image_path"]).name
        gt = item1["ground_truth"]
        p1 = item1["prediction"]
        p2 = item2["prediction"]
        lay = item1["layout"]

        if not item1["exact"] and item2["exact"]:
            fixed_cases.append({
                "image": img_name,
                "ground_truth": gt,
                "v1_prediction": p1,
                "candidate_prediction": p2,
                "layout": lay,
                "source_test_set": "TEST_FINAL",
            })
        elif item1["exact"] and not item2["exact"]:
            regressions.append({
                "image": img_name,
                "ground_truth": gt,
                "v1_prediction": p1,
                "candidate_prediction": p2,
                "layout": lay,
                "source_test_set": "TEST_FINAL",
            })

    print(f"Fixed cases (V1 wrong -> Candidate right): {len(fixed_cases)}")
    print(f"Regressions (V1 right -> Candidate wrong): {len(regressions)}")
    print(f"Net Gain on TEST_FINAL: {len(fixed_cases) - len(regressions):+d} plates")

    # Write Fixed Cases CSV
    fixed_csv = Path("reports/phase3_fixed_cases.csv")
    with open(fixed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "v1_prediction", "candidate_prediction", "layout", "source_test_set"])
        writer.writeheader()
        writer.writerows(fixed_cases)
    print(f"Saved {fixed_csv}")

    # Write Regressions CSV
    regress_csv = Path("reports/phase3_regressions.csv")
    with open(regress_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "v1_prediction", "candidate_prediction", "layout", "source_test_set"])
        writer.writeheader()
        writer.writerows(regressions)
    print(f"Saved {regress_csv}")

    # 2. Confusion Analysis
    print("\nAnalyzing Character Confusions...")
    conf_v1 = build_confusion_matrix(r1_test)
    conf_cand = build_confusion_matrix(r2_test)

    tracked_pairs = [
        "0 -> 6", "6 -> 0", "7 -> 1", "E -> F", "3 -> 0", "2 -> 7",
        "8 -> 9", "X -> Y", "8 -> 0", "9 -> 3",
        "8 -> B", "B -> 8", "0 -> D", "D -> 0", "5 -> S", "S -> 5", "2 -> Z", "Z -> 2"
    ]

    conf_md = [
        "# SO SÁNH MA TRẬN NHẦM LẪN KÝ TỰ: MODEL V1 VS CANDIDATE FT-B (PHASE 3)",
        "",
        "**Thời gian thực hiện:** 2026-09-24",
        "**Tập kiểm định:** `TEST_FINAL` (3,733 mẫu)",
        f"**Candidate:** FT-B (`weights/ocr/experiments/phase3_ft_b/best.pth`)",
        "",
        "---",
        "",
        "## 1. Biến Động Các Cặp Ký Tự Trọng Yếu",
        "| Cặp nhầm lẫn (GT -> Pred) | Số lần ở Model V1 | Số lần ở Candidate FT-B | Thay đổi (FT-B vs V1) |",
        "| :---: | :---: | :---: | :---: |",
    ]

    for p in tracked_pairs:
        c1 = conf_v1.get(p, 0)
        c2 = conf_cand.get(p, 0)
        diff = c2 - c1
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        trend = "GIẢM LỖI (TỐT)" if diff < 0 else ("TĂNG LỖI" if diff > 0 else "GIỮ NGUYÊN")
        conf_md.append(f"| `{p}` | {c1} | {c2} | **{diff_str}** ({trend}) |")

    conf_md.extend([
        "",
        "---",
        "",
        "## 2. Top 10 Lỗi Phổ Biến Nhất ở Model V1",
        "| Hạng | Cặp ký tự (GT -> Pred) | Tần suất |",
        "| :---: | :---: | :---: |",
    ])
    for idx, (p, c) in enumerate(conf_v1.most_common(10), 1):
        conf_md.append(f"| {idx} | `{p}` | {c} |")

    conf_md.extend([
        "",
        "---",
        "",
        "## 3. Top 10 Lỗi Phổ Biến Nhất ở Candidate FT-B",
        "| Hạng | Cặp ký tự (GT -> Pred) | Tần suất |",
        "| :---: | :---: | :---: |",
    ])
    for idx, (p, c) in enumerate(conf_cand.most_common(10), 1):
        conf_md.append(f"| {idx} | `{p}` | {c} |")

    conf_file = Path("reports/phase3_confusion_comparison.md")
    with open(conf_file, "w", encoding="utf-8") as f:
        f.write("\n".join(conf_md))
    print(f"Saved {conf_file}")

    # 3. Main Comparison Report (reports/phase3_v1_vs_candidate.md)
    m1_tf = v1_metrics["TEST_FINAL"]
    m2_tf = cand_metrics["TEST_FINAL"]
    m1_tc = v1_metrics["TEST_CLEAN"]
    m2_tc = cand_metrics["TEST_CLEAN"]
    m1_un = v1_metrics["TEST_UNSEEN_TEXT"]
    m2_un = cand_metrics["TEST_UNSEEN_TEXT"]

    lines = [
        "# BÁO CÁO ĐỐI CHỨNG PYTORCH TOÀN DIỆN: MODEL V1 VS CANDIDATE FT-B (PHASE 3)",
        "",
        "**Thời gian kiểm định:** 2026-09-24",
        "**Framework:** PyTorch (CUDA thuần, không qua ONNX/OpenCV)",
        "**Mô hình V1 (Baseline):** `weights/ocr/best_lprnet_vn.pth`",
        "**Mô hình Candidate (FT-B):** `weights/ocr/experiments/phase3_ft_b/best.pth`",
        "",
        "---",
        "",
        "## 1. Bảng So Sánh Hiệu Năng Cốt Lõi Trên Các Tập Test",
        "| Chỉ số Đánh giá | Model V1 (Baseline) | Candidate FT-B | Delta (Candidate - V1) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **TEST_FINAL Exact Acc (3,733)** | {m1_tf['exact_accuracy']:.2f}% | {m2_tf['exact_accuracy']:.2f}% | **{delta_str(m2_tf['exact_accuracy'], m1_tf['exact_accuracy'])}** |",
        f"| **TEST_CLEAN Exact Acc (3,727)** | {m1_tc['exact_accuracy']:.2f}% | {m2_tc['exact_accuracy']:.2f}% | **{delta_str(m2_tc['exact_accuracy'], m1_tc['exact_accuracy'])}** |",
        f"| **TEST_UNSEEN_TEXT Exact Acc (1,187)** | {m1_un['exact_accuracy']:.2f}% | {m2_un['exact_accuracy']:.2f}% | **{delta_str(m2_un['exact_accuracy'], m1_un['exact_accuracy'])}** |",
        "| | | | |",
        f"| — Biển 1 dòng (`one_row`) | {m1_tf['layout_breakdown']['one_row']['exact_accuracy']:.2f}% | {m2_tf['layout_breakdown']['one_row']['exact_accuracy']:.2f}% | **{delta_str(m2_tf['layout_breakdown']['one_row']['exact_accuracy'], m1_tf['layout_breakdown']['one_row']['exact_accuracy'])}** |",
        f"| — Biển 2 dòng ô tô (`two_rows`) | {m1_tf['layout_breakdown']['two_rows']['exact_accuracy']:.2f}% | {m2_tf['layout_breakdown']['two_rows']['exact_accuracy']:.2f}% | **{delta_str(m2_tf['layout_breakdown']['two_rows']['exact_accuracy'], m1_tf['layout_breakdown']['two_rows']['exact_accuracy'])}** |",
        f"| — Biển 2 dòng xe máy (`motorcycle`) | {m1_tf['layout_breakdown']['two_rows_label_xe_may']['exact_accuracy']:.2f}% | {m2_tf['layout_breakdown']['two_rows_label_xe_may']['exact_accuracy']:.2f}% | **{delta_str(m2_tf['layout_breakdown']['two_rows_label_xe_may']['exact_accuracy'], m1_tf['layout_breakdown']['two_rows_label_xe_may']['exact_accuracy'])}** |",
        "| | | | |",
        f"| **Character Accuracy (TEST_FINAL)** | {m1_tf['char_accuracy']:.2f}% | {m2_tf['char_accuracy']:.2f}% | **{delta_str(m2_tf['char_accuracy'], m1_tf['char_accuracy'])}** |",
        f"| **Character Error Rate (CER)** | {m1_tf['cer']:.4f} | {m2_tf['cer']:.4f} | **{m2_tf['cer'] - m1_tf['cer']:+.4f}** |",
        f"| **Average Edit Distance (TEST_FINAL)** | {m1_tf['avg_edit_distance']:.4f} | {m2_tf['avg_edit_distance']:.4f} | **{m2_tf['avg_edit_distance'] - m1_tf['avg_edit_distance']:+.4f}** |",
        f"| **Character Accuracy (UNSEEN_TEXT)** | {m1_un['char_accuracy']:.2f}% | {m2_un['char_accuracy']:.2f}% | **{delta_str(m2_un['char_accuracy'], m1_un['char_accuracy'])}** |",
        f"| **CER (UNSEEN_TEXT)** | {m1_un['cer']:.4f} | {m2_un['cer']:.4f} | **{m2_un['cer'] - m1_un['cer']:+.4f}** |",
        f"| **Average Edit Distance (UNSEEN_TEXT)**| {m1_un['avg_edit_distance']:.4f} | {m2_un['avg_edit_distance']:.4f} | **{m2_un['avg_edit_distance'] - m1_un['avg_edit_distance']:+.4f}** |",
        "",
        "---",
        "",
        "## 2. Phân Tích Sửa Lỗi và Suy Giảm (Error Analysis)",
        f"- **Số ca Model V1 sai nhưng Candidate đoán đúng (Fixed Cases):** **{len(fixed_cases)}** mẫu (chi tiết tại `reports/phase3_fixed_cases.csv`).",
        f"- **Số ca Model V1 đúng nhưng Candidate đoán sai (Regressions):** **{len(regressions)}** mẫu (chi tiết tại `reports/phase3_regressions.csv`).",
        f"- **Net Gain trên TEST_FINAL:** **`{len(fixed_cases) - len(regressions):+d}`** mẫu đúng tuyệt đối.",
        "",
        "---",
        "",
        "## 3. Đánh Giá Khả Năng Tổng Quát Hóa (Generalization)",
        f"- Trên tập biển số chưa từng xuất hiện trong tập Train (`TEST_UNSEEN_TEXT`):",
        f"  - Model V1: **{m1_un['exact_accuracy']:.2f}%**",
        f"  - Candidate FT-B: **{m2_un['exact_accuracy']:.2f}%**",
        f"  - Delta: **{delta_str(m2_un['exact_accuracy'], m1_un['exact_accuracy'])}**",
        "",
        "---",
        "",
        "## 4. Kết Luận Đánh Giá Phase 3",
        f"- Trạng thái so sánh tổng thể: **[{'BETTER THAN V1' if m2_tf['exact_accuracy'] >= m1_tf['exact_accuracy'] and m2_un['exact_accuracy'] >= m1_un['exact_accuracy'] else ('MIXED' if m2_tf['exact_accuracy'] >= m1_tf['exact_accuracy'] or m2_un['exact_accuracy'] >= m1_un['exact_accuracy'] else 'REGRESSION')}]**.",
        "- Candidate FT-B là checkpoint PyTorch duy nhất được lưu giữ tại `weights/ocr/experiments/phase3_ft_b/best.pth`.",
        "- Không export ONNX, không deploy production trong Phase 3.",
        ""
    ]

    comp_path = Path("reports/phase3_v1_vs_candidate.md")
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {comp_path} successfully!")


if __name__ == "__main__":
    main()
