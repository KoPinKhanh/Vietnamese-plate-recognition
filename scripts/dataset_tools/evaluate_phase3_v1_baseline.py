"""
Official Phase 3 Baseline Evaluator for V1 PyTorch Checkpoint.
Evaluates best_lprnet_vn.pth on:
- VAL (datasets/VNLP_processed/val.csv)
- TEST_FINAL (datasets/VNLP_processed/test.csv)
- TEST_CLEAN (datasets/VNLP_processed/test_clean.csv)
- TEST_UNSEEN_TEXT (datasets/VNLP_processed/test_unseen_text.csv)

Outputs:
- reports/phase3_v1_pytorch_baseline.md
"""

import sys
from pathlib import Path
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.charset import NUM_CLASSES
from ocr.models.lprnet_lite import LPRNetLite
from tools.evaluate_ocr_model import evaluate_manifest


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("weights/ocr/best_lprnet_vn.pth")
    print(f"Loading {ckpt_path} on {device}...")
    model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    manifests = {
        "VAL": Path("datasets/VNLP_processed/val.csv"),
        "TEST_FINAL": Path("datasets/VNLP_processed/test.csv"),
        "TEST_CLEAN": Path("datasets/VNLP_processed/test_clean.csv"),
        "TEST_UNSEEN_TEXT": Path("datasets/VNLP_processed/test_unseen_text.csv"),
    }

    results = {}
    for name, p in manifests.items():
        print(f"Evaluating {name} ({p})...")
        m, _ = evaluate_manifest(model, str(p), device=device, batch_size=64, num_workers=4)
        results[name] = m
        ex = m["exact_accuracy"]
        ca = m["char_accuracy"]
        c = m["cer"]
        ed = m["avg_edit_distance"]
        cor = m["correct"]
        tot = m["total_samples"]
        print(f"  -> Exact: {cor}/{tot} ({ex:.2f}%) | CharAcc: {ca:.2f}% | CER: {c:.4f} | AvgED: {ed:.4f}")

    lines = [
        "# BÁO CÁO BASELINE CHÍNH THỨC MODEL V1 BẰNG PYTORCH (PHASE 3)",
        "",
        "**Thời gian thực hiện:** 2026-09-24",
        "**Framework:** PyTorch (CUDA)",
        "**File Checkpoint V1:** `weights/ocr/best_lprnet_vn.pth`",
        "**SHA256:** `25d2ff3cba0869766a96689a8eda2d1b133634c86023df01e2e41cf17491305c`",
        "**Evaluator:** `tools/evaluate_ocr_model.py` (Chuẩn hóa duy nhất)",
        "",
        "---",
        "",
        "## 1. Kết Quả Tổng Thể trên Các Tập Đánh Giá (Official Phase 3 Baseline)",
        "| Tập dữ liệu | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name, m in results.items():
        lines.append(
            f"| **{name}** | {m['total_samples']} | {m['correct']}/{m['total_samples']} | "
            f"**{m['exact_accuracy']:.2f}%** | {m['char_accuracy']:.2f}% | {m['cer']:.4f} | {m['avg_edit_distance']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Phân Tích Chi Tiết Theo Kiểu Biển (Layout Breakdown)",
    ])

    for name, m in results.items():
        lines.extend([
            f"### Tập dữ liệu: {name}",
            "| Kiểu biển (Layout) | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for lay, st in m["layout_breakdown"].items():
            lay_name = "1 Dòng (one_row)" if lay == "one_row" else ("2 Dòng Ô tô (two_rows)" if lay == "two_rows" else "2 Dòng Xe Máy (motorcycle)")
            lines.append(
                f"| {lay_name} | {st['total']} | {st['correct']}/{st['total']} | "
                f"**{st['exact_accuracy']:.2f}%** | {st['char_accuracy']:.2f}% | {st['cer']:.4f} | {st['avg_edit_distance']:.4f} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Quy Định Bất Biến Cho Phase 3",
        "1. Bảng số liệu trên là **GROUND TRUTH BASELINE DUY NHẤT** được công nhận cho toàn bộ Phase 3.",
        "2. Mọi kết quả fine-tune (FT-A, FT-B, FT-C) đều phải được so sánh đối chiếu với các con số này trên cùng Evaluator PyTorch.",
        f"3. **Chỉ số chọn lọc mô hình:** `VAL Exact Accuracy` (Baseline V1: **{results['VAL']['exact_accuracy']:.2f}%**).",
        ""
    ])

    out_path = Path("reports/phase3_v1_pytorch_baseline.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    print(f"\nSuccessfully wrote {out_path}")


if __name__ == "__main__":
    main()
