# BÁO CÁO SO SÁNH TRÊN TẬP VALIDATION (PHASE 3 VALIDATION COMPARISON)

**Thời gian thực hiện:** 2026-09-24
**Mục tiêu:** So sánh khách quan hiệu năng giữa Model V1 gốc, FT-A và FT-B trên tập VALIDATION chuẩn (3,728 mẫu) để chọn đúng **MỘT CANDIDATE DUY NHẤT** trước khi mở tập TEST.

---

## 1. Bảng So Sánh Chỉ Số Trên Tập Validation Chuẩn (3,728 mẫu)
| Chỉ số Đánh giá | Model V1 (Baseline) | FT-A (LR=1e-4, Ep 1) | FT-B (LR=5e-5, Ep 6) | Delta FT-B vs V1 | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VAL Exact Accuracy** | **84.79%** | 85.60% | **85.76%** | **+0.97 pp** | **VƯỢT TRỘI** |
| **VAL Character Accuracy** | 97.18% | 97.41% | **97.43%** | **+0.24 pp** | **CẢI THIỆN** |
| **VAL CER** | 0.0282 | 0.0259 | **0.0257** | **-0.0024** | **TỐT NHẤT** |
| **Average Edit Distance** | 0.2234 | 0.2047 | **0.2033** | **-0.0201** | **CẢI THIỆN** |
| | | | | | |
| — Biển 1 dòng (`one_row`) | 89.99% | 90.04% | **90.04%** | **+0.05 pp** | **DUY TRÌ VỮNG** |
| — Biển 2 dòng ô tô (`two_rows`) | 80.73% | 80.97% | **81.28%** | **+0.56 pp** | **CẢI THIỆN** |
| — Biển 2 dòng xe máy (`motorcycle`)| 76.21% | 80.86% | **81.22%** | **+5.01 pp** | **ĐỘT PHÁ (+5.0 pp)** |
| | | | | | |
| **VAL Unseen Text (1,248 mẫu)** | 83.89% | 84.70% | **85.18%** | **+1.28 pp** | **TỔNG QUÁT HÓA TĂNG** |

---

## 2. Phân Tích Hiện Tượng và Quyết Định Chọn Candidate
1. **Phân tích FT-A vs FT-B:**
   - Cả FT-A (LR=1e-4) và FT-B (LR=5e-5) đều giải quyết hoàn toàn vấn đề suy giảm trước đó của V2-A nhờ cơ chế **khởi tạo từ checkpoint V1** (Transfer Learning) thay vì train from scratch.
   - FT-B với tốc độ học nhỏ hơn (5e-5) cho phép các trọng số thích ứng mượt mà hơn, tránh hiện tượng drift nhanh, đạt đỉnh phong độ ở Epoch 6 với độ chính xác VAL lên tới **85.76%** (so với 85.60% của FT-A và 84.79% của V1).
   - Đặc biệt, tỷ lệ chính xác trên biển 2 dòng xe máy (`motorcycle`) tăng vọt từ **76.21% lên 81.22% (+5.01 pp)**.
   - Trên tập kiểm định độc lập `val_unseen_text_v2.csv`, FT-B đạt **85.18% (so với V1 83.89%)**.

2. **Lựa chọn Candidate Duy Nhất:**
   - **CANDIDATE ĐƯỢC CHỌN:** **`FT-B`** (`weights/ocr/experiments/phase3_ft_b/best.pth`).
   - **Lý do lựa chọn:** Đạt điểm số cao nhất trên mọi tiêu chí của tập Validation (Exact Acc: 85.76%, CER: 0.0257, Motorcycle: 81.22%, Unseen Val: 85.18%) mà không có bất kỳ dấu hiệu catastrophic forgetting nào.
   - Đủ điều kiện để tiến hành bước đánh giá cuối cùng trên các tập TEST.
