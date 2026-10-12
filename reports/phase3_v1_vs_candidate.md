# BÁO CÁO ĐỐI CHỨNG PYTORCH TOÀN DIỆN: MODEL V1 VS CANDIDATE FT-B (PHASE 3)

**Thời gian kiểm định:** 2026-09-24
**Framework:** PyTorch (CUDA thuần, không qua ONNX/OpenCV)
**Mô hình V1 (Baseline):** `weights/ocr/best_lprnet_vn.pth`
**Mô hình Candidate (FT-B):** `weights/ocr/experiments/phase3_ft_b/best.pth`

---

## 1. Bảng So Sánh Hiệu Năng Cốt Lõi Trên Các Tập Test
| Chỉ số Đánh giá | Model V1 (Baseline) | Candidate FT-B | Delta (Candidate - V1) |
| :--- | :---: | :---: | :---: |
| **TEST_FINAL Exact Acc (3,733)** | 85.27% | 85.75% | **+0.48 pp** |
| **TEST_CLEAN Exact Acc (3,727)** | 85.32% | 85.81% | **+0.48 pp** |
| **TEST_UNSEEN_TEXT Exact Acc (1,187)** | 85.93% | 85.51% | **-0.42 pp** |
| | | | |
| — Biển 1 dòng (`one_row`) | 90.58% | 90.68% | **+0.10 pp** |
| — Biển 2 dòng ô tô (`two_rows`) | 81.47% | 81.16% | **-0.32 pp** |
| — Biển 2 dòng xe máy (`motorcycle`) | 75.71% | 79.29% | **+3.57 pp** |
| | | | |
| **Character Accuracy (TEST_FINAL)** | 97.15% | 97.36% | **+0.21 pp** |
| **Character Error Rate (CER)** | 0.0285 | 0.0264 | **-0.0021** |
| **Average Edit Distance (TEST_FINAL)** | 0.2253 | 0.2076 | **-0.0177** |
| **Character Accuracy (UNSEEN_TEXT)** | 97.24% | 97.36% | **+0.13 pp** |
| **CER (UNSEEN_TEXT)** | 0.0276 | 0.0264 | **-0.0013** |
| **Average Edit Distance (UNSEEN_TEXT)**| 0.2216 | 0.2098 | **-0.0118** |

---

## 2. Phân Tích Sửa Lỗi và Suy Giảm (Error Analysis)
- **Số ca Model V1 sai nhưng Candidate đoán đúng (Fixed Cases):** **93** mẫu (chi tiết tại `reports/phase3_fixed_cases.csv`).
- **Số ca Model V1 đúng nhưng Candidate đoán sai (Regressions):** **75** mẫu (chi tiết tại `reports/phase3_regressions.csv`).
- **Net Gain trên TEST_FINAL:** **`+18`** mẫu đúng tuyệt đối.

---

## 3. Đánh Giá Khả Năng Tổng Quát Hóa (Generalization)
- Trên tập biển số chưa từng xuất hiện trong tập Train (`TEST_UNSEEN_TEXT`):
  - Model V1: **85.93%**
  - Candidate FT-B: **85.51%**
  - Delta: **-0.42 pp**

---

## 4. Kết Luận Đánh Giá Phase 3
- Trạng thái so sánh tổng thể: **[MIXED]**.
- Candidate FT-B là checkpoint PyTorch duy nhất được lưu giữ tại `weights/ocr/experiments/phase3_ft_b/best.pth`.
- Không export ONNX, không deploy production trong Phase 3.
