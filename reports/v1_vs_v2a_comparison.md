# BÁO CÁO SO SÁNH ĐỐI CHỨNG MODEL V1 VS MODEL V2-A (V1 VS V2-A COMPARISON)

**Thời gian kiểm định:** 2026-09-24
**Runtime suy luận:** OpenCV DNN CPU thuần (DNN_BACKEND_OPENCV, DNN_TARGET_CPU)
**Phần cứng kiểm thử:** Intel Core i5-12500H (Host CPU)
**Model V1:** `best_lprnet_vn.onnx` (3.04 MB)
**Model V2-A:** `best_lprnet_v2a.onnx` (3.04 MB)

---

## 1. Bảng So Sánh Hiệu Năng Cốt Lõi
| Chỉ số Đánh giá | Model V1 (Baseline) | Model V2-A (PBL4 Added) | Delta (V2-A - V1) |
| :--- | :---: | :---: | :---: |
| **TEST_FINAL Exact Acc (3,733)** | 85.27% | 65.87% | **-19.39 pp** |
| **TEST_CLEAN Exact Acc (3,727)** | 85.32% | 65.90% | **-19.43 pp** |
| **TEST_UNSEEN_TEXT Exact Acc (1,187)** | 85.93% | 67.40% | **-18.53 pp** |
| | | | |
| — Biển 1 dòng (`one_row`) | 90.58% | 72.20% | **-18.38 pp** |
| — Biển 2 dòng ô tô (`two_rows`) | 81.47% | 63.66% | **-17.81 pp** |
| — Biển 2 dòng xe máy (`motorcycle`) | 75.71% | 49.29% | **-26.43 pp** |
| | | | |
| **Character Accuracy (TEST_FINAL)** | 97.15% | 93.94% | **-3.21 pp** |
| **Character Error Rate (CER)** | 0.0285 | 0.0606 | **+0.0321** |
| **Character Accuracy (UNSEEN_TEXT)** | 97.24% | 94.03% | **-3.21 pp** |
| **CER (UNSEEN_TEXT)** | 0.0276 | 0.0597 | **+0.0321** |
| | | | |
| **Dung lượng ONNX Model** | 3.04 MB | 3.04 MB | 0.00 MB |
| **Laptop CPU Latency trung bình** | 3.41 ms (293.7 FPS) | 3.33 ms (299.9 FPS) | -0.07 ms |
| **Raspberry Pi 5 OCR Latency** | ~12.40 ms (đo trước) | NOT MEASURED (Pi offline) | N/A |

---

## 2. Phân Tích Hiện Tượng Sửa Lỗi và Suy Giảm (Error Analysis)
- **Số ca Model V1 sai nhưng Model V2-A đoán đúng (Fixed Cases):** **82** mẫu (chi tiết tại `reports/v2a_fixed_cases.csv`).
- **Số ca Model V1 đúng nhưng Model V2-A đoán sai (Regressions):** **806** mẫu (chi tiết tại `reports/v2a_regressions.csv`).
- **Net Improvement trên TEST_FINAL:** `-724` mẫu đúng tuyệt đối.

---

## 3. Đánh Giá Khả Năng Tổng Quát Hóa (Generalization Analysis)
- Trên tập biển số chưa từng xuất hiện trong tập Train (`TEST_UNSEEN_TEXT`):
  - Model V1: **85.93%**
  - Model V2-A: **67.40%**
  - Delta Generalization: **-18.53 pp**

---

## 4. Phân Loại Trạng Thái Model V2-A
- Kết luận phân loại: **[REGRESSION]**.
- Quyết định triển khai: **KHÔNG đưa V2-A vào production**. Model V1 tiếp tục giữ vai trò active cho đến khi người dùng và ChatGPT phân tích toàn diện snapshot này.
