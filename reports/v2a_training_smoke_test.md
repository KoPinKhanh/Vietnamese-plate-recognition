# BÁO CÁO THỬ NGHIỆM SMOKE TEST PIPELINE HUẤN LUYỆN V2-A (SMOKE TEST SUMMARY)

**Thời gian thực hiện:** 2026-09-24
**Thiết bị tính toán:** cuda (NVIDIA GeForce RTX 3050 Laptop GPU)
**Cấu hình thử nghiệm:** `configs/ocr_lprnet_v2a.yaml`

---

## 1. Kết quả Kiểm tra Từng Thành phần
| Hạng mục kiểm tra | Tiêu chuẩn đánh giá | Kết quả thực tế | Trạng thái |
| :--- | :--- | :--- | :---: |
| **Khởi tạo DataLoader** | Đọc đủ 31,047 mẫu `train_v2a.csv` | **31,047 mẫu / 486 batches** | **PASS** |
| **Kích thước Input Batch** | `[64, 3, 48, 160]`, RGB `[-1, 1]` | `[64, 3, 48, 160]`, min=-1.00, max=1.00 | **PASS** |
| **Độ dài nhãn vs Time-steps** | `target_length <= 40` | Min=7, Max=9 <= 40 | **PASS** |
| **Forward Pass (LPRNetLite)**| Logits shape `[64, 37, 40]` | `[64, 37, 40]`, 0 NaN | **PASS** |
| **CTC Loss Tính toán** | Không NaN, không Inf, > 0.0 | **Loss = 15.0095** (hợp lệ) | **PASS** |
| **Backward Pass & Gradients**| Đạo hàm chuẩn, không triệt tiêu | Gradients tính đủ 100% tham số | **PASS** |
| **Optimizer Step** | Cập nhật trọng số ổn định | Hoàn thành không ngoại lệ | **PASS** |
| **Mini-overfit (32 mẫu)** | Giảm tổn thất > 50% trong 40 steps | **Loss giảm từ 14.2859 về 0.8351** | **PASS** |

---

## 2. Kết luận
- Toàn bộ quy trình nạp dữ liệu kết hợp VNLP + PBL4 Unique, tiền xử lý, tính đạo hàm CTC và cập nhật trọng số hoạt động hoàn hảo.
- **Hệ thống đủ điều kiện 100% để kích hoạt Full Training V2-A.**
