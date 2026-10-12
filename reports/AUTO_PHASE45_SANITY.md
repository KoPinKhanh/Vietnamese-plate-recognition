# BÁO CÁO XÁC MINH CHUẨN HÓA THÔNG SỐ (PHASE 4.5 SANITY LOCK)

**Thời gian thực hiện:** 2026-09-25  
**Checkpoint kiểm tra:** `weights/ocr/experiments/phase4/control_seed42/best.pth`  
**Mục tiêu:** Rà soát và khóa cứng toàn bộ siêu dữ liệu kiến trúc, bảng ký tự và tensor input/output trực tiếp từ mã nguồn thực tế và checkpoint PyTorch trước khi bắt đầu Phase 5.

---

## 1. Kết Quả Kiểm Tra Trực Tiếp Từ Source Code & Checkpoint

| Thông số / Thuộc tính | Giá trị ghi nhận từ mã nguồn & Checkpoint thực tế | Trạng thái xác minh |
| :--- | :--- | :---: |
| **Strict Model Load** | `model.load_state_dict(state_dict, strict=True)` thành công 100% | **PASS** |
| **Bảng ký tự (CHARSET)** | `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` (10 chữ số + 26 chữ cái tiếng Anh in hoa) | **PASS** |
| **Số ký tự từ vựng (NUM_CHARS)** | **36** ký tự | **PASS** |
| **Chỉ số CTC Blank (BLANK_INDEX)** | **36** (0-indexed, ký tự thứ 37) | **PASS** |
| **Tổng số lớp đầu ra (NUM_CLASSES)** | **37** (36 alphanumeric + 1 CTC blank) | **PASS** |
| **Số lượng tham số mô hình (Parameters)** | **798,565** tham số | **PASS** |
| **Kích thước đầu vào (Input Tensor)** | `[B, 3, 48, 160]` $(B \times C \times H \times W)$, RGB, chuẩn hóa $[-1.0, 1.0]$ | **PASS** |
| **Kích thước đầu ra (Output Tensor)** | `[B, 37, 40]` $(B \times \text{classes} \times \text{time steps})$ | **PASS** |
| **Thời gian downsampling** | Chiều rộng $W=160 \rightarrow 40$ time steps (downsample factor 4) | **PASS** |
| **Best Accuracy ghi trong Checkpoint** | **86.02%** (3,207 / 3,728 mẫu trên tập Validation chuẩn) | **PASS** |
| **Best Epoch ghi trong Checkpoint** | **Epoch 3** | **PASS** |

---

## 2. Làm Rõ Sai Lệch Trong Các Báo Cáo Cũ (Discrepancy Resolution)

1. **Về bảng ký tự (`CHARSET`):**
   - *Báo cáo cũ ghi chú:* `0123456789ABCDEFGHKLMNPRSTUVXYZ-` (bỏ `I`, `J`, `O`, `Q`, `W` và có dấu `-`).
   - *Thực tế mã nguồn (`ocr/charset.py`):* Bảng từ vựng thực tế sử dụng chuẩn đầy đủ 36 ký tự: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` (không chứa dấu `-`, đầy đủ `A-Z`). Việc bỏ các ký tự không dùng của biển số Việt Nam thực chất được xử lý thông qua chuẩn hóa văn bản trước khi encode, nhưng output layer của mạng LPRNetLite hỗ trợ toàn bộ 36 ký tự tiếng Anh + 1 CTC blank.
   - *Kết luận:* Khóa bảng ký tự 36 ký tự chuẩn `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` cho toàn bộ các phase tiếp theo.

2. **Về số lượng tham số:**
   - *Báo cáo cũ ghi:* ~805,093 tham số.
   - *Thực tế mã nguồn:* Chính xác là **798,565 tham số** (ước tính 0.80M tham số).

---

## 3. Quyết Định Sanity Gate

- **Tình trạng Checkpoint Base:** Hoàn toàn hợp lệ, load strict 100%, không bị corrupt, không có mismatch giữa state dict và model definition.
- **Quyết định:** **SANITY PASS — CHO PHÉP BẮT ĐẦU PHASE 5.**
