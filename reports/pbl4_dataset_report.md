# BÁO CÁO CHI TIẾT VỀ DATASET PBL4 (PBL4 DATASET REPORT)

**Nguồn:** `https://github.com/lephamcong/PBL4_LPRNet`  
**Thư mục quan tâm:** `Dataset/BiensoxeVietNam`  
**Vị trí lưu trữ:** `datasets_external/pbl4/PBL4_LPRNet/Dataset/BiensoxeVietNam/`  
**Thời gian kiểm tra:** 2026-09-24  

---

## 1. Tổng quan số lượng & Cấu trúc thư mục
- **Tổng số file ảnh:** **13,320** file (100% định dạng `.jpg`).
  - Thư mục `train/`: 12,320 file ảnh.
  - Thư mục `valid/`: 1,000 file ảnh.
- **Quy tắc đặt tên file (Naming Convention):**
  `[plate_text]_[sample_id]_[aug_id].jpg`
  - Ví dụ: `29A87180_1212_0.jpg`, `29A87180_1212_1.jpg`,..., `29A87180_1212_9.jpg`
  - `plate_text`: Chuỗi ký tự biển số xe (Ground Truth).
  - `sample_id`: Mã định danh mẫu ảnh gốc.
  - `aug_id`: Mã biến thể augmentation offline (từ `0` đến `9`).

---

## 2. Phân tích Bản chất Dữ liệu & Augmentation nội bộ
- **Số lượng biển số duy nhất (Unique Plate Texts):**
  - Trong tập `train`: 421 chuỗi biển số duy nhất.
  - Trong tập `valid`: 136 chuỗi biển số duy nhất.
- **Số lượng mẫu ảnh gốc thực tế (Unique Sample IDs):**
  - Trong tập `train`: 1,232 mẫu ảnh gốc. Mỗi mẫu được augment offline thành 10 biến thể (`_0` đến `_9`). Tổng: 1,232 x 10 = 12,320 ảnh.
  - Trong tập `valid`: 200 mẫu ảnh gốc. Mỗi mẫu được augment offline thành 5 biến thể (`_0, _1, _4, _7, _8`). Tổng: 200 x 5 = 1,000 ảnh.
- **Tổng số mẫu ảnh gốc thực tế của toàn bộ PBL4:** **1,432 ảnh gốc**.
- **Trùng lặp hash nội bộ (Internal Exact Duplicates):** **1,784 file** có SHA256 trùng khớp với một file khác trong chính dataset PBL4 (do một số phép augment không làm thay đổi pixel hoặc augment giống hệt nhau).

---

## 3. Độ phân giải & Tính toàn vẹn của file ảnh
- **File hỏng (Corrupted files):** **0 file** (Toàn bộ 13,320 file đều đọc thành công bằng OpenCV).
- **Độ phân giải hình ảnh:**
  - Min Width: 94 px
  - Max Width: 94 px
  - Median Width: 94.0 px
  - Min Height: 24 px
  - Max Height: 24 px
  - Median Height: 24.0 px
  - *Nhận xét quan trọng:* Toàn bộ ảnh trong PBL4 đã bị crop sát và resize cố định về **94 x 24** theo chuẩn LPRNet gốc của tác giả. Kích thước này nhỏ hơn kích thước input chuẩn hiện tại của project (`160 x 48`). Khi đưa vào pipeline của project, ảnh sẽ cần resize/letterbox lên `160 x 48`.

---

## 4. Phân tích Nhãn & Tính tương thích với `CHARS`
- **Bộ ký tự hiện tại của Project (`CHARS`):** `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` (36 ký tự).
- **Tổng số nhãn kiểm tra:** 13,320 nhãn.
- **Nhãn hợp lệ (Valid Labels):** **13,320 (100.0%)**.
- **Nhãn không hợp lệ (Invalid Labels):** **0 (0.0%)**.
- **Ký tự lạ ngoài CHARS:** Không có (`0`).
- **Phân bố độ dài chuỗi ký tự (Sequence Length Distribution):**
  - Độ dài 7 ký tự: 560 ảnh (4.20%)
  - Độ dài 8 ký tự: 12,475 ảnh (93.66%)
  - Độ dài 9 ký tự: 285 ảnh (2.14%)
- **Tần suất ký tự:** Toàn bộ thuộc 29 ký tự (10 chữ số + 19 chữ cái `A, B, C, D, E, F, G, H, K, L, M, N, P, S, T, V, X, Y, Z`). Các ký tự `J, O, Q, W, U, R, I` không xuất hiện trong PBL4.
- **File chi tiết xuất kèm:**
  - `reports/pbl4_label_analysis.csv`
  - `reports/pbl4_character_frequency.csv`
