# BÁO CÁO TOÀN DIỆN VỀ DATASET VÀ OCR HIỆN TẠI (CURRENT DATASET REPORT)

**Thời gian audit:** 2026-09-24  
**Project Root:** `D:\Vietnam_Plate_Project`  
**Trạng thái:** Audit hoàn chỉnh & Khóa `TEST_FINAL`

---

## 1. Vị trí thực tế của Dataset OCR
- **Thư mục dataset ảnh gốc:** `d:/Vietnam_Plate_Project/datasets/VNLP_detection/detection/`
  - Thư mục con `one_row/`: 19,086 file ảnh (.jpg)
  - Thư mục con `two_rows/`: 12,618 file ảnh (.jpg)
  - Thư mục con `two_rows_label_xe_may/`: 5,593 file ảnh (.jpg)
  - File tài liệu nguồn: `VNLP_readme`
- **File nén lưu trữ gốc:** `datasets/VNLP_detection.zip` (3,702,468,346 bytes ~ 3.45 GB)
- **Thư mục repository VNLP đi kèm:** `datasets/VNLP-main/VNLP-main/` (chứa `readme.md` và mã bài báo MAPR2022 / The Visual Computer 2023 của tác giả Phạm Thế Anh - Đại học Hồng Đức)
- **Thư mục Manifests đã chuẩn hóa:** `d:/Vietnam_Plate_Project/datasets/VNLP_processed/`
  - `all.csv`: 37,297 dòng dữ liệu
  - `train.csv`: 29,836 dòng (80.0%)
  - `val.csv`: 3,728 dòng (10.0%)
  - `test.csv`: 3,733 dòng (10.0%)  <--- **ĐÂY LÀ TEST_FINAL**
  - `test_unseen_text.csv`: 1,187 dòng (31.8% tập test có chuỗi biển số chưa từng xuất hiện trong tập train)

---

## 2. Xác minh nguồn gốc VNLP (Bằng chứng cụ thể)
Đã xác minh 100% dataset OCR hiện tại chính là **VNLP (Vietnamese License Plate Dataset)** qua các bằng chứng:
1. **File `datasets/VNLP_detection/detection/VNLP_readme`:**
   > *"The VNLP dataset is composed of around 37,300 images. Most of the images have been taken at a Transportation Registration Center in Vietnam. The dataset is divided into two collections with respect to the LP shape type: one-row subset (19,086 images) and two-row subset (18,211 images)."*  
   > Contact: `phamtheanh@hdu.edu.vn`.
2. **File `datasets/VNLP-main/VNLP-main/readme.md`:**
   > Cung cấp thông tin tác giả: *Thi-Anh-Loan Trinh, The Anh Pham, and Van-Dung Hoang (2022). "Layout-invariant license plate detection and recognition", MAPR2022* và *The Visual Computer 2023*.
3. **Cấu trúc đặt tên file ảnh và số lượng:**
   - Định dạng chuẩn của VNLP: `[idx]_[id]_[layout_code]_[plate_text]_[x1]_[y1]_[x2]_[y2].jpg`
   - Số lượng ảnh khớp 100%: 37,297 ảnh.

---

## 3. Quy cách Annotation và Ground-Truth
- **Nguồn nhãn:** Parse trực tiếp từ tên file ảnh thông qua hàm `parse_vnlp_filename()` trong `tools/prepare_vnlp.py`.
- **Ví dụ giải mã tên file:** `7_4986_1_54S9449_183_152_520_237.jpg`
  - `parts[2] = 1`: layout là `one_row` (nếu khác 1 là `two_rows`)
  - `parts[3:-4] = 54S9449`: nhãn biển số xe (Ground Truth)
  - `x1 = 183, y1 = 152`: tọa độ góc trên-trái bounding box
  - `x2 = 520, y2 = 237`: tọa độ góc dưới-phải bounding box
- **Định dạng manifest CSV:**
  `image_path, relative_path, plate_text, layout, subset, x1, y1, x2, y2`

---

## 4. Phân chia Train / Val / Test & Đánh giá Rủi ro Leakage
- **Tỷ lệ phân chia:** 80% Train / 10% Val / 10% Test theo phương pháp phân tầng (stratified) theo từng subset (`one_row`, `two_rows`, `two_rows_label_xe_may`) với random seed = 42.
- **Số lượng mẫu:**
  - **TRAIN:** 29,836 mẫu (15,268 one_row, 10,094 two_rows, 4,474 xe máy)
  - **VAL:** 3,728 mẫu (1,908 one_row, 1,261 two_rows, 559 xe máy)
  - **TEST (`TEST_FINAL`):** 3,733 mẫu (1,910 one_row, 1,263 two_rows, 560 xe máy)
- **Khảo sát trùng lặp hình ảnh (Image Hash Leakage):**
  - Trùng hash ảnh tuyệt đối giữa Train và Val: 4 ảnh (0.11%)
  - Trùng hash ảnh tuyệt đối giữa Train và Test: 6 ảnh (0.16%)
- **Khảo sát trùng lặp nội dung biển (Text Overlap):**
  - Do VNLP chụp nhiều góc/frame của cùng một xe tại trung tâm đăng kiểm, có 2,546 / 3,733 mẫu của tập Test (68.20%) mang chuỗi biển số đã từng xuất hiện ở ảnh khác trong tập Train.
  - Nhóm phát triển đã tạo riêng tập `test_unseen_text.csv` gồm 1,187 mẫu (31.80% tập test) chứa các biển số xe hoàn toàn mới (chưa từng thấy trong tập train) để đánh giá khả năng tổng quát hóa thực sự.
- **Rủi ro Augmentation:** Không có rủi ro leakage do augmentation offline; toàn bộ ảnh trong các tập train/val/test đều là ảnh gốc độc lập.

---

## 5. Input, Model và Character Set (CHARS)
- **Input Model OCR:**
  - Kích thước: `1 x 3 x 48 x 160` (Batch, Channels, Height, Width)
  - Định dạng màu: RGB
  - Chuẩn hóa: `[-1.0, 1.0]` thông qua công thức `(pixel / 127.5) - 1.0` (tương đương `cv2.dnn.blobFromImage(..., scalefactor=1/127.5, mean=(127.5, 127.5, 127.5), swapRB=True)`)
- **Kiến trúc mạng:** `LPRNetLite`
  - Fully convolutional với 3 stages (32 -> 64 -> 128 -> 256 channels)
  - Multi-scale feature fusion (Adapter 1 & 2 pooling và nối với stage 3 thành 384 channels)
  - Spatial collapse: chiều cao thu gọn về 1, chiều rộng thu nhỏ 4 lần (160 -> 40 time steps)
  - Đầu ra: Tensor shape `[B, 37, 40]`
  - Dung lượng tham số: ~0.80M tham số (ONNX size: 3.04 MB)
- **Character Set (CHARS):**
  - Bảng ký tự chuẩn: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` (36 ký tự chữ hoa và số)
  - CTC Blank index: 36 (vị trí cuối cùng)
  - Tổng số lớp phân loại (`NUM_CLASSES`): 37
  - Phân tích tập VNLP: Có 32 ký tự xuất hiện thực tế (`0123456789ABCDEFGHIKLMNPRSTUVXYZ`). Các ký tự không xuất hiện trong VNLP là `J`, `O`, `Q`, `W`, và ký tự `I` chỉ xuất hiện 1 lần. Tuy nhiên model và decoder đã hỗ trợ đầy đủ 36 ký tự alphanumeric.
- **Cơ chế tiền xử lý bố cục biển số:**
  - **Biển 1 dòng (`one_row`):** Aspect ratio >= 2.0. Letterbox đệm viền trắng (255, 255, 255) về kích thước chuẩn (160, 48).
  - **Biển 2 dòng (`two_rows`) & Xe máy (`two_rows_label_xe_may`):** Aspect ratio < 2.0. Phân tách dòng trên và dòng dưới bằng thuật toán tìm thung lũng đạo hàm Sobel đứng trong khoảng [0.35H, 0.60H]. Resize cả 2 dòng về chiều cao 48 px, ghép ngang dòng trên và dòng dưới với dải ngăn cách 6 px trắng, sau đó letterbox về (160, 48).

---

## 6. KHÓA TẬP TEST HIỆN TẠI (LOCK TEST_FINAL)
- **Tập test chính thức:** `datasets/VNLP_processed/test.csv` (3,733 mẫu).
- **Quy tắc bất biến:**
  1. Tuyệt đối không xáo trộn, chia lại (re-split) tập test này.
  2. Tuyệt đối không bổ sung dữ liệu mới từ PBL4 hay Synthetic vào tập test này.
  3. Mọi dữ liệu mới từ PBL4 và Generator nếu bị trùng với bất kỳ mẫu nào trong tập test này sẽ bị đưa vào `QUARANTINE_DUPLICATES` để tránh data leakage.
