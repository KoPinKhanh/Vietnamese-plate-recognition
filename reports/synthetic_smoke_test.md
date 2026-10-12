# BÁO CÁO THỬ NGHIỆM SMOKE TEST DỮ LIỆU TỔNG HỢP (SYNTHETIC SMOKE TEST)

**Thời gian thực hiện:** 2026-09-24  
**Script thực hiện:** `scripts/dataset_tools/generate_lprnet_vn.py`  
**Thư mục lưu trữ:**
- Ảnh: `dataset_staging/synthetic/images/`
- Manifest: `dataset_staging/synthetic/labels.csv`

---

## 1. Mục tiêu & Giới hạn thử nghiệm
- **Giới hạn số lượng:** Chỉ tạo đúng **100 ảnh** (không tạo số lượng lớn gây tốn tài nguyên và chưa qua kiểm định).
- **Mục tiêu:** Kiểm tra toàn diện adapter sinh dữ liệu biển số xe tổng hợp phục vụ huấn luyện và kiểm định LPRNet-Lite:
  - Khả năng tải assets (font `MyFont-Regular_ver3.otf`, các ảnh nền `rec*.jpg`, `square*.jpg`).
  - Khả năng kết xuất hình ảnh và tính tương thích với phiên bản Pillow 12+.
  - Tính hợp lệ của chuỗi ký tự Ground Truth so với bảng mã `CHARS`.
  - Khả năng sinh cả hai bố cục biển số (`one_row` và `two_rows`).
  - Bổ sung các ký tự hiếm vốn bị thiếu trong tập dữ liệu VNLP (`J, O, Q, W`).

---

## 2. Kết quả Đo kiểm Smoke Test (100 Mẫu)

| Tiêu chí Kiểm định | Chỉ tiêu Yêu cầu | Kết quả Đo đạc Thực tế | Đánh giá |
| :--- | :--- | :--- | :--- |
| **Generation Success** | 100 ảnh được tạo | **100 / 100 ảnh** | **ĐẠT (PASS)** |
| **Image Readability** | Đọc được bằng OpenCV | **100 / 100 ảnh readable (0 lỗi)** | **ĐẠT (PASS)** |
| **Label vs Image Match** | Khớp định dạng CSV | **100 / 100 mẫu khớp nhãn** | **ĐẠT (PASS)** |
| **CHARS Compatibility** | Ký tự thuộc [A-Z0-9] | **100 / 100 nhãn (0 ký tự lạ)** | **ĐẠT (PASS)** |
| **Layout Bố cục** | Hỗ trợ cả 1 dòng & 2 dòng | **46 one_row, 54 two_rows** | **ĐẠT (PASS)** |
| **Format Manifest CSV** | `filename,label,layout,source` | Đúng 4 cột chuẩn | **ĐẠT (PASS)** |
| **Rare Characters Injected**| Bổ sung J, O, Q, W | **Có mặt: J, O, Q, W** | **ĐẠT (PASS)** |
| **No Train Merge** | Không trộn vào tập train | **Lưu độc lập tại `dataset_staging/`**| **ĐẠT (PASS)** |

---

## 3. Chi tiết Cấu trúc Dữ liệu Sinh ra
- **Quy chuẩn đặt tên:** `00000001.jpg` đến `00000100.jpg`
- **Kích thước ảnh:**
  - Biển 1 dòng (`one_row`): 450 x 110 px (chuẩn tỷ lệ biển dài ~4.1:1).
  - Biển 2 dòng (`two_rows`): 320 x 240 px (chuẩn tỷ lệ biển vuông ~1.33:1).
- **Mẫu dữ liệu thực tế từ `labels.csv`:**
  ```csv
  filename,label,layout,source
  00000001.jpg,56H1234,two_rows,synthetic
  00000002.jpg,37T17748,two_rows,synthetic
  00000003.jpg,77D19876,one_row,synthetic
  00000004.jpg,29A88888,one_row,synthetic
  00000005.jpg,51F12345,one_row,synthetic
  ```

---

## 4. Đánh giá Khả năng Ứng dụng cho Training V2
- Adapter hoạt động mượt mà, sinh ảnh với viền, bóng chữ, hiệu ứng góc nghiêng và nhiễu nhẹ mô phỏng camera thực tế.
- Ground truth được chuẩn hóa loại bỏ hoàn toàn dấu gạch nối `-` và dấu chấm `.`, hoàn toàn sẵn sàng cho CTC Loss mà không cần viết thêm bộ parser trung gian.
- **Kết luận:** Adapter `scripts/dataset_tools/generate_lprnet_vn.py` đã sẵn sàng để có thể sinh hàng nghìn ảnh cho pha mở rộng dữ liệu (Dataset V2) khi được yêu cầu.
