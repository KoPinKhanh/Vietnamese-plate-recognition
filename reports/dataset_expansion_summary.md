# TỔNG KẾT MỞ RỘNG DỮ LIỆU & KIỂM ĐỊNH LEAKAGE (DATASET EXPANSION SUMMARY)

**Thời gian lập báo cáo:** 2026-09-24  
**Project Root:** `D:\Vietnam_Plate_Project`  

---

## 1. Tình trạng Dataset Hiện tại
- **Tên dataset hiện tại:** **VNLP (Vietnamese License Plate Dataset)**.
- **Đường dẫn lưu trữ:** `d:/Vietnam_Plate_Project/datasets/VNLP_detection/detection/` (kèm manifests tại `datasets/VNLP_processed/`).
- **Bằng chứng xác nhận VNLP:**
  1. File `datasets/VNLP_detection/detection/VNLP_readme` ghi rõ thông tin tác giả: `phamtheanh@hdu.edu.vn`.
  2. File `datasets/VNLP-main/VNLP-main/readme.md` chứa citation bài báo gốc MAPR2022 và The Visual Computer 2023.
  3. Cấu trúc tên file chứa tọa độ bbox `[x1]_[y1]_[x2]_[y2]` và nhãn ký tự với chính xác 37,297 file ảnh.
- **Số lượng mẫu phân chia Train / Val / Test:**
  - **TRAIN (`train.csv`):** 29,836 mẫu (80.0%)
  - **VAL (`val.csv`):** 3,728 mẫu (10.0%)
  - **TEST (`test.csv`):** 3,733 mẫu (10.0%)
- **Tập test chuẩn (`TEST_FINAL`):**
  - **Đường dẫn:** `datasets/VNLP_processed/test.csv` (3,733 mẫu).
  - **Trạng thái:** **ĐÃ KHÓA CỐ ĐỊNH (LOCKED)**. Tuyệt đối không thay đổi, không xáo trộn, không merge thêm dữ liệu vào tập test này.

---

## 2. Nguồn dữ liệu Mới 1: PBL4 (`lephamcong/PBL4_LPRNet`)
- **Tình trạng download:** Tải thành công qua Git Sparse Clone vào `datasets_external/pbl4/PBL4_LPRNet/Dataset/BiensoxeVietNam/`.
- **Số lượng ảnh ban đầu (PBL4 Initial Count):** **13,320 ảnh** (12,320 train + 1,000 valid). Toàn bộ ảnh đã được crop ở kích thước `94 x 24` px.
- **Phát hiện cấu trúc nội bộ:** Thực chất bộ dữ liệu chỉ có **1,432 ảnh gốc thực tế** (1,232 ảnh train và 200 ảnh valid) với 557 chuỗi biển số duy nhất. Tác giả PBL4 đã thực hiện offline augmentation thành 10 biến thể cho mỗi ảnh train và 5 biến thể cho mỗi ảnh valid.
- **Trùng lặp tuyệt đối nội bộ (Exact Duplicate):** **128 ảnh** trùng SHA256 chính xác.
- **Trùng lặp tương đồng (Near Duplicate):** **9,845 ảnh** có pHash Hamming distance <= 6 từ ảnh gốc (là các biến thể augment nhẹ của cùng một biển số).
- **Cờ xem xét (Review Flagged):** **2,136 ảnh** có pHash Hamming distance từ 7 đến 10 với cùng nhãn.
- **Nguy cơ rò rỉ tập test (Test Leakage Risk vs `TEST_FINAL`):** **0 (KHÔNG CÓ)**. Không có bất kỳ ảnh nào trong PBL4 trùng hash, pHash hoặc chuỗi ký tự với tập `TEST_FINAL`.
- **Dữ liệu thực sự mới, duy nhất có thể sử dụng (Unique Usable):** **1,211 ảnh** (đã được trích xuất độc lập và đưa vào `dataset_staging/pbl4_unique/`).
- **Tổng số ảnh bị cách ly (Total Quarantined):** **12,109 ảnh** (đã đưa vào `dataset_staging/quarantine_duplicates/`).
- **Tính tương thích bảng mã ký tự (CHARS Compatibility):** **100.0% tương thích** (Toàn bộ 13,320 nhãn đều thuộc bảng 36 ký tự [A-Z0-9]).

---

## 3. Nguồn dữ liệu Mới 2: Vietnamese License Plate Generator
- **Tình trạng download:** Tải thành công từ `https://github.com/NNDam/Vietnamese-License-Plate-Generator.git` vào `datasets_external/vn_plate_generator/Vietnamese-License-Plate-Generator/`.
- **Đánh giá Dependencies:** Yêu cầu thư viện gốc: `numpy`, `opencv-python==4.2.0.32`, `pillow`, `progressbar2`. Không cài đặt đè để tránh làm hỏng môi trường `.venv` hiện tại (OpenCV 5.0.0, Pillow 12.3.0).
- **Adapter được xây dựng:** `scripts/dataset_tools/generate_lprnet_vn.py` chuyển đổi định dạng sinh từ YOLO BBox sang chuỗi nhãn CTC LPRNet và tương thích với Pillow 12+.
- **Thử nghiệm Smoke Test (Synthetic Smoke Test):**
  - Đã sinh chính xác **100 ảnh** thử nghiệm vào `dataset_staging/synthetic/images/`.
  - Manifest chuẩn: `dataset_staging/synthetic/labels.csv` (100 dòng).
  - Tỷ lệ bố cục: 46 ảnh `one_row` (450x110) và 54 ảnh `two_rows` (320x240).
  - Kết quả kiểm định: **100/100 ảnh đọc tốt bằng OpenCV (0 lỗi)**.
  - Tính tương thích nhãn (`CHARS` Compatibility): **100% hợp lệ**. Đã chủ động bổ sung các ký tự hiếm vốn thiếu trong VNLP (`J, O, Q, W`).

---

## 4. Đề xuất Tập Dữ liệu cho Pha Huấn luyện V2 (Dataset V2 Candidates)
Để chuẩn bị cho việc retrain model LPRNet-Lite trong tương lai mà không gây thiên lệch và không vi phạm rò rỉ dữ liệu:
1. **Dữ liệu hiện tại (Current VNLP Train):** 29,836 mẫu.
2. **Dữ liệu mới từ PBL4 sau khi khử trùng lặp:** 1,211 mẫu (`dataset_staging/pbl4_unique/`).
3. **Dữ liệu tổng hợp (Synthetic Data):** Đề xuất sinh từ 5,000 đến 10,000 mẫu bổ sung chuyên biệt cho các ký tự hiếm (`J, O, Q, W, I`), các biển số xe máy 2 dòng và các biến thể góc nghiêng.
4. **Tập Test giữ nguyên:** Giữ nguyên 100% `TEST_FINAL` (3,733 mẫu) để đảm bảo so sánh công bằng và khoa học giữa Model V1 và Model V2.

---

## 5. Danh sách File và Script Mới Đã Tạo
1. `scripts/dataset_tools/deduplicate_dataset.py`: Công cụ khử trùng lặp 2 tầng (SHA256 & pHash) và kiểm soát rò rỉ tập test.
2. `scripts/dataset_tools/generate_lprnet_vn.py`: Adapter sinh dữ liệu tổng hợp chuẩn LPRNet/CTC.
3. `reports/project_tree.txt`: Cây thư mục snapshot của toàn bộ project.
4. `reports/current_dataset_report.md`: Báo cáo chi tiết về Dataset VNLP và hiện trạng OCR.
5. `reports/pbl4_dataset_report.md`: Báo cáo kiểm định dataset PBL4.
6. `reports/pbl4_label_analysis.csv`: Phân tích nhãn chi tiết của 13,320 mẫu PBL4.
7. `reports/pbl4_character_frequency.csv`: Thống kê tần suất từng ký tự trong PBL4.
8. `reports/pbl4_vs_current_duplicates.csv`: Bảng đối chiếu trùng lặp và phân loại hành động (13,320 dòng).
9. `reports/synthetic_smoke_test.md`: Báo cáo nghiệm thu smoke test 100 ảnh tổng hợp.
10. `reports/dataset_expansion_summary.md`: Báo cáo tổng hợp mở rộng dữ liệu (file này).
11. `reports/project_state_for_chatgpt.md`: Bản snapshot trạng thái toàn diện cập nhật cho ChatGPT.
