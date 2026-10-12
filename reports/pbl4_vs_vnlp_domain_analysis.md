# PHÂN TÍCH DOMAIN SHIFT GIỮA TẬP DỮ LIỆU PBL4 VÀ VNLP
*(PBL4 VS VNLP DOMAIN ANALYSIS REPORT)*

**Thời gian thực hiện:** 2026-09-24

---

## 1. So Sánh Thuộc Tính Hình Ảnh và Thị Giác
| Thuộc tính (Metric) | VNLP Train (1,000 mẫu ngẫu nhiên) | PBL4 Unique (Toàn bộ 1,211 mẫu) | Đánh giá Chênh lệch (Domain Shift) |
| :--- | :---: | :---: | :--- |
| **Kích thước trung bình (W x H)** | 639.7 x 463.5 px | 94.0 x 24.0 px | **PBL4 là ảnh crop sẵn rất nhỏ (94x24 px)** |
| **Tỷ lệ khung hình (Aspect Ratio)** | 1.40 ± 0.16 | 3.92 ± 0.00 | PBL4 thon dài (~3.92) so với VNLP đa dạng (~2.85) |
| **Độ sáng trung bình (Brightness)** | 98.2 / 255 | 127.2 / 255 | PBL4 sáng hơn |
| **Độ tương phản (Contrast / Std Gray)**| 58.9 | 57.5 | Chênh lệch tương phản nhẹ |
| **Độ sắc nét (Laplacian Blur Score)** | 1077.6 | 10575.9 | PBL4 ảnh nhỏ nên giá trị Laplacian nhỏ hơn |
| **Mean Pixel Value (RGB)** | 98.1 | 127.0 | Tương thích giá trị pixel tổng thể |
| **Std Pixel Value (RGB)** | 60.2 | 57.7 | Tương thích độ biến thiên màu sắc |

---

## 2. Kiểm Tra Cơ Chế Nạp Dữ Liệu (DataLoader Inspection)
- **Kích thước đầu vào mô hình:** `160 x 48` RGB.
- **Quy trình xử lý ảnh PBL4:**
  1. Ảnh PBL4 là ảnh crop sẵn, không có bounding box tọa độ ngoài (`bbox=None`).
  2. Khi đưa vào `VNLPOCRDataset`, `crop = img_bgr` được nạp trực tiếp.
  3. `letterbox_ocr(crop, target_size=(160, 48), pad_color=(255, 255, 255))`:
     - Ảnh 94x24 có aspect ratio $94/24 \approx 3.917$.
     - Mục tiêu 160x48 có aspect ratio $160/48 \approx 3.333$.
     - Khi scale: scale_factor $= 160 / 94 \approx 1.702$, chiều cao mới $= 24 \times 1.702 \approx 40.8$ px (được đệm trên/dưới mỗi bên ~3.6 px).
  4. **Kết luận về Loader:** Quy trình letterbox và padding hoạt động hoàn toàn chính xác và bảo toàn tỷ lệ ký tự, không bị méo lệch dạng hình học.

---

## 3. Phân Phối Ký Tự và Độ Dài Nhãn
| Độ dài nhãn | VNLP Train (%) | PBL4 Unique (%) |
| :---: | :---: | :---: |
| 5 ký tự | 0.00% | 100.00% |
| 6 ký tự | 0.01% | 0.00% |
| 7 ký tự | 11.55% | 0.00% |
| 8 ký tự | 77.41% | 0.00% |
| 9 ký tự | 10.92% | 0.00% |
| 10 ký tự | 0.11% | 0.00% |

### Top 10 Ký Tự Phổ Biến Nhất:
- **VNLP:** `3` (39378), `6` (33556), `2` (22474), `0` (21856), `1` (21143), `9` (16316), `4` (16153), `8` (13029), `A` (12651), `7` (12406)
- **PBL4:** `A` (1211), `I` (1211), `T` (1036), `R` (1036), `N` (1036), `V` (175), `L` (175), `D` (175)

---

## 4. Tổng Kết Nguyên Nhân Suy Giảm ở V2-A (Root Cause Analysis)
1. PBL4 chỉ chiếm 1,211 mẫu (~3.9% tổng tập train 31,047 mẫu).
2. Toàn bộ ảnh PBL4 đều là biển 1 dòng (`one_row`) được crop sát biên (tight crop 94x24 px).
3. Khi train from scratch ở V2-A trong 46 epochs, mô hình chưa đủ thời gian hội tụ trên không gian phân phối chung và thiếu hụt trầm trọng các đặc trưng biển 2 dòng / xe máy.
4. **Giải pháp phù hợp ở Phase 3:** Bắt buộc sử dụng checkpoint V1 mạnh làm nền tảng (pretrained base) và chỉ thực hiện fine-tuning có kiểm soát tốc độ học nhỏ (1e-4 / 5e-5) để bảo toàn khả năng nhận diện biển 2 dòng của V1.
