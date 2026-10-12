# TỔNG HỢP MANIFEST HUẤN LUYỆN V2-A (TRAIN V2-A MANIFEST SUMMARY)

**Thời gian tạo:** 2026-09-24
**Manifest V2-A:** `datasets/VNLP_processed/train_v2a.csv`
**Sidecar Sources:** `datasets/VNLP_processed/train_v2a_sources.csv`

---

## 1. Thành phần Dữ liệu Huấn luyện V2-A
- **Mẫu huấn luyện VNLP gốc:** **29,836** mẫu (96.10%)
- **Mẫu PBL4 Unique yêu cầu:** **1,211** mẫu
- **Mẫu PBL4 bị từ chối kiểm định:** **0** mẫu
- **Mẫu PBL4 Unique được chấp nhận:** **1,211** mẫu (3.90%)
- **Tổng số mẫu Huấn luyện V2-A:** **31,047** mẫu (100.0%)

---

## 2. Phân bố Bố cục & Nhóm Dữ liệu
- **one_row (Biển 1 dòng ô tô + PBL4):** 16,479 mẫu (53.08%)
- **two_rows (Biển 2 dòng ô tô):** 10,094 mẫu (32.51%)
- **two_rows_label_xe_may (Biển 2 dòng xe máy):** 4,474 mẫu (14.41%)

---

## 3. Phân bố Độ dài Nhãn Biển Số
- Độ dài **6** ký tự: 3 mẫu (0.01%)
- Độ dài **7** ký tự: 3,503 mẫu (11.28%)
- Độ dài **8** ký tự: 24,224 mẫu (78.02%)
- Độ dài **9** ký tự: 3,283 mẫu (10.57%)
- Độ dài **10** ký tự: 34 mẫu (0.11%)

---

## 4. Bảng Tần suất Ký tự trong Toàn bộ Tập Huấn luyện V2-A
| Ký tự | Số lần xuất hiện | Ký tự | Số lần xuất hiện |
| :---: | :---: | :---: | :---: |
| `0` | 22,889 | `G` | 593 |
| `1` | 22,547 | `H` | 523 |
| `2` | 23,243 | `I` | 1 |
| `3` | 40,231 | `K` | 297 |
| `4` | 16,707 | `L` | 774 |
| `5` | 13,435 | `M` | 1,343 |
| `6` | 34,226 | `N` | 1,021 |
| `7` | 12,955 | `P` | 237 |
| `8` | 13,627 | `R` | 362 |
| `9` | 17,067 | `S` | 166 |
| `A` | 13,076 | `T` | 258 |
| `B` | 2,837 | `U` | 136 |
| `C` | 6,851 | `V` | 196 |
| `D` | 766 | `X` | 243 |
| `E` | 532 | `Y` | 252 |
| `F` | 692 | `Z` | 135 |

---

## 5. Kết luận Tính sẵn sàng
- Toàn bộ 31,047 mẫu đều có đường dẫn hợp lệ, ảnh đọc tốt và nhãn tương thích 100% với từ điển 36 ký tự của `LPRNetLite`.
- Tập Train V2-A đã sẵn sàng cho bước smoke test và huấn luyện mô hình V2-A.
