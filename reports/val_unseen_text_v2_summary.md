# TỔNG HỢP TẬP VALIDATION CHƯA THẤY KÝ TỰ (VAL UNSEEN TEXT V2)

**Thời gian tạo:** 2026-09-24
**Tập nguồn Train V2:** `datasets\VNLP_processed\train_v2a.csv` (31,047 mẫu: VNLP Train + PBL4 unique)
**Tập nguồn Validation:** `datasets\VNLP_processed\val.csv` (3728 mẫu)
**Tập tạo mới:** `datasets\VNLP_processed\val_unseen_text_v2.csv`

---

## 1. Thống Kê Tổng Quan
- **Tổng số mẫu trong VAL:** **3728** mẫu
- **Số mẫu VAL có chuỗi ký tự chưa từng xuất hiện trong TRAIN V2:** **1248** mẫu (33.48%)
- **Số chuỗi ký tự VAL đã xuất hiện trong TRAIN V2 (Overlap):** **2480** mẫu (66.52%)

---

## 2. Phân Bố Theo Kiểu Biển (Layout Breakdown)
| Kiểu biển (Layout) | Số mẫu | Tỷ lệ (%) |
| :--- | :---: | :---: |
| 1 Dòng (`one_row`) | 731 | 58.57% |
| 2 Dòng ô tô (`two_rows`) | 319 | 25.56% |
| 2 Dòng xe máy (`two_rows_label_xe_may`) | 198 | 15.87% |

---

## 3. Phân Bố Độ Dài Biển Số (Label Length Distribution)
| Độ dài (Ký tự) | Số mẫu | Tỷ lệ (%) |
| :---: | :---: | :---: |
| 7 | 121 | 9.70% |
| 8 | 970 | 77.72% |
| 9 | 153 | 12.26% |
| 10 | 4 | 0.32% |

---

## 4. Ý Nghĩa và Khuyến Nghị Sử Dụng
- Tập `val_unseen_text_v2.csv` có **1248** mẫu (33.5% tập VAL).
- Tập này hoàn toàn độc lập với TEST và cho phép quan sát sớm khả năng tổng quát hóa trong quá trình fine-tune.
- **Lưu ý:** Metric chính để quyết định checkpoint candidate vẫn là **toàn bộ tập VAL (3,728 mẫu)** để đảm bảo độ tin cậy và kích thước mẫu tối ưu.
