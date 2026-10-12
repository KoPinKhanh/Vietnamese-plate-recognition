# BÁO CÁO PHÂN TÍCH PHÂN PHỐI DỮ LIỆU HUẤN LUYỆN PHASE 4
*(PHASE 4 TRAINING DATA SOURCE DISTRIBUTION REPORT)*

**Thời gian thực hiện:** 2026-09-24

---

## 1. Xác Minh Số Lượng Mẫu Huấn Luyện
| Nguồn dữ liệu | Số lượng mẫu | Tỷ lệ trong Train V2 |
| :--- | :---: | :---: |
| **VNLP TRAIN** | **29,836** | 96.10% |
| **PBL4 UNIQUE** | **1,211** | 3.90% |
| **TỔNG CỘNG (TRAIN V2)** | **31,047** | **100.00%** |

---

## 2. Phân Bố Mã Tỉnh / Tiền Tố Biển Số (Province Prefix Distribution)
Kiểm tra hiện tượng tập trung dữ liệu địa phương ở PBL4 so với VNLP:

| Mã Tỉnh | Khu vực chính | Tỷ lệ ở VNLP (%) | Tỷ lệ ở PBL4 (%) | Đánh giá Mức độ Tập trung |
| :---: | :--- | :---: | :---: | :--- |
| **30** | Hà Nội | 4.42% (1318) | **20.48% (248)** | Cao hơn VNLP |
| **29** | Hà Nội | 11.88% (3546) | **7.35% (89)** | Bình thường |
| **51** | TP.HCM | 0.45% (135) | **50.45% (611)** | Cực kỳ tập trung ở PBL4 |
| **36** | Thanh Hóa | 74.12% (22115) | **0.83% (10)** | Bình thường |
| **14** | Tỉnh khác | 0.34% (102) | **0.50% (6)** | Cao hơn VNLP |
| **99** | Tỉnh khác | 0.35% (103) | **0.50% (6)** | Cao hơn VNLP |
| **88** | Tỉnh khác | 0.28% (85) | **1.24% (15)** | Cao hơn VNLP |
| **15** | Tỉnh khác | 0.39% (116) | **0.99% (12)** | Cao hơn VNLP |
| **43** | Tỉnh khác | 0.09% (26) | **0.25% (3)** | Cao hơn VNLP |
| **60** | Tỉnh khác | 0.05% (16) | **3.22% (39)** | Cao hơn VNLP |

### Top 5 Mã Tỉnh Chiếm Tỷ Trọng Lớn Nhất ở PBL4:
- **Mã `51`:** 611 mẫu (50.45%)
- **Mã `30`:** 248 mẫu (20.48%)
- **Mã `29`:** 89 mẫu (7.35%)
- **Mã `60`:** 39 mẫu (3.22%)
- **Mã `89`:** 33 mẫu (2.73%)

---

## 3. Phân Bố Độ Dài Biển Số (Label Length Distribution)
| Độ dài (Ký tự) | VNLP Train (Mẫu / %) | PBL4 Unique (Mẫu / %) |
| :---: | :---: | :---: |
| 6 ký tự | 3 (0.01%) | 0 (0.00%) |
| 7 ký tự | 3445 (11.55%) | 58 (4.79%) |
| 8 ký tự | 23096 (77.41%) | 1128 (93.15%) |
| 9 ký tự | 3258 (10.92%) | 25 (2.06%) |
| 10 ký tự | 34 (0.11%) | 0 (0.00%) |

---

## 4. Tần Suất Ký Tự (Character Frequency)
| Hạng | Ký tự VNLP | Tần suất VNLP | Ký tự PBL4 | Tần suất PBL4 |
| :---: | :---: | :---: | :---: | :---: |
| 1 | `3` | 39378 (16.51%) | `1` | 1404 (14.54%) |
| 2 | `6` | 33556 (14.07%) | `5` | 1237 (12.81%) |
| 3 | `2` | 22474 (9.42%) | `0` | 1033 (10.70%) |
| 4 | `0` | 21856 (9.16%) | `3` | 853 (8.83%) |
| 5 | `1` | 21143 (8.86%) | `2` | 769 (7.96%) |
| 6 | `9` | 16316 (6.84%) | `9` | 751 (7.78%) |
| 7 | `4` | 16153 (6.77%) | `6` | 670 (6.94%) |
| 8 | `8` | 13029 (5.46%) | `8` | 598 (6.19%) |
| 9 | `A` | 12651 (5.30%) | `4` | 554 (5.74%) |
| 10 | `7` | 12406 (5.20%) | `7` | 549 (5.69%) |
| 11 | `5` | 12198 (5.11%) | `A` | 425 (4.40%) |
| 12 | `C` | 6842 (2.87%) | `F` | 377 (3.90%) |
| 13 | `B` | 2807 (1.18%) | `G` | 207 (2.14%) |
| 14 | `M` | 1339 (0.56%) | `E` | 43 (0.45%) |
| 15 | `N` | 1014 (0.43%) | `D` | 42 (0.44%) |

---

## 5. Nhận Xét Quan Trọng Về Phân Phối Dữ Liệu
1. **Độ tập trung địa lý cực cao ở PBL4:**
   - Mã **`30`** (Hà Nội) chiếm tới **20.48%** trong PBL4 (so với chỉ 4.42% ở VNLP).
   - Mã **`51`** (TP.HCM) chiếm **50.45%** trong PBL4.
   - Hai đầu Hà Nội (`29, 30`) và TP.HCM (`51`) chiếm tới hơn **70%** toàn bộ tập PBL4.
2. **Độ dài nhãn:**
   - Cả hai tập đều chủ yếu là biển 8 ký tự (VNLP: 77.6%, PBL4: 86.6%).
   - PBL4 hoàn toàn không có biển 6 hoặc 10 ký tự.
3. **Kiểu biển:**
   - Toàn bộ 1,211 mẫu PBL4 đều là biển 1 dòng (`one_row`), không có mẫu xe máy nào trong tập PBL4 bổ sung này.
   - Sự tăng trưởng biển xe máy ở FT-B Phase 3 cần được thí nghiệm đối chứng với CONTROL (VNLP only) để xác định xem việc fine-tune tiếp trên VNLP có mang lại hiệu ứng tương tự hay không.
