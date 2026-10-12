# BÁO CÁO BASELINE CHÍNH THỨC MODEL V1 BẰNG PYTORCH (PHASE 3)

**Thời gian thực hiện:** 2026-09-24
**Framework:** PyTorch (CUDA)
**File Checkpoint V1:** `weights/ocr/best_lprnet_vn.pth`
**SHA256:** `25d2ff3cba0869766a96689a8eda2d1b133634c86023df01e2e41cf17491305c`
**Evaluator:** `tools/evaluate_ocr_model.py` (Chuẩn hóa duy nhất)

---

## 1. Kết Quả Tổng Thể trên Các Tập Đánh Giá (Official Phase 3 Baseline)
| Tập dữ liệu | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VAL** | 3728 | 3161/3728 | **84.79%** | 97.18% | 0.0282 | 0.2234 |
| **TEST_FINAL** | 3733 | 3183/3733 | **85.27%** | 97.15% | 0.0285 | 0.2253 |
| **TEST_CLEAN** | 3727 | 3180/3727 | **85.32%** | 97.17% | 0.0283 | 0.2235 |
| **TEST_UNSEEN_TEXT** | 1187 | 1020/1187 | **85.93%** | 97.24% | 0.0276 | 0.2216 |

---

## 2. Phân Tích Chi Tiết Theo Kiểu Biển (Layout Breakdown)
### Tập dữ liệu: VAL
| Kiểu biển (Layout) | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 Dòng (one_row) | 1908 | 1717/1908 | **89.99%** | 98.45% | 0.0155 | 0.1184 |
| 2 Dòng Ô tô (two_rows) | 1261 | 1018/1261 | **80.73%** | 95.87% | 0.0413 | 0.3220 |
| 2 Dòng Xe Máy (motorcycle) | 559 | 426/559 | **76.21%** | 95.82% | 0.0418 | 0.3596 |

### Tập dữ liệu: TEST_FINAL
| Kiểu biển (Layout) | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 Dòng (one_row) | 1910 | 1730/1910 | **90.58%** | 98.35% | 0.0165 | 0.1267 |
| 2 Dòng Ô tô (two_rows) | 1263 | 1029/1263 | **81.47%** | 95.88% | 0.0412 | 0.3175 |
| 2 Dòng Xe Máy (motorcycle) | 560 | 424/560 | **75.71%** | 95.90% | 0.0410 | 0.3536 |

### Tập dữ liệu: TEST_CLEAN
| Kiểu biển (Layout) | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 Dòng (one_row) | 1908 | 1730/1908 | **90.67%** | 98.40% | 0.0160 | 0.1232 |
| 2 Dòng Ô tô (two_rows) | 1259 | 1026/1259 | **81.49%** | 95.87% | 0.0413 | 0.3177 |
| 2 Dòng Xe Máy (motorcycle) | 560 | 424/560 | **75.71%** | 95.90% | 0.0410 | 0.3536 |

### Tập dữ liệu: TEST_UNSEEN_TEXT
| Kiểu biển (Layout) | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 Dòng (one_row) | 696 | 639/696 | **91.81%** | 98.39% | 0.0161 | 0.1250 |
| 2 Dòng Ô tô (two_rows) | 281 | 225/281 | **80.07%** | 95.61% | 0.0439 | 0.3381 |
| 2 Dòng Xe Máy (motorcycle) | 210 | 156/210 | **74.29%** | 95.57% | 0.0443 | 0.3857 |

---

## 3. Quy Định Bất Biến Cho Phase 3
1. Bảng số liệu trên là **GROUND TRUTH BASELINE DUY NHẤT** được công nhận cho toàn bộ Phase 3.
2. Mọi kết quả fine-tune (FT-A, FT-B, FT-C) đều phải được so sánh đối chiếu với các con số này trên cùng Evaluator PyTorch.
3. **Chỉ số chọn lọc mô hình:** `VAL Exact Accuracy` (Baseline V1: **84.79%**).
