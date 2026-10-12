# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH PYTORCH LPRNET V2-A (V2-A PYTORCH EVALUATION)

**Thời gian thực hiện:** 2026-09-24
**File Checkpoint:** `weights\ocr\experiments\v2a\best_lprnet_v2a.pth`
**Epoch tốt nhất:** `26`
**Thiết bị thực thi:** `cuda`

---

## 1. Kết Quả Tổng Thể trên Các Tập Đánh Giá
| Tập dữ liệu | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VAL** | 3728 | 2466/3728 | **66.15%** | 94.02% | 0.0598 | 0.4767 |
| **TEST_FINAL** | 3733 | 2459/3733 | **65.87%** | 93.94% | 0.0606 | 0.4819 |
| **TEST_CLEAN** | 3727 | 2456/3727 | **65.90%** | 93.95% | 0.0605 | 0.4811 |
| **TEST_UNSEEN_TEXT** | 1187 | 800/1187 | **67.40%** | 94.03% | 0.0597 | 0.4819 |

---

## 2. Phân Tích Chi Tiết Theo Kiểu Biển (Layout Breakdown)
### Tập dữ liệu: VAL
| Kiểu biển (Layout) | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 Dòng (one_row) | 1908 | 1372/1908 | **71.91%** | 95.65% | 0.0435 | 0.3407 |
| 2 Dòng Ô tô (two_rows) | 1261 | 819/1261 | **64.95%** | 92.78% | 0.0722 | 0.5615 |
| 2 Dòng Xe Máy (motorcycle) | 559 | 275/559 | **49.19%** | 91.28% | 0.0872 | 0.7496 |

### Tập dữ liệu: TEST_FINAL
| Kiểu biển (Layout) | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 Dòng (one_row) | 1910 | 1379/1910 | **72.20%** | 95.64% | 0.0436 | 0.3414 |
| 2 Dòng Ô tô (two_rows) | 1263 | 804/1263 | **63.66%** | 92.59% | 0.0741 | 0.5709 |
| 2 Dòng Xe Máy (motorcycle) | 560 | 276/560 | **49.29%** | 91.18% | 0.0882 | 0.7607 |

### Tập dữ liệu: TEST_CLEAN
| Kiểu biển (Layout) | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 Dòng (one_row) | 1908 | 1379/1908 | **72.27%** | 95.67% | 0.0433 | 0.3396 |
| 2 Dòng Ô tô (two_rows) | 1259 | 801/1259 | **63.62%** | 92.58% | 0.0742 | 0.5711 |
| 2 Dòng Xe Máy (motorcycle) | 560 | 276/560 | **49.29%** | 91.18% | 0.0882 | 0.7607 |

### Tập dữ liệu: TEST_UNSEEN_TEXT
| Kiểu biển (Layout) | Số mẫu | Đúng / Tổng | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Distance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 Dòng (one_row) | 696 | 524/696 | **75.29%** | 96.07% | 0.0393 | 0.3103 |
| 2 Dòng Ô tô (two_rows) | 281 | 175/281 | **62.28%** | 91.86% | 0.0814 | 0.6299 |
| 2 Dòng Xe Máy (motorcycle) | 210 | 101/210 | **48.10%** | 90.16% | 0.0984 | 0.8524 |

---

## 3. Nhận Xét Khái Quát về Checkpoint PyTorch V2-A
- Checkpoint PyTorch tốt nhất đạt Exact Accuracy trên TEST_FINAL là **65.87%** (so với V1 baseline 85.27%).
- Khả năng tổng quát hóa trên TEST_UNSEEN_TEXT đạt **67.40%** (so với V1 baseline 72.11%).
- Tập loại bỏ rò rỉ hình ảnh TEST_CLEAN đạt **65.90%**.
