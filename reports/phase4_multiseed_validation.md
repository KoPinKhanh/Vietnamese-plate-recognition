# BÁO CÁO TỔNG HỢP KIỂM ĐỊNH ĐỐI CHỨNG ĐA SEED TRÊN VALIDATION (PHASE 4)
*(PHASE 4 MULTI-SEED VALIDATION COMPARISON REPORT)*

**Thời gian thực hiện:** 2026-09-24
**Thiết kế thí nghiệm:** 2 Điều kiện (CONTROL vs PBL4) x 3 Random Seeds (42, 1337, 2026) = 6 Chạy độc lập.
**Mục tiêu cốt lõi:** Phân lập tác động thực sự của 1,211 ảnh PBL4 đối với hiệu năng mô hình (so với continued fine-tuning đơn thuần trên VNLP).

---

## 1. Kết Quả Chi Tiết Từng Chạy (Per-Run Validation Results)
| Experiment Name | Condition | Seed | Best Epoch | VAL Exact (%) | VAL Unseen (%) | 1Row (%) | 2Row (%) | Moto (%) | CharAcc (%) | CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CONTROL_S42** | CONTROL (VNLP) | `42` | Epoch 3 | **86.02%** | 84.86% | 90.57% | 81.44% | 80.86% | 97.48% | 0.0252 |
| **CONTROL_S1337** | CONTROL (VNLP) | `1337` | Epoch 1 | **85.94%** | 85.66% | 90.25% | 81.52% | 81.22% | 97.44% | 0.0256 |
| **CONTROL_S2026** | CONTROL (VNLP) | `2026` | Epoch 2 | **85.84%** | 85.02% | 90.20% | 81.52% | 80.68% | 97.44% | 0.0256 |
| **PBL4_S42** | PBL4 (VNLP+PBL4) | `42` | Epoch 6 | **85.76%** | 85.18% | 90.04% | 81.28% | 81.22% | 97.43% | 0.0257 |
| **PBL4_S1337** | PBL4 (VNLP+PBL4) | `1337` | Epoch 3 | **85.92%** | 85.18% | 90.51% | 81.36% | 80.50% | 97.45% | 0.0255 |
| **PBL4_S2026** | PBL4 (VNLP+PBL4) | `2026` | Epoch 1 | **85.57%** | 85.02% | 90.04% | 80.97% | 80.68% | 97.39% | 0.0261 |

---

## 2. Thống Kê Tổng Hợp Đa Seed (Mean ± Std [Min - Max])
| Chỉ số Đánh giá | CONTROL (VNLP Only) | PBL4 (VNLP + PBL4) | Delta (PBL4 - CONTROL) | Đánh giá |
| :--- | :---: | :---: | :---: | :--- |
| **VAL Exact Accuracy** | 85.94 ± 0.08% [85.84% - 86.02%] | **85.75 ± 0.14%** [85.57% - 85.92%] | **-0.19 pp** | TƯƠNG ĐƯƠNG |
| **VAL Unseen Exact** | 85.18 ± 0.35% [84.86% - 85.66%] | **85.12 ± 0.08%** [85.02% - 85.18%] | **-0.05 pp** | TƯƠNG ĐƯƠNG |
| — Biển 1 dòng (`one_row`) | 90.34 ± 0.16% | 90.20 ± 0.22% | **-0.14 pp** | Ổn định |
| — Biển 2 dòng ô tô (`two_rows`) | 81.50 ± 0.04% | 81.21 ± 0.17% | **-0.29 pp** | Ổn định |
| — Biển 2 dòng xe máy (`motorcycle`) | 80.92 ± 0.22% | 80.80 ± 0.30% | **-0.12 pp** | Cải thiện |
| **VAL CER** | 0.0255 ± 0.0002 | 0.0258 ± 0.0002 | **+0.0003** | TƯƠNG ĐƯƠNG |

---

## 3. Bảng So Sánh Ghép Cặp Theo Random Seed (Paired Seed Comparisons)
| Random Seed | Control VAL (%) | PBL4 VAL (%) | Delta VAL | Control Unseen (%) | PBL4 Unseen (%) | Delta Unseen | Control Moto (%) | PBL4 Moto (%) | Delta Moto |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Seed `42` | 86.02% | 85.76% | **-0.27 pp** | 84.86% | 85.18% | **+0.32 pp** | 80.86% | 81.22% | **+0.36 pp** |
| Seed `1337` | 85.94% | 85.92% | **-0.03 pp** | 85.66% | 85.18% | **-0.48 pp** | 81.22% | 80.50% | **-0.72 pp** |
| Seed `2026` | 85.84% | 85.57% | **-0.27 pp** | 85.02% | 85.02% | **+0.00 pp** | 80.68% | 80.68% | **+0.00 pp** |

---

## 4. Kết Luận Khoa Học Về Tác Động Của Dữ Liệu PBL4
- **Đánh giá hiệu ứng của PBL4:** **[NOT SUPPORTED]**.
- **Phân lập nguyên nhân:**
  1. Cả CONTROL lẫn PBL4 khi fine-tune từ V1 đều cho kết quả tốt hơn V1 gốc (~84.79%), chứng minh bản thân quá trình fine-tune với learning rate nhỏ đã giúp tinh chỉnh trọng số.
  2. Khi so sánh trực diện giữa CONTROL và PBL4 trên cùng random seed, điều kiện có PBL4 đạt Mean VAL là **85.75%** so với **85.94%** của CONTROL (chênh lệch -0.19 pp).
