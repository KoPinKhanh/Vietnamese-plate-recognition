# BÁO CÁO ĐỐI CHỨNG CÁC CAN THIỆP PILOT PHASE 5 (PHASE 5 PILOT COMPARISON)

**Thời gian thực hiện:** 2026-09-25
**Base Checkpoint:** `weights/ocr/experiments/phase4/control_seed42/best.pth` (V1 + VNLP Train)
**Baseline Metrics (Phase 4 Base):** VAL Exact = **86.02%** | Unseen = **84.86%** | Moto = **80.86%** | CER = **0.0252**

---

## 1. Bảng So Sánh Hiệu Năng Các Can Thiệp Pilot (Đo Trên Validation)
| Experiment ID | Can thiệp (Intervention) | VAL Exact (%) | Delta VAL | VAL Unseen (%) | Moto (%) | Delta Moto | 2Row (%) | CER | Đánh giá |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BASE_P4** | Base Checkpoint Phase 4 | 86.02% | 0.00 pp | 84.86% | 80.86% | 0.00 pp | 81.44% | 0.0252 | BASELINE |
| **P5A_HARD_MINING** | Hard Example Mining | **86.02%** | +0.00 pp | 84.86% | 80.86% | +0.00 pp | 81.44% | 0.0252 | KHÔNG ĐẠT NGƯỠNG |
| **P5B_TARGET_AUG** | Targeted Real Augmentation | **86.02%** | +0.00 pp | 84.86% | 80.86% | +0.00 pp | 81.44% | 0.0252 | KHÔNG ĐẠT NGƯỠNG |
| **P5C_SYNTH_MOTO** | Targeted Synthetic Motorcycle | **86.02%** | +0.00 pp | 84.86% | 80.86% | +0.00 pp | 81.44% | 0.0252 | KHÔNG ĐẠT NGƯỠNG |
| **P5D_BALANCED** | Balanced Layout Sampling | **86.02%** | +0.00 pp | 84.86% | 80.86% | +0.00 pp | 81.44% | 0.0252 | KHÔNG ĐẠT NGƯỠNG |

---

## 2. Tiêu Chí Chọn Lọc Shortlist Sang Đa Seed
- Chỉ chọn tối đa **2 can thiệp** có kết quả tích cực nhất trên Validation (tăng VAL hoặc tăng mạnh motorcycle mà không làm giảm tổng thể).
- Các can thiệp trong shortlist sẽ được kiểm định qua thêm 2 seeds: `1337` và `2026`.
