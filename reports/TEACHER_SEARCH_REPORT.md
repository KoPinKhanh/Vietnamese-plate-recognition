# TEACHER SEARCH REPORT: VIETNAMESE LICENSE PLATE RECOGNITION

**Date:** 2026-09-25  
**Project Root:** `D:\Vietnam_Plate_Project`  
**Target:** Identify an OCR Teacher model with representations significantly superior to the current best student (`LPRNetLite` @ 86.02% VAL / 85.96% TEST).

---

## 1. Executive Summary

| Role | Architecture | Checkpoint | VAL Exact | VAL Unseen | CER | Qualification | Decision |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Student Baseline** | `LPRNetLite` | `weights/ocr/experiments/phase4/control_seed42/best.pth` | 86.02% | 84.86% | 0.0252 | *Baseline* | Reference |
| **Teacher A** | `PARSeq-Tiny` | `weights/ocr/teachers/parseq_tiny_vn/best.pth` | **95.20%** | **94.23%** | **0.0074** | **`QUALIFIED_STRONG`** | **CHOSEN FOR KD** |
| **Teacher B** | `SVTR-Tiny` | N/A | N/A | N/A | N/A | *Skipped* | Skipped via Section 16 Rule |
| **Teacher C** | `ResNet-CTC` | N/A | N/A | N/A | N/A | *Skipped* | Skipped via Section 16 Rule |

**Key Finding:** Pretrained `PARSeq-Tiny` fine-tuned with project-consistent two-row rectification achieved **95.20% VAL Exact Accuracy** (+9.18 pp over Student baseline), with **99.16%** on 1-row plates, **95.00%** on 2-row car plates, and **94.23%** on unseen text, cutting the character error rate by over 70%. Because VAL surpassed the 90.0% threshold (Section 16), secondary architectures were skipped and the pipeline immediately advanced to Knowledge Distillation.

---

## 2. Baseline & Invariant Principles
1. **Student Baseline:** `LPRNetLite` (798,565 params) initialized from Phase 4 Control Best (`control_seed42/best.pth`). Baseline checkpoint is strictly protected and never overwritten.
2. **Data Policy:** Training on `datasets/VNLP_processed/train.csv` (29,836 samples); validation on `val.csv` (3,728 samples) and `val_unseen_text_v2.csv` (1,248 samples). No PBL4, no Phase 5 synthetic data.
3. **Locking Principle:** TEST sets remain completely locked until the final student candidate is chosen based solely on validation.

---

## 3. Teacher A: PARSeq-Tiny Architecture & Training
- **Base Architecture:** Vision Transformer Permutation Autoregressive Sequence Model (`baudm/parseq`, NeurIPS 2022).
- **Parameters:** 3,828,901 (~15.3 MB).
- **Native Input:** `[B, 3, 32, 128]`, RGB normalized to `[-1.0, 1.0]`.
- **Pretrained Weights:** Pretrained on large-scale scene text datasets, loaded via `torch.hub`.
- **Preprocessing Consistency:** Identical ground-truth bounding box crop (`pad_ratio=0.03`) and Sobel vertical projection valley rectification for two-row plates as Student.
- **Optimization:** `AdamW` (lr = 1e-4, weight_decay = 1e-4), `CosineAnnealingLR` (T_max = 10), AMP (Float16), batch size 64.

---

## 4. Teacher Validation Performance & Layout Breakdown

### Progress Across Epochs
- **Pretrained Zero-Shot (No VNLP Training):** VAL Unseen = **60.02%**, CER = 0.1621.
- **Fine-Tuning Epoch 1:** VAL Exact = **93.21%** (Unseen: 92.95%), CER = 0.0106.
- **Fine-Tuning Epoch 2:** VAL Exact = **94.15%** (Unseen: 93.19%), CER = 0.0093.
- **Fine-Tuning Epoch 6:** VAL Exact = **95.12%** (Unseen: 94.07%), CER = 0.0073.
- **Fine-Tuning Epoch 8 (Best):** VAL Exact = **95.20%** (Unseen: 94.23%), CER = 0.0074.

### Detailed Validation Breakdown (Best Checkpoint)
| Plate Layout | Samples | Correct | Exact Acc (%) | Char Acc (%) | CER | Avg Edit Dist |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **one_row** | 1,908 | 1,892 | **99.16%** | 99.88% | 0.0012 | 0.0094 |
| **two_rows (car)** | 1,261 | 1,198 | **95.00%** | 99.23% | 0.0077 | 0.0603 |
| **two_rows (motorcycle)**| 559 | 459 | **82.11%** | 97.21% | 0.0279 | 0.2397 |
| **OVERALL VAL** | **3,728** | **3,549** | **95.20%** | **99.26%** | **0.0074** | **0.0612** |
| **VAL_UNSEEN** | **1,248** | **1,176** | **94.23%** | **99.07%** | **0.0093** | **0.0745** |

---

## 5. Teacher Qualification & Decision Gate
- **Qualification Criteria:**
  - `QUALIFIED_STRONG`: VAL Exact $\ge 88.0\%$ and VAL_UNSEEN $\ge 84.86\%$.
  - Achieved: **94.15%** ($\ge 88.0\%$) and **93.19%** ($\ge 84.86\%$).
- **Status:** **`QUALIFIED_STRONG`**
- **Stop Condition (Section 16):** Because VAL Exact exceeds 90.0% with strong generalization across unseen plates, Teacher B (SVTR) and Teacher C (ResNet-CTC) are skipped to optimize compute efficiency.
- **Action:** Proceed directly to Knowledge Distillation (`KD_A`) into `LPRNetLite`.
