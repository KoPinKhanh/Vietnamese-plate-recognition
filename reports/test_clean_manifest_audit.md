# CANONICAL TEST MANIFEST AUDIT
## EVALUATION ELIGIBILITY AUDIT OF `datasets/VNLP_processed/test_clean.csv`

**Date:** 2026-09-25  
**Audit File:** `datasets/VNLP_processed/test_clean.csv`  

---

## 1. MANIFEST SPECIFICATIONS

- **FILE:** `datasets/VNLP_processed/test_clean.csv`
- **ACTUAL ROW COUNT:** **3,727**
- **SHA256:** `e41ea05eb5290b5ec0a21c5bb0e8a3d4bea2bf4e8794b946dd7e7a4f34ee708b`
- **COLUMN SCHEMA:** `['image_path', 'relative_path', 'plate_text', 'layout', 'subset', 'x1', 'y1', 'x2', 'y2']`

---

## 2. CRITICAL FULL-SCENE GATE EVALUATION

| Criterion | Requirement | Audit Result | Gate Status |
| :--- | :--- | :--- | :---: |
| **Full-Scene Images** | Must resolve to original scene images (not crops) | Images resolve to $480 \times 640$ full-scene images in `datasets/VNLP_detection/` | **PASSED (YES)** |
| **Ground-Truth Plate Text** | Clean alphanumeric plate strings | Available in column `plate_text` (100% non-empty) | **PASSED (YES)** |
| **Ground-Truth Bounding Box** | Valid coordinates ($x2 > x1$, $y2 > y1$) | Available in columns `x1`, `y1`, `x2`, `y2` (0 invalid boxes) | **PASSED (YES)** |
| **Layout / Category Labels** | Distinguishable layout tags | Categorized via `layout` and `subset` (`two_rows_label_xe_may` $\implies$ `motorcycle`) | **PASSED (YES)** |
| **Disk Image Accessibility** | 100% images readable on disk | 3,727 / 3,727 files verified readable with OpenCV | **PASSED (YES)** |

---

## 3. CANONICAL TEST DISTRIBUTION

- **Total Samples:** **3,727**
  - **`one_row` (Cars):** **1,908 (51.2%)**
  - **`two_rows` (Cars):** **1,259 (33.8%)**
  - **`motorcycle` (Two-row):** **560 (15.0%)**

---

## 4. GATE CONCLUSION

**GATE PASSED:** The canonical test partition `datasets/VNLP_processed/test_clean.csv` is fully eligible for the Final Locked Routing Benchmark. It contains complete full-scene images with verified plate bounding boxes and ground-truth text annotations.
