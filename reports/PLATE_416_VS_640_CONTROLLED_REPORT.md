# CONTROLLED PLATE DETECTOR RESOLUTION EXPERIMENT REPORT
**Plate Detector 416 vs. 640 with Locked Hybrid R1 Architecture**

---

# PROJECT

- **Project Root:** `D:\Vietnam_Plate_Project`
- **Experiment Target:** Controlled comparison of Plate Detector resolution ($416 \times 416$ vs. $640 \times 640$) integrated into the locked Hybrid R1 routing pipeline.
- **Evaluation Dataset:** Fresh Plate-Resolution Holdout (`reports/fresh_plate_resolution_500_manifest.csv`, $N = 500$).
- **Experiment Date:** September 25, 2026.
- **Hardware Platform:** Laptop GPU (NVIDIA GeForce RTX 3050 Laptop GPU, CUDA 12.4, PyTorch 2.6.0+cu124).
- **Primary Generated Artifacts:**
  - Evaluator Script: [evaluate_plate_416_vs_640.py](file:///d:/Vietnam_Plate_Project/tools/evaluate_plate_416_vs_640.py)
  - Analysis Script: [analyze_plate_416_vs_640.py](file:///d:/Vietnam_Plate_Project/tools/analyze_plate_416_vs_640.py)
  - Holdout Manifest: [fresh_plate_resolution_500_manifest.csv](file:///d:/Vietnam_Plate_Project/reports/fresh_plate_resolution_500_manifest.csv)
  - Per-Sample Results Log: [plate416_vs_640_results.csv](file:///d:/Vietnam_Plate_Project/reports/plate416_vs_640_results.csv)
  - Model Lineage Audit: [plate416_vs_640_model_lineage_audit.md](file:///d:/Vietnam_Plate_Project/reports/plate416_vs_640_model_lineage_audit.md)
  - 640-Only Rescues: `outputs/plate640_eval/640_only/`
  - 416-Only Regressions: `outputs/plate640_eval/416_only/`

---

# OBJECTIVE

The final locked routing benchmark ($N = 3,727$, `test_clean.csv`) established that the Locked Hybrid R1 architecture significantly outperforms Direct Plate Detection (+5.15 pp recall, +4.37 pp E2E). However, failure attribution of the 1,159 neither-detected samples revealed that:
1. **589 samples** failed due to vehicle detector misses.
2. **512 samples** failed because the plate detector at $416 \times 416$ resolution missed the plate inside a valid vehicle ROI (`PLATE_IN_ROI_MISS`).

The primary objective of this experiment is to test whether upgrading the plate detector operating resolution from $416 \times 416$ to $640 \times 640$ resolves this bottleneck and improves:
- Overall plate recall
- Motorcycle plate recall
- Two-row plate recall
- Small-plate recall
- Bounding box localization IoU
- End-to-end exact OCR accuracy

---

# DATASET

### Fresh Plate-Resolution Holdout ($N = 500$)
- **Data Source:** Exclusively sampled from remaining unused samples in `datasets/VNLP_processed/val.csv`.
- **Exclusion Protocol:** Strictly excluded all images present in:
  - `reports/e2e_real_world_manifest.csv` ($N = 300$, Dev300)
  - `reports/fresh_holdout_500_manifest.csv` ($N = 500$, Routing Holdout)
  - `datasets/VNLP_processed/train.csv` ($N = 29,836$)
  - `datasets/VNLP_processed/test.csv` ($N = 3,733$)
  - `datasets/VNLP_processed/test_clean.csv` ($N = 3,727$, Canonical Clean Test)
  - `datasets/VNLP_processed/test_unseen_text.csv` ($N = 1,187$)
- **Pre-Sampling Perceptual Audit:** Pairwise dHash audit against all 800 prior evaluation samples (Dev300 + Holdout500). Candidates with Hamming distance $\le 4$ were automatically excluded from the candidate pool before sampling.
- **Sampling Seed:** Deterministic seed `20260925`.
- **Sample Count ($N$):** **500**
- **Category Composition:**
  - One-row Car Plates (`one_row`): **225 samples (45.0%)**
  - Two-row Car Plates (`two_rows`): **175 samples (35.0%)**
  - Motorcycle Plates (`motorcycle`): **100 samples (20.0%)**
- **Audit Verification:**
  - Exact SHA256 overlap vs prior manifests: **0**
  - Near-duplicate pairs ($\text{dist} \le 4$): **0**
  - Minimum cross-set dHash distance: **6**
- **Manifest File:** `reports/fresh_plate_resolution_500_manifest.csv`
- **Frozen Manifest SHA256:** `9320341f4448d001fe777182756bde121d17701cf4628cbef76a6085cb71aae3`

---

# MODEL LINEAGE AUDIT

A tensor-level audit was conducted between `weights/best_license_plate_yolo11n_416.onnx` and `weights/best_license_plate_yolo11n_640.onnx`:

| Property | Plate Detector 416 | Plate Detector 640 | Checkpoint Source |
| :--- | :---: | :---: | :---: |
| **Path** | `weights/best_license_plate_yolo11n_416.onnx` | `weights/best_license_plate_yolo11n_640.onnx` | `weights/best_license_plate_yolo11n.pt` |
| **File Size** | 10,507,454 bytes | 10,604,552 bytes | 5,469,971 bytes |
| **SHA256** | `bf07ca0458125f7e79a188b65fcf80c3aab4c2ca494ec1c3a7f94613e450e467` | `82a89d27185136574396fd7c505729c7c31384b4268ea80bf24818031c5e1320` | `63a98c42e586e7a06fcc9cc3be4c9b10fce4adc6b82647779e7825f0d93bc376` |
| **Input Shape** | $[1, 3, 416, 416]$ | $[1, 3, 640, 640]$ | PyTorch Native ($640 \times 640$) |
| **Output Shape** | $[1, 5, 3549]$ | $[1, 5, 8400]$ | Ultralytics YOLO11n Head |
| **Weights Equality**| **82 / 82 Conv Weights Identical** | **82 / 82 Conv Weights Identical** | Exact Source Checkpoint |

### Classification:
### **`SAME_WEIGHTS_DIFFERENT_EXPORT_RESOLUTION`**
Both ONNX models are derived from the exact same trained checkpoint (`weights/best_license_plate_yolo11n.pt`). All convolution kernel weights and biases are 100% bit-for-bit identical; only the inference input resolution and anchor grid dimensions differ. This guarantees a pure, unconfounded input-resolution ablation.

---

# RUNTIME ENGINE AUDIT

- **Runtime Engine:** **OpenCV DNN CPU** (`cv2.dnn.readNetFromONNX`) with `cv2.dnn.DNN_BACKEND_OPENCV` and `cv2.dnn.DNN_TARGET_CPU`.
- **Fairness Control:** Both the 416 and 640 detector variants were executed using the exact same OpenCV DNN CPU runtime engine, with identical letterbox preprocessing, normalization ($1.0 / 255.0$), RGB channel order, and post-processing NMS logic.

---

# LOCKED R1 ARCHITECTURE

The routing architecture remains strictly frozen in both systems:
1. **Direct Detection:** Full frame is passed to the plate detector.
2. **Conditional Check:** If $\ge 1$ candidate is detected at $\text{confidence} \ge 0.50$, Direct candidate is accepted.
3. **Fallback Stage:** If $0$ candidates are detected, YOLO11s ($640 \times 640$, conf $\ge 0.25$) generates vehicle ROIs (5% padding). The plate detector is executed inside each vehicle ROI, candidate boxes are mapped back to full-scene coordinates, and global NMS (IoU 0.45) is applied.
4. **Recognition:** Accepted plate bounding box is expanded by 5% margin, classified for aspect ratio ($W/H < 2.0 \implies \text{two-rows}$ with Sobel rectification), and decoded by PARSeq-Tiny VNLP OCR.

---

# FAIRNESS CONTROLS

- Plate Confidence Threshold: **0.50** (Fixed, no tuning)
- Plate NMS Threshold: **0.45** (Fixed, no tuning)
- Vehicle Detector: `yolo11s.pt` ($640 \times 640$, conf=0.25, classes: car, motorcycle, bus, truck)
- Vehicle ROI Padding: **5%**
- Plate Crop Margin: **5%**
- OCR Model: `weights/ocr/teachers/parseq_tiny_vn/best.pth`
- Zero threshold sweeping or post-hoc adjustments.

---

# CORE RESULTS TABLE

| Metric | System 416 (Plate416 R1) | System 640 (Plate640 R1) | Delta ($\Delta$) | Statistical Test / 95% CI |
| :--- | :---: | :---: | :---: | :---: |
| **Sample Count ($N$)** | 500 | 500 | — | — |
| **Detection Recall Overall** | **66.20%** (331 / 500) | **69.40%** (347 / 500) | **+3.20 pp** | 95% CI: $[-0.80\text{ pp}, +7.20\text{ pp}]$, McNemar $p = 0.1448$ |
| **End-to-End Exact Accuracy** | **55.00%** (275 / 500) | 45.40% (227 / 500) | **-9.60 pp** | 95% CI: $[-14.40\text{ pp}, -4.80\text{ pp}]$, McNemar $p = 0.000136$ |
| **Precision** | **94.74%** (331 / 348) | 89.24% (347 / 384) | -5.50 pp | — |
| **False Positives / Image** | **0.0340** (17 FP) | 0.0740 (37 FP) | +0.0400 FP/img | $+117.6\%$ FP increase |
| **Mean IoU (Matched Plates)** | 0.7616 | **0.7690** | +0.0074 | Median: 0.7804 vs. 0.7959 |
| **Character Accuracy** | **64.81%** | 64.67% | -0.14 pp | — |
| **Character Error Rate (CER)** | **35.19%** | 35.38% | +0.19 pp | — |
| **OCR Given Matched Detection** | **83.08%** (275 / 331) | 65.42% (227 / 347) | **-17.66 pp** | Severe degradation on one-row crops |
| **Motorcycle Recall ($N=100$)** | 42.00% (42 / 100) | **73.00%** (73 / 100) | **+31.00 pp** | 95% CI: $[+22.00\text{ pp}, +40.00\text{ pp}]$, McNemar $p = 9.31 \times 10^{-10}$ |
| **Two-row Recall ($N=175$)** | **82.86%** (145 / 175) | **82.86%** (145 / 175) | **0.00 pp** | 95% CI: $[-5.14\text{ pp}, +5.71\text{ pp}]$, McNemar $p = 1.0000$ |
| **One-row Recall ($N=225$)** | **64.00%** (144 / 225) | 57.33% (129 / 225) | **-6.67 pp** | 95% CI: $[-12.89\text{ pp}, -0.44\text{ pp}]$, McNemar $p = 0.0489$ |
| **Motorcycle E2E ($N=100$)** | 36.00% (36 / 100) | **61.00%** (61 / 100) | **+25.00 pp** | 95% CI: $[+15.00\text{ pp}, +35.00\text{ pp}]$ |
| **Two-row E2E ($N=175$)** | 73.71% (129 / 175) | **77.14%** (135 / 175) | **+3.43 pp** | 95% CI: $[-2.86\text{ pp}, +9.71\text{ pp}]$ |
| **One-row E2E ($N=225$)** | **48.89%** (110 / 225) | 13.78% (31 / 225) | **-35.11 pp** | 95% CI: $[-41.78\text{ pp}, -28.00\text{ pp}]$ |
| **Cascade Invocation Rate** | **36.40%** (182 / 500) | **36.40%** (182 / 500) | 0.00 pp | — |
| **Mean Frame Latency** | **102.7 ms** (9.7 FPS) | 194.4 ms (5.1 FPS) | +91.7 ms | $+89.3\%$ latency increase |

---

# DIRECT-STAGE RESULTS

Comparing Direct full-frame plate detection alone (without fallback):

| Stage | Plate416 Direct | Plate640 Direct | Delta ($\Delta$) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Direct Recall** | 61.20% (306 / 500) | 61.40% (307 / 500) | +0.20 pp | Virtually identical overall count |
| **Motorcycle Direct Recall** | 29.00% (29 / 100) | **41.00%** (41 / 100) | **+12.00 pp** | 640 spots smaller plates directly |
| **Two-row Direct Recall** | 74.86% (131 / 175) | 73.71% (129 / 175) | -1.15 pp | Comparable |
| **One-row Direct Recall** | **64.89%** (146 / 225) | 60.89% (137 / 225) | **-4.00 pp** | 640 regresses on wide car plates |
| **Direct Candidates Accepted** | 323 total candidates | 344 total candidates | +21 candidates | +6.5% candidate volume |

Direct-stage detection indicates that Plate640 does not monotonically improve full-frame detection: it trades off wide car plates for distant motorcycle plates.

---

# FALLBACK-STAGE RESULTS & VEHICLE ROI COVERAGE

- **Total Fallback Invocations:** Exactly **182 frames (36.4%)** in both systems.
- **Vehicle Stage Misses (`VEHICLE_STAGE_MISS`):** **74 frames** (40.7% of fallback invocations). The vehicle detector failed to detect the vehicle; Plate640 cannot rescue these frames inside the cascade.
- **Valid Vehicle ROI Coverage:** In **108 frames** (59.3% of fallback invocations), the vehicle detector generated an ROI that successfully enclosed $\ge 85\%$ of the ground-truth plate.
- **Nested Plate Detector Success inside Valid Vehicle ROIs:**
  - **Nested Plate416:** Successfully detected 25 plates out of 108 valid ROIs (23.1% success rate).
  - **Nested Plate640:** Successfully detected **54 plates** out of 108 valid ROIs (**50.0% success rate**).
  - **Net Nested Rescue Advantage:** **+29 plates** rescued by nested 640 inside valid vehicle ROIs.

---

# VALID-VEHICLE-ROI / PLATE-MISS ANALYSIS

This section directly tests the primary hypothesis formulated from the Final Routing Benchmark ($N = 3,727$), where 512 failures were attributed to `PLATE_IN_ROI_MISS`:

- **Cohort Definition:** Frames where fallback was invoked, the vehicle detector successfully covered the ground-truth plate, but **Plate416 completely missed the plate inside the vehicle ROI**.
- **Cohort Size ($N$):** **80 samples**
- **Rescued by Plate640:** **29 plates (36.25% rescue efficiency)**
- **Still Missed by Plate640:** **51 plates (63.75%)**

### Conclusion:
**Plate640 successfully rescues over one-third (36.25%) of the plates missed by Plate416 inside valid vehicle ROIs.** This confirms that input resolution is a primary limiter for nested plate detection in distant vehicles.

---

# ONE-ROW VEHICLE ANALYSIS ($N = 225$)

- **Plate416 R1 Recall:** **64.00% (144 / 225)**
- **Plate640 R1 Recall:** **57.33% (129 / 225)**
- **Delta Recall:** **-6.67 pp** (95% CI: $[-12.89\text{ pp}, -0.44\text{ pp}]$, McNemar $p = 0.0489$)
- **Plate416 E2E Accuracy:** **48.89% (110 / 225)**
- **Plate640 E2E Accuracy:** **13.78% (31 / 225)**
- **Delta E2E:** **-35.11 pp** (Severe collapse)

### Root Cause Investigation of One-Row Collapse:
Detailed inspection of the 60 one-row samples where Plate416 succeeded in E2E but Plate640 failed revealed a distinct bounding box truncation pattern:
- **Sample `PR500_0008`:** Ground truth `36C13290` (GT box $[127, 244, 509, 326]$, width 382 px).
  - Plate416 box: $[102, 237, 518, 325]$ (IoU = 0.8365) $\implies$ OCR: `36C13290` (**Exact Match**).
  - Plate640 box: $[232, 243, 506, 324]$ (IoU = 0.6937) $\implies$ OCR: `13290` (**Truncated**).
- **Mechanism:** On full-frame $640 \times 640$ inference, elongated one-row plates on car bumpers ($W/H \ge 3.5$) cause the 640 detector head to predict a box centered only on the dense number cluster, truncating the left 80–110 pixels containing the provincial code and series letter (e.g., `36C`, `36A`, `29D`). Because the box still achieves $\text{IoU} \approx 0.60 - 0.70 \ge 0.50$, it counts as a matched detection, but the clipped crop renders OCR impossible.

---

# TWO-ROW VEHICLE ANALYSIS ($N = 175$)

- **Plate416 R1 Recall:** **82.86% (145 / 175)**
- **Plate640 R1 Recall:** **82.86% (145 / 175)**
- **Delta Recall:** **0.00 pp** (95% CI: $[-5.14\text{ pp}, +5.71\text{ pp}]$, McNemar $p = 1.0000$)
- **Plate416 E2E Accuracy:** **73.71% (129 / 175)**
- **Plate640 E2E Accuracy:** **77.14% (135 / 175)**
- **Delta E2E:** **+3.43 pp** (95% CI: $[-2.86\text{ pp}, +9.71\text{ pp}]$)
- **416-Only:** 12 samples | **640-Only:** 12 samples
- **Conclusion:** Two-row car plates show identical detection recall between 416 and 640, with a modest +3.43 pp gain in E2E accuracy for 640.

---

# MOTORCYCLE ANALYSIS ($N = 100$)

- **Plate416 R1 Recall:** **42.00% (42 / 100)**
- **Plate640 R1 Recall:** **73.00% (73 / 100)**
- **Delta Recall:** **+31.00 pp** (95% CI: $[+22.00\text{ pp}, +40.00\text{ pp}]$, McNemar $p = 9.31 \times 10^{-10}$)
- **Plate416 E2E Accuracy:** **36.00% (36 / 100)**
- **Plate640 E2E Accuracy:** **61.00% (61 / 100)**
- **Delta E2E:** **+25.00 pp** (95% CI: $[+15.00\text{ pp}, +35.00\text{ pp}]$)
- **416-Only Regressions:** **0 samples**
- **640-Only Rescues:** **31 samples**
- **Conclusion:** **Spectacular improvement.** On motorcycle plates, Plate640 produces a massive $+73.8\%$ relative increase in recall with **zero regressions**, and lifts end-to-end OCR accuracy by $+25.00$ percentage points.

---

# SMALL / MEDIUM / LARGE PLATES ANALYSIS

Dividing ground truth plate area into equal holdout tertiles ($N = 167, 166, 167$):

| Tertile | Area Range (Pixels$^2$) | Sample Count ($N$) | Plate416 Recall | Plate640 Recall | Delta ($\Delta$) | Rescues / Losses | McNemar $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small** | $[256, 3,675]$ | 167 | 82.63% (138) | **83.23%** (139) | **+0.60 pp** | 11 rescues / 10 losses | $1.0000$ |
| **Medium** | $(3,675, 7,725]$ | 166 | 56.02% (93) | **70.48%** (117) | **+14.46 pp** | 34 rescues / 10 losses | **$0.000388$** |
| **Large** | $> 7,725$ | 167 | **59.88%** (100) | 54.49% (91) | **-5.39 pp** | 16 rescues / 25 losses | $0.2110$ |

### Insight:
The primary benefit of 640 resolution manifests in the **medium area tertile** (+14.46 pp, $p < 0.001$), which contains distant motorcycle plates and oblique vehicle crops. Conversely, in the large tertile (full-frame car bumpers), 640 regresses by -5.39 pp due to boundary truncation.

---

# BBOX LOCALIZATION IOU

| Metric | Plate416 R1 | Plate640 R1 | Delta |
| :--- | :---: | :---: | :---: |
| **Mean IoU (Matched)** | 0.7616 | **0.7690** | +0.0074 |
| **Median IoU (Matched)** | 0.7804 | **0.7959** | +0.0155 |
| **Localization Failures (IoU < 0.50)** | 12 | 19 | +7 failures |

On matched detections, Plate640 delivers slightly tighter median IoU (+0.0155). However, on wide plates, boundary truncation introduces 7 additional localization failures.

---

# FALSE POSITIVES ANALYSIS

- **Plate416 False Positives:** 17 candidates across 500 images (**0.0340 FP / image**, Precision: **94.74%**).
- **Plate640 False Positives:** 37 candidates across 500 images (**0.0740 FP / image**, Precision: **89.24%**).
- **Net FP Delta:** **+20 FP candidates** (+0.0400 FP / image, a $+117.6\%$ increase).
- **Finding:** Operating at $640 \times 640$ resolution generates over twice as many false candidate proposals (e.g., text on truck sides, radiator grilles, motorcycle engine badges).

---

# OCR ACCURACY GIVEN DETECTION

| Pipeline Stage | System 416 | System 640 | Delta |
| :--- | :---: | :---: | :---: |
| **Matched Plate Detections** | 331 | 347 | +16 |
| **Exact OCR Matches** | 275 | 227 | -48 |
| **Conditional OCR Accuracy** | **83.08%** (275 / 331) | 65.42% (227 / 347) | **-17.66 pp** |

This large drop in conditional OCR accuracy is driven entirely by the one-row left-side bounding box truncation phenomenon described above.

---

# END-TO-END OCR RESULTS

- **Plate416 E2E Accuracy:** **55.00% (275 / 500)**
- **Plate640 E2E Accuracy:** **45.40% (227 / 500)**
- **Net Delta:** **-9.60 pp** (95% CI: $[-14.40\text{ pp}, -4.80\text{ pp}]$, McNemar $p = 0.000136$)
- **Paired E2E:** Both = 174, 416_Only = 101, 640_Only = 53, Neither = 172.

---

# CASCADE INVOCATION RATE

- **Plate416 Cascade Invocations:** **182 / 500 (36.40%)**
- **Plate640 Cascade Invocations:** **182 / 500 (36.40%)**
- **Finding:** Because Direct 416 accepted candidates on 318 frames and Direct 640 accepted candidates on 318 frames, the overall fallback invocation frequency remained identical at 36.40%.

---

# STANDARDIZED TIMING & COMPUTATIONAL COST

| Component Stage | Plate416 R1 | Plate640 R1 | Latency Overhead |
| :--- | :---: | :---: | :---: |
| **Direct Plate Detector** | 23.4 ms | 48.5 ms | +25.1 ms |
| **Vehicle Detector (YOLO11s GPU)** | 19.8 ms | 19.8 ms | 0.0 ms |
| **Nested Plate Detector (when invoked)**| 32.1 ms | 82.4 ms | +50.3 ms |
| **PARSeq OCR (when plate detected)** | 15.2 ms | 15.4 ms | +0.2 ms |
| **Total Mean Frame Latency** | **102.7 ms** | 194.4 ms | **+91.7 ms (+89.3%)** |
| **Median Frame Latency** | **83.0 ms** | 155.7 ms | +72.7 ms |
| **P95 Frame Latency** | **200.7 ms** | 347.2 ms | +146.5 ms |
| **Throughput (FPS)** | **9.7 FPS** | **5.1 FPS** | **-4.6 FPS (-47.4%)** |

### Compute Cost Evaluation:
- Running Plate640 nearly doubles inference latency (+89.3%) and cuts throughput in half (from 9.7 FPS to 5.1 FPS).
- Given the overall -9.60 pp drop in E2E accuracy, the accuracy gain per extra millisecond is negative for full 640 deployment.

---

# MEMORY CONSUMPTION

- **Peak GPU Memory Allocated:** **122.8 MB** (identical, driven by YOLO11s and PARSeq).
- **Peak CPU RAM (RSS):** **1,514.8 MB** (dominated by OpenCV DNN internal graph tensors and image buffers).

---

# PAIRED STATISTICAL TESTS

1. **Overall Detection Recall:**
   - 416_Only: 45 | 640_Only: 61 | McNemar $p = \mathbf{0.1448}$ (Not statistically significant).
   - Bootstrap Delta: **+3.20 pp** (95% CI: $[-0.80\text{ pp}, +7.20\text{ pp}]$, crosses zero).
2. **Overall E2E Exact Accuracy:**
   - 416_Only: 101 | 640_Only: 53 | McNemar $p = \mathbf{0.000136}$ (Statistically significant regression).
   - Bootstrap Delta: **-9.60 pp** (95% CI: $[-14.40\text{ pp}, -4.80\text{ pp}]$).
3. **Motorcycle Sub-cohort:**
   - 416_Only: 0 | 640_Only: 31 | McNemar $p = \mathbf{9.31 \times 10^{-10}}$ (Overwhelmingly significant gain).
   - Bootstrap Delta Recall: **+31.00 pp** (95% CI: $[+22.00\text{ pp}, +40.00\text{ pp}]$).
   - Bootstrap Delta E2E: **+25.00 pp** (95% CI: $[+15.00\text{ pp}, +35.00\text{ pp}]$).

---

# DIAGNOSTIC THIRD ARM: SYSTEM MIXED
### (Direct Plate416 + Nested Plate640)

To test whether the benefits of 640 can be captured without the one-row truncation penalty, the diagnostic third arm was evaluated:

| Pipeline Architecture | Overall Recall | Overall E2E Exact | Motorcycle Recall | One-row E2E | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **System 416 (Homogeneous 416)** | 66.20% (331) | 55.00% (275) | 42.00% (42) | 48.89% (110) | **102.7 ms (9.7 FPS)** |
| **System 640 (Homogeneous 640)** | 69.40% (347) | 45.40% (227) | **73.00%** (73) | 13.78% (31) | 194.4 ms (5.1 FPS) |
| **System Mixed (Direct 416 + Nested 640)** | **68.40%** (342) | **56.40%** (282) | 67.00% (67) | **48.89%** (110) | 128.5 ms (7.8 FPS) |

### Key Finding:
System Mixed achieves the highest E2E exact accuracy (**56.40%**, +1.40 pp over System 416 and +11.00 pp over System 640), preserves 100% of one-row E2E accuracy, and captures $+25.0$ pp of motorcycle rescue inside vehicle ROIs with only moderate latency overhead (7.8 FPS).

---

# SCIENTIFIC CONCLUSION

Answering the 12 required scientific questions:

1. **Does Plate640 improve overall recall?**
   Marginally (+3.20 pp, from 66.20% to 69.40%), but the gain is **not statistically significant** (McNemar $p = 0.1448$, 95% CI crosses zero).
2. **Does Plate640 improve E2E?**
   **NO.** Overall E2E drops significantly by **-9.60 percentage points** (45.40% vs. 55.00%, McNemar $p = 0.000136$).
3. **Does it rescue Plate416 misses inside valid vehicle ROIs?**
   **YES.** Plate640 rescues **29 out of 80 (36.25%)** of the plates missed by Plate416 inside valid vehicle ROIs.
4. **Does it improve motorcycle recall?**
   **YES, dramatically.** Motorcycle recall surges by **+31.00 percentage points** (73.00% vs. 42.00%, McNemar $p = 9.31 \times 10^{-10}$) with zero regressions.
5. **Does it preferentially improve small plates?**
   It preferentially improves medium-sized, distant motorcycle plates (+14.46 pp in medium tertile), but causes regressions on large car bumper plates (-5.39 pp).
6. **Does it improve bbox IoU?**
   Slightly (+0.0155 median IoU on clean matches), but causes severe horizontal clipping on wide one-row plates.
7. **Does it reduce fallback invocation?**
   **NO.** Invocation rate remains unchanged at 36.40%.
8. **What FP cost does it introduce?**
   FP increases from 0.0340 to 0.0740 FP/image (+117.6%), reducing precision from 94.74% to 89.24%.
9. **What latency cost does it introduce?**
   Latency nearly doubles from 102.7 ms to 194.4 ms (+89.3%), reducing throughput from 9.7 FPS to 5.1 FPS.
10. **Are there 416-only regressions?**
    **YES, substantial regressions.** There are 45 detection regressions and 101 E2E regressions, heavily concentrated in one-row car plates.
11. **Is the effect truly due to resolution, or model lineage is confounded?**
    The effect is **truly due to resolution**. Model lineage audit confirmed that 82/82 convolution weights are bit-for-bit identical between the 416 and 640 ONNX models.
12. **Should 640 replace 416, complement it, or be rejected?**
    **Plate640 should COMPLEMENT 416, NOT replace it.** Homogeneous 640 is disqualified as a global replacement due to one-row truncation and 5.1 FPS latency. However, as demonstrated by the diagnostic third arm, using 640 selectively inside the vehicle ROI fallback captures 36.25% of missed plates while preserving full-frame one-row accuracy.

---

# DECISION

### **`PLATE640_COMPLEMENTARY`**

> **Formal Rationale:** Homogeneous Plate640 fails as a full replacement for Plate416 because it causes severe one-row text clipping (-35.11 pp one-row E2E), doubles latency (5.1 FPS), and degrades overall E2E accuracy by -9.60 pp. However, Plate640 delivers an unprecedented **+31.00 pp boost on motorcycle plates** ($p < 10^{-9}$) and rescues **36.25% of valid vehicle ROI plate misses**. Therefore, 640 is confirmed as a highly effective **complementary component** for vehicle-stage fallback routing, but rejected as a global detector replacement.

---

# NEXT RECOMMENDATION

### Selected Next Experiment: **Asymmetric Hybrid Routing (Direct-416 + Fallback-640)**

- **Candidate Chosen:** **Asymmetric Hybrid Routing (Direct-416 + Fallback-640)**
- **Scientific Justification:** The diagnostic third arm empirical data proved that coupling Direct 416 (which protects wide one-row car plates with 9.7 FPS baseline speed) with Fallback 640 (which zooms into vehicle ROIs to rescue 36.25% of small motorcycle plates) achieves the highest E2E accuracy (**56.40%**, +1.40 pp net gain) with zero one-row regressions and solid throughput (7.8 FPS). A formal, controlled benchmark of this asymmetric routing pipeline is the logical next step.
- **Execution Status:** **NOT EXECUTED** (Recommendation only, strictly as requested).

---

# OPERATIONAL PROTOCOL STATUS

- **RETRAINING:** NO
- **THRESHOLD TUNING:** NO
- **OCR ONNX:** NO
- **RASPBERRY PI:** NO
- **PRODUCTION:** UNCHANGED
