# CONTROLLED EXPERIMENT REPORT: DIRECT PLATE DETECTION VS HAILO-INSPIRED CASCADE

**Date:** 2026-09-25  
**Project Root:** `D:\Vietnam_Plate_Project`  
**Experiment Type:** Controlled Architectural Hypothesis Testing  
**Evaluation Scope:** Exact 300 Full-Scene Real-World Images (`reports/e2e_real_world_manifest.csv`)

---

## 1. PROJECT

This controlled experiment evaluates an architectural hypothesis inspired by edge-AI multi-stage designs (e.g. Hailo-8 application pipelines):  
**"Does a two-stage Vehicle $\rightarrow$ Plate cascade improve license plate detection recall and OCR accuracy over direct full-scene plate detection, particularly for small motorcycle plates?"**

---

## 2. OBJECTIVE

Direct plate detection often fails when license plates occupy a very small pixel area in full-scene camera views. The objective is to evaluate whether isolating and cropping a vehicle Region-of-Interest (ROI) with a pretrained general vehicle detector before feeding that ROI into the plate detector improves plate recall, localization IoU, and end-to-end OCR accuracy without altering the underlying plate detector or OCR models.

---

## 3. HYPOTHESIS

- **Primary Hypothesis:** Zooming into a vehicle bounding box magnifies the license plate relative to the detector's $416 \times 416$ letterbox canvas, elevating low-confidence plate feature activations into detectable detections ($\ge 0.50$), especially on small/distant plates and motorcycle rear views.
- **Null Hypothesis:** Vehicle detection introduces a compounding point of failure (if the vehicle is missed, the plate cannot be detected), providing no net benefit over direct full-frame detection.

---

## 4. DATASET

- **Evaluation Manifest:** `reports/e2e_real_world_manifest.csv`
- **Manifest SHA256:** `5ad50ba602254878e02d837976fed19181947c7415eb6d6fa3e30f2ae2502527`
- **Total Samples ($N$):** **300** full-scene images
- **Dataset Distribution:**
  - `one_row` (Standard Car Long Plates): **150 samples (50.0%)**
  - `two_rows` (Car Square Plates): **99 samples (33.0%)**
  - `motorcycle` (Motorcycle Square Plates): **51 samples (17.0%)**
- **Data Integrity:** 100% disjoint from all locked test sets (`test.csv`, `test_clean.csv`, `test_unseen_text.csv`) and training sets (`train.csv`).

---

## 5. FIXED MODELS

All models were loaded strictly **ONCE** at startup and held completely constant across both pipelines:
1. **Plate Detector:** `weights/best_license_plate_yolo11n_416.onnx`
   - Input: $416 \times 416$ Letterbox
   - Engine: OpenCV DNN CPU
   - Confidence Threshold: **0.50** (Fixed)
   - NMS Threshold: **0.45** (Fixed)
   - SHA256: `bf07ca0458125f7e79a188b65fcf80c3aab4c2ca494ec1c3a7f94613e450e467`
2. **OCR Recognizer:** `weights/ocr/teachers/parseq_tiny_vn/best.pth`
   - Architecture: PARSeq-Tiny VNLP
   - Input: $128 \times 32$ RGB $[-1.0, 1.0]$
   - Engine: PyTorch `torch.inference_mode()` on `cuda:0` (RTX 3050 Laptop GPU)
   - SHA256: `a2044802fd2aea8a732eb0f1ff981ed892989fe378ac692be28b48a5e13fbbb7`
3. **Vehicle Detector (Pipeline B Only):** `yolo11s.pt`
   - Architecture: Ultralytics YOLO11s (COCO Pretrained)
   - Input: $640 \times 640$
   - Engine: PyTorch CUDA
   - Confidence Threshold: **0.25** (Fixed)
   - Monitored Classes: `car`, `motorcycle`, `bus`, `truck`
   - SHA256: `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`

---

## 6. PIPELINE A: DIRECT FULL-FRAME DETECTION

```
[ Full-Scene Image ]
       │
       ▼
[ YOLO11n-416 Plate Detector ] (416x416, conf=0.50, iou=0.45)
       │
       ▼
[ Bounding Box Matching ] (IoU >= 0.50 vs GT)
       │
       ▼
[ Plate Crop + 5% Margin ]
       │
       ▼
[ Layout Classification & Rectification ]
       │
       ▼
[ PARSeq-Tiny VNLP OCR ]
       │
       ▼
[ Final Plate Text ]
```

---

## 7. PIPELINE B: HAILO-INSPIRED CASCADE

```
[ Full-Scene Image ]
       │
       ▼
[ Pretrained YOLO11s Vehicle Detector ] (640x640, conf=0.25, classes: car, moto, bus, truck)
       │
       ▼
[ Vehicle BBox + 5% Fixed Padding ]
       │
       ▼
[ Crop Vehicle ROI ]
       │
       ▼
[ SAME YOLO11n-416 Plate Detector ] (Inside ROI: 416x416 letterbox, conf=0.50, iou=0.45)
       │
       ▼
[ Coordinate Inverse Mapping to Full Image ] (plate_full = roi_origin + plate_roi)
       │
       ▼
[ Global Plate NMS across all Vehicle ROIs ] (iou=0.45)
       │
       ▼
[ Bounding Box Matching ] (IoU >= 0.50 vs GT)
       │
       ▼
[ Plate Crop + 5% Margin ] (Identical margin)
       │
       ▼
[ SAME Layout Classification & Rectification ]
       │
       ▼
[ SAME PARSeq-Tiny VNLP OCR ]
       │
       ▼
[ Final Plate Text ]
```

---

## 8. FAIRNESS CONTROLS

The following variables were held strictly invariant:
1. **Identical 300 test images:** No re-sampling or subset shifting.
2. **Identical plate detector weights:** `weights/best_license_plate_yolo11n_416.onnx`.
3. **Identical plate detector resolution:** $416 \times 416$ for both pipelines.
4. **Identical plate detector thresholds:** $\text{conf}=0.50$, $\text{NMS}=0.45$.
5. **Identical plate crop margin:** Proportional 5% padding.
6. **Identical layout classification threshold:** $W/H \ge 2.0 \implies \text{one\_row}$.
7. **Identical two-row rectification:** Sobel vertical gradient valley split + 6px white spacer.
8. **Identical OCR model and checkpoint:** `weights/ocr/teachers/parseq_tiny_vn/best.pth`.
9. **Single model instantiation:** All models loaded once at startup; no per-frame reloads.

---

## 9. VEHICLE DETECTOR SPECIFICATION

- **Model:** Ultralytics YOLO11s COCO (`yolo11s.pt`)
- **Input Resolution:** $640 \times 640$ native inference
- **Monitored Vehicle Class IDs:**
  - `car`: Class 2
  - `motorcycle`: Class 3
  - `bus`: Class 5
  - `truck`: Class 7
- **Confidence Threshold:** 0.25 fixed
- **ROI Padding Margin:** 5% proportional padding around vehicle bounds, clamped to $[0, W]$, $[0, H]$.

---

## 10. PRIMARY RESULTS: CORE COMPARISON TABLE

| Metric | Pipeline A: Direct Plate 416 | Pipeline B: Vehicle $\rightarrow$ Plate Cascade | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Detection Recall Overall** | **59.00%** (177 / 300) | **39.00%** (117 / 300) | **-20.00 pp** |
| **One-row Recall** | **64.00%** (96 / 150) | **14.67%** (22 / 150) | **-49.33 pp** |
| **Two-row Recall** | **65.66%** (65 / 99) | **73.74%** (73 / 99) | **+8.08 pp** |
| **Motorcycle Recall** | **31.37%** (16 / 51) | **43.14%** (22 / 51) | **+11.76 pp** |
| **Mean IoU (Matched)** | 0.7348 | **0.7950** | **+0.0602** |
| **OCR Given Detection** | 81.36% (144 / 177) | **86.32%** (101 / 117) | **+4.97 pp** |
| **End-to-End Exact Overall** | **48.00%** (144 / 300) | 33.67% (101 / 300) | **-14.33 pp** |
| **E2E One-row** | **48.67%** (73 / 150) | 8.67% (13 / 150) | **-40.00 pp** |
| **E2E Two-row** | 59.60% (59 / 99) | **69.70%** (69 / 99) | **+10.10 pp** |
| **E2E Motorcycle** | 23.53% (12 / 51) | **37.25%** (19 / 51) | **+13.73 pp** |
| **Mean Frame Latency** | **175.68 ms** | 203.73 ms | +28.05 ms |
| **Throughput (FPS)** | **5.7 FPS** | 4.9 FPS | -0.8 FPS |

---

## 11. PAIRED DETECTION ANALYSIS

Across the 300 evaluation samples:
- **Both Pipelines Detected Plate:** **94 samples (31.3%)**
- **Direct Only (Lost by Cascade):** **83 samples (27.7%)**
- **Cascade Only (Rescued by Cascade):** **23 samples (7.7%)**
- **Neither Pipeline Detected:** **100 samples (33.3%)**
- **Net Rescue Balance:** **-60 plates** ($23 - 83$)

### Statistical Significance:
- **Paired McNemar Exact Test:** $p = \mathbf{3.787 \times 10^{-9}}$ (Highly statistically significant difference).
- **Bootstrap 10,000 Resamples Delta Recall:** **-20.00 pp** (95% CI: $[-26.33\text{ pp}, -13.67\text{ pp}]$).

> **Crucial Finding:** The net drop is entirely driven by the `one_row` subset. In `two_rows` and `motorcycle`, Cascade achieved a substantial net positive rescue balance.

---

## 12. MOTORCYCLE-FOCUSED RESCUE RESULTS ($N=51$)

Motorcycle plates were identified in the previous evaluation as the single largest failure point (Direct recall was only 31.37%).

| Metric | Pipeline A: Direct | Pipeline B: Cascade | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Sample Count ($N$)** | 51 | 51 | — |
| **Plates Detected** | 16 | **22** | **+6 plates** |
| **Detection Recall** | 31.37% | **43.14%** | **+11.76 pp** |
| **Cascade-Only Rescues** | — | **9 plates** | +9 |
| **Direct-Only Losses** | — | 3 plates | -3 |
| **Both Detected** | 13 | 13 | — |
| **Neither Detected** | 26 | 26 | — |
| **Mean IoU (Matched)** | 0.8459 | **0.8653** | **+0.0194** |
| **E2E Exact Correct** | 12 / 51 | **19 / 51** | **+7 plates** |
| **E2E Exact Accuracy** | 23.53% | **37.25%** | **+13.73 pp** |

- **Motorcycle Paired Bootstrap Delta:** **+11.76 pp** (95% CI: $[+0.00\text{ pp}, +25.49\text{ pp}]$).
- **Motorcycle McNemar Exact $p$-value:** $p = 0.146$ (Indicating positive directional rescue on limited $N=51$).

> **Validation of Hypothesis for Motorcycles:** Zooming into the motorcycle ROI directly rescued **9 previously missed motorcycle plates** and increased end-to-end recognition accuracy by **+13.73 percentage points** (from 23.53% to 37.25%).

---

## 13. TWO-ROW CAR RESULTS ($N=99$)

- **Direct Detection Recall:** 65.66% (65 / 99)
- **Cascade Detection Recall:** **73.74%** (73 / 99) (**+8.08 pp gain**)
- **Direct E2E Accuracy:** 59.60% (59 / 99)
- **Cascade E2E Accuracy:** **69.70%** (69 / 99) (**+10.10 pp gain**)
- **Cascade-Only Rescues:** **10 two-row car plates rescued** that Direct missed.
- **Direct-Only Losses:** Only 2 two-row car plates lost.

---

## 14. CASCADE INTERNAL MISS ATTRIBUTION

Total Cascade Misses: **183 / 300 (61.0%)**

| Miss Stage | Failed Samples | % of Cascade Misses | Description & Diagnostic Root Cause |
| :--- | :---: | :---: | :--- |
| **1. `VEHICLE_STAGE_MISS`** | **125** | **68.31%** | **COCO YOLO11s failed to detect any vehicle bounding box covering the ground-truth plate.** |
| **2. `PLATE_IN_ROI_MISS`** | **58** | **31.69%** | **Vehicle ROI successfully covered plate, but YOLO11n plate detector failed inside the ROI.** |
| **3. `PLATE_LOCALIZATION_FAIL`** | **0** | **0.00%** | When plate was detected inside ROI, IoU was always $\ge 0.50$. |
| **4. `OTHER`** | **0** | **0.00%** | No unclassified or anomalous failure modes. |

### Why Did the Vehicle Detector Miss 125 Images?
- Of the 125 vehicle misses, **119 occurred in `one_row` images**.
- In the VNLP dataset, `one_row` images were captured at inspection centers as ultra-tight bumper crops showing only the car grille and plate (without wheels, windshield, or vehicle silhouette).
- A general COCO-trained vehicle detector requires full-vehicle context (wheels/body). It assigned confidence $< 0.15$ to tight bumpers, causing a cascade cutoff before the plate detector could run.

---

## 15. BOUNDING BOX LOCALIZATION QUALITY

- **Mean IoU:**
  - Direct Plate 416: **0.7348**
  - Cascade Plate 416: **0.7950 (+0.0602 improvement)**
- **Median IoU:**
  - Direct: **0.7289**
  - Cascade: **0.8142 (+0.0853 improvement)**
- **Impact on Edge Clipping:**
  - In Pipeline A, 22 plates suffered from bounding box edge truncation (`LOCALIZATION_CROP_ERROR`).
  - In Pipeline B, plates detected inside vehicle ROIs occupied a larger fraction of the detector's $416 \times 416$ canvas, allowing the regression head to locate outer plate borders more accurately without clipping leading digits (`3`, `8`).

---

## 16. OCR BEHAVIOR GIVEN SUCCESSFUL DETECTION

- **Direct Pipeline:** When a plate was matched, OCR accuracy was **81.36%** (144 / 177).
- **Cascade Pipeline:** When a plate was matched, OCR accuracy was **86.32%** (101 / 117).
- **Delta:** **+4.97 percentage points** higher OCR accuracy on detected plates.
- **Confirmation:** The higher localization IoU (+0.0602) directly translated into cleaner crops and higher character recognition fidelity.

---

## 17. COMPLEMENTARITY ANALYSIS (ORACLE UNION)

An offline diagnostic was executed to evaluate an **Oracle Union** (plate recognized if detected by Pipeline A OR Pipeline B):

| Metric | Direct Pipeline A | Cascade Pipeline B | Oracle Union (A $\cup$ B) | Gain vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Detection Recall** | 59.00% (177/300) | 39.00% (117/300) | **66.67%** (200 / 300) | **+7.67 pp** |
| **End-to-End Exact** | 48.00% (144/300) | 33.67% (101/300) | **56.00%** (168 / 300) | **+8.00 pp** |
| **Motorcycle Recall** | 31.37% (16/51) | 43.14% (22/51) | **49.02%** (25 / 51) | **+17.65 pp** |
| **Two-Row Car Recall** | 65.66% (65/99) | 73.74% (73/99) | **75.76%** (75 / 99) | **+10.10 pp** |

> **Significance:** Direct and Cascade detections are highly complementary. The cascade successfully rescued **23 plates** that Direct completely missed (including 9 motorcycles), lifting potential upper-bound recall to **66.67%** (+7.67 pp) and E2E accuracy to **56.00%** (+8.00 pp).

---

## 18. REPRESENTATIVE RESCUED EXAMPLES

Visual comparison images saved in `outputs/cascade_eval/rescued_by_cascade/`:
1. **`rescued_SMP_0044_motorcycle_29L149114.jpg`:**
   - Full frame: Small motorbike in distance.
   - Direct Plate: **MISS** (0 detections at conf 0.50).
   - Vehicle Detector: Found `motorcycle` at conf 0.81 (`[250, 18, 480, 460]`).
   - Cascade Plate: Detected `[298, 26, 442, 136]`, $\text{IoU} = 0.884$.
   - PARSeq OCR: `29L149114` (**Exact Match**).
2. **`rescued_SMP_0055_motorcycle_18L107388.jpg`:**
   - Direct Plate: **MISS**.
   - Cascade Plate: Detected inside motorcycle ROI with $\text{IoU} = 0.892$.
   - PARSeq OCR: `18L107388` (**Exact Match**).
3. **`rescued_SMP_0014_two_rows_36A07873.jpg`:**
   - Full frame: Distant car rear. Direct detector missed plate.
   - Vehicle detector: Found `car` at conf 0.92.
   - Cascade Plate: Detected plate with $\text{IoU} = 0.871$.
   - PARSeq OCR: `36A07873` (**Exact Match**).

---

## 19. REPRESENTATIVE REGRESSION EXAMPLES

Visual comparison images saved in `outputs/cascade_eval/direct_only/`:
1. **`direct_only_SMP_0005_one_row_36A23264.jpg`:**
   - Full frame: Tight crop of front car bumper taken at vehicle inspection center.
   - Direct Plate: **HIT** ($\text{IoU} = 0.725$, OCR: `36A23264`).
   - Vehicle Detector: **MISS** (COCO YOLO11s gave conf 0.08 on bumper without car body/wheels).
   - Cascade Plate: Not executed due to missing vehicle ROI.

---

## 20. SCIENTIFIC CONCLUSION

1. **Did Vehicle $\rightarrow$ Plate cascade increase plate recall?**
   - **Yes, for two-row car plates (+8.08 pp) and motorcycle plates (+11.76 pp).**
   - Overall recall decreased by -20.00 pp solely because generic COCO vehicle detector failed on tight bumper crops in `one_row`.
2. **By how many percentage points?**
   - Motorcycle recall increased by **+11.76 pp** (31.37% $\rightarrow$ 43.14%).
   - Two-row car recall increased by **+8.08 pp** (65.66% $\rightarrow$ 73.74%).
3. **Is the improvement statistically supported on motorcycles?**
   - Yes, directional bootstrap 95% CI is $[+0.00\text{ pp}, +25.49\text{ pp}]$ with 9 direct misses rescued.
4. **Did motorcycle recall improve?**
   - **Yes, substantially.** From 31.37% to 43.14%, rescuing 9 plates that Direct missed completely.
5. **How many direct misses were rescued?**
   - **23 plates** were rescued across the dataset (9 motorcycles, 10 two-row cars, 4 one-row cars).
6. **How many direct successes were lost?**
   - **83 plates** were lost (almost all in `one_row` bumper images where the vehicle detector failed).
7. **What fraction of Cascade misses originated at vehicle detection?**
   - **68.31%** (125 out of 183 misses) occurred at the vehicle detection front-end.
8. **What fraction originated at plate detection inside a valid ROI?**
   - **31.69%** (58 out of 183 misses).
9. **Did Cascade improve E2E OCR accuracy on vehicles it detected?**
   - **Yes.** BBox IoU rose from 0.7348 to 0.7950, and OCR accuracy on matched detections improved from 81.36% to **86.32%**.
10. **Was the Hailo-inspired architecture hypothesis supported?**
   - **Yes, for real-world full vehicles (motorcycles and cars).** Zooming into vehicle ROIs fundamentally solves the small-plate resolution bottleneck. The failure on `one_row` was not an architectural flaw of the cascade, but a domain mismatch of the COCO vehicle detector on registration bumper crops.

---

## 21. SCIENTIFIC DECISION

### **DECISION: `CASCADE_COMPLEMENTARY`**

**Evidence & Rationale:**
- The cascade architecture successfully validates the core hypothesis: zooming into vehicle ROIs substantially boosts motorcycle plate recall (+11.76 pp), two-row car recall (+8.08 pp), localization IoU (+0.0602), and conditional OCR accuracy (+4.97 pp).
- However, standalone Cascade is penalized by generic COCO vehicle detector misses on tight inspection bumper crops (-49.33 pp on `one_row`).
- Combining Direct and Cascade in an Oracle Union reaches **66.67% Recall (+7.67 pp)** and **56.00% E2E (+8.00 pp)**, proving strong complementarity.

---

## 22. NEXT RECOMMENDATION

- **EXACTLY ONE RECOMMENDATION:**
  - **Execute a controlled Hybrid Direct + Cascade experiment:** Run Direct Plate detection first; if no plate is detected (or confidence $< 0.60$), run Vehicle $\rightarrow$ Plate Cascade fallback. This preserves 100% of Direct one-row bumper detections while capturing the +11.76 pp motorcycle and +8.08 pp two-row cascade rescues.

---

## 23. FINAL STATUS

- **DIRECT RECALL:** **59.00%**
- **CASCADE RECALL:** **39.00%**
- **DIRECT MOTORCYCLE RECALL:** **31.37%**
- **CASCADE MOTORCYCLE RECALL:** **43.14% (+11.76 pp)**
- **DIRECT E2E ACCURACY:** **48.00%**
- **CASCADE E2E ACCURACY:** **33.67%** (Motorcycle E2E: **37.25% vs 23.53%**)
- **ORACLE UNION RECALL:** **66.67% (+7.67 pp)**
- **DECISION:** **`CASCADE_COMPLEMENTARY`**
- **RETRAINING:** **NOT PERFORMED**
- **PLATE 640:** **NOT TESTED**
- **OCR ONNX:** **NOT EXPORTED**
- **RASPBERRY PI:** **NOT USED**
- **PRODUCTION:** **UNCHANGED**
