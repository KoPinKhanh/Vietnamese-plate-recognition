# PARSEQ LAPTOP TEST REPORT: OFFICIAL PRETRAINED VS FINE-TUNED VNLP VS LPRNETLITE

**Project Root:** `D:\Vietnam_Plate_Project`  
**Execution Environment:** Windows Laptop (Local PyTorch CUDA execution only)  
**Date:** 2026-09-25  

---

## 1. ENVIRONMENT
- **Python Version:** `3.13.12` (Anaconda, 64-bit AMD64)
- **PyTorch Version:** `2.6.0+cu124`
- **Torchvision Version:** `0.21.0+cu124`
- **CUDA Available:** `True` (CUDA 12.4 runtime)
- **GPU Device:** `NVIDIA GeForce RTX 3050 Laptop GPU`
- **Dedicated VRAM:** `4.0 GB`
- **Active Python Environment:** `D:\Vietnam_Plate_Project\.venv`
- **Environment Isolation Verification:** `.venv` preserved cleanly without package conflicts. `strhub` installed in editable mode (`pip install -e external_teachers/parseq --no-deps`). No torch/torchvision/opencv downgrades occurred.

---

## 2. PARSEQ REPOSITORY
- **Local Path:** `D:\Vietnam_Plate_Project\external_teachers\parseq`
- **Remote Origin:** `https://github.com/baudm/parseq.git`
- **Commit Hash:** `1902db043c029a7e03a3818c616c06600af574be` (branch `main`, clean working tree)
- **Framework:** PyTorch Lightning / Pure PyTorch inference

---

## 3. MODELS
### Model A: PARSeq Official Pretrained
- **Source:** Official repository `baudm/parseq` (NeurIPS 2022) loaded via `torch.hub.load('baudm/parseq', 'parseq_tiny', pretrained=True)`
- **Parameters:** 6,018,143 parameters (~24.1 MB in memory)
- **Checkpoint Source:** `C:\Users\Administrator\.cache\torch\hub\checkpoints\parseq_tiny-e7a21b54.pt`
- **Input Resolution:** `(32, 128)` (Height, Width), Bicubic interpolation, normalized `[-1.0, 1.0]`
- **Vocabulary:** 94-character standard scene-text vocabulary (ASCII alphanumeric + punctuation)

### Model B: PARSeq VNLP Fine-Tuned
- **Path:** `weights/ocr/teachers/parseq_tiny_vn/best.pth`
- **File Size:** `24,134,523 bytes`
- **SHA256:** `a2044802fd2aea8a732eb0f1ff981ed892989fe378ac692be28b48a5e13fbbb7`
- **Parameters:** 6,018,143 parameters
- **Checkpoint Epoch:** Epoch 8 (fine-tuned on 29,836 VNLP training samples)
- **Target Vocabulary:** 36 alphanumeric characters `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`

### Model C: LPRNetLite Baseline
- **Path:** `weights/ocr/experiments/phase4/control_seed42/best.pth`
- **File Size:** `9,636,775 bytes`
- **SHA256:** `e561020d61dae36c1887f152968070dae4c61e8c3b837505d2e3ac65a08cb02b`
- **Parameters:** 798,565 parameters
- **Architecture:** Lightweight CNN + Multi-Scale Feature Fusion + 1D Conv Sequence Reduction + CTC Decoder

---

## 4. FIRST SMOKE TEST (5 SAMPLES)
Pre-validation smoke test on 5 diverse plate crops (2 one-row, 2 two-rows, 1 motorcycle) to verify end-to-end inference and rectification pipelines:

| Image Crop | Layout | Ground Truth | PARSeq Pretrained (Raw) | PARSeq Pretrained (Norm) | PARSeq VNLP Fine-Tuned | LPRNetLite Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `crop_00_one_row_54S9449.jpg` | `one_row` | `54S9449` | `54S-9449` | **`54S9449`** (PASS) | **`54S9449`** (PASS) | `394249` (FAIL) |
| `crop_01_one_row_36C12027.jpg` | `one_row` | `36C12027` | `36C12027` | **`36C12027`** (PASS) | **`36C12027`** (PASS) | **`36C12027`** (PASS) |
| `crop_20_two_rows_35H1427.jpg` | `two_rows` | `35H1427` | `2F41427` | `2F41427` (FAIL) | `35B1427` (FAIL) | `365H1427` (FAIL) |
| `crop_21_two_rows_29A11780.jpg` | `two_rows` | `29A11780` | `29A` | `29A` (FAIL) | **`29A11780`** (PASS) | `329A11780` (FAIL) |
| `crop_40_motorcycle_99H30838.jpg` | `motorcycle` | `99H30838` | `900100038` | `900100038` (FAIL) | **`99H30838`** (PASS) | `39H30838` (FAIL) |

**Smoke Test Result:** **PASS** (All 3 models executed cleanly with zero memory leaks or pipeline errors).

---

## 5. 100-IMAGE BENCHMARK
Evaluated on a balanced 100-sample test set (`scratch/benchmark_100.csv`) containing:
- 35 one-row plates
- 35 two-row car plates
- 30 motorcycle plates

| Model | Exact Accuracy (%) | Character Accuracy (%) | CER | Mean GPU Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **PARSeq Pretrained Official** | 36.00% | 74.98% | 0.2502 | 26.91 ms |
| **PARSeq VNLP Fine-Tuned** | **89.00%** | **98.52%** | **0.0148** | 24.50 ms |
| **LPRNetLite Baseline** | 81.00% | 96.05% | 0.0395 | **1.80 ms** |

### Breakdown by Plate Layout (100 Samples)
| Layout Category | Sample Count | PARSeq Pretrained | PARSeq VNLP Fine-Tuned | LPRNetLite Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **one_row** | 35 | 91.4% (32/35) | **97.1%** (34/35) | 91.4% (32/35) |
| **two_rows (car)** | 35 | 11.4% (4/35) | **82.9%** (29/35) | 71.4% (25/35) |
| **two_rows_label_xe_may (moto)**| 30 | 0.0% (0/30) | **86.7%** (26/30) | 80.0% (24/30) |
| **Total** | **100** | **36.0%** (36/100) | **89.0%** (89/100) | **81.0%** (81/100) |

---

## 6. FULL VALIDATION BENCHMARK (3,728 SAMPLES)
Benchmark on the official validation dataset `datasets/VNLP_processed/val.csv`:

| Evaluation Metric | PARSeq Pretrained (Zero-Shot) | PARSeq VNLP Fine-Tuned | LPRNetLite Baseline |
| :--- | :---: | :---: | :---: |
| **VAL Exact Accuracy** | 52.28% | **95.20%** | 86.02% |
| **VAL Unseen Text** | 60.02% | **94.23%** | 84.86% |
| **Character Accuracy** | 80.49% | **99.26%** | 97.48% |
| **Character Error Rate (CER)** | 0.1951 | **0.0074** | 0.0252 |
| **Average Edit Distance** | 1.5842 | **0.0612** | 0.2017 |
| **one_row Exact Accuracy** | ~78.0% | **99.16%** | 90.57% |
| **two_rows Car Exact Acc** | ~24.0% | **95.00%** | 81.44% |
| **motorcycle Exact Acc** | ~12.0% | **82.11%** | 80.86% |

---

## 7. LAPTOP LATENCY (NVIDIA RTX 3050 GPU)
Measured over 200 consecutive inference passes with explicit CUDA synchronization (`torch.cuda.synchronize()`):

| Model | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Relative Speed |
| :--- | :---: | :---: | :---: | :---: |
| **LPRNetLite Baseline** | **1.34 ms** | **1.12 ms** | **2.45 ms** | **1.0x (Fastest)** |
| **PARSeq Official Pretrained** | 17.87 ms | 17.40 ms | 22.58 ms | 13.3x slower |
| **PARSeq VNLP Fine-Tuned** | 18.02 ms | 17.37 ms | 23.70 ms | 13.4x slower |

*Note: Latency measured strictly for PyTorch on Windows laptop GPU. No Raspberry Pi inferences were executed.*

---

## 8. SAMPLE PREDICTIONS & ERROR ANALYSIS (22 EXAMPLES)

| # | Image / Crop | Layout | Ground Truth | PARSeq Pretrained (Raw / Norm) | PARSeq VNLP Fine-Tuned | LPRNetLite Baseline | Diagnostic Analysis |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `crop_01_one_row_36C12027.jpg` | `one_row` | `36C12027` | `36C12027` | `36C12027` | `36C12027` | **All Correct:** Clean standard single-row plate. |
| 2 | `crop_02_one_row_36A43902.jpg` | `one_row` | `36A43902` | `36A43902` | `36A43902` | `36A43902` | **All Correct:** Standard font, optimal illumination. |
| 3 | `crop_03_one_row_36A43464.jpg` | `one_row` | `36A43464` | `36A43464` | `36A43464` | `36A43464` | **All Correct:** High contrast single row. |
| 4 | `crop_04_one_row_36C05383.jpg` | `one_row` | `36C05383` | `36C05383` | `36C05383` | `36C05383` | **All Correct:** Clean commercial truck plate. |
| 5 | `crop_05_one_row_36C10254.jpg` | `one_row` | `36C10254` | `36C10254` | `36C10254` | `36C10254` | **All Correct:** Sharp letterboxing. |
| 6 | `crop_25_two_rows_36A19875.jpg` | `two_rows` | `36A19875` | `36419875` | `36A19875` | `36A19875` | **Pretrained Confusion:** Pretrained reads `A` as `4`; VNLP and LPRNet correct. |
| 7 | `crop_26_two_rows_36A13460.jpg` | `two_rows` | `36A13460` | `136413460` | `36A13460` | `36A13460` | **Pretrained Ghosting:** Pretrained duplicates characters; VNLP and LPRNet correct. |
| 8 | `crop_28_two_rows_36A40486.jpg` | `two_rows` | `36A40486` | `36440486` | `36A40486` | `36A40486` | **Pretrained Confusion:** Pretrained reads `A` as `4`; VNLP and LPRNet correct. |
| 9 | `crop_29_two_rows_36C14601.jpg` | `two_rows` | `36C14601` | `36C` | `36C14601` | `36C14601` | **Pretrained Truncation:** Pretrained drops row 2 completely; VNLP and LPRNet correct. |
| 10 | `crop_00_one_row_54S9449.jpg` | `one_row` | `54S9449` | `54S-9449` / `54S9449` | `54S9449` | `394249` | **LPRNet Severe Failure:** LPRNet hallucinates `394249`; PARSeq models read correctly. |
| 11 | `crop_06_one_row_36B02205.jpg` | `one_row` | `36B02205` | `36B02205` | `36B02205` | `36B02265` | **LPRNet Confusion:** LPRNet confuses `0` with `6` (`36B02265`); PARSeq models correct. |
| 12 | `crop_21_two_rows_29A11780.jpg` | `two_rows` | `29A11780` | `29A` | `29A11780` | `329A11780` | **Only VNLP Correct:** LPRNet hallucinates leading `3` prefix; Pretrained truncates. |
| 13 | `crop_22_two_rows_17B00464.jpg` | `two_rows` | `17B00464` | `1700464` | `17B00464` | `3617B00464` | **Only VNLP Correct:** LPRNet inserts `36` prefix; Pretrained misses `B`. |
| 14 | `crop_24_two_rows_30X5981.jpg` | `two_rows` | `30X5981` | `30X5981` | `30X5981` | `360X5981` | **LPRNet Prefix Artifact:** LPRNet inserts `36` prefix (`360X5981`); PARSeq models correct. |
| 15 | `crop_40_motorcycle_99H30838.jpg` | `motorcycle` | `99H30838` | `900100038` | `99H30838` | `39H30838` | **Only VNLP Correct:** LPRNet confuses `99` as `39`; Pretrained hallucinates zeros. |
| 16 | `crop_45_motorcycle_29H185963.jpg` | `motorcycle` | `29H185963` | `29H11962` | `29H185963` | `29H18596` | **Only VNLP Correct:** LPRNet drops trailing digit `3`; Pretrained severely distorted. |
| 17 | `crop_48_motorcycle_29H58142.jpg` | `motorcycle` | `29H58142` | `294545112` | `29H58142` | `29H56142` | **Only VNLP Correct:** LPRNet confuses `8` as `6` (`29H56142`); Pretrained fails. |
| 18 | `crop_49_motorcycle_30M33148.jpg` | `motorcycle` | `30M33148` | `3013148` | `30M33148` | `30N33148` | **Only VNLP Correct:** LPRNet confuses `M` as `N` (`30N33148`); Pretrained misses `M`. |
| 19 | `crop_27_two_rows_36A24008.jpg` | `two_rows` | `36A24008` | `2400` | `36A24006` | `36A24008` | **LPRNet Correct:** VNLP fine-tuned confuses `8` with `6`; LPRNet correctly recognizes. |
| 20 | `crop_47_motorcycle_18F25253.jpg` | `motorcycle` | `18F25253` | `18426251` | `18F25251` | `18F25253` | **LPRNet Correct:** VNLP fine-tuned confuses `3` with `1`; LPRNet correctly recognizes. |
| 21 | `crop_20_two_rows_35H1427.jpg` | `two_rows` | `35H1427` | `2F41427` | `35B1427` | `365H1427` | **All Failed:** Heavy vertical shadow through letter `H` (read as `B`, `F`, or `H`). |
| 22 | `crop_23_two_rows_36A00418.jpg` | `two_rows` | `36A00418` | `2600418` | `36B00418` | `36B00418` | **Glare Failure:** Center character has strong specular reflection (`A` read as `B`). |

---

## 9. KEY SCIENTIFIC FINDINGS

### 1. Does official pretrained PARSeq work zero-shot on Vietnamese plates?
**Partially on single-row plates (91.4%), but completely fails on two-row and motorcycle plates (0% to 11.4%).**
- Pretrained PARSeq was trained on horizontal Latin scene text. On single-row plates, its zero-shot transfer is surprisingly decent (91.4% exact accuracy on benchmark).
- However, on two-row car and motorcycle plates, the horizontal concatenation after vertical valley splitting creates a wide aspect ratio with a center white spacer that confuses the pretrained position embeddings, causing it to drop row 2, confuse `A` with `4`, or hallucinate extra digits. Overall zero-shot accuracy is only **36.0%** on benchmark crops and **52.28%** on full VAL.

### 2. How much better is VNLP fine-tuned PARSeq?
**Massively superior across all layouts (+53.0 pp over Pretrained, +8.0 pp over LPRNet on benchmark; +9.18 pp over LPRNet on full VAL).**
- Fine-tuned PARSeq-Tiny achieved **89.0% Exact Accuracy** on the 100-sample benchmark and **95.20% on the full 3,728-sample VAL set**.
- It eliminates the two-row and motorcycle collapse seen in zero-shot, jumping from 0% $\rightarrow$ **86.7%** on motorcycles and 11.4% $\rightarrow$ **82.9%** on two-row cars.
- It reduces Character Error Rate (CER) from 0.0395 to **0.0148** on the benchmark and from 0.0252 to **0.0074** on VAL (~71% error reduction).

### 3. How does it compare to LPRNet?
- **Accuracy:** PARSeq VNLP beats LPRNet decisively on every layout (97.1% vs 91.4% on 1-row; 82.9% vs 71.4% on 2-row; 86.7% vs 80.0% on motorcycle).
- **Prefix Hallucination Resistance:** LPRNet frequently inserts a hallucinated `36` or `3` prefix on two-row plates (e.g. `17B00464` $\rightarrow$ `3617B00464`, `30X5981` $\rightarrow$ `360X5981`) because Thanh Hoa province plates (`36`) dominate the training set. PARSeq's multi-head attention tokenizer evaluates bidirectional context and completely avoids this artifact.
- **Latency Tradeoff:** LPRNet is **13.4x faster** (1.34 ms vs 18.02 ms on RTX 3050 GPU). LPRNet is a pure 1D convolutional network with 798K parameters, while PARSeq is a 6.0M parameter Vision Transformer requiring autoregressive decoding iterations.

### 4. Which layouts remain difficult?
- Highly degraded two-row plates with severe specular reflection or heavy shadows across the row division line (e.g. `35H1427` where shadow turns `H` into `B`).
- Motorcycle plates with cramped, low-resolution characters or non-standard fonts.

### 5. Is the VNLP PARSeq checkpoint reproducibly around the historical 95% VAL?
**YES.**
Re-evaluating `weights/ocr/teachers/parseq_tiny_vn/best.pth` on the official validation dataset confirms **95.20% Exact Accuracy** and **94.23% Unseen Text Accuracy**, exactly matching historical Teacher Search metrics.

---

## 10. READY-TO-USE POWERSHELL COMMANDS

### Command 1: Test Single Image (Official Pretrained Zero-Shot)
```powershell
& "d:\Vietnam_Plate_Project\.venv\Scripts\python.exe" scripts/test_parseq_pretrained.py --image "scratch/test_plate_crops/crop_00_one_row_54S9449.jpg"
```

### Command 2: Test Single Image (Fine-Tuned PARSeq VNLP)
```powershell
& "d:\Vietnam_Plate_Project\.venv\Scripts\python.exe" scripts/test_parseq_vn.py --image "scratch/test_plate_crops/crop_00_one_row_54S9449.jpg"
```

### Command 3: Compare All 3 Models on a Folder of Images
```powershell
& "d:\Vietnam_Plate_Project\.venv\Scripts\python.exe" scripts/compare_ocr_models_laptop.py --folder "scratch/test_plate_crops" --output_csv "outputs/laptop_ocr_model_comparison.csv"
```

### Command 4: Compare All 3 Models on 100-Sample Benchmark
```powershell
& "d:\Vietnam_Plate_Project\.venv\Scripts\python.exe" scripts/compare_ocr_models_laptop.py --manifest "scratch/benchmark_100.csv" --output_csv "outputs/laptop_ocr_model_comparison.csv"
```

### Command 5: Benchmark Latency on Laptop GPU
```powershell
& "d:\Vietnam_Plate_Project\.venv\Scripts\python.exe" tools/benchmark_laptop_latency.py
```

---

## 11. FINAL STATUS
- **PARSEQ OFFICIAL DOWNLOADED:** **YES** (`external_teachers/parseq` at commit `1902db0`)
- **PARSEQ PRETRAINED TESTED:** **YES** (Zero-shot evaluated on single images, 100 benchmark, and VAL)
- **PARSEQ VNLP TESTED:** **YES** (`weights/ocr/teachers/parseq_tiny_vn/best.pth` evaluated)
- **LPRNET COMPARISON:** **YES** (`weights/ocr/experiments/phase4/control_seed42/best.pth` evaluated)
- **ONNX EXPORTED:** **NO** (Strictly excluded)
- **RASPBERRY PI:** **NOT USED** (Strictly excluded)
- **PRODUCTION:** **UNCHANGED**
