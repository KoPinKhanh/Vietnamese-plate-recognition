# PROJECT
Vietnam License Plate Recognition System (Pi 5 CPU Deployment Preparation)

# OBJECTIVE
Prepare the remaining laptop models for Raspberry Pi 5 CPU-only deployment by exporting:
1. Vehicle Detector: `yolo11s.pt` $\rightarrow$ `weights/deployment/pi_cpu/yolo11s_640_fp32.onnx`
2. OCR Recognizer: `weights/ocr/teachers/parseq_tiny_vn/best.pth` $\rightarrow$ `weights/deployment/pi_cpu/parseq_tiny_vn_fp32.onnx`

And rigorously verify mathematical and decision-level equivalence between PyTorch and ONNX Runtime CPU (`CPUExecutionProvider`) on deterministic held-out evaluation samples.

---

# ENVIRONMENT
- **Python**: `3.13.2`
- **PyTorch**: `2.6.0+cu124` (CUDA Available: True, GPU: NVIDIA GeForce RTX 3050 Laptop GPU)
- **Ultralytics**: `8.4.159`
- **ONNX**: `1.23.0`
- **ONNX Runtime**: `1.30.0` (CPUExecutionProvider used exclusively for equivalence verification and CPU benchmarks)
- **OpenCV**: `5.0.0`

---

# ORIGINAL MODEL HASHES
- **YOLO11s**: `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` (`yolo11s.pt`)
- **PARSeq-Tiny VNLP**: `a2044802fd2aea8a732eb0f1ff981ed892989fe378ac692be28b48a5e13fbbb7` (`weights/ocr/teachers/parseq_tiny_vn/best.pth`)

---

# VEHICLE YOLO11S EXPORT
- **Export Command / API**: Ultralytics Python API:
  ```python
  model = YOLO("yolo11s.pt")
  model.export(
      format="onnx",
      imgsz=640,
      batch=1,
      dynamic=False,
      half=False,
      opset=17,
      nms=False,
      simplify=True
  )
  ```
- **ONNX Path**: `weights/deployment/pi_cpu/yolo11s_640_fp32.onnx`
- **SHA256**: `01f8e6cc9a9f60e9a0b7491e8efd1c60ed19413e26f4fa9241576d31ce1a75ca`
- **File Size**: 38,051,782 bytes (36.29 MB)
- **Input**: `images` shape `[1, 3, 640, 640]`, dtype `float32`
- **Output**: `output0` shape `[1, 84, 8400]`, dtype `float32`
- **Opset**: 17
- **Model Check**: `onnx.checker.check_model()` PASSED.

---

# VEHICLE PYTORCH VS ONNX EQUIVALENCE
- **Validation Dataset**: Deterministic first 100 full-scene images from `reports/fresh_holdout_500_manifest.csv`.
- **Validation N**: 100 images
- **Execution Provider**: ONNX Runtime CPU (`CPUExecutionProvider`) vs PyTorch CPU.
- **Images with Identical Detection Count**:
  - Total Detections (all 80 COCO classes): **100 / 100 (100.0%)**
  - Relevant Vehicles (`car`, `motorcycle`, `bus`, `truck`): **100 / 100 (100.0%)**
- **Class Agreement Rate**:
  - Relevant Vehicles: **100.00% (98 / 98 matched predictions)**
  - All COCO Classes: **100.00% (162 / 162 matched predictions)**
- **BBox Localization IoU (PyTorch vs ONNX)**:
  - Mean Vehicle BBox IoU: **1.000000**
  - Minimum Vehicle BBox IoU: **0.999996**
  - Minimum Overall BBox IoU: **0.999990**
- **Confidence Absolute Differences**:
  - Mean Vehicle Confidence Abs Diff: **4.032437e-07**
  - Max Vehicle Confidence Abs Diff: **2.354383e-06**
- **Raw Logits Differences**:
  - Mean Absolute Diff: **2.127013e-06**
  - Max Absolute Diff: **7.659912e-03**
- **Failures / Anomalies**: **0** (Zero prediction discrepancies, zero NaNs, zero Infs).

---

# PARSEQ EXPORT AUDIT
- **Architecture**: PARSeq-Tiny VNLP (Permutation Autoregressive Sequence Model with ViT encoder and Transformer decoder, 12 layers, Tiny embedding dimension 192).
- **Forward Signature**:
  - Conceptual: `forward(images: Tensor[1, 3, 32, 128]) -> logits: Tensor[1, 26, 95]`
  - Maximum sequence length: 25 characters + 1 EOS token = 26 positions.
- **Input Preprocessing**:
  - Plate crop $\rightarrow$ Layout classification (`one_row`, `two_rows`, `motorcycle`)
  - Two-row rectification: Vertical stacking of upper and lower text halves if layout is two-row
  - Resize to fixed `32 × 128` (H × W)
  - Color: BGR to RGB
  - Normalization: ImageNet channel normalization `(img - mean) / std`
  - Input tensor is generated once and dispatched to both PyTorch and ONNX.
- **Tokenizer**:
  - Vocabulary size: 95 classes
  - Special tokens: EOS=0, BOS=95, PAD=96
  - Alphanumeric charset: `0-9`, `A-Z`, Vietnamese diacritics/punctuation.
- **Decode Path**:
  - Autoregressive loop of 26 steps where each token is predicted via greedy argmax and fed into the next step.
  - Followed by 1 iterative refinement step querying all positions simultaneously.
- **Exporter Investigation & Resolution**:
  - **Dynamo Exporter (`dynamo=True`)**: Failed due to missing `onnxscript` in the Python 3.13 environment.
  - **Direct Legacy Tracing (`dynamo=False`)**: Failed because PyTorch PARSeq implementation mutated token lists in Python loops and generated dynamic boolean `Where` operations (`/Where_17`) unsupported on ONNX Runtime CPU.
  - **Engineered Solution (`ParseqUnrolledWrapper`)**:
    - Created an explicit wrapper around `rec.model.model`.
    - Registered static upper-triangular masks as precomputed buffers (`tgt_mask`, `query_mask_refine`), completely eliminating boolean `Where` operators.
    - Unrolled the 26 autoregressive steps using tensor concatenation (`torch.cat([tgt_in, next_tok], dim=1)` and `p_i[:, -1:].argmax(dim=-1)`).
    - Preserved exact mathematical forward behavior (PyTorch wrapper diff vs original: $< 5.72 \times 10^{-6}$).
    - Exported cleanly using Torch legacy tracer with Opset 17 and constant folding.

---

# PARSEQ ONNX
- **Path**: `weights/deployment/pi_cpu/parseq_tiny_vn_fp32.onnx`
- **SHA256**: `369fcfbc06ff0ed3db962538b2186874e31067245563ef42e75eaa3eb9c8ae8a`
- **File Size**: 25,155,326 bytes (23.99 MB)
- **Input**: `image` shape `[1, 3, 32, 128]`, dtype `float32`
- **Output**: `logits` shape `[1, 26, 95]`, dtype `float32`
- **Opset**: 17
- **Model Check**: `onnx.checker.check_model()` PASSED.

---

# PARSEQ PYTORCH VS ONNX NUMERICAL EQUIVALENCE
- **Logits Output Shape**: `[1, 26, 95]` (identical)
- **Logits Mean Absolute Error**: **2.441744e-06**
- **Logits Max Absolute Error**: **7.057190e-05**
- **Argmax Token Agreement**: **100.00%** (2,600 / 2,600 tokens across all 100 validation samples)
- **Numerical Discrepancy Gate**: PASSED ($\ll 10^{-3}$ threshold).

---

# PARSEQ TEXT-LEVEL EQUIVALENCE
- **Validation Dataset**: 100 real plate crops from `reports/fresh_holdout_500_manifest.csv`
- **Layout Distribution**:
  - `one_row`: 47 samples
  - `two_rows`: 39 samples
  - `motorcycle`: 14 samples
- **Total Evaluated**: 100 crops
- **Decoded Text Exact Matches**: **100 / 100 (100.00%)**
- **Disagreements**: **0** (Zero text mismatches across all layouts and conditions).

---

# EXISTING PLATE ONNX AUDIT
Both pre-existing plate detector models were validated without re-exporting:

1. **Plate416 Detector**:
   - Path: `weights/best_license_plate_yolo11n_416.onnx`
   - SHA256: `bf07ca0458125f7e79a188b65fcf80c3aab4c2ca494ec1c3a7f94613e450e467`
   - File Size: 10,507,454 bytes (10.02 MB)
   - Input: `[1, 3, 416, 416]`, Output: `[1, 5, 3549]`
   - OpenCV DNN Load: **SUCCESS**

2. **Plate640 Detector**:
   - Path: `weights/best_license_plate_yolo11n_640.onnx`
   - SHA256: `82a89d27185136574396fd7c505729c7c31384b4268ea80bf24818031c5e1320`
   - File Size: 10,604,552 bytes (10.11 MB)
   - Input: `[1, 3, 640, 640]`, Output: `[1, 5, 8400]`
   - OpenCV DNN Load: **SUCCESS**

---

# LAPTOP CPU ONNX LATENCY
*Sanity baseline only measured on laptop CPU using `CPUExecutionProvider` (Warmup: 25, Timed Runs: 100):*

| Model | Input Shape | Mean (ms) | Median (ms) | P95 (ms) |
| :--- | :--- | :---: | :---: | :---: |
| **Vehicle YOLO11s** (`yolo11s_640_fp32.onnx`) | `[1, 3, 640, 640]` | 90.56 | 89.73 | 96.25 |
| **PARSeq-Tiny VNLP** (`parseq_tiny_vn_fp32.onnx`) | `[1, 3, 32, 128]` | 13.57 | 13.51 | 15.06 |

*(Note: Do not use laptop CPU latency to predict Raspberry Pi 5 FPS).*

---

# PI DEPLOYMENT ARTIFACTS
All required models and manifests are prepared and ready for Raspberry Pi 5:

1. `weights/deployment/pi_cpu/yolo11s_640_fp32.onnx` (Vehicle Detector, 36.29 MB)
2. `weights/deployment/pi_cpu/parseq_tiny_vn_fp32.onnx` (PARSeq OCR, 23.99 MB)
3. `weights/deployment/pi_cpu/deployment_manifest.json` (Platform metadata & checksums)
4. `weights/best_license_plate_yolo11n_416.onnx` (Direct Plate Detector, 10.02 MB)
5. `weights/best_license_plate_yolo11n_640.onnx` (Cascade Plate Detector, 10.11 MB)

---

# LIMITATIONS
- **Not Yet Tested on Raspberry Pi 5**: Benchmarks and executions have been verified on Windows Laptop CPU only.
- **No INT8**: All models are strictly full FP32 precision.
- **No Quantization**: Dynamic or static quantization was not applied.
- **No Retraining**: All models utilize strictly trained/frozen project weights.

---

# DECISION
**`PI_ONNX_EXPORT_READY`**

---

# NEXT STEP
**`COPY THE VERIFIED FP32 ONNX ARTIFACTS TO RASPBERRY PI 5 AND RUN INDIVIDUAL CPU BENCHMARKS BEFORE BUILDING THE FULL PIPELINE.`**
