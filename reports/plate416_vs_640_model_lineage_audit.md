# PLATE 416 VS 640 MODEL LINEAGE AUDIT

- **Project Root:** `D:\Vietnam_Plate_Project`
- **Audit Date:** September 25, 2026
- **Investigation Objective:** Audit the origin, weights, architecture, and export parameters of the 416 and 640 license plate detector ONNX artifacts to classify their model lineage.

---

## 1. Candidate Artifacts Inspected

### A. Plate Detector 416 ONNX
- **File Path:** `weights/best_license_plate_yolo11n_416.onnx`
- **File Size:** 10,507,454 bytes (~10.02 MB)
- **SHA256:** `bf07ca0458125f7e79a188b65fcf80c3aab4c2ca494ec1c3a7f94613e450e467`
- **ONNX IR Version:** 7 (Producer: pytorch 2.6.0)
- **Input Shape:** `[1, 3, 416, 416]` (name: `images`)
- **Output Shape:** `[1, 5, 3549]` (name: `output0`, where $3549 = 52 \times 52 + 26 \times 26 + 13 \times 13$, format: `[cx, cy, w, h, conf]`)
- **Metadata Description:** `Ultralytics YOLO11n model trained on D:\Vietnam_Plate_Project\configs\license_plate.yaml`
- **Classes:** 1 (`{0: 'license_plate'}`)
- **Export Date:** 2026-09-23T01:07:10.568039+07:00

### B. Plate Detector 640 ONNX
- **File Path:** `weights/best_license_plate_yolo11n_640.onnx`
- **File Size:** 10,604,552 bytes (~10.11 MB)
- **SHA256:** `82a89d27185136574396fd7c505729c7c31384b4268ea80bf24818031c5e1320`
- **ONNX IR Version:** 7 (Producer: pytorch 2.6.0)
- **Input Shape:** `[1, 3, 640, 640]` (name: `images`)
- **Output Shape:** `[1, 5, 8400]` (name: `output0`, where $8400 = 80 \times 80 + 40 \times 40 + 20 \times 20$, format: `[cx, cy, w, h, conf]`)
- **Metadata Description:** `Ultralytics YOLO11n model trained on D:\Vietnam_Plate_Project\configs\license_plate.yaml`
- **Classes:** 1 (`{0: 'license_plate'}`)
- **Export Date:** 2026-09-23T00:52:18.393744+07:00

### C. Common PyTorch Source Checkpoint
- **File Path:** `weights/best_license_plate_yolo11n.pt`
- **Training Source Path:** `outputs/train/vietnam_plate_yolo11n_fast/weights/best.pt`
- **File Size:** 5,469,971 bytes
- **SHA256:** `63a98c42e586e7a06fcc9cc3be4c9b10fce4adc6b82647779e7825f0d93bc376` (Bit-for-bit identical)
- **Training Config:** `outputs/train/vietnam_plate_yolo11n_fast/args.yaml`
  - Base Model: `yolo11n.pt`
  - Training Data: `D:\Vietnam_Plate_Project\configs\license_plate.yaml`
  - Training Imgsz: 640
  - Epochs: 30, Batch: 16, Optimizer: auto, Seed: 0

---

## 2. Weight Tensor Comparison

Tensor-by-tensor comparison between `best_license_plate_yolo11n_416.onnx` and `best_license_plate_yolo11n_640.onnx`:
- **Total Initializer Tensors:** 192 in both models.
- **Convolution Weight Tensors:** Exactly 82 tensors.
- **Matched Convolution Weights:** **82 / 82 (100.0% bit-for-bit identical)**.
- **Differing Tensors:** Exactly 7 tensors, all corresponding to resolution-dependent anchor grids and position constants (`(1, 2, 3549)` for 416 vs `(1, 2, 8400)` for 640).
- **Core Weights Equality:** **CONFIRMED IDENTICAL**.

---

## 3. Model Lineage Classification

Under Section 8 of the experiment protocol, this model pairing is classified as:

### **SAME_WEIGHTS_DIFFERENT_EXPORT_RESOLUTION**

> **Explanation:** Both models are derived from the exact same trained PyTorch checkpoint (`weights/best_license_plate_yolo11n.pt`), sharing identical learned parameters, convolution feature extractors, and detection head weights. The only difference is the input resolution and corresponding spatial anchor grid ($416 \times 416$ with 3,549 grid cells vs. $640 \times 640$ with 8,400 grid cells).
> 
> Therefore, this experiment constitutes a **pure, unconfounded input-resolution ablation**.
