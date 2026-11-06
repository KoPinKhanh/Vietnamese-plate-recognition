# Dataset Lineage & Reproducibility Guide

Due to storage limitations and Git repository best practices, full-scale training image archives are not tracked directly in this repository. Three self-contained sample images are bundled in [`samples/`](./samples/) for immediate out-of-the-box testing (Plug & Play).

---

## 🌐 Dataset Inventory & Public Sources

| Dataset Source | Access Link | Scope & Role |
| :--- | :--- | :--- |
| **Roboflow VN License Plate** | [Roboflow Universe (8,357 images)](https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0) | Bounding box plate detection training data for YOLO11 (Train: 5,845; Val: 1,680; Test: 832 labeled images). |
| **VNLP (Vietnamese License Plate)** | [Google Drive (Published by VNLP Authors)](https://drive.google.com/drive/folders/19OnlE0gFXEMPyngxRZ980d4z0fGQ-C7T?usp=sharing) | Primary OCR training and evaluation benchmark (37,297 samples: 19,086 1-row rectangular plates, 12,618 2-row square car plates, 5,593 motorcycle plates). Source reference: `datasets/VNLP-main/VNLP-main/readme.md`. |
| **PBL4 Deep Learning** | [Source GitHub Repository](https://github.com/lephamcong/PBL4_Deep-Learning) | Supplementary OCR domain data; see [Source Analysis & Deduplication Report](./reports/pbl4_dataset_report.md). |
| **Vietnamese License Plate Generator** | [Source GitHub Repository](https://github.com/NNDam/Vietnamese-License-Plate-Generator) | Synthetic plate generation tool for hard-negative augmentation; see [Dataset Expansion Summary](./reports/dataset_expansion_summary.md). |

---

## 🛠️ Data Preparation & Manifest Generation

Directories `datasets/`, `datasets_external/`, `dataset_staging/`, and `vietnamese license plate.v1i.yolov11/` are excluded via `.gitignore`. 

To prepare the dataset pipeline after downloading the raw archives:
1. Generate standardized training manifests using [`tools/prepare_vnlp.py`](./tools/prepare_vnlp.py).
2. Training, validation, and test splits are defined in configuration files:
   - Detection: [`configs/license_plate.yaml`](./configs/license_plate.yaml)
   - OCR Student: [`configs/ocr_lprnet.yaml`](./configs/ocr_lprnet.yaml)
   - OCR Fine-tuning: [`configs/ocr_lprnet_v2a.yaml`](./configs/ocr_lprnet_v2a.yaml)

For comprehensive statistical audits, character distributions, and boundary analyses, refer to [`reports/current_dataset_report.md`](./reports/current_dataset_report.md) and [`reports/dataset_expansion_summary.md`](./reports/dataset_expansion_summary.md).
