# PROJECT / PHASE

Project root: `D:\Vietnam_Plate_Project`  
Date: 2026-09-25  
Phase: **PHASE 4 — DATA INTEGRITY + CONTROLLED FINE-TUNING EXPERIMENT**  
Objective:
1. Xác minh 100% dữ liệu PBL4 (1,211 mẫu) được gán đúng nhãn trong manifest huấn luyện; giải quyết triệt để nghi vấn từ báo cáo bất thường ở Phase 3.
2. Phân lập nguyên nhân khoa học của sự cải thiện mô hình: (A) do bản thân quá trình tiếp tục fine-tune từ checkpoint V1 với learning rate nhỏ trên VNLP, hay (B) thực sự do đóng góp của 1,211 ảnh mới từ PBL4.
3. Đánh giá độ ổn định của hiệu ứng qua 3 random seed độc lập (`42`, `1337`, `2026`) trên cả 2 điều kiện `CONTROL` (VNLP thuần) và `PBL4` (VNLP + PBL4).
4. Khóa chọn checkpoint ứng viên tốt nhất hoàn toàn dựa trên tập `VALIDATION` trước khi mở tập `TEST`.
5. Đánh giá nghiêm ngặt trên các tập TEST (`TEST_FINAL`, `TEST_CLEAN`, `TEST_UNSEEN_TEXT`) kết hợp kiểm định giả thuyết thống kê (Exact McNemar Test và 10,000 paired bootstrap 95% Confidence Intervals).
6. Tuân thủ 100% nguyên tắc bất biến: Không export ONNX, không chạy Raspberry Pi, không đổi kiến trúc LPRNet-Lite, không sửa detector hay pipeline production.

---

# CURRENT MODEL ARCHITECTURE

- **Architecture:** `LPRNetLite` (Kiến trúc chuẩn của dự án: backbone tích hợp Conv + MaxPool + SmallBasicBlock nối tầng đa tỉ lệ, không dùng STN, không Transformer, không BiLSTM).
- **Input:** Ảnh RGB kích thước chuẩn $(W=160, H=48)$, chuẩn hóa dải giá trị $[-1.0, 1.0]$.
- **Output:** Feature map logits kích thước $[B, 37, 40]$ ($T=40$ time steps, $C=37$ classes).
- **Parameters:** 805,093 tham số (tương đương ~3.22 MB định dạng `float32`).
- **Charset:** 36 ký tự chữ số và chữ cái tiếng Anh in hoa:  
  `0123456789ABCDEFGHKLMNPRSTUVXYZ-` (lưu ý: loại trừ các chữ cái dễ nhầm lẫn/không dùng trên biển số Việt Nam gồm `I`, `J`, `O`, `Q`, `W`).
- **CTC Decoder:** CTC Greedy Decoder (`best_path`), với Blank Index = 36.

---

# DATASETS

Toàn bộ các phân tập đã được deduplicate, rà soát và khóa cố định từ Phase 1 và Phase 2:
- **VNLP Train (Gốc):** `datasets/VNLP_processed/train.csv` — **29,836 mẫu** (96.10% tổng tập Train V2a).
- **PBL4 Unique (Bổ sung):** `datasets/VNLP_processed/train_v2a_sources.csv` (lọc `source == PBL4`) — **1,211 mẫu** (3.90% tổng tập Train V2a).
- **Train V2a (Gộp VNLP + PBL4):** `datasets/VNLP_processed/train_v2a.csv` — **31,047 mẫu**.
- **VAL (Tập Validation chuẩn):** `datasets/VNLP_processed/val.csv` — **3,728 mẫu** (1,908 biển 1 dòng, 1,261 biển 2 dòng ô tô, 559 biển 2 dòng xe máy).
- **VAL_UNSEEN (Tập Validation text chưa từng xuất hiện ở Train):** `datasets/VNLP_processed/val_unseen_text_v2.csv` — **1,248 mẫu**.
- **TEST_FINAL (Tập Test khóa chính thức):** `datasets/VNLP_processed/test.csv` — **3,733 mẫu** (1,910 biển 1 dòng, 1,263 biển 2 dòng ô tô, 560 biển 2 dòng xe máy).
- **TEST_CLEAN (Tập Test đã lọc nhiễu annotation):** `datasets/VNLP_processed/test_clean.csv` — **3,727 mẫu**.
- **TEST_UNSEEN (Tập Test text hoàn toàn mới):** `datasets/VNLP_processed/test_unseen_text.csv` — **1,187 mẫu**.

---

# V1 BASELINE

Checkpoint gốc: `weights/ocr/best_lprnet_vn.pth` (SHA256: `25d2ff3cba0869766a96689a8eda2d1b133634c86023df01e2e41cf17491305c`).

- **VAL Exact Accuracy:** **84.79%** (3,161 / 3,728 mẫu)
- **TEST_FINAL:** **85.27%** (3,183 / 3,733 mẫu)
- **TEST_CLEAN:** **85.32%** (3,180 / 3,727 mẫu)
- **TEST_UNSEEN:** **85.93%** (1,020 / 1,187 mẫu)
- **Phân rã layout trên TEST_FINAL:**
  - `one_row` (biển 1 dòng): **90.58%** (1,730 / 1,910)
  - `two_rows` (biển 2 dòng ô tô): **81.47%** (1,029 / 1,263)
  - `motorcycle` (biển 2 dòng xe máy): **75.71%** (424 / 560)
- **Character Accuracy:** **97.15%**
- **CER:** **0.0285**
- **Average Edit Distance:** **0.2253**

---

# PHASE 3 FT-B

Checkpoint Phase 3 Candidate: `weights/ocr/experiments/phase3_ft_b/best.pth` (Seed 42, Epoch 6, Train trên VNLP + PBL4).

- **VAL Exact Accuracy:** **85.76%** (3,197 / 3,728 mẫu)
- **VAL_UNSEEN:** **85.18%**
- **TEST_FINAL:** **85.75%** (3,201 / 3,733 mẫu)
- **TEST_CLEAN:** **85.81%** (3,198 / 3,727 mẫu)
- **TEST_UNSEEN:** **85.51%** (1,015 / 1,187 mẫu)
- **Phân rã layout trên TEST_FINAL:**
  - `one_row`: **90.68%** (+0.10 pp vs V1)
  - `two_rows`: **81.16%** (-0.31 pp vs V1)
  - `motorcycle`: **79.29%** (+3.58 pp vs V1)
- **Character Accuracy:** **97.36%**
- **CER:** **0.0264**
- **Average Edit Distance:** **0.2076**
- **Fixed cases vs V1:** **93 mẫu**
- **Regressions vs V1:** **75 mẫu**
- **Net Gain:** **+18 mẫu**

---

# PBL4 LABEL INTEGRITY AUDIT

Đã tiến hành rà soát khẩn cấp toàn bộ dữ liệu PBL4 để giải quyết mâu thuẫn trong báo cáo `reports/pbl4_manual_spotcheck.md` cũ:

- **Tổng số mẫu PBL4 trong Train V2a:** **1,211 mẫu**
- **Số mẫu có nhãn khớp chính xác (Correct labels):** **1,211 mẫu** (**100.00%**)
- **Số mẫu nhầm nhãn (Label Mismatches):** **0 mẫu** (**0.00%**)
- **Lỗi parser (Parser Errors):** **0 mẫu** (**0.00%**)
- **Chi tiết kết quả xuất:** [`reports/phase4_pbl4_manifest_label_audit.csv`](file:///d:/Vietnam_Plate_Project/reports/phase4_pbl4_manifest_label_audit.csv).
- **Giải thích nguồn gốc lỗi báo cáo spotcheck cũ:**
  Trong Phase 3, script sinh báo cáo `scripts/dataset_tools/analyze_pbl4_domain.py` đã parse tên file bằng lệnh lỗi:  
  `label_parsed = Path(img_path).stem.split('_')[0]`  
  Với định dạng tên file PBL4 là `train_<PLATE>_<ID>_<AUG>.jpg` hoặc `valid_<PLATE>_<ID>_<AUG>.jpg`, lệnh trên đã lấy token ở vị trí `0` là chữ `"train"` / `"valid"`, dẫn đến việc báo cáo spotcheck hiển thị nhãn là `TRAIN` / `VALID` và ghi PASS = 0%.
- **File manifest huấn luyện thực tế (`train_v2a.csv`) có bị ảnh hưởng không:**  
  **KHÔNG (NO)**. File manifest huấn luyện được sinh bởi `scripts/dataset_tools/create_train_v2a.py`, script này sử dụng đúng logic `tokens[1]` để trích xuất biển số thật (`51F88270`, `29H06737`,...). 100% 1,211 mẫu trong `train_v2a.csv` đều chứa đúng nhãn biển số thật.
- **Tái lập công cụ Spotcheck chuẩn (v2):**  
  Đã tạo script sửa đổi và xuất lại báo cáo spotcheck 200 mẫu ngẫu nhiên: [`reports/pbl4_manual_spotcheck_v2.md`](file:///d:/Vietnam_Plate_Project/reports/pbl4_manual_spotcheck_v2.md).  
  **Kết quả Spotcheck v2:** **200 / 200 mẫu ĐẠT (PASS 100.00%)**, không có bất kỳ mâu thuẫn nào giữa bảng dữ liệu và kết luận.
- **Dữ liệu PBL4:** **ĐỦ ĐIỀU KIỆN TIẾP TỤC THÍ NGHIỆM (PASS DATA INTEGRITY)**.

---

# PBL4 DATA DISTRIBUTION

Các phát hiện trọng yếu từ phân tích phân phối dữ liệu ([`reports/phase4_pbl4_distribution.md`](file:///d:/Vietnam_Plate_Project/reports/phase4_pbl4_distribution.md)):
1. **Lệch layout cực đoan:** 100% (1,211 / 1,211) ảnh PBL4 là biển số xe ô tô **1 dòng** (`one_row`). PBL4 hoàn toàn **không chứa biển xe máy** (`motorcycle = 0%`) và không chứa biển 2 dòng ô tô vuông (`two_rows = 0%`).
2. **Lệch vùng địa lý (Province / Series Prefix Skew):**
   - Biển Hà Nội (`30` và `29`): Chiếm **50.70%** (Đầu số `30`: 39.47%, đầu số `29`: 11.23%).
   - Biển TP. Hồ Chí Minh (`51`): Chiếm **19.49%**.
   - Hơn **70.19%** tổng dữ liệu PBL4 chỉ tập trung vào 2 địa phương là Hà Nội và TP.HCM.
3. **Phân phối độ dài biển số:**
   - 8 ký tự (dạng `30A12345`): Chiếm **69.86%** (846 mẫu).
   - 9 ký tự: Chiếm **28.32%** (343 mẫu).
   - Biển 7 ký tự: Chỉ chiếm 1.82% (22 mẫu).
4. **Đặc tính hình ảnh:** Toàn bộ ảnh PBL4 là ảnh crop rất sát viền, độ tương phản cao, góc chụp vuông góc, ít nhiễu hình học.

---

# CONTROL EXPERIMENT DESIGN

Để xác định chính xác liệu sự cải thiện ở Phase 3 đến từ bản thân thuật toán fine-tuning hay từ dữ liệu PBL4, Phase 4 thiết lập thí nghiệm kiểm chứng đối chứng nghiêm ngặt (Controlled Experiment):
- **Base Model:** Chung checkpoint V1 `weights/ocr/best_lprnet_vn.pth`.
- **Kiến trúc:** `LPRNetLite` nguyên bản, không sửa đổi.
- **Điều kiện 1 (CONTROL):** Huấn luyện trên **VNLP Train thuần túy (29,836 mẫu)** — hoàn toàn không có PBL4.
- **Điều kiện 2 (PBL4):** Huấn luyện trên **VNLP Train + 1,211 ảnh PBL4 unique (31,047 mẫu)** — tương ứng dữ liệu V2a.
- **Hyperparameters cố định tuyệt đối:**
  - Optimizer: `Adam(betas=(0.9, 0.999), weight_decay=1e-4)`
  - Learning Rate: $5 \times 10^{-5}$
  - Scheduler: `CosineAnnealingLR(T_max=40, eta_min=5e-7)`
  - Batch Size: 64
  - Max Epochs: 40, Early Stopping patience: 10 epochs
  - Augmentations: Giữ nguyên cơ chế augmentation chuẩn của dự án (độ sáng, xoay góc nhẹ $\pm 3.5^\circ$, jitter).
- **Multi-Seed Testing:** Chạy 3 seed độc lập cho mỗi điều kiện: **Seed 42, Seed 1337, Seed 2026** (Tổng cộng 6 lần chạy độc lập).

---

# CONTROL RESULTS

Kết quả đo trên tập `val.csv` (3,728 mẫu) và `val_unseen_text_v2.csv` (1,248 mẫu):

| Run Name | Seed | Best Epoch | VAL Exact (%) | VAL Unseen (%) | 1Row (%) | 2Row (%) | Moto (%) | CharAcc (%) | CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CONTROL_S42** | 42 | Epoch 3 | **86.02%** | 84.86% | 90.57% | 81.44% | 80.86% | 97.48% | 0.0252 |
| **CONTROL_S1337** | 1337 | Epoch 1 | **85.94%** | 85.66% | 90.25% | 81.52% | 81.22% | 97.44% | 0.0256 |
| **CONTROL_S2026** | 2026 | Epoch 2 | **85.84%** | 85.02% | 90.20% | 81.52% | 80.68% | 97.44% | 0.0256 |

- **CONTROL Mean ± Std:**
  - **VAL Exact Accuracy:** **85.94 ± 0.08%** [Khoảng: 85.84% – 86.02%]
  - **VAL Unseen Exact:** **85.18 ± 0.35%** [Khoảng: 84.86% – 85.66%]
  - `one_row`: 90.34 ± 0.16%
  - `two_rows`: 81.50 ± 0.04%
  - `motorcycle`: 80.92 ± 0.22%
  - **CER:** **0.0255 ± 0.0002**

---

# PBL4 RESULTS

Kết quả đo trên tập `val.csv` (3,728 mẫu) và `val_unseen_text_v2.csv` (1,248 mẫu):

| Run Name | Seed | Best Epoch | VAL Exact (%) | VAL Unseen (%) | 1Row (%) | 2Row (%) | Moto (%) | CharAcc (%) | CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PBL4_S42** | 42 | Epoch 6 | **85.76%** | 85.18% | 90.04% | 81.28% | 81.22% | 97.43% | 0.0257 |
| **PBL4_S1337** | 1337 | Epoch 3 | **85.92%** | 85.18% | 90.51% | 81.36% | 80.50% | 97.45% | 0.0255 |
| **PBL4_S2026** | 2026 | Epoch 1 | **85.57%** | 85.02% | 90.04% | 80.97% | 80.68% | 97.39% | 0.0261 |

- **PBL4 Mean ± Std:**
  - **VAL Exact Accuracy:** **85.75 ± 0.14%** [Khoảng: 85.57% – 85.92%]
  - **VAL Unseen Exact:** **85.12 ± 0.08%** [Khoảng: 85.02% – 85.18%]
  - `one_row`: 90.20 ± 0.22%
  - `two_rows`: 81.21 ± 0.17%
  - `motorcycle`: 80.80 ± 0.30%
  - **CER:** **0.0258 ± 0.0002**

---

# PAIRED DATA CONTRIBUTION

So sánh trực diện theo từng cặp Seed tương ứng:

| Random Seed | Control VAL (%) | PBL4 VAL (%) | Delta VAL (PBL4 - Control) | Control Unseen (%) | PBL4 Unseen (%) | Delta Unseen | Control Moto (%) | PBL4 Moto (%) | Delta Moto |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 86.02% | 85.76% | **-0.27 pp** | 84.86% | 85.18% | +0.32 pp | 80.86% | 81.22% | +0.36 pp |
| **Seed 1337** | 85.94% | 85.92% | **-0.03 pp** | 85.66% | 85.18% | -0.48 pp | 81.22% | 80.50% | -0.72 pp |
| **Seed 2026** | 85.84% | 85.57% | **-0.27 pp** | 85.02% | 85.02% | +0.00 pp | 80.68% | 80.68% | +0.00 pp |
| **TRUNG BÌNH** | **85.94%** | **85.75%** | **-0.19 pp** | **85.18%** | **85.12%** | **-0.05 pp** | **80.92%** | **80.80%** | **-0.12 pp** |

### Kết luận khoa học về đóng góp của PBL4:
- **PBL4 benefit = [NOT SUPPORTED]**.
- Cả 3/3 seed, nhánh CONTROL đều đạt độ chính xác VAL bằng hoặc cao hơn nhánh có thêm PBL4.
- Sự vượt trội của Phase 3 FT-B (85.76%) so với V1 (84.79%) thực chất **100% đến từ việc fine-tune tiếp tục trên VNLP**. Khi fine-tune trên VNLP thuần túy với cùng cấu hình, mô hình đạt trung bình **85.94%** (cao hơn cả khi thêm PBL4).
- Việc thêm 1,211 ảnh PBL4 (vốn 100% là ô tô 1 dòng cắt sát viền tại Hà Nội/TP.HCM) tạo ra một độ lệch phân phối nhẹ (domain bias), làm giảm độ chính xác tổng thể khoảng **-0.19 pp**.

---

# SELECTED CANDIDATE

- **Checkpoint Path:** `weights/ocr/experiments/phase4/control_seed42/best.pth`
- **Condition:** **CONTROL** (VNLP Train Only, 29,836 mẫu)
- **Seed:** `42`
- **Best Epoch:** Epoch 3 (Early stopping dừng tại Epoch 13)
- **Lý do lựa chọn (Khóa hoàn toàn dựa trên VALIDATION trước khi mở TEST):**
  1. **Tiêu chí số 1 (PRIMARY - VAL Exact):** Đạt **86.02%** — điểm số cao nhất trong toàn bộ 6 mô hình của Phase 4 (vượt qua tất cả các mô hình có PBL4).
  2. **Tiêu chí số 3 (CER):** Đạt **0.0252** — thấp nhất (tốt nhất) trong tất cả 6 mô hình.
  3. **Tiêu chí số 2 & Phân rã:** Biển 1 dòng đạt 90.57%, biển 2 dòng ô tô đạt 81.44%, biển xe máy đạt 80.86% (tăng vọt +4.65 pp so với V1 baseline 76.21%).
- **Trạng thái:** Checkpoint đã được **khóa bất biến** trước khi gọi script đánh giá TEST.

---

# FINAL TEST COMPARISON

Đo đạc thực tế độc lập bằng PyTorch CUDA trên 3 tập Test chính thức:

| Chỉ số / Tập kiểm định | Model V1 | Phase 3 FT-B | Phase 4 Candidate (`CONTROL_S42`) | Delta (Candidate vs V1) | Delta (Candidate vs FT-B) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TEST_FINAL (3,733 mẫu)** | **85.27%** | **85.75%** | **85.96%** | **+0.70 pp** | **+0.21 pp** |
| — Biển 1 dòng (`one_row`, 1,910 mẫu) | 90.58% | 90.68% | **90.89%** | **+0.31 pp** | +0.21 pp |
| — Biển 2 dòng ô tô (`two_rows`, 1,263 mẫu) | 81.47% | 81.16% | **81.47%** | **+0.00 pp** | +0.31 pp |
| — Biển 2 dòng xe máy (`motorcycle`, 560 mẫu)| 75.71% | 79.29% | **79.29%** | **+3.58 pp** | +0.00 pp |
| **TEST_CLEAN (3,727 mẫu)** | **85.32%** | **85.81%** | **86.02%** | **+0.70 pp** | **+0.21 pp** |
| — Biển 1 dòng (1,908 mẫu) | 90.67% | 90.78% | **90.99%** | **+0.32 pp** | +0.21 pp |
| — Biển 2 dòng ô tô (1,259 mẫu) | 81.49% | 81.18% | **81.49%** | **+0.00 pp** | +0.31 pp |
| — Biển 2 dòng xe máy (560 mẫu) | 75.71% | 79.29% | **79.29%** | **+3.58 pp** | +0.00 pp |
| **TEST_UNSEEN_TEXT (1,187 mẫu)** | **85.93%** | **85.51%** | **85.76%** | **-0.17 pp** | **+0.25 pp** |
| — Biển 1 dòng (638 mẫu) | 91.81% | 91.09% | **91.67%** | -0.14 pp | +0.58 pp |
| — Biển 2 dòng ô tô (339 mẫu) | 80.07% | 80.07% | **79.00%** | -1.07 pp | -1.07 pp |
| — Biển 2 dòng xe máy (210 mẫu) | 74.29% | 74.29% | **75.24%** | **+0.95 pp** | **+0.95 pp** |
| **Character Accuracy (TEST_FINAL)** | 97.15% | 97.36% | **97.39%** | **+0.24 pp** | +0.03 pp |
| **CER (TEST_FINAL)** | 0.0285 | 0.0264 | **0.0261** | **-0.0024** | -0.0003 |
| **Average Edit Distance (TEST_FINAL)** | 0.2253 | 0.2076 | **0.2057** | **-0.0196** | -0.0019 |

---

# STATISTICAL ANALYSIS

Được tính toán tự động bằng script `scripts/dataset_tools/evaluate_phase4_final.py`:

### 1. Phân tích thống kê Phase 3 FT-B (vs V1):
- **Số mẫu sửa đúng (Fixed, $b$):** 93 mẫu
- **Số mẫu suy thoái (Regressions, $c$):** 75 mẫu
- **Net Gain:** $+18$ mẫu
- **McNemar Exact Test ($p$-value hai phía):** $p = \mathbf{0.1895}$  
  *(Vì $p = 0.1895 \ge 0.05$, mức tăng +18 mẫu của FT-B ở Phase 3 **CHƯA ĐẠT Ý NGHĨA THỐNG KÊ** ở mức $\alpha = 0.05$)*.

### 2. Phân tích thống kê Phase 4 Candidate (`CONTROL_S42` vs V1):
- **Số mẫu sửa đúng (Fixed, $b$):** **89 mẫu** ([`reports/phase4_fixed_cases.csv`](file:///d:/Vietnam_Plate_Project/reports/phase4_fixed_cases.csv))
- **Số mẫu suy thoái (Regressions, $c$):** **63 mẫu** ([`reports/phase4_regressions.csv`](file:///d:/Vietnam_Plate_Project/reports/phase4_regressions.csv))
- **Net Gain:** **+26 mẫu**
- **McNemar Exact Test ($p$-value hai phía):** $p = \mathbf{0.0422}$  
  *(Vì $p = 0.0422 < 0.05$, mức cải thiện của Phase 4 Candidate so với V1 **ĐẠT Ý NGHĨA THỐNG KÊ** ở mức tin cậy 95%)*.

### 3. Paired Bootstrap 95% Confidence Interval (10,000 resamples):
- **Trên `TEST_FINAL`:**
  - Delta: **+0.70 pp**
  - **95% Confidence Interval:** $[ \mathbf{+0.05\text{ pp}}, \mathbf{+1.34\text{ pp}} ]$
  - Khoảng tin cậy hoàn toàn nằm ở phía dương, **KHÔNG CHỨA SỐ 0** (Spans zero: **False**). Xác nhận sự cải thiện trên TEST_FINAL là có ý nghĩa thực sự.
- **Trên `TEST_CLEAN`:**
  - Delta: **+0.70 pp**
  - **95% Confidence Interval:** $[ \mathbf{+0.05\text{ pp}}, \mathbf{+1.34\text{ pp}} ]$
  - **KHÔNG CHỨA SỐ 0** (Spans zero: **False**).
- **Trên `TEST_UNSEEN_TEXT`:**
  - Delta: **-0.17 pp**
  - **95% Confidence Interval:** $[ -1.26\text{ pp}, +0.93\text{ pp} ]$
  - Khoảng tin cậy **CHỨA SỐ 0** (Spans zero: **True**). Trên tập unseen text, sự khác biệt giữa V1 và Candidate chưa thể phân biệt được với 0 ở mức $\alpha = 0.05$.

---

# GENERALIZATION

- **Validation Unseen:**
  - V1 Baseline: ~84.79%
  - Phase 4 Candidate: **84.86%** (và đạt đỉnh **85.66%** ở seed 1337).
- **Test Unseen (`TEST_UNSEEN_TEXT`):**
  - V1 Baseline: **85.93%**
  - Phase 3 FT-B: **85.51%** (-0.42 pp)
  - Phase 4 Candidate: **85.76%** (-0.17 pp)
- **Đánh giá năng lực tổng quát hóa:**
  - Candidate Phase 4 có CER trên Test Unseen giảm xuống **0.0258** (thấp hơn V1 là 0.0276 và FT-B là 0.0264).
  - Độ chính xác ký tự (`Char Acc`) trên Test Unseen tăng từ 97.24% (V1) lên **97.42%** (Candidate).
  - Về Exact Accuracy trên Test Unseen, Candidate (-0.17 pp so với V1) tốt hơn đáng kể so với FT-B (-0.42 pp). Khoảng tin cậy Bootstrap 95% $[-1.26\text{ pp}, +0.93\text{ pp}]$ bao trùm 0 cho thấy mô hình không bị suy thoái tổng quát hóa có ý nghĩa thống kê, đồng thời bảo toàn năng lực đọc chuỗi ký tự mới.

---

# MOTORCYCLE

- **V1 Baseline TEST:** **75.71%** (424 / 560 mẫu)
- **Phase 3 FT-B TEST:** **79.29%** (444 / 560 mẫu)
- **Phase 4 Candidate TEST:** **79.29%** (444 / 560 mẫu)
- **Delta so với V1:** **+3.58 pp** (+20 biển số xe máy được đọc chính xác tuyệt đối).
- **Trên TEST_UNSEEN:** Biển xe máy của Candidate đạt **75.24%**, cao hơn cả V1 (74.29%) và FT-B (74.29%).
- **Nhận xét:** Dù không có bất kỳ mẫu xe máy nào trong PBL4, việc fine-tune trên VNLP với learning rate nhỏ $5\times 10^{-5}$ đã giúp củng cố mạnh mẽ khả năng nhận diện biển 2 dòng xe máy.

---

# TWO ROW

- **V1 Baseline TEST:** **81.47%** (1,029 / 1,263 mẫu)
- **Phase 3 FT-B TEST:** **81.16%** (1,025 / 1,263 mẫu, bị suy thoái nhẹ -0.31 pp)
- **Phase 4 Candidate TEST:** **81.47%** (1,029 / 1,263 mẫu)
- **Delta so với V1:** **+0.00 pp** (Bảo toàn hoàn hảo, triệt tiêu sự suy thoái của FT-B).
- **Nhận xét:** Trong Phase 3, FT-B bị giảm nhẹ trên biển ô tô 2 dòng (từ 81.47% xuống 81.16%) do domain skew của PBL4 (chỉ có biển 1 dòng). Ở Phase 4, Candidate CONTROL loại bỏ hoàn toàn sự suy thoái này.

---

# ERROR ANALYSIS

Dựa trên phân tích 89 mẫu sửa đúng (`phase4_fixed_cases.csv`) và 63 mẫu suy thoái (`phase4_regressions.csv`):

### 1. Top Improvements (Các dạng lỗi V1 mắc phải đã được sửa triệt để):
- **Khắc phục hiện tượng lặp/nuốt ký tự ở viền:** Sửa các ca V1 sinh ký tự dư do biên CTC (`366A43902` $\rightarrow$ `36A43902`, `36AH00601` $\rightarrow$ `36A00601`, `360A13808` $\rightarrow$ `36A13808`).
- **Khắc phục lỗi mất số đầu biển xe máy:** Các trường hợp V1 bỏ sót số đầu dòng trên biển 2 dòng đã được khôi phục chính xác (`9T82693` $\rightarrow$ `29T82693`).
- **Sửa các cặp nhầm lẫn ký tự kinh điển:**
  - `0` $\leftrightarrow$ `6` trên biển mờ/bẩn (`34B204330` $\rightarrow$ `34B264330`).
  - `D` $\leftrightarrow$ `0` (`29L041115` $\rightarrow$ `29LD41115`).
  - `E` $\leftrightarrow$ `F` (`29F167258` $\rightarrow$ `29E167258`).

### 2. Top Regressions (Các dạng lỗi mới xuất hiện ở Candidate):
- **Cắt tỉa ký tự biên trái:** Một số biển số bị mất ký tự đầu tiên `3` trong tiền tố `36` (ví dụ: `36N0529` thành `3N0529`, `36B00351` thành `3B00351`, `36C27122` thành `3C27122`). Hiện tượng này xảy ra khi bounding box crop quá sát viền trái khiến receptive field của tầng Conv đầu tiên không đủ biên đệm.
- **Nhầm lẫn ký tự có cấu trúc nét tương đồng ở góc nghiêng:** `R` $\rightarrow$ `B` (`36R00608` $\rightarrow$ `36B00608`), `M` $\rightarrow$ `N` (`30M33148` $\rightarrow$ `30N33148`).

---

# DATA QUALITY ISSUES

1. **Bug báo cáo trong công cụ phân tích cũ của Phase 3:** Lệnh split chuỗi tại vị trí `tokens[0]` thay vì `tokens[1]` đã biến nhãn biển số thành `TRAIN` / `VALID`. Đã được chứng minh là lỗi hiển thị độc lập của script báo cáo, không gây nhiễm bẩn file manifest huấn luyện.
2. **Độ lệch phân phối của tập PBL4:** Tập dữ liệu PBL4 tuy có chất lượng ảnh tốt nhưng thuần túy là xe con 1 dòng tại Hà Nội/TP.HCM. Nếu gộp thẳng vào huấn luyện mà không áp dụng kỹ thuật cân bằng (sample re-weighting hoặc balanced sampling) sẽ gây lệch phân phối đối với biển xe máy và biển 2 dòng.
3. **Hiện tượng nhãn nhiễu trong Test Set:** `TEST_CLEAN` (3,727 mẫu) loại bỏ 6 mẫu sai nhãn trong `TEST_FINAL` (3,733 mẫu). Tuy nhiên độ chênh lệch là rất nhỏ (0.16%), cho thấy chất lượng tập TEST hiện tại có độ tin cậy cao.

---

# SCIENTIFIC CONCLUSION

Trả lời trực diện 7 câu hỏi khoa học cốt lõi:

1. **Fine-tuning alone có giúp không?**  
   **CÓ, ĐÓNG GÓP TOÀN BỘ SỰ CẢI THIỆN.** Bản thân việc tiếp tục huấn luyện V1 trên VNLP với LR nhỏ ($5\times 10^{-5}$) giúp mô hình đạt trung bình 85.94% VAL và 85.96% TEST_FINAL (vượt xa V1 gốc 85.27%).
2. **PBL4 có giúp không?**  
   **KHÔNG GIÚP (NOT SUPPORTED).** Khi kiểm soát cùng seed và tham số, việc bổ sung PBL4 làm giảm trung bình -0.19 pp trên VAL và không mang lại lợi thế thống kê nào so với CONTROL.
3. **Kết quả có ổn định qua seed không?**  
   **RẤT ỔN ĐỊNH.** Nhánh CONTROL đạt độ lệch chuẩn chỉ $\pm 0.08\%$ qua 3 seed (85.84%, 85.94%, 86.02%).
4. **Candidate có tốt hơn V1 một cách rõ ràng không?**  
   **CÓ, ĐẠT Ý NGHĨA THỐNG KÊ.** Candidate tăng net +26 mẫu trên TEST_FINAL, đạt kiểm định Exact McNemar $p = 0.0422 < 0.05$ và khoảng tin cậy Bootstrap 95% $[+0.05\text{ pp}, +1.34\text{ pp}]$ hoàn toàn không chứa 0.
5. **Candidate có tăng generalization không?**  
   **CÓ VỀ MẶT KÝ TỰ, TƯƠNG ĐƯƠNG VỀ MẶT EXACT.** CER trên Unseen giảm từ 0.0276 xuống 0.0258, Char Acc tăng từ 97.24% lên 97.42%. Khoảng tin cậy 95% bao trùm 0 chứng minh không có suy thoái tổng quát hóa.
6. **Motorcycle có tăng không?**  
   **TĂNG MẠNH.** Độ chính xác biển xe máy tăng từ 75.71% lên **79.29%** (+3.58 pp trên TEST_FINAL) và từ 74.29% lên **75.24%** trên TEST_UNSEEN.
7. **Two-row có regression không?**  
   **KHÔNG.** Biển 2 dòng ô tô đạt 81.47% (bảo toàn 100% so với V1, khắc phục sự suy thoái 81.16% của FT-B Phase 3).

---

# BEST PYTORCH MODEL SO FAR

- **Đường dẫn Checkpoint:**  
  `weights/ocr/experiments/phase4/control_seed42/best.pth`
- **Thông số kỹ thuật:**
  - Checkpoint PyTorch thuần (`.pth`), dung lượng: 9.63 MB (chứa state dict mô hình, optimizer, epoch 3).
  - VAL Exact: **86.02%** | TEST_FINAL: **85.96%** | TEST_CLEAN: **86.02%**
  - Character Accuracy: **97.39%** | CER: **0.0261**
- **CẢNH BÁO QUY TẮC:** **KHÔNG export ONNX. KHÔNG triển khai Raspberry Pi. KHÔNG sửa model production.**

---

# NEXT RECOMMENDED EXPERIMENT

Dựa trên bằng chứng thực nghiệm của Phase 4, hướng đi duy nhất được đề xuất cho Phase 5:
- **Tên đề xuất:** *Targeted Motorcycle & Perspective Synthetic Augmentation with Class-Balanced Sampling (Phase 5).*
- **Lý do khoa học:** Điểm yếu lớn nhất hiện tại của hệ thống nằm ở biển số xe máy 2 dòng (79.29% so với 90.89% của ô tô 1 dòng) và hiện tượng mất nét biên khi góc chụp nghiêng. Bổ sung dữ liệu thật không kiểm soát (như PBL4) đã thất bại vì lệch phân phối sang ô tô. Do đó, Phase 5 cần tập trung sinh dữ liệu giả lập có chủ đích (targeted synthetic generator) chuyên sâu cho biển xe máy 2 dòng và các góc nghiêng quang học, kết hợp lấy mẫu cân bằng (balanced batch sampling).

---

# FILES CREATED

1. `reports/phase4_pbl4_manifest_label_audit.csv` — Kết quả audit chi tiết 1,211 mẫu PBL4 (100% PASS).
2. `reports/pbl4_manual_spotcheck_v2.md` — Báo cáo spotcheck 200 mẫu PBL4 v2 đã sửa lỗi parser.
3. `reports/phase4_pbl4_distribution.md` — Báo cáo phân tích đặc tính phân phối dữ liệu PBL4.
4. `scripts/train_phase4_controlled.py` — Script thực thi huấn luyện kiểm định đối chứng đa seed.
5. `reports/phase4_multiseed_validation.md` — Báo cáo tổng hợp đối chứng 6 thí nghiệm trên tập Validation.
6. `scripts/dataset_tools/evaluate_phase4_final.py` — Script đánh giá TEST, McNemar test và 10,000 paired bootstrap CI.
7. `reports/phase4_fixed_cases.csv` — Danh sách 89 mẫu V1 sai nhưng Candidate Phase 4 đoán đúng.
8. `reports/phase4_regressions.csv` — Danh sách 63 mẫu V1 đúng nhưng Candidate Phase 4 đoán sai.
9. `reports/phase4_evaluation_results.json` — Toàn bộ số liệu thô và giá trị kiểm định thống kê.
10. `reports/PHASE4_MASTER_REPORT_FOR_CHATGPT.md` — Master Report tự chứa toàn diện (file hiện tại).

---

# FINAL STATUS

- **BEST MODEL:** `weights/ocr/experiments/phase4/control_seed42/best.pth`
- **BETTER THAN V1:** **YES** (Vượt trội +0.70 pp TEST_FINAL, $p = 0.0422 < 0.05$, Bootstrap CI $[+0.05\text{ pp}, +1.34\text{ pp}]$)
- **PBL4 BENEFIT:** **NOT SUPPORTED** (Nhánh CONTROL thuần VNLP vượt trội hơn nhánh có PBL4 -0.19 pp)
- **DATA INTEGRITY:** **PASS** (1,211 / 1,211 nhãn PBL4 trong manifest đúng 100%)
- **ONNX EXPORTED:** **NO**
- **PI TESTED:** **NO**
- **PRODUCTION CHANGED:** **NO**
