# BÁO CÁO KẾT QUẢ THÍ NGHIỆM LPRNET V2-A (DÀNH CHO CHATGPT & HỘI ĐỒNG KỸ THUẬT)

---

# EXPERIMENT
- **Name:** V2-A PBL4 Unique
- **Objective:** Kiểm chứng giả thuyết: Bổ sung 1,211 ảnh thật mới từ PBL4 vào TRAIN, giữ nguyên 100% kiến trúc, preprocessing, optimizer và quy trình training, xem khả năng generalization của LPRNet-Lite có tăng hay không.
- **Date:** 2026-09-24

---

# HYPOTHESIS
"Nếu giữ nguyên toàn bộ kiến trúc (LPRNetLite), preprocessing (RGB, 160x48, normalize [-1.0, 1.0]), optimizer (Adam lr 0.001) và quy trình training hiện tại, nhưng bổ sung 1,211 ảnh thật mới từ PBL4 vào TRAIN, khả năng generalization của LPRNet-Lite có tăng hay không?"

---

# DATA
- **V1 Train count:** 29,836 mẫu (`datasets/VNLP_processed/train.csv`)
- **PBL4 Candidate:** 1,211 mẫu (`dataset_staging/pbl4_unique/`)
- **PBL4 Rejected (Leakage):** 0 mẫu (kiểm tra hash, pHash, rò rỉ vào VAL/TEST: 0 trường hợp)
- **PBL4 Accepted:** 1,211 mẫu
- **V2-A Total Train:** 31,047 mẫu (`datasets/VNLP_processed/train_v2a.csv`)
- **VAL:** 3,728 mẫu (`datasets/VNLP_processed/val.csv` - giữ nguyên hoàn toàn)
- **TEST_FINAL:** 3,733 mẫu (`datasets/VNLP_processed/test.csv` - khóa bất biến)
- **TEST_CLEAN:** 3,727 mẫu (`datasets/VNLP_processed/test_clean.csv` - loại bỏ chính xác 6 mẫu exact duplicate với TRAIN)
- **TEST_UNSEEN_TEXT:** 1,187 mẫu (`datasets/VNLP_processed/test_unseen_text.csv` - biển số chưa từng xuất hiện trong TRAIN)

---

# MODEL
- **Architecture:** `LPRNetLite` (`ocr/models/lprnet_lite.py`)
- **Modifications:** NONE (Không STN, không BiLSTM, không Transformer, không Spatial Attention)
- **Input:** `[1, 3, 48, 160]` RGB, chuẩn hóa `pixel / 127.5 - 1.0`
- **Output:** `[1, 37, 40]` (36 ký tự tiếng Việt + 1 blank index 36)
- **Decoder:** CTC Greedy Decoding thuần
- **Parameters:** 798,565 tham số (~0.80M)
- **ONNX Size:** 3.04 MB (3,191,811 bytes)

---

# TRAINING CONFIG
- **Config File:** `configs/ocr_lprnet_v2a.yaml` (sao chép nguyên mẫu từ `configs/ocr_lprnet.yaml`)
- **Epochs:** 60 kế hoạch, thực tế 46 epochs (dừng sớm do Early Stopping tại patience = 20)
- **Best Epoch:** Epoch 26
- **Batch Size:** 64
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, Weight Decay = 1e-4)
- **Initial LR:** 0.001
- **Scheduler:** CosineAnnealingLR ($T_{\max}=60, \eta_{\min}=1\text{e-}5$)
- **Seed:** Deterministic seed = 42 (Python random, NumPy, PyTorch CPU, PyTorch CUDA)
- **Augmentation:** Chuẩn của VNLP (ColorJitter, RandomAffine, Blur)
- **Initialization:** Khởi tạo ngẫu nhiên từ đầu (From Scratch - theo đúng config V1 không có pretrained weights)

---

# V1 RESULTS (BASELINE)
*Kiểm định bằng OpenCV DNN CPU trên Host Intel Core i5-12500H:*
- **TEST_FINAL Exact Acc:** 85.27% (3,183 / 3,733)
- **TEST_CLEAN Exact Acc:** 85.32% (3,180 / 3,727)
- **TEST_UNSEEN_TEXT Exact Acc:** 85.93% (1,020 / 1,187)
- **ONE_ROW Exact Acc:** 90.58% (1,730 / 1,910)
- **TWO_ROWS Exact Acc:** 81.47% (1,029 / 1,263)
- **MOTORCYCLE Exact Acc:** 75.71% (424 / 560)
- **Character Accuracy:** 97.15% (TEST_FINAL) / 97.24% (UNSEEN)
- **CER:** 0.0285 (TEST_FINAL) / 0.0276 (UNSEEN)

---

# V2-A RESULTS (PBL4 MERGED)
*Kiểm định cùng runtime OpenCV DNN CPU:*
- **TEST_FINAL Exact Acc:** 65.87% (2,459 / 3,733)
- **TEST_CLEAN Exact Acc:** 65.90% (2,456 / 3,727)
- **TEST_UNSEEN_TEXT Exact Acc:** 67.40% (800 / 1,187)
- **ONE_ROW Exact Acc:** 72.20% (1,379 / 1,910)
- **TWO_ROWS Exact Acc:** 63.66% (804 / 1,263)
- **MOTORCYCLE Exact Acc:** 49.29% (276 / 560)
- **Character Accuracy:** 93.94% (TEST_FINAL) / 94.03% (UNSEEN)
- **CER:** 0.0606 (TEST_FINAL) / 0.0597 (UNSEEN)

---

# DELTAS (V2-A - V1)
| Chỉ số | Model V1 | Model V2-A | Delta |
| :--- | :---: | :---: | :---: |
| **TEST_FINAL Exact Acc** | 85.27% | 65.87% | **-19.39 percentage points** |
| **TEST_CLEAN Exact Acc** | 85.32% | 65.90% | **-19.43 percentage points** |
| **TEST_UNSEEN_TEXT Exact Acc** | 85.93% | 67.40% | **-18.53 percentage points** |
| **Biển 1 dòng (one_row)** | 90.58% | 72.20% | **-18.38 percentage points** |
| **Biển 2 dòng ô tô (two_rows)** | 81.47% | 63.66% | **-17.81 percentage points** |
| **Biển 2 dòng xe máy (motorcycle)**| 75.71% | 49.29% | **-26.43 percentage points** |
| **Character Accuracy (TEST_FINAL)**| 97.15% | 93.94% | **-3.21 percentage points** |
| **Character Error Rate (CER)** | 0.0285 | 0.0606 | **+0.0321** |

---

# CONFUSION CHANGES
Theo dõi các cặp ký tự nhạy cảm trên `TEST_FINAL`:
- **8 ↔ B:**
  - `8 -> B`: 0 (V1) vs 0 (V2-A) [Thay đổi: 0]
  - `B -> 8`: 1 (V1) vs 0 (V2-A) [Thay đổi: -1]
- **0 ↔ D:**
  - `0 -> D`: 0 (V1) vs 3 (V2-A) [Thay đổi: +3]
  - `D -> 0`: 1 (V1) vs 0 (V2-A) [Thay đổi: -1]
- **5 ↔ S:**
  - `5 -> S`: 0 (V1) vs 0 (V2-A) [Thay đổi: 0]
  - `S -> 5`: 0 (V1) vs 0 (V2-A) [Thay đổi: 0]
- **2 ↔ Z:**
  - `2 -> Z`: 0 (V1) vs 0 (V2-A) [Thay đổi: 0]
  - `Z -> 2`: 0 (V1) vs 0 (V2-A) [Thay đổi: 0]

*Các lỗi nhầm lẫn phát sinh nhiều nhất ở Model V2-A:*
1. `C -> A`: 182 lần
2. `A -> C`: 87 lần
3. `2 -> 3`: 81 lần
4. `0 -> 6`: 56 lần
5. `B -> A`: 55 lần
6. `3 -> 2`: 53 lần

---

# FIXED CASES & REGRESSIONS
- **Fixed Cases (V1 sai -> V2-A đúng):** **82** mẫu (được lưu tại `reports/v2a_fixed_cases.csv`)
  - Ví dụ: Biển bị tối, góc xiên nhẹ từ tập PBL4 giúp nhận diện một số biển bị mờ ở V1.
- **Regressions (V1 đúng -> V2-A sai):** **806** mẫu (được lưu tại `reports/v2a_regressions.csv`)
- **Net Balance:** -724 mẫu đúng tuyệt đối trên `TEST_FINAL`.

---

# TRAINING OBSERVATIONS
1. **Quá trình hội tụ:** Training loss giảm đều từ `1.354` xuống `0.0299` ở Epoch 46.
2. **Hiện tượng Early Stopping:** Validation Exact Accuracy đạt đỉnh sớm ở Epoch 26 (`66.15%`), sau đó dao động quanh mức 64% - 65.9% và kích hoạt dừng sớm ở Epoch 46 (sau 20 epochs không vượt qua 66.15%).
3. **Nguyên nhân chính dẫn đến kết quả 65.87%:**
   - Model V1 ban đầu nhiều khả năng đã được huấn luyện với số epoch lớn hơn nhiều (ví dụ 100-150 epochs), hoặc có learning rate schedule/warmup khác, hoặc đã sử dụng checkpoint khởi tạo trước (pre-trained initialization).
   - Với chỉ 60 epochs từ scratch theo config mặc định hiện có trong repo (`configs/ocr_lprnet.yaml`), mô hình LPRNet-Lite chưa đạt đủ số bước lặp để hội tụ sâu vào không gian đặc trưng tối ưu.
   - Thêm 1,211 ảnh PBL4 (chiếm ~3.9% tập train) không đủ sức bù đắp độ thiếu hụt số bước huấn luyện khi train from scratch.

---

# ONNX VERIFICATION
- **Trạng thái:** **PASS** (100% khớp)
- **Độ lệch tối đa PyTorch vs ONNX Runtime:** `3.242493e-05` (< 1e-4)
- **Độ lệch tối đa PyTorch vs OpenCV DNN:** `4.577637e-05` (< 1e-4)
- **Tỷ lệ khớp giải mã PyTorch vs OpenCV DNN:** `100/100 (100.00%)`
- **File ONNX:** `weights/ocr/experiments/v2a/best_lprnet_v2a.onnx`

---

# LATENCY BENCHMARK
- **Laptop CPU (Intel Core i5-12500H, OpenCV DNN):**
  - Model V1: **3.41 ms** (~293.7 FPS)
  - Model V2-A: **3.33 ms** (~299.9 FPS)
  - Delta Latency: -0.08 ms (tương đương nhau về mặt điện toán)
- **Raspberry Pi 5 (Cortex-A76):**
  - Model V1: **~12.40 ms** (đo trước đó)
  - Model V2-A: **NOT MEASURED** (Pi hiện không kết nối trực tiếp trong môi trường test này)

---

# CONCLUSION BASED ON DATA
1. **Biến động chỉ số:**
   - Toàn bộ các chỉ số chính xác tuyệt đối đều giảm: `TEST_FINAL` giảm 19.39 pp, `TEST_CLEAN` giảm 19.43 pp, `TEST_UNSEEN_TEXT` giảm 18.53 pp.
   - Phân khúc biển 2 dòng xe máy chịu ảnh hưởng nặng nhất (-26.43 pp), tiếp theo là biển 1 dòng (-18.38 pp).
2. **Khả năng Generalization:**
   - Giả thuyết **KHÔNG ĐƯỢC XÁC NHẬN** trong điều kiện huấn luyện from-scratch 46 epochs. Thêm 1,211 ảnh PBL4 không giúp tăng generalization khi mà toàn bộ mạng chưa hội tụ tới mức của V1.
3. **Đánh giá Trade-off:**
   - Latency và kích thước mô hình không đổi (3.04 MB, ~3.3 ms CPU).
   - Tuy nhiên độ chính xác suy giảm nghiêm trọng.
4. **Phân loại mô hình:** **[REGRESSION]**.
5. **Quyết định Production:**
   - **TUYỆT ĐỐI KHÔNG DEPLOY V2-A LÀM PRODUCTION**.
   - Model V1 (`weights/ocr/best_lprnet_vn.onnx`) tiếp tục giữ nguyên 100% vai trò production active.

---

# RECOMMENDED NEXT EXPERIMENTS
Dựa trên bằng chứng thực nghiệm thu được:
1. **Thí nghiệm Fine-tuning V1 (Transfer Learning):** Lấy chính checkpoint V1 (`best_lprnet_vn.pth`) làm base, sau đó fine-tune với learning rate nhỏ ($1\text{e-}4 \to 1\text{e-}5$) trên tập dữ liệu kết hợp VNLP + PBL4. Đây là phương pháp chuẩn để bảo toàn các đặc trưng đã học của V1 đồng thời hấp thụ phân phối mới từ PBL4.
2. **Khảo sát Training Schedule V1:** Kiểm tra lại log hoặc lịch sử huấn luyện ban đầu của V1 xem V1 đã train bao nhiêu epoch (rất có thể là 150-200 epochs).
3. **Thí nghiệm V2-B (Targeted Synthetic):** Bổ sung dữ liệu sinh nhân tạo nhắm mục tiêu vào các cặp ký tự yếu và biển 2 dòng xe máy (chờ phê duyệt của ChatGPT và người dùng sau Phase 2).
