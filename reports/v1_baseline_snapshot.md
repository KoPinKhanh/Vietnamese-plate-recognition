# SNAPSHOT BASELINE MODEL V1 (V1 BASELINE SNAPSHOT)

**Thời gian ghi nhận:** 2026-09-24  
**Mục đích:** Khóa và lưu vết toàn bộ thông số, mã băm SHA256 và hiệu năng của Model V1 trước khi thực hiện thí nghiệm V2-A nhằm đảm bảo Model V1 không bị thay đổi hoặc ghi đè.

---

## 1. File Trọng số & Mã băm Toàn vẹn (SHA256)
- **Model ONNX V1:** `weights/ocr/best_lprnet_vn.onnx`
  - Dung lượng: `3,191,184 bytes` (~3.04 MB)
  - **SHA256:** `6add22435d43c5d340ae775a0801a4cb106499b9795cbd181fcc8a8c2221258b`
- **Checkpoint PyTorch V1 Best:** `weights/ocr/best_lprnet_vn.pth`
  - Dung lượng: `10,465,421 bytes` (~9.98 MB)
  - **SHA256:** `25d2ff3cba0869766a96689a8eda2d1b133634c86023df01e2e41cf17491305c`
- **Checkpoint PyTorch V1 Last:** `weights/ocr/last_lprnet_vn.pth`
  - Dung lượng: `10,465,229 bytes` (~9.98 MB)
  - **SHA256:** `3b7f95e4c9e6c336b67aac534ee68f49b08416ce4fcfdc263473fe78492c2138`

---

## 2. Kiến trúc & Cấu hình Kỹ thuật
- **Kiến trúc mạng:** `LPRNetLite` (định nghĩa tại `ocr/models/lprnet_lite.py`)
- **Khởi tạo:** Huấn luyện từ đầu (train from scratch, random initialization)
- **Input Tensor:** `[1, 3, 48, 160]`, định dạng RGB, float32 trong khoảng `[-1.0, 1.0]`
- **Output Tensor:** `[1, 37, 40]` (37 classes, 40 time-steps)
- **Bảng mã ký tự (CHARS):** `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` (36 ký tự chữ hoa và số)
- **CTC Blank Index:** `36` (vị trí cuối cùng)
- **Tổng số lớp (`NUM_CLASSES`):** `37`
- **Tiền xử lý:**
  - Biển 1 dòng (`one_row`): Letterbox đệm viền trắng (255, 255, 255) về `(160, 48)`
  - Biển 2 dòng (`two_rows` / `motorcycle`): Tìm thung lũng Sobel đứng, cắt thành dòng trên & dưới, resize về chiều cao 48 px, ghép ngang tuần tự với dải đệm trắng 6 px (`rectify_two_row_plate`), letterbox về `(160, 48)`

---

## 3. Cấu hình Huấn luyện Gốc (`configs/ocr_lprnet.yaml`)
- **Train Manifest:** `datasets/VNLP_processed/train.csv` (29,836 mẫu)
- **Validation Manifest:** `datasets/VNLP_processed/val.csv` (3,728 mẫu)
- **Test Manifest (`TEST_FINAL`):** `datasets/VNLP_processed/test.csv` (3,733 mẫu)
- **Unseen Text Manifest:** `datasets/VNLP_processed/test_unseen_text.csv` (1,187 mẫu)
- **Batch Size:** `64`
- **Epochs:** `60`
- **Learning Rate:** `0.001`
- **Weight Decay:** `0.0001`
- **Scheduler:** `cosine` (min_lr: `0.00001`)
- **Early Stopping Patience:** `20`
- **Optimizer:** `Adam` (betas=(0.9, 0.999))
- **Loss:** `torch.nn.CTCLoss(blank=36, zero_infinity=True)`
- **Mixed Precision (AMP):** `True`
- **Gradient Clipping Norm:** `5.0`
- **Seed:** `42`
- **Command huấn luyện gốc:**
  `python scripts/train_ocr_lprnet.py --config configs/ocr_lprnet.yaml --mode train`

---

## 4. Các Chỉ số Hiệu năng Baseline V1
- **TEST_FINAL (3,733 mẫu):**
  - Exact Plate Accuracy: **85.27%** (3,183 / 3,733 mẫu)
  - one_row: **90.58%** (1,730 / 1,910 mẫu)
  - two_rows: **81.47%** (1,029 / 1,263 mẫu)
  - motorcycle: **75.71%** (424 / 560 mẫu)
  - Character Accuracy: **95.68%**
  - CER: **0.0432**
- **TEST_UNSEEN_TEXT (1,187 mẫu):**
  - Exact Plate Accuracy: **72.11%** (856 / 1,187 mẫu)
  - Character Accuracy: **91.95%**
  - CER: **0.0805**
- **Độ trễ suy luận (Latency):**
  - Laptop Intel Core i5-12500H CPU: ~**3.25 ms** (>300 FPS)
  - Raspberry Pi 5 CPU: ~**12.40 ms** (~80 FPS)
