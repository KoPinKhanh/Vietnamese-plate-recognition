# SNAPSHOT BÁO CÁO TOÀN DIỆN DỰ ÁN NHẬN DIỆN BIỂN SỐ XE VIỆT NAM
*(Tài liệu chuẩn bị cho Kỹ sư AI / ChatGPT phân tích)*

---

## 1. PROJECT OVERVIEW
- **Project Root:** `D:\Vietnam_Plate_Project`
- **Platform Host:** Windows 11 Pro 64-bit (Build 10.0.26100)
- **Target Hardware Edge:** Raspberry Pi 5 (Quad-core ARM Cortex-A76 @ 2.4GHz, RAM 8GB, OS: Raspberry Pi OS 64-bit Bookworm)
- **Host Hardware:** 12th Gen Intel Core i5-12500H (12 Cores, 16 Threads), 16GB RAM, NVIDIA RTX 3050 Laptop GPU (CUDA 12.4)
- **Python Version:** 3.13.12 (trong môi trường virtualenv `.venv`)
- **Git Status:** 
  - Thư mục root hiện tại **không phải là Git repository** trực tiếp (không có thư mục `.git`).
  - Toàn bộ source code được lưu trữ local trên phân vùng `D:`.

---

## 2. CURRENT GOAL
- Xây dựng hệ thống End-to-End (E2E) tự động nhận diện biển số xe cơ giới Việt Nam (ô tô biển dài 1 dòng, ô tô biển vuông 2 dòng, và xe máy 2 dòng) triển khai thực tế trên **Raspberry Pi 5** kết hợp USB Camera / Video Stream.
- Yêu cầu hiệu năng: Tối ưu hóa suy luận thuần CPU thông qua **OpenCV DNN** và **NumPy**, loại bỏ hoàn toàn runtime nặng (PyTorch, Torchvision, Ultralytics, ONNX Runtime) trên thiết bị biên.
- Mục tiêu tốc độ: Đạt **25 - 35+ FPS** cho toàn bộ pipeline E2E trên Raspberry Pi 5 với độ chính xác nhận diện ký tự và biển số cao.

---

## 3. CURRENT PIPELINE
Pipeline thực tế được hiện thực hóa qua `scripts/pi_recognize_image.py`, `scripts/pi_recognize_camera.py`, và `scripts/pi_recognize_video.py`:

```
               [ CAMERA / VIDEO / IMAGE INPUT ]
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │   STAGE 1: LICENSE PLATE DETECTOR    │
            │   Model: YOLO11n ONNX (416x416)      │
            │   Engine: OpenCV DNN (CPU)           │
            │   Code: detector/pi_yolo_detector.py │
            └──────────────────────────────────────┘
                               │
             Bounding Box [x1, y1, x2, y2] + Confidence
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │    STAGE 2: CROP & PREPROCESSING     │
            │  - Proportional margin pad (+5%)     │
            │  - Aspect ratio test: w/h >= 2.0 ?   │
            │    * >= 2.0 -> ONE-ROW (biển dài)    │
            │    * < 2.0  -> TWO-ROWS (biển vuông) │
            │  - Two-row Rectification:            │
            │    Sobel vertical gradient valley    │
            │    split -> Top & Bot row resize 48h │
            │    -> Concat ngang + 6px trắng spacer│
            │  - Letterbox pad to (160, 48) white  │
            │  Code: ocr/preprocess.py             │
            └──────────────────────────────────────┘
                               │
               Tensor RGB [1, 3, 48, 160], [-1.0, 1.0]
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │       STAGE 3: OCR INFERENCE         │
            │   Model: LPRNet-Lite ONNX            │
            │   Engine: OpenCV DNN (CPU)           │
            │   Code: ocr/pi_lprnet_ocr.py         │
            └──────────────────────────────────────┘
                               │
                  Logits Tensor [1, 37, 40]
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │      STAGE 4: CTC GREEDY DECODE      │
            │  - Pure NumPy Argmax along class axis│
            │  - Collapse duplicate tokens         │
            │  - Filter CTC blank token (index 36) │
            │  - Map indices -> CHARS [0-9, A-Z]   │
            │  Code: ocr/ctc_decoder.py            │
            └──────────────────────────────────────┘
                               │
                    Raw OCR String Candidate
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │   STAGE 5: STABILIZATION & VOTING    │
            │  - Temporal IoU tracking qua các frame│
            │  - Sliding window (size=5, min_vote=3│
            │  - Loại bỏ jitter nhấp nháy ký tự     │
            │  - Cooldown chống spam log terminal  │
            │  Code: ocr/plate_stabilizer.py       │
            └──────────────────────────────────────┘
                               │
                               ▼
                  [ FINAL RECOGNIZED PLATE ]
                (Ví dụ: "30A-123.45" | 31.5 FPS)
```

---

## 4. ACTIVE MODELS
Bảng tổng hợp các model trọng số hiện có trong project:

| Model File | Dung lượng | Loại Model | Mục đích | Input Shape / Format | Output Shape / Format | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `weights/ocr/best_lprnet_vn.onnx` | 3.04 MB | LPRNet-Lite | OCR nhận dạng chuỗi ký tự biển số | `[1, 3, 48, 160]`, RGB, `[-1, 1]` | `[1, 37, 40]`, float32 | **ACTIVE** |
| `weights/ocr/best_lprnet_vn.pth` | 9.98 MB | LPRNetLite PyTorch | Checkpoint OCR tốt nhất sau huấn luyện | `[B, 3, 48, 160]` | `[B, 37, 40]` | **ACTIVE** |
| `weights/ocr/last_lprnet_vn.pth` | 9.98 MB | LPRNetLite PyTorch | Checkpoint OCR epoch cuối cùng | `[B, 3, 48, 160]` | `[B, 37, 40]` | BACKUP |
| `weights/best_license_plate_yolo11n_416.onnx` | 10.02 MB | YOLO11n ONNX | Plate Detection tối ưu cho Pi 5 (416x416) | `[1, 3, 416, 416]`, RGB, `[0, 1]` | `[1, 5, 3549]` (cx, cy, w, h, conf) | **ACTIVE** |
| `weights/best_license_plate_yolo11n_320.onnx` | 9.99 MB | YOLO11n ONNX | Plate Detection siêu nhanh (320x320) | `[1, 3, 320, 320]`, RGB, `[0, 1]` | `[1, 5, 2100]` | STANDBY |
| `weights/best_license_plate_yolo11n_512.onnx` | 10.06 MB | YOLO11n ONNX | Plate Detection độ phân giải 512x512 | `[1, 3, 512, 512]`, RGB, `[0, 1]` | `[1, 5, 5376]` | STANDBY |
| `weights/best_license_plate_yolo11n_640.onnx` | 10.11 MB | YOLO11n ONNX | Plate Detection chuẩn 640x640 | `[1, 3, 640, 640]`, RGB, `[0, 1]` | `[1, 5, 8400]` | STANDBY |
| `weights/best_license_plate_yolo11n.pt` | 5.22 MB | YOLO11n PyTorch | Model gốc huấn luyện detector (mAP50 99.49%) | `[B, 3, 640, 640]` | Ultralytics format | STANDBY |
| `weights/yolo26n.pt` / `yolo11n.pt` | ~5.5 MB | YOLO Pretrained | Pretrained weights gốc | N/A | N/A | BASE WEIGHTS |

---

## 5. CHI TIẾT OCR HIỆN TẠI
- **Kiến trúc:** `LPRNetLite` (được định nghĩa trong `ocr/models/lprnet_lite.py`).
- **Triển khai:** Hoàn toàn là mạng tích chập sâu không đệ quy (fully-convolutional), tích hợp trích xuất đặc trưng đa tầng (multi-scale feature adapters) từ stage 1, stage 2 và stage 3, sau đó gộp vào tầng phân loại 384 channels.
- **Input Resolution:** Chiều cao 48 px, Chiều rộng 160 px, 3 channels RGB (`[1, 3, 48, 160]`).
- **Chuẩn hóa giá trị pixel:** `pixel_norm = (pixel_rgb / 127.5) - 1.0` (dải giá trị `[-1.0, 1.0]`).
- **Kích thước đầu ra:** `[1, 37, 40]` tương ứng 37 classes ký tự trên 40 bước thời gian (time steps dọc theo chiều ngang).
- **Decoder:** CTC Greedy Decoder (`ocr/ctc_decoder.py`) triển khai trên cả PyTorch Tensor lẫn NumPy mảng thuần (phục vụ suy luận không cần PyTorch).
- **Xử lý bố cục biển số:**
  - `one_row`: Letterbox trực tiếp với viền trắng về `(160, 48)`.
  - `two_rows` và `two_rows_label_xe_may`: Tự động tìm đường phân cách giữa 2 dòng chữ bằng năng lượng cạnh đạo hàm Sobel đứng (`find_horizontal_split`). Sau đó cắt làm 2 hàng, resize cả 2 về cùng chiều cao 48 px, ghép ngang tuần tự (hàng trên bên trái, hàng dưới bên phải) cách nhau bởi 6 px đệm trắng (`rectify_two_row_plate`).

---

## 6. CHARACTER SET (CHARS)
- **Biến CHARSET trong `ocr/charset.py`:**
  `CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"`
- **Tổng số ký tự hợp lệ (`NUM_CHARS`):** 36 ký tự (10 chữ số + 26 chữ cái tiếng Anh in hoa).
- **CTC Blank Index (`BLANK_INDEX`):** `36` (vị trí cuối cùng).
- **Tổng số lớp đầu ra (`NUM_CLASSES`):** `37`.
- **Ký tự không được hỗ trợ trong từ điển:**
  - Không hỗ trợ dấu gạch ngang (`-`), dấu chấm (`.`), khoảng trắng (` `).
  - Không hỗ trợ chữ cái tiếng Việt có dấu (`Đ`, `Ư`, `Ơ`,...).
  - Không hỗ trợ chữ thường (`a-z`).
- **Lưu ý thực tế từ dataset VNLP:**
  - VNLP chỉ xuất hiện 32 ký tự: `0123456789ABCDEFGHIKLMNPRSTUVXYZ`.
  - Các ký tự `J`, `O`, `Q`, `W` không có trong dữ liệu huấn luyện VNLP thực tế, và chữ `I` chỉ xuất hiện đúng 1 lần. Model hiện tại đã có vị trí cho 36 ký tự nhưng chưa được học phân phối đủ cho J, O, Q, W.

---

## 7. CHI TIẾT PLATE DETECTOR
- **Kiến trúc:** YOLO11n (Ultralytics).
- **Trọng số tối ưu:** `weights/best_license_plate_yolo11n_416.onnx` (10.02 MB).
- **Runtime triển khai:** OpenCV DNN (`cv2.dnn.readNetFromONNX`), backend `DNN_BACKEND_OPENCV`, target `DNN_TARGET_CPU`.
- **Cấu hình suy luận:**
  - Resolution: 416x416
  - Normalization: `scalefactor = 1/255.0, swapRB = True` (`[0.0, 1.0]`)
  - Confidence Threshold: `conf = 0.50`
  - NMS IoU Threshold: `iou = 0.45`
- **Chỉ số hiệu năng Detector gốc:**
  - Precision: **99.61%**
  - Recall: **99.18%**
  - mAP50: **99.49%**
  - mAP50-95: **70.42%**

---

## 8. TÌNH TRẠNG DATASET & XÁC MINH VNLP
- **Dataset OCR hiện tại:** VNLP (Vietnamese License Plate Dataset).
- **VNLP currently used:** **YES** (100% xác nhận).
- **Bằng chứng xác nhận:**
  1. `datasets/VNLP_detection/detection/VNLP_readme` ghi rõ tác giả `phamtheanh@hdu.edu.vn`.
  2. `datasets/VNLP-main/VNLP-main/readme.md` chứa citation bài báo MAPR2022 / The Visual Computer 2023.
  3. Cấu trúc tên file chứa tọa độ bbox và số lượng ảnh: 37,297 ảnh (19,086 one_row, 12,618 two_rows, 5,593 xe máy).
- **Dataset Detection:** Roboflow Vietnamese License Plate Dataset (8,357 ảnh: 5,845 train, 1,680 valid, 832 test).

---

## 9. TRAIN / VAL / TEST SPLITS & TEST_FINAL
- **Quy chuẩn chia tách:** Phân tầng theo layout, tỷ lệ 80/10/10, random seed = 42.
- **Số lượng mẫu:**
  - `TRAIN = datasets/VNLP_processed/train.csv`: 29,836 mẫu (80.0%)
  - `VAL = datasets/VNLP_processed/val.csv`: 3,728 mẫu (10.0%)
  - `TEST = datasets/VNLP_processed/test.csv`: 3,733 mẫu (10.0%)
  - `TEST_FINAL = datasets/VNLP_processed/test.csv` (ĐÃ KHÓA CỐ ĐỊNH).
- **Đánh giá rủi ro Leakage:**
  - Trùng lặp hash ảnh giữa Train và Test: 6 mẫu (0.16%).
  - Trùng lặp chuỗi ký tự biển (Text Overlap): 2,546 / 3,733 (68.20%) do cùng một xe được ghi hình ở nhiều góc trong VNLP.
  - Phân nhánh độc lập: `datasets/VNLP_processed/test_unseen_text.csv` có 1,187 mẫu hoàn toàn chưa từng xuất hiện trong tập Train.

---

## 10. CURRENT BENCHMARK (ĐO ĐẠC THỰC TẾ TRÊN TEST_FINAL)
Đánh giá chi tiết model `best_lprnet_vn.onnx` trên tập `TEST_FINAL` (3,733 mẫu):

| Chỉ số / Phân loại | Tesseract Baseline | LPRNet-Lite Hiện Tại (weights/ocr/best_lprnet_vn.onnx) |
| :--- | :--- | :--- |
| **Độ chính xác biển tổng thể (Exact Plate Acc)** | 14.00% | **85.27%** (3,183 / 3,733 mẫu đúng 100%) |
| **Biển 1 dòng (one_row)** | 26.75% | **90.58%** (1,730 / 1,910 mẫu) |
| **Biển 2 dòng ô tô (two_rows)** | 2.75% | **81.47%** (1,029 / 1,263 mẫu) |
| **Biển 2 dòng xe máy (motorcycle)** | 0.00% | **75.71%** (424 / 560 mẫu) |
| **Tập biển số mới hoàn toàn (Unseen Text)** | N/A | **72.11%** (856 / 1,187 mẫu) |
| **Độ chính xác ký tự (Character Accuracy)** | 44.43% | **95.68%** |
| **Character Error Rate (CER)** | 0.5557 | **0.0432** |
| **Latency Suy luận OCR (Laptop CPU)** | ~180 ms | **3.25 ms** (>300 FPS) |
| **Latency Suy luận OCR (Raspberry Pi 5 CPU)**| ~450 ms | **12.40 ms** (~80 FPS) |
| **Latency Pipeline E2E (Pi 5: Detect + OCR)** | ~550 ms | **31.70 ms** (~31.5 FPS) |

---

## 11. CURRENTLY WORKING (CÓ BẰNG CHỨNG THỰC TẾ)
1. Đã export và kiểm định thành công mô hình `best_lprnet_vn.onnx` (Opset 12, 3.04 MB), khớp 100% giữa PyTorch và OpenCV DNN.
2. Thuật toán `rectify_two_row_plate` bằng Sobel vertical gradient valley hoạt động ổn định trên cả biển 2 dòng ô tô và xe máy.
3. Bộ suy luận E2E thuần OpenCV DNN (`scripts/pi_recognize_image.py`, `scripts/pi_recognize_camera.py`) chạy mượt mà không cần cài PyTorch/Ultralytics trên môi trường biên.
4. Đã hiện thực hóa bộ lọc ổn định thời gian đa khung hình `PlateStabilizer` (`ocr/plate_stabilizer.py`), hỗ trợ IoU tracking và sliding-window majority voting triệt tiêu hiện tượng nhảy chữ (jitter).
5. Đã hoàn thành bộ test suite cho CTC decoder và PlateStabilizer (`tests/test_ctc_decoder.py`, `tests/test_plate_stabilizer.py`).

---

## 12. CURRENTLY NOT WORKING & LIMITATIONS
1. **Góc nghiêng và phối cảnh mạnh (Perspective Distortion):** Hiện tại pipeline chưa có Spatial Transformer Network (STN) hoặc Perspective Warp 4 điểm; các biển chụp xiên chéo > 25 độ có tỷ lệ nhận diện giảm rõ rệt.
2. **Ký tự hiếm:** Do dataset VNLP hoàn toàn thiếu các ký tự `J`, `O`, `Q`, `W` và chỉ có 1 mẫu chữ `I`, model hiện tại không thể nhận diện chính xác các biển đặc thù (biển quân sự, ngoại giao, xe cơ quan quốc tế mang ký tự đặc biệt).
3. **Phân biệt cặp ký tự dễ lẫn:** Tỷ lệ nhầm lẫn cao nhất trong 550 mẫu sai tập test tập trung vào các cặp: `8` ↔ `B`, `0` ↔ `D`, `5` ↔ `S`, `2` ↔ `Z`.
4. **Biển xe máy bị che khuất hoặc ốc bắt biển:** Các biển xe máy có ốc bắt biển đè lên chữ số ở hàng trên gây đứt nét dẫn đến nhận diện sai (chính xác xe máy hiện tại chỉ đạt 75.71%).

---

## 13. KNOWN ISSUES & TECHNICAL DEBT
1. **Hardcoded IP & Path trong các script remote:**
   - `tools/remote_exec.py`: Hardcoded `PI_HOST = "172.16.10.145"` và `KEY_FILE = r"C:\Users\Administrator\.ssh\id_rsa"`.
   - `scratch/execute_pi_ocr_pipeline.py`: Hardcoded `host = "192.168.2.43"`.
   - `scripts/benchmark_models.py` và `scripts/pi_verify_models.py`: Hardcoded đường dẫn `/home/cytron/vietnam_plate/...`.
   - `configs/license_plate.yaml`: Hardcoded đường dẫn Windows tuyệt đối `d:/Vietnam_Plate_Project/...`.
2. **Text Overlap trong VNLP:** 68.2% mẫu trong `test.csv` có chuỗi biển số xuất hiện trong tập `train.csv`. Mặc dù ảnh vật lý là khác nhau nhưng điều này có thể dẫn đến việc đánh giá độ chính xác cao hơn khả năng nhận diện thực tế trên môi trường chưa từng thấy (out-of-distribution).
3. **Thiếu hỗ trợ Hailo:** Mặc dù `README.md` đề cập tới Hailo-8, hiện tại toàn bộ code chỉ chạy OpenCV DNN CPU; chưa có file `.hef` và chưa có tích hợp HailoRT/Tappas.

---

## 14. EXPERIMENT HISTORY
1. **Tesseract OCR Baseline:** Đã thử nghiệm trên 3,733 ảnh test VNLP. Kết quả thất bại nặng nề trên biển 2 dòng (Exact Acc chỉ đạt 2.75% cho ô tô 2 dòng và 0% cho xe máy).
2. **YOLO11n Plate Detector:** Đã huấn luyện trên 8,357 ảnh, xuất thành công các biến thể ONNX ở kích thước 320, 416, 512, 640. Nghiệm thu kích thước 416x416 là điểm cân bằng hoàn hảo trên Pi 5 (Recall 89.09% tổng thể, 60.71% biển nhỏ, latency ~18.5 ms).
3. **LPRNet-Lite Architecture:** Đã thử nghiệm hội tụ qua các bước overfit 32 mẫu, smoke test 1,000 mẫu và huấn luyện chính thức 60 epochs với Cosine Annealing LR và Mixed Precision. Kết quả nâng độ chính xác tổng thể từ 14.0% lên 85.27%.
4. **Hai dòng Rectification:** Đã so sánh việc đọc trực tiếp crop 2 dòng so với thuật toán cắt Sobel + ghép ngang. Việc cắt ghép ngang giúp mạng 1D CTC đọc được cả 2 dòng biển số một cách liền mạch với chi phí tiền xử lý chỉ tốn ~0.8 ms.

---

## 15. IMPORTANT FILES MAPPING

| File Path | Vai trò & Trách nhiệm | Trạng thái |
| :--- | :--- | :--- |
| `weights/ocr/best_lprnet_vn.onnx` | Model ONNX OCR chính thức triển khai | **ACTIVE** |
| `weights/best_license_plate_yolo11n_416.onnx` | Model ONNX Detector chính thức triển khai | **ACTIVE** |
| `ocr/pi_lprnet_ocr.py` | Engine suy luận OCR thuần OpenCV DNN | **ACTIVE** |
| `detector/pi_yolo_detector.py` | Engine suy luận Detector thuần OpenCV DNN | **ACTIVE** |
| `ocr/preprocess.py` | Pipeline tiền xử lý, letterbox, và Sobel rectification | **ACTIVE** |
| `ocr/charset.py` | Định nghĩa từ điển 36 ký tự và CTC Blank | **ACTIVE** |
| `ocr/ctc_decoder.py` | CTC Greedy decoder hỗ trợ NumPy và PyTorch | **ACTIVE** |
| `ocr/plate_stabilizer.py` | Module khử rung lắc ký tự và voting thời gian thực | **ACTIVE** |
| `scripts/pi_recognize_camera.py` | Pipeline E2E hoàn chỉnh cho Webcam / Camera USB | **ACTIVE** |
| `scripts/pi_recognize_image.py` | Pipeline E2E cho ảnh tĩnh và thư mục ảnh | **ACTIVE** |
| `scripts/train_ocr_lprnet.py` | Script huấn luyện LPRNet-Lite trên GPU | **ACTIVE** |
| `tools/prepare_vnlp.py` | Script parse và chia tách dataset VNLP 80/10/10 | **ACTIVE** |
| `tools/audit_ocr_manifests.py` | Script kiểm tra leakage và sinh `test_unseen_text.csv` | **ACTIVE** |
| `tools/evaluate_ocr_model.py` | Script đánh giá chuyên sâu và so sánh baseline | **ACTIVE** |
| `tools/export_and_verify_ocr_onnx.py`| Script xuất ONNX và kiểm thử sai số PyTorch vs OpenCV | **ACTIVE** |
| `configs/ocr_lprnet.yaml` | File cấu hình huấn luyện OCR | **ACTIVE** |
| `configs/license_plate.yaml` | File cấu hình dataset Detector | **ACTIVE** |
| `scripts/ocr_tesseract.py` | Module Tesseract OCR cũ | LEGACY |
| `scripts/crop_plate.py` | Script crop biển số cơ bản bằng Ultralytics | LEGACY |

---

## 16. HAILO HARDWARE STATUS
- **Hardware detected:** Không phát hiện thiết bị Hailo trên máy trạm Windows host hiện tại.
- **HailoRT:** Chưa được cài đặt trên host và không có thư viện trong `.venv`.
- **HEF Models:** Không có bất kỳ file `.hef` nào trong repository.
- **Trạng thái thực tế:** Toàn bộ pipeline hiện tại được thiết kế chạy **100% CPU** thông qua tối ưu hóa tập lệnh OpenCV DNN (NEON trên Raspberry Pi 5 và AVX2 trên Intel x86_64).

---

## 17. DATASET EXPANSION STATUS

### PBL4:
- **Downloaded:** YES (`datasets_external/pbl4/PBL4_LPRNet/Dataset/BiensoxeVietNam/`)
- **Original images:** 13,320 images (1,432 original unique scene samples: 1,232 train, 200 valid)
- **Exact duplicate:** 128 images internal exact duplicates (0 duplicate with VNLP)
- **Near duplicate:** 9,845 images (pHash Hamming distance <= 6)
- **Review flagged:** 2,136 images (pHash distance 7-10 with same label)
- **Test leakage:** 0 images (0 duplicate with `TEST_FINAL`)
- **Unique usable:** 1,211 images (saved to `dataset_staging/pbl4_unique/`)
- **Quarantined:** 12,109 images (saved to `dataset_staging/quarantine_duplicates/`)

### Synthetic generator:
- **Downloaded:** YES (`datasets_external/vn_plate_generator/Vietnamese-License-Plate-Generator/`)
- **Environment:** Dedicated adapter script (`scripts/dataset_tools/generate_lprnet_vn.py`), Pillow 12+ compatible
- **Smoke test:** 100 images generated into `dataset_staging/synthetic/`
- **Valid:** 100 / 100 (100.0%)
- **Invalid:** 0 / 100 (0.0%)
- **CHARS compatibility:** 100% compliant with 36-character alphanumeric charset + CTC blank, successfully injected rare letters `J, O, Q, W`

### DATASET V2 CANDIDATES
- **Current VNLP Train:** 29,836 samples (`datasets/VNLP_processed/train.csv`)
- **PBL4 unique usable:** 1,211 samples (`dataset_staging/pbl4_unique/`)
- **Synthetic generated:** 100 smoke-tested samples (ready to expand to 5,000 - 10,000 samples)
- **Self-captured if any:** 0 (none currently available)
- **Locked Test Set:** 3,733 samples (`TEST_FINAL` in `datasets/VNLP_processed/test.csv`)

### READY FOR DATASET V2
- **READY FOR DATASET STAGING:** **YES**
- **READY FOR TRAINING V2 NOW:** **NO**
- **Reason:** Toàn bộ công đoạn kiểm định duplicate, cách ly rò rỉ tập test và staging dữ liệu đã hoàn tất 100% (không có rò rỉ vào `TEST_FINAL`). Tuy nhiên, theo quy tắc an toàn bất biến của dự án: "KHÔNG RETRAIN MODEL TRONG NHIỆM VỤ NÀY" và "KHÔNG OVERWRITE MODEL HIỆN TẠI". Quá trình huấn luyện Model V2 sẽ được triển khai trong một phiên làm việc riêng sau khi người dùng và ChatGPT phân tích snapshot báo cáo này.

---

## 18. RECOMMENDED NEXT STEPS
1. **Duyệt qua báo cáo snapshot:** Gửi toàn bộ nội dung file `reports/project_state_for_chatgpt.md` cho ChatGPT hoặc kỹ sư phụ trách để thống nhất kiến trúc và chiến lược augmentation.
2. **Sinh thêm dữ liệu tổng hợp mục tiêu:** Chạy `scripts/dataset_tools/generate_lprnet_vn.py` để sinh thêm khoảng 5,000 - 8,000 ảnh tập trung vào các trường hợp hiện tại còn yếu (xe máy 2 dòng, góc nghiêng nhẹ, biển mờ, ký tự hiếm `J, O, Q, W`).
3. **Thực hiện huấn luyện LPRNet-Lite V2:** Sử dụng script `scripts/train_ocr_lprnet.py` với cấu hình kết hợp VNLP Train (29,836) + PBL4 Unique (1,211) + Synthetic Expanded (~5,000). Đánh giá độc lập trên `TEST_FINAL` (3,733 ảnh) để đo lường mức cải thiện so với mốc baseline 85.27%.
4. **Tối ưu hóa Edge Deployment:** Khảo sát bổ sung bộ nắn góc phối cảnh nhẹ (Perspective Rectification / Spatial Transformer) trước khi đưa vào OpenCV DNN trên Raspberry Pi 5.
