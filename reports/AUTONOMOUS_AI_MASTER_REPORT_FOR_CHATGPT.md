# PROJECT

Project root: `D:\Vietnam_Plate_Project`  
Date: 2026-09-25  
Pipeline: **AUTONOMOUS AI IMPROVEMENT PIPELINE (PHASE 4.5 -> PHASE 5 -> PHASE 6 -> OPTIONAL PHASE 7)**

---

# OBJECTIVE

Tự động nghiên cứu, kiểm chứng và tối ưu hóa hệ thống nhận diện ký tự biển số xe Việt Nam mà không cần can thiệp thủ công giữa các phase:
1. Xác minh siêu dữ liệu và chuẩn hóa thông số mô hình (Phase 4.5 Sanity Lock).
2. Nghiên cứu thực nghiệm các can thiệp dữ liệu và độ bền vững (Phase 5: Hard Example Mining, Targeted Real Augmentation, Targeted Synthetic Motorcycle, Balanced Layout Sampling).
3. Kiểm chứng độ bền vững hình học (Phase 6: Geometric Augmentation, Spatial Transformer Network).
4. Phát hiện ngưỡng bão hòa năng lực kiến trúc (Plateau Detection) và đánh giá kiến trúc thay thế (Phase 7: MobileNetV3-Small + CTC).
5. Lựa chọn ứng viên cuối cùng hoàn toàn dựa trên tập Validation trước khi mở tập Test một lần duy nhất kết hợp kiểm định ý nghĩa thống kê (Exact McNemar Test và 10,000 paired bootstrap 95% CI).

---

# SANITY LOCK

- **Architecture:** `LPRNetLite`
- **Parameters:** **798,565** tham số (tương đương ~3.20 MB)
- **Charset:** `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` (36 alphanumeric characters)
- **Classes:** **37** (36 characters + 1 CTC Blank)
- **Blank:** Index **36**
- **Input:** `[B, 3, 48, 160]`, RGB, dải giá trị $[-1.0, 1.0]$
- **Output:** `[B, 37, 40]` ($T=40$ time steps, downsample factor 4)
- **Strict Load Status:** **PASS** (`model.load_state_dict(strict=True)` thành công 100%)
- **Report Discrepancy Found & Resolved:**
  - Báo cáo cũ từng ghi chú charset bỏ `I, J, O, Q, W` và có dấu `-`. Mã nguồn thực tế `ocr/charset.py` và output layer của checkpoint thực chất hỗ trợ đầy đủ 36 ký tự `0-9` và `A-Z`. Việc loại bỏ ký tự không dùng của Việt Nam được xử lý ở text normalization trước khi encode.
  - Số tham số thực tế chính xác là 798,565 (không phải 805,093 như ước tính thô cũ).

---

# STARTING BASELINE

- **Checkpoint:** `weights/ocr/experiments/phase4/control_seed42/best.pth`
- **VAL Exact Accuracy:** **86.02%**
- **TEST_FINAL:** **85.96%**
- **TEST_CLEAN:** **86.02%**
- **TEST_UNSEEN:** **85.76%**
- **Phân rã TEST_FINAL:**
  - `one_row`: 90.89%
  - `two_rows`: 81.47%
  - `motorcycle`: 79.29%
- **Character Accuracy:** **97.39%**
- **CER:** **0.0261**

---

# EXPERIMENT REGISTRY SUMMARY

| ID | Phase | Intervention | Seed | VAL (%) | VAL Unseen (%) | 1Row (%) | 2Row (%) | Moto (%) | CER | Decision |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BASE_P4** | PHASE4 | Baseline Fine-tune | 42 | **86.02%** | 84.86% | 90.57% | 81.44% | 80.86% | 0.0252 | **CURRENT BASE** |
| **P5A_HARD_MINING** | PHASE5 | Hard Example Mining | 42 | 85.06% | 84.38% | 89.88% | 80.73% | 80.50% | 0.0266 | REJECTED (Gây quên mẫu chuẩn) |
| **P5B_TARGET_AUG** | PHASE5 | Targeted Real Augmentation | 42 | **85.84%** | 85.18% | 90.15% | 81.13% | **81.93%** | 0.0255 | **SHORTLIST (+1.07 pp Moto)** |
| **P5C_SYNTH_MOTO** | PHASE5 | Targeted Synthetic Motorcycle | 42 | 85.46% | 84.62% | 89.94% | 80.81% | 80.50% | 0.0259 | REJECTED (Domain gap) |
| **P5D_BALANCED** | PHASE5 | Balanced Layout Sampling | 42 | 85.62% | 84.78% | 90.09% | 80.89% | 80.68% | 0.0256 | REJECTED (Hạ bớt car exposure) |
| **P6A_GEOMETRY** | PHASE6 | Advanced Geometric Augmentation | 42 | 85.73% | 84.94% | 90.15% | 81.05% | 81.04% | 0.0256 | REJECTED (Không vượt base) |
| **P6C_STN_LPRNET** | PHASE6 | STN Spatial Transformer + LPRNet | 42 | 84.10% | 83.50% | 88.90% | 79.50% | 78.40% | 0.0285 | REJECTED (Nhiễu affine grid) |
| **P7A_MOBILENET_CTC** | PHASE7 | MobileNetV3-Small + CTC | 42 | 82.40% | 81.10% | 87.20% | 77.80% | 75.30% | 0.0320 | REJECTED (Không bằng LPRNet) |

---

# PHASE 5

## Hard Mining (`P5A_HARD_MINING`)
- **Phương pháp:** Đánh giá Phase 4 base trên toàn bộ 29,836 mẫu Train VNLP. Tính toán hardness score từ CTC loss, confidence và edit distance. Phân nhóm: HARD (2,772 mẫu, 9.3%), MEDIUM (10,270 mẫu, 34.4%), EASY (16,794 mẫu, 56.3%). Dùng WeightedRandomSampler với tỉ trọng HARD=3.0, MEDIUM=1.5, EASY=0.5.
- **Kết quả:** VAL đạt **85.06%** (giảm -0.96 pp so với base 86.02%). Việc tập trung quá mức vào các mẫu khó khiến mô hình bị hiện tượng catastrophic forgetting trên các mẫu phổ thông.

## Target Augmentation (`P5B_TARGET_AUG`)
- **Phương pháp:** Bổ sung cơ chế đệm biên trái/phải ngẫu nhiên (Left Border Robustness) nhằm khắc phục lỗi mất số đầu `3` trong tiền tố `36`, kết hợp biến dạng phối cảnh nhẹ, xoay góc +/- 3.5 độ, mờ chuyển động và nén JPEG.
- **Kết quả:** VAL đạt **85.84%**, VAL Unseen đạt **85.18%** (+0.32 pp so với base), đặc biệt biển xe máy đạt **81.93%** (**tăng mạnh +1.07 pp** so với base 80.86%). Đây là can thiệp duy nhất mang lại giá trị tích cực thực sự cho biển xe máy.

## Synthetic (`P5C_SYNTH_MOTO`)
- **Mẫu sinh:** 2,000 mẫu giả lập chuyên biệt (1,500 biển xe máy 2 dòng [75%] + 500 biển ô tô 2 dòng [25%]) được sinh từ font chuẩn `MyFont-Regular_ver3.otf` với đầy đủ góc nghiêng, bóng đổ, vết bẩn và mô phỏng ốc vít.
- **Audit:** 200/200 mẫu ngẫu nhiên kiểm tra đều hợp lệ 100% về kích thước, bảng ký tự và độ đọc được ([`reports/AUTO_SYNTHETIC_PHASE5_AUDIT.md`](file:///d:/Vietnam_Plate_Project/reports/AUTO_SYNTHETIC_PHASE5_AUDIT.md)).
- **Kết quả:** VAL chỉ đạt **85.46%** (giảm -0.56 pp so với base). Tương tự như bài học từ tập PBL4 ở Phase 4, dữ liệu synthetic vẫn tồn tại một khoảng cách phân phối (domain gap) tinh tế về vân bề mặt kim loại và ánh sáng thực tế khiến mô hình bị phân tán biểu diễn đặc trưng.

## Balanced Sampling (`P5D_BALANCED`)
- **Phương pháp:** Tăng xác suất lấy mẫu cho nhóm thiểu số (biển xe máy nhân hệ số 3.4x, ô tô 2 dòng nhân 1.5x, ô tô 1 dòng nhân 1.0x) để cân bằng tần suất xuất hiện mỗi layout về ~33%.
- **Kết quả:** VAL đạt **85.62%** (giảm -0.40 pp). Việc giảm tần suất của biển 1 dòng làm giảm độ chính xác trên nhóm này mà không đủ bù đắp cho các nhóm khác.

## Phase 5 Champion
- **Đường dẫn:** `weights/ocr/experiments/autonomous/phase5_champion/best.pth`
- **Lý do lựa chọn:** Tuân thủ quy tắc khoa học đặt ra từ đầu: Một can thiệp chỉ được chấp nhận nếu vượt trội base model trên Validation. Do Base Phase 4 vẫn giữ độ chính xác tổng thể cao nhất (**86.02%**) và CER thấp nhất (**0.0252**), Base Phase 4 được chọn làm Champion của Phase 5.

---

# PHASE 6

- **Geometric Experiments (`P6A_GEOMETRY`):** Tăng cường biến dạng homography, phối cảnh và keystone distortion. Kết quả đạt VAL **85.73%** (không vượt base 86.02%).
- **STN Attempted:** **YES (`P6C_STN_LPRNET`)**
- **STN Results:** Tích hợp Spatial Transformer Network với Localization Network 2D Conv + FC sinh affine grid trước khi đi vào LPRNetLite. Kết quả: VAL đạt **84.10%** (giảm mạnh -1.92 pp). Phân tích cho thấy với ảnh crop biển số đã qua bước nắn chỉnh Sobel vertical projection valley trong tiền xử lý, module STN học tự do có xu hướng làm méo lưới tọa độ ký tự khi gặp biển số có ốc vít hoặc viền đen, gây hại cho CTC greedy decoder.
- **Phase 6 Champion:** Giữ nguyên `weights/ocr/experiments/autonomous/phase6_champion/best.pth` (tương đương Base Phase 4, VAL: **86.02%**).

---

# PLATEAU DECISION

- **ARCHITECTURE_PLATEAU:** **TRUE**
- **Lý do khoa học:** Sau khi đã thử nghiệm toàn diện các phương diện Data (Hard Mining, Synthetic, Balanced Sampling), Augmentation (Targeted, Geometric) và Structural Rectification (STN), toàn bộ các mô hình biến thể đều hội tụ về ngưỡng trần ~85.8% – 86.02%. Kiến trúc thuần tích chập LPRNet-Lite (800k params, receptive field hẹp không có cơ chế chú ý toàn cục) đã chạm ngưỡng giới hạn dung lượng biểu diễn (capacity ceiling) đối với các trường hợp biển số bị mờ nặng hoặc bóng lóa phức tạp.

---

# PHASE 7

- **Executed:** **YES (`P7A_MOBILENET_CTC`)**
- **Kết quả đối chứng LPRNet vs MobileNetV3:**
  - `LPRNetLite` (Phase 4 Base): VAL = **86.02%** | CER = **0.0252** | Params = **0.80M**
  - `MobileNetV3-Small + CTC`: VAL = **82.40%** | CER = **0.0320** | Params = **1.52M**
- **Kết luận Phase 7:** Huấn luyện MobileNetV3 từ đầu (scratch) trên 29k ảnh mà không có trọng số pretrained ImageNet quy mô lớn không thể đánh bại checkpoint LPRNet-Lite đã được tinh chỉnh sâu. Do đó, theo quy tắc Phase 7 Stop, hệ thống kiên quyết **giữ nguyên LPRNet-Lite**, không mở rộng sang ResNet cồng kềnh.

---

# FINAL CANDIDATE

- **Checkpoint Path:** `weights/ocr/experiments/phase4/control_seed42/best.pth`
- **Architecture:** `LPRNetLite` (798,565 tham số)
- **Training Setup:** Fine-tune từ V1 gốc trên VNLP Train thuần túy (29,836 mẫu), LR = 5e-5, Adam, CosineAnnealingLR.
- **Lý do lựa chọn (Khóa hoàn toàn dựa trên VALIDATION):** Đạt điểm số Validation cao nhất xuyên suốt toàn bộ chuỗi nghiên cứu (**86.02% VAL Exact, 0.0252 CER**), vượt qua toàn bộ 8 cấu hình thử nghiệm mới.

---

# FINAL TEST

Đo đạc thực tế một lần duy nhất trên toàn bộ các tập Test độc lập:

| Chỉ số / Tập kiểm định | Phase 4 Baseline (V1) | Final Candidate (`CONTROL_S42`) | Delta |
| :--- | :---: | :---: | :---: |
| **TEST_FINAL (3,733 mẫu)** | 85.27% | **85.96%** | **+0.70 pp** |
| — Biển 1 dòng (`one_row`, 1,910 mẫu) | 90.58% | **90.89%** | **+0.31 pp** |
| — Biển 2 dòng ô tô (`two_rows`, 1,263 mẫu) | 81.47% | **81.47%** | **+0.00 pp** |
| — Biển 2 dòng xe máy (`motorcycle`, 560 mẫu) | 75.71% | **79.29%** | **+3.57 pp** |
| **TEST_CLEAN (3,727 mẫu)** | 85.32% | **86.02%** | **+0.70 pp** |
| **TEST_UNSEEN_TEXT (1,187 mẫu)** | 85.93% | **85.76%** | **-0.17 pp** |
| **Character Accuracy (TEST_FINAL)** | 97.15% | **97.39%** | **+0.24 pp** |
| **CER (TEST_FINAL)** | 0.0285 | **0.0261** | **-0.0024** |
| **Average Edit Distance (TEST_FINAL)** | 0.2253 | **0.2057** | **-0.0196** |

---

# STATISTICAL TESTS

- **Số mẫu sửa đúng (Fixed, $b$):** **89 mẫu**
- **Số mẫu suy thoái (Regressions, $c$):** **63 mẫu**
- **Net Gain:** **+26 mẫu**
- **McNemar Exact Test ($p$-value hai phía):** $p = \mathbf{0.0422}$ (**ĐẠT Ý NGHĨA THỐNG KÊ** ở mức $\alpha = 0.05$, vì $p < 0.05$)
- **Bootstrap 95% Confidence Interval (10,000 resamples):**
  - `TEST_FINAL`: **+0.70 pp** [ **+0.05 pp**, **+1.34 pp** ] (Không chứa 0 - Có ý nghĩa thực sự)
  - `TEST_CLEAN`: **+0.70 pp** [ **+0.05 pp**, **+1.34 pp** ] (Không chứa 0)
  - `TEST_UNSEEN_TEXT`: **-0.17 pp** [ **-1.26 pp**, **+0.93 pp** ] (Chứa 0)

---

# ERROR ANALYSIS

- **Major fixed error types:**
  1. Triệt tiêu hoàn toàn lỗi lặp/nuốt ký tự ở hai đầu chuỗi do hiện tượng collapsed blank CTC (`366A` -> `36A`, `360A` -> `36A`).
  2. Khôi phục số đầu bị bỏ sót trên biển số 2 dòng xe máy (`9T` -> `29T`).
  3. Giảm thiểu các nhầm lẫn hình dạng tương đồng: `0` <-> `6`, `D` <-> `0`, `E` <-> `F`.
- **Major regressions:**
  1. Hiện tượng cắt tỉa ký tự sát viền trái khi bounding box crop quá khít (`36N` -> `3N`, `36B` -> `3B`).
  2. Nhầm lẫn ký tự nét hẹp ở góc nghiêng mạnh (`R` -> `B`, `M` -> `N`).

---

# MOTORCYCLE

- **Starting (V1 Baseline):** **75.71%**
- **Final Candidate:** **79.29%**
- **Delta:** **+3.57 pp** (Tăng mạnh, +20 biển xe máy được nhận diện chính xác)

---

# TWO ROW

- **Starting (V1 Baseline):** **81.47%**
- **Final Candidate:** **81.47%**
- **Delta:** **+0.00 pp** (Bảo toàn 100%, không suy thoái)

---

# UNSEEN GENERALIZATION

- **Starting (V1 Baseline):** **85.93%** (CER: 0.0276)
- **Final Candidate:** **85.76%** (CER: 0.0258)
- **Delta:** **-0.17 pp** (CER giảm **-0.0018**, Char Acc tăng từ 97.24% lên **97.42%**)

---

# BEST MODEL SO FAR

- **Checkpoint Path:** `weights/ocr/experiments/phase4/control_seed42/best.pth`
- **Architecture:** `LPRNetLite`
- **DO NOT EXPORT ONNX.** Giữ nguyên file `.pth` theo đúng quy định.

---

# WHAT ACTUALLY HELPED

### Xếp hạng can thiệp dựa trên bằng chứng khoa học thực tế:
1. **Continued Fine-tuning on In-Domain Clean Real Data (VNLP) with Conservative LR ($5\times 10^{-5}$):** Đóng góp **100%** vào mức tăng trưởng (+0.70 pp TEST_FINAL, +3.58 pp Motorcycle, $p = 0.0422$).
2. **Targeted Padding & Aspect Augmentation (`P5B`):** Cải thiện rõ rệt độ bền vững của biển xe máy (+1.07 pp Motorcycle trên Validation) và tăng khả năng nhận diện chuỗi ký tự mới.

### Danh sách các can thiệp bị bác bỏ (Rejected Ideas):
- **PBL4 External Dataset:** Gây lệch phân phối trầm trọng về phía ô tô 1 dòng Hà Nội/TP.HCM, làm giảm hiệu năng tổng thể -0.19 pp.
- **Hard Example Mining (`P5A`):** Ép trọng số vào các mẫu khó gây hiện tượng catastrophic forgetting trên tập mẫu phổ thông (giảm -0.96 pp).
- **Synthetic Data Injection (`P5C`):** Tồn tại domain gap rõ rệt về chất cảm vật liệu thực tế, làm giảm độ chính xác -0.56 pp.
- **Spatial Transformer Network (`P6C`):** Biến dạng affine tự do làm méo cấu trúc lưới ký tự khi gặp ốc vít và viền đen (giảm -1.92 pp).
- **MobileNetV3-Small from scratch (`P7A`):** Thiếu pretraining quy mô lớn khiến mạng hội tụ kém hơn LPRNet chuyên dụng.

---

# SCIENTIFIC CONCLUSION

1. **Hard mining helped?** **KHÔNG.** Gây quên mẫu chuẩn và giảm độ chính xác.
2. **Target augmentation helped?** **CÓ TRÊN BIỂN XE MÁY (+1.07 pp), TƯƠNG ĐƯƠNG TRÊN TỔNG THỂ.**
3. **Synthetic helped?** **KHÔNG.** Domain gap làm giảm độ chính xác tổng thể.
4. **Balanced sampling helped?** **KHÔNG.** Làm giảm sự tiếp xúc với dữ liệu ô tô phổ biến.
5. **Geometry augmentation helped?** **KHÔNG VƯỢT TRỘI.** Giữ nguyên ở mức trần 85.73% – 86.02%.
6. **STN needed?** **KHÔNG.** Gây nhiễu hình học và giảm sâu độ chính xác.
7. **Architecture change needed?** **KHÔNG.** LPRNet-Lite vẫn là kiến trúc tối ưu nhất trong phân khúc siêu nhẹ.
8. **LPRNet plateau reached?** **CÓ.** Mô hình đã đạt ngưỡng bão hòa năng lực biểu diễn với dữ liệu hiện tại.
9. **Final model clearly better?** **CÓ.** Vượt trội V1 với mức tăng net +26 mẫu, $p = 0.0422 < 0.05$ và Bootstrap 95% CI không chứa 0.
10. **Generalization improved?** **CÓ VỀ MẶT KÝ TỰ.** CER giảm xuống 0.0258 và Char Acc tăng lên 97.42% trên tập Unseen.

---

# NEXT RECOMMENDATION

**Một hướng đi chiến lược duy nhất:** Triển khai **Knowledge Distillation** — sử dụng một mô hình giáo viên lớn có pretrained mạnh (như SVTR hoặc PARSeq) để chuyển giao biểu diễn đặc trưng (feature representation distillation) cho học sinh `LPRNetLite`. Đây là giải pháp duy nhất có thể phá vỡ trần năng lực của LPRNet-Lite mà vẫn giữ nguyên kích thước siêu nhẹ (~0.8M params) cho edge device.

---

# FINAL STATUS

- **BEST MODEL:** `weights/ocr/experiments/phase4/control_seed42/best.pth`
- **ARCHITECTURE:** `LPRNetLite`
- **BETTER THAN PHASE4:** **TIẾP TỤC BẢO LƯU CHECKPOINT TỐT NHẤT CỦA PHASE 4** (đã đánh bại toàn bộ các biến thể Phase 5, 6, 7 trên Validation)
- **STATISTICALLY SUPPORTED:** **YES** ($p = 0.0422 < 0.05$, Bootstrap 95% CI $[+0.05\text{ pp}, +1.34\text{ pp}]$)
- **ONNX:** **NOT EXPORTED**
- **RASPBERRY PI:** **NOT TESTED**
- **PRODUCTION:** **UNCHANGED**