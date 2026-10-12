# PHÂN TÍCH MÂU THUẪN CHỈ SỐ TEST_UNSEEN_TEXT CỦA MODEL V1
*(UNSEEN METRIC DISCREPANCY ANALYSIS REPORT)*

**Thời gian thực hiện:** 2026-09-24  
**Tập dữ liệu kiểm tra:** `datasets/VNLP_processed/test_unseen_text.csv`  
**Model đối chứng:** `weights/ocr/best_lprnet_vn.pth` & `weights/ocr/best_lprnet_vn.onnx`

---

## 1. Kiểm Toán Tệp Manifest `test_unseen_text.csv`
- **Đường dẫn tuyệt đối:** `D:\Vietnam_Plate_Project\datasets\VNLP_processed\test_unseen_text.csv`
- **Mã băm SHA256:** `243aea90e30940873407ec98505a7412a7b05fa32f0210a36ec37113c234366e`
- **Tổng số dòng dữ liệu:** `1,187` mẫu (khớp 100% với định nghĩa ban đầu)
- **Số lượng ảnh tồn tại trên đĩa:** `1,187 / 1,187` ảnh (100% hợp lệ, không thiếu file)
- **Số chuỗi biển số duy nhất (Unique Labels):** `1,085` nhãn
- **Số chuỗi biển số xuất hiện trong `train.csv`:** `0` (100% chuỗi ký tự biển số chưa từng xuất hiện trong tập Train)
- **Số lượng ảnh trùng lặp nhị phân (Bitwise Duplicate) với `train.csv`:** `1` mẫu (đây là 1 trong 6 mẫu exact duplicate giữa VNLP Train và Test đã được phân lập ở Phase 2)
- **Phân phối kiểu biển (Layout Distribution):**
  - Biển 1 dòng (`one_row`): `696` mẫu (58.64%)
  - Biển 2 dòng ô tô (`two_rows`): `281` mẫu (23.67%)
  - Biển 2 dòng xe máy (`two_rows_label_xe_may`): `210` mẫu (17.69%)
  - Tổng cộng: `1,187` mẫu.
- **Kiểm tra các phiên bản khác:** Không tồn tại bất kỳ file manifest nào khác (`test_unseen*`, `unseen*`, `backup*`) trong thư mục dataset.

---

## 2. Giải Trình 7 Câu Hỏi Trọng Tâm

### Câu hỏi 1: Vì sao từng có kết quả 72.11% (856 / 1,187)?
- **Nguồn gốc lịch sử:** Con số `72.11% (856 / 1,187)` xuất hiện lần đầu tiên tại dòng 194 của báo cáo `reports/project_state_for_chatgpt.md` được tạo trong Phase 1.
- **Bằng chứng từ Transcript và Execution Logs:**
  - Kiểm tra toàn bộ 185 bước đầu tiên của transcript cuộc trò chuyện cho thấy: trong suốt Phase 1, script `tools/evaluate_ocr_model.py` **chưa từng được chạy** với tham số `--unseen_csv`.
  - Trong Phase 1, trợ lý AI chỉ đọc tệp nhật ký lỗi có sẵn `outputs/lprnet_errors.csv` (chứa 550 mẫu sai của toàn bộ `test.csv`) và tập `scratch/laptop_predictions.json` (60 mẫu test ngẫu nhiên).
  - Khi soạn thảo bảng tổng kết Phase 1 tại Step 180, con số `72.11% (856 / 1,187)` đã được đưa vào như một ước tính lý thuyết/tổng hợp thủ công mà không có lệnh thực thi (execution run) trực tiếp trên file manifest này.

### Câu hỏi 2: Vì sao hiện tại đo đạc thực tế là 85.93% (1,020 / 1,187)?
- **Bằng chứng số học tuyệt đối từ tệp lỗi gốc:**
  - Tệp `outputs/lprnet_errors.csv` (được sinh ra khi đánh giá mô hình V1 trên toàn bộ 3,733 mẫu của `test.csv`) ghi nhận chính xác **550** mẫu sai của Model V1.
  - Khi đối chiếu đường dẫn của 550 mẫu sai này với 1,187 mẫu của `test_unseen_text.csv`, chỉ có chính xác **167** mẫu sai nằm trong tập `test_unseen_text.csv`.
  - Số mẫu đoán đúng thực tế: $1,187 - 167 = 1,020$ mẫu đúng.
  - Tỷ lệ chính xác thực tế: $\frac{1,020}{1,187} = 85.9309\% \approx \mathbf{85.93\%}$.
- **Bằng chứng thực thi độc lập:**
  - Chạy `tools/evaluate_ocr_model.py` trên PyTorch CUDA: kết quả trả về `1,020 / 1,187` (**85.93%**, CER: 0.0276, Char Acc: 97.24%).
  - Chạy `scripts/dataset_tools/benchmark_v1_vs_v2a.py` trên OpenCV DNN CPU: kết quả trả về `1,020 / 1,187` (**85.93%**, CER: 0.0276, Char Acc: 97.24%).

### Câu hỏi 3: Có phải cùng manifest không?
- **CÓ.** File `datasets/VNLP_processed/test_unseen_text.csv` có SHA256 cố định và không thay đổi bất kỳ dòng nào kể từ khi được sinh bởi `tools/audit_ocr_manifests.py`.

### Câu hỏi 4: Có phải cùng model không?
- **CÓ.** Cả hai lần đều tham chiếu đến `best_lprnet_vn.pth` (PyTorch) và bản xuất `best_lprnet_vn.onnx` (OpenCV DNN) với SHA256 bất biến `25d2ff3c...` và `6add2243...`.

### Câu hỏi 5: Có phải cùng preprocessing không?
- **CÓ.** Cùng sử dụng pipeline chuẩn:
  1. Crop bounding box tỷ lệ mở rộng 3% (`pad_ratio=0.03`).
  2. Nắn chỉnh biển 2 dòng bằng thuật toán thung lũng gradient Sobel đứng (`rectify_two_row_plate`).
  3. Letterbox về kích thước `160 x 48` với viền trắng `(255, 255, 255)`.
  4. Chuẩn hóa RGB về đoạn `[-1.0, 1.0]`.

### Câu hỏi 6: Có phải cùng decoder không?
- **CÓ.** Cả hai đều dùng giải mã CTC Greedy Decoding (`ctc_greedy_decode_torch` và `ctc_greedy_decode_numpy`) với blank index = 36.

### Câu hỏi 7: Metric nào mới là baseline hợp lệ?
- **KẾT LUẬN:** **85.93% (1,020 / 1,187)** là baseline thực tế duy nhất hợp lệ, có thể tái lập 100% bằng code và dữ liệu gốc. Con số 72.11% là một sai số ghi chép trong tài liệu markdown của Phase 1.

---

## 3. Bảng Đối Chiếu Hiện Trạng Baseline V1
| Phân loại | Tỷ lệ chính xác (Exact Acc) | Số mẫu đúng / Tổng | Ghi chú |
| :--- | :---: | :---: | :--- |
| **TEST_FINAL Tổng thể** | **85.27%** | 3,183 / 3,733 | 550 lỗi |
| **Phần biển đã gặp chữ (Seen Text)** | **84.96%** | 2,163 / 2,546 | 383 lỗi |
| **Phần biển chưa gặp chữ (Unseen Text)** | **85.93%** | 1,020 / 1,187 | 167 lỗi |

*Nhận định AI:* Model V1 có độ chính xác trên tập biển số chưa từng gặp (85.93%) tương đương, thậm chí hơi nhỉnh hơn so với tập biển số đã từng gặp trong train (84.96%). Điều này chứng minh LPRNet-Lite V1 không bị overfit thuộc lòng chuỗi ký tự, mà đã học được đặc trưng hình học ký tự thực sự trên phân phối của tập VNLP.
