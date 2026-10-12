# TRẠNG THÁI DỰ ÁN SAU THÍ NGHIỆM V2-A (PROJECT STATE AFTER PHASE 2 / V2-A)

**Thời gian cập nhật:** 2026-09-24  
**Giai đoạn:** Hoàn tất Phase 2 (Thí nghiệm V2-A kiểm tra giả thuyết dữ liệu thực tế PBL4)

---

## 1. Trạng Thái Mô Hình và Production
- **Production Active Model:** `weights/ocr/best_lprnet_vn.onnx` (Model V1 - **GIỮ NGUYÊN 100%**)
  - SHA256: `6add22435d43c5d340ae775a0801a4cb106499b9795cbd181fcc8a8c2221258b`
  - PyTorch Checkpoint: `weights/ocr/best_lprnet_vn.pth`
  - SHA256: `25d2ff3cba0869766a96689a8eda2d1b133634c86023df01e2e41cf17491305c`
  - Độ chính xác TEST_FINAL: **85.27%**
  - Khả năng tổng quát hóa (TEST_UNSEEN_TEXT): **85.93%**
- **Mô hình Thí nghiệm V2-A:** `weights/ocr/experiments/v2a/best_lprnet_v2a.onnx`
  - SHA256: `0740c153524cfbac0eef84da19753d87d23c397003b2545735d8dfd111f04207`
  - PyTorch Checkpoint: `weights/ocr/experiments/v2a/best_lprnet_v2a.pth`
  - Độ chính xác TEST_FINAL: **65.87%**
  - Khả năng tổng quát hóa (TEST_UNSEEN_TEXT): **67.40%**
  - Đánh giá phân loại: **[REGRESSION]**
  - Ứng viên Candidate Production: **KHÔNG** (Không đủ tiêu chuẩn thay thế V1)

---

## 2. Dữ Liệu và Manifests Hiện Có
- **Tập Train V1:** `datasets/VNLP_processed/train.csv` (29,836 mẫu)
- **Tập Train V2-A:** `datasets/VNLP_processed/train_v2a.csv` (31,047 mẫu = 29,836 VNLP + 1,211 PBL4)
- **Tập Train V2-A Sidecar Metadata:** `datasets/VNLP_processed/train_v2a_sources.csv`
- **Tập Validation:** `datasets/VNLP_processed/val.csv` (3,728 mẫu - khóa bất biến)
- **Tập TEST_FINAL:** `datasets/VNLP_processed/test.csv` (3,733 mẫu - khóa bất biến)
- **Tập TEST_CLEAN:** `datasets/VNLP_processed/test_clean.csv` (3,727 mẫu - đã loại 6 mẫu exact bitwise duplicate với TRAIN)
- **Tập TEST_UNSEEN_TEXT:** `datasets/VNLP_processed/test_unseen_text.csv` (1,187 mẫu - khóa bất biến)
- **PBL4 Unique Staging:** `dataset_staging/pbl4_unique/` (1,211 mẫu - 100% hợp lệ, không rò rỉ vào VAL/TEST)

---

## 3. Tổng Hợp Các Báo Cáo Kỹ Thuật Đã Tạo Trong Phase 2
1. `reports/v1_baseline_snapshot.md`: Snapshot bối cảnh và mã băm SHA256 bảo vệ tính bất biến của Model V1.
2. `reports/test_clean_removed_samples.csv`: Danh sách 6 mẫu trùng khớp chính xác bitwise giữa TRAIN và TEST.
3. `reports/test_clean_summary.md`: Tổng hợp quy trình tạo tập TEST_CLEAN.
4. `reports/train_v2a_manifest_summary.md`: Thống kê phân phối nhãn, ký tự và nguồn gốc tập huấn luyện V2-A.
5. `reports/v2a_training_smoke_test.md`: Biên bản kiểm tra forward/backward và overfit 32 mẫu trước huấn luyện.
6. `reports/v2a_training_summary.md`: Nhật ký huấn luyện 46 epochs và thông số hội tụ của V2-A.
7. `reports/v2a_pytorch_evaluation.md`: Đánh giá checkpoint PyTorch trên 4 tập dữ liệu và breakdown theo layout.
8. `reports/v2a_onnx_verification.md`: Kiểm tra độ tương thích số học ONNX vs OpenCV DNN (đạt mức sai số ~4e-5, 100% match).
9. `reports/v1_vs_v2a_comparison.md`: Bảng so sánh đối chứng toàn diện cùng runtime OpenCV DNN CPU.
10. `reports/v2a_fixed_cases.csv`: 82 ca V1 đoán sai nhưng V2-A đoán đúng.
11. `reports/v2a_regressions.csv`: 806 ca V1 đoán đúng nhưng V2-A đoán sai.
12. `reports/v1_confusion_matrix.csv` & `reports/v2a_confusion_matrix.csv`: Ma trận nhầm lẫn ký tự.
13. `reports/v1_vs_v2a_confusion_summary.md`: Phân tích biến động các cặp nhầm lẫn ký tự cốt lõi (8/B, 0/D, 5/S, 2/Z).
14. `reports/v2a_result_for_chatgpt.md`: Bản báo cáo tiêu chuẩn gửi cho ChatGPT và hội đồng kỹ thuật.
15. `reports/project_state_after_v2a.md`: Bản cập nhật trạng thái dự án hiện tại.

---

## 4. Quyết Định Kỹ Thuật Cho Các Bước Tiếp Theo
- **Giữ nguyên Production:** V1 (`best_lprnet_vn.onnx`) tiếp tục là model chính thức.
- **Không thực hiện V2-B / Không sinh synthetic plate** trong Phase 2 theo chỉ thị.
- **Đợi phản hồi từ ChatGPT / Người dùng** dựa trên `reports/v2a_result_for_chatgpt.md`.
