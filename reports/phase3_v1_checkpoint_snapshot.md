# BÁO CÁO SNAPSHOT BẢO VỆ CHECKPOINT MODEL V1 (PHASE 3)

**Thời gian tạo:** 2026-09-24  
**Giai đoạn:** Bắt đầu Phase 3 - Hoàn thiện Chất lượng Mô hình AI (PyTorch Fine-tuning)

---

## 1. Thông Tin Checkpoint PyTorch V1 Cốt Lõi
- **Đường dẫn file:** `weights/ocr/best_lprnet_vn.pth`
- **Mã băm SHA256:** `25d2ff3cba0869766a96689a8eda2d1b133634c86023df01e2e41cf17491305c`
- **Dung lượng file:** `10,465,421 bytes` (~9.98 MB)
- **Kiến trúc mô hình:** `LPRNetLite` (`ocr/models/lprnet_lite.py`)
- **Tổng số tham số (Parameters):** `798,565` tham số (~0.80M)
- **Kích thước đầu vào (Input Shape):** `[1, 3, 48, 160]` (RGB, Chuẩn hóa `[-1.0, 1.0]`)
- **Kích thước đầu ra (Output Shape):** `[1, 37, 40]` (Time steps: 40, Classes: 37)
- **Bộ ký tự (Charset):** `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` (36 ký tự)
- **CTC Blank Index:** `36`
- **Missing Keys khi load:** `0`
- **Unexpected Keys khi load:** `0`
- **Các trường trong Checkpoint:** `['epoch', 'model_state_dict', 'optimizer_state_dict', 'best_exact_acc', 'val_metrics', 'config', 'charset']`

---

## 2. Cam Kết Bảo Vệ Checkpoint
1. Checkpoint `weights/ocr/best_lprnet_vn.pth` là tài sản cơ sở bất biến và chuẩn mực cao nhất của dự án hiện tại.
2. Tuyệt đối không ghi đè, sửa đổi, hay di chuyển checkpoint này trong suốt quá trình thực hiện Phase 3.
3. Mọi mô hình huấn luyện thử nghiệm fine-tune sẽ được lưu vào các thư mục độc lập:
   - `weights/ocr/experiments/phase3_ft_a/`
   - `weights/ocr/experiments/phase3_ft_b/`
