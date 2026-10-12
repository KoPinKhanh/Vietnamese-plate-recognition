# BÁO CÁO KIỂM THỬ XUẤT ONNX MODEL V2-A (V2-A ONNX VERIFICATION REPORT)

**Thời gian thực hiện:** 2026-09-24
**File ONNX V2-A xuất:** `weights\ocr\experiments\v2a\best_lprnet_v2a.onnx`
**Dung lượng file:** `3.04 MB`
**Opset Version:** 12
**Input Shape:** `[1, 3, 48, 160]` | **Output Shape:** `[1, 37, 40]`

---

## 1. Kết quả So khớp Số học (Numerical Agreement)
| Tiêu chí | Ngưỡng cho phép | Đo đạc thực tế | Đánh giá |
| :--- | :---: | :---: | :---: |
| **Độ lệch tối đa PyTorch vs ONNX Runtime** | < 1e-4 | `3.242493e-05` | **PASS** |
| **Độ lệch tối đa PyTorch vs OpenCV DNN** | < 1e-4 | `4.577637e-05` | **PASS** |
| **Tỷ lệ khớp giải mã PyTorch vs ONNX Runtime** | 100% | `100/100 (100.00%)` | **PASS** |
| **Tỷ lệ khớp giải mã PyTorch vs OpenCV DNN** | >= 99% | `100/100 (100.00%)` | **PASS** |
| **Độ chính xác biển số trên 100 mẫu thử** | N/A | `67/100 (67.00%)` | **THAM CHIẾU** |

---

## 2. Kết luận
- Trạng thái tổng thể: **[PASS]**.
- Mô hình ONNX V2-A đạt sự tương thích số học tuyệt đối với OpenCV DNN trên CPU.
- Đủ điều kiện triển khai kiểm thử và benchmark cùng runtime với Model V1.
