# Nguồn dữ liệu và cách tái lập

Các bộ ảnh huấn luyện không được lưu trong GitHub vì dung lượng lớn. Ba ảnh trong [`samples/`](./samples/) chỉ dùng để chạy thử, không phải tập đánh giá.

| Nguồn | Liên kết | Vai trò |
| --- | --- | --- |
| VNLP (Vietnamese License Plate) | [Thư mục Google Drive do tác giả VNLP công bố](https://drive.google.com/drive/folders/19OnlE0gFXEMPyngxRZ980d4z0fGQ-C7T?usp=sharing) | Nguồn ảnh OCR chính; liên kết được trích từ `datasets/VNLP-main/VNLP-main/readme.md` trong bản dữ liệu tại máy. |
| PBL4 Deep Learning | [Repository nguồn](https://github.com/lephamcong/PBL4_Deep-Learning) | Dữ liệu bổ sung; xem [báo cáo nguồn và trùng lặp](./reports/pbl4_dataset_report.md). |
| Vietnamese License Plate Generator | [Repository nguồn](https://github.com/NNDam/Vietnamese-License-Plate-Generator) | Sinh ảnh tổng hợp thử nghiệm; xem [báo cáo mở rộng dữ liệu](./reports/dataset_expansion_summary.md). |

Thư mục `datasets/`, `datasets_external/` và `dataset_staging/` bị `.gitignore` loại trừ. Sau khi tải VNLP, chuẩn bị manifest bằng [`tools/prepare_vnlp.py`](./tools/prepare_vnlp.py). Các cấu hình [`configs/ocr_lprnet.yaml`](./configs/ocr_lprnet.yaml) và [`configs/ocr_lprnet_v2a.yaml`](./configs/ocr_lprnet_v2a.yaml) chỉ ra đường dẫn train/val/test được kỳ vọng. Các tập đã xử lý và phép chia tập riêng của dự án không được cung cấp trong repo; tải dữ liệu gốc không tự động tái tạo chính xác mọi kết quả đã báo cáo.

Các kết quả công bố và giới hạn của tập dữ liệu được trình bày trong [`reports/current_dataset_report.md`](./reports/current_dataset_report.md) và [`reports/dataset_expansion_summary.md`](./reports/dataset_expansion_summary.md). Người nộp cần kiểm tra quyền truy cập các liên kết nguồn bằng tài khoản khác trước khi gửi giảng viên.
