# Nguồn dữ liệu và cách tái lập

Các bộ ảnh huấn luyện không được lưu trực tiếp trên GitHub vì dung lượng lớn (tuân thủ giới hạn file và tối ưu repo). Ba ảnh trong [`samples/`](./samples/) được cung cấp sẵn để chạy thử nhanh (Plug & Play).

| Nguồn tập dữ liệu | Liên kết truy cập | Vai trò & Quy mô |
| :--- | :--- | :--- |
| **Roboflow VN License Plate** | [Roboflow Universe (8,357 ảnh)](https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0) | Dữ liệu huấn luyện mô hình YOLO11 phát hiện vị trí biển số (Train: 5,845; Val: 1,680; Test: 832 ảnh có nhãn). |
| **VNLP (Vietnamese License Plate)** | [Google Drive do tác giả VNLP công bố](https://drive.google.com/drive/folders/19OnlE0gFXEMPyngxRZ980d4z0fGQ-C7T?usp=sharing) | Nguồn ảnh OCR chính (37,297 mẫu đa dạng: 19,086 biển dài 1 dòng, 12,618 biển vuông ô tô 2 dòng, 5,593 biển xe máy). Liên kết trích từ `datasets/VNLP-main/VNLP-main/readme.md`. |
| **PBL4 Deep Learning** | [Repository nguồn GitHub](https://github.com/lephamcong/PBL4_Deep-Learning) | Dữ liệu OCR bổ sung; xem chi tiết tại [báo cáo nguồn và kiểm thử trùng lặp](./reports/pbl4_dataset_report.md). |
| **Vietnamese License Plate Generator** | [Repository nguồn GitHub](https://github.com/NNDam/Vietnamese-License-Plate-Generator) | Sinh ảnh tổng hợp thử nghiệm; xem chi tiết tại [báo cáo mở rộng dữ liệu](./reports/dataset_expansion_summary.md). |

Thư mục `datasets/`, `datasets_external/`, `dataset_staging/` và `vietnamese license plate.v1i.yolov11/` được cấu hình trong `.gitignore`. Sau khi tải tập VNLP, chuẩn bị manifest bằng lệnh [`tools/prepare_vnlp.py`](./tools/prepare_vnlp.py). Các cấu hình [`configs/license_plate.yaml`](./configs/license_plate.yaml), [`configs/ocr_lprnet.yaml`](./configs/ocr_lprnet.yaml) và [`configs/ocr_lprnet_v2a.yaml`](./configs/ocr_lprnet_v2a.yaml) chỉ ra các đường dẫn train/val/test tương ứng.

Các kết quả công bố, phân tích phân phối và giới hạn của tập dữ liệu được trình bày đầy đủ trong [`reports/current_dataset_report.md`](./reports/current_dataset_report.md) và [`reports/dataset_expansion_summary.md`](./reports/dataset_expansion_summary.md).
