# TỔNG HỢP XÂY DỰNG TẬP KIỂM ĐỊNH SẠCH (TEST_CLEAN SUMMARY)

**Thời gian tạo:** 2026-09-24
**Tập nguồn TEST_FINAL:** `datasets/VNLP_processed/test.csv` (3,733 mẫu - ĐÃ KHÓA)
**Tập đích TEST_CLEAN:** `datasets/VNLP_processed/test_clean.csv`
**File ghi nhận mẫu loại bỏ:** `reports/test_clean_removed_samples.csv`

---

## 1. Kết quả Phân tách & Lọc Trùng lặp
- **Số lượng mẫu TEST_FINAL ban đầu:** **3733**
- **Số lượng exact image duplicate với TRAIN bị loại bỏ:** **6**
- **Số lượng mẫu TEST_CLEAN thu được:** **3727**
- **Số lượng exact duplicate còn lại giữa TRAIN và TEST_CLEAN:** **0 (HOÀN TOÀN KHÔNG CÒN)**

---

## 2. Chi tiết 6 Mẫu Bị Loại Bỏ (Exact Bitwise Match)
| STT | File trong TEST_FINAL | File đối ứng trong TRAIN | Nhãn Test | SHA256 |
| :---: | :--- | :--- | :---: | :--- |
| 1 | `1_2984_1_36A38030_144_330_478_417.jpg` | `1_3074_1_36A38230_146_330_475_414.jpg` | `36A38030` | `83ba2498eb22b6f5...` |
| 2 | `6_3353_1_36C14506_9_134_480_221.jpg` | `6_3355_1_36C14536_6_127_480_224.jpg` | `36C14506` | `edcb8cc8676aacd4...` |
| 3 | `2_249_0_36A28345_482_326_539_407.jpg` | `2_242_0_36A26345_479_325_536_405.jpg` | `36A28345` | `bf14a5c9497a56cf...` |
| 4 | `1_742_0_36A37893_111_240_182_307.jpg` | `1_1708_0_36C37893_115_239_179_310.jpg` | `36A37893` | `e6f69fba350282f4...` |
| 5 | `1_674_0_36A37142_116_227_185_290.jpg` | `1_673_0_36A37142_117_228_181_293.jpg` | `36A37142` | `21fcd13c6986ccdf...` |
| 6 | `1_717_0_36A37817_104_234_163_297.jpg` | `1_718_0_36A37817_101_233_164_297.jpg` | `36A37817` | `61b0a2cbe6a827e1...` |

---

## 3. Quy tắc Đánh giá Đa chiều trong Thí nghiệm
Từ Phase 2 trở đi, dự án sử dụng song song 3 tập đánh giá độc lập:
1. **TEST_FINAL (3,733 mẫu):** Giữ nguyên vẹn tuyệt đối để so sánh lịch sử với Model V1 Baseline (mốc 85.27%).
2. **TEST_CLEAN (3,727 mẫu):** Đã loại bỏ 100% rủi ro rò rỉ hình ảnh tuyệt đối với tập huấn luyện, phản ánh độ chính xác thực tế khi không có mẫu nhìn thấy trước.
3. **TEST_UNSEEN_TEXT (1,187 mẫu):** Đánh giá khả năng tổng quát hóa (Generalization) trên các chuỗi biển số hoàn toàn mới chưa từng xuất hiện trong tập Train.
