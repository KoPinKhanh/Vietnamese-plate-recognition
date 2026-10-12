# SO SÁNH MA TRẬN NHẦM LẪN KÝ TỰ: MODEL V1 VS CANDIDATE FT-B (PHASE 3)

**Thời gian thực hiện:** 2026-09-24
**Tập kiểm định:** `TEST_FINAL` (3,733 mẫu)
**Candidate:** FT-B (`weights/ocr/experiments/phase3_ft_b/best.pth`)

---

## 1. Biến Động Các Cặp Ký Tự Trọng Yếu
| Cặp nhầm lẫn (GT -> Pred) | Số lần ở Model V1 | Số lần ở Candidate FT-B | Thay đổi (FT-B vs V1) |
| :---: | :---: | :---: | :---: |
| `0 -> 6` | 12 | 5 | **-7** (GIẢM LỖI (TỐT)) |
| `6 -> 0` | 8 | 5 | **-3** (GIẢM LỖI (TỐT)) |
| `7 -> 1` | 8 | 0 | **-8** (GIẢM LỖI (TỐT)) |
| `E -> F` | 6 | 4 | **-2** (GIẢM LỖI (TỐT)) |
| `3 -> 0` | 5 | 1 | **-4** (GIẢM LỖI (TỐT)) |
| `2 -> 7` | 5 | 4 | **-1** (GIẢM LỖI (TỐT)) |
| `8 -> 9` | 5 | 7 | **+2** (TĂNG LỖI) |
| `X -> Y` | 5 | 3 | **-2** (GIẢM LỖI (TỐT)) |
| `8 -> 0` | 5 | 2 | **-3** (GIẢM LỖI (TỐT)) |
| `9 -> 3` | 4 | 6 | **+2** (TĂNG LỖI) |
| `8 -> B` | 0 | 1 | **+1** (TĂNG LỖI) |
| `B -> 8` | 1 | 0 | **-1** (GIẢM LỖI (TỐT)) |
| `0 -> D` | 0 | 0 | **0** (GIỮ NGUYÊN) |
| `D -> 0` | 1 | 1 | **0** (GIỮ NGUYÊN) |
| `5 -> S` | 0 | 0 | **0** (GIỮ NGUYÊN) |
| `S -> 5` | 0 | 0 | **0** (GIỮ NGUYÊN) |
| `2 -> Z` | 0 | 0 | **0** (GIỮ NGUYÊN) |
| `Z -> 2` | 0 | 0 | **0** (GIỮ NGUYÊN) |

---

## 2. Top 10 Lỗi Phổ Biến Nhất ở Model V1
| Hạng | Cặp ký tự (GT -> Pred) | Tần suất |
| :---: | :---: | :---: |
| 1 | `0 -> 6` | 12 |
| 2 | `6 -> 0` | 8 |
| 3 | `7 -> 1` | 8 |
| 4 | `E -> F` | 6 |
| 5 | `3 -> 0` | 5 |
| 6 | `2 -> 7` | 5 |
| 7 | `8 -> 9` | 5 |
| 8 | `X -> Y` | 5 |
| 9 | `8 -> 0` | 5 |
| 10 | `9 -> 3` | 4 |

---

## 3. Top 10 Lỗi Phổ Biến Nhất ở Candidate FT-B
| Hạng | Cặp ký tự (GT -> Pred) | Tần suất |
| :---: | :---: | :---: |
| 1 | `8 -> 9` | 7 |
| 2 | `9 -> 3` | 6 |
| 3 | `0 -> 6` | 5 |
| 4 | `6 -> 0` | 5 |
| 5 | `3 -> 5` | 5 |
| 6 | `6 -> 8` | 5 |
| 7 | `8 -> 6` | 4 |
| 8 | `R -> B` | 4 |
| 9 | `E -> F` | 4 |
| 10 | `2 -> 7` | 4 |