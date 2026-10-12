# BIÊN BẢN KIỂM TRA CHẤT LƯỢNG MẪU PBL4 V2 (200 MẪU NGẪU NHIÊN)
*(PBL4 MANUAL & AUTOMATED QUALITY SPOTCHECK V2 — FIXED PARSER)*

**Thời gian thực hiện:** 2026-09-24
**Mục đích:** Sửa đổi lỗi parser trong báo cáo Phase 3 (vốn lấy nhầm tiền tố 'train'/'valid' làm nhãn), xác thực chính xác 200 mẫu ngẫu nhiên độc lập.
**Ngẫu nhiên hóa:** Seed = `42` cố định

---

## 1. Kết Quả Tổng Quan
- **Tổng số mẫu kiểm tra:** 200 / 1,211 mẫu (16.52%)
- **Số mẫu đạt chuẩn toàn diện (PASS):** **200** / 200 (100.00%)
- **Số mẫu không đạt chuẩn (FAIL):** **0** / 200
- **Tỷ lệ khớp nhãn giữa filename và manifest (`parsed_plate == manifest_plate`):** **200 / 200 (100.00%)**
- **Tỷ lệ ký tự thuộc bộ CHARSET OCR:** **200 / 200 (100.00%)**
- **Tỷ lệ độ dài biển số chuẩn (7-9 ký tự):** **200 / 200 (100.00%)**
- **Tỷ lệ ảnh đọc được không lỗi:** **200 / 200 (100.00%)**
- **Tỷ lệ ảnh crop trọn vẹn (không mất góc):** **200 / 200 (100.00%)**

---

## 2. Bảng Dữ Liệu Chi Tiết 200 Mẫu
| STT | Filename | Parsed Plate | Manifest Plate | Chars Valid | Length Valid | Image Readable | Crop Complete | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `train_48A05177_0091_1.jpg` | **48A05177** | **48A05177** | True | True | True | True | **PASS** |
| 2 | `train_30F18976_0959_0.jpg` | **30F18976** | **30F18976** | True | True | True | True | **PASS** |
| 3 | `train_51F44614_0441_0.jpg` | **51F44614** | **51F44614** | True | True | True | True | **PASS** |
| 4 | `train_51F22029_0348_0.jpg` | **51F22029** | **51F22029** | True | True | True | True | **PASS** |
| 5 | `train_51F15585_0121_0.jpg` | **51F15585** | **51F15585** | True | True | True | True | **PASS** |
| 6 | `train_51A16598_0220_0.jpg` | **51A16598** | **51A16598** | True | True | True | True | **PASS** |
| 7 | `train_36A04104_1123_0.jpg` | **36A04104** | **36A04104** | True | True | True | True | **PASS** |
| 8 | `valid_29B10916_1395_0.jpg` | **29B10916** | **29B10916** | True | True | True | True | **PASS** |
| 9 | `train_30H54663_0692_0.jpg` | **30H54663** | **30H54663** | True | True | True | True | **PASS** |
| 10 | `valid_30A97061_1217_0.jpg` | **30A97061** | **30A97061** | True | True | True | True | **PASS** |
| 11 | `train_52Y6490_0103_0.jpg` | **52Y6490** | **52Y6490** | True | True | True | True | **PASS** |
| 12 | `train_30F27580_1162_0.jpg` | **30F27580** | **30F27580** | True | True | True | True | **PASS** |
| 13 | `train_30F25697_1253_0.jpg` | **30F25697** | **30F25697** | True | True | True | True | **PASS** |
| 14 | `train_30X4243_1187_0.jpg` | **30X4243** | **30X4243** | True | True | True | True | **PASS** |
| 15 | `train_51F11585_0370_8.jpg` | **51F11585** | **51F11585** | True | True | True | True | **PASS** |
| 16 | `train_51F16159_0104_0.jpg` | **51F16159** | **51F16159** | True | True | True | True | **PASS** |
| 17 | `train_99A36456_1081_0.jpg` | **99A36456** | **99A36456** | True | True | True | True | **PASS** |
| 18 | `train_30F20449_0962_0.jpg` | **30F20449** | **30F20449** | True | True | True | True | **PASS** |
| 19 | `valid_29LD00143_0005_0.jpg` | **29LD00143** | **29LD00143** | True | True | True | True | **PASS** |
| 20 | `train_51F04877_0781_0.jpg` | **51F04877** | **51F04877** | True | True | True | True | **PASS** |
| 21 | `train_51H96668_0037_0.jpg` | **51H96668** | **51H96668** | True | True | True | True | **PASS** |
| 22 | `train_51F13251_0904_0.jpg` | **51F13251** | **51F13251** | True | True | True | True | **PASS** |
| 23 | `train_61A22959_0431_0.jpg` | **61A22959** | **61A22959** | True | True | True | True | **PASS** |
| 24 | `valid_30A90296_1008_0.jpg` | **30A90296** | **30A90296** | True | True | True | True | **PASS** |
| 25 | `train_51F44614_0729_0.jpg` | **51F44614** | **51F44614** | True | True | True | True | **PASS** |
| 26 | `train_30E20701_1149_0.jpg` | **30E20701** | **30E20701** | True | True | True | True | **PASS** |
| 27 | `train_51A72110_0191_2.jpg` | **51A72110** | **51A72110** | True | True | True | True | **PASS** |
| 28 | `train_52Y6490_0180_0.jpg` | **52Y6490** | **52Y6490** | True | True | True | True | **PASS** |
| 29 | `train_51G00672_0307_0.jpg` | **51G00672** | **51G00672** | True | True | True | True | **PASS** |
| 30 | `train_51A72078_0152_0.jpg` | **51A72078** | **51A72078** | True | True | True | True | **PASS** |
| 31 | `train_51F07973_0423_0.jpg` | **51F07973** | **51F07973** | True | True | True | True | **PASS** |
| 32 | `train_51F98466_0809_0.jpg` | **51F98466** | **51F98466** | True | True | True | True | **PASS** |
| 33 | `train_30V3993_0456_0.jpg` | **30V3993** | **30V3993** | True | True | True | True | **PASS** |
| 34 | `train_51G42861_0572_0.jpg` | **51G42861** | **51G42861** | True | True | True | True | **PASS** |
| 35 | `train_34A23352_0006_0.jpg` | **34A23352** | **34A23352** | True | True | True | True | **PASS** |
| 36 | `train_51G22237_0282_0.jpg` | **51G22237** | **51G22237** | True | True | True | True | **PASS** |
| 37 | `train_51G10096_0329_0.jpg` | **51G10096** | **51G10096** | True | True | True | True | **PASS** |
| 38 | `train_51F27809_0920_0.jpg` | **51F27809** | **51F27809** | True | True | True | True | **PASS** |
| 39 | `train_30F65939_1091_0.jpg` | **30F65939** | **30F65939** | True | True | True | True | **PASS** |
| 40 | `train_62A07858_0023_0.jpg` | **62A07858** | **62A07858** | True | True | True | True | **PASS** |
| 41 | `valid_29A46511_0993_0.jpg` | **29A46511** | **29A46511** | True | True | True | True | **PASS** |
| 42 | `train_51A01204_0168_1.jpg` | **51A01204** | **51A01204** | True | True | True | True | **PASS** |
| 43 | `train_51G39466_0612_2.jpg` | **51G39466** | **51G39466** | True | True | True | True | **PASS** |
| 44 | `train_30G40856_0014_0.jpg` | **30G40856** | **30G40856** | True | True | True | True | **PASS** |
| 45 | `valid_29D03693_1408_0.jpg` | **29D03693** | **29D03693** | True | True | True | True | **PASS** |
| 46 | `train_51F59881_0411_0.jpg` | **51F59881** | **51F59881** | True | True | True | True | **PASS** |
| 47 | `train_51G25481_0480_0.jpg` | **51G25481** | **51G25481** | True | True | True | True | **PASS** |
| 48 | `valid_30A55695_1277_0.jpg` | **30A55695** | **30A55695** | True | True | True | True | **PASS** |
| 49 | `train_51C59323_0515_0.jpg` | **51C59323** | **51C59323** | True | True | True | True | **PASS** |
| 50 | `train_30G25495_1154_0.jpg` | **30G25495** | **30G25495** | True | True | True | True | **PASS** |
| 51 | `train_30F71554_0936_0.jpg` | **30F71554** | **30F71554** | True | True | True | True | **PASS** |
| 52 | `train_51F15585_0660_9.jpg` | **51F15585** | **51F15585** | True | True | True | True | **PASS** |
| 53 | `train_51F59011_0890_8.jpg` | **51F59011** | **51F59011** | True | True | True | True | **PASS** |
| 54 | `train_30G43342_1190_0.jpg` | **30G43342** | **30G43342** | True | True | True | True | **PASS** |
| 55 | `train_36A01418_1038_0.jpg` | **36A01418** | **36A01418** | True | True | True | True | **PASS** |
| 56 | `train_61A83247_0054_0.jpg` | **61A83247** | **61A83247** | True | True | True | True | **PASS** |
| 57 | `train_51G31691_0219_0.jpg` | **51G31691** | **51G31691** | True | True | True | True | **PASS** |
| 58 | `train_51A72444_0475_0.jpg` | **51A72444** | **51A72444** | True | True | True | True | **PASS** |
| 59 | `train_51G37307_0229_6.jpg` | **51G37307** | **51G37307** | True | True | True | True | **PASS** |
| 60 | `train_51G21670_0603_4.jpg` | **51G21670** | **51G21670** | True | True | True | True | **PASS** |
| 61 | `train_51F07568_0208_0.jpg` | **51F07568** | **51F07568** | True | True | True | True | **PASS** |
| 62 | `train_51F31118_0139_0.jpg` | **51F31118** | **51F31118** | True | True | True | True | **PASS** |
| 63 | `train_30G29750_0973_0.jpg` | **30G29750** | **30G29750** | True | True | True | True | **PASS** |
| 64 | `train_51A85325_0389_0.jpg` | **51A85325** | **51A85325** | True | True | True | True | **PASS** |
| 65 | `valid_29A44039_1269_0.jpg` | **29A44039** | **29A44039** | True | True | True | True | **PASS** |
| 66 | `train_51A72444_0476_2.jpg` | **51A72444** | **51A72444** | True | True | True | True | **PASS** |
| 67 | `train_68A19066_0043_0.jpg` | **68A19066** | **68A19066** | True | True | True | True | **PASS** |
| 68 | `train_51G39466_0614_0.jpg` | **51G39466** | **51G39466** | True | True | True | True | **PASS** |
| 69 | `train_51F32488_0045_0.jpg` | **51F32488** | **51F32488** | True | True | True | True | **PASS** |
| 70 | `valid_29D21596_1070_0.jpg` | **29D21596** | **29D21596** | True | True | True | True | **PASS** |
| 71 | `train_51F13251_0796_0.jpg` | **51F13251** | **51F13251** | True | True | True | True | **PASS** |
| 72 | `train_51F78904_0893_2.jpg` | **51F78904** | **51F78904** | True | True | True | True | **PASS** |
| 73 | `train_30F92619_1009_0.jpg` | **30F92619** | **30F92619** | True | True | True | True | **PASS** |
| 74 | `train_51F15585_0773_7.jpg` | **51F15585** | **51F15585** | True | True | True | True | **PASS** |
| 75 | `train_51F69259_0901_0.jpg` | **51F69259** | **51F69259** | True | True | True | True | **PASS** |
| 76 | `train_51G51332_0117_9.jpg` | **51G51332** | **51G51332** | True | True | True | True | **PASS** |
| 77 | `train_51F31118_0228_0.jpg` | **51F31118** | **51F31118** | True | True | True | True | **PASS** |
| 78 | `train_30G16380_1138_0.jpg` | **30G16380** | **30G16380** | True | True | True | True | **PASS** |
| 79 | `train_51F07973_0082_0.jpg` | **51F07973** | **51F07973** | True | True | True | True | **PASS** |
| 80 | `valid_30A05197_1225_0.jpg` | **30A05197** | **30A05197** | True | True | True | True | **PASS** |
| 81 | `train_51F69259_0793_0.jpg` | **51F69259** | **51F69259** | True | True | True | True | **PASS** |
| 82 | `train_51F07973_0261_0.jpg` | **51F07973** | **51F07973** | True | True | True | True | **PASS** |
| 83 | `train_98A13467_0954_0.jpg` | **98A13467** | **98A13467** | True | True | True | True | **PASS** |
| 84 | `train_51G51008_0365_0.jpg` | **51G51008** | **51G51008** | True | True | True | True | **PASS** |
| 85 | `train_62A07017_0825_9.jpg` | **62A07017** | **62A07017** | True | True | True | True | **PASS** |
| 86 | `train_51A55433_0748_0.jpg` | **51A55433** | **51A55433** | True | True | True | True | **PASS** |
| 87 | `train_51F3029_0336_0.jpg` | **51F3029** | **51F3029** | True | True | True | True | **PASS** |
| 88 | `train_51F22029_0511_7.jpg` | **51F22029** | **51F22029** | True | True | True | True | **PASS** |
| 89 | `valid_29A51796_0834_8.jpg` | **29A51796** | **29A51796** | True | True | True | True | **PASS** |
| 90 | `train_51F27809_0817_0.jpg` | **51F27809** | **51F27809** | True | True | True | True | **PASS** |
| 91 | `valid_30A76121_1102_0.jpg` | **30A76121** | **30A76121** | True | True | True | True | **PASS** |
| 92 | `train_60A03868_0190_0.jpg` | **60A03868** | **60A03868** | True | True | True | True | **PASS** |
| 93 | `valid_30A71103_1097_0.jpg` | **30A71103** | **30A71103** | True | True | True | True | **PASS** |
| 94 | `train_51G51292_0236_0.jpg` | **51G51292** | **51G51292** | True | True | True | True | **PASS** |
| 95 | `train_51G27471_0559_0.jpg` | **51G27471** | **51G27471** | True | True | True | True | **PASS** |
| 96 | `train_51A11122_0446_0.jpg` | **51A11122** | **51A11122** | True | True | True | True | **PASS** |
| 97 | `valid_15A00635_1120_0.jpg` | **15A00635** | **15A00635** | True | True | True | True | **PASS** |
| 98 | `train_89A20061_1309_7.jpg` | **89A20061** | **89A20061** | True | True | True | True | **PASS** |
| 99 | `train_30S2639_1159_0.jpg` | **30S2639** | **30S2639** | True | True | True | True | **PASS** |
| 100 | `train_30F74338_1100_0.jpg` | **30F74338** | **30F74338** | True | True | True | True | **PASS** |
| 101 | `train_43A29999_1375_0.jpg` | **43A29999** | **43A29999** | True | True | True | True | **PASS** |
| 102 | `train_51A69172_0756_0.jpg` | **51A69172** | **51A69172** | True | True | True | True | **PASS** |
| 103 | `train_51A72110_0192_0.jpg` | **51A72110** | **51A72110** | True | True | True | True | **PASS** |
| 104 | `train_30G10428_1053_0.jpg` | **30G10428** | **30G10428** | True | True | True | True | **PASS** |
| 105 | `train_51G44313_0056_0.jpg` | **51G44313** | **51G44313** | True | True | True | True | **PASS** |
| 106 | `train_51G42971_0265_0.jpg` | **51G42971** | **51G42971** | True | True | True | True | **PASS** |
| 107 | `train_80B6789_0898_0.jpg` | **80B6789** | **80B6789** | True | True | True | True | **PASS** |
| 108 | `valid_29A25885_0969_0.jpg` | **29A25885** | **29A25885** | True | True | True | True | **PASS** |
| 109 | `train_51F22261_0454_2.jpg` | **51F22261** | **51F22261** | True | True | True | True | **PASS** |
| 110 | `valid_29D05685_0948_0.jpg` | **29D05685** | **29D05685** | True | True | True | True | **PASS** |
| 111 | `train_30E47662_1273_0.jpg` | **30E47662** | **30E47662** | True | True | True | True | **PASS** |
| 112 | `train_50LD04411_0066_0.jpg` | **50LD04411** | **50LD04411** | True | True | True | True | **PASS** |
| 113 | `valid_29A46511_1001_0.jpg` | **29A46511** | **29A46511** | True | True | True | True | **PASS** |
| 114 | `train_51F59881_0412_9.jpg` | **51F59881** | **51F59881** | True | True | True | True | **PASS** |
| 115 | `train_60A35981_0204_0.jpg` | **60A35981** | **60A35981** | True | True | True | True | **PASS** |
| 116 | `train_51A72078_0720_0.jpg` | **51A72078** | **51A72078** | True | True | True | True | **PASS** |
| 117 | `train_62A05195_0542_0.jpg` | **62A05195** | **62A05195** | True | True | True | True | **PASS** |
| 118 | `train_30E06618_1125_0.jpg` | **30E06618** | **30E06618** | True | True | True | True | **PASS** |
| 119 | `train_51F27809_0817_9.jpg` | **51F27809** | **51F27809** | True | True | True | True | **PASS** |
| 120 | `train_98A17339_1303_0.jpg` | **98A17339** | **98A17339** | True | True | True | True | **PASS** |
| 121 | `train_51A96141_0193_0.jpg` | **51A96141** | **51A96141** | True | True | True | True | **PASS** |
| 122 | `valid_14A25074_1152_0.jpg` | **14A25074** | **14A25074** | True | True | True | True | **PASS** |
| 123 | `train_37A99999_0686_0.jpg` | **37A99999** | **37A99999** | True | True | True | True | **PASS** |
| 124 | `train_51F61712_0263_2.jpg` | **51F61712** | **51F61712** | True | True | True | True | **PASS** |
| 125 | `train_51G37307_0533_1.jpg` | **51G37307** | **51G37307** | True | True | True | True | **PASS** |
| 126 | `train_51A72110_0707_1.jpg` | **51A72110** | **51A72110** | True | True | True | True | **PASS** |
| 127 | `valid_29A57224_1278_0.jpg` | **29A57224** | **29A57224** | True | True | True | True | **PASS** |
| 128 | `valid_29A29441_1166_0.jpg` | **29A29441** | **29A29441** | True | True | True | True | **PASS** |
| 129 | `train_30A99279_1306_0.jpg` | **30A99279** | **30A99279** | True | True | True | True | **PASS** |
| 130 | `train_51F78904_0786_0.jpg` | **51F78904** | **51F78904** | True | True | True | True | **PASS** |
| 131 | `train_89A13780_0955_0.jpg` | **89A13780** | **89A13780** | True | True | True | True | **PASS** |
| 132 | `train_30E94515_1013_0.jpg` | **30E94515** | **30E94515** | True | True | True | True | **PASS** |
| 133 | `train_48A05177_0222_0.jpg` | **48A05177** | **48A05177** | True | True | True | True | **PASS** |
| 134 | `train_51G29034_0159_0.jpg` | **51G29034** | **51G29034** | True | True | True | True | **PASS** |
| 135 | `train_51F63034_0528_0.jpg` | **51F63034** | **51F63034** | True | True | True | True | **PASS** |
| 136 | `train_51F20754_0571_0.jpg` | **51F20754** | **51F20754** | True | True | True | True | **PASS** |
| 137 | `train_30F94265_0015_0.jpg` | **30F94265** | **30F94265** | True | True | True | True | **PASS** |
| 138 | `train_51F21255_0791_0.jpg` | **51F21255** | **51F21255** | True | True | True | True | **PASS** |
| 139 | `train_30G90379_0051_0.jpg` | **30G90379** | **30G90379** | True | True | True | True | **PASS** |
| 140 | `train_89A08824_1234_0.jpg` | **89A08824** | **89A08824** | True | True | True | True | **PASS** |
| 141 | `train_30G25495_1153_0.jpg` | **30G25495** | **30G25495** | True | True | True | True | **PASS** |
| 142 | `valid_29A42050_1083_0.jpg` | **29A42050** | **29A42050** | True | True | True | True | **PASS** |
| 143 | `train_51A01204_0257_0.jpg` | **51A01204** | **51A01204** | True | True | True | True | **PASS** |
| 144 | `train_51A01204_0417_0.jpg` | **51A01204** | **51A01204** | True | True | True | True | **PASS** |
| 145 | `train_88A23016_1250_0.jpg` | **88A23016** | **88A23016** | True | True | True | True | **PASS** |
| 146 | `valid_29C66625_1207_0.jpg` | **29C66625** | **29C66625** | True | True | True | True | **PASS** |
| 147 | `train_51A77529_0155_2.jpg` | **51A77529** | **51A77529** | True | True | True | True | **PASS** |
| 148 | `valid_29A23950_0906_0.jpg` | **29A23950** | **29A23950** | True | True | True | True | **PASS** |
| 149 | `train_52Y6490_0273_0.jpg` | **52Y6490** | **52Y6490** | True | True | True | True | **PASS** |
| 150 | `train_51F07973_0083_1.jpg` | **51F07973** | **51F07973** | True | True | True | True | **PASS** |
| 151 | `train_51F06532_0148_0.jpg` | **51F06532** | **51F06532** | True | True | True | True | **PASS** |
| 152 | `train_51F64619_0487_0.jpg` | **51F64619** | **51F64619** | True | True | True | True | **PASS** |
| 153 | `train_51G37307_0395_9.jpg` | **51G37307** | **51G37307** | True | True | True | True | **PASS** |
| 154 | `train_60A35981_0552_0.jpg` | **60A35981** | **60A35981** | True | True | True | True | **PASS** |
| 155 | `valid_19N1866_1063_0.jpg` | **19N1866** | **19N1866** | True | True | True | True | **PASS** |
| 156 | `train_61A39534_0548_0.jpg` | **61A39534** | **61A39534** | True | True | True | True | **PASS** |
| 157 | `train_50Z1630_0200_0.jpg` | **50Z1630** | **50Z1630** | True | True | True | True | **PASS** |
| 158 | `train_51F22261_0035_0.jpg` | **51F22261** | **51F22261** | True | True | True | True | **PASS** |
| 159 | `train_51F15585_0450_0.jpg` | **51F15585** | **51F15585** | True | True | True | True | **PASS** |
| 160 | `train_30G11155_1056_0.jpg` | **30G11155** | **30G11155** | True | True | True | True | **PASS** |
| 161 | `train_51G00672_0209_0.jpg` | **51G00672** | **51G00672** | True | True | True | True | **PASS** |
| 162 | `train_30E97662_1301_0.jpg` | **30E97662** | **30E97662** | True | True | True | True | **PASS** |
| 163 | `valid_30A88888_1335_0.jpg` | **30A88888** | **30A88888** | True | True | True | True | **PASS** |
| 164 | `valid_29D05685_0949_0.jpg` | **29D05685** | **29D05685** | True | True | True | True | **PASS** |
| 165 | `train_51F15585_0798_0.jpg` | **51F15585** | **51F15585** | True | True | True | True | **PASS** |
| 166 | `valid_30A89685_1108_0.jpg` | **30A89685** | **30A89685** | True | True | True | True | **PASS** |
| 167 | `train_30E28131_0972_0.jpg` | **30E28131** | **30E28131** | True | True | True | True | **PASS** |
| 168 | `train_30G27838_1259_0.jpg` | **30G27838** | **30G27838** | True | True | True | True | **PASS** |
| 169 | `train_30F95280_1014_0.jpg` | **30F95280** | **30F95280** | True | True | True | True | **PASS** |
| 170 | `train_51F15585_0772_0.jpg` | **51F15585** | **51F15585** | True | True | True | True | **PASS** |
| 171 | `train_30G21513_0027_0.jpg` | **30G21513** | **30G21513** | True | True | True | True | **PASS** |
| 172 | `train_30F27580_1161_0.jpg` | **30F27580** | **30F27580** | True | True | True | True | **PASS** |
| 173 | `train_51F88270_0358_0.jpg` | **51F88270** | **51F88270** | True | True | True | True | **PASS** |
| 174 | `valid_15A66666_1411_0.jpg` | **15A66666** | **15A66666** | True | True | True | True | **PASS** |
| 175 | `train_51F20754_0135_0.jpg` | **51F20754** | **51F20754** | True | True | True | True | **PASS** |
| 176 | `train_51F44614_0765_0.jpg` | **51F44614** | **51F44614** | True | True | True | True | **PASS** |
| 177 | `train_89A07394_1048_0.jpg` | **89A07394** | **89A07394** | True | True | True | True | **PASS** |
| 178 | `train_51F07973_0356_1.jpg` | **51F07973** | **51F07973** | True | True | True | True | **PASS** |
| 179 | `train_51A02923_0216_0.jpg` | **51A02923** | **51A02923** | True | True | True | True | **PASS** |
| 180 | `valid_30A26144_1072_0.jpg` | **30A26144** | **30A26144** | True | True | True | True | **PASS** |
| 181 | `valid_30A47837_1196_0.jpg` | **30A47837** | **30A47837** | True | True | True | True | **PASS** |
| 182 | `train_88A14436_0956_0.jpg` | **88A14436** | **88A14436** | True | True | True | True | **PASS** |
| 183 | `train_51F21255_0900_0.jpg` | **51F21255** | **51F21255** | True | True | True | True | **PASS** |
| 184 | `train_51G51332_0591_7.jpg` | **51G51332** | **51G51332** | True | True | True | True | **PASS** |
| 185 | `train_51B30620_1396_0.jpg` | **51B30620** | **51B30620** | True | True | True | True | **PASS** |
| 186 | `train_30Z1040_1242_0.jpg` | **30Z1040** | **30Z1040** | True | True | True | True | **PASS** |
| 187 | `train_60A20880_0183_0.jpg` | **60A20880** | **60A20880** | True | True | True | True | **PASS** |
| 188 | `train_51G21670_0517_0.jpg` | **51G21670** | **51G21670** | True | True | True | True | **PASS** |
| 189 | `train_52Z9755_0433_0.jpg` | **52Z9755** | **52Z9755** | True | True | True | True | **PASS** |
| 190 | `train_51G51936_0896_8.jpg` | **51G51936** | **51G51936** | True | True | True | True | **PASS** |
| 191 | `train_77L7777_0733_0.jpg` | **77L7777** | **77L7777** | True | True | True | True | **PASS** |
| 192 | `train_30F88368_0028_0.jpg` | **30F88368** | **30F88368** | True | True | True | True | **PASS** |
| 193 | `train_34A31825_1172_0.jpg` | **34A31825** | **34A31825** | True | True | True | True | **PASS** |
| 194 | `train_30F99666_1316_0.jpg` | **30F99666** | **30F99666** | True | True | True | True | **PASS** |
| 195 | `train_51G51332_0120_0.jpg` | **51G51332** | **51G51332** | True | True | True | True | **PASS** |
| 196 | `train_51G00672_0306_0.jpg` | **51G00672** | **51G00672** | True | True | True | True | **PASS** |
| 197 | `train_43A27208_0024_0.jpg` | **43A27208** | **43A27208** | True | True | True | True | **PASS** |
| 198 | `train_51F22261_0124_2.jpg` | **51F22261** | **51F22261** | True | True | True | True | **PASS** |
| 199 | `train_51C03283_0570_0.jpg` | **51C03283** | **51C03283** | True | True | True | True | **PASS** |
| 200 | `train_61A22959_0430_0.jpg` | **61A22959** | **61A22959** | True | True | True | True | **PASS** |

---

## 3. Kết Luận Kiểm Định
1. **Giải thích nguyên nhân lỗi ở báo cáo Phase 3:**
   - Trong Phase 3, script tạo báo cáo `analyze_pbl4_domain.py` đã dùng lệnh `p.stem.split('_')[0]`. Do file có dạng `train_51F88270_0508_2.jpg`, lệnh này lấy ra chuỗi `'TRAIN'` thay vì chuỗi biển số `'51F88270'`.
   - Tệp manifest huấn luyện thực tế `datasets/VNLP_processed/train_v2a.csv` do script `create_train_v2a.py` tạo ra đã dùng đúng parser `tokens[1]`, do đó **100% nhãn huấn luyện trong manifest luôn chính xác tuyệt đối**.
2. **Xác nhận chất lượng dữ liệu PBL4:**
   - Toàn bộ 200 mẫu ngẫu nhiên kiểm tra đều đạt trạng thái **PASS (100.00%)**.
   - Không có ảnh hỏng, không có nhãn lệch, không có ký tự ngoài CHARSET.
