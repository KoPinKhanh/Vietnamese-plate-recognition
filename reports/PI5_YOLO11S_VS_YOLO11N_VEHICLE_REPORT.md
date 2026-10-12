# PROJECT
Vietnam License Plate Recognition System (Pi 5 CPU Deployment)

# OBJECTIVE
Perform a controlled empirical evaluation of **YOLO11s-640** (baseline) versus **YOLO11n-640** (candidate) as the vehicle fallback detector on Raspberry Pi 5 CPU.
Evaluate whether YOLO11n can resolve the primary latency bottleneck while preserving adequate vehicle ROI coverage (especially motorcycle ROI coverage) for plate fallback routing.

# RASPBERRY PI SYSTEM

Model:
Raspberry Pi 5 Model B Rev 1.0 (Broadcom BCM2712, 4 cores Cortex-A76 @ 2.4 GHz)

RAM:
2.0 GiB LPDDR4X

OS:
Debian GNU/Linux 12 (bookworm)

Python:
3.11.2

ONNX Runtime:
1.30.0

Temperature:
Start: temp=51.0'C | After YOLO11s: temp=68.1'C | After YOLO11n: temp=65.9'C

Throttling:
Start: throttled=0x0 | After YOLO11s: throttled=0x0 | After YOLO11n: throttled=0x0

# MODELS

YOLO11s:

Path: /home/cytron/vietnam_plate/models/pi_cpu/yolo11s_640_fp32.onnx  
SHA256: 01f8e6cc9a9f60e9a0b7491e8efd1c60ed19413e26f4fa9241576d31ce1a75ca  
Size: 38,051,782 bytes (36.29 MB)

YOLO11n:

Path: /home/cytron/vietnam_plate/models/pi_cpu/yolo11n_640_fp32.onnx  
SHA256: 51233245cd84dcea4fc73d84b48fb0dad028c4e8744856bcded3da32abe44a02  
Size: 10,741,396 bytes (10.24 MB)

# ONNX EQUIVALENCE

YOLO11n PT vs ONNX:
PASS (100.0% class agreement, 1.000000 mean IoU, min IoU 0.999995, raw tensor mean abs diff 1.98e-6 on laptop CPU)

# FAIRNESS CONTROLS

Input:
1 × 3 × 640 × 640 FP32 (Identical letterbox, normalization, and RGB channel order)

Confidence:
0.25

Classes:
COCO relevant vehicle classes: car (2), motorcycle (3), bus (5), truck (7)

Runtime:
ONNX Runtime CPUExecutionProvider

Threads:
intra_op_num_threads = 4, inter_op_num_threads = 1, ORT_SEQUENTIAL

# DATASET

N:
500 full-scene images from `fresh_holdout_500_manifest.csv`

Layout distribution:
- one_row: 225 images (45.0%)
- two_rows: 175 images (35.0%)
- motorcycle: 100 images (20.0%)

Fallback cohort N:
183 images where Direct Plate416 produced zero accepted candidates (84 one_row, 37 two_rows, 62 motorcycle)

# CORE RESULTS

| Metric | YOLO11s | YOLO11n | Delta |
| :--- | ------: | ------: | ----: |
| Overall valid ROI coverage | 70.60% (353/500) | 73.20% (366/500) | +2.60 pp |
| One-row valid ROI | 45.33% (102/225) | 52.44% (118/225) | +7.11 pp |
| Two-row valid ROI | 92.57% (162/175) | 93.14% (163/175) | +0.57 pp |
| Motorcycle valid ROI | 89.00% (89/100) | 85.00% (85/100) | -4.00 pp |
| Fallback-cohort valid ROI | 67.76% (124/183) | 73.77% (135/183) | +6.01 pp |
| Motorcycle fallback valid ROI | 88.71% (55/62) | 82.26% (51/62) | -6.45 pp |
| Mean vehicle proposals/image | 0.99 | 1.03 | +0.05 |
| Operational false proposals | 5 (1.0%) | 4 (0.8%) | -1 |
| Mean latency | 508.69 ms | 416.99 ms | -91.70 ms (18.0% reduction) |
| Median latency | 483.44 ms | 387.69 ms | -95.75 ms |
| P95 latency | 593.19 ms | 669.31 ms | +76.12 ms |
| FPS (Throughput) | 1.97 FPS | 2.40 FPS | 1.22x speedup |
| Peak RAM | 308.7 MB | 151.4 MB | -157.3 MB |

# PAIRED ROI ANALYSIS

Both:
318 (63.6%)

YOLO11s only:
35 (7.0%)

YOLO11n only:
48 (9.6%)

Neither:
99 (19.8%)

Net difference:
+13 samples (+2.60 pp)

# MOTORCYCLE ANALYSIS
- Motorcycle samples: N = 100
- YOLO11s motorcycle valid ROI coverage: 89.00% (89/100)
- YOLO11n motorcycle valid ROI coverage: 85.00% (85/100)
- Net delta: -4.00 pp
- Paired breakdown:
  - Both valid: 79
  - YOLO11s only: 10
  - YOLO11n only: 6
  - Neither valid: 5

# FALLBACK COHORT ANALYSIS
- Sub-cohort N = 183 images where Direct Plate416 failed
- YOLO11s valid ROI coverage: 67.76% (124/183)
- YOLO11n valid ROI coverage: 73.77% (135/183)
- Delta on fallback cohort: +6.01 pp
- Within motorcycle fallback cohort (N = 62):
  - YOLO11s: 88.71% (55/62)
  - YOLO11n: 82.26% (51/62)
  - Delta: -6.45 pp

# CONFIDENCE ANALYSIS
Valid vehicle ROI confidence metrics:
- YOLO11s: Mean = 0.7437, Median = 0.8328, P10 = 0.3591
- YOLO11n: Mean = 0.6528, Median = 0.6816, P10 = 0.3663

# LATENCY
- YOLO11s 640: Mean 508.69 ms | Median 483.44 ms | P95 593.19 ms | 1.97 FPS
- YOLO11n 640: Mean 416.99 ms | Median 387.69 ms | P95 669.31 ms | 2.40 FPS
- Speedup: **1.22x faster**
- Latency Reduction: **18.0% reduction** (91.7 ms saved per execution)

# MEMORY
- YOLO11s Peak Process RAM: 308.7 MB
- YOLO11n Peak Process RAM: 151.4 MB
- Memory delta: -157.3 MB

# THERMALS
- Thermal progression:
  - Start: temp=51.0'C (throttled=throttled=0x0)
  - After YOLO11s: temp=68.1'C (throttled=throttled=0x0)
  - After YOLO11n: temp=65.9'C (throttled=throttled=0x0)
- Thermal / Electrical contamination: **NONE** (throttling remained throttled=0x0)

# STATISTICS
- **Overall ROI Coverage McNemar Exact Test**:
  - Discordant pairs: S_ONLY = 35, N_ONLY = 48
  - Two-tailed p-value = 0.18746
- **Motorcycle ROI Coverage McNemar Exact Test**:
  - Discordant pairs: S_ONLY = 10, N_ONLY = 6
  - Two-tailed p-value = 0.45450
- **Paired Bootstrap 95% Confidence Intervals (10,000 resamples)**:
  - Overall Delta 95% CI: [-1.00 pp, +6.20 pp]
  - Motorcycle Delta 95% CI: [-12.00 pp, +4.00 pp]

# SCIENTIFIC / ENGINEERING CONCLUSION

1. **How much faster is YOLO11n?**
   YOLO11n executes at **416.99 ms** versus **508.69 ms** for YOLO11s, delivering a **1.22x speedup** (18.0% reduction in CPU time).
2. **How much RAM does it save?**
   YOLO11n uses a significantly smaller parameter footprint (10.2 MB vs 36.3 MB), operating within normal Pi process memory headroom.
3. **Does overall valid vehicle ROI coverage change?**
   YOLO11s achieves 70.60% while YOLO11n achieves 73.20% (delta: +2.60 pp, 95% CI: [-1.00 pp, +6.20 pp], McNemar p = 0.18746).
4. **Does motorcycle coverage change?**
   YOLO11s covers 89.00% of motorcycles while YOLO11n covers 85.00% (delta: -4.00 pp, McNemar p = 0.45450).
5. **Does fallback-cohort coverage change?**
   On the 183 critical samples where Direct Plate416 produced zero candidates, valid vehicle ROI coverage is 67.76% (YOLO11s) vs 73.77% (YOLO11n) (delta: +6.01 pp).
6. **How many YOLO11s detections are lost?**
   Across 500 samples, exactly 35 samples had a valid ROI with YOLO11s but not with YOLO11n.
7. **How many YOLO11n-only rescues appear?**
   Exactly 48 samples had a valid ROI with YOLO11n that YOLO11s missed.
8. **Is the speed gain worth the accuracy tradeoff?**
   No. YOLO11n achieves only an 18.0% latency reduction (1.22x speedup), failing the predeclared >=30% speedup gate. Furthermore, on the critical motorcycle fallback cohort (where Direct Plate416 failed), valid vehicle ROI coverage drops by -6.45 pp (from 88.71% to 82.26%, losing 4 critical motorcycle plate candidates). The modest 91.7 ms latency improvement does not justify degrading the primary routing fallback objective.
9. **Which vehicle detector should be used in the Pi Hybrid pipeline?**
   YOLO11N_NOT_SUPPORTED.

# DECISION

YOLO11N_NOT_SUPPORTED

# NEXT RECOMMENDATION

RETAIN YOLO11S AS BASELINE VEHICLE DETECTOR.
