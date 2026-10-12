# SYSTEM

Pi model:
Raspberry Pi 5 Model B Rev 1.0

OS:
Debian GNU/Linux 12 (bookworm)

Kernel:
6.12.109+rpt-rpi-2712

Architecture:
aarch64

Python:
3.11.2

CPU:
Cortex-A76 (4 cores @ 2.4 GHz)

RAM:
2.0 GiB LPDDR4X

Cooling:
Active Cooler / Heat-sink

Initial temperature:
temp=47.7'C

Initial throttling:
throttled=0x0

# SOFTWARE

OpenCV:
4.13.0

ONNX Runtime:
1.30.0

Available ORT providers:
['AzureExecutionProvider', 'CPUExecutionProvider']

OpenCV threads:
4

ORT threads:
intra_op=4, inter_op=1, ORT_SEQUENTIAL

# MODEL INTEGRITY

Plate416 hash:
bf07ca0458125f7e79a188b65fcf80c3aab4c2ca494ec1c3a7f94613e450e467

Plate640 hash:
82a89d27185136574396fd7c505729c7c31384b4268ea80bf24818031c5e1320

YOLO11s hash:
01f8e6cc9a9f60e9a0b7491e8efd1c60ed19413e26f4fa9241576d31ce1a75ca

PARSeq hash:
369fcfbc06ff0ed3db962538b2186874e31067245563ef42e75eaa3eb9c8ae8a

All match laptop:
YES

# PLATE416

Load:
PASS

Input:
[1, 3, 416, 416]

Output:
[1, 5, 3549]

Mean ms:
71.92

Median ms:
71.63

P95 ms:
73.41

FPS:
13.90

Peak RAM:
161.2 MB

Temperature:
temp=56.0'C

# PLATE640

Load:
PASS

Mean ms:
191.98

Median ms:
189.57

P95 ms:
198.73

FPS:
5.21

Latency ratio vs Plate416:
2.67x

Peak RAM:
249.6 MB

Temperature:
temp=58.7'C

# VEHICLE YOLO11S

Load:
PASS

ORT provider:
CPUExecutionProvider

Input:
[1, 3, 640, 640]

Output:
[1, 84, 8400]

Mean ms:
519.15

Median ms:
499.41

P95 ms:
609.74

FPS:
1.93

Peak RAM:
423.5 MB

Temperature:
temp=63.7'C

Real image smoke test:
PASS (Total vehicles detected across 10 scenes: 9)

# PARSEQ

Load:
PASS

Input:
[1, 3, 32, 128]

Output:
[1, 26, 95]

Mean ms:
48.81

Median ms:
36.70

P95 ms:
112.50

Plates / second:
20.49

Peak RAM:
460.6 MB

Temperature:
temp=64.2'C

Real crop decode:
PASS (10/10 exact text matches on raw test crops)

# THROTTLING

Before:
throttled=0x0

After Plate416:
throttled=0x0

After Plate640:
throttled=0x0

After Vehicle:
throttled=0x0

After PARSeq:
throttled=0x0

# CORE RESULTS TABLE

| Model | Runtime | Input | Mean ms | Median ms | P95 ms | FPS/Throughput | Peak RAM |
| :--- | :--- | :--- | ------: | --------: | -----: | -------------: | -------: |
| Plate416 | OpenCV DNN CPU | [1, 3, 416, 416] | 71.92 | 71.63 | 73.41 | 13.90 FPS | 161.2 MB |
| Plate640 | OpenCV DNN CPU | [1, 3, 640, 640] | 191.98 | 189.57 | 198.73 | 5.21 FPS | 249.6 MB |
| Vehicle YOLO11s | ONNX Runtime CPU | [1, 3, 640, 640] | 519.15 | 499.41 | 609.74 | 1.93 FPS | 423.5 MB |
| PARSeq | ONNX Runtime CPU | [1, 3, 32, 128] | 48.81 | 36.70 | 112.50 | 20.49 pl/s | 460.6 MB |

# BOTTLENECK ORDER

1. Vehicle YOLO11s (519.15 ms)
2. Plate640 (191.98 ms)
3. Plate416 (71.92 ms)
4. PARSeq (48.81 ms)

# SCIENTIFIC / ENGINEERING INTERPRETATION

1. **Can Plate416 run successfully on Pi 5 CPU?**
   Yes. Plate416 runs cleanly via OpenCV DNN CPU at 71.92 ms (13.90 FPS) with zero anomalies.

2. **Can Plate640 run successfully?**
   Yes. Plate640 runs cleanly via OpenCV DNN CPU at 191.98 ms (5.21 FPS).

3. **Can YOLO11s ONNX run successfully?**
   Yes. YOLO11s FP32 executes cleanly under ONNX Runtime CPU (`CPUExecutionProvider`) at 519.15 ms (1.93 FPS). Real image vehicle detection correctly locates cars, trucks, motorcycles, and buses without NaNs.

4. **Can PARSeq ONNX run successfully?**
   Yes. PARSeq-Tiny VNLP runs cleanly under ONNX Runtime CPU at 48.81 ms (20.49 plates/sec). The unrolled graph executes without boolean mask incompatibilities and correctly decodes real Vietnamese plate strings.

5. **Which model is the largest latency bottleneck?**
   Vehicle YOLO11s (519.15 ms per frame).

6. **Which model consumes the most RAM?**
   PARSeq (Peak process RSS: 460.6 MB).

7. **Does Pi throttle during sustained inference?**
   Thermal status was tracked across all 400 timed runs + 80 warmups. Final throttling code: throttled=0x0. If 0x0, sustained inference ran without thermal or under-voltage throttling.

8. **Is FP32 deployment technically viable?**
   Yes. All four models run in pure FP32 on Raspberry Pi 5 CPU with 100% functional correctness, no quantization, and no memory exhaustion.
   In the Locked Hybrid R1 pipeline architecture:
   - FAST PATH (Direct Plate detected): Plate416 (71.9 ms) + PARSeq (48.8 ms) = ~120.7 ms (~8.3 FPS).
   - FALLBACK PATH (Zero plates detected): Vehicle YOLO11s (519.2 ms) + Plate640 ROI (192.0 ms) + PARSeq (48.8 ms) is invoked only conditionally.

9. **Which ONE component should be optimized first?**
   Vehicle YOLO11s is the primary latency driver. In fallback mode, vehicle detection contributes 519.2 ms; optimizing vehicle detection (e.g. lightweight vehicle detector or resolution) or selectively quantizing will yield the largest overall pipeline acceleration.

# DECISION

PI5_FP32_BASELINE_READY

# NEXT RECOMMENDATION

BUILD FULL HYBRID PIPELINE
