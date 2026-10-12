"""
Benchmark Latency on Windows Laptop for:
1. LPRNetLite Baseline
2. PARSeq Official Pretrained
3. PARSeq VNLP Fine-Tuned
"""

import time
import numpy as np
import torch
from pathlib import Path
import sys

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.charset import NUM_CLASSES, BLANK_INDEX
from ocr.models.lprnet_lite import LPRNetLite
from ocr.ctc_decoder import ctc_greedy_decode_torch
from teacher_adapters.parseq_vn import parseq_predict_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Benchmarking on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Load models
parseq_pre = torch.hub.load("baudm/parseq", "parseq_tiny", pretrained=True, trust_repo=True).to(device).eval()
parseq_vn = torch.hub.load("baudm/parseq", "parseq_tiny", pretrained=False, trust_repo=True)
ckpt_vn = torch.load("weights/ocr/teachers/parseq_tiny_vn/best.pth", map_location=device, weights_only=False)
parseq_vn.load_state_dict(ckpt_vn["model_state_dict"])
parseq_vn = parseq_vn.to(device).eval()

lprnet = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
ckpt_lpr = torch.load("weights/ocr/experiments/phase4/control_seed42/best.pth", map_location=device, weights_only=False)
lprnet.load_state_dict(ckpt_lpr["model_state_dict"])
lprnet = lprnet.to(device).eval()

dummy_parseq = torch.randn(1, 3, 32, 128, device=device)
dummy_lprnet = torch.randn(1, 3, 48, 160, device=device)

# Warmup 20 passes
for _ in range(20):
    with torch.no_grad():
        _ = parseq_pre(dummy_parseq)
        _ = parseq_vn(dummy_parseq)
        _ = lprnet(dummy_lprnet)
if device.type == "cuda":
    torch.cuda.synchronize()

n_iters = 200

# 1. LPRNetLite
lpr_times = []
for _ in range(n_iters):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = lprnet(dummy_lprnet)
        _ = ctc_greedy_decode_torch(out, blank_index=BLANK_INDEX)
    if device.type == "cuda":
        torch.cuda.synchronize()
    lpr_times.append((time.perf_counter() - t0) * 1000.0)

# 2. PARSeq Pretrained
pre_times = []
for _ in range(n_iters):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = parseq_pre(dummy_parseq)
        probs = logits.softmax(-1)
        _, _ = parseq_pre.tokenizer.decode(probs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    pre_times.append((time.perf_counter() - t0) * 1000.0)

# 3. PARSeq VNLP
vn_times = []
for _ in range(n_iters):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _, _ = parseq_predict_fn(parseq_vn, dummy_parseq)
    if device.type == "cuda":
        torch.cuda.synchronize()
    vn_times.append((time.perf_counter() - t0) * 1000.0)

print("\n" + "=" * 65)
print(f"LATENCY BENCHMARK ON {device} ({n_iters} iterations):")
print("Model                   | Mean (ms) | Median (ms) | P95 (ms)")
print("----------------------- | --------- | ----------- | --------")
print(f"LPRNetLite Baseline     | {np.mean(lpr_times):8.2f}  | {np.median(lpr_times):10.2f}  | {np.percentile(lpr_times, 95):7.2f}")
print(f"PARSeq Official (Pre)   | {np.mean(pre_times):8.2f}  | {np.median(pre_times):10.2f}  | {np.percentile(pre_times, 95):7.2f}")
print(f"PARSeq VNLP Fine-Tuned  | {np.mean(vn_times):8.2f}  | {np.median(vn_times):10.2f}  | {np.percentile(vn_times, 95):7.2f}")
print("=" * 65)
