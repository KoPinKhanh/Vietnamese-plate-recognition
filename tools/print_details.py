import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.analyze_fresh_holdout_500 import analyze_holdout

data = analyze_holdout(Path("reports/fresh_holdout_500_results.csv"))

print("=== SIZE ANALYSIS ===")
for k, v in data["size"].items():
    print(f"  {k:<8}: N={v['n']:<3} | Dir={v['d_rec']:>6.2f}% | R1={v['r1_rec']:>6.2f}% | Delta={v['delta']:>+6.2f} pp | Rescues={v['rescues']} | Losses={v['losses']}")

print("\n=== UNSEEN TEXT ANALYSIS ===")
u = data["unseen"]
print(f"  Unseen plate text (N={u['unseen_count']}): Direct E2E={u['unseen_d_e2e']:.2f}% | R1 E2E={u['unseen_r1_e2e']:.2f}% | Delta={u['unseen_r1_e2e'] - u['unseen_d_e2e']:+.2f} pp")
print(f"  Seen plate text   (N={u['seen_count']}): Direct E2E={u['seen_d_e2e']:.2f}% | R1 E2E={u['seen_r1_e2e']:.2f}% | Delta={u['seen_r1_e2e'] - u['seen_d_e2e']:+.2f} pp")

print("\n=== FALLBACK MISS ATTRIBUTION ===")
for k, v in data["fallback"]["miss_stages"].items():
    print(f"  {k}: {v} ({v/data['fallback']['invocations']*100:.1f}%)")

print("\n=== OCR & CHAR METRICS ===")
print(f"  Direct: CharAcc={data['direct']['char_acc']:.2f}% | CER={data['direct']['cer']:.2f}% | OCR Given Det={data['direct']['ocr_given_det']:.2f}% | Precision={data['direct']['precision']:.2f}%")
print(f"  R1    : CharAcc={data['r1']['char_acc']:.2f}% | CER={data['r1']['cer']:.2f}% | OCR Given Det={data['r1']['ocr_given_det']:.2f}% | Precision={data['r1']['precision']:.2f}%")

print("\n=== DEVELOPMENT VS HOLDOUT COMPARISON ===")
dev_ref = {
    "Recall": (59.00, 66.00, +7.00),
    "E2E": (48.00, 54.33, +6.33),
    "one_row recall": (64.00, 64.67, +0.67),
    "two_rows recall": (65.66, 76.77, +11.11),
    "motorcycle recall": (31.37, 49.02, +17.65),
}

fresh_ref = {
    "Recall": (data["direct"]["rec"], data["r1"]["rec"], data["paired"]["delta_rec"]),
    "E2E": (data["direct"]["e2e_acc"], data["r1"]["e2e_acc"], data["paired"]["delta_e2e"]),
    "one_row recall": (data["layout"]["one_row"]["d_rec"], data["layout"]["one_row"]["r1_rec"], data["layout"]["one_row"]["delta_rec"]),
    "two_rows recall": (data["layout"]["two_rows"]["d_rec"], data["layout"]["two_rows"]["r1_rec"], data["layout"]["two_rows"]["delta_rec"]),
    "motorcycle recall": (data["layout"]["motorcycle"]["d_rec"], data["layout"]["motorcycle"]["r1_rec"], data["layout"]["motorcycle"]["delta_rec"]),
}

print(f"{'Metric':<20} | {'Dev Direct':>10} | {'Dev R1':>10} | {'Dev Delta':>10} | {'Fresh Direct':>12} | {'Fresh R1':>10} | {'Fresh Delta':>12}")
print("-" * 95)
for m in dev_ref:
    d_dir, d_r1, d_del = dev_ref[m]
    f_dir, f_r1, f_del = fresh_ref[m]
    print(f"{m:<20} | {d_dir:>9.2f}% | {d_r1:>9.2f}% | {d_del:>+9.2f} pp | {f_dir:>11.2f}% | {f_r1:>9.2f}% | {f_del:>+11.2f} pp")
