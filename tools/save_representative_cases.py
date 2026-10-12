import csv
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw

fp_dir = Path("outputs/final_routing_test/false_positives")
neither_dir = Path("outputs/final_routing_test/neither_detected")
fp_dir.mkdir(parents=True, exist_ok=True)
neither_dir.mkdir(parents=True, exist_ok=True)

with open("reports/final_locked_routing_results.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Find Direct and R1 FP samples
# A false positive occurs when candidate count > (1 if detected else 0)
d_fp_rows = [r for r in rows if int(r.get("direct_candidates_count", 0)) > (1 if r["direct_detected"].lower() == "true" else 0)]
r1_fp_rows = [r for r in rows if int(r.get("r1_candidates_count", 0)) > (1 if r["r1_detected"].lower() == "true" else 0)]

print(f"Total Direct FP rows: {len(d_fp_rows)}")
print(f"Total R1 FP rows: {len(r1_fp_rows)}")

# Save top 5 representative FP samples
for i, r in enumerate(r1_fp_rows[:5]):
    img_path = Path(r["image_path"])
    if img_path.exists():
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        # Draw GT in green
        def parse_box(b_str):
            cleaned = b_str.strip().strip("[]()").replace(" ", "")
            return [float(x) for x in cleaned.split(",") if x]

        gt_box = parse_box(r["gt_bbox"])
        draw.rectangle(gt_box, outline="green", width=2)
        # Draw R1 bbox in red
        if r["r1_bbox"]:
            p_box = parse_box(r["r1_bbox"])
            draw.rectangle(p_box, outline="red", width=2)
        s_id = r["sample_id"]
        img.save(fp_dir / f"fp_sample_{i+1}_{s_id}.jpg")

# Find Neither detected samples
neither_rows = [r for r in rows if r["direct_detected"].lower() == "false" and r["r1_detected"].lower() == "false"]
print(f"Total Neither detected rows: {len(neither_rows)}")

# Failure stages of neither
stages = Counter(r["failure_stage"] for r in neither_rows)
print("Neither failure stages:", stages)

# Save representative neither detected samples (3 of each failure type)
saved_types = Counter()
for r in neither_rows:
    st = r["failure_stage"]
    if saved_types[st] < 3:
        img_path = Path(r["image_path"])
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            gt_box = parse_box(r["gt_bbox"])
            draw.rectangle(gt_box, outline="green", width=2)
            if r["vehicle_bbox"]:
                v_box = parse_box(r["vehicle_bbox"])
                draw.rectangle(v_box, outline="blue", width=2)
            s_id = r["sample_id"]
            img.save(neither_dir / f"neither_{st}_{saved_types[st]+1}_{s_id}.jpg")
            saved_types[st] += 1

print("Done saving representative samples.")
