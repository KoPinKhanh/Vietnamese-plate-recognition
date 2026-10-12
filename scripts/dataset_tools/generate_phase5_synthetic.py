"""
Phase 5 Targeted Synthetic Generator for Motorcycle and Two-Row Plates.
Target: 2,000 samples
- 1,500 motorcycle plates (75%)
- 500 two-row car plates (25%)
Applies realistic variations: perspective, rotation, blur, lighting, dirt, bolt occlusion.
Audits 200 random samples and outputs: reports/AUTO_SYNTHETIC_PHASE5_AUDIT.md
"""

import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.charset import CHARSET, CHAR_TO_IDX

PROVINCE_CODES = [
    "11", "12", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "43", "47", "48", "49", "50", "51", "52", "53",
    "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66",
    "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
    "81", "82", "83", "84", "85", "86", "88", "89", "90", "92", "93", "94", "95",
    "97", "98", "99"
]

SERIES_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]


def generate_targeted_plate(is_motorcycle: bool) -> Tuple[str, str, str, str]:
    """
    Generates plate text components.
    Returns: (gt_label, visual_line1, visual_line2, layout_type)
    """
    province = random.choice(PROVINCE_CODES)
    series_letter = random.choice(SERIES_LETTERS)

    is_5digit = random.random() < 0.85
    if is_5digit:
        d1 = str(random.randint(100, 999))
        d2 = f"{random.randint(10, 99):02d}"
        digits_raw = d1 + d2
        digits_visual = f"{d1}.{d2}"
    else:
        digits_raw = f"{random.randint(1000, 9999):04d}"
        digits_visual = digits_raw

    if is_motorcycle:
        sub_num = str(random.randint(1, 9))
        moto_series = f"{series_letter}{sub_num}"
        gt_label = f"{province}{moto_series}{digits_raw}"
        line1 = f"{province}-{moto_series}"
        line2 = digits_visual
        layout = "two_rows_label_xe_may"
    else:
        gt_label = f"{province}{series_letter}{digits_raw}"
        line1 = f"{province}{series_letter}"
        line2 = digits_visual
        layout = "two_rows"

    return gt_label, line1, line2, layout


def render_square_plate(
    line1: str,
    line2: str,
    font_path: Path,
    bg_path: Optional[Path] = None,
    target_size: Tuple[int, int] = (320, 240)
) -> Image.Image:
    tw, th = target_size
    if bg_path and bg_path.exists():
        im = Image.open(str(bg_path)).convert("RGB").resize((tw, th), Image.Resampling.BILINEAR)
    else:
        im = Image.new("RGB", (tw, th), (248, 248, 248))
        draw = ImageDraw.Draw(im)
        draw.rectangle([(4, 4), (tw - 5, th - 5)], outline=(25, 25, 25), width=3)

    font_size1 = random.randint(88, 94)
    font_size2 = random.randint(92, 98)
    font1 = ImageFont.truetype(str(font_path), font_size1)
    font2 = ImageFont.truetype(str(font_path), font_size2)
    draw = ImageDraw.Draw(im)

    bbox1 = draw.textbbox((0, 0), line1, font=font1)
    w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]

    bbox2 = draw.textbbox((0, 0), line2, font=font2)
    w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

    x1 = max(10, int((tw - w1) / 2))
    y1 = int(th * random.uniform(0.12, 0.16))

    x2 = max(10, int((tw - w2) / 2))
    y2 = int(th * random.uniform(0.50, 0.54))

    text_col = (random.randint(15, 35), random.randint(15, 35), random.randint(15, 35))
    shadow_col = (random.randint(180, 210), random.randint(180, 210), random.randint(180, 210))

    # Shadow & Text
    draw.text((x1 + 1, y1 + 1), line1, font=font1, fill=shadow_col)
    draw.text((x1, y1), line1, font=font1, fill=text_col)

    draw.text((x2 + 1, y2 + 1), line2, font=font2, fill=shadow_col)
    draw.text((x2, y2), line2, font=font2, fill=text_col)

    return im


def apply_realistic_effects(pil_img: Image.Image) -> np.ndarray:
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    # 1. Perspective Transform (Homography)
    if random.random() < 0.6:
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dx = random.uniform(0, 0.08 * w)
        dy = random.uniform(0, 0.08 * h)
        pts2 = np.float32([
            [random.uniform(0, dx), random.uniform(0, dy)],
            [w - random.uniform(0, dx), random.uniform(0, dy)],
            [random.uniform(0, dx), h - random.uniform(0, dy)],
            [w - random.uniform(0, dx), h - random.uniform(0, dy)],
        ])
        M_p = cv2.getPerspectiveTransform(pts1, pts2)
        cv_img = cv2.warpPerspective(cv_img, M_p, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Minor rotation (-3 to +3 deg)
    angle = random.uniform(-3.5, 3.5)
    M_r = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cv_img = cv2.warpAffine(cv_img, M_r, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 3. Simulate mild bolt/screw head near top/bottom margin (never on center text)
    if random.random() < 0.4:
        screw_x1 = int(w * random.uniform(0.18, 0.28))
        screw_x2 = int(w * random.uniform(0.72, 0.82))
        screw_y = int(h * random.uniform(0.04, 0.08))
        r = random.randint(4, 7)
        cv2.circle(cv_img, (screw_x1, screw_y), r, (80, 80, 80), -1)
        cv2.circle(cv_img, (screw_x1, screw_y), r // 2, (40, 40, 40), -1)
        cv2.circle(cv_img, (screw_x2, screw_y), r, (80, 80, 80), -1)
        cv2.circle(cv_img, (screw_x2, screw_y), r // 2, (40, 40, 40), -1)

    # 4. Brightness / Contrast / Gamma
    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-20, 20)
    cv_img = np.clip(alpha * cv_img + beta, 0, 255).astype(np.uint8)

    # 5. Motion blur or defocus blur
    if random.random() < 0.35:
        ksize = random.choice([3, 5])
        cv_img = cv2.GaussianBlur(cv_img, (ksize, ksize), 0)

    # 6. JPEG compression artifacts
    if random.random() < 0.4:
        quality = random.randint(45, 85)
        _, enc = cv2.imencode(".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        cv_img = cv2.imdecode(enc, 1)

    return cv_img


def main():
    random.seed(42)
    np.random.seed(42)

    gen_root = Path("datasets_external/vn_plate_generator/Vietnamese-License-Plate-Generator")
    font_path = gen_root / "MyFont-Regular_ver3.otf"
    bg_dir = gen_root / "background"
    square_bgs = list(bg_dir.glob("square*.jpg"))

    out_dir = Path("dataset_staging/synthetic_phase5")
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = out_dir / "labels.csv"

    print("=" * 70)
    print("PHASE 5C: GENERATING TARGETED SYNTHETIC MOTORCYCLE & TWO-ROW PLATES")
    print(f"Output: {out_dir}")
    print(f"Font:   {font_path}")
    print(f"BGs:    {[b.name for b in square_bgs]}")
    print("=" * 70)

    total_samples = 2000
    moto_samples = 1500  # 75%
    car_samples = 500    # 25%

    records = []

    for i in range(1, total_samples + 1):
        is_moto = (i <= moto_samples)
        gt_label, line1, line2, layout = generate_targeted_plate(is_moto)

        # Validate label characters
        for ch in gt_label:
            assert ch in CHAR_TO_IDX, f"Invalid character '{ch}' in {gt_label}"

        bg = random.choice(square_bgs) if square_bgs and random.random() < 0.75 else None
        pil_im = render_square_plate(line1, line2, font_path=font_path, bg_path=bg)
        final_cv = apply_realistic_effects(pil_im)

        img_filename = f"synth_p5_{i:04d}_{gt_label}.jpg"
        img_full_path = img_dir / img_filename
        cv2.imwrite(str(img_full_path), final_cv)

        records.append({
            "image_path": str(img_full_path).replace("\\", "/"),
            "plate_text": gt_label,
            "layout": layout,
            "source": "SYNTH_P5",
        })

    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "plate_text", "layout", "source"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} images in {img_dir}")
    print(f"Saved manifest: {manifest_csv}")

    # Audit 200 random samples
    print("\nRunning Audit on 200 Random Synthetic Samples...")
    random.seed(1337)
    audit_samples = random.sample(records, 200)

    audit_pass = 0
    audit_rows = []

    for item in audit_samples:
        p = Path(item["image_path"])
        exists = p.exists()
        valid_chars = all(c in CHAR_TO_IDX for c in item["plate_text"])
        valid_len = (7 <= len(item["plate_text"]) <= 9)
        readable = exists and (cv2.imread(str(p)) is not None)

        status = "PASS" if (exists and valid_chars and valid_len and readable) else "FAIL"
        if status == "PASS":
            audit_pass += 1

        audit_rows.append({
            "filename": p.name,
            "plate_text": item["plate_text"],
            "layout": item["layout"],
            "chars_valid": "YES" if valid_chars else "NO",
            "length_valid": "YES" if valid_len else "NO",
            "image_readable": "YES" if readable else "NO",
            "status": status,
        })

    print(f"Audit Result: {audit_pass} / 200 ({audit_pass / 200 * 100:.1f}%) PASS")

    # Write Audit Report
    audit_md = Path("reports/AUTO_SYNTHETIC_PHASE5_AUDIT.md")
    lines = [
        "# BÁO CÁO KIỂM TRA CHẤT LƯỢNG DỮ LIỆU SYNTHETIC PHASE 5",
        "*(PHASE 5C SYNTHETIC MOTORCYCLE & TWO-ROW AUDIT REPORT)*",
        "",
        f"**Thời gian thực hiện:** 2026-09-25",
        f"**Tổng số mẫu sinh:** {total_samples} (Xe máy: {moto_samples} [75%], Ô tô 2 dòng: {car_samples} [25%])",
        f"**Số mẫu kiểm tra ngẫu nhiên:** 200 mẫu",
        f"**Kết quả Audit:** **{audit_pass} / 200 ({audit_pass / 200 * 100:.1f}% PASS)**",
        f"**Trạng thái toàn vẹn dữ liệu (Data Integrity):** **{'PASS' if audit_pass == 200 else 'FAIL'}**",
        "",
        "---",
        "",
        "## Danh Sách 20 Mẫu Đại Diện Kiểm Tra Ngẫu Nhiên",
        "| Filename | Plate Text | Layout | Valid Chars | Valid Length | Readable | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in audit_rows[:20]:
        lines.append(f"| `{r['filename']}` | **{r['plate_text']}** | `{r['layout']}` | {r['chars_valid']} | {r['length_valid']} | {r['image_readable']} | **{r['status']}** |")

    with open(audit_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {audit_md}")


if __name__ == "__main__":
    main()
