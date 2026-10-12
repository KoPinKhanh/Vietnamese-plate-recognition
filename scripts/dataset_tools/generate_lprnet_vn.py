"""
Synthetic Vietnamese License Plate Generator Adapter for LPRNet.
Adapts Vietnamese-License-Plate-Generator assets to produce:
  - Plate images: dataset_staging/synthetic/images/
  - Manifest CSV: dataset_staging/synthetic/labels.csv (filename, label, layout, source)

Features:
  - 100% compatible with CHARS (0-9, A-Z)
  - Supports both one_row (long plates) and two_rows (square car & motorcycle plates)
  - Renders realistic visual separators (hyphens, dots) while keeping ground truth clean
  - Injects rare characters (J, O, Q, W) to expand representation
  - Compatible with modern Pillow (10+)
"""

import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Standard 36 alphanumeric characters
from ocr.charset import CHARSET, CHAR_TO_IDX

# Realistic Vietnamese province codes
PROVINCE_CODES = [
    "11", "12", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "43", "47", "48", "49", "50", "51", "52", "53",
    "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66",
    "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
    "81", "82", "83", "84", "85", "86", "88", "89", "90", "92", "93", "94", "95",
    "97", "98", "99"
]

# Standard vehicle series letters
STANDARD_SERIES = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "S", "T", "U", "V", "X", "Y", "Z"]
RARE_SERIES = ["J", "O", "Q", "W"]  # Special/diplomatic/military/test coverage


def generate_plate_text(layout: str, include_rare_p: float = 0.15) -> Tuple[str, str, str]:
    """
    Generates realistic plate components.
    Returns:
        gt_label: Normalized clean alphanumeric string (e.g. '30A12345')
        visual_line1: Visual string for line 1 (e.g. '30A' or '30A-123.45' for 1-row)
        visual_line2: Visual string for line 2 (e.g. '123.45' for 2-row, or '' for 1-row)
    """
    province = random.choice(PROVINCE_CODES)
    series_pool = RARE_SERIES if random.random() < include_rare_p else STANDARD_SERIES
    series_letter = random.choice(series_pool)

    # 4-digit (0.2 prob) or 5-digit (0.8 prob)
    is_5digit = random.random() < 0.8
    if is_5digit:
        d1 = str(random.randint(100, 999))
        d2 = f"{random.randint(10, 99):02d}"
        digits_raw = d1 + d2
        digits_visual = f"{d1}.{d2}"
    else:
        digits_raw = f"{random.randint(1000, 9999):04d}"
        digits_visual = digits_raw

    if layout == "one_row":
        gt_label = f"{province}{series_letter}{digits_raw}"
        visual_line1 = f"{province}{series_letter}-{digits_visual}"
        visual_line2 = ""
    else: # two_rows
        is_motorcycle = random.random() < 0.5
        if is_motorcycle:
            sub_num = str(random.randint(1, 9))
            moto_series = f"{series_letter}{sub_num}"
            gt_label = f"{province}{moto_series}{digits_raw}"
            visual_line1 = f"{province}-{moto_series}"
            visual_line2 = digits_visual
        else: # square car
            gt_label = f"{province}{series_letter}{digits_raw}"
            visual_line1 = f"{province}{series_letter}"
            visual_line2 = digits_visual

    return gt_label, visual_line1, visual_line2


def render_one_row(
    visual_text: str,
    font_path: Path,
    bg_path: Optional[Path] = None,
    target_size: Tuple[int, int] = (450, 110)
) -> Image.Image:
    """Renders single-row plate."""
    tw, th = target_size
    if bg_path and bg_path.exists():
        im = Image.open(str(bg_path)).convert("RGB").resize((tw, th), Image.Resampling.BILINEAR)
    else:
        im = Image.new("RGB", (tw, th), (250, 250, 250))
        # Draw simulated black border
        draw = ImageDraw.Draw(im)
        draw.rectangle([(4, 4), (tw - 5, th - 5)], outline=(20, 20, 20), width=3)

    font_size = 85 if len(visual_text) > 10 else 92
    font = ImageFont.truetype(str(font_path), font_size)
    draw = ImageDraw.Draw(im)

    bbox = draw.textbbox((0, 0), visual_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = max(10, int((tw - text_w) / 2))
    text_y = max(8, int((th - text_h) / 2) - 4)

    # Slight text shadow for realism
    shadow_col = (random.randint(180, 220), random.randint(180, 220), random.randint(180, 220))
    text_col = (random.randint(10, 35), random.randint(10, 35), random.randint(10, 35))
    
    draw.text((text_x + 1, text_y + 1), visual_text, font=font, fill=shadow_col)
    draw.text((text_x, text_y), visual_text, font=font, fill=text_col)
    return im


def render_two_rows(
    line1: str,
    line2: str,
    font_path: Path,
    bg_path: Optional[Path] = None,
    target_size: Tuple[int, int] = (320, 240)
) -> Image.Image:
    """Renders two-row plate."""
    tw, th = target_size
    if bg_path and bg_path.exists():
        im = Image.open(str(bg_path)).convert("RGB").resize((tw, th), Image.Resampling.BILINEAR)
    else:
        im = Image.new("RGB", (tw, th), (250, 250, 250))
        draw = ImageDraw.Draw(im)
        draw.rectangle([(4, 4), (tw - 5, th - 5)], outline=(20, 20, 20), width=3)

    font_size1 = 92
    font_size2 = 96
    font1 = ImageFont.truetype(str(font_path), font_size1)
    font2 = ImageFont.truetype(str(font_path), font_size2)
    draw = ImageDraw.Draw(im)

    # Line 1 bbox
    bbox1 = draw.textbbox((0, 0), line1, font=font1)
    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]

    # Line 2 bbox
    bbox2 = draw.textbbox((0, 0), line2, font=font2)
    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]

    x1 = max(10, int((tw - w1) / 2))
    y1 = int(th * 0.14)

    x2 = max(10, int((tw - w2) / 2))
    y2 = int(th * 0.52)

    shadow_col = (random.randint(180, 220), random.randint(180, 220), random.randint(180, 220))
    text_col = (random.randint(10, 35), random.randint(10, 35), random.randint(10, 35))

    # Line 1
    draw.text((x1 + 1, y1 + 1), line1, font=font1, fill=shadow_col)
    draw.text((x1, y1), line1, font=font1, fill=text_col)

    # Line 2
    draw.text((x2 + 1, y2 + 1), line2, font=font2, fill=shadow_col)
    draw.text((x2, y2), line2, font=font2, fill=text_col)
    return im


def apply_photorealistic_effects(pil_img: Image.Image) -> np.ndarray:
    """Applies mild perspective jitter, noise, and lighting variation."""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    # 1. Subtle rotation (-3.0 to +3.0 deg)
    angle = random.uniform(-3.0, 3.0)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cv_img = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 2. Subtle contrast/brightness
    alpha = random.uniform(0.90, 1.10)
    beta = random.uniform(-15, 15)
    cv_img = np.clip(alpha * cv_img + beta, 0, 255).astype(np.uint8)

    # 3. Subtle blur or noise
    if random.random() < 0.3:
        cv_img = cv2.GaussianBlur(cv_img, (3, 3), 0)
    elif random.random() < 0.2:
        noise = np.random.normal(0, random.uniform(2, 6), cv_img.shape).astype(np.float32)
        cv_img = np.clip(cv_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return cv_img


def generate_synthetic_dataset(
    num_samples: int,
    output_dir: Path,
    generator_root: Path,
    seed: int = 42,
    one_row_ratio: float = 0.5,
) -> List[Dict[str, str]]:
    random.seed(seed)
    np.random.seed(seed)

    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "labels.csv"

    font_path = generator_root / "MyFont-Regular_ver3.otf"
    if not font_path.exists():
        raise FileNotFoundError(f"Font not found: {font_path}")

    bg_dir = generator_root / "background"
    rec_bgs = list(bg_dir.glob("rec*.jpg"))
    square_bgs = list(bg_dir.glob("square*.jpg"))

    print(f"Generating {num_samples} synthetic plates into {output_dir}...")
    print(f"  Rec backgrounds:    {[b.name for b in rec_bgs]}")
    print(f"  Square backgrounds: {[b.name for b in square_bgs]}")
    print(f"  Font path:          {font_path.name}")

    manifest_rows = []
    
    for idx in range(1, num_samples + 1):
        layout = "one_row" if random.random() < one_row_ratio else "two_rows"
        gt_label, line1, line2 = generate_plate_text(layout)

        # Sanity check: must be strictly alphanumeric
        for ch in gt_label:
            assert ch in CHAR_TO_IDX, f"Generated character '{ch}' outside CHARS!"

        if layout == "one_row":
            bg = random.choice(rec_bgs) if rec_bgs and random.random() < 0.8 else None
            pil_img = render_one_row(line1, font_path=font_path, bg_path=bg)
        else:
            bg = random.choice(square_bgs) if square_bgs and random.random() < 0.8 else None
            pil_img = render_two_rows(line1, line2, font_path=font_path, bg_path=bg)

        # Realistic postprocessing
        final_bgr = apply_photorealistic_effects(pil_img)

        # Filename format: e.g. 00000001.jpg
        fname = f"{idx:08d}.jpg"
        save_path = images_out / fname
        cv2.imwrite(str(save_path), final_bgr)

        manifest_rows.append({
            "filename": fname,
            "label": gt_label,
            "layout": layout,
            "source": "synthetic",
        })

    # Save CSV
    with open(manifest_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["filename", "label", "layout", "source"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n[DONE] Generated {len(manifest_rows)} synthetic plates.")
    print(f"  Images dir:   {images_out}")
    print(f"  Manifest CSV: {manifest_path}")
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Vietnamese plates for LPRNet")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="dataset_staging/synthetic", help="Output directory")
    parser.add_argument(
        "--generator_root",
        type=str,
        default="datasets_external/vn_plate_generator/Vietnamese-License-Plate-Generator",
        help="Path to generator assets"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--one_row_ratio", type=float, default=0.5, help="Ratio of one-row plates")
    args = parser.parse_args()

    generate_synthetic_dataset(
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
        generator_root=Path(args.generator_root),
        seed=args.seed,
        one_row_ratio=args.one_row_ratio,
    )


if __name__ == "__main__":
    main()
