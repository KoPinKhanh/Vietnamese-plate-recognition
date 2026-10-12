"""
Baseline OCR script using Tesseract for Vietnamese license plate recognition.
Supports both one-row and two-row plates with automatic or manual layout detection.
"""

import argparse
import sys
from pathlib import Path
import cv2
import pytesseract

# Import project OCR modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr.plate_text_utils import normalize_plate_text, format_plate_text
from ocr.preprocess import preprocess_plate, PREPROCESS_STRATEGIES

# Ensure pytesseract knows the Tesseract executable path if default on Windows
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if Path(TESSERACT_EXE).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


def detect_plate_layout(image) -> str:
    """
    Infers plate layout from aspect ratio (Width / Height):
    - One-row plates typically have W/H around 4.0 - 6.0
    - Two-row plates typically have W/H around 1.2 - 1.8
    Threshold: 2.2
    """
    h, w = image.shape[:2]
    if h == 0:
        return "one_row"
    aspect_ratio = w / float(h)
    return "one_row" if aspect_ratio > 2.2 else "two_rows"


def run_tesseract(
    image,
    layout: str = "auto",
    strategy: str = "clahe_otsu",
    custom_psm: int = None,
    whitelist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
) -> dict:
    """
    Runs Tesseract OCR on a cropped plate image.
    
    Returns:
        dict containing:
            - 'raw_text': direct string from Tesseract
            - 'normalized_text': uppercase cleaned [A-Z0-9]
            - 'formatted_text': human-friendly display text
            - 'inferred_layout': 'one_row' or 'two_rows'
            - 'psm_used': PSM integer mode
    """
    if layout == "auto":
        inferred_layout = detect_plate_layout(image)
    else:
        inferred_layout = layout
        
    if inferred_layout == "one_row":
        target_h = 80
        psm = custom_psm if custom_psm is not None else 7
        preprocessed = preprocess_plate(image, strategy=strategy, target_height=target_h, border_pad=10)
        config_parts = [f"--psm {psm}", "--oem 3"]
        if whitelist:
            config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
        config_str = " ".join(config_parts)
        try:
            raw_text = pytesseract.image_to_string(preprocessed, config=config_str)
        except Exception as e:
            raw_text = ""
    else:
        # Two-row plate: split top & bottom row with PSM 7 for best line isolation
        h, w = image.shape[:2]
        top_crop = image[:int(h * 0.52), :]
        bot_crop = image[int(h * 0.42):, :]
        
        p_top = preprocess_plate(top_crop, strategy=strategy, target_height=70, border_pad=10)
        p_bot = preprocess_plate(bot_crop, strategy=strategy, target_height=70, border_pad=10)
        
        psm = custom_psm if custom_psm is not None else 7
        config_parts = [f"--psm {psm}", "--oem 3"]
        if whitelist:
            config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
        config_str = " ".join(config_parts)
        
        try:
            top_raw = pytesseract.image_to_string(p_top, config=config_str).strip()
            bot_raw = pytesseract.image_to_string(p_bot, config=config_str).strip()
            raw_text = f"{top_raw}\n{bot_raw}" if top_raw or bot_raw else ""
            
            # Fallback to full block PSM 6 if both empty
            if not raw_text.strip():
                p_full = preprocess_plate(image, strategy=strategy, target_height=150, border_pad=10)
                raw_text = pytesseract.image_to_string(p_full, config="--psm 6 --oem 3")
        except Exception as e:
            raw_text = ""
        
    norm_text = normalize_plate_text(raw_text)
    fmt_text = format_plate_text(norm_text)
    
    return {
        "raw_text": raw_text.strip(),
        "normalized_text": norm_text,
        "formatted_text": fmt_text,
        "inferred_layout": inferred_layout,
        "psm_used": psm,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Tesseract OCR for Vietnamese License Plates.")
    parser.add_argument("--image", type=str, required=True, help="Path to cropped plate image or folder.")
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "one_row", "two_rows"])
    parser.add_argument("--preprocess", type=str, default="clahe_otsu", choices=PREPROCESS_STRATEGIES)
    parser.add_argument("--psm", type=int, default=None, help="Force specific PSM mode (e.g., 6, 7, 8, 11).")
    parser.add_argument("--whitelist", type=str, default="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return parser.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.image)
    
    if not img_path.exists():
        print(f"Error: Path '{img_path}' does not exist.")
        sys.exit(1)
        
    if img_path.is_file():
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error: Unable to read image '{img_path}'.")
            sys.exit(1)
            
        res = run_tesseract(
            img,
            layout=args.layout,
            strategy=args.preprocess,
            custom_psm=args.psm,
            whitelist=args.whitelist,
        )
        print("=" * 45)
        print("         TESSERACT OCR RESULT         ")
        print("=" * 45)
        print(f"Image File      : {img_path.name}")
        print(f"Inferred Layout : {res['inferred_layout']} (PSM {res['psm_used']})")
        print(f"Raw Text        : {repr(res['raw_text'])}")
        print(f"Normalized Text : {res['normalized_text']}")
        print(f"Formatted Text  : {res['formatted_text']}")
        print("=" * 45)
        
    elif img_path.is_dir():
        image_files = sorted(
            [p for p in img_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        print(f"Running Tesseract OCR on {len(image_files)} images in {img_path}...\n")
        print(f"{'Filename':<35} | {'Layout':<9} | {'Normalized':<12} | {'Formatted'}")
        print("-" * 75)
        for p in image_files[:50]:  # limit to first 50
            img = cv2.imread(str(p))
            if img is None:
                continue
            res = run_tesseract(
                img,
                layout=args.layout,
                strategy=args.preprocess,
                custom_psm=args.psm,
                whitelist=args.whitelist,
            )
            print(f"{p.name[:35]:<35} | {res['inferred_layout']:<9} | {res['normalized_text']:<12} | {res['formatted_text']}")


if __name__ == "__main__":
    main()
