"""
Full End-to-End License Plate Recognition Pipeline for Images.
Flow: Vehicle Image -> YOLO11n Plate Detection -> Crop Plate -> Preprocessing -> OCR -> Text Normalization -> Annotated Image.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr.plate_text_utils import normalize_plate_text, format_plate_text
from scripts.ocr_tesseract import run_tesseract
from scripts.crop_plate import crop_with_padding


def draw_plate_box(image, box, text, formatted_text, conf=None):
    """Draws a stylish, high-contrast bounding box and label banner above the plate."""
    x1, y1, x2, y2 = [int(v) for v in box]
    
    # Bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    display_label = f"{formatted_text}" if formatted_text else "License Plate"
    if conf is not None:
        display_label += f" ({conf:.2f})"
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    
    (lw, lh), baseline = cv2.getTextSize(display_label, font, scale, thickness)
    
    # Label background banner
    label_y1 = max(0, y1 - lh - 12)
    label_y2 = y1
    label_x1 = x1
    label_x2 = min(image.shape[1], x1 + lw + 16)
    
    cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), (0, 200, 0), -1)
    cv2.putText(
        image,
        display_label,
        (label_x1 + 8, label_y2 - 6),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return image


def recognize_image(
    image_path: Path,
    model: YOLO,
    det_conf: float = 0.5,
    padding: float = 0.05,
    preprocess_strategy: str = "otsu",
):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, []
        
    results = model.predict(source=img, conf=det_conf, verbose=False)
    boxes = results[0].boxes
    
    recognized_plates = []
    vis_img = img.copy()
    
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            conf = float(b.conf[0].cpu().numpy())
            
            crop = crop_with_padding(img, xyxy, padding=padding)
            if crop.size == 0:
                continue
                
            ocr_res = run_tesseract(crop, layout="auto", strategy=preprocess_strategy)
            
            recognized_plates.append({
                "bbox": xyxy,
                "det_conf": conf,
                "raw_text": ocr_res["raw_text"],
                "normalized_text": ocr_res["normalized_text"],
                "formatted_text": ocr_res["formatted_text"],
                "inferred_layout": ocr_res["inferred_layout"],
            })
            
            draw_plate_box(
                vis_img,
                xyxy,
                ocr_res["normalized_text"],
                ocr_res["formatted_text"],
                conf=conf,
            )
            
    return vis_img, recognized_plates


def main():
    parser = argparse.ArgumentParser(description="End-to-End Vietnamese License Plate Recognition.")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or image folder.")
    parser.add_argument("--model", type=str, default="weights/best_license_plate_yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO detector confidence threshold.")
    parser.add_argument("--padding", type=float, default=0.05, help="Crop padding around plate.")
    parser.add_argument("--preprocess", type=str, default="otsu", help="OCR preprocessing strategy.")
    parser.add_argument("--output", type=str, default="outputs/recognition", help="Output directory.")
    args = parser.parse_args()
    
    source_p = Path(args.source)
    if not source_p.exists():
        print(f"Error: Source '{source_p}' does not exist.")
        sys.exit(1)
        
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading YOLO detector: {args.model}...")
    model = YOLO(args.model)
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [source_p] if source_p.is_file() else sorted([p for p in source_p.iterdir() if p.suffix.lower() in valid_exts])
    
    print(f"Processing {len(image_files)} image(s)...")
    
    for p in image_files:
        vis_img, plates = recognize_image(
            p,
            model=model,
            det_conf=args.conf,
            padding=args.padding,
            preprocess_strategy=args.preprocess,
        )
        if vis_img is None:
            continue
            
        save_path = out_dir / f"recog_{p.name}"
        cv2.imwrite(str(save_path), vis_img)
        
        print(f"\n[+] {p.name}:")
        if plates:
            for i, pl in enumerate(plates, 1):
                print(f"    Plate #{i}: {pl['normalized_text']} (Formatted: {pl['formatted_text']}, Conf: {pl['det_conf']:.2f}, Layout: {pl['inferred_layout']})")
        else:
            print("    No license plates detected.")
            
    print(f"\nAll annotated results saved to: {out_dir}")


if __name__ == "__main__":
    main()
