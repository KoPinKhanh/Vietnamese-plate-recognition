"""
Tool to verify license plate crop quality across test images.
Runs YOLO detector on 50+ test images, crops detected plates,
computes size/aspect ratio statistics, and builds a visual contact sheet grid.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def build_contact_sheet(crop_images, cols=5, tile_size=(240, 80), margin=4):
    """
    Assembles a list of cropped plate images into a neat contact sheet grid.
    Each tile is resized to tile_size (w, h) while preserving aspect ratio, centered with dark border.
    """
    if not crop_images:
        return None
        
    tw, th = tile_size
    n = len(crop_images)
    rows = (n + cols - 1) // cols
    
    sheet_w = cols * tw + (cols + 1) * margin
    sheet_h = rows * th + (rows + 1) * margin
    sheet = np.full((sheet_h, sheet_w, 3), 40, dtype=np.uint8)  # dark background
    
    for idx, crop in enumerate(crop_images):
        r = idx // cols
        c = idx % cols
        x = margin + c * (tw + margin)
        y = margin + r * (th + margin)
        
        # Fit inside tile keeping aspect ratio
        ch, cw = crop.shape[:2]
        if ch == 0 or cw == 0:
            continue
        scale = min(tw / cw, th / ch)
        scaled_w = max(1, int(cw * scale))
        scaled_h = max(1, int(ch * scale))
        
        resized = cv2.resize(crop, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        
        # Center in tile
        pad_x = (tw - scaled_w) // 2
        pad_y = (th - scaled_h) // 2
        
        tile_area = np.zeros((th, tw, 3), dtype=np.uint8)
        tile_area[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = resized
        
        # Draw border
        cv2.rectangle(tile_area, (0, 0), (tw - 1, th - 1), (100, 100, 100), 1)
        cv2.putText(tile_area, f"#{idx+1}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        sheet[y:y + th, x:x + tw] = tile_area
        
    return sheet


def main():
    parser = argparse.ArgumentParser(description="Check license plate crop quality.")
    parser.add_argument("--test-dir", type=str, default="vietnamese license plate.v1i.yolov11/test/images")
    parser.add_argument("--model", type=str, default="weights/best_license_plate_yolo11n.pt")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--output-sheet", type=str, default="outputs/crop_quality_contact_sheet.jpg")
    parser.add_argument("--output-report", type=str, default="outputs/crop_quality_report.txt")
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    assert test_dir.exists(), f"Test dir not found: {test_dir}"
    
    model = YOLO(args.model)
    image_paths = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:args.num_samples]
    
    crops = []
    widths = []
    heights = []
    aspect_ratios = []
    confidences = []
    
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        results = model.predict(source=img, conf=args.conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
            
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            conf = float(b.conf[0].cpu().numpy())
            bw = x2 - x1
            bh = y2 - y1
            
            pad_w = int(bw * args.padding)
            pad_h = int(bh * args.padding)
            
            cx1 = max(0, int(x1 - pad_w))
            cy1 = max(0, int(y1 - pad_h))
            cx2 = min(iw, int(x2 + pad_w))
            cy2 = min(ih, int(y2 + pad_h))
            
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                crops.append(crop)
                ch, cw = crop.shape[:2]
                widths.append(cw)
                heights.append(ch)
                aspect_ratios.append(cw / max(1, ch))
                confidences.append(conf)
                
    sheet = build_contact_sheet(crops, cols=5, tile_size=(240, 80))
    if sheet is not None:
        Path(args.output_sheet).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output_sheet, sheet)
        print(f"Contact sheet saved to: {args.output_sheet}")
        
    report = [
        "=" * 50,
        "     CROP QUALITY ASSESSMENT REPORT",
        "=" * 50,
        f"Images evaluated: {len(image_paths)}",
        f"Total plates detected: {len(crops)}",
        f"Average confidence: {np.mean(confidences):.3f}" if confidences else "N/A",
        f"Plate Width (px) : Min={min(widths):.0f}, Mean={np.mean(widths):.1f}, Max={max(widths):.0f}" if widths else "N/A",
        f"Plate Height (px): Min={min(heights):.0f}, Mean={np.mean(heights):.1f}, Max={max(heights):.0f}" if heights else "N/A",
        f"Aspect Ratio (W/H): Min={min(aspect_ratios):.2f}, Mean={np.mean(aspect_ratios):.2f}, Max={max(aspect_ratios):.2f}" if aspect_ratios else "N/A",
        "=" * 50,
    ]
    report_text = "\n".join(report)
    print(report_text)
    
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_text)


if __name__ == "__main__":
    main()
