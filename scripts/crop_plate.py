"""
Script to detect license plates in images using YOLO11n detector and crop them.
Saves cropped plates and optional detection debug images with bounding boxes.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Crop license plates using YOLO11n detector.")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to an image or directory of images.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="weights/best_license_plate_yolo11n.pt",
        help="Path to trained YOLO detector weights.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Fractional padding around bounding box (default: 0.05 = 5%).",
    )
    parser.add_argument(
        "--output-crops",
        type=str,
        default="outputs/plate_crops",
        help="Directory to save cropped plates.",
    )
    parser.add_argument(
        "--output-debug",
        type=str,
        default="outputs/detection_debug",
        help="Directory to save detection debug visualization.",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug visualization images.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (useful for testing).",
    )
    return parser.parse_args()


def get_image_files(source_path: Path):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source_path.is_file():
        if source_path.suffix.lower() in valid_exts:
            return [source_path]
        return []
    elif source_path.is_dir():
        return [p for p in source_path.iterdir() if p.suffix.lower() in valid_exts]
    return []


def crop_with_padding(image, box, padding=0.05):
    """
    Crops image at box [x1, y1, x2, y2] with specified fractional padding, clamped to bounds.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    
    pad_w = int(bw * padding)
    pad_h = int(bh * padding)
    
    crop_x1 = max(0, int(x1 - pad_w))
    crop_y1 = max(0, int(y1 - pad_h))
    crop_x2 = min(w, int(x2 + pad_w))
    crop_y2 = min(h, int(y2 + pad_h))
    
    return image[crop_y1:crop_y2, crop_x1:crop_x2]


def main():
    args = parse_args()
    
    source = Path(args.source)
    if not source.exists():
        print(f"Error: Source '{source}' does not exist.")
        sys.exit(1)
        
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model weights not found at '{model_path}'.")
        sys.exit(1)
        
    print(f"Loading YOLO detector model from: {model_path}")
    model = YOLO(str(model_path))
    
    image_paths = get_image_files(source)
    if not image_paths:
        print(f"No valid images found at '{source}'.")
        sys.exit(1)
        
    if args.max_images:
        image_paths = image_paths[:args.max_images]
        
    out_crops_dir = Path(args.output_crops)
    out_crops_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_debug:
        out_debug_dir = Path(args.output_debug)
        out_debug_dir.mkdir(parents=True, exist_ok=True)
        
    print(f"Processing {len(image_paths)} images with conf={args.conf}, padding={args.padding}...")
    
    total_detected = 0
    total_crops = 0
    
    for idx, img_p in enumerate(image_paths, 1):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
            
        results = model.predict(source=img, conf=args.conf, verbose=False)
        boxes = results[0].boxes
        
        num_boxes = len(boxes) if boxes is not None else 0
        if num_boxes > 0:
            total_detected += 1
            
        debug_img = img.copy() if args.save_debug else None
        
        for b_idx, box_data in enumerate(boxes):
            xyxy = box_data.xyxy[0].cpu().numpy()
            conf = float(box_data.conf[0].cpu().numpy())
            
            crop = crop_with_padding(img, xyxy, padding=args.padding)
            crop_filename = f"{img_p.stem}_crop_{b_idx + 1}.jpg" if num_boxes > 1 else f"{img_p.stem}_crop.jpg"
            crop_path = out_crops_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)
            total_crops += 1
            
            if args.save_debug and debug_img is not None:
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"LP {conf:.2f}"
                cv2.putText(
                    debug_img,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                
        if args.save_debug and debug_img is not None:
            debug_path = out_debug_dir / f"{img_p.stem}_debug.jpg"
            cv2.imwrite(str(debug_path), debug_img)
            
        if idx % 100 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] Done. Total plates cropped: {total_crops}")
            
    print(f"\nFinished!")
    print(f"Processed images: {len(image_paths)}")
    print(f"Images with at least 1 plate: {total_detected} ({(total_detected/len(image_paths))*100:.1f}%)")
    print(f"Total plate crops saved: {total_crops} -> {out_crops_dir}")
    if args.save_debug:
        print(f"Debug images saved -> {out_debug_dir}")


if __name__ == "__main__":
    main()
