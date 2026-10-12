"""
End-to-End License Plate Recognition for Video and Live Camera Streams.
Performs YOLO11n plate detection, crops plates, runs OCR every N frames with text persistence,
and renders real-time annotated video output.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ocr.plate_text_utils import normalize_plate_text, format_plate_text
from scripts.ocr_tesseract import run_tesseract
from scripts.crop_plate import crop_with_padding
from scripts.recognize_plate import draw_plate_box


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time License Plate Recognition on Video / Camera.")
    parser.add_argument("--source", type=str, default="0", help="Video file path or camera index (e.g., 0).")
    parser.add_argument("--model", type=str, default="weights/best_license_plate_yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument("--padding", type=float, default=0.05, help="Padding around detected plate.")
    parser.add_argument("--preprocess", type=str, default="otsu", help="OCR preprocessing strategy.")
    parser.add_argument("--ocr-every", type=int, default=5, help="Run OCR every N frames to maintain high FPS.")
    parser.add_argument("--save", action="store_true", help="Save annotated output video.")
    parser.add_argument("--output", type=str, default="outputs/recognition_video.mp4", help="Output video path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open GUI display window.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check source: webcam index or video file
    if args.source.isdigit():
        source_input = int(args.source)
        is_live = True
    else:
        source_input = str(Path(args.source).resolve())
        is_live = False
        if not Path(source_input).exists():
            print(f"Error: Video file '{source_input}' does not exist.")
            sys.exit(1)
            
    print(f"Loading YOLO detector from: {args.model}")
    model = YOLO(args.model)
    
    cap = cv2.VideoCapture(source_input)
    if not cap.isOpened():
        print(f"Error: Failed to open video source '{args.source}'.")
        sys.exit(1)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    writer = None
    if args.save:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))
        print(f"Saving output video to: {out_p}")
        
    frame_idx = 0
    cached_detections = []
    prev_time = time.time()
    
    print("\nStarting video processing. Press 'q' in the window to stop.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            curr_time = time.time()
            instant_fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time
            
            should_run_ocr = (frame_idx % args.ocr_every == 0) or (frame_idx == 1)
            
            # Detect plates
            results = model.predict(source=frame, conf=args.conf, verbose=False)
            boxes = results[0].boxes
            
            if boxes is not None and len(boxes) > 0:
                if should_run_ocr:
                    cached_detections = []
                    for b in boxes:
                        xyxy = b.xyxy[0].cpu().numpy()
                        conf = float(b.conf[0].cpu().numpy())
                        crop = crop_with_padding(frame, xyxy, padding=args.padding)
                        
                        ocr_res = {"normalized_text": "", "formatted_text": ""}
                        if crop.size > 0:
                            ocr_res = run_tesseract(crop, layout="auto", strategy=args.preprocess)
                            
                        cached_detections.append({
                            "bbox": xyxy,
                            "conf": conf,
                            "text": ocr_res["normalized_text"],
                            "formatted": ocr_res["formatted_text"],
                        })
                else:
                    # Update bounding boxes with current frame detection
                    for i, b in enumerate(boxes):
                        xyxy = b.xyxy[0].cpu().numpy()
                        conf = float(b.conf[0].cpu().numpy())
                        if i < len(cached_detections):
                            cached_detections[i]["bbox"] = xyxy
                            cached_detections[i]["conf"] = conf
                        else:
                            cached_detections.append({
                                "bbox": xyxy,
                                "conf": conf,
                                "text": "",
                                "formatted": "",
                            })
            else:
                if should_run_ocr:
                    cached_detections = []
                    
            # Draw detections
            for det in cached_detections:
                draw_plate_box(
                    frame,
                    det["bbox"],
                    det["text"],
                    det["formatted"],
                    conf=det["conf"],
                )
                
            # Draw FPS banner
            cv2.putText(
                frame,
                f"FPS: {instant_fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            
            if writer is not None:
                writer.write(frame)
                
            if not args.no_show:
                cv2.imshow("Vietnam License Plate Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                    
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nFinished processing. Processed {frame_idx} frames.")


if __name__ == "__main__":
    main()
