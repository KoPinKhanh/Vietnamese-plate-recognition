"""
Raspberry Pi 5 End-to-End License Plate Recognition for Video Files.
Combines YOLO11n-416 ONNX Detector + LPRNet-Lite ONNX OCR using pure OpenCV DNN and NumPy.

Usage:
    python3 scripts/pi_recognize_video.py \
        --source test_video.mp4 \
        --detector models/best_license_plate_yolo11n_416.onnx \
        --ocr models/best_lprnet_vn.onnx \
        --detect-every 1 \
        --output outputs/e2e/videos/output.mp4
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.pi_yolo_detector import PlateDetector
from ocr.pi_lprnet_ocr import PlateOCR
from scripts.pi_recognize_image import crop_with_padding, draw_plate_prediction, get_process_memory_mb


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Plate Recognition on Video via OpenCV DNN.")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--detector", type=str, default="models/best_license_plate_yolo11n_416.onnx")
    parser.add_argument("--ocr", type=str, default="models/best_lprnet_vn.onnx")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--crop-pad", type=float, default=0.05)
    parser.add_argument("--detect-every", type=int, default=1, help="Run detector+OCR every N frames.")
    parser.add_argument("--output", type=str, default="outputs/e2e/videos/recognized_video.mp4")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process (0 = all).")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.source)
    if not video_path.exists():
        print(f"Error: Input video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Failed to open video file: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))
    if not writer.isOpened():
        # Fallback to XVID / AVI if mp4v fails
        alt_out_p = out_p.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(alt_out_p), fourcc, fps, (width, height))
        out_p = alt_out_p

    print("=" * 60)
    print("PIPELINE INITIALIZATION")
    print(f"  Source:       {video_path} ({width}x{height} @ {fps:.1f} fps, {total_frames} frames)")
    print(f"  Detector:     {args.detector} (imgsz={args.imgsz}, conf={args.conf})")
    print(f"  OCR:          {args.ocr}")
    print(f"  Detect Every: {args.detect_every} frames")
    print(f"  Output:       {out_p}")
    print("=" * 60)

    detector = PlateDetector(
        model_path=args.detector,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        num_threads=args.threads
    )
    ocr = PlateOCR(model_path=args.ocr)

    rss_mb = get_process_memory_mb()
    print(f"Pipeline ready. Process RSS RAM: {rss_mb:.1f} MB\n")

    frame_idx = 0
    processed_count = 0
    cached_results: List[Dict[str, Any]] = []

    total_proc_time = 0.0
    total_det_time = 0.0
    total_ocr_time = 0.0

    t_pipeline_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if args.max_frames > 0 and frame_idx > args.max_frames:
            break

        t_frame_start = time.perf_counter()
        annotated_frame = frame.copy()

        should_detect = (frame_idx % args.detect_every == 0) or (frame_idx == 1)

        if should_detect:
            processed_count += 1
            # 1. Detection
            detections, det_timings = detector.detect(frame)
            total_det_time += det_timings["total_ms"]

            current_results = []
            frame_ocr_ms = 0.0

            # 2. OCR for each plate
            for det in detections:
                bbox = det["bbox"]
                conf = det["confidence"]

                crop, _ = crop_with_padding(frame, bbox, pad_ratio=args.crop_pad)
                if crop.size == 0 or crop.shape[1] < 20 or crop.shape[0] < 12:
                    continue

                ocr_res = ocr.recognize(crop, layout="auto")
                frame_ocr_ms += ocr_res["timings"]["total_ms"]

                current_results.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "plate_text": ocr_res["text"],
                    "layout": ocr_res["layout"],
                })

            total_ocr_time += frame_ocr_ms
            cached_results = current_results

        # Draw annotations (from current or cached result)
        for res in cached_results:
            draw_plate_prediction(
                annotated_frame,
                res["bbox"],
                res["plate_text"],
                res["confidence"]
            )

        # Draw FPS and frame index overlay
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_idx} | Plates: {len(cached_results)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(annotated_frame)

        t_frame_end = time.perf_counter()
        frame_time_ms = (t_frame_end - t_frame_start) * 1000.0
        total_proc_time += frame_time_ms

        if frame_idx % 10 == 0 or frame_idx == total_frames:
            avg_fps = frame_idx / ((time.perf_counter() - t_pipeline_start) + 1e-5)
            print(f"  Frame {frame_idx:03d}/{total_frames:03d} | Current Plates: {len(cached_results)} | Avg Pipeline Speed: {avg_fps:.2f} FPS")

    cap.release()
    writer.release()

    total_wall_time = time.perf_counter() - t_pipeline_start
    overall_fps = frame_idx / total_wall_time if total_wall_time > 0 else 0.0

    print("\n" + "=" * 60)
    print("VIDEO PROCESSING SUMMARY:")
    print(f"  Total frames processed: {frame_idx}")
    print(f"  Detection frames run:   {processed_count}")
    print(f"  Total wall clock time:  {total_wall_time:.2f} s")
    print(f"  Overall video FPS:      {overall_fps:.2f} FPS")
    if processed_count > 0:
        print(f"  Avg Detector latency:   {(total_det_time / processed_count):.2f} ms")
        print(f"  Avg OCR latency / frame: {(total_ocr_time / processed_count):.2f} ms")
    print(f"  Output saved to:        {out_p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
