"""
Raspberry Pi 5 Real-Time License Plate Recognition from USB Camera (OpenCV V4L2 / MJPG).
Combines YOLO11n-416 ONNX Detector + LPRNet-Lite ONNX OCR using pure OpenCV DNN and NumPy.
Includes optional Multi-Frame Temporal Stabilization and Duplicate Suppression.

Usage:
    python3 scripts/pi_recognize_camera.py \
        --cam-id 0 \
        --width 1280 \
        --height 720 \
        --fps 30 \
        --detect-every 2 \
        --conf 0.5 \
        --iou 0.45
"""

import argparse
from collections import deque
from datetime import datetime
import os
from pathlib import Path
import sys
import time
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.pi_yolo_detector import PlateDetector
from ocr.pi_lprnet_ocr import PlateOCR, normalize_plate_text
from ocr.plate_stabilizer import PlateStabilizer
from scripts.pi_recognize_image import crop_with_padding, get_process_memory_mb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Vietnamese License Plate Recognition from USB Camera on Raspberry Pi 5."
    )
    # Camera arguments
    parser.add_argument("--cam-id", type=int, default=0, help="Camera device index (e.g. 0 for /dev/video0).")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera frame height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera capture FPS.")
    
    # Model arguments
    parser.add_argument("--detector", type=str, default="models/best_license_plate_yolo11n_416.onnx", help="Path to YOLO11n ONNX model.")
    parser.add_argument("--ocr", type=str, default="models/best_lprnet_vn.onnx", help="Path to LPRNet ONNX model.")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO detector input size.")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--crop-pad", type=float, default=0.05, help="Proportional padding around plate bounding box.")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads for OpenCV DNN.")

    # Pipeline optimization
    parser.add_argument("--detect-every", type=int, default=2, help="Run detector+OCR every N frames (1 = every frame).")
    
    # Display and storage
    parser.add_argument("--display-scale", type=float, default=1.0, help="Scaling factor for cv2.imshow preview window.")
    parser.add_argument("--headless", action="store_true", help="Run without cv2.imshow GUI window (useful for benchmarks).")
    parser.add_argument("--save-video", action="store_true", help="Record annotated output video.")
    parser.add_argument("--output", type=str, default="outputs/e2e/videos/camera_recording.mp4", help="Video output filepath.")
    parser.add_argument("--save-crops", action="store_true", help="Save debug plate crops.")
    parser.add_argument("--max-crops", type=int, default=30, help="Maximum debug crops to save.")
    parser.add_argument("--crops-dir", type=str, default="outputs/e2e/camera_crops", help="Directory for debug crops.")
    parser.add_argument("--snapshots-dir", type=str, default="outputs/e2e/camera_snapshots", help="Directory for snapshots (key 's').")
    
    # Temporal stabilization (Phase 2)
    parser.add_argument("--stabilize", action="store_true", help="Enable multi-frame temporal voting & stabilization.")
    parser.add_argument("--vote-window", type=int, default=5, help="Sliding window size for multi-frame voting.")
    parser.add_argument("--min-votes", type=int, default=3, help="Minimum votes to mark a plate as STABLE.")
    parser.add_argument("--plate-cooldown", type=float, default=3.0, help="Cooldown in seconds between duplicate terminal logs.")
    parser.add_argument("--iou-track", type=float, default=0.40, help="IoU threshold to associate bounding boxes across frames.")
    
    # Benchmark and logging controls
    parser.add_argument("--print-ocr", action="store_true", help="Print OCR recognition results directly to terminal.")
    parser.add_argument("--duration", type=float, default=0.0, help="Run for N seconds then exit automatically (0 = run indefinitely).")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (0 = unlimited).")

    return parser.parse_args()


def decode_fourcc(fourcc_val: int) -> str:
    """Decodes integer FOURCC to 4-character ASCII string."""
    try:
        chars = [chr((fourcc_val >> (8 * i)) & 0xFF) for i in range(4)]
        return "".join(chars)
    except Exception:
        return "UNKNOWN"


def draw_plate_overlay(
    annotated_img: np.ndarray,
    bbox: List[int],
    plate_text: str,
    confidence: float,
    is_stable: Optional[bool] = None,
    votes_info: Optional[str] = None,
):
    """Draws bounding box and stylized text label on the frame."""
    x1, y1, x2, y2 = bbox
    ih, iw = annotated_img.shape[:2]

    # Color: Green if stable / normal, Orange if unstable candidate
    if is_stable is True:
        box_color = (0, 220, 0)       # Vibrant green
        bg_color = (0, 160, 0)
    elif is_stable is False:
        box_color = (0, 180, 255)     # Amber / Yellow-orange
        bg_color = (0, 120, 200)
    else:
        box_color = (0, 255, 0)       # Standard green
        bg_color = (0, 160, 0)

    # Box
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)

    # Label text
    display_text = plate_text if plate_text else "UNKNOWN"
    if votes_info:
        label = f"{display_text} [{votes_info}] | {confidence:.2f}"
    elif is_stable is True:
        label = f"{display_text} [STABLE] | {confidence:.2f}"
    else:
        label = f"{display_text} | {confidence:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    bg_y1 = max(0, y1 - th - 8)
    bg_y2 = y1
    bg_x1 = x1
    bg_x2 = min(iw, x1 + tw + 8)

    cv2.rectangle(annotated_img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.putText(
        annotated_img,
        label,
        (bg_x1 + 4, bg_y2 - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_system_stats(
    annotated_img: np.ndarray,
    cam_info: str,
    fps: float,
    detect_every: int,
    plates_count: int,
    stabilize_enabled: bool,
):
    """Draws top-left semi-transparent status overlay HUD."""
    overlay_lines = [
        f"Camera: {cam_info}",
        f"FPS: {fps:.1f}",
        f"Detect every: {detect_every}",
        f"Plates: {plates_count}" + (" (Stabilized)" if stabilize_enabled else ""),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    line_h = 24

    hud_w = 320
    hud_h = len(overlay_lines) * line_h + 12
    sub_img = annotated_img[8:8 + hud_h, 10:10 + hud_w]
    if sub_img.shape[0] == hud_h and sub_img.shape[1] == hud_w:
        dark_rect = np.zeros_like(sub_img)
        cv2.addWeighted(sub_img, 0.4, dark_rect, 0.6, 0, sub_img)

    for i, line in enumerate(overlay_lines):
        y_pos = 28 + i * line_h
        cv2.putText(
            annotated_img,
            line,
            (16, y_pos),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def main():
    args = parse_args()

    # 1. Open Camera with V4L2 backend
    print(f"\nOpening Camera /dev/video{args.cam_id} using OpenCV V4L2...")
    cap = cv2.VideoCapture(args.cam_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera /dev/video{args.cam_id}")
        sys.exit(1)

    # Set MJPG FOURCC first
    fourcc_req = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_req)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    # Try setting buffer size to 1 to minimize latency
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Read back actual hardware values
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc = decode_fourcc(actual_fourcc_int)
    cam_info_str = f"{actual_w}x{actual_h} {actual_fourcc}"

    print("=" * 60)
    print("CAMERA INITIALIZATION")
    print(f"  Device:               /dev/video{args.cam_id}")
    print(f"  Requested resolution: {args.width}x{args.height}")
    print(f"  Actual resolution:    {actual_w}x{actual_h}")
    print(f"  Requested FPS:        {args.fps:.1f}")
    print(f"  Actual FPS:           {actual_fps:.1f}")
    print(f"  FOURCC:               {actual_fourcc}")
    print("=" * 60)
    print("MODELS & PIPELINE INITIALIZATION")
    print(f"  Detector:             {args.detector} (imgsz={args.imgsz}, conf={args.conf}, iou={args.iou})")
    print(f"  OCR:                  {args.ocr}")
    print(f"  Detect Every:         {args.detect_every}")
    print(f"  Display scale:        {args.display_scale}")
    print(f"  Headless mode:        {args.headless}")
    print(f"  Print OCR to terminal:{'ENABLED' if args.print_ocr else 'DISABLED'}")
    if args.stabilize:
        print(f"  Stabilization:        ENABLED (window={args.vote_window}, min_votes={args.min_votes}, cooldown={args.plate_cooldown}s)")
    else:
        print(f"  Stabilization:        DISABLED")
    print("=" * 60)

    # 2. Load Models Once
    t_load_start = time.perf_counter()
    detector = PlateDetector(
        model_path=args.detector,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        num_threads=args.threads,
    )
    ocr = PlateOCR(model_path=args.ocr)
    t_load_end = time.perf_counter()
    print(f"Both networks loaded successfully in {(t_load_end - t_load_start):.2f}s.")
    rss_mb = get_process_memory_mb()
    print(f"Process Resident Memory (RSS): {rss_mb:.1f} MB\n")

    # 3. Setup Optional Video Recording
    writer = None
    if args.save_video:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        rec_fps = actual_fps if actual_fps > 0 else 30.0
        fourcc_code = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc_code, rec_fps, (actual_w, actual_h))
        if not writer.isOpened():
            alt_out_p = out_p.with_suffix(".avi")
            fourcc_code = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(alt_out_p), fourcc_code, rec_fps, (actual_w, actual_h))
            out_p = alt_out_p
        print(f"Recording video to: {out_p}")

    # Setup directories
    if args.save_crops:
        crops_p = Path(args.crops_dir)
        crops_p.mkdir(parents=True, exist_ok=True)
    snapshots_p = Path(args.snapshots_dir)
    snapshots_p.mkdir(parents=True, exist_ok=True)

    # Setup Stabilizer
    stabilizer = PlateStabilizer(
        vote_window=args.vote_window,
        min_votes=args.min_votes,
        iou_threshold=args.iou_track,
        plate_cooldown=args.plate_cooldown,
    ) if args.stabilize else None

    # Performance metrics accumulators
    frame_idx = 0
    detection_runs = 0
    consecutive_read_fails = 0
    MAX_READ_FAILS = 30
    read_failure_count = 0
    saved_crops_count = 0

    total_capture_ms = 0.0
    total_det_prep_ms = 0.0
    total_det_fwd_ms = 0.0
    total_det_post_ms = 0.0
    total_crop_layout_ms = 0.0
    total_ocr_prep_ms = 0.0
    total_ocr_fwd_ms = 0.0
    total_ocr_dec_ms = 0.0
    total_drawing_ms = 0.0
    total_full_loop_ms = 0.0

    recent_frame_times = deque(maxlen=30)
    cached_display_results: List[Dict[str, Any]] = []
    paused = False

    window_name = "Vietnam Plate Recognition - Pi 5"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    t_start_wall = time.perf_counter()

    try:
        while True:
            t_loop_start = time.perf_counter()

            # Check time duration limit
            if args.duration > 0 and (t_loop_start - t_start_wall) >= args.duration:
                print(f"\nReached benchmark duration limit ({args.duration}s). Exiting loop...")
                break

            # Check max frames limit
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print(f"\nReached max frames limit ({args.max_frames}). Exiting loop...")
                break

            # 1. Capture Frame
            t_cap_0 = time.perf_counter()
            ret, frame = cap.read()
            t_cap_1 = time.perf_counter()

            if not ret or frame is None:
                consecutive_read_fails += 1
                read_failure_count += 1
                if consecutive_read_fails >= MAX_READ_FAILS:
                    print(f"\nERROR: Exceeded {MAX_READ_FAILS} consecutive frame read failures! Exiting.")
                    break
                time.sleep(0.01)
                continue

            consecutive_read_fails = 0
            frame_idx += 1
            cap_ms = (t_cap_1 - t_cap_0) * 1000.0
            total_capture_ms += cap_ms

            if paused:
                if not args.headless:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = False
                continue

            annotated_frame = frame.copy()
            should_detect = (frame_idx % args.detect_every == 0) or (frame_idx == 1)

            if should_detect:
                detection_runs += 1

                # 2. Detection
                detections, det_timings = detector.detect(frame)
                total_det_prep_ms += det_timings["prep_ms"]
                total_det_fwd_ms += det_timings["forward_ms"]
                total_det_post_ms += det_timings["post_ms"]

                current_results = []
                frame_crop_layout_ms = 0.0
                frame_ocr_prep_ms = 0.0
                frame_ocr_fwd_ms = 0.0
                frame_ocr_dec_ms = 0.0

                # 3. Crop & OCR each plate
                for det in detections:
                    t_cl_0 = time.perf_counter()
                    bbox = det["bbox"]
                    conf = det["confidence"]

                    crop, _ = crop_with_padding(frame, bbox, pad_ratio=args.crop_pad)
                    if crop.size == 0 or crop.shape[1] < 20 or crop.shape[0] < 12:
                        continue

                    # OCR recognition (handles layout estimation & rectification internally)
                    t_cl_1 = time.perf_counter()
                    ocr_res = ocr.recognize(crop, layout="auto")
                    t_ocr_end = time.perf_counter()

                    t_ocr_timings = ocr_res["timings"]
                    frame_ocr_prep_ms += t_ocr_timings["prep_ms"]
                    frame_ocr_fwd_ms += t_ocr_timings["forward_ms"]
                    frame_ocr_dec_ms += t_ocr_timings["decode_ms"]

                    # Subtract OCR time to isolate crop and layout estimation
                    cl_ms = (t_cl_1 - t_cl_0) * 1000.0 + max(0.0, ((t_ocr_end - t_cl_1) * 1000.0 - t_ocr_timings["total_ms"]))
                    frame_crop_layout_ms += cl_ms

                    plate_text = ocr_res["text"]
                    layout = ocr_res["layout"]

                    # Save debug crop if enabled
                    if args.save_crops and saved_crops_count < args.max_crops:
                        crop_filename = f"crop_{frame_idx:04d}_{saved_crops_count:02d}_{plate_text or 'blank'}.jpg"
                        cv2.imwrite(str(Path(args.crops_dir) / crop_filename), crop)
                        if ocr_res["rectified_image"] is not None:
                            rect_filename = f"crop_{frame_idx:04d}_{saved_crops_count:02d}_rectified.jpg"
                            cv2.imwrite(str(Path(args.crops_dir) / rect_filename), ocr_res["rectified_image"])
                        saved_crops_count += 1

                    current_results.append({
                        "bbox": bbox,
                        "confidence": conf,
                        "plate_text": plate_text,
                        "layout": layout,
                        "is_stable": False,
                        "votes": 1,
                        "total_samples": 1,
                    })

                total_crop_layout_ms += frame_crop_layout_ms
                total_ocr_prep_ms += frame_ocr_prep_ms
                total_ocr_fwd_ms += frame_ocr_fwd_ms
                total_ocr_dec_ms += frame_ocr_dec_ms

                # 4. Stabilization / Voting (Phase 2)
                if stabilizer:
                    stabilized_dets = stabilizer.update(current_results, current_time=time.perf_counter())
                    cached_display_results = stabilized_dets

                    # Terminal logging
                    if args.print_ocr:
                        for item in stabilized_dets:
                            is_st = bool(item.get("is_stable", False))
                            v_str = f"{item.get('votes', 1)}/{item.get('total_samples', 1)}"
                            plate_str = item.get("plate_text", "") or "UNKNOWN"
                            print("[OCR]")
                            print(f"Frame: {frame_idx}")
                            print(f"Plate: {plate_str}")
                            print(f"Detector confidence: {item['confidence']:.3f}")
                            print(f"Layout: {item.get('layout', 'unknown')}")
                            print(f"Stable: {'YES' if is_st else 'NO'}")
                            print(f"Votes: {v_str}")

                            if item.get("is_new_stable_event"):
                                print("[STABLE]")
                                print(f"Frame: {frame_idx}")
                                print(f"Plate: {plate_str}")
                                print(f"Votes: {v_str}")
                                print(f"Detector confidence: {item['confidence']:.3f}")
                    else:
                        for item in stabilized_dets:
                            if item.get("is_new_stable_event"):
                                votes_str = f"{item['votes']}/{item['total_samples']}"
                                print(f"[STABLE PLATE] Text: {item['plate_text']} | Conf: {item['confidence']:.2f} | Votes: {votes_str}")
                else:
                    cached_display_results = current_results
                    if args.print_ocr:
                        for item in current_results:
                            plate_str = item.get("plate_text", "") or "UNKNOWN"
                            print("[OCR]")
                            print(f"Frame: {frame_idx}")
                            print(f"Plate: {plate_str}")
                            print(f"Detector confidence: {item['confidence']:.3f}")
                            print(f"Layout: {item.get('layout', 'unknown')}")
                            print("Stable: NO")
                            print("Votes: 1/1")

            # 5. Drawing & Overlays
            t_draw_0 = time.perf_counter()
            for res in cached_display_results:
                is_st = res.get("is_stable", None)
                votes_info = f"{res['votes']}/{res['total_samples']}" if (args.stabilize and not is_st and res.get('votes', 0) > 0) else None
                draw_plate_overlay(
                    annotated_frame,
                    res["bbox"],
                    res["plate_text"],
                    res["confidence"],
                    is_stable=is_st,
                    votes_info=votes_info,
                )

            # Calculate Moving FPS
            recent_frame_times.append(time.perf_counter())
            if len(recent_frame_times) >= 2:
                moving_fps = len(recent_frame_times) / (recent_frame_times[-1] - recent_frame_times[0])
            else:
                moving_fps = 0.0

            draw_system_stats(
                annotated_frame,
                cam_info_str,
                moving_fps,
                args.detect_every,
                len(cached_display_results),
                args.stabilize,
            )

            if writer:
                writer.write(annotated_frame)

            t_draw_1 = time.perf_counter()
            total_drawing_ms += (t_draw_1 - t_draw_0) * 1000.0

            # 6. Realtime Display & Keyboard Interaction
            if not args.headless:
                if args.display_scale != 1.0:
                    disp_w = int(round(actual_w * args.display_scale))
                    disp_h = int(round(actual_h * args.display_scale))
                    display_frame = cv2.resize(annotated_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
                else:
                    display_frame = annotated_frame

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nUser pressed 'q'. Exiting...")
                    break
                elif key == ord('s'):
                    snap_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_path = snapshots_p / f"snapshot_{snap_time}_f{frame_idx:05d}.jpg"
                    cv2.imwrite(str(snap_path), annotated_frame)
                    print(f"  [Snapshot] Saved: {snap_path}")
                elif key == ord('p'):
                    paused = True
                    print("  [Stream Paused] Press 'p' again to resume or 'q' to quit.")

            t_loop_end = time.perf_counter()
            total_full_loop_ms += (t_loop_end - t_loop_start) * 1000.0

            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx:04d} | Moving Speed: {moving_fps:.1f} FPS | Active Plates: {len(cached_display_results)}")

    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    total_wall_time = time.perf_counter() - t_start_wall
    overall_realtime_fps = frame_idx / total_wall_time if total_wall_time > 0 else 0.0

    print("\n" + "=" * 60)
    print("CAMERA PROCESSING SUMMARY")
    print(f"  Total frames captured:    {frame_idx}")
    print(f"  Detection frames run:     {detection_runs}")
    print(f"  Frame read failures:      {read_failure_count}")
    print(f"  Total wall clock time:    {total_wall_time:.2f} s")
    print(f"  Realtime Pipeline FPS:    {overall_realtime_fps:.2f} FPS")
    print("-" * 60)
    print("AVERAGE LATENCY BREAKDOWN (per frame / detection):")
    if frame_idx > 0:
        print(f"  Camera Capture:           {(total_capture_ms / frame_idx):.2f} ms")
        print(f"  Drawing & HUD:            {(total_drawing_ms / frame_idx):.2f} ms")
        print(f"  Full Loop (all frames):   {(total_full_loop_ms / frame_idx):.2f} ms")
    if detection_runs > 0:
        print(f"  Detector Preprocess:      {(total_det_prep_ms / detection_runs):.2f} ms")
        print(f"  Detector Forward (CPU):   {(total_det_fwd_ms / detection_runs):.2f} ms")
        print(f"  Detector Postprocess:     {(total_det_post_ms / detection_runs):.2f} ms")
        print(f"  Crop & Layout:            {(total_crop_layout_ms / detection_runs):.2f} ms")
        print(f"  OCR Preprocess:           {(total_ocr_prep_ms / detection_runs):.2f} ms")
        print(f"  OCR Forward (CPU):        {(total_ocr_fwd_ms / detection_runs):.2f} ms")
        print(f"  OCR CTC Decode:           {(total_ocr_dec_ms / detection_runs):.2f} ms")
        total_ocr_avg = (total_ocr_prep_ms + total_ocr_fwd_ms + total_ocr_dec_ms) / detection_runs
        print(f"  Total OCR (per det frame):{total_ocr_avg:.2f} ms")

    if args.stabilize and stabilizer:
        print("-" * 60)
        print("TEMPORAL STABILIZATION METRICS:")
        print(f"  Vote Window:              {args.vote_window}")
        print(f"  Min Votes:                {args.min_votes}")
        print(f"  Raw OCR Predictions:      {stabilizer.total_raw_predictions}")
        print(f"  Stable Plate Events:      {stabilizer.total_stable_events}")
        print(f"  Unstable Transitions:     {stabilizer.unstable_transitions}")
    print("=" * 60)


if __name__ == "__main__":
    main()
