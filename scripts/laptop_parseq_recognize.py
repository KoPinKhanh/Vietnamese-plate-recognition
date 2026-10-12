"""
Real Laptop End-to-End Pipeline: YOLO11n ONNX Plate Detector + PARSeq-Tiny VNLP OCR.
Supports IMAGE, VIDEO, and CAMERA sources with optional temporal stabilization,
debug overlays, crop saving, and low-confidence logging.

Usage Examples:
  # Image:
  python scripts/laptop_parseq_recognize.py --source image --input "path/to/image.jpg" --save --show-debug

  # Video:
  python scripts/laptop_parseq_recognize.py --source video --input "path/to/video.mp4" --save --show-debug --stabilize

  # Camera:
  python scripts/laptop_parseq_recognize.py --source camera --camera-id 0 --show-debug --stabilize
"""

import argparse
import csv
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.pi_yolo_detector import PlateDetector
from ocr.parseq_vn_ocr import ParseqVNRecognizer
from ocr.plate_stabilizer import PlateStabilizer
from ocr.plate_text_utils import normalize_plate_text, format_plate_text


def crop_with_padding(img: np.ndarray, bbox: List[int], pad_ratio: float = 0.05) -> Tuple[np.ndarray, List[int]]:
    """Crops plate from image with proportional padding clamped to boundaries."""
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    pad_w = int(round(bw * pad_ratio))
    pad_h = int(round(bh * pad_ratio))

    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(iw, x2 + pad_w)
    cy2 = min(ih, y2 + pad_h)

    crop = img[cy1:cy2, cx1:cx2]
    return crop, [cx1, cy1, cx2, cy2]


def draw_detection_overlay(
    image: np.ndarray,
    bbox: List[int],
    plate_text: str,
    det_conf: float,
    ocr_conf: float,
    layout: str = "one_row",
    is_stable: Optional[bool] = None,
    track_id: Optional[int] = None,
) -> np.ndarray:
    """
    Renders high-visibility bounding box and metadata banner above the detected plate.
    """
    x1, y1, x2, y2 = bbox
    ih, iw = image.shape[:2]

    # Color scheme: Green for stable, Amber for high confidence, Cyan for standard
    if is_stable is True:
        box_color = (0, 220, 0)      # Vivid Green
        banner_bg = (0, 160, 0)
    elif ocr_conf >= 0.90:
        box_color = (0, 215, 255)    # Gold / Amber
        banner_bg = (0, 140, 200)
    else:
        box_color = (255, 180, 0)    # Blue / Cyan
        banner_bg = (180, 100, 0)

    # 1. Draw bounding box with subtle corner accents
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

    # 2. Text strings
    main_text = plate_text if plate_text else "UNKNOWN"
    sub_text = f"Det:{det_conf:.2f} | OCR:{ocr_conf:.2f} | {layout}"
    if track_id is not None:
        sub_text = f"ID:{track_id} | " + sub_text
    if is_stable is not None:
        sub_text += " [STABLE]" if is_stable else " [RAW]"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_main_scale = 0.65
    font_sub_scale = 0.42
    thick_main = 2
    thick_sub = 1

    (tw1, th1), _ = cv2.getTextSize(main_text, font, font_main_scale, thick_main)
    (tw2, th2), _ = cv2.getTextSize(sub_text, font, font_sub_scale, thick_sub)

    banner_w = max(tw1, tw2) + 14
    banner_h = th1 + th2 + 16

    # Position banner above box; if not enough room, position below or inside
    if y1 - banner_h - 4 >= 0:
        by1 = y1 - banner_h - 4
        by2 = y1 - 4
    elif y2 + banner_h + 4 <= ih:
        by1 = y2 + 4
        by2 = y2 + banner_h + 4
    else:
        by1 = max(0, y1)
        by2 = min(ih, y1 + banner_h)

    bx1 = max(0, min(iw - banner_w, x1))
    bx2 = min(iw, bx1 + banner_w)

    # Semi-transparent banner background
    overlay = image.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), banner_bg, -1)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    # Border for banner
    cv2.rectangle(image, (bx1, by1), (bx2, by2), box_color, 1)

    # Render main text
    ty1 = by1 + th1 + 5
    cv2.putText(image, main_text, (bx1 + 6, ty1), font, font_main_scale, (255, 255, 255), thick_main, cv2.LINE_AA)

    # Render sub text
    ty2 = ty1 + th2 + 6
    cv2.putText(image, sub_text, (bx1 + 6, ty2), font, font_sub_scale, (230, 240, 240), thick_sub, cv2.LINE_AA)

    return image


def draw_debug_panel(
    image: np.ndarray,
    timings: Dict[str, float],
    fps: float,
    num_plates: int,
    ocr_name: str = "PARSeq-Tiny",
    device_name: str = "CUDA",
) -> np.ndarray:
    """Draws top-left status and latency dashboard overlay."""
    panel_w = 260
    panel_h = 135
    ih, iw = image.shape[:2]

    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    cv2.rectangle(image, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 80), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        (f"LAPTOP PIPELINE ({ocr_name})", (0, 255, 255), 0.45, 1),
        (f"Device: {device_name} | FPS: {fps:.1f}", (255, 255, 255), 0.45, 1),
        (f"Det YOLO11n:  {timings.get('detector_ms', 0.0):.1f} ms", (200, 220, 255), 0.42, 1),
        (f"OCR ({num_plates} pl):    {timings.get('ocr_ms', 0.0):.1f} ms", (200, 220, 255), 0.42, 1),
        (f"Pre/Stab/Draw: {timings.get('other_ms', 0.0):.1f} ms", (200, 220, 255), 0.42, 1),
        (f"Total Frame:   {timings.get('total_ms', 0.0):.1f} ms", (0, 255, 128), 0.45, 1),
    ]

    y = 28
    for text, color, scale, thick in lines:
        cv2.putText(image, text, (18, y), font, scale, color, thick, cv2.LINE_AA)
        y += 19

    return image


class RealLaptopPipeline:
    """
    Unified Laptop License Plate Recognition Pipeline.
    Integrates YOLO11n-416 ONNX Detector + PARSeq-Tiny VNLP PyTorch OCR.
    """

    def __init__(
        self,
        detector_path: str = "weights/best_license_plate_yolo11n_416.onnx",
        ocr_model: str = "parseq",
        ocr_checkpoint: str = "weights/ocr/teachers/parseq_tiny_vn/best.pth",
        device: Optional[str] = None,
        use_amp: bool = False,
        imgsz: int = 416,
        det_conf: float = 0.50,
        det_iou: float = 0.45,
        ocr_conf_thresh: float = 0.0,
        crop_pad: float = 0.05,
        enable_stabilizer: bool = True,
        save_crops: bool = False,
        log_low_confidence: bool = False,
        low_conf_thresh: float = 0.80,
        collect: bool = False,
        output_dir: Path = Path("outputs/parseq_laptop"),
    ):
        self.det_conf = det_conf
        self.det_iou = det_iou
        self.ocr_conf_thresh = ocr_conf_thresh
        self.crop_pad = crop_pad
        self.save_crops = save_crops
        self.log_low_confidence = log_low_confidence
        self.low_conf_thresh = low_conf_thresh
        self.collect = collect
        self.enable_stabilizer = enable_stabilizer
        self.output_dir = Path(output_dir)

        # Output folders
        self.img_out_dir = self.output_dir / "images"
        self.vid_out_dir = self.output_dir / "videos"
        self.raw_crops_dir = self.output_dir / "crops" / "raw"
        self.proc_crops_dir = self.output_dir / "crops" / "processed"
        self.collect_dir = Path("dataset_staging/real_world_review")
        self.low_conf_csv = self.output_dir / "low_confidence.csv"

        for p in (self.img_out_dir, self.vid_out_dir, self.raw_crops_dir, self.proc_crops_dir, self.collect_dir):
            p.mkdir(parents=True, exist_ok=True)

        # 1. Initialize Detector (OpenCV DNN CPU)
        print(f"[Pipeline] Loading YOLO11n Detector from {detector_path} (imgsz={imgsz}, conf={det_conf}, iou={det_iou})...")
        self.detector = PlateDetector(
            model_path=str(detector_path),
            imgsz=imgsz,
            conf=det_conf,
            iou=det_iou,
        )

        # 2. Initialize OCR Model
        self.ocr_model_type = ocr_model.lower()
        if self.ocr_model_type == "parseq":
            self.ocr = ParseqVNRecognizer(
                checkpoint=ocr_checkpoint,
                device=device,
                use_amp=use_amp,
            )
            self.ocr_name = "PARSeq-Tiny VNLP"
        elif self.ocr_model_type == "lprnet":
            from ocr.pi_lprnet_ocr import PlateOCR
            lpr_path = "weights/ocr/best_lprnet_vn.onnx"
            print(f"[Pipeline] Loading A/B Reference LPRNet from {lpr_path}...")
            self.ocr = PlateOCR(model_path=lpr_path)
            self.ocr_name = "LPRNet-Lite (A/B)"
        else:
            raise ValueError(f"Unknown OCR type: {ocr_model}")

        # 3. Initialize Temporal Stabilizer
        if self.enable_stabilizer:
            self.stabilizer = PlateStabilizer(vote_window=5, min_votes=3, iou_threshold=0.40)
        else:
            self.stabilizer = None

        # Low confidence CSV setup
        if self.log_low_confidence and not self.low_conf_csv.exists():
            with open(self.low_conf_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "source", "frame", "bbox", "layout",
                    "det_conf", "ocr_text", "ocr_conf", "raw_crop_path", "processed_crop_path"
                ])

        # Automatic initial warmup
        self.warmup(10)

    def warmup(self, num_runs: int = 15):
        """Warms up detector and OCR before benchmarking."""
        print(f"[Pipeline] Running {num_runs} warmup iterations...")
        dummy_frame = np.full((416, 416, 3), 114, dtype=np.uint8)
        for _ in range(5):
            _ = self.detector.detect(dummy_frame)

        if hasattr(self.ocr, "warmup"):
            self.ocr.warmup(num_runs)
        print("[Pipeline] Warmup complete.")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        source_name: str = "frame",
        show_debug: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, float]]:
        """
        Executes end-to-end detection, layout rectification, OCR, temporal stabilization, and overlay rendering.
        """
        t_frame_start = time.perf_counter()
        annotated = frame.copy()

        # 1. Plate Detection
        detections, det_timings = self.detector.detect(frame)
        detector_ms = det_timings["total_ms"]

        # 2. Crop, Rectify & OCR
        t_crop_start = time.perf_counter()
        plate_results = []
        raw_crops = []
        layouts = []
        filtered_detections = []

        total_ocr_ms = 0.0
        total_prep_ms = 0.0

        for idx, det in enumerate(detections, 1):
            bbox = det["bbox"]
            conf = det["confidence"]
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1

            # Discard degenerate small detections
            if bw < 18 or bh < 12:
                continue

            crop, padded_box = crop_with_padding(frame, bbox, pad_ratio=self.crop_pad)
            if crop.size == 0:
                continue

            # Run OCR
            if self.ocr_model_type == "parseq":
                ocr_res = self.ocr.recognize(crop, layout="auto")
                plate_text = ocr_res["text"]
                raw_text = ocr_res["raw_text"]
                ocr_conf = ocr_res["confidence"]
                layout = ocr_res["layout"]
                rectified_img = ocr_res["rectified_image"]
                processed_img = ocr_res["processed_image"]
                plate_ocr_ms = ocr_res["ocr_ms"]
                total_prep_ms += ocr_res["timings"]["prep_ms"]
            else:
                # LPRNet fallback
                ocr_res = self.ocr.recognize(crop, layout="auto")
                plate_text = ocr_res["text"]
                raw_text = plate_text
                ocr_conf = 0.90
                layout = ocr_res["layout"]
                rectified_img = ocr_res["rectified_image"]
                processed_img = crop
                plate_ocr_ms = ocr_res["timings"]["total_ms"]
                total_prep_ms += ocr_res["timings"]["prep_ms"]

            total_ocr_ms += plate_ocr_ms

            # Filter by OCR confidence threshold if set
            if self.ocr_conf_thresh > 0.0 and ocr_conf < self.ocr_conf_thresh:
                continue

            # Crop saving
            raw_crop_path = ""
            proc_crop_path = ""
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]

            if self.save_crops:
                raw_name = f"{ts_str}_f{frame_idx:04d}_p{idx:02d}_{plate_text}_{ocr_conf:.2f}_raw.jpg"
                proc_name = f"{ts_str}_f{frame_idx:04d}_p{idx:02d}_{plate_text}_{ocr_conf:.2f}_proc.jpg"
                raw_crop_path = str(self.raw_crops_dir / raw_name)
                proc_crop_path = str(self.proc_crops_dir / proc_name)
                cv2.imwrite(raw_crop_path, crop)
                target_proc = rectified_img if rectified_img is not None else processed_img
                cv2.imwrite(proc_crop_path, target_proc)

            # Low confidence logging
            is_low_conf = (ocr_conf < self.low_conf_thresh)
            if self.log_low_confidence and is_low_conf:
                if not raw_crop_path:
                    raw_name = f"low_{ts_str}_f{frame_idx:04d}_p{idx:02d}_{plate_text}_{ocr_conf:.2f}_raw.jpg"
                    proc_name = f"low_{ts_str}_f{frame_idx:04d}_p{idx:02d}_{plate_text}_{ocr_conf:.2f}_proc.jpg"
                    raw_crop_path = str(self.raw_crops_dir / raw_name)
                    proc_crop_path = str(self.proc_crops_dir / proc_name)
                    cv2.imwrite(raw_crop_path, crop)
                    target_proc = rectified_img if rectified_img is not None else processed_img
                    cv2.imwrite(proc_crop_path, target_proc)

                with open(self.low_conf_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        ts_str, source_name, frame_idx, f"[{x1},{y1},{x2},{y2}]",
                        layout, f"{conf:.3f}", plate_text, f"{ocr_conf:.3f}",
                        raw_crop_path, proc_crop_path
                    ])

            # Real-world data collection for review
            if self.collect and is_low_conf:
                coll_name = f"review_{ts_str}_f{frame_idx:04d}_{plate_text}_{ocr_conf:.2f}.jpg"
                cv2.imwrite(str(self.collect_dir / coll_name), crop)

            plate_info = {
                "bbox": bbox,
                "confidence": conf,
                "layout": layout,
                "raw_text": raw_text,
                "plate_text": plate_text,
                "ocr_conf": ocr_conf,
                "ocr_ms": plate_ocr_ms,
                "rectified": rectified_img is not None,
            }
            plate_results.append(plate_info)

        t_ocr_end = time.perf_counter()

        # 3. Temporal Stabilization
        t_stab_start = time.perf_counter()
        if self.stabilizer is not None and plate_results:
            stabilized_plates = self.stabilizer.update(plate_results, current_time=t_stab_start)
            # Preserve ocr_conf, ocr_ms, and original metadata
            for s_idx, s_item in enumerate(stabilized_plates):
                if s_idx < len(plate_results):
                    s_item["ocr_conf"] = plate_results[s_idx].get("ocr_conf", 0.95)
                    s_item["ocr_ms"] = plate_results[s_idx].get("ocr_ms", 0.0)
                else:
                    s_item.setdefault("ocr_conf", 0.95)
                    s_item.setdefault("ocr_ms", 0.0)
        else:
            stabilized_plates = []
            for p in plate_results:
                stabilized_plates.append({
                    "bbox": p["bbox"],
                    "confidence": p["confidence"],
                    "layout": p["layout"],
                    "raw_text": p["raw_text"],
                    "plate_text": p["plate_text"],
                    "ocr_conf": p["ocr_conf"],
                    "ocr_ms": p["ocr_ms"],
                    "is_stable": None,
                    "track_id": None,
                })
        t_stab_end = time.perf_counter()
        stabilizer_ms = (t_stab_end - t_stab_start) * 1000.0

        # 4. Rendering / Drawing
        t_draw_start = time.perf_counter()
        for item in stabilized_plates:
            draw_detection_overlay(
                image=annotated,
                bbox=item["bbox"],
                plate_text=item["plate_text"],
                det_conf=item["confidence"],
                ocr_conf=item.get("ocr_conf", 0.95),
                layout=item.get("layout", "one_row"),
                is_stable=item.get("is_stable"),
                track_id=item.get("track_id"),
            )
        t_draw_end = time.perf_counter()
        draw_ms = (t_draw_end - t_draw_start) * 1000.0

        t_frame_end = time.perf_counter()
        total_frame_ms = (t_frame_end - t_frame_start) * 1000.0
        fps = 1000.0 / total_frame_ms if total_frame_ms > 0 else 0.0

        timings = {
            "detector_ms": detector_ms,
            "crop_prep_ms": total_prep_ms,
            "ocr_ms": total_ocr_ms,
            "stabilizer_ms": stabilizer_ms,
            "draw_ms": draw_ms,
            "other_ms": total_frame_ms - (detector_ms + total_ocr_ms),
            "total_ms": total_frame_ms,
            "fps": fps,
        }

        # 5. Debug Dashboard Overlay
        if show_debug:
            device_str = getattr(self.ocr, "device_str", "CPU").upper()
            draw_debug_panel(
                annotated,
                timings=timings,
                fps=fps,
                num_plates=len(stabilized_plates),
                ocr_name=self.ocr_name,
                device_name=device_str,
            )

        return annotated, stabilized_plates, timings

    def run_image(
        self,
        image_path: Union[str, Path],
        save: bool = True,
        show_debug: bool = False,
        no_gui: bool = False,
    ) -> Dict[str, Any]:
        """Runs end-to-end pipeline on a single image."""
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        print(f"\n[IMAGE MODE] Processing: {image_path} ({img.shape[1]}x{img.shape[0]})")
        annotated, plates, timings = self.process_frame(
            img,
            frame_idx=1,
            source_name=image_path.name,
            show_debug=show_debug,
        )

        print("-" * 65)
        print(f"Detected Plates: {len(plates)}")
        for idx, p in enumerate(plates, 1):
            formatted = format_plate_text(p["plate_text"])
            print(f"  Plate #{idx}: '{p['plate_text']}' ({formatted}) | "
                  f"DetConf: {p['confidence']:.2f} | OCRConf: {p.get('ocr_conf', 0.0):.4f} | "
                  f"Layout: {p['layout']} | BBox: {p['bbox']} | OCR: {p.get('ocr_ms', 0.0):.1f} ms")

        print("-" * 65)
        print(f"Timing Breakdown: Det={timings['detector_ms']:.1f}ms | OCR={timings['ocr_ms']:.1f}ms | "
              f"Total={timings['total_ms']:.1f}ms | FPS={timings['fps']:.1f}")

        out_path = None
        if save:
            out_name = f"pred_{image_path.stem}.jpg"
            out_path = self.img_out_dir / out_name
            cv2.imwrite(str(out_path), annotated)
            print(f"Saved annotated image to: {out_path}")

        if not no_gui:
            try:
                cv2.imshow("PARSeq Laptop Pipeline - Image Result", annotated)
                print("Press any key in the GUI window to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"[GUI Notice] cv2.imshow not available ({e}). Skipping window display.")

        return {
            "image_path": str(image_path),
            "output_path": str(out_path) if out_path else None,
            "plates": plates,
            "timings": timings,
        }

    def run_video(
        self,
        video_path: Union[str, Path],
        save: bool = True,
        show_debug: bool = False,
        no_gui: bool = False,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs end-to-end pipeline on a video stream."""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_vframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n[VIDEO MODE] Source: {video_path}")
        print(f"Resolution: {vw}x{vh} @ {vfps:.1f} FPS | Total Frames: {total_vframes}")

        # Warmup
        self.warmup(10)

        # Video writer setup
        writer = None
        out_video_path = None
        if save:
            out_name = f"pred_{video_path.stem}.mp4"
            out_video_path = self.vid_out_dir / out_name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, vfps, (vw, vh))

        frame_idx = 0
        all_det_ms = []
        all_ocr_ms = []
        all_total_ms = []
        total_plates_detected = 0

        print("[VIDEO MODE] Processing frames...")
        t_vid_start = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            annotated, plates, timings = self.process_frame(
                frame,
                frame_idx=frame_idx,
                source_name=video_path.name,
                show_debug=show_debug,
            )

            all_det_ms.append(timings["detector_ms"])
            all_ocr_ms.append(timings["ocr_ms"])
            all_total_ms.append(timings["total_ms"])
            total_plates_detected += len(plates)

            if writer:
                writer.write(annotated)

            if not no_gui:
                try:
                    cv2.imshow("PARSeq Laptop Pipeline - Video Stream", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[VIDEO MODE] User requested quit.")
                        break
                except Exception:
                    pass

            # Terminal progress every 15 frames
            if frame_idx % 15 == 0 or frame_idx == total_vframes:
                avg_fps = 1000.0 / (sum(all_total_ms[-15:]) / len(all_total_ms[-15:]))
                print(f"  Frame [{frame_idx:04d}/{total_vframes}] | Plates: {len(plates)} | "
                      f"Det: {timings['detector_ms']:.1f}ms | OCR: {timings['ocr_ms']:.1f}ms | "
                      f"Total: {timings['total_ms']:.1f}ms ({avg_fps:.1f} FPS)")

        cap.release()
        if writer:
            writer.release()
        if not no_gui:
            cv2.destroyAllWindows()

        elapsed_vid = time.perf_counter() - t_vid_start
        mean_det = sum(all_det_ms) / len(all_det_ms) if all_det_ms else 0.0
        mean_ocr = sum(all_ocr_ms) / len(all_ocr_ms) if all_ocr_ms else 0.0
        mean_tot = sum(all_total_ms) / len(all_total_ms) if all_total_ms else 0.0
        mean_fps = 1000.0 / mean_tot if mean_tot > 0 else 0.0

        print("=" * 65)
        print(f"[VIDEO COMPLETE] Processed {frame_idx} frames in {elapsed_vid:.2f}s")
        print(f"Total Plates Detected: {total_plates_detected}")
        print(f"Mean Detector Latency: {mean_det:.2f} ms")
        print(f"Mean PARSeq OCR Time:  {mean_ocr:.2f} ms / frame")
        print(f"Mean Total Pipeline:   {mean_tot:.2f} ms / frame")
        print(f"Mean End-to-End FPS:   {mean_fps:.2f} FPS")
        if out_video_path:
            print(f"Saved Result Video:    {out_video_path}")
        print("=" * 65)

        return {
            "video_path": str(video_path),
            "output_path": str(out_video_path) if out_video_path else None,
            "frames_processed": frame_idx,
            "plates_detected": total_plates_detected,
            "mean_det_ms": mean_det,
            "mean_ocr_ms": mean_ocr,
            "mean_total_ms": mean_tot,
            "mean_fps": mean_fps,
        }

    def run_camera(
        self,
        camera_id: int = 0,
        show_debug: bool = True,
        no_gui: bool = False,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs end-to-end pipeline on live camera feed."""
        print(f"\n[CAMERA MODE] Opening camera index: {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"[CAMERA ERROR] Cannot open camera device {camera_id}. Verify camera connection or permissions.")
            return {"status": "NOT_ACCESSIBLE", "camera_id": camera_id}

        cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAMERA MODE] Opened successfully ({cw}x{ch}).")
        print("Controls: 'q' = Quit | 's' = Save Frame | 'c' = Save Crops")

        # Warmup
        self.warmup(10)

        frame_idx = 0
        all_total_ms = []
        plates_count = 0
        t_last_fps_log = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAMERA WARNING] Failed to read frame from camera.")
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                print(f"[CAMERA MODE] Reached max frames ({max_frames}). Stopping.")
                break

            annotated, plates, timings = self.process_frame(
                frame,
                frame_idx=frame_idx,
                source_name=f"cam{camera_id}",
                show_debug=show_debug,
            )

            all_total_ms.append(timings["total_ms"])
            plates_count += len(plates)

            # Periodic FPS print (every 2.5 seconds, no spam)
            now = time.perf_counter()
            if now - t_last_fps_log >= 2.5:
                window_ms = all_total_ms[-30:] if len(all_total_ms) >= 30 else all_total_ms
                curr_fps = 1000.0 / (sum(window_ms) / len(window_ms)) if window_ms else 0.0
                print(f"  [Live Camera] Frame {frame_idx:05d} | Active Plates: {len(plates)} | "
                      f"Det: {timings['detector_ms']:.1f}ms | OCR: {timings['ocr_ms']:.1f}ms | "
                      f"FPS: {curr_fps:.1f}")
                t_last_fps_log = now

            if not no_gui:
                try:
                    cv2.imshow("PARSeq Laptop Pipeline - Camera Stream", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[CAMERA MODE] User quit.")
                        break
                    elif key == ord('s'):
                        save_name = self.img_out_dir / f"cam_snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(str(save_name), annotated)
                        print(f"  -> Saved snapshot: {save_name}")
                    elif key == ord('c'):
                        print(f"  -> Manual crop save triggered ({len(plates)} plates).")
                        for idx, p in enumerate(plates, 1):
                            crop, _ = crop_with_padding(frame, p["bbox"], pad_ratio=self.crop_pad)
                            c_name = self.raw_crops_dir / f"manual_{frame_idx}_{idx}_{p['plate_text']}.jpg"
                            cv2.imwrite(str(c_name), crop)
                except Exception as e:
                    # In headless environments cv2.imshow raises error
                    if frame_idx == 1:
                        print(f"[GUI Notice] cv2.imshow error ({e}). Continuing in headless mode.")
            else:
                # If running headless with max_frames, yield brief sleep
                time.sleep(0.005)

        cap.release()
        if not no_gui:
            cv2.destroyAllWindows()

        mean_tot = sum(all_total_ms) / len(all_total_ms) if all_total_ms else 0.0
        mean_fps = 1000.0 / mean_tot if mean_tot > 0 else 0.0

        return {
            "status": "TESTED",
            "camera_id": camera_id,
            "frames_processed": frame_idx,
            "plates_detected": plates_count,
            "mean_total_ms": mean_tot,
            "mean_fps": mean_fps,
        }


def main():
    parser = argparse.ArgumentParser(description="Real Laptop License Plate Pipeline - YOLO11n + PARSeq-Tiny VNLP")
    parser.add_argument("--source", type=str, required=True, choices=["image", "video", "camera"],
                        help="Input source type: 'image', 'video', or 'camera'")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input image or video file (required for image/video)")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--detector", type=str, default="weights/best_license_plate_yolo11n_416.onnx",
                        help="Path to YOLO11n ONNX detector checkpoint")
    parser.add_argument("--ocr", type=str, default="parseq", choices=["parseq", "lprnet"],
                        help="OCR engine: 'parseq' (default) or 'lprnet' (A/B reference)")
    parser.add_argument("--ocr-checkpoint", type=str, default="weights/ocr/teachers/parseq_tiny_vn/best.pth",
                        help="Path to PARSeq-Tiny VNLP PyTorch weights (.pth)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Inference device for PARSeq: 'cuda' or 'cpu'")
    parser.add_argument("--use-amp", action="store_true",
                        help="Enable AMP mixed precision for PARSeq inference")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="YOLO detector input resolution (default: 416)")
    parser.add_argument("--det-conf", type=float, default=0.50,
                        help="YOLO detector confidence threshold (default: 0.50)")
    parser.add_argument("--det-iou", type=float, default=0.45,
                        help="YOLO detector NMS IoU threshold (default: 0.45)")
    parser.add_argument("--ocr-conf", type=float, default=0.0,
                        help="Minimum sequence confidence threshold to accept plate (default: 0.0)")
    parser.add_argument("--crop-pad", type=float, default=0.05,
                        help="Proportional padding margin around detected bounding box (default: 0.05)")
    parser.add_argument("--stabilize", dest="stabilize", action="store_true", default=None,
                        help="Enable temporal stabilization (default ON for video/camera, OFF for image)")
    parser.add_argument("--no-stabilize", dest="stabilize", action="store_false",
                        help="Disable temporal stabilization")
    parser.add_argument("--save", action="store_true",
                        help="Save output annotated image or video to outputs/parseq_laptop/")
    parser.add_argument("--show-debug", action="store_true",
                        help="Render latency and FPS dashboard overlay on frames")
    parser.add_argument("--save-crops", action="store_true",
                        help="Save raw detector crops and preprocessed OCR crops to disk")
    parser.add_argument("--log-low-confidence", action="store_true",
                        help="Log plates with OCR confidence below threshold to CSV and disk")
    parser.add_argument("--low-conf-thresh", type=float, default=0.80,
                        help="Low confidence threshold for review logging (default: 0.80)")
    parser.add_argument("--collect", action="store_true",
                        help="Save hard / low-confidence crops to dataset_staging/real_world_review/")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run without displaying cv2.imshow windows (ideal for headless / automation)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit processing to N frames (useful for video/camera benchmarks)")
    parser.add_argument("--output-dir", type=str, default="outputs/parseq_laptop",
                        help="Base output directory")

    args = parser.parse_args()

    # Determine default stabilization setting
    if args.stabilize is None:
        stabilize = True if args.source in ("video", "camera") else False
    else:
        stabilize = args.stabilize

    pipeline = RealLaptopPipeline(
        detector_path=args.detector,
        ocr_model=args.ocr,
        ocr_checkpoint=args.ocr_checkpoint,
        device=args.device,
        use_amp=args.use_amp,
        imgsz=args.imgsz,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        ocr_conf_thresh=args.ocr_conf,
        crop_pad=args.crop_pad,
        enable_stabilizer=stabilize,
        save_crops=args.save_crops,
        log_low_confidence=args.log_low_confidence,
        low_conf_thresh=args.low_conf_thresh,
        collect=args.collect,
        output_dir=Path(args.output_dir),
    )

    if args.source == "image":
        if not args.input:
            parser.error("--input is required when --source is 'image'")
        input_path = Path(args.input)
        pipeline.run_image(
            image_path=input_path,
            save=args.save,
            show_debug=args.show_debug,
            no_gui=args.no_gui,
        )

    elif args.source == "video":
        if not args.input:
            parser.error("--input is required when --source is 'video'")
        input_path = Path(args.input)
        pipeline.run_video(
            video_path=input_path,
            save=args.save,
            show_debug=args.show_debug,
            no_gui=args.no_gui,
            max_frames=args.max_frames,
        )

    elif args.source == "camera":
        pipeline.run_camera(
            camera_id=args.camera_id,
            show_debug=args.show_debug,
            no_gui=args.no_gui,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
