"""
Raspberry Pi 5 End-to-End Vietnamese License Plate Recognition Pipeline.
Combines YOLO11n-416 ONNX Detector + LPRNet-Lite ONNX OCR using pure OpenCV DNN and NumPy.

Usage:
    python3 scripts/pi_recognize_image.py \
        --detector models/best_license_plate_yolo11n_416.onnx \
        --ocr models/best_lprnet_vn.onnx \
        --image test_images/example.jpg \
        --imgsz 416 --conf 0.5 --iou 0.45 --crop-pad 0.05
"""

import argparse
import ast
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.pi_yolo_detector import PlateDetector
from ocr.pi_lprnet_ocr import PlateOCR, normalize_plate_text


def get_process_memory_mb() -> float:
    """Reads resident set size (RSS) memory in MB on Linux."""
    try:
        with open("/proc/self/statm", "r") as f:
            fields = f.read().split()
            rss_pages = int(fields[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (rss_pages * page_size) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def box_iou(box1: List[int], box2: List[int]) -> float:
    """Calculates Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / float(union) if union > 0 else 0.0


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


def draw_plate_prediction(
    annotated_img: np.ndarray,
    bbox: List[int],
    plate_text: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0)
):
    """Draws bounding box and recognized license plate text banner."""
    x1, y1, x2, y2 = bbox
    ih, iw = annotated_img.shape[:2]

    # Draw box
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

    # Banner text
    label = f"{plate_text} | det {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    bg_y1 = max(0, y1 - th - 8)
    bg_y2 = y1
    bg_x1 = x1
    bg_x2 = min(iw, x1 + tw + 8)

    cv2.rectangle(annotated_img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 180, 0), -1)
    cv2.putText(
        annotated_img,
        label,
        (bg_x1 + 4, bg_y2 - 4),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


class EndToEndPipeline:
    """
    Complete License Plate Detection + OCR Pipeline.
    Loads both networks once and processes frames or image batches.
    """
    def __init__(
        self,
        detector_path: str,
        ocr_path: str,
        imgsz: int = 416,
        conf: float = 0.5,
        iou: float = 0.45,
        crop_pad: float = 0.05,
        min_crop_w: int = 20,
        min_crop_h: int = 12,
        threads: Optional[int] = None,
    ):
        self.crop_pad = crop_pad
        self.min_crop_w = min_crop_w
        self.min_crop_h = min_crop_h

        print("Initializing End-to-End Pipeline on CPU...")
        self.detector = PlateDetector(
            model_path=detector_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            num_threads=threads
        )
        self.ocr = PlateOCR(model_path=ocr_path)
        print("Both networks loaded successfully.")

    def process_image(
        self,
        img_bgr: np.ndarray,
        image_name: str = "image",
        save_debug_crops: bool = True,
        crops_out_dir: Optional[Path] = None,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, float]]:
        """
        Executes complete end-to-end detection and recognition on an image.
        Returns:
            results: List of detections with recognized text and layout
            annotated_img: Image with drawn boxes and labels
            timings: Breakdown of execution time in milliseconds
        """
        t_start = time.perf_counter()

        # 1. Detector
        detections, det_timings = self.detector.detect(img_bgr)

        # 2. Crop & OCR each detection
        t_crop_start = time.perf_counter()
        annotated_img = img_bgr.copy()
        results = []

        total_ocr_prep_ms = 0.0
        total_ocr_fwd_ms = 0.0
        total_ocr_dec_ms = 0.0

        for idx, det in enumerate(detections, 1):
            bbox = det["bbox"]
            conf = det["confidence"]
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1

            # Skip small crops
            if bw < self.min_crop_w or bh < self.min_crop_h:
                print(f"  [Skip] Detection {idx} too small ({bw}x{bh} < {self.min_crop_w}x{self.min_crop_h})")
                continue

            # Padded crop
            crop, padded_box = crop_with_padding(img_bgr, bbox, pad_ratio=self.crop_pad)
            if crop.size == 0:
                continue

            # OCR Recognition
            ocr_res = self.ocr.recognize(crop, layout="auto")
            plate_text = ocr_res["text"]
            layout = ocr_res["layout"]
            ocr_timings = ocr_res["timings"]

            total_ocr_prep_ms += ocr_timings["prep_ms"]
            total_ocr_fwd_ms += ocr_timings["forward_ms"]
            total_ocr_dec_ms += ocr_timings["decode_ms"]

            # Save debug crops if requested
            if save_debug_crops and crops_out_dir:
                crop_path = crops_out_dir / f"{image_name}_plate_{idx:02d}.jpg"
                cv2.imwrite(str(crop_path), crop)
                if ocr_res["rectified_image"] is not None:
                    rect_path = crops_out_dir / f"{image_name}_plate_{idx:02d}_rectified.jpg"
                    cv2.imwrite(str(rect_path), ocr_res["rectified_image"])

            # Draw
            draw_plate_prediction(annotated_img, bbox, plate_text, conf)

            results.append({
                "bbox": bbox,
                "confidence": conf,
                "layout": layout,
                "plate_text": plate_text,
                "ocr_timings": ocr_timings,
            })

        t_end = time.perf_counter()
        total_pipeline_ms = (t_end - t_start) * 1000.0
        crop_layout_ms = (t_end - t_crop_start) * 1000.0 - (total_ocr_prep_ms + total_ocr_fwd_ms + total_ocr_dec_ms)

        timings = {
            "det_prep_ms": det_timings["prep_ms"],
            "det_fwd_ms": det_timings["forward_ms"],
            "det_post_ms": det_timings["post_ms"],
            "crop_layout_ms": max(0.0, crop_layout_ms),
            "ocr_prep_ms": total_ocr_prep_ms,
            "ocr_fwd_ms": total_ocr_fwd_ms,
            "ocr_dec_ms": total_ocr_dec_ms,
            "total_ocr_ms": total_ocr_prep_ms + total_ocr_fwd_ms + total_ocr_dec_ms,
            "total_ms": total_pipeline_ms,
            "e2e_fps": 1000.0 / total_pipeline_ms if total_pipeline_ms > 0 else 0.0,
        }

        return results, annotated_img, timings


def run_benchmark_1plate(pipeline: EndToEndPipeline, img_path: str, warmup: int = 5, runs: int = 30):
    """Benchmarks end-to-end latency on a typical 1-plate image."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load benchmark image: {img_path}")

    print(f"\n--- Running End-to-End Benchmark (1 Plate, {runs} runs, {warmup} warmup) ---")
    for _ in range(warmup):
        pipeline.process_image(img, save_debug_crops=False)

    timings_list = []
    for _ in range(runs):
        _, _, t = pipeline.process_image(img, save_debug_crops=False)
        timings_list.append(t)

    avg_det_prep = np.mean([t["det_prep_ms"] for t in timings_list])
    avg_det_fwd = np.mean([t["det_fwd_ms"] for t in timings_list])
    avg_det_post = np.mean([t["det_post_ms"] for t in timings_list])
    avg_crop = np.mean([t["crop_layout_ms"] for t in timings_list])
    avg_ocr_prep = np.mean([t["ocr_prep_ms"] for t in timings_list])
    avg_ocr_fwd = np.mean([t["ocr_fwd_ms"] for t in timings_list])
    avg_ocr_dec = np.mean([t["ocr_dec_ms"] for t in timings_list])
    avg_total = np.mean([t["total_ms"] for t in timings_list])
    e2e_fps = 1000.0 / avg_total if avg_total > 0 else 0.0

    print("=" * 60)
    print("PERFORMANCE — 1 PLATE BENCHMARK RESULTS:")
    print(f"  detector preprocess: {avg_det_prep:.2f} ms")
    print(f"  detector forward:    {avg_det_fwd:.2f} ms")
    print(f"  detector postprocess: {avg_det_post:.2f} ms")
    print(f"  crop/layout:         {avg_crop:.2f} ms")
    print(f"  OCR preprocess:      {avg_ocr_prep:.2f} ms")
    print(f"  OCR forward:         {avg_ocr_fwd:.2f} ms")
    print(f"  OCR decode:          {avg_ocr_dec:.2f} ms")
    print(f"  total:               {avg_total:.2f} ms")
    print(f"  E2E FPS:             {e2e_fps:.2f} FPS")
    print("=" * 60)


def run_gt_evaluation(pipeline: EndToEndPipeline, manifest_csv: str, img_dir: Optional[str] = None):
    """Evaluates End-to-End metrics (Detection Recall, OCR Given Det, E2E Exact)."""
    print(f"\n--- Running Ground-Truth End-to-End Evaluation on {manifest_csv} ---")
    records = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)

    total_gt = len(records)
    detected_gt = 0
    e2e_exact = 0

    base_dir = Path(img_dir) if img_dir else Path(manifest_csv).parent

    for r in records:
        img_p = r.get("image_path", r.get("filename", ""))
        full_p = base_dir / img_p if not os.path.isabs(img_p) else Path(img_p)
        if not full_p.exists() and (base_dir / "images" / img_p).exists():
            full_p = base_dir / "images" / img_p

        subset = r.get("subset", r.get("layout", "unknown"))
        gt_text = normalize_plate_text(r.get("plate_text", ""))

        # Parse GT bbox
        gt_box = None
        if "x1" in r and r["x1"]:
            gt_box = [int(float(r["x1"])), int(float(r["y1"])), int(float(r["x2"])), int(float(r["y2"]))]
        elif "bbox" in r and r["bbox"]:
            try:
                gt_box = ast.literal_eval(r["bbox"])
            except Exception:
                gt_box = None

        img = cv2.imread(str(full_p))
        if img is None:
            print(f"Warning: could not load image {full_p}")
            continue

        results, _, _ = pipeline.process_image(img, save_debug_crops=False)

        # Match detections with GT
        is_detected = False
        is_e2e_exact = False

        if gt_box is not None:
            best_iou = 0.0
            matched_ocr = None
            for det in results:
                iou = box_iou(det["bbox"], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    matched_ocr = det["plate_text"]

            if best_iou >= 0.5:
                is_detected = True
                detected_gt += 1
                if matched_ocr and normalize_plate_text(matched_ocr) == gt_text:
                    is_e2e_exact = True
                    e2e_exact += 1
        else:
            if len(results) > 0:
                is_detected = True
                detected_gt += 1
                for det in results:
                    if normalize_plate_text(det["plate_text"]) == gt_text:
                        is_e2e_exact = True
                        e2e_exact += 1
                        break

    det_recall = (detected_gt / total_gt) * 100.0 if total_gt > 0 else 0.0
    ocr_given_det = (e2e_exact / detected_gt) * 100.0 if detected_gt > 0 else 0.0
    e2e_acc = (e2e_exact / total_gt) * 100.0 if total_gt > 0 else 0.0

    print("=" * 60)
    print("END-TO-END ACCURACY RESULTS:")
    print(f"  Total Ground Truth:               {total_gt}")
    print(f"  Detection Recall:                 {detected_gt}/{total_gt} ({det_recall:.2f}%)")
    print(f"  OCR Exact Given Detection:        {e2e_exact}/{detected_gt} ({ocr_given_det:.2f}%)")
    print(f"  End-to-End Exact Plate Accuracy:  {e2e_exact}/{total_gt} ({e2e_acc:.2f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Plate Detection + OCR Pipeline")
    parser.add_argument("--detector", type=str, default="models/best_license_plate_yolo11n_416.onnx")
    parser.add_argument("--ocr", type=str, default="models/best_lprnet_vn.onnx")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--crop-pad", type=float, default=0.05)
    parser.add_argument("--output-dir", type=str, default="outputs/e2e")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    out_base = Path(args.output_dir)
    images_out = out_base / "images"
    crops_out = out_base / "plate_crops"
    images_out.mkdir(parents=True, exist_ok=True)
    crops_out.mkdir(parents=True, exist_ok=True)

    pipeline = EndToEndPipeline(
        detector_path=args.detector,
        ocr_path=args.ocr,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        crop_pad=args.crop_pad,
    )

    rss_mb = get_process_memory_mb()
    print(f"Process RSS RAM: {rss_mb:.1f} MB" if rss_mb > 0 else "RAM: N/A")

    if args.csv:
        run_gt_evaluation(pipeline, args.csv)
    elif args.image:
        img_p = Path(args.image)
        if not img_p.exists():
            print(f"Error: {img_p} not found.")
            sys.exit(1)

        if args.benchmark:
            run_benchmark_1plate(pipeline, str(img_p), warmup=args.warmup, runs=args.runs)
        else:
            img = cv2.imread(str(img_p))
            results, annotated, timings = pipeline.process_image(
                img, image_name=img_p.stem, save_debug_crops=True, crops_out_dir=crops_out
            )
            out_img_path = images_out / f"{img_p.stem}_recognized.jpg"
            cv2.imwrite(str(out_img_path), annotated)

            print(f"\nImage:       {img_p.name}")
            print(f"Detections:  {len(results)}")
            for i, r in enumerate(results, 1):
                b = r["bbox"]
                print(f"  [{i}] Box: [{b[0]}, {b[1]}, {b[2]}, {b[3]}] | Conf: {r['confidence']:.2f} | Layout: {r['layout']} | Text: {r['plate_text']}")
            print(f"Detector ms: {timings['det_fwd_ms']:.2f}")
            print(f"OCR ms:      {timings['total_ocr_ms']:.2f}")
            print(f"Total ms:    {timings['total_ms']:.2f}")
            print(f"Saved:       {out_img_path}")
    elif args.image_dir:
        p_dir = Path(args.image_dir)
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        imgs = sorted([p for p in p_dir.iterdir() if p.suffix.lower() in valid_exts])
        print(f"\nProcessing {len(imgs)} images from {p_dir}...")
        for img_p in imgs:
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            results, annotated, timings = pipeline.process_image(
                img, image_name=img_p.stem, save_debug_crops=True, crops_out_dir=crops_out
            )
            out_img_path = images_out / f"{img_p.stem}_recognized.jpg"
            cv2.imwrite(str(out_img_path), annotated)
            print("-" * 50)
            print(f"Filename:        {img_p.name}")
            print(f"Detections:      {len(results)}")
            for i, r in enumerate(results, 1):
                b = r["bbox"]
                print(f"  [{i}] bbox:           [{b[0]}, {b[1]}, {b[2]}, {b[3]}]")
                print(f"      det confidence: {r['confidence']:.2f}")
                print(f"      layout:         {r['layout']}")
                print(f"      OCR prediction: {r['plate_text']}")
            print(f"Detector ms:     {timings['det_fwd_ms']:.2f}")
            print(f"OCR ms:          {timings['total_ocr_ms']:.2f}")
            print(f"Total ms:        {timings['total_ms']:.2f}")
            print(f"Saved:           {out_img_path.name}")


if __name__ == "__main__":
    main()
