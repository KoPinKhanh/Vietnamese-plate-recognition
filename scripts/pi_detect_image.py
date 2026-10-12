"""
Raspberry Pi 5 License Plate Detector using OpenCV DNN.
Supports multiple input resolutions (640, 512, 416, 320) dynamically.
Performs letterboxing, inference, YOLO11 output decoding, NMS,
coordinate mapping back to original image, visualization, and plate cropping.
Zero dependencies on PyTorch, Ultralytics, or ONNXRuntime.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resizes and pads image to new_shape while strictly maintaining aspect ratio.
    Returns:
        padded_img, scale, pad_x, pad_y
    """
    shape = img.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad_w = int(round(shape[1] * scale))
    new_unpad_h = int(round(shape[0] * scale))

    dw = new_shape[1] - new_unpad_w  # total horizontal padding
    dh = new_shape[0] - new_unpad_h  # total vertical padding

    pad_x = dw / 2.0
    pad_y = dh / 2.0

    if shape[::-1] != (new_unpad_w, new_unpad_h):
        img_resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img

    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))

    padded_img = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded_img, scale, pad_x, pad_y


def load_network(model_path: str, num_threads: int = None):
    """Loads ONNX model using OpenCV DNN with CPU target."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    net = cv2.dnn.readNetFromONNX(str(model_path))

    # Explicitly set backend and target if supported
    if hasattr(cv2.dnn, "DNN_BACKEND_OPENCV"):
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    if hasattr(cv2.dnn, "DNN_TARGET_CPU"):
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    if num_threads is not None and num_threads > 0:
        cv2.setNumThreads(num_threads)

    return net


def infer_imgsz(model_path: str, user_imgsz: int = None) -> int:
    """Infers input size from user argument or model filename."""
    if user_imgsz is not None and user_imgsz > 0:
        return user_imgsz
    m_str = str(model_path)
    for sz in [640, 512, 416, 320]:
        if f"_{sz}" in m_str:
            return sz
    return 640


def postprocess_yolo11(output, conf_thresh, iou_thresh, scale, pad_x, pad_y, orig_w, orig_h):
    """
    Decodes YOLO11 single-class output tensor:
    Output shape: [1, 5, N] or [1, N, 5] (where columns are cx, cy, w, h, score).
    Returns list of dicts with mapped bounding boxes [x1, y1, x2, y2], scores, and class_ids.
    """
    preds = np.squeeze(output)

    # Transpose to [N, 5] if format is [5, N]
    if preds.ndim == 2:
        if preds.shape[0] == 5 and preds.shape[1] != 5:
            preds = preds.T
    else:
        raise ValueError(f"Unexpected output tensor dimension: {output.shape}")

    num_predictions = preds.shape[0]
    if num_predictions == 0:
        return []

    # Filter by confidence threshold
    scores = preds[:, 4]
    mask = scores >= conf_thresh
    filtered_preds = preds[mask]
    filtered_scores = scores[mask]

    if len(filtered_preds) == 0:
        return []

    boxes = []
    confidences = []

    for i in range(len(filtered_preds)):
        cx, cy, w, h = filtered_preds[i, :4]
        score = float(filtered_scores[i])

        x = cx - w / 2.0
        y = cy - h / 2.0

        boxes.append([int(round(x)), int(round(y)), int(round(w)), int(round(h))])
        confidences.append(score)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        score_threshold=conf_thresh,
        nms_threshold=iou_thresh,
    )

    if len(indices) == 0:
        return []

    indices = np.array(indices).flatten()
    detections = []

    for idx in indices:
        bx, by, bw, bh = boxes[idx]
        conf = confidences[idx]

        orig_x1 = (bx - pad_x) / scale
        orig_y1 = (by - pad_y) / scale
        orig_x2 = (bx + bw - pad_x) / scale
        orig_y2 = (by + bh - pad_y) / scale

        orig_x1 = max(0, min(orig_w, orig_x1))
        orig_y1 = max(0, min(orig_h, orig_y1))
        orig_x2 = max(0, min(orig_w, orig_x2))
        orig_y2 = max(0, min(orig_h, orig_y2))

        if orig_x2 > orig_x1 and orig_y2 > orig_y1:
            detections.append({
                "box": [int(round(orig_x1)), int(round(orig_y1)), int(round(orig_x2)), int(round(orig_y2))],
                "conf": conf,
                "label": "license_plate",
            })

    return detections


def detect_image(
    image_path: str,
    net,
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.45,
    imgsz: int = 640,
):
    """
    Executes full pipeline for a single image:
    Returns (detections, annotated_image, crops, timing_dict)
    """
    input_size = (imgsz, imgsz)
    t_start = time.time()

    # 1. Image Read
    t_read_start = time.time()
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image at: {image_path}")
    orig_h, orig_w = img.shape[:2]
    read_ms = (time.time() - t_read_start) * 1000.0

    # 2. Letterbox & Preprocess
    t_prep_start = time.time()
    padded_img, scale, pad_x, pad_y = letterbox(img, new_shape=input_size)
    blob = cv2.dnn.blobFromImage(
        padded_img,
        scalefactor=1.0 / 255.0,
        size=input_size,
        swapRB=True,
        crop=False,
    )
    prep_ms = (time.time() - t_prep_start) * 1000.0

    # 3. Forward Pass
    net.setInput(blob)
    t_inf_start = time.time()
    output = net.forward()
    inf_ms = (time.time() - t_inf_start) * 1000.0

    # 4. Postprocess & NMS
    t_post_start = time.time()
    detections = postprocess_yolo11(
        output,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_w=orig_w,
        orig_h=orig_h,
    )
    post_ms = (time.time() - t_post_start) * 1000.0

    # 5. Drawing & Plate Cropping
    t_draw_start = time.time()
    annotated = img.copy()
    crops = []

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]

        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label_text = f"license_plate {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        bg_y1 = max(0, y1 - th - 8)
        bg_y2 = y1
        bg_x1 = x1
        bg_x2 = min(orig_w, x1 + tw + 8)

        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 200, 0), -1)
        cv2.putText(
            annotated,
            label_text,
            (bg_x1 + 4, bg_y2 - 4),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    draw_ms = (time.time() - t_draw_start) * 1000.0
    total_ms = (time.time() - t_start) * 1000.0

    timing = {
        "read_ms": read_ms,
        "prep_ms": prep_ms,
        "inf_ms": inf_ms,
        "post_ms": post_ms,
        "draw_ms": draw_ms,
        "total_ms": total_ms,
    }

    return detections, annotated, crops, timing


def parse_args():
    parser = argparse.ArgumentParser(description="License Plate Detector using OpenCV DNN.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_license_plate_yolo11n_640.onnx",
        help="Path to ONNX model file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image resolution (e.g., 640, 512, 416, 320). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image file.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to directory of images to process.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run end-to-end benchmark on the first image.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of benchmark iterations (default: 30).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Set cv2 number of threads.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    imgsz = infer_imgsz(args.model, args.imgsz)

    det_out_dir = Path(args.output_dir) / "detections"
    crops_out_dir = Path(args.output_dir) / "plate_crops"
    det_out_dir.mkdir(parents=True, exist_ok=True)
    crops_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("      OPENCV DNN VIETNAMESE LICENSE PLATE DETECTOR      ")
    print("=" * 60)
    print(f"OpenCV version   : {cv2.__version__}")
    print(f"OpenCV threads   : {cv2.getNumThreads()}")
    print(f"Model path       : {args.model}")
    print(f"Input size       : {imgsz}x{imgsz}")
    print(f"Confidence thresh: {args.conf}")
    print(f"NMS IoU thresh   : {args.iou}")

    net = load_network(args.model, num_threads=args.threads)
    if args.threads is not None:
        print(f"Configured threads: {cv2.getNumThreads()}")

    image_paths = []
    if args.image:
        p = Path(args.image)
        if not p.exists():
            print(f"Error: Image '{p}' does not exist.")
            sys.exit(1)
        image_paths.append(p)
    elif args.image_dir:
        p_dir = Path(args.image_dir)
        if not p_dir.exists():
            print(f"Error: Directory '{p_dir}' does not exist.")
            sys.exit(1)
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted([p for p in p_dir.iterdir() if p.suffix.lower() in valid_exts])
    else:
        print("Error: Specify either --image or --image-dir.")
        sys.exit(1)

    if not image_paths:
        print("No valid images found to process.")
        sys.exit(1)

    print(f"\nProcessing {len(image_paths)} image(s)...")
    print("-" * 60)

    for img_p in image_paths:
        detections, annotated, crops, timing = detect_image(
            str(img_p),
            net=net,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            imgsz=imgsz,
        )

        out_detected_path = det_out_dir / f"{img_p.stem}_detected.jpg"
        cv2.imwrite(str(out_detected_path), annotated)

        for idx, crop in enumerate(crops, 1):
            crop_path = crops_out_dir / f"{img_p.stem}_plate_{idx:02d}.jpg"
            cv2.imwrite(str(crop_path), crop)

        num_det = len(detections)
        print(f"Image       : {img_p.name}")
        print(f"Detections  : {num_det}")
        if num_det > 0:
            for i, d in enumerate(detections, 1):
                b = d["box"]
                print(f"  [{i}] Conf: {d['conf']:.4f} | Box: [x1={b[0]}, y1={b[1]}, x2={b[2]}, y2={b[3]}]")
        print(f"Inference ms: {timing['inf_ms']:.2f}")
        print(f"Total ms    : {timing['total_ms']:.2f}")
        print("-" * 60)

    if args.benchmark and image_paths:
        bench_img = str(image_paths[0])
        print(f"\nRunning End-to-End Benchmark on: {Path(bench_img).name}")
        print(f"Warmup iterations: {args.warmup}")
        for _ in range(args.warmup):
            detect_image(bench_img, net=net, conf_thresh=args.conf, iou_thresh=args.iou, imgsz=imgsz)

        print(f"Benchmark iterations: {args.runs}")
        timings = {"prep": [], "inf": [], "post": [], "draw": [], "total": []}

        for _ in range(args.runs):
            _, _, _, t = detect_image(bench_img, net=net, conf_thresh=args.conf, iou_thresh=args.iou, imgsz=imgsz)
            timings["prep"].append(t["prep_ms"])
            timings["inf"].append(t["inf_ms"])
            timings["post"].append(t["post_ms"])
            timings["draw"].append(t["draw_ms"])
            timings["total"].append(t["total_ms"])

        avg_prep = float(np.mean(timings["prep"]))
        avg_inf = float(np.mean(timings["inf"]))
        avg_post = float(np.mean(timings["post"]))
        avg_draw = float(np.mean(timings["draw"]))
        avg_total = float(np.mean(timings["total"]))

        min_inf = float(np.min(timings["inf"]))
        max_inf = float(np.max(timings["inf"]))
        min_total = float(np.min(timings["total"]))
        max_total = float(np.max(timings["total"]))

        raw_inf_fps = 1000.0 / avg_inf if avg_inf > 0 else 0.0
        e2e_fps = 1000.0 / avg_total if avg_total > 0 else 0.0

        print("=" * 60)
        print(f"END-TO-END BENCHMARK RESULTS ({imgsz}x{imgsz}):")
        print(f"Preprocess ms    : {avg_prep:.2f} ms")
        print(f"Inference ms     : {avg_inf:.2f} ms (Min: {min_inf:.2f}, Max: {max_inf:.2f})")
        print(f"Postprocess ms   : {avg_post:.2f} ms")
        print(f"Drawing/Crop ms  : {avg_draw:.2f} ms")
        print(f"Total ms         : {avg_total:.2f} ms (Min: {min_total:.2f}, Max: {max_total:.2f})")
        print(f"Raw inference FPS: {raw_inf_fps:.2f}")
        print(f"End-to-end FPS   : {e2e_fps:.2f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
