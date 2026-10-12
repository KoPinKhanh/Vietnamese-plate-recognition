"""
Modular YOLO11 License Plate Detector for Raspberry Pi 5.
Runs CPU inference using OpenCV DNN.
Zero dependencies on PyTorch, Ultralytics, or ONNX Runtime.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (416, 416), color: Tuple[int, int, int] = (114, 114, 114)):
    """Resizes and pads image to new_shape preserving aspect ratio."""
    shape = img.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    scale = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad_w = int(round(shape[1] * scale))
    new_unpad_h = int(round(shape[0] * scale))

    dw = new_shape[1] - new_unpad_w
    dh = new_shape[0] - new_unpad_h

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


def postprocess_yolo11(output: np.ndarray, conf_thresh: float, iou_thresh: float, scale: float, pad_x: float, pad_y: float, orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
    """Decodes YOLO11 output tensor [1, 5, N] or [1, N, 5], applies NMS, and rescales boxes."""
    preds = np.squeeze(output)
    if preds.ndim == 2:
        if preds.shape[0] == 5 and preds.shape[1] != 5:
            preds = preds.T
    else:
        raise ValueError(f"Unexpected output tensor dimension: {output.shape}")

    num_predictions = preds.shape[0]
    if num_predictions == 0:
        return []

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
                "bbox": [int(round(orig_x1)), int(round(orig_y1)), int(round(orig_x2)), int(round(orig_y2))],
                "confidence": float(conf),
            })
    return detections


class PlateDetector:
    """
    OpenCV DNN License Plate Detector.
    Encapsulates YOLO11 model loading and forward execution on CPU.
    """
    def __init__(
        self,
        model_path: str,
        imgsz: int = 416,
        conf: float = 0.5,
        iou: float = 0.45,
        num_threads: Optional[int] = None,
    ):
        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Detector model not found: {self.model_path}")

        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        if hasattr(cv2.dnn, "DNN_BACKEND_OPENCV"):
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        if hasattr(cv2.dnn, "DNN_TARGET_CPU"):
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        if num_threads is not None and num_threads > 0:
            cv2.setNumThreads(num_threads)

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Runs plate detection on a frame (BGR).
        Returns:
            detections: List of {"bbox": [x1, y1, x2, y2], "confidence": float}
            timings: {"prep_ms": float, "forward_ms": float, "post_ms": float, "total_ms": float}
        """
        import time
        orig_h, orig_w = frame.shape[:2]
        input_size = (self.imgsz, self.imgsz)

        t0 = time.perf_counter()
        padded_img, scale, pad_x, pad_y = letterbox(frame, new_shape=input_size)
        blob = cv2.dnn.blobFromImage(
            padded_img,
            scalefactor=1.0 / 255.0,
            size=input_size,
            swapRB=True,
            crop=False
        )
        t_pre = time.perf_counter()

        self.net.setInput(blob)
        output = self.net.forward()
        t_fwd = time.perf_counter()

        detections = postprocess_yolo11(
            output,
            conf_thresh=self.conf,
            iou_thresh=self.iou,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            orig_w=orig_w,
            orig_h=orig_h
        )
        t_post = time.perf_counter()

        timings = {
            "prep_ms": (t_pre - t0) * 1000.0,
            "forward_ms": (t_fwd - t_pre) * 1000.0,
            "post_ms": (t_post - t_fwd) * 1000.0,
            "total_ms": (t_post - t0) * 1000.0,
        }
        return detections, timings
