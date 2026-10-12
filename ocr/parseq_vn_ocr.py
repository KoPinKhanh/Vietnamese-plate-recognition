"""
PARSeq-Tiny VNLP License Plate OCR Wrapper for Laptop Pipeline.
Loads fine-tuned PARSeq model once at startup, executes inference under torch.inference_mode(),
and integrates Vietnamese license plate layout rectification (one_row vs two_rows).

Model: weights/ocr/teachers/parseq_tiny_vn/best.pth
Architecture: PARSeq-Tiny (Baudm/parseq) fine-tuned on VNLP.
"""

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

from ocr.preprocess import rectify_two_row_plate
from ocr.plate_text_utils import normalize_plate_text


class ParseqVNRecognizer:
    """
    Production-grade Laptop Wrapper for PARSeq-Tiny VNLP OCR.
    Loads checkpoint strictly ONCE at initialization and runs high-accuracy
    inference on Vietnamese license plate crops.
    """

    def __init__(
        self,
        checkpoint: Union[str, Path] = "weights/ocr/teachers/parseq_tiny_vn/best.pth",
        device: Optional[str] = None,
        use_amp: bool = False,
        target_size: Tuple[int, int] = (128, 32),
    ):
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = PROJECT_ROOT / self.checkpoint_path

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PARSeq checkpoint not found: {self.checkpoint_path}")

        # Device selection: CUDA preferred if available
        if device is None or device == "auto":
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device
        self.device = torch.device(self.device_str)

        self.use_amp = use_amp and (self.device.type == "cuda")
        self.target_size = target_size  # (width=128, height=32)

        print(f"[PARSeqVNRecognizer] Initializing on device: {self.device} (AMP: {self.use_amp})...")
        t0 = time.perf_counter()

        # Load PARSeq-Tiny architecture from torch.hub cache
        self.model = torch.hub.load("baudm/parseq", "parseq_tiny", pretrained=False, trust_repo=True)

        # Load weights
        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=True)

        # Eval mode and transfer to target device
        self.model = self.model.to(self.device).eval()

        t_load = (time.perf_counter() - t0) * 1000.0
        print(f"[PARSeqVNRecognizer] Loaded checkpoint ({self.checkpoint_path.name}) in {t_load:.1f} ms. Model is in eval mode.")

    @staticmethod
    def estimate_plate_layout(crop_bgr: np.ndarray) -> str:
        """
        Determines plate layout based on aspect ratio (W / H):
        - Vietnamese long plate (one_row): W/H >= 2.0 (typically ~4.3:1)
        - Vietnamese square/motorcycle plate (two_rows): W/H < 2.0 (typically ~1.4:1)
        """
        h, w = crop_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return "one_row"
        aspect_ratio = float(w) / float(h)
        return "one_row" if aspect_ratio >= 2.0 else "two_rows"

    def prepare_input(
        self,
        plate_crop: np.ndarray,
        layout: str = "auto"
    ) -> Tuple[torch.Tensor, str, Optional[np.ndarray], np.ndarray]:
        """
        Preprocesses a raw plate crop:
        1. Layout handling: If two_rows or aspect_ratio < 1.9, rectifies via rectify_two_row_plate.
        2. Resizes to (128, 32).
        3. Converts BGR to RGB and normalizes to [-1.0, 1.0].
        4. Permutes to [3, 32, 128] torch tensor.
        
        Returns:
            tensor: [3, 32, 128] torch.Tensor
            detected_layout: "one_row" or "two_rows"
            rectified_bgr: rectified intermediate image or None
            processed_bgr: image resized to (128, 32)
        """
        h, w = plate_crop.shape[:2]
        if h <= 0 or w <= 0:
            dummy_bgr = np.full((self.target_size[1], self.target_size[0], 3), 255, dtype=np.uint8)
            dummy_tensor = torch.zeros((3, self.target_size[1], self.target_size[0]), dtype=torch.float32)
            return dummy_tensor, "one_row", None, dummy_bgr

        if layout == "auto":
            detected_layout = self.estimate_plate_layout(plate_crop)
        else:
            detected_layout = layout

        # Two-row rectification logic
        rectified_bgr = None
        aspect_ratio = float(w) / float(max(1, h))
        if detected_layout in ("two_rows", "two_rows_label_xe_may", "motorcycle") or (layout == "auto" and aspect_ratio < 1.9):
            rectified_bgr, _, _ = rectify_two_row_plate(plate_crop, target_h=48)
            working_crop = rectified_bgr
            detected_layout = "two_rows"
        else:
            working_crop = plate_crop
            detected_layout = "one_row"

        # Resize to PARSeq target size (tw=128, th=32)
        tw, th = self.target_size
        resized_bgr = cv2.resize(working_crop, (tw, th), interpolation=cv2.INTER_LINEAR)

        # Normalize RGB to [-1.0, 1.0]
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - 0.5) / 0.5
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()

        return tensor, detected_layout, rectified_bgr, resized_bgr

    @torch.inference_mode()
    def recognize(
        self,
        plate_crop: np.ndarray,
        layout: str = "auto"
    ) -> Dict[str, Any]:
        """
        Runs OCR on a single plate crop (BGR).
        
        Args:
            plate_crop: np.ndarray of shape (H, W, 3) BGR.
            layout: "auto", "one_row", or "two_rows".
            
        Returns:
            Dict containing:
                text: Normalized uppercase alphanumeric string (e.g. '30A12345')
                confidence: Sequence mean probability float [0.0, 1.0]
                raw_text: Raw output string before character cleaning
                layout: 'one_row' or 'two_rows'
                rectified_image: intermediate rectified image (if two-row) or None
                processed_image: (128, 32) preprocessed image fed to PARSeq
                ocr_ms: Total OCR execution time in milliseconds
                timings: Dict of prep_ms, forward_ms, decode_ms, total_ms
        """
        t0 = time.perf_counter()

        tensor, detected_layout, rectified_bgr, processed_bgr = self.prepare_input(plate_crop, layout=layout)
        t_prep = time.perf_counter()

        input_batch = tensor.unsqueeze(0).to(self.device, non_blocking=True)

        if self.use_amp:
            with torch.amp.autocast("cuda"):
                logits = self.model(input_batch)
        else:
            logits = self.model(input_batch)
        t_fwd = time.perf_counter()

        probs = logits.softmax(-1)
        preds, confs = self.model.tokenizer.decode(probs)
        t_dec = time.perf_counter()

        raw_pred = preds[0] if preds else ""
        norm_pred = normalize_plate_text(raw_pred)

        if confs and len(confs) > 0:
            c = confs[0]
            conf = float(c.mean().item()) if hasattr(c, "mean") else 0.95
        else:
            conf = 0.95

        total_ms = (t_dec - t0) * 1000.0
        prep_ms = (t_prep - t0) * 1000.0
        fwd_ms = (t_fwd - t_prep) * 1000.0
        dec_ms = (t_dec - t_fwd) * 1000.0

        return {
            "text": norm_pred,
            "confidence": conf,
            "raw_text": raw_pred,
            "layout": detected_layout,
            "rectified_image": rectified_bgr,
            "processed_image": processed_bgr,
            "ocr_ms": total_ms,
            "timings": {
                "prep_ms": prep_ms,
                "forward_ms": fwd_ms,
                "decode_ms": dec_ms,
                "total_ms": total_ms,
            }
        }

    @torch.inference_mode()
    def recognize_batch(
        self,
        plate_crops: List[np.ndarray],
        layouts: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch recognition for multiple plate crops in a single forward pass.
        """
        if not plate_crops:
            return []

        if layouts is None:
            layouts = ["auto"] * len(plate_crops)
        elif len(layouts) != len(plate_crops):
            layouts = ["auto"] * len(plate_crops)

        t0 = time.perf_counter()
        tensors = []
        meta_info = []

        for crop, lay in zip(plate_crops, layouts):
            tensor, detected_layout, rectified_bgr, processed_bgr = self.prepare_input(crop, layout=lay)
            tensors.append(tensor)
            meta_info.append((detected_layout, rectified_bgr, processed_bgr))

        t_prep = time.perf_counter()
        batch_tensor = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)

        if self.use_amp:
            with torch.amp.autocast("cuda"):
                logits = self.model(batch_tensor)
        else:
            logits = self.model(batch_tensor)
        t_fwd = time.perf_counter()

        probs = logits.softmax(-1)
        preds, confs = self.model.tokenizer.decode(probs)
        t_dec = time.perf_counter()

        total_batch_ms = (t_dec - t0) * 1000.0
        fwd_ms = (t_fwd - t_prep) * 1000.0
        per_plate_ms = total_batch_ms / max(1, len(plate_crops))

        results = []
        for i in range(len(plate_crops)):
            raw_pred = preds[i] if i < len(preds) else ""
            norm_pred = normalize_plate_text(raw_pred)
            c = confs[i] if i < len(confs) else None
            conf = float(c.mean().item()) if (c is not None and hasattr(c, "mean")) else 0.95
            detected_layout, rectified_bgr, processed_bgr = meta_info[i]

            results.append({
                "text": norm_pred,
                "confidence": conf,
                "raw_text": raw_pred,
                "layout": detected_layout,
                "rectified_image": rectified_bgr,
                "processed_image": processed_bgr,
                "ocr_ms": per_plate_ms,
                "timings": {
                    "forward_ms": fwd_ms / len(plate_crops),
                    "total_ms": per_plate_ms,
                }
            })
        return results

    def warmup(self, num_runs: int = 15):
        """Warms up CUDA kernels and torch.hub modules to eliminate first-run latency spikes."""
        dummy = np.full((48, 160, 3), 128, dtype=np.uint8)
        print(f"[PARSeqVNRecognizer] Warming up ({num_runs} iterations)...", end="", flush=True)
        for _ in range(num_runs):
            _ = self.recognize(dummy, layout="one_row")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print(" Warmup complete.")
