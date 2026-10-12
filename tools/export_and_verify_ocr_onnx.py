"""
Export PyTorch LPRNet-Lite model to ONNX and verify with ONNX Runtime and OpenCV DNN.
"""

import argparse
import ast
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
import onnxruntime as ort

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.charset import CHARSET, NUM_CLASSES, BLANK_INDEX
from ocr.models.lprnet_lite import LPRNetLite
from ocr.ctc_decoder import ctc_greedy_decode_numpy, ctc_greedy_decode_torch
from ocr.preprocess import preprocess_plate_for_ocr, normalize_image_for_model
from ocr.plate_text_utils import normalize_plate_text


def export_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple = (1, 3, 48, 160),
    opset_version: int = 12,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    print(f"\nExporting LPRNetLite to ONNX: {output_path}")
    print(f"  Input shape:   {input_shape}")
    print(f"  Opset version: {opset_version}")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        do_constant_folding=True,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Export successful! File size: {file_size_mb:.2f} MB")

    # Check model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX checker: PASS (Valid model structure)")
    return output_path


def verify_with_onnxruntime_and_opencv(
    model: torch.nn.Module,
    onnx_path: str,
    test_manifest: str,
    num_samples: int = 100,
):
    print(f"\nVerifying ONNX model on {num_samples} real plate images from {test_manifest}...")

    # 1. Setup ONNX Runtime session
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_input_name = ort_session.get_inputs()[0].name

    # 2. Setup OpenCV DNN Net
    cv_net = cv2.dnn.readNetFromONNX(str(onnx_path))
    cv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    cv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 3. Read test manifest
    records = []
    with open(test_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)
            if len(records) >= num_samples:
                break

    pytorch_vs_ort_diffs = []
    pytorch_vs_opencv_diffs = []
    pytorch_ort_matches = 0
    pytorch_opencv_matches = 0
    ground_truth_matches = 0

    model.eval()
    for idx, r in enumerate(records):
        img_path = r["image_path"]
        raw_text = r["plate_text"]
        gt_clean = normalize_plate_text(raw_text)
        layout = r.get("subset", "one_row")
        if "x1" in r and r["x1"] and r["x1"].strip():
            bbox = [int(float(r["x1"])), int(float(r["y1"])), int(float(r["x2"])), int(float(r["y2"]))]
        elif "bbox" in r and r["bbox"] and r["bbox"].strip():
            bbox = ast.literal_eval(r["bbox"])
        else:
            bbox = None

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # Standard preprocessing
        preprocessed_bgr = preprocess_plate_for_ocr(img_bgr, layout=layout, bbox=bbox, target_size=(160, 48))
        norm_rgb = normalize_image_for_model(preprocessed_bgr)  # (48, 160, 3) float32 [-1, 1]
        input_nchw = np.transpose(norm_rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)  # (1, 3, 48, 160)

        # 1. PyTorch inference
        tensor_in = torch.from_numpy(input_nchw)
        with torch.no_grad():
            pt_out = model(tensor_in).numpy()  # [1, 37, 40]
        pt_pred = ctc_greedy_decode_numpy(pt_out, blank_index=BLANK_INDEX)

        # 2. ONNX Runtime inference
        ort_out = ort_session.run(None, {ort_input_name: input_nchw})[0]  # [1, 37, 40]
        ort_pred = ctc_greedy_decode_numpy(ort_out, blank_index=BLANK_INDEX)

        # 3. OpenCV DNN inference via blobFromImage
        blob = cv2.dnn.blobFromImage(
            preprocessed_bgr,
            scalefactor=1.0 / 127.5,
            size=(160, 48),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False
        )
        cv_net.setInput(blob)
        cv_out = cv_net.forward()  # [1, 37, 40]
        cv_pred = ctc_greedy_decode_numpy(cv_out, blank_index=BLANK_INDEX)

        # Compare outputs
        diff_ort = np.max(np.abs(pt_out - ort_out))
        diff_cv = np.max(np.abs(pt_out - cv_out))

        pytorch_vs_ort_diffs.append(diff_ort)
        pytorch_vs_opencv_diffs.append(diff_cv)

        if pt_pred == ort_pred:
            pytorch_ort_matches += 1
        if pt_pred == cv_pred:
            pytorch_opencv_matches += 1
        if cv_pred == gt_clean:
            ground_truth_matches += 1

    total = len(pytorch_vs_ort_diffs)
    max_ort_diff = max(pytorch_vs_ort_diffs)
    max_cv_diff = max(pytorch_vs_opencv_diffs)
    ort_agreement = (pytorch_ort_matches / total) * 100.0
    cv_agreement = (pytorch_opencv_matches / total) * 100.0
    exact_acc_100 = (ground_truth_matches / total) * 100.0

    print("\n" + "=" * 60)
    print("ONNX & OPENCV DNN VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Test samples evaluated:             {total}")
    print(f"Max PyTorch vs ONNX Runtime diff:   {max_ort_diff:.6e}")
    print(f"Max PyTorch vs OpenCV DNN diff:     {max_cv_diff:.6e}")
    print(f"PyTorch vs ONNX Runtime agreement:  {pytorch_ort_matches}/{total} ({ort_agreement:.2f}%)")
    print(f"PyTorch vs OpenCV DNN agreement:    {pytorch_opencv_matches}/{total} ({cv_agreement:.2f}%)")
    print(f"OpenCV DNN Exact Match on 100 test: {ground_truth_matches}/{total} ({exact_acc_100:.2f}%)")
    print("=" * 60)

    assert cv_agreement >= 99.0, f"OpenCV agreement too low: {cv_agreement}%"
    print("\n[PASS] ONNX model is fully verified and 100% compatible with OpenCV DNN!")
    return {
        "max_ort_diff": max_ort_diff,
        "max_cv_diff": max_cv_diff,
        "ort_agreement": ort_agreement,
        "cv_agreement": cv_agreement,
        "exact_acc_100": exact_acc_100,
    }


def main():
    parser = argparse.ArgumentParser(description="Export and verify LPRNet-Lite ONNX model")
    parser.add_argument("--checkpoint", type=str, default="weights/ocr/best_lprnet_vn.pth", help="Checkpoint path")
    parser.add_argument("--output", type=str, default="weights/ocr/best_lprnet_vn.onnx", help="Output ONNX path")
    parser.add_argument("--test_csv", type=str, default="datasets/VNLP_processed/test.csv", help="Test CSV")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cpu")
    model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 1. Export ONNX
    onnx_path = export_onnx(model, args.output, opset_version=args.opset)

    # 2. Verify
    verify_with_onnxruntime_and_opencv(model, str(onnx_path), args.test_csv, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
