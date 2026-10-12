"""
Export YOLO11s vehicle detector to ONNX for Pi 5 CPU deployment.
Loads yolo11s.pt and exports to weights/deployment/pi_cpu/yolo11s_640_fp32.onnx.
Fixed shape [1, 3, 640, 640], FP32, Opset 17, Raw detector output.
"""

import hashlib
import shutil
from pathlib import Path
import onnx
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("EXCELSIOR / ULTRALYTICS YOLO11s VEHICLE DETECTOR ONNX EXPORT")
    print("=" * 80)

    pt_path = PROJECT_ROOT / "yolo11s.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Source model not found: {pt_path}")

    target_dir = PROJECT_ROOT / "weights/deployment/pi_cpu"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_onnx = target_dir / "yolo11s_640_fp32.onnx"

    print(f"Loading PyTorch checkpoint: {pt_path}")
    print(f"Source SHA256: {compute_sha256(pt_path)}")
    model = YOLO(str(pt_path))

    print("\nStarting ONNX export...")
    # Export using Ultralytics API
    exported_path = model.export(
        format="onnx",
        imgsz=640,
        batch=1,
        dynamic=False,
        half=False,
        opset=17,
        nms=False,
        simplify=True,
    )
    exported_p = Path(exported_path)
    print(f"Ultralytics exported to: {exported_p}")

    # Move/Copy to target deployment path
    if exported_p.resolve() != target_onnx.resolve():
        shutil.copy2(exported_p, target_onnx)
        print(f"Copied exported model to: {target_onnx}")

    # Inspect and audit exported ONNX model
    print("\nAuditing exported ONNX model structure...")
    onnx_model = onnx.load(str(target_onnx))
    onnx.checker.check_model(onnx_model)
    print("onnx.checker.check_model(): PASS")

    file_size = target_onnx.stat().st_size
    sha256 = compute_sha256(target_onnx)
    opset = [op.version for op in onnx_model.opset_import if op.domain == "" or op.domain == "ai.onnx"][0]

    inputs = []
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        inputs.append((inp.name, shape, dtype))

    outputs = []
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        dtype = out.type.tensor_type.elem_type
        outputs.append((out.name, shape, dtype))

    print("-" * 80)
    print(f"FILE        : {target_onnx}")
    print(f"FILE SIZE   : {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
    print(f"SHA256      : {sha256}")
    print(f"OPSET       : {opset}")
    print(f"INPUTS      : {inputs}")
    print(f"OUTPUTS     : {outputs}")
    print("-" * 80)
    print("YOLO11s ONNX Export Successful!")
    print("=" * 80)


if __name__ == "__main__":
    main()
