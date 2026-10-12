"""
Export PARSeq-Tiny VNLP OCR model to ONNX for Pi 5 CPU deployment.
Checkpoint: weights/ocr/teachers/parseq_tiny_vn/best.pth
Target artifact: weights/deployment/pi_cpu/parseq_tiny_vn_fp32.onnx
Fixed shape [1, 3, 32, 128], FP32, Opset 17.
Uses ParseqUnrolledWrapper to expose deterministic tensor-in / tensor-out graph
without dynamic boolean masks or in-place token assignments.
"""

import hashlib
import sys
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ocr.parseq_vn_ocr import ParseqVNRecognizer


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class ParseqUnrolledWrapper(nn.Module):
    """
    Wraps core PARSeq model to unroll autoregressive decoding steps into
    pure tensor operations and register static attention masks as buffers.
    This eliminates unsupported ONNX boolean Where operations and data-dependent
    in-place token mutations during ONNX tracing.
    """
    def __init__(self, core_model: nn.Module, tokenizer):
        super().__init__()
        self.model = core_model
        self.max_length = int(self.model.max_label_length)  # 25
        self.num_steps = self.max_length + 1  # 26
        self.bos_id = int(tokenizer.bos_id)  # 95
        self.pad_id = int(tokenizer.pad_id)  # 96
        self.eos_id = int(tokenizer.eos_id)  # 0
        self.refine_iters = int(self.model.refine_iters)  # 1

        # Precompute static attention masks as registered buffers
        tgt_mask = torch.triu(torch.ones((self.num_steps, self.num_steps), dtype=torch.bool), 1)
        self.register_buffer("tgt_mask", tgt_mask)

        triu_mask = torch.triu(torch.ones((self.num_steps, self.num_steps), dtype=torch.bool), 2)
        query_mask_refine = torch.where(triu_mask, torch.zeros_like(tgt_mask), tgt_mask)
        self.register_buffer("query_mask_refine", query_mask_refine)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        bs = images.shape[0]
        memory = self.model.encode(images)
        pos_queries = self.model.pos_queries[:, : self.num_steps].expand(bs, -1, -1)

        # Step 0: Initial BOS token
        tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=images.device)
        logits = []

        # Autoregressive generation loop
        for i in range(self.num_steps):
            j = i + 1
            tgt_out = self.model.decode(
                tgt_in,
                memory,
                self.tgt_mask[:j, :j],
                tgt_query=pos_queries[:, i:j],
                tgt_query_mask=self.tgt_mask[i:j, :j],
            )
            p_i = self.model.head(tgt_out)
            logits.append(p_i)
            if j < self.num_steps:
                next_tok = p_i[:, -1:].argmax(dim=-1)
                tgt_in = torch.cat([tgt_in, next_tok], dim=1)

        logits = torch.cat(logits, dim=1)

        # Iterative refinement step
        if self.refine_iters > 0:
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=images.device)
            for _ in range(self.refine_iters):
                tgt_in_refine = torch.cat([bos, logits[:, :-1].argmax(dim=-1)], dim=1)
                tgt_padding_mask = (tgt_in_refine == self.eos_id).int().cumsum(dim=-1) > 0
                tgt_out = self.model.decode(
                    tgt_in_refine,
                    memory,
                    self.tgt_mask,
                    tgt_padding_mask,
                    pos_queries,
                    self.query_mask_refine[:, : tgt_in_refine.shape[1]],
                )
                logits = self.model.head(tgt_out)

        return logits


def main():
    print("=" * 80)
    print("PARSeq-Tiny VNLP OCR ONNX EXPORT (Pi 5 CPU Deployment)")
    print("=" * 80)

    ckpt_path = PROJECT_ROOT / "weights/ocr/teachers/parseq_tiny_vn/best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    target_dir = PROJECT_ROOT / "weights/deployment/pi_cpu"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_onnx = target_dir / "parseq_tiny_vn_fp32.onnx"

    print(f"Loading PARSeq-Tiny VNLP checkpoint: {ckpt_path}")
    source_sha256 = compute_sha256(ckpt_path)
    print(f"Source SHA256: {source_sha256}")

    rec = ParseqVNRecognizer(str(ckpt_path), device="cpu")
    core_model = rec.model.model.eval()
    tokenizer = rec.model.tokenizer

    wrapper = ParseqUnrolledWrapper(core_model, tokenizer).eval()

    # Pre-export sanity check with dummy input
    dummy_input = torch.randn(1, 3, 32, 128, dtype=torch.float32)
    with torch.no_grad():
        out_orig = rec.model(dummy_input)
        out_wrap = wrapper(dummy_input)
        pre_diff = float((out_orig - out_wrap).abs().max())
    print(f"PyTorch Wrapper Equivalence Pre-check Max Diff: {pre_diff:.6e}")
    assert pre_diff < 1e-4, f"Pre-check wrapper diff too large: {pre_diff}"

    print(f"\nExporting to ONNX: {target_onnx}")
    print("Input: [1, 3, 32, 128], FP32, Opset 17, Fixed shape")

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(target_onnx),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    print("Torch ONNX export complete.")

    # Audit exported ONNX
    print("\nAuditing ONNX structure...")
    onnx_m = onnx.load(str(target_onnx))
    onnx.checker.check_model(onnx_m)
    print("onnx.checker.check_model(): PASS")

    file_size = target_onnx.stat().st_size
    onnx_sha256 = compute_sha256(target_onnx)
    opset = [op.version for op in onnx_m.opset_import if op.domain == "" or op.domain == "ai.onnx"][0]

    inputs = []
    for inp in onnx_m.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        inputs.append((inp.name, shape, dtype))

    outputs = []
    for out in onnx_m.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        dtype = out.type.tensor_type.elem_type
        outputs.append((out.name, shape, dtype))

    print("-" * 80)
    print(f"FILE        : {target_onnx}")
    print(f"FILE SIZE   : {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
    print(f"SHA256      : {onnx_sha256}")
    print(f"OPSET       : {opset}")
    print(f"INPUTS      : {inputs}")
    print(f"OUTPUTS     : {outputs}")
    print("-" * 80)

    # Sanity check with ONNX Runtime CPU
    print("\nVerifying with ONNX Runtime CPUExecutionProvider...")
    sess = ort.InferenceSession(str(target_onnx), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"image": dummy_input.numpy()})[0]
    num_diff = float((out_orig.numpy() - ort_out).max())
    print(f"Dummy Run ONNX Runtime Output Shape: {ort_out.shape}")
    print(f"Max Absolute Diff (PyTorch vs ONNX Runtime): {num_diff:.6e}")
    assert num_diff < 1e-3, f"ONNX Runtime numerical diff too large: {num_diff}"
    print("PARSeq ONNX Export and CPU Sanity Check: PASS!")
    print("=" * 80)


if __name__ == "__main__":
    main()
