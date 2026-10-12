"""
Teacher Adapter for PARSeq (Baudm/parseq) on Vietnamese License Plate OCR.
Loads pretrained PARSeq Tiny from torch.hub, fine-tunes on train.csv (29,836 samples),
evaluates on val.csv and val_unseen_text_v2.csv, and checks Teacher Qualification.
Outputs: weights/ocr/teachers/parseq_tiny_vn/best.pth
"""

import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.preprocess import crop_plate_gt, rectify_two_row_plate
from tools.evaluate_teacher import evaluate_teacher, TeacherEvaluationDataset, teacher_collate_fn


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PARSeqTrainDataset(Dataset):
    """Dataset for training PARSeq Teacher with project-consistent preprocessing."""
    def __init__(self, manifest_csv: str, target_size: Tuple[int, int] = (128, 32)):
        self.crop_plate_gt = crop_plate_gt
        self.rectify_two_row_plate = rectify_two_row_plate
        self.target_size = target_size
        self.records = []
        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bbox = None
                if "x1" in row and row["x1"] and row["x1"].strip():
                    try:
                        bbox = [int(float(row["x1"])), int(float(row["y1"])), int(float(row["x2"])), int(float(row["y2"]))]
                    except Exception:
                        bbox = None
                layout = row.get("subset", row.get("layout", "one_row"))
                self.records.append({
                    "image_path": row["image_path"],
                    "plate_text": row["plate_text"],
                    "layout": layout,
                    "bbox": bbox,
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        img_bgr = cv2.imread(item["image_path"])
        if img_bgr is None:
            img_bgr = np.full((100, 300, 3), 255, dtype=np.uint8)

        # 1. Ground truth crop
        if item["bbox"] is not None and len(item["bbox"]) == 4:
            crop = self.crop_plate_gt(img_bgr, item["bbox"], pad_ratio=0.03)
        else:
            crop = img_bgr

        # 2. Rectify two-row plates
        if item["layout"] in ("two_rows", "two_rows_label_xe_may"):
            crop, _, _ = self.rectify_two_row_plate(crop, target_h=48)

        # 3. Light OCR augmentation
        if random.random() < 0.5:
            # Contrast/brightness
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-15, 15)
            crop = np.clip(alpha * crop + beta, 0, 255).astype(np.uint8)
            # Slight rotation
            if random.random() < 0.3:
                h, w = crop.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), random.uniform(-2.5, 2.5), 1.0)
                crop = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 4. Resize to target size (128, 32)
        tw, th = self.target_size
        resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)

        # 5. Normalize [-1.0, 1.0] RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - 0.5) / 0.5
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()

        return {
            "image": tensor,
            "raw_text": item["plate_text"].upper(),
        }


def parseq_train_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[str]]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = [item["raw_text"] for item in batch]
    return images, labels


def compute_parseq_loss(model: Any, images: torch.Tensor, labels: List[str], device: torch.device) -> torch.Tensor:
    tgt = model.tokenizer.encode(labels, device)
    memory = model.model.encode(images)
    tgt_perms = model.gen_tgt_perms(tgt)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    tgt_padding_mask = (tgt_in == model.pad_id) | (tgt_in == model.eos_id)

    loss = 0
    loss_numel = 0
    n = (tgt_out != model.pad_id).sum().item()
    for i, perm in enumerate(tgt_perms):
        tgt_mask, query_mask = model.generate_attn_masks(perm)
        out = model.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
        logits = model.model.head(out).flatten(end_dim=1)
        loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=model.pad_id)
        loss_numel += n
        if i == 1:
            tgt_out = torch.where(tgt_out == model.eos_id, model.pad_id, tgt_out)
            n = (tgt_out != model.pad_id).sum().item()
    return loss / max(1, loss_numel)


def parseq_predict_fn(model: Any, images: torch.Tensor) -> Tuple[List[str], List[float]]:
    logits = model(images)
    probs = logits.softmax(-1)
    preds, confs = model.tokenizer.decode(probs)
    # Post-process: uppercase, remove spaces/punctuation
    clean_preds = []
    clean_confs = []
    for p, c in zip(preds, confs):
        s = p.upper().replace("-", "").replace(".", "").replace(" ", "").replace("_", "")
        clean_preds.append(s)
        mean_c = float(c.mean().item()) if hasattr(c, "mean") else 0.95
        clean_confs.append(mean_c)
    return clean_preds, clean_confs


def train_parseq_teacher(
    train_manifest: Path = Path("datasets/VNLP_processed/train.csv"),
    val_manifest: Path = Path("datasets/VNLP_processed/val.csv"),
    val_unseen_manifest: Path = Path("datasets/VNLP_processed/val_unseen_text_v2.csv"),
    output_dir: Path = Path("weights/ocr/teachers/parseq_tiny_vn"),
    lr: float = 1e-4,
    max_epochs: int = 10,
    patience: int = 4,
    batch_size: int = 64,
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 75)
    print("TEACHER SEARCH: TRAINING PARSEQ-TINY ON VNLP")
    print(f"Pretrained Base: baudm/parseq (parseq_tiny)")
    print(f"Output: {output_dir}")
    print(f"Hyperparameters: LR={lr}, MaxEpochs={max_epochs}, Patience={patience}, BatchSize={batch_size}")
    print("=" * 75)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "best.pth"

    # Preload unseen set
    unseen_paths_set = set()
    with open(val_unseen_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            unseen_paths_set.add(r["image_path"])

    # Load Pretrained PARSeq
    print("Loading pretrained PARSeq Tiny from torch.hub...")
    model = torch.hub.load("baudm/parseq", "parseq_tiny", pretrained=True, trust_repo=True).to(device)

    # Initial zero-shot validation
    print("Evaluating initial pretrained zero-shot on Validation...")
    init_m, init_records = evaluate_teacher(model, str(val_manifest), parseq_predict_fn, device=device, batch_size=128)
    init_unseen = [r for r in init_records if r["image_path"] in unseen_paths_set]
    init_uex = (sum(1 for r in init_unseen if r["exact"]) / len(init_unseen)) * 100.0 if init_unseen else 0.0
    print(f"Initial Zero-Shot | VAL: {init_m['exact_accuracy']:.2f}% (Unseen: {init_uex:.2f}%) | CER: {init_m['cer']:.4f}")

    best_val_exact = init_m["exact_accuracy"]
    best_val_cer = init_m["cer"]
    best_unseen_exact = init_uex
    best_metrics = init_m
    best_epoch = 0
    epochs_no_improve = 0
    history_records = []

    # Dataset & Dataloader
    train_ds = PARSeqTrainDataset(str(train_manifest), target_size=(128, 32))
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=parseq_train_collate_fn, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        start_t = time.time()

        for images, labels in train_ld:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                loss = compute_parseq_loss(model, images, labels, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_ld)
        scheduler.step()
        epoch_dur = time.time() - start_t

        # Evaluate on Validation
        val_m, val_records = evaluate_teacher(model, str(val_manifest), parseq_predict_fn, device=device, batch_size=128)
        unseen_recs = [r for r in val_records if r["image_path"] in unseen_paths_set]
        uex = (sum(1 for r in unseen_recs if r["exact"]) / len(unseen_recs)) * 100.0 if unseen_recs else 0.0

        vex = val_m["exact_accuracy"]
        vcer = val_m["cer"]
        rm = val_m["layout_breakdown"]["two_rows_label_xe_may"]["exact_accuracy"]

        is_improved = False
        if vex > best_val_exact + 0.05:
            is_improved = True
        elif abs(vex - best_val_exact) <= 0.05 and vcer < best_val_cer - 0.0003:
            is_improved = True

        history_records.append({
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "val_exact": round(vex, 2),
            "val_unseen": round(uex, 2),
            "val_moto": round(rm, 2),
            "val_cer": round(vcer, 4),
            "duration_s": round(epoch_dur, 1),
            "is_best": is_improved,
        })

        if is_improved:
            best_val_exact = vex
            best_val_cer = vcer
            best_unseen_exact = uex
            best_metrics = val_m
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_exact_acc": best_val_exact,
                "val_metrics": val_m,
                "val_unseen_exact": uex,
            }, best_ckpt_path)
            print(f"  Epoch [{epoch:02d}/{max_epochs:02d}] Loss: {avg_loss:.4f} | VAL: {vex:.2f}% (Unseen: {uex:.2f}%) | Moto: {rm:.1f}% | CER: {vcer:.4f} | {epoch_dur:.1f}s [*BEST*]")
        else:
            epochs_no_improve += 1
            print(f"  Epoch [{epoch:02d}/{max_epochs:02d}] Loss: {avg_loss:.4f} | VAL: {vex:.2f}% (Unseen: {uex:.2f}%) | Moto: {rm:.1f}% | CER: {vcer:.4f} | {epoch_dur:.1f}s")

        if epochs_no_improve >= patience:
            print(f"  [Early Stopping] No improvement for {patience} epochs. Stopping at epoch {epoch}.")
            break

    # Determine Teacher Qualification
    # Student Baseline: VAL = 86.02%, Unseen = 84.86%, Moto = 80.86%, CER = 0.0252
    if best_val_exact >= 88.0 and best_unseen_exact >= 84.86:
        qualification = "QUALIFIED_STRONG"
    elif best_val_exact >= 87.0 and best_val_cer <= 0.0252:
        qualification = "QUALIFIED_MODERATE"
    else:
        qualification = "NOT_QUALIFIED"

    print("\n" + "=" * 60)
    print(f"PARSEQ TEACHER COMPLETE | Best Epoch: {best_epoch}")
    print(f"VAL Exact: {best_val_exact:.2f}% (Student Base: 86.02%) | Delta: {best_val_exact - 86.02:+.2f} pp")
    print(f"VAL Unseen: {best_unseen_exact:.2f}% (Student Base: 84.86%) | Delta: {best_unseen_exact - 84.86:+.2f} pp")
    print(f"CER: {best_val_cer:.4f} (Student Base: 0.0252)")
    print(f"Qualification: [{qualification}]")
    print("=" * 60)

    # Save training history
    history_file = output_dir / "training_history.csv"
    if history_records:
        with open(history_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_records[0].keys()))
            writer.writeheader()
            writer.writerows(history_records)

    # Update TEACHER_KD_EXPERIMENT_REGISTRY.csv
    registry_file = Path("reports/TEACHER_KD_EXPERIMENT_REGISTRY.csv")
    if registry_file.exists():
        rows = []
        with open(registry_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["experiment_id"] != "TEACHER_PARSEQ":
                    rows.append(row)
        rows.append({
            "experiment_id": "TEACHER_PARSEQ",
            "stage": "TEACHER",
            "architecture": "PARSeq-Tiny",
            "pretrained": "baudm/parseq",
            "seed": seed,
            "status": "COMPLETE",
            "best_epoch": best_epoch,
            "val_exact": round(best_val_exact, 2),
            "val_unseen": round(best_unseen_exact, 2),
            "one_row": round(best_metrics["layout_breakdown"]["one_row"]["exact_accuracy"], 2),
            "two_rows": round(best_metrics["layout_breakdown"]["two_rows"]["exact_accuracy"], 2),
            "motorcycle": round(best_metrics["layout_breakdown"]["two_rows_label_xe_may"]["exact_accuracy"], 2),
            "char_acc": round(best_metrics["char_accuracy"], 2),
            "cer": round(best_val_cer, 4),
            "checkpoint": str(best_ckpt_path).replace("\\", "/"),
            "decision": qualification,
            "notes": f"Pretrained PARSeq Tiny FT; Val delta: {best_val_exact - 86.02:+.2f} pp",
        })
        with open(registry_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    res = {
        "architecture": "PARSeq-Tiny",
        "pretrained": "baudm/parseq",
        "seed": seed,
        "best_epoch": best_epoch,
        "val_exact": round(best_val_exact, 2),
        "val_unseen": round(best_unseen_exact, 2),
        "one_row": round(best_metrics["layout_breakdown"]["one_row"]["exact_accuracy"], 2),
        "two_rows": round(best_metrics["layout_breakdown"]["two_rows"]["exact_accuracy"], 2),
        "motorcycle": round(best_metrics["layout_breakdown"]["two_rows_label_xe_may"]["exact_accuracy"], 2),
        "char_acc": round(best_metrics["char_accuracy"], 2),
        "cer": round(best_val_cer, 4),
        "checkpoint_path": str(best_ckpt_path).replace("\\", "/"),
        "qualification": qualification,
    }
    return res


if __name__ == "__main__":
    train_parseq_teacher()

