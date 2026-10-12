"""
Knowledge Distillation Training Pipeline for Vietnamese License Plate OCR.
Distills knowledge from a qualified Teacher (e.g. PARSeq) into Student (LPRNetLite).

Invariant Principles:
  1. Base Student initialized strictly from Phase 4 Control Best:
     weights/ocr/experiments/phase4/control_seed42/best.pth
  2. NEVER overwrite Phase 4 control checkpoint.
  3. Base supervised loss: CTC Loss on Ground Truth.
  4. Sequence-level distillation with confidence filtering:
     - Teacher is frozen (eval mode, requires_grad=False).
     - Teacher predictions decoded with confidence.
     - Teacher pseudo-labels applied only when confidence >= threshold.
     - Ground truth is never overridden by low-confidence teacher errors.
  5. Evaluation strictly on VALIDATION (val.csv, val_unseen_text_v2.csv).
  6. TEST sets remain untouched during training and model selection.
  7. Registry updated in reports/TEACHER_KD_EXPERIMENT_REGISTRY.csv.
"""

import argparse
import ast
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.charset import CHARSET, NUM_CLASSES, BLANK_INDEX, encode_text
from ocr.models.lprnet_lite import LPRNetLite
from ocr.preprocess import crop_plate_gt, rectify_two_row_plate, letterbox_ocr, normalize_image_for_model
from tools.evaluate_ocr_model import evaluate_manifest
from teacher_adapters.parseq_vn import parseq_predict_fn


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class KDTrainingDataset(Dataset):
    """
    Dataset that yields both Student input [3, 48, 160] and Teacher input [3, 32, 128]
    from the exact same rectified and augmented plate crop.
    """
    def __init__(
        self,
        manifest_csv: str,
        student_size: Tuple[int, int] = (160, 48),  # (W, H)
        teacher_size: Tuple[int, int] = (128, 32),  # (W, H)
        is_training: bool = True,
        augment_p: float = 0.5,
    ):
        self.manifest_path = Path(manifest_csv)
        self.student_size = student_size
        self.teacher_size = teacher_size
        self.is_training = is_training
        self.augment_p = augment_p

        self.records: List[Dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bbox = None
                if "x1" in row and row["x1"] and row["x1"].strip():
                    try:
                        bbox = [int(float(row["x1"])), int(float(row["y1"])), int(float(row["x2"])), int(float(row["y2"]))]
                    except Exception:
                        bbox = None
                elif "bbox" in row and row["bbox"] and row["bbox"].strip():
                    try:
                        bbox = ast.literal_eval(row["bbox"])
                    except Exception:
                        bbox = None

                layout = row.get("subset", row.get("layout", "one_row"))
                self.records.append({
                    "image_path": row["image_path"],
                    "plate_text": row["plate_text"].strip().upper(),
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
            crop = crop_plate_gt(img_bgr, item["bbox"], pad_ratio=0.03)
        else:
            crop = img_bgr

        # 2. Rectify two-row plates if applicable
        if item["layout"] in ("two_rows", "two_rows_label_xe_may"):
            crop, _, _ = rectify_two_row_plate(crop, target_h=48)

        # 3. Light OCR augmentation
        if self.is_training and random.random() < self.augment_p:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-15, 15)
            crop = np.clip(alpha * crop + beta, 0, 255).astype(np.uint8)
            if random.random() < 0.25:
                h, w = crop.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), random.uniform(-2.5, 2.5), 1.0)
                crop = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 4. Student input: letterbox to (160, 48), normalize [-1.0, 1.0]
        student_letterboxed = letterbox_ocr(crop, target_size=self.student_size, pad_color=(255, 255, 255))
        student_norm = normalize_image_for_model(student_letterboxed)
        student_tensor = torch.from_numpy(student_norm).permute(2, 0, 1).float()

        # 5. Teacher input: resize to (128, 32), normalize [-1.0, 1.0] RGB
        tw, th = self.teacher_size
        teacher_resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
        teacher_rgb = cv2.cvtColor(teacher_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        teacher_norm = (teacher_rgb - 0.5) / 0.5
        teacher_tensor = torch.from_numpy(teacher_norm).permute(2, 0, 1).float()

        # 6. Encode Ground Truth text
        gt_encoded = encode_text(item["plate_text"])

        return {
            "student_image": student_tensor,
            "teacher_image": teacher_tensor,
            "gt_label": torch.tensor(gt_encoded, dtype=torch.long),
            "gt_len": len(gt_encoded),
            "raw_text": item["plate_text"],
            "image_path": item["image_path"],
            "layout": item["layout"],
        }


def kd_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    student_images = torch.stack([item["student_image"] for item in batch], dim=0)
    teacher_images = torch.stack([item["teacher_image"] for item in batch], dim=0)

    # Flatten GT labels for CTCLoss
    gt_targets = torch.cat([item["gt_label"] for item in batch], dim=0)
    gt_lengths = torch.tensor([item["gt_len"] for item in batch], dtype=torch.long)
    raw_texts = [item["raw_text"] for item in batch]
    layouts = [item["layout"] for item in batch]

    return {
        "student_images": student_images,
        "teacher_images": teacher_images,
        "gt_targets": gt_targets,
        "gt_lengths": gt_lengths,
        "raw_texts": raw_texts,
        "layouts": layouts,
    }


def load_teacher_model(teacher_ckpt_path: str, device: torch.device) -> Any:
    """Loads trained PARSeq Teacher from checkpoint."""
    print(f"Loading Teacher from: {teacher_ckpt_path}")
    model = torch.hub.load("baudm/parseq", "parseq_tiny", pretrained=False, trust_repo=True)
    ckpt = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("Teacher loaded and frozen successfully.")
    return model


def train_kd_experiment(
    exp_name: str,
    seed: int,
    teacher_ckpt: str,
    student_init_ckpt: str,
    train_manifest: str = "datasets/VNLP_processed/train.csv",
    val_manifest: str = "datasets/VNLP_processed/val.csv",
    val_unseen_manifest: str = "datasets/VNLP_processed/val_unseen_text_v2.csv",
    output_dir_base: str = "weights/ocr/experiments/kd",
    lambda_kd: float = 0.25,
    conf_threshold: float = 0.85,
    lr: float = 5e-5,
    max_epochs: int = 30,
    patience: int = 8,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print(f"KNOWLEDGE DISTILLATION RUN: {exp_name} (Seed {seed})")
    print(f"Teacher Checkpoint: {teacher_ckpt}")
    print(f"Student Init Checkpoint: {student_init_ckpt}")
    print(f"Hyperparameters: lambda_kd={lambda_kd}, conf_thresh={conf_threshold}, lr={lr}, max_epochs={max_epochs}")
    print("=" * 75)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir_base) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "best.pth"

    # Pre-load unseen paths set
    unseen_paths_set = set()
    with open(val_unseen_manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            unseen_paths_set.add(r["image_path"])

    # 1. Load Frozen Teacher
    teacher_model = load_teacher_model(teacher_ckpt, device)

    # 2. Load Student initialized from Phase 4 Control Best
    student_model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
    ckpt = torch.load(student_init_ckpt, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    student_model.load_state_dict(state)
    print(f"Student loaded from {student_init_ckpt}. Params: {sum(p.numel() for p in student_model.parameters()):,}")

    # 3. Initial Baseline Validation
    init_m, init_records = evaluate_manifest(student_model, val_manifest, device=device, batch_size=128, num_workers=num_workers)
    init_unseen = [r for r in init_records if r["image_path"] in unseen_paths_set]
    init_uex = (sum(1 for r in init_unseen if r["exact"]) / len(init_unseen)) * 100.0 if init_unseen else 0.0
    print(f"Student Baseline Check | VAL: {init_m['exact_accuracy']:.2f}% (Unseen: {init_uex:.2f}%) | CER: {init_m['cer']:.4f}")

    best_val_exact = init_m["exact_accuracy"]
    best_val_cer = init_m["cer"]
    best_unseen_exact = init_uex
    best_metrics = init_m
    best_epoch = 0
    epochs_no_improve = 0
    history_records = []

    # Pre-save initial model as best fallback
    torch.save({
        "epoch": 0,
        "model_state_dict": student_model.state_dict(),
        "best_exact_acc": best_val_exact,
        "val_metrics": init_m,
        "val_unseen_exact": init_uex,
        "lambda_kd": lambda_kd,
        "seed": seed,
    }, best_ckpt_path)

    # 4. DataLoaders
    train_dataset = KDTrainingDataset(train_manifest, is_training=True, augment_p=0.5)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=kd_collate_fn,
        pin_memory=True,
    )

    # 5. Optimizer, Scheduler, Loss
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=5e-7)
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_INDEX, reduction="mean", zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # 6. Training Loop
    for epoch in range(1, max_epochs + 1):
        student_model.train()
        total_loss = 0.0
        total_gt_loss = 0.0
        total_kd_loss = 0.0
        kd_samples_used = 0
        total_samples = 0
        start_t = time.time()

        for batch in train_loader:
            student_imgs = batch["student_images"].to(device, non_blocking=True)
            teacher_imgs = batch["teacher_images"].to(device, non_blocking=True)
            gt_targets = batch["gt_targets"].to(device, non_blocking=True)
            gt_lengths = batch["gt_lengths"].to(device, non_blocking=True)
            raw_texts = batch["raw_texts"]
            b_size = student_imgs.size(0)
            total_samples += b_size

            # Teacher inference (frozen)
            with torch.no_grad():
                t_preds, t_confs = parseq_predict_fn(teacher_model, teacher_imgs)

            # Build Teacher KD target labels
            teacher_labels_list = []
            teacher_weights_list = []
            for gt_str, t_pred, t_conf in zip(raw_texts, t_preds, t_confs):
                if t_conf >= conf_threshold and t_pred and all(c in CHARSET for c in t_pred):
                    # Valid confident teacher prediction
                    encoded_t = encode_text(t_pred)
                    teacher_labels_list.append(torch.tensor(encoded_t, dtype=torch.long))
                    # Confidence-weighted: full weight if matches GT, discounted if high-conf alternate
                    w = t_conf if t_pred == gt_str else (0.5 * t_conf if t_conf >= 0.95 else 0.0)
                    teacher_weights_list.append(w)
                    if w > 0:
                        kd_samples_used += 1
                else:
                    teacher_labels_list.append(torch.tensor(encode_text(gt_str), dtype=torch.long))
                    teacher_weights_list.append(0.0)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                # Student forward: [B, 37, 40]
                student_logits = student_model(student_imgs)
                # CTC log_probs: [T, B, C] = [40, B, 37]
                log_probs = student_logits.log_softmax(1).permute(2, 0, 1)
                input_lengths = torch.full((b_size,), 40, dtype=torch.long, device=device)

                # Ground Truth CTC Loss
                loss_gt = ctc_loss_fn(log_probs, gt_targets, input_lengths, gt_lengths)

                # Teacher Distillation Loss
                weights_tensor = torch.tensor(teacher_weights_list, device=device, dtype=torch.float32)
                if weights_tensor.sum() > 0:
                    t_targets = torch.cat(teacher_labels_list, dim=0).to(device)
                    t_lengths = torch.tensor([len(l) for l in teacher_labels_list], dtype=torch.long, device=device)
                    # Compute per-sample CTC loss using reduction='none'
                    ctc_none = nn.CTCLoss(blank=BLANK_INDEX, reduction="none", zero_infinity=True)
                    per_sample_t_loss = ctc_none(log_probs, t_targets, input_lengths, t_lengths)
                    loss_kd = (per_sample_t_loss * weights_tensor).sum() / (weights_tensor.sum() + 1e-6)
                else:
                    loss_kd = torch.tensor(0.0, device=device)

                loss = loss_gt + lambda_kd * loss_kd

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_gt_loss += loss_gt.item()
            total_kd_loss += loss_kd.item() if isinstance(loss_kd, torch.Tensor) else 0.0

        scheduler.step()
        epoch_dur = time.time() - start_t
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_gt_loss = total_gt_loss / n_batches
        avg_kd_loss = total_kd_loss / n_batches
        kd_pct = (kd_samples_used / total_samples) * 100.0

        # Evaluate on Validation
        val_m, val_records = evaluate_manifest(student_model, val_manifest, device=device, batch_size=128, num_workers=num_workers)
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
            "loss_gt": round(avg_gt_loss, 4),
            "loss_kd": round(avg_kd_loss, 4),
            "kd_pct": round(kd_pct, 1),
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
                "model_state_dict": student_model.state_dict(),
                "best_exact_acc": best_val_exact,
                "val_metrics": val_m,
                "val_unseen_exact": uex,
                "lambda_kd": lambda_kd,
                "seed": seed,
            }, best_ckpt_path)
            print(f"  Epoch [{epoch:02d}/{max_epochs:02d}] Loss: {avg_loss:.4f} (KD: {avg_kd_loss:.4f}, {kd_pct:.0f}% used) | VAL: {vex:.2f}% (Unseen: {uex:.2f}%) | Moto: {rm:.1f}% | CER: {vcer:.4f} | {epoch_dur:.1f}s [*BEST*]", flush=True)
        else:
            epochs_no_improve += 1
            print(f"  Epoch [{epoch:02d}/{max_epochs:02d}] Loss: {avg_loss:.4f} (KD: {avg_kd_loss:.4f}, {kd_pct:.0f}% used) | VAL: {vex:.2f}% (Unseen: {uex:.2f}%) | Moto: {rm:.1f}% | CER: {vcer:.4f} | {epoch_dur:.1f}s", flush=True)

        if epochs_no_improve >= patience:
            print(f"  [Early Stopping] No improvement for {patience} epochs. Stopping at epoch {epoch}.", flush=True)
            break

    # Save history
    hist_file = output_dir / "history.csv"
    if history_records:
        with open(hist_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_records[0].keys()))
            writer.writeheader()
            writer.writerows(history_records)

    # Acceptance rule
    is_useful = (best_val_exact > 86.02) and (best_unseen_exact >= 84.5) and (best_val_cer <= 0.0252)
    decision = "QUALIFIED" if is_useful else "INSUFFICIENT_IMPROVEMENT"

    # Update Registry
    registry_file = Path("reports/TEACHER_KD_EXPERIMENT_REGISTRY.csv")
    if registry_file.exists():
        rows = []
        with open(registry_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["experiment_id"] != exp_name:
                    rows.append(row)
        rows.append({
            "experiment_id": exp_name,
            "stage": "KD",
            "architecture": "LPRNetLite",
            "pretrained": "Phase4_Control_S42",
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
            "decision": decision,
            "notes": f"KD-A lambda={lambda_kd}; Val delta: {best_val_exact - 86.02:+.2f} pp",
        })
        with open(registry_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + "=" * 60)
    print(f"KD EXPERIMENT COMPLETE: {exp_name}")
    print(f"VAL Exact: {best_val_exact:.2f}% (Baseline: 86.02%) | Delta: {best_val_exact - 86.02:+.2f} pp")
    print(f"VAL Unseen: {best_unseen_exact:.2f}% (Baseline: 84.86%) | Delta: {best_unseen_exact - 84.86:+.2f} pp")
    print(f"CER: {best_val_cer:.4f} (Baseline: 0.0252)")
    print(f"Decision: [{decision}]")
    print("=" * 60)

    return {
        "exp_name": exp_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_exact": round(best_val_exact, 2),
        "best_val_unseen": round(best_unseen_exact, 2),
        "val_metrics": best_metrics,
        "best_ckpt_path": str(best_ckpt_path).replace("\\", "/"),
        "decision": decision,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation Runner")
    parser.add_argument("--exp_name", type=str, default="kd_a_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher_ckpt", type=str, default="weights/ocr/teachers/parseq_tiny_vn/best.pth")
    parser.add_argument("--student_init", type=str, default="weights/ocr/experiments/phase4/control_seed42/best.pth")
    parser.add_argument("--lambda_kd", type=float, default=0.25)
    parser.add_argument("--conf_threshold", type=float, default=0.85)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    train_kd_experiment(
        exp_name=args.exp_name,
        seed=args.seed,
        teacher_ckpt=args.teacher_ckpt,
        student_init_ckpt=args.student_init,
        lambda_kd=args.lambda_kd,
        conf_threshold=args.conf_threshold,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
