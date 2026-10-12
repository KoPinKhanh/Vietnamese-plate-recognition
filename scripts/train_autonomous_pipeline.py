"""
Autonomous AI Improvement Pipeline: Phase 4.5 -> Phase 5 -> Phase 6 -> Optional Phase 7.
Fully automated research pipeline for Vietnamese License Plate OCR.
Enforces:
  - Model selection strictly based on VALIDATION (val.csv, val_unseen_text_v2.csv)
  - TEST opened only once at the end after locking candidate
  - Registry-based resume/restart safety (reports/AUTO_EXPERIMENT_REGISTRY.csv)
  - No ONNX export, no Raspberry Pi, pure PyTorch .pth
"""

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.charset import CHARSET, NUM_CLASSES, BLANK_INDEX, CHAR_TO_IDX, encode_text, decode_indices
from ocr.models.lprnet_lite import LPRNetLite
from ocr.preprocess import (
    crop_plate_gt,
    rectify_two_row_plate,
    letterbox_ocr,
    normalize_image_for_model,
)
from tools.evaluate_ocr_model import evaluate_manifest


REGISTRY_PATH = Path("reports/AUTO_EXPERIMENT_REGISTRY.csv")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EnhancedOCRAugmentor:
    """
    Targeted OCR Augmentations addressing real-world failure modes:
      - Left/right crop jitter and padding (left border robustness)
      - Mild perspective transform
      - Mild rotation (-3.5 to +3.5 deg)
      - Motion and defocus blur
      - Contrast, brightness, gamma
      - Mild JPEG compression artifacts
    """
    def __init__(self, p: float = 0.6, geom_p: float = 0.4):
        self.p = p
        self.geom_p = geom_p

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img_bgr

        img = img_bgr.copy()
        h, w = img.shape[:2]

        # 1. Left/Right border padding jitter (prevents dropping edge characters like '3' in '36')
        if random.random() < 0.4:
            pad_left = random.randint(2, 10)
            pad_right = random.randint(2, 10)
            pad_top = random.randint(1, 4)
            pad_bottom = random.randint(1, 4)
            img = cv2.copyMakeBorder(
                img, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            h, w = img.shape[:2]

        # 2. Mild perspective / homography
        if random.random() < self.geom_p:
            dx = random.uniform(0, 0.05 * w)
            dy = random.uniform(0, 0.06 * h)
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([
                [random.uniform(0, dx), random.uniform(0, dy)],
                [w - random.uniform(0, dx), random.uniform(0, dy)],
                [random.uniform(0, dx), h - random.uniform(0, dy)],
                [w - random.uniform(0, dx), h - random.uniform(0, dy)],
            ])
            M_p = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img, M_p, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 3. Mild rotation (-3.0 to +3.0 degrees)
        if random.random() < 0.35:
            angle = random.uniform(-3.0, 3.0)
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # 4. Brightness, Contrast, Gamma
        if random.random() < 0.45:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-18, 18)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # 5. Motion blur or defocus blur
        if random.random() < 0.3:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 6. JPEG compression
        if random.random() < 0.35:
            q = random.randint(50, 85)
            _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            img = cv2.imdecode(enc, 1)

        return img


class AdvancedGeometryAugmentor:
    """Phase 6 Geometric Augmentor: heavier perspective, keystone, border dilation."""
    def __init__(self, p: float = 0.7):
        self.p = p

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img_bgr

        img = img_bgr.copy()
        h, w = img.shape[:2]

        # Always add border buffer
        pad_l = random.randint(4, 14)
        pad_r = random.randint(4, 14)
        img = cv2.copyMakeBorder(img, 2, 2, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        h, w = img.shape[:2]

        # Perspective & Keystone
        dx = random.uniform(0.02 * w, 0.08 * w)
        dy = random.uniform(0.02 * h, 0.08 * h)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.uniform(0, dx), random.uniform(0, dy)],
            [w - random.uniform(0, dx), random.uniform(0, dy)],
            [random.uniform(0, dx), h - random.uniform(0, dy)],
            [w - random.uniform(0, dx), h - random.uniform(0, dy)],
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # Mild blur
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        return img


class AutonomousOCRDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        target_size: Tuple[int, int] = (160, 48),
        is_training: bool = False,
        augmentor_type: str = "default",  # "none", "default", "targeted", "geometry"
    ):
        self.manifest_path = Path(manifest_csv)
        self.target_size = target_size
        self.is_training = is_training

        if augmentor_type == "targeted":
            self.augmentor = EnhancedOCRAugmentor(p=0.6, geom_p=0.4) if is_training else None
        elif augmentor_type == "geometry":
            self.augmentor = AdvancedGeometryAugmentor(p=0.7) if is_training else None
        elif augmentor_type == "default":
            from ocr.dataset import OCRAugmentor
            self.augmentor = OCRAugmentor(p=0.5) if is_training else None
        else:
            self.augmentor = None

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
            crop = crop_plate_gt(img_bgr, item["bbox"], pad_ratio=0.03)
        else:
            crop = img_bgr

        # 2. Rectify two-row plates
        if item["layout"] in ("two_rows", "two_rows_label_xe_may"):
            crop, _, _ = rectify_two_row_plate(crop, target_h=self.target_size[1])

        # 3. Augmentation
        if self.augmentor is not None:
            crop = self.augmentor(crop)

        # 4. Letterbox to (160, 48)
        letterboxed = letterbox_ocr(crop, target_size=self.target_size, pad_color=(255, 255, 255))
        normalized = normalize_image_for_model(letterboxed)
        img_tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        encoded = encode_text(item["plate_text"])

        return {
            "image": img_tensor,
            "label": torch.tensor(encoded, dtype=torch.long),
            "label_len": len(encoded),
            "raw_text": item["plate_text"],
            "layout": item["layout"],
            "image_path": item["image_path"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    target_lengths = torch.tensor([item["label_len"] for item in batch], dtype=torch.long)
    targets = torch.cat([item["label"] for item in batch], dim=0)
    input_lengths = torch.full((len(batch),), 40, dtype=torch.long)
    raw_texts = [item["raw_text"] for item in batch]
    layouts = [item["layout"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "input_lengths": input_lengths,
        "raw_texts": raw_texts,
        "layouts": layouts,
        "image_paths": image_paths,
    }


def load_registry() -> Dict[str, Dict[str, Any]]:
    reg = {}
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                reg[r["experiment_id"]] = r
    return reg


def update_registry(row: Dict[str, Any]):
    reg = load_registry()
    reg[row["experiment_id"]] = row
    fieldnames = [
        "experiment_id", "phase", "intervention", "seed", "base_checkpoint",
        "dataset", "status", "best_epoch", "val_exact", "val_unseen",
        "one_row", "two_rows", "motorcycle", "char_acc", "cer",
        "checkpoint_path", "decision", "notes"
    ]
    with open(REGISTRY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in sorted(reg.keys()):
            writer.writerow(reg[k])


def train_single_experiment(
    exp_id: str,
    phase: str,
    intervention: str,
    seed: int,
    base_ckpt_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    val_unseen_manifest: Path,
    output_dir: Path,
    augmentor_type: str = "default",
    sampler_type: str = "none",  # "none", "hard_mining", "balanced_layout"
    hardness_csv: Optional[Path] = None,
    lr: float = 2e-5,
    max_epochs: int = 25,
    patience: int = 8,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print(f"PIPELINE RUN: {exp_id} | Phase: {phase} | Intervention: {intervention}")
    print(f"Base: {base_ckpt_path.name} | Seed: {seed} | Augmentor: {augmentor_type} | Sampler: {sampler_type}")
    print(f"Output: {output_dir}")
    print("=" * 75)

    update_registry({
        "experiment_id": exp_id,
        "phase": phase,
        "intervention": intervention,
        "seed": seed,
        "base_checkpoint": str(base_ckpt_path).replace("\\", "/"),
        "dataset": train_manifest.name,
        "status": "RUNNING",
        "best_epoch": 0,
        "val_exact": 0.0,
        "val_unseen": 0.0,
        "one_row": 0.0,
        "two_rows": 0.0,
        "motorcycle": 0.0,
        "char_acc": 0.0,
        "cer": 0.0,
        "checkpoint_path": "",
        "decision": "PENDING",
        "notes": f"Started {time.strftime('%Y-%m-%d %H:%M:%S')}",
    })

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "best.pth"

    # Preload unseen paths
    unseen_paths_set = set()
    with open(val_unseen_manifest, "r", encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            unseen_paths_set.add(r["image_path"])

    # Load Base Model
    model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
    ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    # Initial Validation
    init_val_m, init_records = evaluate_manifest(model, str(val_manifest), device=device, batch_size=128, num_workers=num_workers)
    init_unseen = [r for r in init_records if r["image_path"] in unseen_paths_set]
    init_uex = (sum(1 for r in init_unseen if r["exact"]) / len(init_unseen)) * 100.0 if init_unseen else 0.0

    best_val_exact = init_val_m["exact_accuracy"]
    best_val_cer = init_val_m["cer"]
    best_unseen_exact = init_uex
    best_metrics = init_val_m
    best_epoch = 0
    epochs_no_improve = 0

    # Save initial model as default best
    torch.save({
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "best_exact_acc": best_val_exact,
        "val_metrics": init_val_m,
        "val_unseen_exact": best_unseen_exact,
        "experiment_id": exp_id,
        "seed": seed,
        "charset": CHARSET,
    }, best_ckpt_path)

    # Build Dataset
    train_dataset = AutonomousOCRDataset(
        manifest_csv=str(train_manifest),
        is_training=True,
        augmentor_type=augmentor_type,
    )

    # Build Sampler
    sampler = None
    shuffle = True
    if sampler_type == "hard_mining" and hardness_csv and hardness_csv.exists():
        # Load hardness tiers
        tier_weights = {"HARD": 3.0, "MEDIUM": 1.5, "EASY": 0.5}
        sample_weights = []
        with open(hardness_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                w = tier_weights.get(r["tier"], 1.0)
                sample_weights.append(w)
        # Ensure length matches
        if len(sample_weights) == len(train_dataset):
            weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
            sampler = WeightedRandomSampler(weights_tensor, num_samples=len(train_dataset), replacement=True)
            shuffle = False
            print(f"  [Sampler] Using Hard Example WeightedRandomSampler (HARD=3.0, MED=1.5, EASY=0.5)")

    elif sampler_type == "balanced_layout":
        layout_weights = {"two_rows_label_xe_may": 3.4, "two_rows": 1.5, "one_row": 1.0}
        sample_weights = []
        for rec in train_dataset.records:
            w = layout_weights.get(rec["layout"], 1.0)
            sample_weights.append(w)
        weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights_tensor, num_samples=len(train_dataset), replacement=True)
        shuffle = False
        print(f"  [Sampler] Using Layout-Balanced WeightedRandomSampler (Moto=3.4, 2Row=1.5, 1Row=1.0)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    criterion = nn.CTCLoss(blank=BLANK_INDEX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        start_t = time.time()

        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            input_lengths = batch["input_lengths"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(images)
                log_probs = logits.permute(2, 0, 1).log_softmax(2)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        scheduler.step()
        epoch_dur = time.time() - start_t

        # Validation
        val_m, val_records = evaluate_manifest(model, str(val_manifest), device=device, batch_size=128, num_workers=num_workers)
        unseen_recs = [r for r in val_records if r["image_path"] in unseen_paths_set]
        uex = (sum(1 for r in unseen_recs if r["exact"]) / len(unseen_recs)) * 100.0 if unseen_recs else 0.0

        vex = val_m["exact_accuracy"]
        vcer = val_m["cer"]
        rm = val_m["layout_breakdown"]["two_rows_label_xe_may"]["exact_accuracy"]

        # Selection logic (Primary: Val Exact Acc, Secondary: CER)
        is_improved = False
        if vex > best_val_exact + 0.05:
            is_improved = True
        elif abs(vex - best_val_exact) <= 0.05 and vcer < best_val_cer - 0.0004:
            is_improved = True

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
                "experiment_id": exp_id,
                "seed": seed,
                "charset": CHARSET,
            }, best_ckpt_path)
            print(f"  Epoch [{epoch:02d}/{max_epochs:02d}] Loss: {avg_loss:.4f} | VAL: {vex:.2f}% (Unseen: {uex:.2f}%) | Moto: {rm:.1f}% | CER: {vcer:.4f} | {epoch_dur:.1f}s [*BEST*]")
        else:
            epochs_no_improve += 1
            print(f"  Epoch [{epoch:02d}/{max_epochs:02d}] Loss: {avg_loss:.4f} | VAL: {vex:.2f}% (Unseen: {uex:.2f}%) | Moto: {rm:.1f}% | CER: {vcer:.4f} | {epoch_dur:.1f}s")

        if epochs_no_improve >= patience:
            print(f"  [Early Stopping] No improvement for {patience} epochs. Stopping at epoch {epoch}.")
            break

    res = {
        "experiment_id": exp_id,
        "phase": phase,
        "intervention": intervention,
        "seed": seed,
        "base_checkpoint": str(base_ckpt_path).replace("\\", "/"),
        "dataset": train_manifest.name,
        "status": "COMPLETE",
        "best_epoch": best_epoch,
        "val_exact": round(best_val_exact, 2),
        "val_unseen": round(best_unseen_exact, 2),
        "one_row": round(best_metrics["layout_breakdown"]["one_row"]["exact_accuracy"], 2),
        "two_rows": round(best_metrics["layout_breakdown"]["two_rows"]["exact_accuracy"], 2),
        "motorcycle": round(best_metrics["layout_breakdown"]["two_rows_label_xe_may"]["exact_accuracy"], 2),
        "char_acc": round(best_metrics["char_accuracy"], 2),
        "cer": round(best_val_cer, 4),
        "checkpoint_path": str(best_ckpt_path).replace("\\", "/"),
        "decision": "ACCEPTED" if best_val_exact >= 86.02 else "EVALUATING",
        "notes": f"Finished best_epoch={best_epoch}",
    }
    update_registry(res)
    return res


def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Research Pipeline")
    parser.add_argument("--run_all", action="store_true", help="Run full pipeline autonomously")
    parser.add_argument("--phase", type=str, default="5", help="Target phase to run (5, 6, 7)")
    args = parser.parse_args()

    v1_ckpt = Path("weights/ocr/best_lprnet_vn.pth")
    phase4_base = Path("weights/ocr/experiments/phase4/control_seed42/best.pth")
    vnlp_train = Path("datasets/VNLP_processed/train.csv")
    synth_train = Path("datasets/VNLP_processed/train_p5_synth.csv")
    val_csv = Path("datasets/VNLP_processed/val.csv")
    val_unseen_csv = Path("datasets/VNLP_processed/val_unseen_text_v2.csv")
    hardness_csv = Path("datasets/VNLP_processed/train_hardness.csv")

    auto_dir = Path("weights/ocr/experiments/autonomous")
    auto_dir.mkdir(parents=True, exist_ok=True)

    reg = load_registry()

    # =========================================================================
    # PHASE 5 PILOTS: P5A, P5B, P5C, P5D
    # =========================================================================
    pilots = {
        "P5A_HARD_MINING": {
            "phase": "PHASE5",
            "intervention": "Hard Example Mining",
            "seed": 42,
            "manifest": vnlp_train,
            "dir": auto_dir / "p5a_hard_mining_s42",
            "aug": "default",
            "sampler": "hard_mining",
            "hardness": hardness_csv,
            "lr": 2e-5,
        },
        "P5B_TARGET_AUG": {
            "phase": "PHASE5",
            "intervention": "Targeted Real Augmentation",
            "seed": 42,
            "manifest": vnlp_train,
            "dir": auto_dir / "p5b_target_aug_s42",
            "aug": "targeted",
            "sampler": "none",
            "hardness": None,
            "lr": 2e-5,
        },
        "P5C_SYNTH_MOTO": {
            "phase": "PHASE5",
            "intervention": "Targeted Synthetic Motorcycle",
            "seed": 42,
            "manifest": synth_train,
            "dir": auto_dir / "p5c_synth_moto_s42",
            "aug": "default",
            "sampler": "none",
            "hardness": None,
            "lr": 2e-5,
        },
        "P5D_BALANCED": {
            "phase": "PHASE5",
            "intervention": "Balanced Layout Sampling",
            "seed": 42,
            "manifest": vnlp_train,
            "dir": auto_dir / "p5d_balanced_s42",
            "aug": "default",
            "sampler": "balanced_layout",
            "hardness": None,
            "lr": 2e-5,
        },
    }

    print("\n" + "=" * 75)
    print("EXECUTING PHASE 5 PILOT EXPERIMENTS")
    print("=" * 75)

    pilot_results = {}
    for p_id, p_cfg in pilots.items():
        if p_id in reg and reg[p_id]["status"] == "COMPLETE" and Path(reg[p_id]["checkpoint_path"]).exists():
            print(f"[SKIP] {p_id} already COMPLETE: Val={reg[p_id]['val_exact']}%")
            pilot_results[p_id] = reg[p_id]
        else:
            res = train_single_experiment(
                exp_id=p_id,
                phase=p_cfg["phase"],
                intervention=p_cfg["intervention"],
                seed=p_cfg["seed"],
                base_ckpt_path=phase4_base,
                train_manifest=p_cfg["manifest"],
                val_manifest=val_csv,
                val_unseen_manifest=val_unseen_csv,
                output_dir=p_cfg["dir"],
                augmentor_type=p_cfg["aug"],
                sampler_type=p_cfg["sampler"],
                hardness_csv=p_cfg["hardness"],
                lr=p_cfg["lr"],
            )
            pilot_results[p_id] = res

    # Write Phase 5 Pilot Report
    write_phase5_pilot_report(pilot_results)


def write_phase5_pilot_report(results: Dict[str, Dict[str, Any]]):
    base_val = 86.02
    base_unseen = 84.86
    base_1row = 90.57
    base_2row = 81.44
    base_moto = 80.86
    base_cer = 0.0252

    lines = [
        "# BÁO CÁO ĐỐI CHỨNG CÁC CAN THIỆP PILOT PHASE 5 (PHASE 5 PILOT COMPARISON)",
        "",
        "**Thời gian thực hiện:** 2026-09-25",
        "**Base Checkpoint:** `weights/ocr/experiments/phase4/control_seed42/best.pth` (V1 + VNLP Train)",
        "**Baseline Metrics (Phase 4 Base):** VAL Exact = **86.02%** | Unseen = **84.86%** | Moto = **80.86%** | CER = **0.0252**",
        "",
        "---",
        "",
        "## 1. Bảng So Sánh Hiệu Năng Các Can Thiệp Pilot (Đo Trên Validation)",
        "| Experiment ID | Can thiệp (Intervention) | VAL Exact (%) | Delta VAL | VAL Unseen (%) | Moto (%) | Delta Moto | 2Row (%) | CER | Đánh giá |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        f"| **BASE_P4** | Base Checkpoint Phase 4 | 86.02% | 0.00 pp | 84.86% | 80.86% | 0.00 pp | 81.44% | 0.0252 | BASELINE |",
    ]

    for p_id in ["P5A_HARD_MINING", "P5B_TARGET_AUG", "P5C_SYNTH_MOTO", "P5D_BALANCED"]:
        if p_id in results:
            r = results[p_id]
            vex = float(r["val_exact"])
            vue = float(r["val_unseen"])
            vmot = float(r["motorcycle"])
            v2r = float(r["two_rows"])
            vcer = float(r["cer"])
            d_val = vex - base_val
            d_mot = vmot - base_moto

            # Evaluate utility
            useful = False
            if d_val > 0.05 or (d_mot > 0.5 and d_val >= -0.10):
                if vue >= base_unseen - 0.40 and v2r >= base_2row - 0.40 and vcer <= base_cer + 0.0005:
                    useful = True

            eval_str = "**HỨA HẸN (SHORTLIST)**" if useful else "KHÔNG ĐẠT NGƯỠNG"
            lines.append(f"| **{p_id}** | {r['intervention']} | **{vex:.2f}%** | {d_val:+.2f} pp | {vue:.2f}% | {vmot:.2f}% | {d_mot:+.2f} pp | {v2r:.2f}% | {vcer:.4f} | {eval_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Tiêu Chí Chọn Lọc Shortlist Sang Đa Seed",
        "- Chỉ chọn tối đa **2 can thiệp** có kết quả tích cực nhất trên Validation (tăng VAL hoặc tăng mạnh motorcycle mà không làm giảm tổng thể).",
        "- Các can thiệp trong shortlist sẽ được kiểm định qua thêm 2 seeds: `1337` và `2026`.",
        "",
    ])

    out_p = Path("reports/AUTO_PHASE5_PILOT_COMPARISON.md")
    with open(out_p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_p} successfully!")


if __name__ == "__main__":
    main()
