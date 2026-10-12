"""
Phase 3 Fine-Tuning Pipeline for LPRNet-Lite (PyTorch Only).
Runs Controlled Fine-Tuning from Checkpoint V1 (best_lprnet_vn.pth) on VNLP + PBL4 Unique.

Supports:
  - FT-A: lr=1e-4, epochs=30, patience=10, eta_min=1e-6
  - FT-B: lr=5e-5, epochs=40, patience=10, eta_min=1e-6
  - FT-C: lr=1e-5, epochs=30, patience=10, eta_min=1e-6 (Optional)

Tracks:
  - VAL Exact Accuracy, Char Accuracy, CER, Avg Edit Distance
  - Breakdown: one_row, two_rows, motorcycle
  - Catastrophic Forgetting warnings
  - Outputs epoch logs to CSV
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.charset import CHARSET, NUM_CLASSES, BLANK_INDEX
from ocr.models.lprnet_lite import LPRNetLite
from ocr.dataset import create_ocr_dataloader
from ocr.ctc_decoder import ctc_greedy_decode_torch
from ocr.plate_text_utils import normalize_plate_text, calculate_cer, levenshtein_distance
from tools.evaluate_ocr_model import evaluate_manifest


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(
    exp_name: str,
    v1_ckpt_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    log_csv_path: Path,
    lr: float = 1e-4,
    max_epochs: int = 30,
    patience: int = 10,
    eta_min: float = 1e-6,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    seed: int = 42,
    num_workers: int = 4,
):
    print("\n" + "=" * 75)
    print(f"PHASE 3 FINE-TUNING EXPERIMENT: {exp_name}")
    print(f"Pretrained Base: {v1_ckpt_path}")
    print(f"Train Manifest:  {train_manifest}")
    print(f"Val Manifest:    {val_manifest}")
    print(f"Hyperparameters: LR={lr}, MaxEpochs={max_epochs}, Patience={patience}, EtaMin={eta_min}, BatchSize={batch_size}")
    print("=" * 75)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "best.pth"
    last_ckpt_path = output_dir / "last.pth"

    # 1. Initialize model and load V1 weights
    model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
    ckpt = torch.load(v1_ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch! Missing: {missing}, Unexpected: {unexpected}")
    print(f"Loaded V1 weights successfully! (Missing: 0, Unexpected: 0)")

    # 2. Pre-training verification on VAL
    print("\nEvaluating initial model on VAL before fine-tuning...")
    init_val_metrics, _ = evaluate_manifest(model, str(val_manifest), device=device, batch_size=batch_size, num_workers=num_workers)
    v1_val_exact = init_val_metrics["exact_accuracy"]
    v1_val_cer = init_val_metrics["cer"]
    v1_one_row = init_val_metrics["layout_breakdown"]["one_row"]["exact_accuracy"]
    v1_two_rows = init_val_metrics["layout_breakdown"]["two_rows"]["exact_accuracy"]
    v1_moto = init_val_metrics["layout_breakdown"]["two_rows_label_xe_may"]["exact_accuracy"]

    print(f"Baseline V1 VAL: Exact={v1_val_exact:.2f}% | 1Row={v1_one_row:.2f}% | 2Row={v1_two_rows:.2f}% | Moto={v1_moto:.2f}% | CER={v1_val_cer:.4f}")

    # 3. DataLoaders
    train_loader = create_ocr_dataloader(
        manifest_csv=str(train_manifest),
        batch_size=batch_size,
        is_training=True,
        num_workers=num_workers,
    )
    val_loader_eval_path = str(val_manifest)

    # 4. Optimizer, Criterion, Scheduler, Scaler
    criterion = nn.CTCLoss(blank=BLANK_INDEX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_val_exact = v1_val_exact
    best_val_cer = v1_val_cer
    best_epoch = 0
    epochs_no_improve = 0

    # Save initial baseline as initial best if desired
    torch.save({
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "best_exact_acc": best_val_exact,
        "val_metrics": init_val_metrics,
        "experiment": exp_name,
        "charset": CHARSET,
    }, best_ckpt_path)

    log_records = [{
        "epoch": 0,
        "train_loss": 0.0,
        "val_exact": v1_val_exact,
        "val_char_acc": init_val_metrics["char_accuracy"],
        "val_cer": v1_val_cer,
        "one_row_acc": v1_one_row,
        "two_rows_acc": v1_two_rows,
        "motorcycle_acc": v1_moto,
        "learning_rate": lr,
        "status": "V1_INITIAL_BASELINE",
    }]

    print("\nStarting Fine-tuning loop...")
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
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_dur = time.time() - start_t

        # Validation evaluation
        val_m, _ = evaluate_manifest(model, val_loader_eval_path, device=device, batch_size=batch_size, num_workers=num_workers)
        vex = val_m["exact_accuracy"]
        vca = val_m["char_accuracy"]
        vcer = val_m["cer"]
        r1 = val_m["layout_breakdown"]["one_row"]["exact_accuracy"]
        r2 = val_m["layout_breakdown"]["two_rows"]["exact_accuracy"]
        rm = val_m["layout_breakdown"]["two_rows_label_xe_may"]["exact_accuracy"]

        # Catastrophic forgetting check
        warnings = []
        if rm < v1_moto - 3.0:
            warnings.append(f"[WARN: Moto dropped {v1_moto - rm:.2f}pp]")
        if r2 < v1_two_rows - 3.0:
            warnings.append(f"[WARN: 2Row dropped {v1_two_rows - r2:.2f}pp]")
        warn_str = " ".join(warnings)

        print(f"Epoch [{epoch:02d}/{max_epochs:02d}] Loss: {avg_loss:.4f} | Val Exact: {vex:.2f}% | CharAcc: {vca:.2f}% | CER: {vcer:.4f} | [1Row: {r1:.1f}% | 2Row: {r2:.1f}% | Moto: {rm:.1f}%] | LR: {current_lr:.6f} | {epoch_dur:.1f}s {warn_str}")

        # Model selection: Primary = Val Exact, Secondary = CER
        is_improved = False
        if vex > best_val_exact + 0.05:
            is_improved = True
        elif abs(vex - best_val_exact) <= 0.05 and vcer < best_val_cer - 0.0005:
            is_improved = True

        status_tag = ""
        if is_improved:
            best_val_exact = vex
            best_val_cer = vcer
            best_epoch = epoch
            epochs_no_improve = 0
            status_tag = "BEST"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_exact_acc": best_val_exact,
                "val_metrics": val_m,
                "experiment": exp_name,
                "charset": CHARSET,
            }, best_ckpt_path)
            print(f"  >>> [*] Best model updated! (Val Exact: {best_val_exact:.2f}%, CER: {best_val_cer:.4f} at epoch {epoch})")
        else:
            epochs_no_improve += 1

        # Always save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "last_val_exact": vex,
            "val_metrics": val_m,
            "experiment": exp_name,
            "charset": CHARSET,
        }, last_ckpt_path)

        log_records.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_exact": round(vex, 2),
            "val_char_acc": round(vca, 2),
            "val_cer": round(vcer, 4),
            "one_row_acc": round(r1, 2),
            "two_rows_acc": round(r2, 2),
            "motorcycle_acc": round(rm, 2),
            "learning_rate": current_lr,
            "status": status_tag,
        })

        if epochs_no_improve >= patience:
            print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping at epoch {epoch}.")
            break

    # Write log CSV
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["epoch", "train_loss", "val_exact", "val_char_acc", "val_cer", "one_row_acc", "two_rows_acc", "motorcycle_acc", "learning_rate", "status"])
        writer.writeheader()
        writer.writerows(log_records)
    print(f"Saved training log to {log_csv_path}")

    print("\n" + "=" * 60)
    print(f"EXPERIMENT {exp_name} COMPLETE")
    print(f"Best Epoch: {best_epoch} with Val Exact: {best_val_exact:.2f}% (Baseline V1: {v1_val_exact:.2f}%)")
    print(f"Best Checkpoint: {best_ckpt_path}")
    print("=" * 60)

    return {
        "exp_name": exp_name,
        "best_epoch": best_epoch,
        "best_val_exact": best_val_exact,
        "best_val_cer": best_val_cer,
        "best_ckpt_path": best_ckpt_path,
        "v1_val_exact": v1_val_exact,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Fine-tuning Runner")
    parser.add_argument("--exp", type=str, choices=["FT_A", "FT_B", "FT_C"], required=True, help="Experiment name")
    parser.add_argument("--v1_ckpt", type=str, default="weights/ocr/best_lprnet_vn.pth")
    parser.add_argument("--train_manifest", type=str, default="datasets/VNLP_processed/train_v2a.csv")
    parser.add_argument("--val_manifest", type=str, default="datasets/VNLP_processed/val.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    v1_ckpt = Path(args.v1_ckpt)
    train_manifest = Path(args.train_manifest)
    val_manifest = Path(args.val_manifest)

    if args.exp == "FT_A":
        run_experiment(
            exp_name="PHASE3_FT_A",
            v1_ckpt_path=v1_ckpt,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            output_dir=Path("weights/ocr/experiments/phase3_ft_a"),
            log_csv_path=Path("reports/phase3_ft_a_epochs.csv"),
            lr=1e-4,
            max_epochs=30,
            patience=10,
            eta_min=1e-6,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.exp == "FT_B":
        run_experiment(
            exp_name="PHASE3_FT_B",
            v1_ckpt_path=v1_ckpt,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            output_dir=Path("weights/ocr/experiments/phase3_ft_b"),
            log_csv_path=Path("reports/phase3_ft_b_epochs.csv"),
            lr=5e-5,
            max_epochs=40,
            patience=10,
            eta_min=5e-7,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.exp == "FT_C":
        run_experiment(
            exp_name="PHASE3_FT_C",
            v1_ckpt_path=v1_ckpt,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            output_dir=Path("weights/ocr/experiments/phase3_ft_c"),
            log_csv_path=Path("reports/phase3_ft_c_epochs.csv"),
            lr=1e-5,
            max_epochs=30,
            patience=10,
            eta_min=1e-6,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
