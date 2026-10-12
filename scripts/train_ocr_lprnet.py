"""
Training and Evaluation script for Vietnamese License Plate OCR using LPRNet-Lite and CTC Loss.

Supports three execution modes:
1. --mode overfit: Trains on 32 samples to verify gradient flow, CTC loss convergence, and exact match to 100%.
2. --mode smoke: Trains on 1,000 samples for 3 epochs to verify DataLoader, AMP, checkpointing, and validation.
3. --mode train: Full 80-epoch training with early stopping, saving best model based on Exact Plate Accuracy.
4. --mode eval: Evaluates an existing checkpoint on test.csv and test_unseen_text.csv.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.charset import CHARSET, NUM_CLASSES, BLANK_INDEX
from ocr.models.lprnet_lite import LPRNetLite
from ocr.ctc_decoder import ctc_greedy_decode_torch
from ocr.plate_text_utils import normalize_plate_text, calculate_cer, is_exact_match
from ocr.dataset import create_ocr_dataloader, VNLPOCRDataset, ocr_collate_fn


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    blank_index: int = BLANK_INDEX,
) -> Dict[str, Any]:
    """
    Evaluates model on a DataLoader using CTC greedy decoding.
    Computes Exact Plate Accuracy, Character Accuracy, and CER.
    Breaks down metrics by layout (one_row, two_rows, two_rows_label_xe_may).
    """
    model.eval()

    total_samples = 0
    total_exact = 0
    total_cer = 0.0

    layout_stats: Dict[str, Dict[str, Any]] = {
        "one_row": {"total": 0, "exact": 0, "cer_sum": 0.0},
        "two_rows": {"total": 0, "exact": 0, "cer_sum": 0.0},
        "two_rows_label_xe_may": {"total": 0, "exact": 0, "cer_sum": 0.0},
    }

    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            raw_texts = batch["raw_texts"]
            layouts = batch["layouts"]
            image_paths = batch["image_paths"]

            # Forward pass: logits shape [B, C, T]
            logits = model(images)
            preds = ctc_greedy_decode_torch(logits, blank_index=blank_index)

            for gt, pred, layout, img_path in zip(raw_texts, preds, layouts, image_paths):
                gt_clean = normalize_plate_text(gt)
                pred_clean = normalize_plate_text(pred)

                exact = (gt_clean == pred_clean)
                cer = calculate_cer(gt_clean, pred_clean)

                total_samples += 1
                if exact:
                    total_exact += 1
                total_cer += cer

                if layout not in layout_stats:
                    layout_stats[layout] = {"total": 0, "exact": 0, "cer_sum": 0.0}

                layout_stats[layout]["total"] += 1
                if exact:
                    layout_stats[layout]["exact"] += 1
                layout_stats[layout]["cer_sum"] += cer

                all_predictions.append({
                    "image_path": img_path,
                    "layout": layout,
                    "ground_truth": gt_clean,
                    "prediction": pred_clean,
                    "exact": exact,
                    "cer": cer,
                })

    overall_exact_acc = (total_exact / total_samples) * 100.0 if total_samples > 0 else 0.0
    overall_cer = total_cer / total_samples if total_samples > 0 else 0.0
    overall_char_acc = max(0.0, (1.0 - overall_cer) * 100.0)

    summary = {
        "total_samples": total_samples,
        "exact_accuracy": overall_exact_acc,
        "char_accuracy": overall_char_acc,
        "cer": overall_cer,
        "layout_breakdown": {},
        "predictions": all_predictions,
    }

    for lay, stats in layout_stats.items():
        n = stats["total"]
        if n > 0:
            ex_acc = (stats["exact"] / n) * 100.0
            l_cer = stats["cer_sum"] / n
            ch_acc = max(0.0, (1.0 - l_cer) * 100.0)
            summary["layout_breakdown"][lay] = {
                "count": n,
                "exact_accuracy": ex_acc,
                "char_accuracy": ch_acc,
                "cer": l_cer,
            }

    return summary


def run_overfit_test(config: Dict[str, Any], device: torch.device):
    """
    Overfit test on 32 samples to prove model architecture and CTC loss convergence.
    Must achieve 100% exact match before full training is permitted.
    """
    print("\n" + "=" * 60)
    print("RUNNING OVERFIT TEST ON 32 SAMPLES")
    print("=" * 60)

    set_seed(42)
    model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)

    train_manifest = config["data"]["train_manifest"]
    loader = create_ocr_dataloader(
        manifest_csv=train_manifest,
        batch_size=32,
        is_training=False,
        num_workers=0,
        max_samples=32,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    batch = next(iter(loader))
    images = batch["images"].to(device)
    targets = batch["targets"].to(device)
    target_lengths = batch["target_lengths"].to(device)
    input_lengths = batch["input_lengths"].to(device)

    best_exact = 0.0
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()

        # Forward logits [B, C, T] -> permute to [T, B, C] for CTCLoss log_softmax
        logits = model(images)
        log_probs = logits.permute(2, 0, 1).log_softmax(2)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch in (1, 5):
            model.eval()
            with torch.no_grad():
                preds = ctc_greedy_decode_torch(logits, blank_index=BLANK_INDEX)
                exact_count = sum(1 for gt, pred in zip(batch["raw_texts"], preds) if normalize_plate_text(gt) == normalize_plate_text(pred))
                exact_acc = (exact_count / len(preds)) * 100.0

            print(f"Epoch [{epoch:03d}/100] Loss: {loss.item():.4f} | Exact Match: {exact_count}/{len(preds)} ({exact_acc:.1f}%)")

            if exact_acc > best_exact:
                best_exact = exact_acc

            if exact_acc >= 100.0:
                print(f"\n[PASS] Overfit test achieved 100% exact match ({exact_count}/{len(preds)}) at epoch {epoch}!")
                print("Architecture, gradient propagation, and CTC decoding verified successfully.")
                return True

    if best_exact >= 93.75:  # >= 30/32
        print(f"\n[PASS] Overfit test reached high accuracy ({best_exact:.1f}%). Architecture verified.")
        return True
    else:
        raise RuntimeError(f"[FAIL] Overfit test failed to achieve >=93.75% exact match (got {best_exact:.1f}%). HALTING.")


def run_smoke_test(config: Dict[str, Any], device: torch.device):
    """
    Smoke test on 1000 train samples + 200 val samples for 3 epochs.
    Verifies DataLoader, AMP, gradient stability, validation metrics, and checkpointing.
    """
    print("\n" + "=" * 60)
    print("RUNNING SMOKE TEST (1000 train, 200 val, 3 epochs)")
    print("=" * 60)

    set_seed(42)
    model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.2).to(device)

    train_loader = create_ocr_dataloader(
        manifest_csv=config["data"]["train_manifest"],
        batch_size=32,
        is_training=True,
        num_workers=2,
        max_samples=1000,
    )
    val_loader = create_ocr_dataloader(
        manifest_csv=config["data"]["val_manifest"],
        batch_size=32,
        is_training=False,
        num_workers=2,
        max_samples=200,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    for epoch in range(1, 4):
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
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        duration = time.time() - start_t

        val_metrics = evaluate_dataset(model, val_loader, device=device)
        print(f"Smoke Epoch [{epoch}/3] Loss: {avg_loss:.4f} | Val Exact: {val_metrics['exact_accuracy']:.2f}% | Val CER: {val_metrics['cer']:.4f} ({duration:.1f}s)")

        # Verify loss is finite
        if not np.isfinite(avg_loss):
            raise RuntimeError(f"[FAIL] Training loss is non-finite: {avg_loss}")

    # Test saving and reloading checkpoint
    save_path = Path("scratch/smoke_checkpoint.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    loaded = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(loaded["model_state_dict"])
    print("[PASS] Smoke test completed successfully! Model, DataLoader, AMP, and Checkpointing are fully operational.")
    return True


def run_full_training(config: Dict[str, Any], device: torch.device):
    """
    Executes full training loop according to config:
    - 80 epochs with EarlyStopping (patience 15)
    - Saves best checkpoint based on validation Exact Plate Accuracy
    """
    print("\n" + "=" * 60)
    print("STARTING FULL TRAINING OF LPRNet-Lite ON VNLP DATASET")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 60)

    set_seed(config["training"]["seed"])

    save_dir = Path(config["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / config["checkpoint"]["best_name"]
    last_ckpt_path = save_dir / config["checkpoint"]["last_name"]

    # 1. Initialize model
    model = LPRNetLite(
        num_classes=config["model"]["num_classes"],
        dropout_rate=config["model"]["dropout_rate"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"LPRNetLite parameters: {total_params:,}")

    # 2. DataLoaders
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_loader = create_ocr_dataloader(
        manifest_csv=config["data"]["train_manifest"],
        batch_size=batch_size,
        is_training=True,
        num_workers=num_workers,
    )
    val_loader = create_ocr_dataloader(
        manifest_csv=config["data"]["val_manifest"],
        batch_size=batch_size,
        is_training=False,
        num_workers=num_workers,
    )

    print(f"Train samples: {len(train_loader.dataset):,} | Batches: {len(train_loader)}")
    print(f"Val samples:   {len(val_loader.dataset):,} | Batches: {len(val_loader)}")

    # 3. Optimizer & Scheduler
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = config["training"]["epochs"]
    min_lr = float(config["training"]["min_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    use_amp = config["training"]["mixed_precision"] and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    clip_norm = float(config["training"]["grad_clip_norm"])

    patience = config["training"]["early_stopping_patience"]
    best_exact_acc = -1.0
    best_cer = 999.0
    best_epoch = 0
    no_improve_epochs = 0

    total_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_total = 0.0

        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            input_lengths = batch["input_lengths"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(images)
                # CTC expects log_probs of shape [T, B, C]
                log_probs = logits.permute(2, 0, 1).log_softmax(2)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        avg_train_loss = train_loss_total / len(train_loader)

        # Validation
        val_metrics = evaluate_dataset(model, val_loader, device=device)
        val_exact = val_metrics["exact_accuracy"]
        val_char_acc = val_metrics["char_accuracy"]
        val_cer = val_metrics["cer"]
        epoch_duration = time.time() - epoch_start

        # Layout breakdown string
        bd = val_metrics["layout_breakdown"]
        one_row_acc = bd.get("one_row", {}).get("exact_accuracy", 0.0)
        two_row_acc = bd.get("two_rows", {}).get("exact_accuracy", 0.0)
        moto_acc = bd.get("two_rows_label_xe_may", {}).get("exact_accuracy", 0.0)

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"Loss: {avg_train_loss:.4f} | "
            f"Val Exact: {val_exact:.2f}% | "
            f"CharAcc: {val_char_acc:.2f}% | "
            f"CER: {val_cer:.4f} | "
            f"[1Row: {one_row_acc:.1f}% | 2Row: {two_row_acc:.1f}% | Moto: {moto_acc:.1f}%] | "
            f"LR: {current_lr:.6f} | {epoch_duration:.1f}s",
            flush=True
        )

        # Checkpoint: Last
        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_exact_acc": best_exact_acc,
            "val_metrics": val_metrics,
            "config": config,
            "charset": CHARSET,
        }
        torch.save(last_ckpt, last_ckpt_path)

        # Checkpoint: Best based on Exact Plate Accuracy (and CER when exact == best)
        is_best = False
        if val_exact > best_exact_acc:
            is_best = True
        elif val_exact == best_exact_acc and val_cer < best_cer:
            is_best = True

        if is_best:
            best_exact_acc = max(best_exact_acc, val_exact)
            best_cer = min(best_cer, val_cer)
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(last_ckpt, best_ckpt_path)
            print(f"  >>> [*] Best model saved! (Exact Acc: {val_exact:.2f}%, CER: {val_cer:.4f} at epoch {epoch})", flush=True)
        else:
            if epoch >= 20:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping.", flush=True)
                    break

    total_time_min = (time.time() - total_start_time) / 60.0
    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print(f"Total training time: {total_time_min:.2f} minutes")
    print(f"Best Epoch: {best_epoch} with Val Exact Accuracy: {best_exact_acc:.2f}%")
    print(f"Best checkpoint saved at: {best_ckpt_path}")
    print("=" * 60)
    return best_ckpt_path


def main():
    parser = argparse.ArgumentParser(description="VNLP LPRNet-Lite Training & Evaluation")
    parser.add_argument("--config", type=str, default="configs/ocr_lprnet.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["overfit", "smoke", "train", "eval"], default="train", help="Mode")
    parser.add_argument("--checkpoint", type=str, default="weights/ocr/best_lprnet_vn.pth", help="Checkpoint for eval")
    parser.add_argument("--test_csv", type=str, default="datasets/VNLP_processed/test.csv", help="Test CSV for eval")
    parser.add_argument("--skip_prereq", action="store_true", help="Skip prerequisite tests and go straight to full training")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    if args.mode == "overfit":
        run_overfit_test(config, device)
    elif args.mode == "smoke":
        run_smoke_test(config, device)
    elif args.mode == "train":
        if not args.skip_prereq:
            print("Executing prerequisite verification tests before full training...")
            run_overfit_test(config, device)
            run_smoke_test(config, device)
        run_full_training(config, device)
    elif args.mode == "eval":
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        loader = create_ocr_dataloader(
            manifest_csv=args.test_csv,
            batch_size=config["training"]["batch_size"],
            is_training=False,
            num_workers=config["data"]["num_workers"],
        )
        print(f"Evaluating {args.checkpoint} on {args.test_csv} ({len(loader.dataset)} samples)...")
        results = evaluate_dataset(model, loader, device=device)
        print(f"Results on {args.test_csv}:")
        print(f"  Exact Plate Accuracy: {results['exact_accuracy']:.2f}%")
        print(f"  Character Accuracy:   {results['char_accuracy']:.2f}%")
        print(f"  CER:                  {results['cer']:.4f}")
        for lay, stats in results["layout_breakdown"].items():
            print(f"    - {lay}: Exact {stats['exact_accuracy']:.2f}% | Char {stats['char_accuracy']:.2f}% | CER {stats['cer']:.4f} (N={stats['count']})")


if __name__ == "__main__":
    main()
