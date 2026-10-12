"""
Hard Example Mining on VNLP Train Set.
Evaluates Phase 4 best model on train.csv (29,836 samples).
Computes exact match, edit distance, CTC loss, and confidence.
Assigns hardness score and hardness tiers (HARD, MEDIUM, EASY).
Outputs: datasets/VNLP_processed/train_hardness.csv
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ocr.charset import NUM_CLASSES, BLANK_INDEX, decode_indices
from ocr.models.lprnet_lite import LPRNetLite
from ocr.dataset import create_ocr_dataloader


def compute_levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("MINING HARD EXAMPLES ON VNLP TRAIN SET")
    print(f"Device: {device}")
    print("=" * 70)

    ckpt_path = Path("weights/ocr/experiments/phase4/control_seed42/best.pth")
    train_manifest = Path("datasets/VNLP_processed/train.csv")
    out_csv = Path("datasets/VNLP_processed/train_hardness.csv")

    model = LPRNetLite(num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    loader = create_ocr_dataloader(
        manifest_csv=str(train_manifest),
        batch_size=128,
        is_training=False,
        num_workers=4,
    )

    ctc_criterion = nn.CTCLoss(blank=BLANK_INDEX, reduction="none", zero_infinity=True)

    records = []
    total = 0
    correct_count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)
            input_lengths = batch["input_lengths"].to(device, non_blocking=True)
            raw_texts = batch["raw_texts"]
            layouts = batch["layouts"]
            image_paths = batch["image_paths"]

            logits = model(images)  # [B, 37, 40]
            log_probs = logits.permute(2, 0, 1).log_softmax(2)  # [40, B, 37]
            losses = ctc_criterion(log_probs, targets, input_lengths, target_lengths)  # [B]

            # Softmax probabilities for confidence
            probs = logits.softmax(1)  # [B, 37, 40]
            max_probs, preds = torch.max(logits, dim=1)  # [B, 40]

            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()
            losses_np = losses.cpu().numpy()

            for i in range(len(raw_texts)):
                gt = raw_texts[i]
                lay = layouts[i]
                img_p = image_paths[i]
                loss_val = float(losses_np[i])

                # CTC greedy decode
                raw_seq = preds_np[i]
                collapsed = []
                conf_list = []
                for t in range(len(raw_seq)):
                    c = raw_seq[t]
                    if c != BLANK_INDEX and (t == 0 or c != raw_seq[t - 1]):
                        collapsed.append(c)
                        conf_list.append(float(probs_np[i, c, t]))

                pred_str = decode_indices(collapsed)
                is_correct = (pred_str == gt)
                ed = compute_levenshtein(pred_str, gt)
                mean_conf = float(np.mean(conf_list)) if conf_list else 0.0

                if is_correct:
                    correct_count += 1

                # Hardness Score: combines edit distance, loss, and confidence
                # Higher score = harder example
                # Loss capped at 5.0 for stability
                loss_clamped = min(loss_val, 5.0)
                hardness_score = (ed * 2.0) + loss_clamped + (1.0 - mean_conf)

                # Tier assignment
                if not is_correct or loss_val > 0.40 or mean_conf < 0.85:
                    tier = "HARD"
                elif loss_val > 0.08 or mean_conf < 0.95 or ed > 0:
                    tier = "MEDIUM"
                else:
                    tier = "EASY"

                records.append({
                    "image_path": img_p,
                    "plate_text": gt,
                    "layout": lay,
                    "prediction": pred_str,
                    "correct": 1 if is_correct else 0,
                    "edit_distance": ed,
                    "ctc_loss": round(loss_val, 4),
                    "confidence": round(mean_conf, 4),
                    "hardness_score": round(hardness_score, 4),
                    "tier": tier,
                })
                total += 1

    print(f"Evaluated {total} training samples. Accuracy: {correct_count / total * 100.0:.2f}%")
    tier_counts = {"HARD": sum(1 for r in records if r["tier"] == "HARD"),
                   "MEDIUM": sum(1 for r in records if r["tier"] == "MEDIUM"),
                   "EASY": sum(1 for r in records if r["tier"] == "EASY")}
    print(f"Tier Breakdown: HARD={tier_counts['HARD']} ({tier_counts['HARD']/total*100:.1f}%), "
          f"MEDIUM={tier_counts['MEDIUM']} ({tier_counts['MEDIUM']/total*100:.1f}%), "
          f"EASY={tier_counts['EASY']} ({tier_counts['EASY']/total*100:.1f}%)")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "plate_text", "layout", "prediction", "correct",
            "edit_distance", "ctc_loss", "confidence", "hardness_score", "tier"
        ])
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {out_csv} successfully!")


if __name__ == "__main__":
    main()
