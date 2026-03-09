"""Evaluate trained model — compute mIoU and Dice per task on test set."""

import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CANONICAL_PROMPTS,
    CHECKPOINT_DIR,
    DEVICE,
    IMAGE_SIZE,
    THRESHOLD,
    seed_everything,
)
from data.dataset import get_dataloader
from model.clipseg_finetune import CLIPSegFinetune


def compute_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = THRESHOLD):
    """Compute Dice and IoU for a single prediction-target pair."""
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = target.astype(np.float32)

    intersection = (pred_bin * target_bin).sum()
    dice = (2.0 * intersection) / (pred_bin.sum() + target_bin.sum() + 1e-8)
    union = pred_bin.sum() + target_bin.sum() - intersection
    iou = intersection / (union + 1e-8)

    return {"dice": dice, "iou": iou}


def evaluate(threshold: float = THRESHOLD):
    seed_everything()

    # Load model
    model = CLIPSegFinetune().to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[error] No checkpoint found at {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[eval] Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_dice={ckpt['val_dice']:.4f})")

    # Evaluate per task
    results = {}
    for task in ["crack", "taping"]:
        test_loader = get_dataloader("test", batch_size=BATCH_SIZE, task_filter=task)
        if len(test_loader.dataset) == 0:
            print(f"[warn] No test data for {task}")
            continue

        dices, ious = [], []

        with torch.no_grad():
            for images, prompts, masks, meta in tqdm(test_loader, desc=f"Eval {task}"):
                images = images.to(DEVICE)
                logits = model(images, list(prompts))
                preds = torch.sigmoid(logits).cpu().numpy()
                masks_np = masks.numpy()

                for i in range(preds.shape[0]):
                    m = compute_metrics(preds[i], masks_np[i], threshold)
                    dices.append(m["dice"])
                    ious.append(m["iou"])

        results[task] = {
            "dice_mean": np.mean(dices),
            "dice_std": np.std(dices),
            "iou_mean": np.mean(ious),
            "iou_std": np.std(ious),
            "n_samples": len(dices),
        }

    # Print results table
    print("\n" + "=" * 60)
    print(f"{'Task':<12} {'Dice':>14} {'mIoU':>14} {'N':>6}")
    print("-" * 60)
    all_dice, all_iou = [], []
    for task, r in results.items():
        print(f"{task:<12} {r['dice_mean']:.4f}±{r['dice_std']:.4f}"
              f"  {r['iou_mean']:.4f}±{r['iou_std']:.4f}  {r['n_samples']:>6}")
        all_dice.append(r["dice_mean"])
        all_iou.append(r["iou_mean"])

    if all_dice:
        print("-" * 60)
        print(f"{'Combined':<12} {np.mean(all_dice):.4f}"
              f"{'':>8} {np.mean(all_iou):.4f}")
    print("=" * 60)

    # Optional: threshold sweep
    print("\n[threshold sweep]")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"  threshold={t:.1f}: (run with --threshold {t} for full results)")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    evaluate(threshold=args.threshold)
