"""Joint training loop for CLIPSeg on crack + taping tasks."""

import os
import sys
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    BCE_WEIGHT,
    CHECKPOINT_DIR,
    CRACK_POS_WEIGHT,
    DEVICE,
    DICE_WEIGHT,
    EPOCHS,
    GRAD_CLIP,
    LEARNING_RATE,
    PATIENCE,
    WEIGHT_DECAY,
    seed_everything,
)
from data.dataset import get_dataloader
from model.clipseg_finetune import CLIPSegFinetune
from model.loss import CombinedLoss


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    return (2.0 * intersection / (pred_bin.sum() + target.sum() + 1e-8)).item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection / (union + 1e-8)).item()


def train():
    seed_everything()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"[train] Device: {DEVICE}")
    print(f"[train] Epochs: {EPOCHS}, BS: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    # Data
    train_loader = get_dataloader("train", batch_size=BATCH_SIZE)
    val_loader = get_dataloader("valid", batch_size=BATCH_SIZE)

    print(f"[train] Train samples: {len(train_loader.dataset)}")
    print(f"[train] Val samples: {len(val_loader.dataset)}")

    # Model
    model = CLIPSegFinetune().to(DEVICE)

    # Loss — use pos_weight for crack imbalance
    criterion = CombinedLoss(
        bce_weight=BCE_WEIGHT,
        dice_weight=DICE_WEIGHT,
        pos_weight=CRACK_POS_WEIGHT,
    )

    # Optimizer + scheduler
    optimizer = AdamW(model.trainable_parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_dice = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for images, prompts, masks, meta in pbar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images, list(prompts))
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_dice_sum = 0.0
        val_iou_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for images, prompts, masks, meta in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(images, list(prompts))
                preds = torch.sigmoid(logits)

                for i in range(preds.shape[0]):
                    val_dice_sum += compute_dice(preds[i], masks[i])
                    val_iou_sum += compute_iou(preds[i], masks[i])
                    val_count += 1

        avg_val_dice = val_dice_sum / max(val_count, 1)
        avg_val_iou = val_iou_sum / max(val_count, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_dice"].append(avg_val_dice)
        history["val_iou"].append(avg_val_iou)

        print(f"Epoch {epoch}/{EPOCHS} — loss: {avg_train_loss:.4f} | "
              f"val_dice: {avg_val_dice:.4f} | val_iou: {avg_val_iou:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            patience_counter = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": avg_val_dice,
                "val_iou": avg_val_iou,
                "history": history,
            }, ckpt_path)
            print(f"  → Saved best model (dice={avg_val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[early stop] No improvement for {PATIENCE} epochs.")
                break

    print(f"\n[train] Done. Best val Dice: {best_val_dice:.4f}")
    return history


if __name__ == "__main__":
    train()
