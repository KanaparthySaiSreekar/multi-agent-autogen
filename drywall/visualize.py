"""Generate side-by-side visualizations: Original | Ground Truth | Prediction."""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config import (
    CANONICAL_PROMPTS,
    CHECKPOINT_DIR,
    DATA_PREPARED_DIR,
    DEVICE,
    IMAGE_SIZE,
    OUTPUT_MASKS_DIR,
    OUTPUT_VISUALS_DIR,
    THRESHOLD,
    seed_everything,
)
from data.dataset import DrywallSegDataset
from model.clipseg_finetune import CLIPSegFinetune


def visualize(n_good: int = 4, n_fail: int = 3):
    """Generate visualization images for each task."""
    seed_everything()
    os.makedirs(OUTPUT_VISUALS_DIR, exist_ok=True)

    # Load model
    model = CLIPSegFinetune().to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for task in ["crack", "taping"]:
        dataset = DrywallSegDataset(split="test", task_filter=task)
        if len(dataset) == 0:
            continue

        prompt = CANONICAL_PROMPTS[task]
        scores = []

        # Score all test images
        with torch.no_grad():
            for idx in range(len(dataset)):
                img_tensor, _, mask_tensor, meta = dataset[idx]
                img_tensor_dev = img_tensor.unsqueeze(0).to(DEVICE)
                logits = model(img_tensor_dev, [prompt])
                pred = torch.sigmoid(logits).squeeze(0).cpu()

                # Compute dice
                pred_bin = (pred > THRESHOLD).float()
                inter = (pred_bin * mask_tensor).sum()
                dice = (2 * inter / (pred_bin.sum() + mask_tensor.sum() + 1e-8)).item()
                scores.append((idx, dice))

        # Sort by dice — best and worst
        scores.sort(key=lambda x: x[1], reverse=True)
        good_indices = [s[0] for s in scores[:n_good]]
        fail_indices = [s[0] for s in scores[-n_fail:]]

        for label, indices in [("good", good_indices), ("fail", fail_indices)]:
            for rank, idx in enumerate(indices):
                img_tensor, _, mask_tensor, meta = dataset[idx]

                # Denormalize image for display
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                img_display = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

                # Get prediction
                with torch.no_grad():
                    logits = model(img_tensor.unsqueeze(0).to(DEVICE), [prompt])
                    pred = torch.sigmoid(logits).squeeze(0).cpu().numpy()

                gt_mask = mask_tensor.numpy()
                pred_mask = (pred > THRESHOLD).astype(np.float32)

                # Create figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(img_display)
                axes[0].set_title("Original")
                axes[0].axis("off")

                axes[1].imshow(img_display)
                axes[1].imshow(gt_mask, alpha=0.4, cmap="Reds")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(img_display)
                axes[2].imshow(pred_mask, alpha=0.4, cmap="Blues")
                dice_val = scores[idx][1] if label == "good" else [s[1] for s in scores if s[0] == idx][0]
                axes[2].set_title(f"Prediction (Dice={dice_val:.3f})")
                axes[2].axis("off")

                fig.suptitle(f'{task} — "{prompt}" — {label} example {rank + 1}', fontsize=14)
                plt.tight_layout()

                out_path = os.path.join(
                    OUTPUT_VISUALS_DIR, f"{task}_{label}_{rank + 1}.png"
                )
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"[vis] Saved {out_path}")

    print(f"\n[visualize] All visuals saved to {OUTPUT_VISUALS_DIR}")


if __name__ == "__main__":
    visualize()
