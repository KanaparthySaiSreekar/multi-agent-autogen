"""Run inference on test set — save predicted masks as PNGs."""

import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from config import (
    CANONICAL_PROMPTS,
    CHECKPOINT_DIR,
    DEVICE,
    IMAGE_SIZE,
    OUTPUT_MASKS_DIR,
    THRESHOLD,
    seed_everything,
)
from data.dataset import DrywallSegDataset
from model.clipseg_finetune import CLIPSegFinetune


def predict():
    seed_everything()
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

    # Load model
    model = CLIPSegFinetune().to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[predict] Loaded model from epoch {ckpt['epoch']}")

    # Model size
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"[predict] Model size: {param_mb:.1f} MB")

    # Run on test set for both tasks
    total_time = 0.0
    n_images = 0

    for task in ["crack", "taping"]:
        dataset = DrywallSegDataset(split="test", task_filter=task)
        if len(dataset) == 0:
            print(f"[warn] No test data for {task}")
            continue

        prompt = CANONICAL_PROMPTS[task]
        print(f"\n[predict] {task}: {len(dataset)} images, prompt: '{prompt}'")

        for idx in tqdm(range(len(dataset)), desc=f"Predict {task}"):
            img_tensor, _, mask_tensor, meta = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

            start = time.time()
            with torch.no_grad():
                logits = model(img_tensor, [prompt])
                pred = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            elapsed = time.time() - start
            total_time += elapsed
            n_images += 1

            # Threshold and save
            mask_pred = (pred > THRESHOLD).astype(np.uint8) * 255

            # Generate filename: {image_id}__{prompt_slug}.png
            img_name = os.path.splitext(os.path.basename(meta["image_path"]))[0]
            prompt_slug = prompt.replace(" ", "_")
            out_name = f"{img_name}__{prompt_slug}.png"
            out_path = os.path.join(OUTPUT_MASKS_DIR, out_name)
            cv2.imwrite(out_path, mask_pred)

    avg_time = total_time / max(n_images, 1)
    print(f"\n[predict] Done. {n_images} images processed.")
    print(f"[predict] Avg inference time: {avg_time * 1000:.1f} ms/image")
    print(f"[predict] Masks saved to {OUTPUT_MASKS_DIR}")


if __name__ == "__main__":
    predict()
