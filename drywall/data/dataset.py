"""PyTorch Dataset for drywall segmentation: returns (image, text_prompt, mask, metadata)."""

import csv
import os
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    CANONICAL_PROMPTS,
    DATA_PREPARED_DIR,
    IMAGE_SIZE,
    MANIFEST_CSV,
    PROMPT_POOLS,
)


class DrywallSegDataset(Dataset):
    """Dataset that yields (image, prompt, mask, metadata) tuples."""

    def __init__(self, split: str = "train", task_filter: str | None = None):
        """
        Args:
            split: one of 'train', 'valid', 'test'
            task_filter: if set, only include this task ('crack' or 'taping')
        """
        self.split = split
        self.is_train = split == "train"
        self.entries = []

        with open(MANIFEST_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                if task_filter and row["task_type"] != task_filter:
                    continue
                self.entries.append(row)

        # Image transforms (CLIP normalization)
        self.img_transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        # Augmentations applied to both image and mask
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        task = entry["task_type"]

        # Load image
        img_path = os.path.join(DATA_PREPARED_DIR, entry["image_path"])
        image = Image.open(img_path).convert("RGB")

        # Load mask
        mask_path = os.path.join(DATA_PREPARED_DIR, entry["mask_path"])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image.height, image.width), dtype=np.uint8)

        # Prompt selection
        if self.is_train:
            prompt = random.choice(PROMPT_POOLS[task])
        else:
            prompt = CANONICAL_PROMPTS[task]

        # Synchronized augmentations
        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = np.fliplr(mask).copy()

            # Color jitter (image only)
            image = self.color_jitter(image)

        # Resize mask to target size (NEAREST to preserve binary values)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask).float() / 255.0  # [0, 1]

        # Transform image
        img_tensor = self.img_transform(image)

        metadata = {
            "image_path": entry["image_path"],
            "mask_path": entry["mask_path"],
            "task_type": task,
        }

        return img_tensor, prompt, mask_tensor, metadata


def get_dataloader(split: str, batch_size: int = 8, task_filter: str | None = None,
                   num_workers: int = 2):
    """Convenience function to create a DataLoader."""
    dataset = DrywallSegDataset(split=split, task_filter=task_filter)
    shuffle = split == "train"
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
