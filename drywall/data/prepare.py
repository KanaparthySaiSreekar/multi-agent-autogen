"""Parse COCO JSON annotations and rasterize them into binary mask PNGs.

Generates a manifest CSV: image_path, mask_path, task_type, split
"""

import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DATA_RAW_DIR, DATA_PREPARED_DIR, MANIFEST_CSV


def _rasterize_annotations(coco: dict, images_dir: str, masks_dir: str):
    """Convert COCO annotations to binary mask PNGs.

    Returns list of (image_path, mask_path) pairs.
    """
    os.makedirs(masks_dir, exist_ok=True)

    # Build lookup: image_id → image info
    id_to_img = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    img_to_anns: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    pairs = []
    for img_id, img_info in id_to_img.items():
        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in img_to_anns.get(img_id, []):
            # Prefer polygon segmentation
            if ann.get("segmentation") and isinstance(ann["segmentation"], list):
                for poly in ann["segmentation"]:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    pts = pts.astype(np.int32)
                    cv2.fillPoly(mask, [pts], 255)
            elif ann.get("bbox"):
                # Fallback: use bounding box as coarse mask
                x, y, bw, bh = [int(v) for v in ann["bbox"]]
                mask[y : y + bh, x : x + bw] = 255

        fname = img_info["file_name"]
        mask_name = os.path.splitext(fname)[0] + "_mask.png"
        mask_path = os.path.join(masks_dir, mask_name)
        cv2.imwrite(mask_path, mask)

        img_path = os.path.join(images_dir, fname)
        pairs.append((img_path, mask_path))

    return pairs


def prepare_all():
    """Process all tasks and splits, write manifest CSV."""
    os.makedirs(DATA_PREPARED_DIR, exist_ok=True)
    rows = []

    for task_name in ["taping", "crack"]:
        task_dir = os.path.join(DATA_RAW_DIR, task_name)
        if not os.path.isdir(task_dir):
            print(f"[warn] {task_dir} not found, skipping {task_name}")
            continue

        for split in ["train", "valid", "test"]:
            split_dir = os.path.join(task_dir, split)
            ann_file = os.path.join(split_dir, "_annotations.coco.json")

            if not os.path.isfile(ann_file):
                print(f"[warn] {ann_file} not found, skipping")
                continue

            images_dir = split_dir  # Roboflow puts images next to JSON
            # Also check images/ subdirectory
            if os.path.isdir(os.path.join(split_dir, "images")):
                images_dir = os.path.join(split_dir, "images")

            masks_dir = os.path.join(DATA_PREPARED_DIR, task_name, split, "masks")

            with open(ann_file, "r") as f:
                coco = json.load(f)

            pairs = _rasterize_annotations(coco, images_dir, masks_dir)
            for img_path, mask_path in pairs:
                rows.append({
                    "image_path": os.path.relpath(img_path, DATA_PREPARED_DIR),
                    "mask_path": os.path.relpath(mask_path, DATA_PREPARED_DIR),
                    "task_type": task_name,
                    "split": split,
                })

            # Stats
            n_images = len(pairs)
            if n_images > 0:
                sample_mask = cv2.imread(pairs[0][1], cv2.IMREAD_GRAYSCALE)
                pix_ratio = (sample_mask > 0).sum() / sample_mask.size if sample_mask is not None else 0
                print(f"[{task_name}/{split}] {n_images} images, sample fg ratio: {pix_ratio:.4f}")
            else:
                print(f"[{task_name}/{split}] 0 images")

    # Write manifest
    with open(MANIFEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "mask_path", "task_type", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[done] Manifest written to {MANIFEST_CSV} ({len(rows)} entries)")


if __name__ == "__main__":
    prepare_all()
