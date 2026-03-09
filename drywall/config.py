"""Central configuration for the Drywall QA segmentation project."""

import os
import random
import numpy as np
import torch

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PREPARED_DIR = os.path.join(PROJECT_ROOT, "data", "prepared")
MANIFEST_CSV = os.path.join(DATA_PREPARED_DIR, "manifest.csv")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_MASKS_DIR = os.path.join(PROJECT_ROOT, "outputs", "masks")
OUTPUT_VISUALS_DIR = os.path.join(PROJECT_ROOT, "outputs", "visuals")

# ── Model ─────────────────────────────────────────────────────────────────────
CLIPSEG_CHECKPOINT = "CIDAS/clipseg-rd64-refined"
IMAGE_SIZE = 352
THRESHOLD = 0.5

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 10
GRAD_CLIP = 1.0

# ── Loss weights ──────────────────────────────────────────────────────────────
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
CRACK_POS_WEIGHT = 10.0  # cracks are thin → heavy positive weighting

# ── Prompt pools (augmentation during training) ───────────────────────────────
PROMPT_POOLS = {
    "crack": [
        "segment crack",
        "segment wall crack",
        "segment the crack in the wall",
        "find cracks",
        "detect crack",
    ],
    "taping": [
        "segment taping area",
        "segment joint tape",
        "segment drywall seam",
        "segment drywall joint",
        "find taping area",
    ],
}

# Canonical prompts used at evaluation / inference time
CANONICAL_PROMPTS = {
    "crack": "segment crack",
    "taping": "segment taping area",
}

# ── Dataset names on Roboflow ─────────────────────────────────────────────────
ROBOFLOW_DATASETS = {
    "taping": {
        "workspace": "construction-bj tried",
        "project": "drywall-join-detect",
        "version": 2,
        "format": "coco-segmentation",
    },
    "crack": {
        "workspace": "university-bswxt",
        "project": "crack-bphdr",
        "version": 2,
        "format": "coco-segmentation",
    },
}

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
