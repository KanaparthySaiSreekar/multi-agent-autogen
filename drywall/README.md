# Prompted Segmentation for Drywall QA

Text-conditioned segmentation model for drywall defect detection using fine-tuned CLIPSeg. Given an image and a natural language prompt (e.g., "segment crack" or "segment taping area"), the model outputs a binary mask highlighting the relevant defect.

## Setup

```bash
pip install -r requirements.txt
```

## Usage (Colab — recommended)

Open `notebook.ipynb` in Google Colab and run cells sequentially. The notebook handles:
1. Installing dependencies
2. Downloading datasets from Roboflow (requires free API key)
3. Preparing binary masks from COCO annotations
4. Training CLIPSeg (frozen encoders, trainable decoder)
5. Evaluating mIoU and Dice per task
6. Generating predicted masks and visualizations

## Usage (Local)

```bash
# 1. Download data (needs ROBOFLOW_API_KEY env var or interactive prompt)
python data/download.py

# 2. Prepare binary masks
python data/prepare.py

# 3. Train
python train.py

# 4. Evaluate
python evaluate.py

# 5. Predict
python predict.py

# 6. Visualize
python visualize.py
```

## Project Structure

```
drywall/
├── config.py                # Seeds, hyperparams, paths, prompt mappings
├── requirements.txt
├── notebook.ipynb           # Colab notebook — runs full pipeline
├── data/
│   ├── download.py          # Roboflow API download
│   ├── prepare.py           # COCO annotations → binary masks
│   └── dataset.py           # PyTorch Dataset
├── model/
│   ├── clipseg_finetune.py  # CLIPSeg with frozen encoders
│   └── loss.py              # BCE + Dice combined loss
├── train.py                 # Joint training loop
├── evaluate.py              # mIoU, Dice metrics
├── predict.py               # Inference → PNG masks
├── visualize.py             # Side-by-side comparisons
├── outputs/masks/           # Predicted masks
├── outputs/visuals/         # Visualizations
└── checkpoints/             # Model weights
```

## Model

- **Architecture**: CLIPSeg (CIDAS/clipseg-rd64-refined)
- **Strategy**: Freeze CLIP vision+text encoders, fine-tune only the decoder (~10-15M params)
- **Loss**: 0.5 * BCE + 0.5 * Dice, with pos_weight=10 for crack class imbalance
- **Training**: Joint on both tasks, AdamW, CosineAnnealingLR, early stopping

## Reproducibility

- Seed: 42 (torch, numpy, random, cudnn.deterministic)
- Image size: 352×352
- Batch size: 8
- Learning rate: 1e-4
- Epochs: 50 (early stopping patience=10)
