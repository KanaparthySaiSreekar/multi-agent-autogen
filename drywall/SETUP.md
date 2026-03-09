# Detailed Setup Guide

A step-by-step guide to get the Drywall QA segmentation project running, from account creation to training results.

---

## 1. Prerequisites

| Requirement | Required? | Notes |
|---|---|---|
| Google account | Yes (Colab) / No (local) | For Google Colab access |
| Roboflow account | Yes | Free tier, no credit card needed |
| Python 3.10+ | Only for local | Not needed if using Colab |
| NVIDIA GPU + CUDA | Recommended | CPU works but training will be very slow |

---

## 2. Getting Your Roboflow API Key

The datasets are hosted on Roboflow, and you need a free API key to download them.

### Create an account

1. Go to [roboflow.com](https://roboflow.com) and click **Sign Up**
2. Sign up with Google, GitHub, or email — no credit card required
3. Complete the onboarding (you can skip optional steps)

### Find your API key

1. Click your **profile icon** (top-right corner)
2. Click **Settings**
3. In the left sidebar, click **Roboflow API Key** (or go directly to [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api))
4. Copy the **Private API Key** — it starts with `rf_...`

### Security note

Never commit your API key to git. The project supports two safe ways to provide it:
- **Environment variable:** `export ROBOFLOW_API_KEY="rf_your_key_here"`
- **Interactive prompt:** just run `python data/download.py` and paste when asked

---

## 3. About the Datasets

Both datasets are **public** on Roboflow Universe — you do not need to create or upload any data. The `data/download.py` script downloads them automatically using your API key.

### Taping dataset

- **What:** Images of drywall joints/seams with segmentation annotations
- **Roboflow location:** workspace `construction-bj tried`, project `drywall-join-detect`, version 2
- **Search on Universe:** go to [universe.roboflow.com](https://universe.roboflow.com) and search "Drywall-Join-Detect"

### Crack dataset

- **What:** Images of wall cracks with segmentation annotations
- **Roboflow location:** workspace `university-bswxt`, project `crack-bphdr`, version 2
- **Search on Universe:** go to [universe.roboflow.com](https://universe.roboflow.com) and search "Crack"

### How to find workspace/project slugs from a URL

If you find a dataset on Roboflow Universe, its URL structure is:
```
https://universe.roboflow.com/<workspace>/<project>/dataset/<version>
```
Use the `<workspace>` and `<project>` portions. If the download fails because slugs have changed, update the values in `config.py` under `ROBOFLOW_DATASETS`.

---

## 4. Option A: Running on Google Colab (Recommended)

Google Colab gives you a free GPU and pre-installed CUDA — no local setup needed.

### Step 1: Open Colab

Go to [colab.research.google.com](https://colab.research.google.com).

### Step 2: Upload the notebook

- Click **File > Upload notebook**
- Select `notebook.ipynb` from this project

### Step 3: Enable GPU

- Click **Runtime > Change runtime type**
- Under **Hardware accelerator**, select **T4 GPU**
- Click **Save**

### Step 4: Upload project files

You need the full project in Colab. Two options:

**Option A — Zip upload (simplest):**
1. Zip the entire `drywall/` folder on your local machine
2. In Colab, click the **Files** panel (folder icon, left sidebar)
3. Click the **Upload** button and upload `drywall.zip`
4. In the notebook, uncomment and run the cell under "2. Clone / Upload Project":
   ```python
   !unzip drywall.zip
   %cd drywall
   ```

**Option B — Clone from GitHub:**
If you've pushed this repo to GitHub, uncomment the git clone cell and update the URL:
```python
!git clone https://github.com/YOUR_USERNAME/drywall.git
%cd drywall
```

### Step 5: Set your API key

In cell 4 (under "3. API Key"), paste your Roboflow API key between the quotes:
```python
os.environ["ROBOFLOW_API_KEY"] = "rf_your_key_here"
```

### Step 6: Run all cells

Click **Runtime > Run all**, or run cells one by one sequentially.

### Expected runtime (T4 GPU)

| Step | Cell(s) | Approx. time |
|---|---|---|
| Install dependencies | Cell 2 | ~1 min |
| Download data | Cell 5 | ~2-5 min |
| Prepare masks | Cell 6 | ~1 min |
| Train (50 epochs) | Cell 8 | ~30-60 min |
| Evaluate | Cell 9 | ~2 min |
| Predict + Visualize | Cells 10-12 | ~5 min |

### (Optional) Save checkpoints to Google Drive

Uncomment cell 3 to mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Then copy checkpoints after training:
```python
!cp -r checkpoints/ /content/drive/MyDrive/drywall_checkpoints/
```

---

## 5. Option B: Running Locally

### Step 1: Clone or download the project

```bash
git clone https://github.com/YOUR_USERNAME/drywall.git
cd drywall
```

### Step 2: Create a Python environment

Using **venv**:
```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows
```

Or using **conda**:
```bash
conda create -n drywall python=3.10 -y
conda activate drywall
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** `pycocotools` can be tricky on Windows. If installation fails, try:
> ```bash
> pip install pycocotools-windows
> ```
> Or install Visual C++ Build Tools first, then retry `pip install pycocotools`.

### Step 4: Set your API key

```bash
# Linux / macOS
export ROBOFLOW_API_KEY="rf_your_key_here"

# Windows (PowerShell)
$env:ROBOFLOW_API_KEY = "rf_your_key_here"

# Windows (cmd)
set ROBOFLOW_API_KEY=rf_your_key_here
```

Or skip this — `download.py` will prompt you interactively.

### Step 5: Run the pipeline

Run each script in order:

```bash
# 1. Download datasets from Roboflow
python data/download.py

# 2. Convert COCO annotations to binary masks
python data/prepare.py

# 3. Fine-tune CLIPSeg (frozen encoders, trainable decoder)
python train.py

# 4. Evaluate mIoU and Dice metrics
python evaluate.py

# 5. Generate predicted masks on test set
python predict.py

# 6. Create side-by-side visualization images
python visualize.py
```

### GPU vs CPU

- The project auto-detects CUDA via `config.py` (`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`)
- **With GPU:** Training takes ~30-60 min (50 epochs, batch size 8)
- **Without GPU:** Training may take several hours. Consider reducing `EPOCHS` in `config.py`

---

## 6. Pipeline Walkthrough

Here's what each step does and what it produces:

### `data/download.py` — Download datasets

- Connects to Roboflow API and downloads both datasets in COCO segmentation format
- **Output:** `data/raw/taping/` and `data/raw/crack/` — each containing images and `_annotations.coco.json`
- Skips download if data already exists

### `data/prepare.py` — Prepare binary masks

- Reads COCO JSON annotations and renders polygon annotations as binary mask PNGs
- Creates train/val/test splits
- **Output:** `data/prepared/` with mask images and `manifest.csv` (maps each image to its mask, prompt, and split)

### `train.py` — Train CLIPSeg

- Loads the pretrained `CIDAS/clipseg-rd64-refined` model
- Freezes CLIP vision and text encoders, fine-tunes only the decoder
- Trains jointly on both crack and taping tasks using prompt augmentation
- Loss: 0.5 * BCE + 0.5 * Dice (with pos_weight=10 for cracks)
- Early stopping with patience=10
- **Output:** `checkpoints/best_model.pt`

### `evaluate.py` — Compute metrics

- Loads the best checkpoint and runs inference on the test split
- Reports per-task mIoU and Dice scores
- **Output:** Metrics printed to console

### `predict.py` — Generate masks

- Runs inference on test images and saves predicted binary masks
- **Output:** `outputs/masks/` — PNG mask files

### `visualize.py` — Create visualizations

- Creates side-by-side comparison images (input | ground truth | prediction)
- **Output:** `outputs/visuals/` — PNG comparison images

### Expected folder structure after full pipeline

```
drywall/
├── data/
│   ├── raw/
│   │   ├── taping/          # Downloaded images + COCO JSON
│   │   └── crack/           # Downloaded images + COCO JSON
│   └── prepared/
│       ├── manifest.csv     # Image-mask-prompt mapping
│       ├── taping/          # Binary mask PNGs
│       └── crack/           # Binary mask PNGs
├── checkpoints/
│   └── best_model.pt        # Trained model weights
├── outputs/
│   ├── masks/               # Predicted mask PNGs
│   └── visuals/             # Side-by-side comparison PNGs
└── ...
```

---

## 7. Troubleshooting / FAQ

### "Dataset not found" or 404 error from Roboflow

The workspace or project slugs may have changed on Roboflow. To fix:
1. Go to [universe.roboflow.com](https://universe.roboflow.com)
2. Search for the dataset (e.g., "Drywall-Join-Detect" or "Crack")
3. Open the dataset page — the URL shows the current workspace and project slugs
4. Update the `ROBOFLOW_DATASETS` dict in `config.py`

### "CUDA out of memory"

Reduce the batch size in `config.py`:
```python
BATCH_SIZE = 4  # default is 8; try 4 or 2
```
You can also reduce `IMAGE_SIZE` from 352 to 224, but this may affect accuracy.

### "No module named X"

Install the missing package:
```bash
pip install <package_name>
```
All required packages are listed in `requirements.txt`. Re-run `pip install -r requirements.txt` to ensure everything is installed.

### pycocotools won't install on Windows

Option 1 — use the Windows-specific package:
```bash
pip install pycocotools-windows
```

Option 2 — install build tools first:
1. Download and install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Retry: `pip install pycocotools`

### Roboflow rate limits

The free tier has API rate limits. If you hit them:
- Wait a few minutes and retry
- The download script skips datasets that are already downloaded, so re-running is safe

### Training is too slow on CPU

- Use Google Colab (free T4 GPU) — see Option A above
- Reduce `EPOCHS` in `config.py` (e.g., from 50 to 10 for a quick test)
- Reduce `IMAGE_SIZE` from 352 to 224

### Colab session disconnects mid-training

- Mount Google Drive (cell 3 in notebook) and periodically copy checkpoints
- The training script saves the best model to `checkpoints/best_model.pt`, so you can resume from there if you modify `train.py` to support checkpoint loading
