"""Download datasets from Roboflow in COCO segmentation format."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roboflow import Roboflow
from config import DATA_RAW_DIR, ROBOFLOW_DATASETS


def download_datasets(api_key: str):
    """Download taping and crack datasets from Roboflow."""
    rf = Roboflow(api_key=api_key)

    for task_name, info in ROBOFLOW_DATASETS.items():
        dest = os.path.join(DATA_RAW_DIR, task_name)
        if os.path.exists(dest) and os.listdir(dest):
            print(f"[skip] {task_name} already downloaded at {dest}")
            continue

        print(f"[download] {task_name}: {info['workspace']}/{info['project']} v{info['version']}")
        project = rf.workspace(info["workspace"]).project(info["project"])
        version = project.version(info["version"])
        dataset = version.download(info["format"], location=dest, overwrite=True)
        print(f"[done] {task_name} → {dest}")


if __name__ == "__main__":
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        api_key = input("Enter your Roboflow API key: ").strip()
    if not api_key:
        print("Error: No API key provided. Set ROBOFLOW_API_KEY or enter it when prompted.")
        sys.exit(1)
    download_datasets(api_key)
