"""Download the three public datasets into the local Kaggle cache (~1.6 GB)."""
import kagglehub

from fraudguard.datasets import DATASETS

if __name__ == "__main__":
    for ds in DATASETS.values():
        print(ds.key, kagglehub.dataset_download(ds.kaggle))
