"""Build leakage-safe feature tables with a temporal train/validation/test split.

python -m fraudguard.prepare            # all datasets
python -m fraudguard.prepare paysim     # one dataset
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fraudguard.datasets import DATASETS, ENGINEER, LABEL_COLUMN, RAW_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SPLIT = (0.6, 0.8)  # cumulative time fractions: train | validation | test


def temporal_split(t: pd.Series) -> np.ndarray:
    """Cut on time values (not row positions) so no timestamp straddles two splits."""
    c1, c2 = np.quantile(t.to_numpy(), SPLIT)
    return np.where(t <= c1, "train", np.where(t <= c2, "validation", "test"))


def prepare(key: str) -> Path:
    ds = DATASETS[key]
    s = time.time()
    raw = pd.read_csv(ds.raw_path(), usecols=RAW_COLUMNS[key] + [LABEL_COLUMN[key]])
    df = ENGINEER[key](raw)
    del raw
    df["split"] = pd.Categorical(temporal_split(df["t"]), categories=["train", "validation", "test"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"{key}.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    summary = df.groupby("split", observed=True)["y"].agg(["size", "sum", "mean"])
    print(f"{key}: {len(df):,} rows, {time.time() - s:.0f}s\n{summary}", flush=True)
    return out


if __name__ == "__main__":
    for k in (sys.argv[1:] or list(DATASETS)):
        prepare(k)
