from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlbench.data.base import TabularSplit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FULL = PROJECT_ROOT / "data" / "higgs_full.parquet"


def _sample_path(n_sample: int, sample_seed: int) -> Path:
    # note to self: size and seed are in the file name so a different sample never overwrites this one
    return PROJECT_ROOT / "data" / f"higgs_sample_{n_sample}_seed{sample_seed}.parquet"


def _load_sample(n_sample: int, sample_seed: int) -> pd.DataFrame:
    path = _sample_path(n_sample, sample_seed)
    if path.exists():
        return pd.read_parquet(path)

    # note to self: the sample is drawn ONCE, then saved, so every model and every seed sees the same rows
    full = pd.read_parquet(FULL)  # about 1.3 GB as float32, fine for 12 GB of RAM
    rng = np.random.default_rng(sample_seed)
    idx = np.sort(rng.choice(len(full), size=n_sample, replace=False))
    sample = full.iloc[idx].reset_index(drop=True)
    del full
    sample.to_parquet(path)
    return sample


def load_higgs(
    split_seed: int = 123,
    test_size: float = 0.2,
    val_size: float = 0.1,
    n_sample: int = 1_000_000,
    sample_seed: int = 2024,
) -> TabularSplit:
    df = _load_sample(n_sample, sample_seed)

    y = df["label"].to_numpy().astype("int64")
    X = df.drop(columns=["label"])
    feature_names = X.columns.tolist()
    X = X.to_numpy(dtype=np.float32)

    # note to self: same split procedure as adult, split first and fit the scaler on train only
    idx = np.arange(len(X))
    trainval_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=split_seed
    )
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=rel_val, stratify=y[trainval_idx], random_state=split_seed
    )

    scaler = StandardScaler().fit(X[train_idx])
    return TabularSplit(
        scaler.transform(X[train_idx]).astype(np.float32), y[train_idx],
        scaler.transform(X[val_idx]).astype(np.float32), y[val_idx],
        scaler.transform(X[test_idx]).astype(np.float32), y[test_idx],
        feature_names=feature_names,
    )