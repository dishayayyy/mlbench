from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlbench.data.base import TabularSplit

# note to self: data/ is gitignored, so this cache never gets committed
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE = PROJECT_ROOT / "data" / "adult_raw.parquet"


def _load_raw() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_openml("adult", version=2, as_frame=True).frame
    df.to_parquet(CACHE)
    return df


def load_adult(split_seed: int = 123, test_size: float = 0.2, val_size: float = 0.1) -> TabularSplit:
    df = _load_raw()

    y = (df["class"].astype(str) == ">50K").astype("int64").to_numpy()
    # note to self: fnlwgt is a census sampling weight, not a real feature, so I drop it
    X = df.drop(columns=["class", "fnlwgt"]).copy()

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        # note to self: missing values are real in this dataset, so give them their own category
        X[c] = X[c].astype("object").where(X[c].notna(), "Unknown")

    # note to self: split first, then fit the scaler/encoder on train only, to avoid leakage
    idx = np.arange(len(X))
    trainval_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=split_seed
    )
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=rel_val, stratify=y[trainval_idx], random_state=split_seed
    )

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    X_train = pre.fit_transform(X.iloc[train_idx]).astype(np.float32)
    X_val = pre.transform(X.iloc[val_idx]).astype(np.float32)
    X_test = pre.transform(X.iloc[test_idx]).astype(np.float32)

    return TabularSplit(
        X_train, y[train_idx], X_val, y[val_idx], X_test, y[test_idx],
        feature_names=pre.get_feature_names_out().tolist(),
    )