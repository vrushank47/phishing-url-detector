"""Dataset loading and splitting (NumPy + pandas + scikit-learn only).

Kept separate from train.py so the baselines script can reuse the exact
same features and split without importing TensorFlow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .features import extract_features

SEED = 42
DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset_phishing.csv"


def load_dataset(path: Path = DATASET_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y). Features are recomputed from each URL string, so
    training and inference share one pipeline (no train/serve skew)."""
    df = pd.read_csv(path)
    if "status" not in df.columns or "url" not in df.columns:
        raise ValueError("Dataset must contain 'url' and 'status' columns")
    y = df["status"].map({"legitimate": 0, "phishing": 1}).astype(np.int32).to_numpy()
    X = np.vstack([extract_features(u) for u in df["url"].astype(str)])
    return X, y


def split_data(X: np.ndarray, y: np.ndarray):
    """Stratified 80/20 split with a fixed seed."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
