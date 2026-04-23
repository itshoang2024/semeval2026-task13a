"""Build validation splits: random stratified, leave-one-language-out, length/style bins."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def _composite_stratify_key(df: pd.DataFrame, label_col: str, lang_col: str) -> pd.Series:
    """Create a composite stratification key from label × language."""
    return df[label_col].astype(str) + "_" + df[lang_col].astype(str)


def build_random_stratified(
    df: pd.DataFrame,
    label_col: str = "label",
    lang_col: str = "language",
    n_splits: int = 10,
    seed: int = 42,
) -> dict:
    """K-fold stratified by label × language to preserve both proportions."""
    strat_key = _composite_stratify_key(df, label_col, lang_col)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, strat_key)):
        splits[f"fold_{fold}"] = {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
        }
    return splits


def build_holdout_split(
    df: pd.DataFrame,
    label_col: str = "label",
    lang_col: str = "language",
    val_size: float = 0.1,
    seed: int = 42,
) -> dict:
    """Single train/val split stratified by label × language."""
    strat_key = _composite_stratify_key(df, label_col, lang_col)
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=val_size,
        stratify=strat_key,
        random_state=seed,
    )
    return {
        "holdout": {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
        }
    }


def build_lolo_language(df: pd.DataFrame, lang_col: str = "language") -> dict:
    """Leave-one-language-out: train on N-1 languages, validate on the held-out one."""
    splits = {}
    for lang in df[lang_col].unique():
        val_idx = df[df[lang_col] == lang].index.tolist()
        train_idx = df[df[lang_col] != lang].index.tolist()
        splits[f"holdout_{lang}"] = {"train": train_idx, "val": val_idx}
    return splits


def build_length_style_bins(df: pd.DataFrame, text_col: str = "code", n_bins: int = 3) -> dict:
    """Hold out code from extreme length bins as proxy OOD."""
    lengths = df[text_col].str.len()
    df = df.copy()
    df["_length_bin"] = pd.qcut(lengths, q=n_bins, labels=[f"bin_{i}" for i in range(n_bins)])

    splits = {}
    for bin_name in df["_length_bin"].unique():
        val_idx = df[df["_length_bin"] == bin_name].index.tolist()
        train_idx = df[df["_length_bin"] != bin_name].index.tolist()
        splits[f"holdout_{bin_name}"] = {"train": train_idx, "val": val_idx}
    return splits


def save_splits(splits: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
