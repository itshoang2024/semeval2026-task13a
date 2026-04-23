"""Data preprocessing for code detection."""

import re

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data(train_path: str, test_path: str, val_path: str | None = None) -> tuple:
    """Load parquet data files. Returns (train, test) or (train, val, test)."""
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    if val_path:
        val_df = pd.read_parquet(val_path)
        return train_df, val_df, test_df
    return train_df, test_df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_code(text: str) -> str:
    """Minimal cleaning: normalize line endings, strip trailing whitespace per line."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "code") -> pd.DataFrame:
    """Clean code text and compute basic length features."""
    df = df.copy()
    df[text_col] = df[text_col].fillna("").apply(clean_code)
    df["code_length"] = df[text_col].str.len()
    df["num_lines"] = df[text_col].str.count("\n") + 1
    return df


# ---------------------------------------------------------------------------
# Outlier flagging
# ---------------------------------------------------------------------------

def flag_outliers(
    df: pd.DataFrame,
    text_col: str = "code",
    extreme_percentile: float = 0.99,
) -> pd.DataFrame:
    """Flag pathological samples. Does NOT drop rows.

    Adds columns:
      - is_empty: code has 0 characters
      - is_binary_ish: >20% non-printable or non-ASCII characters
      - is_minified: single very long line (>500 chars) with no newlines
      - is_extreme_length: code_length > given percentile threshold
    """
    df = df.copy()

    if "code_length" not in df.columns:
        df["code_length"] = df[text_col].str.len()

    # Empty
    df["is_empty"] = df["code_length"] == 0

    # Binary-ish: high ratio of non-printable chars
    def _is_binary_ish(code: str) -> bool:
        if not code:
            return False
        non_printable = sum(1 for c in code if not c.isprintable() and c not in "\n\r\t")
        return (non_printable / len(code)) > 0.20

    df["is_binary_ish"] = df[text_col].apply(_is_binary_ish)

    # Minified: single long line, no newlines
    df["is_minified"] = (df[text_col].str.count("\n") == 0) & (df["code_length"] > 500)

    # Extreme length
    threshold = df["code_length"].quantile(extreme_percentile)
    df["is_extreme_length"] = df["code_length"] > threshold

    flagged = df[["is_empty", "is_binary_ish", "is_minified", "is_extreme_length"]].any(axis=1).sum()
    print(f"Outlier flags: {flagged:,} / {len(df):,} rows flagged "
          f"(empty={df['is_empty'].sum()}, binary={df['is_binary_ish'].sum()}, "
          f"minified={df['is_minified'].sum()}, extreme_len={df['is_extreme_length'].sum()})")
    return df


# ---------------------------------------------------------------------------
# Feature extraction pipeline
# ---------------------------------------------------------------------------

def extract_all_features(
    df: pd.DataFrame,
    text_col: str = "code",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extract all handcrafted features: stylometric + structural + AST-lite.

    Returns a DataFrame with ~40 feature columns (same index as input).
    """
    from src.features.stylometric import extract_stylometric
    from src.features.structural import extract_structural
    from src.features.ast_features import extract_ast_lite

    def _extract_row(code: str) -> dict:
        feats = {}
        feats.update(extract_stylometric(code))
        feats.update(extract_structural(code))
        feats.update(extract_ast_lite(code))
        return feats

    apply_fn = df[text_col].progress_apply if show_progress else df[text_col].apply
    feat_records = apply_fn(_extract_row)
    feat_df = pd.DataFrame(feat_records.tolist(), index=df.index)
    return feat_df
