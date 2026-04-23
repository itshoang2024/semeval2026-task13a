"""Data preprocessing for code detection."""

import re

import pandas as pd


def load_raw_data(train_path: str, test_path: str, val_path: str | None = None) -> tuple:
    """Load parquet data files. Returns (train, test) or (train, val, test)."""
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    if val_path:
        val_df = pd.read_parquet(val_path)
        return train_df, val_df, test_df
    return train_df, test_df


def clean_code(text: str) -> str:
    """Minimal cleaning: normalize line endings, strip trailing whitespace per line."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "code") -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].fillna("").apply(clean_code)
    df["code_length"] = df[text_col].str.len()
    df["num_lines"] = df[text_col].str.count("\n") + 1
    return df
