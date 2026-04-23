"""TF-IDF + LogReg runner (Run 1)."""

from pathlib import Path

import pandas as pd
import numpy as np

from src.models.tfidf_lr import build_tfidf_lr_pipeline, predict_proba
from src.data.preprocess import load_raw_data, preprocess_dataframe
from src.utils.metrics import compute_metrics
from src.utils.io import save_json


def _load_data(paths: dict, logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed data if available, otherwise fall back to raw + preprocess."""
    processed = {
        "train": paths.get("processed_train"),
        "val": paths.get("processed_val"),
        "test": paths.get("processed_test"),
    }

    if all(v and Path(v).exists() for v in processed.values()):
        logger.info("Loading pre-processed data from data/processed/")
        train_df = pd.read_parquet(processed["train"])
        val_df = pd.read_parquet(processed["val"])
        test_df = pd.read_parquet(processed["test"])
    else:
        logger.info("Processed data not found — loading raw and preprocessing")
        train_df, val_df, test_df = load_raw_data(
            paths["train"], paths["test"], paths.get("validation")
        )
        train_df = preprocess_dataframe(train_df)
        val_df = preprocess_dataframe(val_df)
        test_df = preprocess_dataframe(test_df)

    return train_df, val_df, test_df


def run_tfidf(cfg: dict, logger):
    paths = cfg["paths"]
    output_dir = cfg["output"]["dir"]

    train_df, val_df, test_df = _load_data(paths, logger)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    # Build pipeline
    model_cfg = cfg["model"]
    pipeline = build_tfidf_lr_pipeline(
        ngram_range=tuple(model_cfg["ngram_range"]),
        analyzer=model_cfg["analyzer"],
        max_features=model_cfg["max_features"],
        C=model_cfg["C"],
        class_weight=model_cfg["class_weight"],
    )

    # Train on full train set, validate on official validation
    X_train = train_df[text_col].tolist()
    y_train = train_df[label_col].tolist()
    X_val = val_df[text_col].tolist()
    y_val = val_df[label_col].tolist()

    pipeline.fit(X_train, y_train)
    logger.info("Training complete")

    # Validate
    val_probs = predict_proba(pipeline, X_val)
    val_preds = (val_probs >= 0.5).astype(int)
    metrics = compute_metrics(np.array(y_val), val_preds)
    logger.info(f"Val metrics: {metrics}")

    # Save
    save_json({"run_name": cfg["run_name"], "validation": {"random": metrics}}, f"{output_dir}/metrics.json")
    pd.DataFrame({"prob": val_probs, "pred": val_preds, "true": y_val}).to_csv(f"{output_dir}/oof.csv", index=False)

    # Test predictions
    if cfg["output"].get("save_submission"):
        test_probs = predict_proba(pipeline, test_df[text_col].tolist())
        test_preds = (test_probs >= 0.5).astype(int)
        submission = pd.DataFrame({"ID": test_df["ID"], "label": test_preds})
        submission.to_csv(f"{output_dir}/submission.csv", index=False)
        logger.info(f"Submission saved to {output_dir}/submission.csv")
