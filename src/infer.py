"""Inference entry point."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import CodeDataset, HeadTailCodeDataset
from src.data.preprocess import preprocess_dataframe
from src.utils.io import ensure_dir, load_config
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and create a submission file.")
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    parser.add_argument("--checkpoint", help="Path to model checkpoint. Defaults to <output.dir>/best_model.pt")
    parser.add_argument("--test-path", help="Override cfg['paths']['test']")
    parser.add_argument("--output-dir", help="Override cfg['output']['dir']")
    parser.add_argument("--submission-path", help="Where to save submission.csv")
    parser.add_argument("--batch-size", type=int, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--threshold", type=float, help="Optional threshold on P(AI) for binary labels")
    return parser.parse_args()


def _load_test_dataframe(cfg: dict, test_path: str | None, logger) -> pd.DataFrame:
    path = test_path or cfg["paths"]["test"]
    text_col = cfg["data"]["text_column"]

    logger.info(f"Loading test data from {path}")
    test_df = pd.read_parquet(path)
    test_df = preprocess_dataframe(test_df, text_col=text_col)
    logger.info(f"Test: {len(test_df):,}")
    return test_df


def _submission_ids(cfg: dict, test_df: pd.DataFrame, logger) -> np.ndarray:
    if "ID" in test_df.columns:
        return test_df["ID"].values

    sample_path = cfg.get("paths", {}).get("sample_submission")
    if sample_path and Path(sample_path).exists():
        sample_df = pd.read_csv(sample_path)
        if "ID" in sample_df.columns and len(sample_df) == len(test_df):
            logger.info("Using IDs from sample_submission.csv")
            return sample_df["ID"].values

    logger.warning("Test data has no ID column; using row indices as IDs")
    return np.arange(len(test_df))


def _checkpoint_path(cfg: dict, checkpoint: str | None) -> Path:
    if checkpoint:
        return Path(checkpoint)
    return Path(cfg["output"]["dir"]) / "best_model.pt"


def _load_state_dict(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _predict_transformer(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(**batch)
            else:
                out = model(**batch)

            all_logits.append(out["logits"].cpu())

    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    return preds, probs, logits.numpy()


def _dataset_class_and_kwargs(model_cfg: dict) -> tuple[type, dict]:
    long_ctx = model_cfg.get("long_context")
    if long_ctx and long_ctx.get("strategy") == "head_tail":
        return HeadTailCodeDataset, {
            "head_tokens": long_ctx["head_tokens"],
            "tail_tokens": long_ctx["tail_tokens"],
        }
    return CodeDataset, {"max_length": model_cfg["max_length"]}


def _run_transformer_inference(
    cfg: dict,
    args: argparse.Namespace,
    logger,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    from src.models.codebert_cls import CodeClassifier, load_tokenizer

    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    checkpoint_path = _checkpoint_path(cfg, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_df = _load_test_dataframe(cfg, args.test_path, logger)
    texts = test_df[cfg["data"]["text_column"]].tolist()

    tokenizer = load_tokenizer(model_cfg["backbone"])
    DatasetCls, ds_kwargs = _dataset_class_and_kwargs(model_cfg)
    test_ds = DatasetCls(texts, labels=None, tokenizer=tokenizer, **ds_kwargs)

    batch_size = args.batch_size or train_cfg.get("batch_size", 16) * 2
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    model = CodeClassifier(
        model_cfg["backbone"],
        num_labels=model_cfg.get("num_labels", 2),
    )
    model.load_state_dict(_load_state_dict(checkpoint_path))
    model.to(device)

    use_amp = bool(train_cfg.get("mixed_precision")) and device == "cuda"
    preds, probs, logits = _predict_transformer(model, test_loader, device, use_amp)

    if args.threshold is not None:
        logger.info(f"Applying threshold on P(AI): {args.threshold}")
        preds = (probs[:, 1] >= args.threshold).astype(int)

    return test_df, preds, probs, logits


def _run_tfidf_inference(
    cfg: dict,
    args: argparse.Namespace,
    logger,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, None]:
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("TF-IDF inference needs joblib and a saved sklearn pipeline.") from exc

    model_path = Path(args.checkpoint) if args.checkpoint else Path(cfg["output"]["dir"]) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            "TF-IDF inference requires a saved sklearn pipeline. "
            f"Expected {model_path}; current run_tfidf.py does not save one by default."
        )

    test_df = _load_test_dataframe(cfg, args.test_path, logger)
    texts = test_df[cfg["data"]["text_column"]].tolist()

    logger.info(f"Loading TF-IDF pipeline from {model_path}")
    pipeline = joblib.load(model_path)
    probs_ai = pipeline.predict_proba(texts)[:, 1]
    threshold = 0.5 if args.threshold is None else args.threshold
    preds = (probs_ai >= threshold).astype(int)
    probs = np.column_stack([1.0 - probs_ai, probs_ai])
    return test_df, preds, probs, None


def _save_outputs(
    cfg: dict,
    args: argparse.Namespace,
    test_df: pd.DataFrame,
    preds: np.ndarray,
    probs: np.ndarray | None,
    logits: np.ndarray | None,
    logger,
) -> None:
    output_dir = Path(cfg["output"]["dir"])
    ensure_dir(output_dir)

    submission_path = Path(args.submission_path) if args.submission_path else output_dir / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    sub_df = pd.DataFrame({
        "ID": _submission_ids(cfg, test_df, logger),
        "label": preds.astype(int),
    })
    sub_df.to_csv(submission_path, index=False)
    logger.info(f"Saved submission -> {submission_path} ({len(sub_df):,} rows)")

    if cfg.get("output", {}).get("save_logits", False) and probs is not None:
        logits_df = pd.DataFrame({
            "ID": sub_df["ID"].values,
            "prob_human": probs[:, 0],
            "prob_ai": probs[:, 1],
        })
        if logits is not None:
            logits_df["logit_0"] = logits[:, 0]
            logits_df["logit_1"] = logits[:, 1]

        logits_path = output_dir / "test_logits.csv"
        logits_df.to_csv(logits_path, index=False)
        logger.info(f"Saved test logits -> {logits_path}")


def main():
    args = _parse_args()
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        raise ValueError("--threshold must be in [0, 1]")

    cfg = load_config(args.config)
    if args.output_dir:
        cfg.setdefault("output", {})["dir"] = args.output_dir

    seed_everything(cfg.get("seed", 42))

    run_name = cfg["run_name"]
    output_dir = cfg.get("output", {}).get("dir", f"runs/{run_name}")
    cfg.setdefault("output", {})["dir"] = output_dir
    ensure_dir(output_dir)

    logger = setup_logger(run_name, log_file=f"{output_dir}/infer.log")
    logger.info(f"Starting inference for {run_name}")
    logger.info(f"Config: {cfg}")

    model_type = cfg.get("model", {}).get("type", cfg.get("model", {}).get("architecture", ""))

    if model_type == "cls_classifier":
        test_df, preds, probs, logits = _run_transformer_inference(cfg, args, logger)
    elif model_type == "tfidf_lr":
        test_df, preds, probs, logits = _run_tfidf_inference(cfg, args, logger)
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")

    _save_outputs(cfg, args, test_df, preds, probs, logits, logger)
    logger.info("Inference complete")


if __name__ == "__main__":
    main()
