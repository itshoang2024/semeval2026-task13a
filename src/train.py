"""Main training entry point. Dispatches to the right model based on config."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import load_config, save_json, ensure_dir
from src.utils.seed import seed_everything
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    run_name = cfg["run_name"]
    output_dir = cfg.get("output", {}).get("dir", f"runs/{run_name}")
    ensure_dir(output_dir)

    logger = setup_logger(run_name, log_file=f"{output_dir}/train.log")
    logger.info(f"Starting {run_name}")
    logger.info(f"Config: {cfg}")

    # Save config copy to run dir
    import shutil
    shutil.copy2(args.config, f"{output_dir}/config.yaml")

    # Dispatch based on model type
    model_type = cfg.get("model", {}).get("type", cfg.get("model", {}).get("architecture", ""))

    if model_type == "tfidf_lr":
        from src.runners.run_tfidf import run_tfidf
        run_tfidf(cfg, logger)
    elif model_type == "cls_classifier":
        from src.runners.run_transformer import run_transformer
        run_transformer(cfg, logger)
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Finished {run_name}")


if __name__ == "__main__":
    main()
