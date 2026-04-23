"""Inference entry point."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import load_config
from src.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    # TODO: Load model checkpoint and run inference on test set
    print(f"Inference for {cfg['run_name']} — implement per-run logic here")


if __name__ == "__main__":
    main()
