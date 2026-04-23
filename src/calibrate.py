"""Threshold calibration utilities."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import load_config, save_json
from src.utils.metrics import find_best_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # TODO: Load OOF probabilities, search best threshold per group
    print(f"Calibration for {cfg['run_name']} — implement after ensemble")


if __name__ == "__main__":
    main()
