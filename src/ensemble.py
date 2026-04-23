"""Ensemble and blending (Run 7)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io import load_config, save_json
from src.utils.metrics import compute_metrics, find_best_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg["output"]["dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # TODO: Load OOF predictions from each model, blend, calibrate threshold
    print(f"Ensemble for {cfg['run_name']} — implement after runs 3-6 complete")


if __name__ == "__main__":
    main()
