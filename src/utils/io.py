"""I/O helpers for configs, metrics, and data."""

import json
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "inherits_from" in cfg:
        base = load_config(cfg["inherits_from"])
        base.update(cfg)
        return base

    return cfg


def save_json(data: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
