"""Runs 2/3: UniXcoder CLS classifier (same architecture, different backbone)."""

# UniXcoder uses the same architecture as CodeBERT CLS classifier.
# This module re-exports for clarity in the config system.

from src.models.codebert_cls import CodeClassifier, load_tokenizer

__all__ = ["CodeClassifier", "load_tokenizer"]
