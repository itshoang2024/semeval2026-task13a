"""Stylometric features for code authorship detection."""

import re
import numpy as np


def extract_stylometric(code: str) -> dict:
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]

    line_lengths = [len(l) for l in lines]
    indent_levels = [len(l) - len(l.lstrip()) for l in non_empty] if non_empty else [0]

    return {
        "num_lines": len(lines),
        "num_non_empty_lines": len(non_empty),
        "blank_line_ratio": 1 - len(non_empty) / max(len(lines), 1),
        "mean_line_length": float(np.mean(line_lengths)) if line_lengths else 0,
        "max_line_length": max(line_lengths) if line_lengths else 0,
        "std_line_length": float(np.std(line_lengths)) if line_lengths else 0,
        "whitespace_ratio": code.count(" ") / max(len(code), 1),
        "tab_ratio": code.count("\t") / max(len(code), 1),
        "mean_indent": float(np.mean(indent_levels)),
        "max_indent": max(indent_levels),
        "std_indent": float(np.std(indent_levels)),
        "char_count": len(code),
        "alpha_ratio": sum(c.isalpha() for c in code) / max(len(code), 1),
        "digit_ratio": sum(c.isdigit() for c in code) / max(len(code), 1),
        "special_char_ratio": sum(not c.isalnum() and not c.isspace() for c in code) / max(len(code), 1),
        "uppercase_ratio": sum(c.isupper() for c in code) / max(len(code), 1),
    }
