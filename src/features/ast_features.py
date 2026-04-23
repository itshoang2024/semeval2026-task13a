"""Lightweight AST-inspired features (regex-based, no parser dependency)."""

import re


def extract_ast_lite(code: str) -> dict:
    lines = code.split("\n")
    indent_levels = []
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            indent_levels.append(indent)

    max_nesting = 0
    current = 0
    for ch in code:
        if ch == "{":
            current += 1
            max_nesting = max(max_nesting, current)
        elif ch == "}":
            current = max(0, current - 1)

    # Identifier analysis
    identifiers = re.findall(r'\b[a-zA-Z_]\w*\b', code)
    id_lengths = [len(i) for i in identifiers] if identifiers else [0]

    return {
        "max_nesting_depth": max_nesting,
        "mean_indent_depth": sum(indent_levels) / max(len(indent_levels), 1),
        "max_indent_depth": max(indent_levels) if indent_levels else 0,
        "num_identifiers": len(identifiers),
        "mean_identifier_length": sum(id_lengths) / max(len(id_lengths), 1),
        "std_identifier_length": (sum((l - sum(id_lengths)/len(id_lengths))**2 for l in id_lengths) / len(id_lengths))**0.5 if len(id_lengths) > 1 else 0,
        "unique_identifier_ratio": len(set(identifiers)) / max(len(identifiers), 1),
    }
