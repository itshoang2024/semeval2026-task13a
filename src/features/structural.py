"""Structural code features (language-agnostic)."""

import re


def extract_structural(code: str) -> dict:
    return {
        "num_functions": len(re.findall(r'\bdef\b|\bfunction\b|\bfunc\b|(?:public|private|protected)\s+\w+\s*\(', code)),
        "num_classes": len(re.findall(r'\bclass\b', code)),
        "num_imports": len(re.findall(r'\bimport\b|\b#include\b|\busing\b|\brequire\b', code)),
        "num_loops": len(re.findall(r'\bfor\b|\bwhile\b|\bdo\b', code)),
        "num_conditionals": len(re.findall(r'\bif\b|\belse\b|\bswitch\b|\bcase\b', code)),
        "num_returns": len(re.findall(r'\breturn\b', code)),
        "num_try_catch": len(re.findall(r'\btry\b|\bcatch\b|\bexcept\b', code)),
        "brace_count": code.count("{") + code.count("}"),
        "paren_count": code.count("(") + code.count(")"),
        "bracket_count": code.count("[") + code.count("]"),
        "semicolon_count": code.count(";"),
        "comma_count": code.count(","),
        "dot_count": code.count("."),
        "colon_count": code.count(":"),
        "operator_count": sum(code.count(op) for op in ["==", "!=", "<=", ">=", "+=", "-=", "&&", "||"]),
        "comment_line_count": len(re.findall(r'^\s*(?://|#|/\*|\*)', code, re.MULTILINE)),
        "comment_density": len(re.findall(r'^\s*(?://|#|/\*|\*)', code, re.MULTILINE)) / max(code.count("\n") + 1, 1),
    }
