"""Evaluation metrics for binary code detection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
    }


def compute_per_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict:
    results = {}
    for g in np.unique(groups):
        mask = groups == g
        results[str(g)] = compute_metrics(y_true[mask], y_pred[mask])
    return results


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, step: float = 0.01) -> tuple[float, float]:
    """Search for threshold that maximizes macro-F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.01, 1.0, step):
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(y_true, y_pred, target_names=["human", "ai"])
