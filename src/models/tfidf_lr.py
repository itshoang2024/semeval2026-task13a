"""Run 1: TF-IDF char n-gram + Logistic Regression baseline."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_tfidf_lr_pipeline(
    ngram_range: tuple = (3, 5),
    analyzer: str = "char_wb",
    max_features: int = 200_000,
    C: float = 4.0,
    class_weight: str = "balanced",
) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
        )),
    ])


def predict_proba(pipeline: Pipeline, texts: list[str]) -> np.ndarray:
    return pipeline.predict_proba(texts)[:, 1]
