"""Run 5+: Hybrid meta-model (neural logits + handcrafted features -> CatBoost)."""

import numpy as np
from catboost import CatBoostClassifier, Pool


def build_meta_model(depth=6, learning_rate=0.05, iterations=1000, early_stopping_rounds=100, seed=42):
    return CatBoostClassifier(
        depth=depth, learning_rate=learning_rate, iterations=iterations,
        eval_metric="Logloss", random_seed=seed, verbose=100,
        early_stopping_rounds=early_stopping_rounds, task_type="CPU",
    )


def prepare_meta_features(neural_probs, handcrafted_features):
    if neural_probs.ndim == 1:
        neural_probs = np.column_stack([1 - neural_probs, neural_probs])
    return np.hstack([neural_probs, handcrafted_features])


def train_meta_model(model, X_train, y_train, X_val, y_val):
    model.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val))
    return model
