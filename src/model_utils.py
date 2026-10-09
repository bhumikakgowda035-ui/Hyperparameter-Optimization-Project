"""Shared utilities for the hyperparameter optimization project."""
from pathlib import Path
import json
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_data():
    """Load a public dataset bundled with scikit-learn; no download required."""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.20, random_state=42, stratify=data.target
    )
    return X_train, X_test, y_train, y_test


def build_model(params=None, random_state=42):
    params = params or {}
    defaults = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
    }
    defaults.update(params)
    return RandomForestClassifier(random_state=random_state, n_jobs=-1, **defaults)


def evaluate_cv(params=None):
    X_train, _, y_train, _ = load_data()
    model = build_model(params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=1)
    return float(np.mean(scores))


def save_json(path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def timed_fit(model, X, y):
    start = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - start
