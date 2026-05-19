"""Loads the trained model + SHAP explainer once at startup."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass
class LoadedModel:
    model: Any
    explainer: Any | None
    feature_names: list[str]
    model_version: str
    metrics: dict[str, float]


_loaded: LoadedModel | None = None


def load_model(model_path: str) -> LoadedModel:
    """Load the model artifact from disk. Expected payload schema:

        {
            "model": <fitted estimator with predict_proba>,
            "explainer": <shap.TreeExplainer or None>,
            "feature_names": [...],
            "model_version": "0.1.0",
            "metrics": {"roc_auc": ..., "pr_auc": ...},
        }
    """
    global _loaded
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Train one via `python -m src.train`."
        )

    artifact = joblib.load(path)
    _loaded = LoadedModel(
        model=artifact["model"],
        explainer=artifact.get("explainer"),
        feature_names=artifact["feature_names"],
        model_version=artifact.get("model_version", "0.0.0"),
        metrics=artifact.get("metrics", {}),
    )
    return _loaded


def get_model() -> LoadedModel:
    if _loaded is None:
        raise RuntimeError("Model not loaded. Call load_model() at app startup.")
    return _loaded


def is_loaded() -> bool:
    return _loaded is not None
