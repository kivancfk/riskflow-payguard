"""SHAP-based explanations for individual predictions."""
from __future__ import annotations

from typing import Any

import numpy as np


def top_risk_factors(
    explainer: Any,
    x_row: np.ndarray,
    feature_names: list[str],
    k: int = 5,
) -> list[dict]:
    """Return the top-k features by absolute SHAP value for a single prediction.

    Returned shape matches the API's RiskFactor schema:
        [{"feature": ..., "shap_value": float, "direction": "increases_risk"|"decreases_risk"}, ...]
    """
    shap_values = explainer.shap_values(x_row)
    # TreeExplainer on binary classifier returns a list [neg_class, pos_class] in older shap
    # or a single array in newer shap; handle both.
    if isinstance(shap_values, list):
        shap_row = np.asarray(shap_values[1]).ravel()
    else:
        shap_row = np.asarray(shap_values).ravel()

    idx = np.argsort(np.abs(shap_row))[::-1][:k]
    return [
        {
            "feature": feature_names[i],
            "shap_value": float(shap_row[i]),
            "direction": "increases_risk" if shap_row[i] > 0 else "decreases_risk",
        }
        for i in idx
    ]
