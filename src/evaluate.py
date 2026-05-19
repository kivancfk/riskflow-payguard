"""Evaluation metrics for fraud models.

Headline metrics:
    - ROC-AUC (overall ranking quality)
    - PR-AUC (better for severe class imbalance)
    - Recall @ fixed FPR (e.g. recall at FPR=1%)
    - Confusion matrix at the chosen operating threshold
    - Business cost at chosen threshold (see src.business_cost ideas)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def headline_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def recall_at_fpr(y_true: np.ndarray, y_proba: np.ndarray, target_fpr: float) -> float:
    raise NotImplementedError
