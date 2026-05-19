"""Probability calibration.

LightGBM probabilities are not naturally well-calibrated, which matters here because
the decision engine thresholds on the probability itself. Use isotonic or Platt
scaling on a held-out validation set.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV


def calibrate(base_estimator, X_val, y_val, method: str = "isotonic"):
    """Wrap a fitted estimator with a calibrated probability layer."""
    calibrated = CalibratedClassifierCV(base_estimator, method=method, cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated
