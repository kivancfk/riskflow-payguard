"""Fraud-focused evaluation metrics for binary risk models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_THRESHOLD = 0.50

DEFAULT_REVIEW_FRACTIONS: tuple[float, ...] = (
    0.005,
    0.01,
    0.02,
    0.05,
)


def _as_float_vector(
    values: ArrayLike,
    *,
    value_name: str,
) -> np.ndarray:
    """Convert array-like input into a finite 1D float array."""
    try:
        vector = np.asarray(
            values,
            dtype="float64",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{value_name} must contain numerical values"
        ) from error

    if vector.ndim != 1:
        raise ValueError(
            f"{value_name} must be one-dimensional"
        )

    if vector.size == 0:
        raise ValueError(
            f"{value_name} must contain at least one value"
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            f"{value_name} must contain only finite values"
        )

    return vector


def _validate_binary_inputs(
    y_true: ArrayLike,
    fraud_probabilities: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned binary targets and fraud probabilities."""
    target_values = _as_float_vector(
        y_true,
        value_name="y_true",
    )
    probabilities = _as_float_vector(
        fraud_probabilities,
        value_name="fraud_probabilities",
    )

    if len(target_values) != len(probabilities):
        raise ValueError(
            "y_true and fraud_probabilities must have "
            "the same length"
        )

    if not np.isin(
        target_values,
        [0.0, 1.0],
    ).all():
        raise ValueError(
            "y_true must contain only binary values 0 and 1"
        )

    observed_classes = set(
        target_values.astype("int8").tolist()
    )

    if observed_classes != {0, 1}:
        raise ValueError(
            "y_true must contain both binary classes"
        )

    if (
        (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
    ):
        raise ValueError(
            "fraud_probabilities must be between 0 and 1"
        )

    return (
        target_values.astype("int8"),
        probabilities,
    )


def _validate_threshold(threshold: float) -> float:
    """Return a valid probability threshold."""
    try:
        validated_threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "threshold must be numerical"
        ) from error

    if not math.isfinite(validated_threshold):
        raise ValueError(
            "threshold must be finite"
        )

    if not 0.0 <= validated_threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    return validated_threshold


def _validate_review_fractions(
    review_fractions: Sequence[float],
) -> tuple[float, ...]:
    """Validate requested review-capacity fractions."""
    try:
        fractions = tuple(
            float(value)
            for value in review_fractions
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "review_fractions must contain "
            "numerical values"
        ) from error

    if not fractions:
        raise ValueError(
            "review_fractions must not be empty"
        )

    if not all(
        math.isfinite(value)
        for value in fractions
    ):
        raise ValueError(
            "review_fractions must contain "
            "only finite values"
        )

    if any(
        value <= 0.0 or value > 1.0
        for value in fractions
    ):
        raise ValueError(
            "review_fractions must be greater than "
            "0 and no greater than 1"
        )

    if len(fractions) != len(set(fractions)):
        raise ValueError(
            "review_fractions must not contain duplicates"
        )

    return fractions


def _validate_transaction_amounts(
    transaction_amounts: ArrayLike,
    *,
    expected_length: int,
) -> np.ndarray:
    """Validate optional transaction amounts."""
    amounts = _as_float_vector(
        transaction_amounts,
        value_name="transaction_amounts",
    )

    if len(amounts) != expected_length:
        raise ValueError(
            "transaction_amounts must have the same "
            "length as y_true"
        )

    if (amounts < 0.0).any():
        raise ValueError(
            "transaction_amounts must not contain "
            "negative values"
        )

    return amounts


def evaluate_probability_metrics(
    y_true: ArrayLike,
    fraud_probabilities: ArrayLike,
) -> dict[str, float | int]:
    """Calculate threshold-independent ranking and loss metrics."""
    target, probabilities = _validate_binary_inputs(
        y_true,
        fraud_probabilities,
    )

    return {
        "transaction_count": int(len(target)),
        "fraud_count": int(target.sum()),
        "fraud_rate": float(target.mean()),
        "pr_auc": float(
            average_precision_score(
                target,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                target,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                target,
                probabilities,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(
                target,
                probabilities,
            )
        ),
    }


def evaluate_threshold_metrics(
    y_true: ArrayLike,
    fraud_probabilities: ArrayLike,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float | int]:
    """Calculate binary metrics at one reference threshold."""
    target, probabilities = _validate_binary_inputs(
        y_true,
        fraud_probabilities,
    )
    validated_threshold = _validate_threshold(
        threshold
    )

    predictions = (
        probabilities >= validated_threshold
    ).astype("int8")

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            target,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    negative_count = (
        int(true_negative)
        + int(false_positive)
    )
    positive_count = (
        int(true_positive)
        + int(false_negative)
    )

    return {
        "threshold": validated_threshold,
        "predicted_fraud_count": int(
            predictions.sum()
        ),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "precision": float(
            precision_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "false_positive_rate": (
            float(false_positive / negative_count)
            if negative_count
            else 0.0
        ),
        "false_negative_rate": (
            float(false_negative / positive_count)
            if positive_count
            else 0.0
        ),
    }


def evaluate_review_capacity(
    y_true: ArrayLike,
    fraud_probabilities: ArrayLike,
    *,
    review_fractions: Sequence[float] = (
        DEFAULT_REVIEW_FRACTIONS
    ),
    transaction_amounts: ArrayLike | None = None,
) -> list[dict[str, float | int]]:
    """Measure fraud capture in fixed-size review queues."""
    target, probabilities = _validate_binary_inputs(
        y_true,
        fraud_probabilities,
    )
    fractions = _validate_review_fractions(
        review_fractions
    )

    amounts = None

    if transaction_amounts is not None:
        amounts = _validate_transaction_amounts(
            transaction_amounts,
            expected_length=len(target),
        )

    ranked_indices = np.argsort(
        -probabilities,
        kind="stable",
    )

    total_fraud_count = int(target.sum())

    total_fraud_amount = None

    if amounts is not None:
        total_fraud_amount = float(
            amounts[target == 1].sum()
        )

    records: list[
        dict[str, float | int]
    ] = []

    for review_fraction in fractions:
        reviewed_count = min(
            len(target),
            max(
                1,
                math.ceil(
                    len(target) * review_fraction
                ),
            ),
        )

        reviewed_indices = ranked_indices[
            :reviewed_count
        ]
        reviewed_target = target[
            reviewed_indices
        ]

        frauds_reviewed = int(
            reviewed_target.sum()
        )

        record: dict[str, float | int] = {
            "review_fraction": review_fraction,
            "reviewed_count": reviewed_count,
            "actual_review_rate": float(
                reviewed_count / len(target)
            ),
            "frauds_reviewed": frauds_reviewed,
            "fraud_recall": float(
                frauds_reviewed
                / total_fraud_count
            ),
            "review_precision": float(
                frauds_reviewed
                / reviewed_count
            ),
        }

        if (
            amounts is not None
            and total_fraud_amount is not None
        ):
            reviewed_fraud_amount = float(
                amounts[
                    reviewed_indices
                ][
                    reviewed_target == 1
                ].sum()
            )

            record[
                "fraud_amount_captured"
            ] = reviewed_fraud_amount
            record[
                "fraud_amount_recall"
            ] = (
                float(
                    reviewed_fraud_amount
                    / total_fraud_amount
                )
                if total_fraud_amount > 0.0
                else 0.0
            )

        records.append(record)

    return records


def evaluate_fraud_model(
    y_true: ArrayLike,
    fraud_probabilities: ArrayLike,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    review_fractions: Sequence[float] = (
        DEFAULT_REVIEW_FRACTIONS
    ),
    transaction_amounts: ArrayLike | None = None,
) -> dict[str, Any]:
    """Build a complete JSON-compatible fraud evaluation report."""
    return {
        "probability_metrics": (
            evaluate_probability_metrics(
                y_true,
                fraud_probabilities,
            )
        ),
        "threshold_metrics": (
            evaluate_threshold_metrics(
                y_true,
                fraud_probabilities,
                threshold=threshold,
            )
        ),
        "review_capacity": (
            evaluate_review_capacity(
                y_true,
                fraud_probabilities,
                review_fractions=review_fractions,
                transaction_amounts=(
                    transaction_amounts
                ),
            )
        ),
    }
