"""Probability-calibration diagnostics for binary fraud models."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.evaluation import evaluate_probability_metrics


DEFAULT_BIN_COUNT = 10
DEFAULT_CLIP_EPSILON = 1e-6
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_TOLERANCE = 1e-10


def _as_float_vector(
    values: ArrayLike,
    *,
    value_name: str,
) -> np.ndarray:
    """Convert input into a finite one-dimensional float vector."""

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


def _validate_inputs(
    y_true: ArrayLike,
    fraud_probabilities: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned binary outcomes and probabilities."""

    target = _as_float_vector(
        y_true,
        value_name="y_true",
    )
    probabilities = _as_float_vector(
        fraud_probabilities,
        value_name="fraud_probabilities",
    )

    if len(target) != len(probabilities):
        raise ValueError(
            "y_true and fraud_probabilities must have "
            "the same length"
        )

    if not np.isin(
        target,
        [0.0, 1.0],
    ).all():
        raise ValueError(
            "y_true must contain only binary values 0 and 1"
        )

    if set(target.astype("int8").tolist()) != {0, 1}:
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
        target.astype("int8"),
        probabilities,
    )


def _validate_bin_count(bin_count: int) -> int:
    """Return a valid fixed-width reliability-bin count."""

    if (
        isinstance(bin_count, bool)
        or not isinstance(
            bin_count,
            (int, np.integer),
        )
    ):
        raise ValueError(
            "bin_count must be an integer"
        )

    validated_bin_count = int(bin_count)

    if validated_bin_count < 2:
        raise ValueError(
            "bin_count must be at least 2"
        )

    return validated_bin_count


def _validate_clip_epsilon(
    clip_epsilon: float,
) -> float:
    """Return a valid probability clipping epsilon."""

    try:
        validated_epsilon = float(clip_epsilon)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "clip_epsilon must be numerical"
        ) from error

    if not math.isfinite(validated_epsilon):
        raise ValueError(
            "clip_epsilon must be finite"
        )

    if not 0.0 < validated_epsilon < 0.5:
        raise ValueError(
            "clip_epsilon must be between 0 and 0.5"
        )

    return validated_epsilon


def _validate_max_iterations(
    max_iterations: int,
) -> int:
    """Return a valid calibration-regression iteration limit."""

    if (
        isinstance(max_iterations, bool)
        or not isinstance(
            max_iterations,
            (int, np.integer),
        )
        or int(max_iterations) <= 0
    ):
        raise ValueError(
            "max_iterations must be a positive integer"
        )

    return int(max_iterations)


def _validate_tolerance(
    tolerance: float,
) -> float:
    """Return a valid calibration-regression tolerance."""

    try:
        validated_tolerance = float(tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "tolerance must be numerical"
        ) from error

    if not math.isfinite(validated_tolerance):
        raise ValueError(
            "tolerance must be finite"
        )

    if validated_tolerance <= 0.0:
        raise ValueError(
            "tolerance must be greater than 0"
        )

    return validated_tolerance


def build_reliability_table(
    y_true: ArrayLike,
    fraud_probabilities: ArrayLike,
    *,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> list[dict[str, float | int | None]]:
    """Build fixed-width probability reliability bins."""

    target, probabilities = _validate_inputs(
        y_true,
        fraud_probabilities,
    )
    validated_bin_count = _validate_bin_count(
        bin_count
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        validated_bin_count + 1,
        dtype="float64",
    )

    bin_indices = np.searchsorted(
        bin_edges,
        probabilities,
        side="right",
    ) - 1

    bin_indices = np.clip(
        bin_indices,
        0,
        validated_bin_count - 1,
    )

    total_count = len(target)
    records: list[
        dict[str, float | int | None]
    ] = []

    for bin_index in range(validated_bin_count):
        bin_mask = bin_indices == bin_index
        transaction_count = int(bin_mask.sum())
        fraud_count = int(target[bin_mask].sum())

        mean_probability: float | None = None
        fraud_rate: float | None = None
        absolute_gap: float | None = None

        if transaction_count:
            mean_probability = float(
                probabilities[bin_mask].mean()
            )
            fraud_rate = float(
                target[bin_mask].mean()
            )
            absolute_gap = abs(
                mean_probability - fraud_rate
            )

        records.append(
            {
                "bin_index": bin_index,
                "lower_bound": float(
                    bin_edges[bin_index]
                ),
                "upper_bound": float(
                    bin_edges[bin_index + 1]
                ),
                "transaction_count": transaction_count,
                "transaction_share": float(
                    transaction_count / total_count
                ),
                "mean_probability": mean_probability,
                "fraud_rate": fraud_rate,
                "absolute_gap": absolute_gap,
                "fraud_count": fraud_count,
            }
        )

    return records


def calculate_calibration_errors(
    reliability_table: list[
        dict[str, float | int | None]
    ],
) -> dict[str, float]:
    """Calculate ECE and MCE from a reliability table."""

    if not reliability_table:
        raise ValueError(
            "reliability_table must not be empty"
        )

    expected_error = 0.0
    non_empty_gaps: list[float] = []

    for record in reliability_table:
        transaction_count = record.get(
            "transaction_count"
        )
        transaction_share = record.get(
            "transaction_share"
        )
        absolute_gap = record.get(
            "absolute_gap"
        )

        if (
            not isinstance(transaction_count, int)
            or transaction_count < 0
        ):
            raise ValueError(
                "reliability_table contains an invalid "
                "transaction_count"
            )

        if not isinstance(
            transaction_share,
            (int, float),
        ):
            raise ValueError(
                "reliability_table contains an invalid "
                "transaction_share"
            )

        if transaction_count == 0:
            if absolute_gap is not None:
                raise ValueError(
                    "empty reliability bins must have "
                    "no absolute_gap"
                )
            continue

        if not isinstance(
            absolute_gap,
            (int, float),
        ):
            raise ValueError(
                "non-empty reliability bins must have "
                "an absolute_gap"
            )

        validated_share = float(
            transaction_share
        )
        validated_gap = float(
            absolute_gap
        )

        if (
            not math.isfinite(validated_share)
            or not 0.0 <= validated_share <= 1.0
        ):
            raise ValueError(
                "transaction_share must be between 0 and 1"
            )

        if (
            not math.isfinite(validated_gap)
            or validated_gap < 0.0
        ):
            raise ValueError(
                "absolute_gap must be finite and non-negative"
            )

        expected_error += (
            validated_share * validated_gap
        )
        non_empty_gaps.append(validated_gap)

    if not non_empty_gaps:
        raise ValueError(
            "reliability_table must contain a non-empty bin"
        )

    return {
        "expected_calibration_error": float(
            expected_error
        ),
        "maximum_calibration_error": float(
            max(non_empty_gaps)
        ),
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Calculate a numerically stable logistic sigmoid."""

    result = np.empty_like(
        values,
        dtype="float64",
    )
    non_negative = values >= 0.0

    result[non_negative] = 1.0 / (
        1.0 + np.exp(-values[non_negative])
    )

    negative_exponential = np.exp(
        values[~non_negative]
    )
    result[~non_negative] = (
        negative_exponential
        / (1.0 + negative_exponential)
    )

    return result


def estimate_calibration_regression(
    y_true: ArrayLike,
    fraud_probabilities: ArrayLike,
    *,
    clip_epsilon: float = DEFAULT_CLIP_EPSILON,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, float | int]:
    """Estimate calibration intercept and slope with logistic IRLS."""

    target, probabilities = _validate_inputs(
        y_true,
        fraud_probabilities,
    )
    validated_epsilon = _validate_clip_epsilon(
        clip_epsilon
    )
    validated_max_iterations = (
        _validate_max_iterations(
            max_iterations
        )
    )
    validated_tolerance = _validate_tolerance(
        tolerance
    )

    clipped_probabilities = np.clip(
        probabilities,
        validated_epsilon,
        1.0 - validated_epsilon,
    )

    probability_logits = np.log(
        clipped_probabilities
        / (1.0 - clipped_probabilities)
    )

    if np.ptp(probability_logits) == 0.0:
        raise ValueError(
            "calibration regression requires varying "
            "fraud probabilities"
        )

    design_matrix = np.column_stack(
        (
            np.ones(
                len(probability_logits),
                dtype="float64",
            ),
            probability_logits,
        )
    )

    coefficients = np.zeros(
        2,
        dtype="float64",
    )

    for iteration in range(
        1,
        validated_max_iterations + 1,
    ):
        linear_predictor = (
            design_matrix @ coefficients
        )
        fitted_probabilities = _sigmoid(
            linear_predictor
        )

        weights = (
            fitted_probabilities
            * (1.0 - fitted_probabilities)
        )
        weights = np.clip(
            weights,
            np.finfo("float64").eps,
            None,
        )

        gradient = design_matrix.T @ (
            target - fitted_probabilities
        )
        information_matrix = (
            design_matrix.T
            @ (
                weights[:, np.newaxis]
                * design_matrix
            )
        )

        try:
            step = np.linalg.solve(
                information_matrix,
                gradient,
            )
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "calibration regression could not "
                "estimate unique coefficients"
            ) from error

        if not np.isfinite(step).all():
            raise ValueError(
                "calibration regression produced "
                "non-finite coefficients"
            )

        coefficients = coefficients + step

        if np.max(np.abs(step)) <= (
            validated_tolerance
        ):
            return {
                "calibration_intercept": float(
                    coefficients[0]
                ),
                "calibration_slope": float(
                    coefficients[1]
                ),
                "probability_clip_epsilon": (
                    validated_epsilon
                ),
                "iterations": iteration,
            }

    raise RuntimeError(
        "calibration regression did not converge "
        f"within {validated_max_iterations} iterations"
    )


def evaluate_calibration(
    y_true: ArrayLike,
    fraud_probabilities: ArrayLike,
    *,
    bin_count: int = DEFAULT_BIN_COUNT,
    clip_epsilon: float = DEFAULT_CLIP_EPSILON,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Build a JSON-compatible probability-calibration report."""

    target, probabilities = _validate_inputs(
        y_true,
        fraud_probabilities,
    )

    reliability_table = build_reliability_table(
        target,
        probabilities,
        bin_count=bin_count,
    )
    calibration_errors = (
        calculate_calibration_errors(
            reliability_table
        )
    )
    regression_metrics = (
        estimate_calibration_regression(
            target,
            probabilities,
            clip_epsilon=clip_epsilon,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    )

    probability_metrics = (
        evaluate_probability_metrics(
            target,
            probabilities,
        )
    )
    probability_metrics.update(
        {
            "minimum_probability": float(
                probabilities.min()
            ),
            "maximum_probability": float(
                probabilities.max()
            ),
            "mean_probability": float(
                probabilities.mean()
            ),
        }
    )

    return {
        "probability_metrics": probability_metrics,
        "calibration_metrics": {
            "bin_count": len(reliability_table),
            **calibration_errors,
            **regression_metrics,
        },
        "reliability_table": reliability_table,
    }
