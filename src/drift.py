"""Temporal score, performance, and categorical drift diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.calibration_metrics import (
    DEFAULT_BIN_COUNT,
    build_reliability_table,
    calculate_calibration_errors,
)
from src.evaluation import evaluate_probability_metrics
from src.preprocessing import CategoricalEncoder


DEFAULT_PSI_BIN_COUNT = 10
DEFAULT_PSI_EPSILON = 1e-6
DEFAULT_TEMPORAL_WINDOW_COUNT = 5


def _as_probability_vector(
    values: ArrayLike,
    *,
    value_name: str,
) -> np.ndarray:
    """Return a finite one-dimensional probability vector."""

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

    if (
        (vector < 0.0).any()
        or (vector > 1.0).any()
    ):
        raise ValueError(
            f"{value_name} must be between 0 and 1"
        )

    return vector


def _validate_binary_target(
    y_true: ArrayLike,
    *,
    expected_length: int,
    value_name: str = "y_true",
) -> np.ndarray:
    """Return an aligned binary target containing both classes."""

    try:
        target = np.asarray(
            y_true,
            dtype="float64",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{value_name} must contain numerical values"
        ) from error

    if target.ndim != 1:
        raise ValueError(
            f"{value_name} must be one-dimensional"
        )

    if len(target) != expected_length:
        raise ValueError(
            f"{value_name} must have length "
            f"{expected_length}"
        )

    if not np.isfinite(target).all():
        raise ValueError(
            f"{value_name} must contain only finite values"
        )

    if not np.isin(
        target,
        [0.0, 1.0],
    ).all():
        raise ValueError(
            f"{value_name} must contain only 0 and 1"
        )

    normalized_target = target.astype("int8")

    if set(normalized_target.tolist()) != {0, 1}:
        raise ValueError(
            f"{value_name} must contain both binary classes"
        )

    return normalized_target


def _validate_positive_integer(
    value: int,
    *,
    value_name: str,
) -> int:
    """Return a positive integer configuration value."""

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, np.integer),
        )
        or int(value) <= 0
    ):
        raise ValueError(
            f"{value_name} must be a positive integer"
        )

    return int(value)


def _validate_psi_epsilon(
    epsilon: float,
) -> float:
    """Return a valid PSI zero-share replacement."""

    try:
        validated_epsilon = float(epsilon)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "epsilon must be numerical"
        ) from error

    if (
        not math.isfinite(validated_epsilon)
        or not 0.0 < validated_epsilon < 0.5
    ):
        raise ValueError(
            "epsilon must be finite and between 0 and 0.5"
        )

    return validated_epsilon


def _python_scalar(value: object) -> object:
    """Convert NumPy-like scalar values to Python values."""

    item_method = getattr(
        value,
        "item",
        None,
    )

    if callable(item_method):
        return item_method()

    return value


def build_psi_reference_edges(
    reference_scores: ArrayLike,
    *,
    bin_count: int = DEFAULT_PSI_BIN_COUNT,
) -> np.ndarray:
    """Build deterministic reference-quantile PSI edges."""

    scores = _as_probability_vector(
        reference_scores,
        value_name="reference_scores",
    )
    validated_bin_count = _validate_positive_integer(
        bin_count,
        value_name="bin_count",
    )

    if validated_bin_count < 2:
        raise ValueError(
            "bin_count must be at least 2"
        )

    quantile_edges = np.quantile(
        scores,
        np.linspace(
            0.0,
            1.0,
            validated_bin_count + 1,
        ),
        method="linear",
    ).astype(
        "float64",
        copy=False,
    )

    quantile_edges[0] = 0.0
    quantile_edges[-1] = 1.0

    edges = np.unique(
        quantile_edges
    )

    return edges.astype(
        "float64",
        copy=False,
    )


def calculate_population_stability_index(
    reference_scores: ArrayLike,
    comparison_scores: ArrayLike,
    *,
    bin_count: int = DEFAULT_PSI_BIN_COUNT,
    epsilon: float = DEFAULT_PSI_EPSILON,
) -> dict[str, Any]:
    """Compare two score distributions with PSI."""

    reference = _as_probability_vector(
        reference_scores,
        value_name="reference_scores",
    )
    comparison = _as_probability_vector(
        comparison_scores,
        value_name="comparison_scores",
    )
    validated_bin_count = _validate_positive_integer(
        bin_count,
        value_name="bin_count",
    )

    if validated_bin_count < 2:
        raise ValueError(
            "bin_count must be at least 2"
        )

    validated_epsilon = _validate_psi_epsilon(
        epsilon
    )
    bin_edges = build_psi_reference_edges(
        reference,
        bin_count=validated_bin_count,
    )

    reference_counts, _ = np.histogram(
        reference,
        bins=bin_edges,
    )
    comparison_counts, _ = np.histogram(
        comparison,
        bins=bin_edges,
    )

    reference_shares = (
        reference_counts.astype("float64")
        / len(reference)
    )
    comparison_shares = (
        comparison_counts.astype("float64")
        / len(comparison)
    )

    adjusted_reference = np.maximum(
        reference_shares,
        validated_epsilon,
    )
    adjusted_comparison = np.maximum(
        comparison_shares,
        validated_epsilon,
    )

    adjusted_reference = (
        adjusted_reference
        / adjusted_reference.sum()
    )
    adjusted_comparison = (
        adjusted_comparison
        / adjusted_comparison.sum()
    )

    contributions = (
        adjusted_comparison
        - adjusted_reference
    ) * np.log(
        adjusted_comparison
        / adjusted_reference
    )

    bin_records: list[
        dict[str, float | int]
    ] = []

    for bin_index in range(
        len(bin_edges) - 1
    ):
        bin_records.append(
            {
                "bin_index": bin_index,
                "lower_bound": float(
                    bin_edges[bin_index]
                ),
                "upper_bound": float(
                    bin_edges[bin_index + 1]
                ),
                "reference_count": int(
                    reference_counts[bin_index]
                ),
                "comparison_count": int(
                    comparison_counts[bin_index]
                ),
                "reference_share": float(
                    reference_shares[bin_index]
                ),
                "comparison_share": float(
                    comparison_shares[bin_index]
                ),
                "adjusted_reference_share": float(
                    adjusted_reference[bin_index]
                ),
                "adjusted_comparison_share": float(
                    adjusted_comparison[bin_index]
                ),
                "psi_contribution": float(
                    contributions[bin_index]
                ),
            }
        )

    return {
        "psi": float(
            contributions.sum()
        ),
        "requested_bin_count": (
            validated_bin_count
        ),
        "actual_bin_count": int(
            len(bin_edges) - 1
        ),
        "epsilon": validated_epsilon,
        "bin_edges": [
            float(value)
            for value in bin_edges
        ],
        "bins": bin_records,
    }


def evaluate_score_drift(
    reference_raw_scores: ArrayLike,
    comparison_raw_scores: ArrayLike,
    reference_calibrated_probabilities: ArrayLike,
    comparison_calibrated_probabilities: ArrayLike,
    *,
    bin_count: int = DEFAULT_PSI_BIN_COUNT,
    epsilon: float = DEFAULT_PSI_EPSILON,
) -> dict[str, Any]:
    """Report raw and calibrated score PSI separately."""

    reference_raw = _as_probability_vector(
        reference_raw_scores,
        value_name="reference_raw_scores",
    )
    comparison_raw = _as_probability_vector(
        comparison_raw_scores,
        value_name="comparison_raw_scores",
    )
    reference_calibrated = (
        _as_probability_vector(
            reference_calibrated_probabilities,
            value_name=(
                "reference_calibrated_probabilities"
            ),
        )
    )
    comparison_calibrated = (
        _as_probability_vector(
            comparison_calibrated_probabilities,
            value_name=(
                "comparison_calibrated_probabilities"
            ),
        )
    )

    if len(reference_raw) != len(
        reference_calibrated
    ):
        raise ValueError(
            "Reference raw and calibrated scores "
            "must have the same length"
        )

    if len(comparison_raw) != len(
        comparison_calibrated
    ):
        raise ValueError(
            "Comparison raw and calibrated scores "
            "must have the same length"
        )

    return {
        "raw_score_psi": (
            calculate_population_stability_index(
                reference_raw,
                comparison_raw,
                bin_count=bin_count,
                epsilon=epsilon,
            )
        ),
        "calibrated_score_psi": (
            calculate_population_stability_index(
                reference_calibrated,
                comparison_calibrated,
                bin_count=bin_count,
                epsilon=epsilon,
            )
        ),
    }


def _validate_transaction_ids(
    transaction_ids: ArrayLike,
) -> np.ndarray:
    """Return complete unique transaction identifiers."""

    identifiers = np.asarray(
        transaction_ids,
        dtype="object",
    )

    if identifiers.ndim != 1:
        raise ValueError(
            "transaction_ids must be one-dimensional"
        )

    if identifiers.size == 0:
        raise ValueError(
            "transaction_ids must not be empty"
        )

    if pd.isna(identifiers).any():
        raise ValueError(
            "transaction_ids must not contain missing values"
        )

    if pd.Index(identifiers).duplicated().any():
        raise ValueError(
            "transaction_ids must not contain duplicates"
        )

    return identifiers


def _validate_decision_thresholds(
    review_threshold: float | None,
    block_threshold: float | None,
) -> tuple[float, float] | None:
    """Validate optional decision-volume thresholds."""

    if (
        review_threshold is None
        and block_threshold is None
    ):
        return None

    if (
        review_threshold is None
        or block_threshold is None
    ):
        raise ValueError(
            "review_threshold and block_threshold "
            "must be provided together"
        )

    try:
        validated_review = float(
            review_threshold
        )
        validated_block = float(
            block_threshold
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "decision thresholds must be numerical"
        ) from error

    if (
        not math.isfinite(validated_review)
        or not math.isfinite(validated_block)
    ):
        raise ValueError(
            "decision thresholds must be finite"
        )

    if not (
        0.0
        <= validated_review
        <= 1.0
        and 0.0
        <= validated_block
        <= 1.0
    ):
        raise ValueError(
            "decision thresholds must be between 0 and 1"
        )

    if validated_review > validated_block:
        raise ValueError(
            "review_threshold must be <= block_threshold"
        )

    return (
        validated_review,
        validated_block,
    )


def _decision_volume_metrics(
    probabilities: np.ndarray,
    *,
    review_threshold: float,
    block_threshold: float,
) -> dict[str, float | int]:
    """Calculate decision counts using existing semantics."""

    block_mask = (
        probabilities >= block_threshold
    )
    review_mask = (
        (probabilities >= review_threshold)
        & ~block_mask
    )
    allow_mask = ~(
        review_mask | block_mask
    )

    row_count = len(probabilities)

    allow_count = int(
        allow_mask.sum()
    )
    review_count = int(
        review_mask.sum()
    )
    block_count = int(
        block_mask.sum()
    )

    return {
        "review_threshold": review_threshold,
        "block_threshold": block_threshold,
        "allow_count": allow_count,
        "allow_rate": float(
            allow_count / row_count
        ),
        "review_count": review_count,
        "review_rate": float(
            review_count / row_count
        ),
        "block_count": block_count,
        "block_rate": float(
            block_count / row_count
        ),
    }


def evaluate_temporal_performance_windows(
    transaction_ids: ArrayLike,
    y_true: ArrayLike,
    raw_scores: ArrayLike,
    calibrated_probabilities: ArrayLike,
    *,
    window_count: int = (
        DEFAULT_TEMPORAL_WINDOW_COUNT
    ),
    calibration_bin_count: int = (
        DEFAULT_BIN_COUNT
    ),
    review_threshold: float | None = None,
    block_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Evaluate contiguous chronological row-count windows."""

    identifiers = _validate_transaction_ids(
        transaction_ids
    )
    raw = _as_probability_vector(
        raw_scores,
        value_name="raw_scores",
    )
    calibrated = _as_probability_vector(
        calibrated_probabilities,
        value_name="calibrated_probabilities",
    )

    row_count = len(identifiers)

    if len(raw) != row_count:
        raise ValueError(
            "raw_scores must align with transaction_ids"
        )

    if len(calibrated) != row_count:
        raise ValueError(
            "calibrated_probabilities must align "
            "with transaction_ids"
        )

    target = _validate_binary_target(
        y_true,
        expected_length=row_count,
    )
    validated_window_count = (
        _validate_positive_integer(
            window_count,
            value_name="window_count",
        )
    )
    validated_bin_count = (
        _validate_positive_integer(
            calibration_bin_count,
            value_name="calibration_bin_count",
        )
    )

    if validated_bin_count < 2:
        raise ValueError(
            "calibration_bin_count must be at least 2"
        )

    if validated_window_count > row_count:
        raise ValueError(
            "window_count must not exceed row count"
        )

    decision_thresholds = (
        _validate_decision_thresholds(
            review_threshold,
            block_threshold,
        )
    )

    index_windows = np.array_split(
        np.arange(row_count),
        validated_window_count,
    )

    records: list[
        dict[str, Any]
    ] = []

    for window_index, indices in enumerate(
        index_windows
    ):
        window_target = target[indices]

        if set(
            window_target.tolist()
        ) != {0, 1}:
            raise ValueError(
                f"Temporal window {window_index} "
                "must contain both binary classes"
            )

        window_raw = raw[indices]
        window_calibrated = calibrated[
            indices
        ]

        raw_metrics = (
            evaluate_probability_metrics(
                window_target,
                window_raw,
            )
        )
        calibrated_metrics = (
            evaluate_probability_metrics(
                window_target,
                window_calibrated,
            )
        )

        reliability_table = (
            build_reliability_table(
                window_target,
                window_calibrated,
                bin_count=validated_bin_count,
            )
        )
        calibration_errors = (
            calculate_calibration_errors(
                reliability_table
            )
        )

        first_position = int(
            indices[0]
        )
        stop_position = int(
            indices[-1] + 1
        )

        record: dict[str, Any] = {
            "window_index": window_index,
            "start_position": first_position,
            "stop_position_exclusive": (
                stop_position
            ),
            "first_transaction_id": (
                _python_scalar(
                    identifiers[first_position]
                )
            ),
            "last_transaction_id": (
                _python_scalar(
                    identifiers[
                        stop_position - 1
                    ]
                )
            ),
            "row_count": int(
                len(indices)
            ),
            "fraud_count": int(
                window_target.sum()
            ),
            "fraud_rate": float(
                window_target.mean()
            ),
            "mean_raw_score": float(
                window_raw.mean()
            ),
            "mean_calibrated_probability": (
                float(
                    window_calibrated.mean()
                )
            ),
            "raw_score_metrics": raw_metrics,
            "calibrated_probability_metrics": {
                **calibrated_metrics,
                **calibration_errors,
            },
        }

        if decision_thresholds is not None:
            (
                validated_review,
                validated_block,
            ) = decision_thresholds

            record["decision_metrics"] = (
                _decision_volume_metrics(
                    window_calibrated,
                    review_threshold=(
                        validated_review
                    ),
                    block_threshold=(
                        validated_block
                    ),
                )
            )

        records.append(record)

    return records


def _validate_segment_name(
    segment_name: str,
) -> str:
    """Return a non-empty diagnostic segment name."""

    if not isinstance(segment_name, str):
        raise ValueError(
            "segment_name must be a string"
        )

    normalized_name = segment_name.strip()

    if not normalized_name:
        raise ValueError(
            "segment_name must not be empty"
        )

    return normalized_name


def evaluate_categorical_rates(
    features: pd.DataFrame,
    encoder: CategoricalEncoder,
    *,
    segment_name: str,
) -> list[dict[str, float | int | str]]:
    """Measure missing, unknown, and known category rates."""

    validated_segment_name = (
        _validate_segment_name(
            segment_name
        )
    )

    if encoder.missing_code == (
        encoder.unknown_code
    ):
        raise ValueError(
            "Encoder missing and unknown codes "
            "must be different"
        )

    encoded_features = encoder.transform(
        features,
        frame_name=(
            f"{validated_segment_name} features"
        ),
    )

    row_count = len(encoded_features)
    records: list[
        dict[str, float | int | str]
    ] = []

    for feature_name in (
        encoder.categorical_features
    ):
        encoded_values = encoded_features[
            feature_name
        ].to_numpy()

        missing_count = int(
            (
                encoded_values
                == encoder.missing_code
            ).sum()
        )
        unknown_count = int(
            (
                encoded_values
                == encoder.unknown_code
            ).sum()
        )
        known_count = (
            row_count
            - missing_count
            - unknown_count
        )

        records.append(
            {
                "segment_name": (
                    validated_segment_name
                ),
                "feature_name": feature_name,
                "row_count": row_count,
                "missing_count": missing_count,
                "missing_rate": float(
                    missing_count / row_count
                ),
                "unknown_count": unknown_count,
                "unknown_rate": float(
                    unknown_count / row_count
                ),
                "known_count": known_count,
                "known_rate": float(
                    known_count / row_count
                ),
            }
        )

    return records


def compare_categorical_rates(
    reference_features: pd.DataFrame,
    comparison_features: pd.DataFrame,
    encoder: CategoricalEncoder,
    *,
    reference_name: str = "calibration_fit",
    comparison_name: str = "comparison",
) -> list[dict[str, float | int | str]]:
    """Compare category-state rates with a reference segment."""

    reference_records = (
        evaluate_categorical_rates(
            reference_features,
            encoder,
            segment_name=reference_name,
        )
    )
    comparison_records = (
        evaluate_categorical_rates(
            comparison_features,
            encoder,
            segment_name=comparison_name,
        )
    )

    reference_by_feature = {
        str(record["feature_name"]): record
        for record in reference_records
    }
    comparison_by_feature = {
        str(record["feature_name"]): record
        for record in comparison_records
    }

    records: list[
        dict[str, float | int | str]
    ] = []

    for feature_name in (
        encoder.categorical_features
    ):
        reference = reference_by_feature[
            feature_name
        ]
        comparison = comparison_by_feature[
            feature_name
        ]

        records.append(
            {
                "feature_name": feature_name,
                "reference_segment": (
                    str(
                        reference[
                            "segment_name"
                        ]
                    )
                ),
                "comparison_segment": (
                    str(
                        comparison[
                            "segment_name"
                        ]
                    )
                ),
                "reference_row_count": int(
                    reference["row_count"]
                ),
                "comparison_row_count": int(
                    comparison["row_count"]
                ),
                "reference_missing_count": int(
                    reference["missing_count"]
                ),
                "comparison_missing_count": int(
                    comparison["missing_count"]
                ),
                "reference_missing_rate": float(
                    reference["missing_rate"]
                ),
                "comparison_missing_rate": float(
                    comparison["missing_rate"]
                ),
                "missing_rate_change": float(
                    comparison["missing_rate"]
                    - reference["missing_rate"]
                ),
                "reference_unknown_count": int(
                    reference["unknown_count"]
                ),
                "comparison_unknown_count": int(
                    comparison["unknown_count"]
                ),
                "reference_unknown_rate": float(
                    reference["unknown_rate"]
                ),
                "comparison_unknown_rate": float(
                    comparison["unknown_rate"]
                ),
                "unknown_rate_change": float(
                    comparison["unknown_rate"]
                    - reference["unknown_rate"]
                ),
                "reference_known_count": int(
                    reference["known_count"]
                ),
                "comparison_known_count": int(
                    comparison["known_count"]
                ),
                "reference_known_rate": float(
                    reference["known_rate"]
                ),
                "comparison_known_rate": float(
                    comparison["known_rate"]
                ),
                "known_rate_change": float(
                    comparison["known_rate"]
                    - reference["known_rate"]
                ),
            }
        )

    return records
