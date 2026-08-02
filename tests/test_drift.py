"""Tests for temporal fraud drift diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.drift import (
    calculate_population_stability_index,
    compare_categorical_rates,
    evaluate_categorical_rates,
    evaluate_score_drift,
    evaluate_temporal_performance_windows,
)
from src.model_data import FeatureContract
from src.preprocessing import CategoricalEncoder


FEATURE_CONTRACT = FeatureContract(
    feature_columns=(
        "category_a",
        "numeric_feature",
        "category_b",
    ),
    categorical_features=(
        "category_a",
        "category_b",
    ),
    numerical_features=(
        "numeric_feature",
    ),
)


def _training_features() -> pd.DataFrame:
    """Return categorical encoder training data."""

    return pd.DataFrame(
        {
            "category_a": [
                "a",
                "b",
                None,
                "a",
            ],
            "numeric_feature": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "category_b": [
                "x",
                "y",
                "x",
                None,
            ],
        }
    )


def _encoder() -> CategoricalEncoder:
    """Fit the frozen training-only category mapping."""

    return CategoricalEncoder.fit(
        _training_features(),
        FEATURE_CONTRACT,
    )


def _temporal_arrays() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return three chronological four-row windows."""

    transaction_ids = np.arange(
        100,
        112,
    )

    target = np.tile(
        np.array(
            [0, 1, 0, 1],
            dtype="int8",
        ),
        3,
    )

    raw_scores = np.tile(
        np.array(
            [0.15, 0.45, 0.70, 0.90],
            dtype="float64",
        ),
        3,
    )

    calibrated_probabilities = np.tile(
        np.array(
            [0.10, 0.40, 0.80, 0.95],
            dtype="float64",
        ),
        3,
    )

    return (
        transaction_ids,
        target,
        raw_scores,
        calibrated_probabilities,
    )


def test_population_stability_index_is_zero_for_identical_data(
) -> None:
    scores = [
        0.05,
        0.10,
        0.20,
        0.40,
        0.70,
        0.90,
    ]

    result = calculate_population_stability_index(
        scores,
        scores,
        bin_count=4,
    )

    assert result["psi"] == pytest.approx(
        0.0
    )
    assert result["requested_bin_count"] == 4
    assert result["actual_bin_count"] <= 4
    assert sum(
        record["reference_count"]
        for record in result["bins"]
    ) == len(scores)
    assert sum(
        record["comparison_count"]
        for record in result["bins"]
    ) == len(scores)

    json.dumps(result)


def test_population_stability_index_detects_shift_deterministically(
) -> None:
    reference = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
    ]
    comparison = [
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    first = calculate_population_stability_index(
        reference,
        comparison,
        bin_count=4,
    )
    second = calculate_population_stability_index(
        reference,
        comparison,
        bin_count=4,
    )

    assert first == second
    assert first["psi"] > 0.0


def test_population_stability_index_removes_duplicate_edges(
) -> None:
    result = calculate_population_stability_index(
        [0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2],
        bin_count=4,
    )

    assert result["requested_bin_count"] == 4
    assert result["actual_bin_count"] == 2
    assert result["bin_edges"] == [
        0.0,
        0.2,
        1.0,
    ]
    assert result["psi"] == pytest.approx(
        0.0
    )


def test_score_drift_reports_raw_and_calibrated_psi(
) -> None:
    result = evaluate_score_drift(
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.05, 0.10, 0.20, 0.30],
        [0.40, 0.50, 0.60, 0.70],
        bin_count=4,
    )

    assert set(result) == {
        "raw_score_psi",
        "calibrated_score_psi",
    }
    assert result[
        "raw_score_psi"
    ]["psi"] > 0.0
    assert result[
        "calibrated_score_psi"
    ]["psi"] > 0.0


def test_temporal_windows_preserve_order_and_metrics(
) -> None:
    (
        transaction_ids,
        target,
        raw_scores,
        calibrated_probabilities,
    ) = _temporal_arrays()

    windows = (
        evaluate_temporal_performance_windows(
            transaction_ids,
            target,
            raw_scores,
            calibrated_probabilities,
            window_count=3,
            calibration_bin_count=5,
        )
    )

    assert len(windows) == 3

    assert [
        window["first_transaction_id"]
        for window in windows
    ] == [
        100,
        104,
        108,
    ]
    assert [
        window["last_transaction_id"]
        for window in windows
    ] == [
        103,
        107,
        111,
    ]

    first = windows[0]

    assert first["start_position"] == 0
    assert first["stop_position_exclusive"] == 4
    assert first["row_count"] == 4
    assert first["fraud_count"] == 2
    assert first["fraud_rate"] == 0.5
    assert first["mean_raw_score"] == (
        pytest.approx(0.55)
    )
    assert first[
        "mean_calibrated_probability"
    ] == pytest.approx(0.5625)

    assert first[
        "raw_score_metrics"
    ]["transaction_count"] == 4
    assert first[
        "calibrated_probability_metrics"
    ]["transaction_count"] == 4
    assert first[
        "calibrated_probability_metrics"
    ][
        "expected_calibration_error"
    ] >= 0.0

    assert "decision_metrics" not in first

    json.dumps(windows)


def test_temporal_windows_include_decision_rates(
) -> None:
    (
        transaction_ids,
        target,
        raw_scores,
        calibrated_probabilities,
    ) = _temporal_arrays()

    windows = (
        evaluate_temporal_performance_windows(
            transaction_ids,
            target,
            raw_scores,
            calibrated_probabilities,
            window_count=3,
            review_threshold=0.30,
            block_threshold=0.80,
        )
    )

    for window in windows:
        decision_metrics = window[
            "decision_metrics"
        ]

        assert decision_metrics[
            "allow_count"
        ] == 1
        assert decision_metrics[
            "review_count"
        ] == 1
        assert decision_metrics[
            "block_count"
        ] == 2

        assert decision_metrics[
            "allow_rate"
        ] == 0.25
        assert decision_metrics[
            "review_rate"
        ] == 0.25
        assert decision_metrics[
            "block_rate"
        ] == 0.50


def test_temporal_windows_require_both_classes(
) -> None:
    (
        transaction_ids,
        target,
        raw_scores,
        calibrated_probabilities,
    ) = _temporal_arrays()

    target[:4] = 0

    with pytest.raises(
        ValueError,
        match=(
            "Temporal window 0 must contain "
            "both binary classes"
        ),
    ):
        evaluate_temporal_performance_windows(
            transaction_ids,
            target,
            raw_scores,
            calibrated_probabilities,
            window_count=3,
        )


@pytest.mark.parametrize(
    (
        "review_threshold",
        "block_threshold",
        "message",
    ),
    [
        (
            0.30,
            None,
            "provided together",
        ),
        (
            0.80,
            0.30,
            "must be <=",
        ),
        (
            -0.10,
            0.80,
            "between 0 and 1",
        ),
    ],
)
def test_temporal_windows_reject_invalid_thresholds(
    review_threshold: float | None,
    block_threshold: float | None,
    message: str,
) -> None:
    (
        transaction_ids,
        target,
        raw_scores,
        calibrated_probabilities,
    ) = _temporal_arrays()

    with pytest.raises(
        ValueError,
        match=message,
    ):
        evaluate_temporal_performance_windows(
            transaction_ids,
            target,
            raw_scores,
            calibrated_probabilities,
            window_count=3,
            review_threshold=review_threshold,
            block_threshold=block_threshold,
        )


def test_temporal_windows_reject_duplicate_transaction_ids(
) -> None:
    (
        transaction_ids,
        target,
        raw_scores,
        calibrated_probabilities,
    ) = _temporal_arrays()

    transaction_ids[-1] = transaction_ids[0]

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        evaluate_temporal_performance_windows(
            transaction_ids,
            target,
            raw_scores,
            calibrated_probabilities,
            window_count=3,
        )


def test_categorical_rates_distinguish_category_states(
) -> None:
    encoder = _encoder()

    features = pd.DataFrame(
        {
            "category_a": [
                "a",
                "new-a",
                None,
                "b",
            ],
            "numeric_feature": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "category_b": [
                "x",
                None,
                "new-b",
                "y",
            ],
        }
    )

    records = evaluate_categorical_rates(
        features,
        encoder,
        segment_name="policy_selection",
    )

    assert [
        record["feature_name"]
        for record in records
    ] == [
        "category_a",
        "category_b",
    ]

    for record in records:
        assert record["row_count"] == 4
        assert record["missing_count"] == 1
        assert record["unknown_count"] == 1
        assert record["known_count"] == 2
        assert record["missing_rate"] == 0.25
        assert record["unknown_rate"] == 0.25
        assert record["known_rate"] == 0.50


def test_categorical_comparison_reports_rate_changes_without_vocab_growth(
) -> None:
    encoder = _encoder()
    original_vocabularies = dict(
        encoder.category_vocabularies
    )

    reference = pd.DataFrame(
        {
            "category_a": [
                "a",
                "b",
                None,
                "a",
            ],
            "numeric_feature": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "category_b": [
                "x",
                "y",
                "x",
                None,
            ],
        }
    )

    comparison = pd.DataFrame(
        {
            "category_a": [
                "new-a",
                "new-b",
                None,
                "a",
            ],
            "numeric_feature": [
                5.0,
                6.0,
                7.0,
                8.0,
            ],
            "category_b": [
                "new-x",
                "new-y",
                None,
                "x",
            ],
        }
    )

    records = compare_categorical_rates(
        reference,
        comparison,
        encoder,
        reference_name="calibration_fit",
        comparison_name="policy_selection",
    )

    assert len(records) == 2

    for record in records:
        assert record[
            "reference_unknown_rate"
        ] == 0.0
        assert record[
            "comparison_unknown_rate"
        ] == 0.5
        assert record[
            "unknown_rate_change"
        ] == 0.5

        assert record[
            "reference_missing_rate"
        ] == 0.25
        assert record[
            "comparison_missing_rate"
        ] == 0.25
        assert record[
            "missing_rate_change"
        ] == 0.0

        assert record[
            "known_rate_change"
        ] == -0.5

    assert encoder.category_vocabularies == (
        original_vocabularies
    )

    json.dumps(records)


def test_categorical_rates_reject_feature_order_mismatch(
) -> None:
    encoder = _encoder()

    reordered = _training_features().loc[
        :,
        [
            "numeric_feature",
            "category_a",
            "category_b",
        ],
    ]

    with pytest.raises(
        ValueError,
        match="expected feature order",
    ):
        evaluate_categorical_rates(
            reordered,
            encoder,
            segment_name="comparison",
        )


@pytest.mark.parametrize(
    ("reference", "comparison", "message"),
    [
        (
            [-0.1, 0.8],
            [0.2, 0.8],
            "between 0 and 1",
        ),
        (
            [0.2, 0.8],
            [0.2, np.nan],
            "finite values",
        ),
        (
            [[0.2], [0.8]],
            [0.2, 0.8],
            "one-dimensional",
        ),
    ],
)
def test_psi_rejects_invalid_scores(
    reference: list[float],
    comparison: list[float],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        calculate_population_stability_index(
            reference,
            comparison,
        )
