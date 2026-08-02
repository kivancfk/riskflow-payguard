"""Tests for binary probability-calibration metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.calibration_metrics import (
    build_reliability_table,
    calculate_calibration_errors,
    estimate_calibration_regression,
    evaluate_calibration,
)


def test_reliability_table_preserves_fixed_bins() -> None:
    table = build_reliability_table(
        [0, 1, 0, 1],
        [0.05, 0.15, 0.85, 1.0],
    )

    assert len(table) == 10

    assert table[0] == {
        "bin_index": 0,
        "lower_bound": 0.0,
        "upper_bound": 0.1,
        "transaction_count": 1,
        "transaction_share": 0.25,
        "mean_probability": 0.05,
        "fraud_rate": 0.0,
        "absolute_gap": 0.05,
        "fraud_count": 0,
    }

    assert table[2]["transaction_count"] == 0
    assert table[2]["mean_probability"] is None
    assert table[2]["fraud_rate"] is None
    assert table[2]["absolute_gap"] is None

    assert table[8]["transaction_count"] == 1
    assert table[8]["mean_probability"] == 0.85

    assert table[9]["transaction_count"] == 1
    assert table[9]["mean_probability"] == 1.0
    assert table[9]["fraud_rate"] == 1.0


def test_calibration_errors_match_known_values() -> None:
    table = build_reliability_table(
        [0, 1, 0, 1],
        [0.05, 0.15, 0.85, 1.0],
    )

    errors = calculate_calibration_errors(table)

    assert errors[
        "expected_calibration_error"
    ] == pytest.approx(0.4375)

    assert errors[
        "maximum_calibration_error"
    ] == pytest.approx(0.85)


def test_calibration_regression_recovers_ideal_mapping() -> None:
    probabilities = (
        [0.2] * 5
        + [0.8] * 5
    )
    target = [
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
    ]

    metrics = estimate_calibration_regression(
        target,
        probabilities,
    )

    assert metrics[
        "calibration_intercept"
    ] == pytest.approx(
        0.0,
        abs=1e-8,
    )
    assert metrics[
        "calibration_slope"
    ] == pytest.approx(
        1.0,
        abs=1e-8,
    )
    assert metrics["iterations"] > 0


def test_evaluate_calibration_returns_complete_report() -> None:
    report = evaluate_calibration(
        [0, 0, 1, 0, 1, 1],
        [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        bin_count=5,
    )

    probability_metrics = report[
        "probability_metrics"
    ]
    calibration_metrics = report[
        "calibration_metrics"
    ]

    assert probability_metrics[
        "transaction_count"
    ] == 6
    assert probability_metrics["fraud_count"] == 3
    assert probability_metrics["fraud_rate"] == 0.5
    assert probability_metrics[
        "minimum_probability"
    ] == 0.1
    assert probability_metrics[
        "maximum_probability"
    ] == 0.9
    assert probability_metrics[
        "mean_probability"
    ] == pytest.approx(0.5)

    assert calibration_metrics["bin_count"] == 5
    assert (
        calibration_metrics[
            "expected_calibration_error"
        ]
        >= 0.0
    )
    assert (
        calibration_metrics[
            "maximum_calibration_error"
        ]
        >= 0.0
    )

    assert len(report["reliability_table"]) == 5

    json.dumps(report)


def test_calibration_metrics_reject_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        evaluate_calibration(
            [0, 1],
            [0.2],
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ([0, 2], "binary values"),
        ([0, 0], "both binary classes"),
        ([0, np.nan], "finite values"),
    ],
)
def test_calibration_metrics_reject_invalid_targets(
    target: list[float],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        evaluate_calibration(
            target,
            [0.2, 0.8],
        )


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ([-0.1, 0.8], "between 0 and 1"),
        ([0.2, 1.1], "between 0 and 1"),
        ([0.2, np.nan], "finite values"),
    ],
)
def test_calibration_metrics_reject_invalid_probabilities(
    probabilities: list[float],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        evaluate_calibration(
            [0, 1],
            probabilities,
        )


@pytest.mark.parametrize(
    ("keyword_arguments", "message"),
    [
        (
            {"bin_count": 1},
            "at least 2",
        ),
        (
            {"clip_epsilon": 0.0},
            "between 0 and 0.5",
        ),
        (
            {"max_iterations": 0},
            "positive integer",
        ),
        (
            {"tolerance": 0.0},
            "greater than 0",
        ),
    ],
)
def test_calibration_metrics_reject_invalid_configuration(
    keyword_arguments: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        evaluate_calibration(
            [0, 0, 1, 1],
            [0.1, 0.4, 0.6, 0.9],
            **keyword_arguments,
        )


def test_calibration_regression_requires_varying_scores() -> None:
    with pytest.raises(
        ValueError,
        match="requires varying",
    ):
        estimate_calibration_regression(
            [0, 0, 1, 1],
            [0.5, 0.5, 0.5, 0.5],
        )
