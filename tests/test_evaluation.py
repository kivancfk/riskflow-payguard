"""Tests for fraud-focused model evaluation metrics."""

from __future__ import annotations

import json
import math

import pytest

from src.evaluation import (
    evaluate_fraud_model,
    evaluate_probability_metrics,
    evaluate_review_capacity,
    evaluate_threshold_metrics,
)


Y_TRUE = [0, 1, 0, 1]
PROBABILITIES = [0.1, 0.9, 0.8, 0.4]


def test_probability_metrics_match_known_values() -> None:
    metrics = evaluate_probability_metrics(
        Y_TRUE,
        PROBABILITIES,
    )

    expected_log_loss = -(
        math.log(0.9)
        + math.log(0.9)
        + math.log(0.2)
        + math.log(0.4)
    ) / 4

    assert metrics["transaction_count"] == 4
    assert metrics["fraud_count"] == 2
    assert metrics["fraud_rate"] == pytest.approx(
        0.5
    )
    assert metrics["pr_auc"] == pytest.approx(
        5 / 6
    )
    assert metrics["roc_auc"] == pytest.approx(
        0.75
    )
    assert metrics["log_loss"] == pytest.approx(
        expected_log_loss
    )
    assert metrics["brier_score"] == pytest.approx(
        0.255
    )


def test_threshold_metrics_match_known_confusion_matrix(
) -> None:
    metrics = evaluate_threshold_metrics(
        Y_TRUE,
        PROBABILITIES,
        threshold=0.5,
    )

    assert metrics == {
        "threshold": 0.5,
        "predicted_fraud_count": 2,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "false_positive_rate": 0.5,
        "false_negative_rate": 0.5,
    }


def test_review_capacity_uses_highest_probabilities(
) -> None:
    records = evaluate_review_capacity(
        Y_TRUE,
        PROBABILITIES,
        review_fractions=(0.25, 0.50),
    )

    assert records[0] == {
        "review_fraction": 0.25,
        "reviewed_count": 1,
        "actual_review_rate": 0.25,
        "frauds_reviewed": 1,
        "fraud_recall": 0.5,
        "review_precision": 1.0,
    }

    assert records[1] == {
        "review_fraction": 0.50,
        "reviewed_count": 2,
        "actual_review_rate": 0.5,
        "frauds_reviewed": 1,
        "fraud_recall": 0.5,
        "review_precision": 0.5,
    }


def test_review_capacity_reports_fraud_amount_capture(
) -> None:
    records = evaluate_review_capacity(
        Y_TRUE,
        PROBABILITIES,
        review_fractions=(0.25, 0.50),
        transaction_amounts=[
            10.0,
            20.0,
            30.0,
            40.0,
        ],
    )

    assert records[0][
        "fraud_amount_captured"
    ] == pytest.approx(20.0)
    assert records[0][
        "fraud_amount_recall"
    ] == pytest.approx(1 / 3)

    assert records[1][
        "fraud_amount_captured"
    ] == pytest.approx(20.0)
    assert records[1][
        "fraud_amount_recall"
    ] == pytest.approx(1 / 3)


def test_review_ranking_is_stable_for_probability_ties(
) -> None:
    records = evaluate_review_capacity(
        [1, 0, 1, 0],
        [0.5, 0.5, 0.5, 0.5],
        review_fractions=(0.5,),
        transaction_amounts=[
            10.0,
            20.0,
            30.0,
            40.0,
        ],
    )

    assert records[0]["reviewed_count"] == 2
    assert records[0]["frauds_reviewed"] == 1
    assert records[0][
        "fraud_amount_captured"
    ] == pytest.approx(10.0)
    assert records[0][
        "fraud_amount_recall"
    ] == pytest.approx(0.25)


def test_complete_report_is_json_serializable() -> None:
    report = evaluate_fraud_model(
        Y_TRUE,
        PROBABILITIES,
        threshold=0.5,
        review_fractions=(0.25,),
        transaction_amounts=[
            10.0,
            20.0,
            30.0,
            40.0,
        ],
    )

    serialized = json.dumps(report)

    assert "probability_metrics" in report
    assert "threshold_metrics" in report
    assert "review_capacity" in report
    assert isinstance(serialized, str)


def test_evaluation_rejects_invalid_vectors() -> None:
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        evaluate_fraud_model(
            [0, 1],
            [0.1],
        )

    with pytest.raises(
        ValueError,
        match="binary values",
    ):
        evaluate_fraud_model(
            [0, 2],
            [0.1, 0.9],
        )

    with pytest.raises(
        ValueError,
        match="both binary classes",
    ):
        evaluate_fraud_model(
            [0, 0],
            [0.1, 0.2],
        )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        evaluate_fraud_model(
            [0, 1],
            [0.1, 1.2],
        )


def test_evaluation_rejects_invalid_configuration(
) -> None:
    with pytest.raises(
        ValueError,
        match="threshold",
    ):
        evaluate_threshold_metrics(
            Y_TRUE,
            PROBABILITIES,
            threshold=1.1,
        )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        evaluate_review_capacity(
            Y_TRUE,
            PROBABILITIES,
            review_fractions=(),
        )

    with pytest.raises(
        ValueError,
        match="no greater than 1",
    ):
        evaluate_review_capacity(
            Y_TRUE,
            PROBABILITIES,
            review_fractions=(0.0,),
        )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        evaluate_review_capacity(
            Y_TRUE,
            PROBABILITIES,
            review_fractions=(0.5, 0.5),
        )

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        evaluate_review_capacity(
            Y_TRUE,
            PROBABILITIES,
            review_fractions=(0.5,),
            transaction_amounts=[10.0],
        )

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        evaluate_review_capacity(
            Y_TRUE,
            PROBABILITIES,
            review_fractions=(0.5,),
            transaction_amounts=[
                10.0,
                -20.0,
                30.0,
                40.0,
            ],
        )
