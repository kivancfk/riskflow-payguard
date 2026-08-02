"""Tests for deterministic fraud decision-policy optimization."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from api.decision_engine import decide
from src.policy import (
    PolicyConstraints,
    PolicyCostAssumptions,
    PolicyThresholds,
    assign_policy_decisions,
    evaluate_policy,
    evaluate_policy_constraints,
    generate_candidate_thresholds,
    search_policy_thresholds,
)


def _cost_assumptions(
) -> PolicyCostAssumptions:
    """Return explicit synthetic policy costs."""

    return PolicyCostAssumptions(
        cost_per_review=2.0,
        cost_per_legitimate_block=5.0,
        fraud_loss_multiplier=1.0,
        review_fraud_capture_rate=0.5,
    )


def _separated_policy_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return perfectly separated calibrated scores."""

    return (
        np.array(
            [0, 0, 1, 1],
            dtype="int8",
        ),
        np.array(
            [0.1, 0.2, 0.8, 0.9],
            dtype="float64",
        ),
        np.array(
            [10.0, 10.0, 100.0, 100.0],
            dtype="float64",
        ),
    )


def test_policy_decisions_match_existing_engine_boundaries(
) -> None:
    thresholds = PolicyThresholds(
        review_threshold=0.30,
        block_threshold=0.70,
    )
    probabilities = [
        0.29,
        0.30,
        0.69,
        0.70,
        0.95,
    ]

    decisions = assign_policy_decisions(
        probabilities,
        thresholds,
    )

    expected = [
        decide(
            probability,
            review_threshold=0.30,
            block_threshold=0.70,
        )
        for probability in probabilities
    ]

    assert decisions.tolist() == expected


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        (
            "cost_per_review",
            -1.0,
            "non-negative",
        ),
        (
            "cost_per_legitimate_block",
            np.inf,
            "finite",
        ),
        (
            "fraud_loss_multiplier",
            0.0,
            "greater than 0",
        ),
        (
            "review_fraud_capture_rate",
            1.1,
            "between 0 and 1",
        ),
    ],
)
def test_cost_assumptions_reject_invalid_values(
    field_name: str,
    value: float,
    message: str,
) -> None:
    arguments = {
        "cost_per_review": 2.0,
        "cost_per_legitimate_block": 5.0,
        "fraud_loss_multiplier": 1.0,
        "review_fraud_capture_rate": 0.5,
    }
    arguments[field_name] = value

    with pytest.raises(
        ValueError,
        match=message,
    ):
        PolicyCostAssumptions(
            **arguments
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        (
            "max_review_rate",
            -0.1,
        ),
        (
            "min_block_precision",
            1.1,
        ),
        (
            "min_fraud_amount_capture",
            np.nan,
        ),
    ],
)
def test_constraints_reject_invalid_values(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        PolicyConstraints(
            **{
                field_name: value,
            }
        )


def test_evaluate_policy_calculates_known_costs_and_capture(
) -> None:
    metrics = evaluate_policy(
        [0, 1, 0, 1],
        [0.1, 0.4, 0.8, 0.9],
        [10.0, 100.0, 20.0, 200.0],
        thresholds=PolicyThresholds(
            review_threshold=0.30,
            block_threshold=0.85,
        ),
        cost_assumptions=_cost_assumptions(),
    )

    assert metrics["allow_count"] == 1
    assert metrics["review_count"] == 2
    assert metrics["block_count"] == 1

    assert metrics[
        "review_legitimate_count"
    ] == 1
    assert metrics[
        "review_fraud_count"
    ] == 1
    assert metrics[
        "block_fraud_count"
    ] == 1
    assert metrics[
        "block_legitimate_count"
    ] == 0

    assert metrics["review_precision"] == 0.5
    assert metrics["block_precision"] == 1.0
    assert metrics["fraud_recall"] == 1.0

    assert metrics[
        "expected_fraud_capture_rate"
    ] == 0.75

    assert metrics[
        "fraud_amount_captured"
    ] == pytest.approx(250.0)
    assert metrics[
        "fraud_amount_capture_rate"
    ] == pytest.approx(250.0 / 300.0)

    assert metrics[
        "total_possible_fraud_loss"
    ] == 300.0
    assert metrics[
        "prevented_fraud_loss"
    ] == 250.0
    assert metrics[
        "remaining_fraud_loss"
    ] == 50.0

    assert metrics["total_cost"] == 54.0
    assert metrics[
        "average_cost_per_transaction"
    ] == 13.5


def test_policy_constraints_report_all_violations(
) -> None:
    metrics = evaluate_policy(
        [0, 1, 0, 1],
        [0.1, 0.4, 0.8, 0.9],
        [10.0, 100.0, 20.0, 200.0],
        thresholds=PolicyThresholds(
            review_threshold=0.30,
            block_threshold=0.85,
        ),
        cost_assumptions=_cost_assumptions(),
    )

    feasible, violations = (
        evaluate_policy_constraints(
            metrics,
            PolicyConstraints(
                max_review_rate=0.25,
                min_fraud_amount_capture=0.90,
            ),
        )
    )

    assert not feasible
    assert violations == (
        "max_review_rate",
        "min_fraud_amount_capture",
    )


def test_candidate_thresholds_are_observed_sorted_and_deterministic(
) -> None:
    probabilities = [
        0.1,
        0.2,
        0.8,
        0.9,
    ]

    first = generate_candidate_thresholds(
        probabilities,
        quantile_count=5,
    )
    second = generate_candidate_thresholds(
        probabilities,
        quantile_count=5,
    )

    assert first == second
    assert first == tuple(
        sorted(set(first))
    )
    assert first[0] == 0.0
    assert first[-1] == 1.0

    assert set(first).issubset(
        {
            0.0,
            0.1,
            0.2,
            0.8,
            0.9,
            1.0,
        }
    )


def test_search_selects_zero_cost_separated_policy(
) -> None:
    target, probabilities, amounts = (
        _separated_policy_data()
    )

    result = search_policy_thresholds(
        target,
        probabilities,
        amounts,
        cost_assumptions=(
            PolicyCostAssumptions(
                cost_per_review=100.0,
                cost_per_legitimate_block=50.0,
                fraud_loss_multiplier=1.0,
                review_fraud_capture_rate=0.0,
            )
        ),
        constraints=PolicyConstraints(),
        quantile_count=5,
    )

    assert result.selected_thresholds == (
        PolicyThresholds(
            review_threshold=0.8,
            block_threshold=0.8,
        )
    )
    assert result.selected_metrics[
        "total_cost"
    ] == 0.0

    assert result.selected_metrics[
        "allow_count"
    ] == 2
    assert result.selected_metrics[
        "review_count"
    ] == 0
    assert result.selected_metrics[
        "block_count"
    ] == 2

    assert result.selected_metrics[
        "block_precision"
    ] == 1.0
    assert result.selected_metrics[
        "fraud_recall"
    ] == 1.0


def test_search_raises_when_constraints_are_infeasible(
) -> None:
    target, probabilities, amounts = (
        _separated_policy_data()
    )

    with pytest.raises(
        ValueError,
        match="No feasible policy",
    ):
        search_policy_thresholds(
            target,
            probabilities,
            amounts,
            cost_assumptions=(
                _cost_assumptions()
            ),
            constraints=PolicyConstraints(
                max_intervention_rate=0.0,
                min_fraud_recall=1.0,
            ),
            quantile_count=5,
        )


def test_search_tie_break_prefers_no_intervention(
) -> None:
    target, probabilities, _ = (
        _separated_policy_data()
    )

    result = search_policy_thresholds(
        target,
        probabilities,
        [0.0, 0.0, 0.0, 0.0],
        cost_assumptions=(
            PolicyCostAssumptions(
                cost_per_review=0.0,
                cost_per_legitimate_block=0.0,
                fraud_loss_multiplier=1.0,
                review_fraud_capture_rate=1.0,
            )
        ),
        constraints=PolicyConstraints(),
        quantile_count=5,
    )

    assert result.selected_thresholds == (
        PolicyThresholds(
            review_threshold=1.0,
            block_threshold=1.0,
        )
    )
    assert result.selected_metrics[
        "intervention_rate"
    ] == 0.0


def test_search_includes_all_allow_and_reference_comparisons(
) -> None:
    target, probabilities, amounts = (
        _separated_policy_data()
    )

    result = search_policy_thresholds(
        target,
        probabilities,
        amounts,
        cost_assumptions=(
            PolicyCostAssumptions(
                cost_per_review=100.0,
                cost_per_legitimate_block=50.0,
                fraud_loss_multiplier=1.0,
                review_fraud_capture_rate=0.0,
            )
        ),
        constraints=PolicyConstraints(),
        quantile_count=5,
        reference_thresholds=(
            PolicyThresholds(
                review_threshold=0.0,
                block_threshold=0.0,
            )
        ),
    )

    assert result.all_allow_metrics[
        "total_cost"
    ] == 200.0

    assert result.selected_metrics[
        "cost_savings_vs_all_allow"
    ] == 200.0
    assert result.selected_metrics[
        "cost_reduction_rate_vs_all_allow"
    ] == 1.0

    assert result.reference_policy is not None
    assert result.reference_policy[
        "metrics"
    ]["total_cost"] == 100.0
    assert result.reference_policy[
        "selected_cost_savings"
    ] == 100.0


def test_search_is_deterministic() -> None:
    target, probabilities, amounts = (
        _separated_policy_data()
    )

    keyword_arguments = {
        "cost_assumptions": (
            _cost_assumptions()
        ),
        "constraints": PolicyConstraints(
            max_review_rate=0.50,
            max_block_rate=0.50,
        ),
        "quantile_count": 5,
        "reference_thresholds": (
            PolicyThresholds(
                review_threshold=0.30,
                block_threshold=0.70,
            )
        ),
    }

    first = search_policy_thresholds(
        target,
        probabilities,
        amounts,
        **keyword_arguments,
    )
    second = search_policy_thresholds(
        target,
        probabilities,
        amounts,
        **keyword_arguments,
    )

    assert first == second


def test_search_result_is_json_compatible() -> None:
    target, probabilities, amounts = (
        _separated_policy_data()
    )

    result = search_policy_thresholds(
        target,
        probabilities,
        amounts,
        cost_assumptions=(
            _cost_assumptions()
        ),
        constraints=PolicyConstraints(),
        quantile_count=5,
    )

    json.dumps(
        asdict(result)
    )


@pytest.mark.parametrize(
    ("target", "probabilities", "amounts", "message"),
    [
        (
            [0, 2],
            [0.2, 0.8],
            [10.0, 20.0],
            "only 0 and 1",
        ),
        (
            [0, 1],
            [0.2, np.nan],
            [10.0, 20.0],
            "finite values",
        ),
        (
            [0, 1],
            [0.2, 0.8],
            [10.0, -20.0],
            "non-negative",
        ),
    ],
)
def test_search_rejects_invalid_inputs(
    target: list[float],
    probabilities: list[float],
    amounts: list[float],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        search_policy_thresholds(
            target,
            probabilities,
            amounts,
            cost_assumptions=(
                _cost_assumptions()
            ),
            constraints=PolicyConstraints(),
        )


@pytest.mark.parametrize(
    ("keyword_arguments", "message"),
    [
        (
            {
                "quantile_count": 1,
            },
            "at least 2",
        ),
        (
            {
                "objective_round_digits": 16,
            },
            "between 0 and 15",
        ),
    ],
)
def test_search_rejects_invalid_configuration(
    keyword_arguments: dict[str, int],
    message: str,
) -> None:
    target, probabilities, amounts = (
        _separated_policy_data()
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        search_policy_thresholds(
            target,
            probabilities,
            amounts,
            cost_assumptions=(
                _cost_assumptions()
            ),
            constraints=PolicyConstraints(),
            **keyword_arguments,
        )
