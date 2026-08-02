"""Deterministic fraud decision-policy evaluation and optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


ALLOW = "ALLOW"
REVIEW = "REVIEW"
BLOCK = "BLOCK"

PolicyDecision = Literal[
    "ALLOW",
    "REVIEW",
    "BLOCK",
]

DEFAULT_QUANTILE_COUNT = 51
DEFAULT_OBJECTIVE_ROUND_DIGITS = 12
MAX_OBJECTIVE_ROUND_DIGITS = 15

CONSTRAINT_TOLERANCE = 1e-12


def _validate_finite_number(
    value: float,
    *,
    value_name: str,
) -> float:
    """Return a finite numerical value."""

    if isinstance(value, bool):
        raise ValueError(
            f"{value_name} must be numerical"
        )

    try:
        normalized_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{value_name} must be numerical"
        ) from error

    if not math.isfinite(normalized_value):
        raise ValueError(
            f"{value_name} must be finite"
        )

    return normalized_value


def _validate_non_negative_number(
    value: float,
    *,
    value_name: str,
) -> float:
    """Return a finite non-negative numerical value."""

    normalized_value = _validate_finite_number(
        value,
        value_name=value_name,
    )

    if normalized_value < 0.0:
        raise ValueError(
            f"{value_name} must be non-negative"
        )

    return normalized_value


def _validate_probability_value(
    value: float,
    *,
    value_name: str,
) -> float:
    """Return a finite value within the probability range."""

    normalized_value = _validate_finite_number(
        value,
        value_name=value_name,
    )

    if not 0.0 <= normalized_value <= 1.0:
        raise ValueError(
            f"{value_name} must be between 0 and 1"
        )

    return normalized_value


@dataclass(frozen=True)
class PolicyThresholds:
    """Validated ALLOW, REVIEW, and BLOCK boundaries."""

    review_threshold: float
    block_threshold: float

    def __post_init__(self) -> None:
        review_threshold = (
            _validate_probability_value(
                self.review_threshold,
                value_name="review_threshold",
            )
        )
        block_threshold = (
            _validate_probability_value(
                self.block_threshold,
                value_name="block_threshold",
            )
        )

        if review_threshold > block_threshold:
            raise ValueError(
                "review_threshold must be <= "
                "block_threshold"
            )

        object.__setattr__(
            self,
            "review_threshold",
            review_threshold,
        )
        object.__setattr__(
            self,
            "block_threshold",
            block_threshold,
        )


@dataclass(frozen=True)
class PolicyCostAssumptions:
    """Explicit monetary assumptions for policy evaluation.

    Fraud loss is calculated as transaction amount multiplied by
    ``fraud_loss_multiplier``.

    Reviewed fraud retains the uncaptured share of that loss:

    ``1 - review_fraud_capture_rate``.
    """

    cost_per_review: float
    cost_per_legitimate_block: float
    fraud_loss_multiplier: float
    review_fraud_capture_rate: float

    def __post_init__(self) -> None:
        cost_per_review = (
            _validate_non_negative_number(
                self.cost_per_review,
                value_name="cost_per_review",
            )
        )
        cost_per_legitimate_block = (
            _validate_non_negative_number(
                self.cost_per_legitimate_block,
                value_name=(
                    "cost_per_legitimate_block"
                ),
            )
        )
        fraud_loss_multiplier = (
            _validate_finite_number(
                self.fraud_loss_multiplier,
                value_name="fraud_loss_multiplier",
            )
        )
        review_fraud_capture_rate = (
            _validate_probability_value(
                self.review_fraud_capture_rate,
                value_name=(
                    "review_fraud_capture_rate"
                ),
            )
        )

        if fraud_loss_multiplier <= 0.0:
            raise ValueError(
                "fraud_loss_multiplier must be "
                "greater than 0"
            )

        object.__setattr__(
            self,
            "cost_per_review",
            cost_per_review,
        )
        object.__setattr__(
            self,
            "cost_per_legitimate_block",
            cost_per_legitimate_block,
        )
        object.__setattr__(
            self,
            "fraud_loss_multiplier",
            fraud_loss_multiplier,
        )
        object.__setattr__(
            self,
            "review_fraud_capture_rate",
            review_fraud_capture_rate,
        )


@dataclass(frozen=True)
class PolicyConstraints:
    """Optional operational requirements for a policy."""

    max_review_rate: float | None = None
    max_block_rate: float | None = None
    max_intervention_rate: float | None = None
    min_review_precision: float | None = None
    min_block_precision: float | None = None
    min_fraud_recall: float | None = None
    min_fraud_amount_capture: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "max_review_rate",
            "max_block_rate",
            "max_intervention_rate",
            "min_review_precision",
            "min_block_precision",
            "min_fraud_recall",
            "min_fraud_amount_capture",
        ):
            value = getattr(
                self,
                field_name,
            )

            if value is None:
                continue

            normalized_value = (
                _validate_probability_value(
                    value,
                    value_name=field_name,
                )
            )

            object.__setattr__(
                self,
                field_name,
                normalized_value,
            )


@dataclass(frozen=True)
class PolicySearchResult:
    """Selected thresholds and reproducible search metadata."""

    selected_thresholds: PolicyThresholds
    selected_metrics: dict[str, float | int]
    cost_assumptions: PolicyCostAssumptions
    constraints: PolicyConstraints
    candidate_thresholds: tuple[float, ...]
    quantile_count: int
    objective_round_digits: int
    evaluated_candidate_count: int
    feasible_candidate_count: int
    selected_selection_key: tuple[
        float,
        int,
        float,
        float,
        float,
        float,
    ]
    all_allow_metrics: dict[str, float | int]
    all_allow_feasible: bool
    all_allow_constraint_violations: tuple[str, ...]
    reference_policy: dict[str, Any] | None


def _as_float_vector(
    values: ArrayLike,
    *,
    value_name: str,
) -> np.ndarray:
    """Return a finite one-dimensional numerical vector."""

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


def _validate_policy_inputs(
    y_true: ArrayLike,
    calibrated_probabilities: ArrayLike,
    transaction_amounts: ArrayLike,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Validate aligned policy-selection arrays."""

    target = _as_float_vector(
        y_true,
        value_name="y_true",
    )
    probabilities = _as_float_vector(
        calibrated_probabilities,
        value_name="calibrated_probabilities",
    )
    amounts = _as_float_vector(
        transaction_amounts,
        value_name="transaction_amounts",
    )

    if not (
        len(target)
        == len(probabilities)
        == len(amounts)
    ):
        raise ValueError(
            "y_true, calibrated_probabilities, and "
            "transaction_amounts must have the same length"
        )

    if not np.isin(
        target,
        [0.0, 1.0],
    ).all():
        raise ValueError(
            "y_true must contain only 0 and 1"
        )

    normalized_target = target.astype("int8")

    if set(normalized_target.tolist()) != {0, 1}:
        raise ValueError(
            "y_true must contain both binary classes"
        )

    if (
        (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
    ):
        raise ValueError(
            "calibrated_probabilities must be "
            "between 0 and 1"
        )

    if (amounts < 0.0).any():
        raise ValueError(
            "transaction_amounts must be non-negative"
        )

    return (
        normalized_target,
        probabilities,
        amounts,
    )


def _validate_quantile_count(
    quantile_count: int,
) -> int:
    """Return a valid candidate quantile count."""

    if (
        isinstance(quantile_count, bool)
        or not isinstance(
            quantile_count,
            (int, np.integer),
        )
        or int(quantile_count) < 2
    ):
        raise ValueError(
            "quantile_count must be an integer "
            "of at least 2"
        )

    return int(quantile_count)


def _validate_objective_round_digits(
    objective_round_digits: int,
) -> int:
    """Return valid deterministic objective precision."""

    if (
        isinstance(objective_round_digits, bool)
        or not isinstance(
            objective_round_digits,
            (int, np.integer),
        )
        or not (
            0
            <= int(objective_round_digits)
            <= MAX_OBJECTIVE_ROUND_DIGITS
        )
    ):
        raise ValueError(
            "objective_round_digits must be an integer "
            "between 0 and 15"
        )

    return int(objective_round_digits)


def assign_policy_decisions(
    calibrated_probabilities: ArrayLike,
    thresholds: PolicyThresholds,
) -> np.ndarray:
    """Map calibrated probabilities to policy decisions."""

    if not isinstance(
        thresholds,
        PolicyThresholds,
    ):
        raise ValueError(
            "thresholds must be a PolicyThresholds instance"
        )

    probabilities = _as_float_vector(
        calibrated_probabilities,
        value_name="calibrated_probabilities",
    )

    if (
        (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
    ):
        raise ValueError(
            "calibrated_probabilities must be "
            "between 0 and 1"
        )

    decisions = np.full(
        len(probabilities),
        ALLOW,
        dtype="<U6",
    )

    review_mask = (
        probabilities
        >= thresholds.review_threshold
    )
    block_mask = (
        probabilities
        >= thresholds.block_threshold
    )

    decisions[review_mask] = REVIEW
    decisions[block_mask] = BLOCK

    return decisions


def _safe_rate(
    numerator: float | int,
    denominator: float | int,
) -> float:
    """Return zero when a rate denominator is zero."""

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def _evaluate_assigned_policy(
    target: np.ndarray,
    transaction_amounts: np.ndarray,
    decisions: np.ndarray,
    cost_assumptions: PolicyCostAssumptions,
) -> dict[str, float | int]:
    """Calculate cost, volume, fraud, and amount metrics."""

    row_count = len(target)

    allow_mask = decisions == ALLOW
    review_mask = decisions == REVIEW
    block_mask = decisions == BLOCK

    legitimate_mask = target == 0
    fraud_mask = target == 1

    allow_legitimate_mask = (
        allow_mask & legitimate_mask
    )
    allow_fraud_mask = (
        allow_mask & fraud_mask
    )
    review_legitimate_mask = (
        review_mask & legitimate_mask
    )
    review_fraud_mask = (
        review_mask & fraud_mask
    )
    block_legitimate_mask = (
        block_mask & legitimate_mask
    )
    block_fraud_mask = (
        block_mask & fraud_mask
    )

    allow_count = int(
        allow_mask.sum()
    )
    review_count = int(
        review_mask.sum()
    )
    block_count = int(
        block_mask.sum()
    )

    allow_legitimate_count = int(
        allow_legitimate_mask.sum()
    )
    allow_fraud_count = int(
        allow_fraud_mask.sum()
    )
    review_legitimate_count = int(
        review_legitimate_mask.sum()
    )
    review_fraud_count = int(
        review_fraud_mask.sum()
    )
    block_legitimate_count = int(
        block_legitimate_mask.sum()
    )
    block_fraud_count = int(
        block_fraud_mask.sum()
    )

    fraud_count = int(
        fraud_mask.sum()
    )
    legitimate_count = (
        row_count - fraud_count
    )

    full_fraud_losses = (
        transaction_amounts
        * cost_assumptions.fraud_loss_multiplier
    )

    residual_review_fraction = (
        1.0
        - cost_assumptions.review_fraud_capture_rate
    )

    transaction_costs = np.zeros(
        row_count,
        dtype="float64",
    )

    transaction_costs[review_mask] += (
        cost_assumptions.cost_per_review
    )
    transaction_costs[
        review_fraud_mask
    ] += (
        full_fraud_losses[
            review_fraud_mask
        ]
        * residual_review_fraction
    )
    transaction_costs[
        allow_fraud_mask
    ] += full_fraud_losses[
        allow_fraud_mask
    ]
    transaction_costs[
        block_legitimate_mask
    ] += (
        cost_assumptions
        .cost_per_legitimate_block
    )

    total_fraud_amount = float(
        transaction_amounts[
            fraud_mask
        ].sum()
    )

    reviewed_fraud_amount = float(
        transaction_amounts[
            review_fraud_mask
        ].sum()
    )
    blocked_fraud_amount = float(
        transaction_amounts[
            block_fraud_mask
        ].sum()
    )

    fraud_amount_captured = (
        blocked_fraud_amount
        + (
            reviewed_fraud_amount
            * cost_assumptions
            .review_fraud_capture_rate
        )
    )

    total_possible_fraud_loss = float(
        full_fraud_losses[
            fraud_mask
        ].sum()
    )

    prevented_fraud_loss = float(
        full_fraud_losses[
            block_fraud_mask
        ].sum()
        + (
            full_fraud_losses[
                review_fraud_mask
            ].sum()
            * cost_assumptions
            .review_fraud_capture_rate
        )
    )

    fraud_intervention_count = (
        review_fraud_count
        + block_fraud_count
    )

    expected_fraud_capture_count = (
        block_fraud_count
        + (
            review_fraud_count
            * cost_assumptions
            .review_fraud_capture_rate
        )
    )

    intervention_count = (
        review_count + block_count
    )
    total_cost = float(
        transaction_costs.sum()
    )

    return {
        "transaction_count": row_count,
        "legitimate_count": legitimate_count,
        "fraud_count": fraud_count,
        "fraud_rate": float(
            target.mean()
        ),
        "allow_count": allow_count,
        "allow_rate": _safe_rate(
            allow_count,
            row_count,
        ),
        "review_count": review_count,
        "review_rate": _safe_rate(
            review_count,
            row_count,
        ),
        "block_count": block_count,
        "block_rate": _safe_rate(
            block_count,
            row_count,
        ),
        "intervention_count": intervention_count,
        "intervention_rate": _safe_rate(
            intervention_count,
            row_count,
        ),
        "allow_legitimate_count": (
            allow_legitimate_count
        ),
        "allow_fraud_count": allow_fraud_count,
        "review_legitimate_count": (
            review_legitimate_count
        ),
        "review_fraud_count": (
            review_fraud_count
        ),
        "block_legitimate_count": (
            block_legitimate_count
        ),
        "block_fraud_count": block_fraud_count,
        "review_precision": _safe_rate(
            review_fraud_count,
            review_count,
        ),
        "block_precision": _safe_rate(
            block_fraud_count,
            block_count,
        ),
        "fraud_recall": _safe_rate(
            fraud_intervention_count,
            fraud_count,
        ),
        "expected_fraud_capture_count": float(
            expected_fraud_capture_count
        ),
        "expected_fraud_capture_rate": (
            _safe_rate(
                expected_fraud_capture_count,
                fraud_count,
            )
        ),
        "total_fraud_amount": total_fraud_amount,
        "reviewed_fraud_amount": (
            reviewed_fraud_amount
        ),
        "blocked_fraud_amount": (
            blocked_fraud_amount
        ),
        "fraud_amount_captured": (
            fraud_amount_captured
        ),
        "fraud_amount_capture_rate": (
            _safe_rate(
                fraud_amount_captured,
                total_fraud_amount,
            )
        ),
        "total_possible_fraud_loss": (
            total_possible_fraud_loss
        ),
        "prevented_fraud_loss": (
            prevented_fraud_loss
        ),
        "remaining_fraud_loss": float(
            total_possible_fraud_loss
            - prevented_fraud_loss
        ),
        "total_cost": total_cost,
        "average_cost_per_transaction": (
            _safe_rate(
                total_cost,
                row_count,
            )
        ),
    }


def evaluate_policy(
    y_true: ArrayLike,
    calibrated_probabilities: ArrayLike,
    transaction_amounts: ArrayLike,
    *,
    thresholds: PolicyThresholds,
    cost_assumptions: PolicyCostAssumptions,
) -> dict[str, float | int]:
    """Evaluate one explicit calibrated decision policy."""

    if not isinstance(
        cost_assumptions,
        PolicyCostAssumptions,
    ):
        raise ValueError(
            "cost_assumptions must be a "
            "PolicyCostAssumptions instance"
        )

    (
        target,
        probabilities,
        amounts,
    ) = _validate_policy_inputs(
        y_true,
        calibrated_probabilities,
        transaction_amounts,
    )

    decisions = assign_policy_decisions(
        probabilities,
        thresholds,
    )

    metrics = _evaluate_assigned_policy(
        target,
        amounts,
        decisions,
        cost_assumptions,
    )

    return {
        "review_threshold": (
            thresholds.review_threshold
        ),
        "block_threshold": (
            thresholds.block_threshold
        ),
        **metrics,
    }


def evaluate_all_allow_policy(
    y_true: ArrayLike,
    calibrated_probabilities: ArrayLike,
    transaction_amounts: ArrayLike,
    *,
    cost_assumptions: PolicyCostAssumptions,
) -> dict[str, float | int]:
    """Evaluate the no-intervention all-allow benchmark."""

    (
        target,
        probabilities,
        amounts,
    ) = _validate_policy_inputs(
        y_true,
        calibrated_probabilities,
        transaction_amounts,
    )

    if not isinstance(
        cost_assumptions,
        PolicyCostAssumptions,
    ):
        raise ValueError(
            "cost_assumptions must be a "
            "PolicyCostAssumptions instance"
        )

    decisions = np.full(
        len(probabilities),
        ALLOW,
        dtype="<U6",
    )

    return _evaluate_assigned_policy(
        target,
        amounts,
        decisions,
        cost_assumptions,
    )


def evaluate_policy_constraints(
    metrics: dict[str, float | int],
    constraints: PolicyConstraints,
) -> tuple[
    bool,
    tuple[str, ...],
]:
    """Return policy feasibility and violated constraint names."""

    if not isinstance(
        constraints,
        PolicyConstraints,
    ):
        raise ValueError(
            "constraints must be a "
            "PolicyConstraints instance"
        )

    checks = (
        (
            "max_review_rate",
            "review_rate",
            "maximum",
        ),
        (
            "max_block_rate",
            "block_rate",
            "maximum",
        ),
        (
            "max_intervention_rate",
            "intervention_rate",
            "maximum",
        ),
        (
            "min_review_precision",
            "review_precision",
            "minimum",
        ),
        (
            "min_block_precision",
            "block_precision",
            "minimum",
        ),
        (
            "min_fraud_recall",
            "fraud_recall",
            "minimum",
        ),
        (
            "min_fraud_amount_capture",
            "fraud_amount_capture_rate",
            "minimum",
        ),
    )

    violations: list[str] = []

    for (
        constraint_name,
        metric_name,
        comparison_type,
    ) in checks:
        constraint_value = getattr(
            constraints,
            constraint_name,
        )

        if constraint_value is None:
            continue

        if metric_name not in metrics:
            raise ValueError(
                "Policy metrics do not contain "
                f"required field: {metric_name}"
            )

        metric_value = float(
            metrics[metric_name]
        )

        if comparison_type == "maximum":
            violated = (
                metric_value
                > (
                    constraint_value
                    + CONSTRAINT_TOLERANCE
                )
            )
        else:
            violated = (
                metric_value
                < (
                    constraint_value
                    - CONSTRAINT_TOLERANCE
                )
            )

        if violated:
            violations.append(
                constraint_name
            )

    return (
        not violations,
        tuple(violations),
    )


def generate_candidate_thresholds(
    calibrated_probabilities: ArrayLike,
    *,
    quantile_count: int = DEFAULT_QUANTILE_COUNT,
) -> tuple[float, ...]:
    """Create sorted observed-quantile threshold candidates."""

    probabilities = _as_float_vector(
        calibrated_probabilities,
        value_name="calibrated_probabilities",
    )

    if (
        (probabilities < 0.0).any()
        or (probabilities > 1.0).any()
    ):
        raise ValueError(
            "calibrated_probabilities must be "
            "between 0 and 1"
        )

    validated_quantile_count = (
        _validate_quantile_count(
            quantile_count
        )
    )

    observed_quantiles = np.quantile(
        probabilities,
        np.linspace(
            0.0,
            1.0,
            validated_quantile_count,
        ),
        method="nearest",
    )

    candidates = np.unique(
        np.concatenate(
            (
                np.array(
                    [0.0, 1.0],
                    dtype="float64",
                ),
                observed_quantiles.astype(
                    "float64",
                    copy=False,
                ),
            )
        )
    )

    return tuple(
        float(value)
        for value in candidates
    )


def _selection_key(
    thresholds: PolicyThresholds,
    metrics: dict[str, float | int],
    *,
    objective_round_digits: int,
) -> tuple[
    float,
    int,
    float,
    float,
    float,
    float,
]:
    """Build the documented deterministic policy tie-break key."""

    return (
        round(
            float(metrics["total_cost"]),
            objective_round_digits,
        ),
        int(
            metrics[
                "block_legitimate_count"
            ]
        ),
        round(
            float(
                metrics["intervention_rate"]
            ),
            objective_round_digits,
        ),
        round(
            float(metrics["block_rate"]),
            objective_round_digits,
        ),
        -thresholds.block_threshold,
        -thresholds.review_threshold,
    )


def search_policy_thresholds(
    y_true: ArrayLike,
    calibrated_probabilities: ArrayLike,
    transaction_amounts: ArrayLike,
    *,
    cost_assumptions: PolicyCostAssumptions,
    constraints: PolicyConstraints,
    quantile_count: int = DEFAULT_QUANTILE_COUNT,
    objective_round_digits: int = (
        DEFAULT_OBJECTIVE_ROUND_DIGITS
    ),
    reference_thresholds: (
        PolicyThresholds | None
    ) = None,
) -> PolicySearchResult:
    """Select the lowest-cost feasible threshold pair."""

    if not isinstance(
        cost_assumptions,
        PolicyCostAssumptions,
    ):
        raise ValueError(
            "cost_assumptions must be a "
            "PolicyCostAssumptions instance"
        )

    if not isinstance(
        constraints,
        PolicyConstraints,
    ):
        raise ValueError(
            "constraints must be a "
            "PolicyConstraints instance"
        )

    (
        target,
        probabilities,
        amounts,
    ) = _validate_policy_inputs(
        y_true,
        calibrated_probabilities,
        transaction_amounts,
    )

    validated_quantile_count = (
        _validate_quantile_count(
            quantile_count
        )
    )
    validated_round_digits = (
        _validate_objective_round_digits(
            objective_round_digits
        )
    )

    candidate_thresholds = (
        generate_candidate_thresholds(
            probabilities,
            quantile_count=(
                validated_quantile_count
            ),
        )
    )

    selected_thresholds: (
        PolicyThresholds | None
    ) = None
    selected_metrics: (
        dict[str, float | int] | None
    ) = None
    selected_key: (
        tuple[
            float,
            int,
            float,
            float,
            float,
            float,
        ]
        | None
    ) = None

    evaluated_candidate_count = 0
    feasible_candidate_count = 0

    for review_index, review_threshold in enumerate(
        candidate_thresholds
    ):
        for block_threshold in candidate_thresholds[
            review_index:
        ]:
            evaluated_candidate_count += 1

            thresholds = PolicyThresholds(
                review_threshold=(
                    review_threshold
                ),
                block_threshold=(
                    block_threshold
                ),
            )

            decisions = assign_policy_decisions(
                probabilities,
                thresholds,
            )
            metrics = _evaluate_assigned_policy(
                target,
                amounts,
                decisions,
                cost_assumptions,
            )

            feasible, _ = (
                evaluate_policy_constraints(
                    metrics,
                    constraints,
                )
            )

            if not feasible:
                continue

            feasible_candidate_count += 1

            candidate_key = _selection_key(
                thresholds,
                metrics,
                objective_round_digits=(
                    validated_round_digits
                ),
            )

            if (
                selected_key is None
                or candidate_key < selected_key
            ):
                selected_thresholds = thresholds
                selected_metrics = metrics
                selected_key = candidate_key

    if (
        selected_thresholds is None
        or selected_metrics is None
        or selected_key is None
    ):
        raise ValueError(
            "No feasible policy threshold pair "
            "satisfies the active constraints"
        )

    all_allow_metrics = (
        evaluate_all_allow_policy(
            target,
            probabilities,
            amounts,
            cost_assumptions=cost_assumptions,
        )
    )
    (
        all_allow_feasible,
        all_allow_violations,
    ) = evaluate_policy_constraints(
        all_allow_metrics,
        constraints,
    )

    all_allow_cost = float(
        all_allow_metrics["total_cost"]
    )
    selected_cost = float(
        selected_metrics["total_cost"]
    )

    selected_metrics = {
        "review_threshold": (
            selected_thresholds
            .review_threshold
        ),
        "block_threshold": (
            selected_thresholds
            .block_threshold
        ),
        **selected_metrics,
        "cost_savings_vs_all_allow": float(
            all_allow_cost - selected_cost
        ),
        "cost_reduction_rate_vs_all_allow": (
            _safe_rate(
                all_allow_cost - selected_cost,
                all_allow_cost,
            )
        ),
    }

    reference_policy: dict[
        str,
        Any,
    ] | None = None

    if reference_thresholds is not None:
        reference_metrics = evaluate_policy(
            target,
            probabilities,
            amounts,
            thresholds=reference_thresholds,
            cost_assumptions=cost_assumptions,
        )
        (
            reference_feasible,
            reference_violations,
        ) = evaluate_policy_constraints(
            reference_metrics,
            constraints,
        )

        reference_policy = {
            "thresholds": {
                "review_threshold": (
                    reference_thresholds
                    .review_threshold
                ),
                "block_threshold": (
                    reference_thresholds
                    .block_threshold
                ),
            },
            "metrics": reference_metrics,
            "feasible": reference_feasible,
            "constraint_violations": (
                reference_violations
            ),
            "selected_cost_savings": float(
                float(
                    reference_metrics[
                        "total_cost"
                    ]
                )
                - selected_cost
            ),
        }

    return PolicySearchResult(
        selected_thresholds=(
            selected_thresholds
        ),
        selected_metrics=selected_metrics,
        cost_assumptions=cost_assumptions,
        constraints=constraints,
        candidate_thresholds=(
            candidate_thresholds
        ),
        quantile_count=(
            validated_quantile_count
        ),
        objective_round_digits=(
            validated_round_digits
        ),
        evaluated_candidate_count=(
            evaluated_candidate_count
        ),
        feasible_candidate_count=(
            feasible_candidate_count
        ),
        selected_selection_key=selected_key,
        all_allow_metrics=all_allow_metrics,
        all_allow_feasible=(
            all_allow_feasible
        ),
        all_allow_constraint_violations=(
            all_allow_violations
        ),
        reference_policy=reference_policy,
    )
