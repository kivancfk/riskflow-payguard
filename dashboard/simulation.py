"""Read-only threshold scenario simulation for Phase 6."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone

from sqlalchemy import select

from api.logging_db import (
    PredictionEvent,
    PredictionStore,
)
from api.model_loader import LoadedPolicy
from src.policy import (
    CONSTRAINT_TOLERANCE,
    PolicyConstraints,
    PolicyThresholds,
    assign_policy_decisions,
    evaluate_policy,
    evaluate_policy_constraints,
)


DECISION_ORDER = (
    "ALLOW",
    "REVIEW",
    "BLOCK",
)


@dataclass(frozen=True)
class SimulationFilters:
    """Optional time filters for a threshold scenario."""

    start_utc: datetime | None = None
    end_utc: datetime | None = None


@dataclass(frozen=True)
class DecisionWorkload:
    """Decision volumes for one policy threshold pair."""

    total_events: int

    allow_count: int
    review_count: int
    block_count: int

    allow_rate: float
    review_rate: float
    block_rate: float

    intervention_count: int
    intervention_rate: float


@dataclass(frozen=True)
class DecisionTransition:
    """One actual-to-candidate decision transition."""

    frozen_decision: str
    candidate_decision: str
    event_count: int
    event_rate: float


@dataclass(frozen=True)
class OperationalConstraintEvaluation:
    """Evaluation of workload-only frozen operational constraints."""

    max_review_rate: float | None
    max_block_rate: float | None
    max_intervention_rate: float | None

    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class LabeledPolicyComparison:
    """Fraud and economic metrics available only on eligible labeled events."""

    evaluated_event_count: int

    frozen_metrics: dict[
        str,
        float | int,
    ]
    candidate_metrics: dict[
        str,
        float | int,
    ]

    frozen_feasible: bool
    frozen_constraint_violations: tuple[
        str,
        ...,
    ]

    candidate_feasible: bool
    candidate_constraint_violations: tuple[
        str,
        ...,
    ]


@dataclass(frozen=True)
class ThresholdSimulationResult:
    """Complete immutable threshold-simulation result."""

    total_events: int

    labeled_event_count: int
    label_coverage_rate: float

    economics_eligible_event_count: int
    economics_eligible_labeled_rate: float
    economics_excluded_missing_amount_count: int

    frozen_thresholds: PolicyThresholds
    candidate_thresholds: PolicyThresholds

    frozen_workload: DecisionWorkload
    candidate_workload: DecisionWorkload

    changed_decision_count: int
    changed_decision_rate: float

    transitions: tuple[
        DecisionTransition,
        ...,
    ]

    frozen_operational_constraints: (
        OperationalConstraintEvaluation
    )
    candidate_operational_constraints: (
        OperationalConstraintEvaluation
    )

    labeled_comparison: (
        LabeledPolicyComparison
        | None
    )

    labeled_evaluation_unavailable_reason: (
        str
        | None
    )


def _safe_rate(
    numerator: int,
    denominator: int,
) -> float:
    """Return zero when the denominator is empty."""

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def _validate_utc_datetime(
    value: datetime,
    *,
    field_name: str,
) -> None:
    """Require an explicitly timezone-aware UTC datetime."""

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} must be timezone-aware"
        )

    if (
        value.utcoffset()
        != timezone.utc.utcoffset(
            value
        )
    ):
        raise ValueError(
            f"{field_name} must use UTC"
        )


def _validate_filters(
    filters: SimulationFilters,
) -> None:
    """Validate read-only scenario filters."""

    if filters.start_utc is not None:
        _validate_utc_datetime(
            filters.start_utc,
            field_name="start_utc",
        )

    if filters.end_utc is not None:
        _validate_utc_datetime(
            filters.end_utc,
            field_name="end_utc",
        )

    if (
        filters.start_utc is not None
        and filters.end_utc is not None
        and filters.start_utc
        > filters.end_utc
    ):
        raise ValueError(
            "start_utc must be earlier than or equal to end_utc"
        )


def _load_events(
    store: PredictionStore,
    filters: SimulationFilters,
) -> list[
    PredictionEvent
]:
    """Load scenario events without mutating persistence."""

    statement = select(
        PredictionEvent
    )

    if filters.start_utc is not None:
        statement = statement.where(
            PredictionEvent.scored_at_utc
            >= filters.start_utc
        )

    if filters.end_utc is not None:
        statement = statement.where(
            PredictionEvent.scored_at_utc
            <= filters.end_utc
        )

    statement = statement.order_by(
        PredictionEvent.scored_at_utc,
        PredictionEvent.prediction_id,
    )

    with store.session_factory() as session:
        return list(
            session.scalars(
                statement
            )
        )


def _validate_event_contract(
    events: list[
        PredictionEvent
    ],
    loaded_policy: LoadedPolicy,
) -> None:
    """Require logged scores to belong to the loaded frozen policy."""

    bundle = (
        loaded_policy.bundle
    )

    for event in events:
        expected_contract = (
            bundle.baseline_model_version,
            bundle.policy_version,
            (
                bundle
                .calibration_selection
                .selected_method
            ),
            bundle.thresholds.review_threshold,
            bundle.thresholds.block_threshold,
            loaded_policy.artifact_sha256,
        )

        actual_contract = (
            event.model_version,
            event.policy_version,
            event.calibration_method,
            event.review_threshold,
            event.block_threshold,
            event.policy_artifact_sha256,
        )

        if (
            actual_contract
            != expected_contract
        ):
            raise ValueError(
                "Persisted prediction event does not match "
                "the loaded frozen policy contract"
            )

    if not events:
        return

    frozen_recomputed = (
        assign_policy_decisions(
            [
                event.calibrated_probability
                for event in events
            ],
            bundle.thresholds,
        )
    )

    stored_decisions = [
        event.decision
        for event in events
    ]

    if (
        frozen_recomputed.tolist()
        != stored_decisions
    ):
        raise ValueError(
            "Persisted decisions do not match the "
            "loaded frozen policy thresholds"
        )


def _build_workload(
    decisions: list[str],
) -> DecisionWorkload:
    """Aggregate ALLOW, REVIEW, and BLOCK workload."""

    total_events = len(
        decisions
    )

    counts = Counter(
        decisions
    )

    allow_count = counts[
        "ALLOW"
    ]
    review_count = counts[
        "REVIEW"
    ]
    block_count = counts[
        "BLOCK"
    ]

    intervention_count = (
        review_count
        + block_count
    )

    return DecisionWorkload(
        total_events=(
            total_events
        ),
        allow_count=(
            allow_count
        ),
        review_count=(
            review_count
        ),
        block_count=(
            block_count
        ),
        allow_rate=(
            _safe_rate(
                allow_count,
                total_events,
            )
        ),
        review_rate=(
            _safe_rate(
                review_count,
                total_events,
            )
        ),
        block_rate=(
            _safe_rate(
                block_count,
                total_events,
            )
        ),
        intervention_count=(
            intervention_count
        ),
        intervention_rate=(
            _safe_rate(
                intervention_count,
                total_events,
            )
        ),
    )


def _build_transitions(
    frozen_decisions: list[str],
    candidate_decisions: list[str],
) -> tuple[
    DecisionTransition,
    ...,
]:
    """Count actual frozen-to-candidate decision transitions."""

    if len(
        frozen_decisions
    ) != len(
        candidate_decisions
    ):
        raise RuntimeError(
            "Decision transition inputs must have equal length"
        )

    total_events = len(
        frozen_decisions
    )

    counts = Counter(
        zip(
            frozen_decisions,
            candidate_decisions,
        )
    )

    transitions: list[
        DecisionTransition
    ] = []

    for frozen_decision in (
        DECISION_ORDER
    ):
        for candidate_decision in (
            DECISION_ORDER
        ):
            event_count = counts[
                (
                    frozen_decision,
                    candidate_decision,
                )
            ]

            if event_count == 0:
                continue

            transitions.append(
                DecisionTransition(
                    frozen_decision=(
                        frozen_decision
                    ),
                    candidate_decision=(
                        candidate_decision
                    ),
                    event_count=(
                        event_count
                    ),
                    event_rate=(
                        _safe_rate(
                            event_count,
                            total_events,
                        )
                    ),
                )
            )

    return tuple(
        transitions
    )


def _evaluate_operational_constraints(
    workload: DecisionWorkload,
    constraints: PolicyConstraints,
) -> OperationalConstraintEvaluation:
    """Evaluate workload-only constraints without using fraud labels."""

    checks = (
        (
            "max_review_rate",
            workload.review_rate,
        ),
        (
            "max_block_rate",
            workload.block_rate,
        ),
        (
            "max_intervention_rate",
            workload.intervention_rate,
        ),
    )

    violations: list[
        str
    ] = []

    for (
        constraint_name,
        metric_value,
    ) in checks:
        constraint_value = getattr(
            constraints,
            constraint_name,
        )

        if (
            constraint_value
            is None
        ):
            continue

        if (
            metric_value
            > (
                constraint_value
                + CONSTRAINT_TOLERANCE
            )
        ):
            violations.append(
                constraint_name
            )

    return OperationalConstraintEvaluation(
        max_review_rate=(
            constraints
            .max_review_rate
        ),
        max_block_rate=(
            constraints
            .max_block_rate
        ),
        max_intervention_rate=(
            constraints
            .max_intervention_rate
        ),
        feasible=(
            not violations
        ),
        violations=tuple(
            violations
        ),
    )


def _build_labeled_comparison(
    events: list[
        PredictionEvent
    ],
    loaded_policy: LoadedPolicy,
    candidate_thresholds: PolicyThresholds,
) -> tuple[
    int,
    int,
    int,
    LabeledPolicyComparison | None,
    str | None,
]:
    """Evaluate fraud/economic metrics only when labeled data supports them."""

    labeled_events = [
        event
        for event in events
        if (
            event.actual_label
            is not None
        )
    ]

    labeled_event_count = len(
        labeled_events
    )

    eligible_events = [
        event
        for event in labeled_events
        if (
            event.transaction_amount
            is not None
        )
    ]

    eligible_event_count = len(
        eligible_events
    )

    excluded_missing_amount_count = (
        labeled_event_count
        - eligible_event_count
    )

    if labeled_event_count == 0:
        return (
            labeled_event_count,
            eligible_event_count,
            excluded_missing_amount_count,
            None,
            (
                "No ground-truth labels are available "
                "for fraud or economic evaluation."
            ),
        )

    if eligible_event_count == 0:
        return (
            labeled_event_count,
            eligible_event_count,
            excluded_missing_amount_count,
            None,
            (
                "Labeled events do not contain transaction "
                "amounts required for economic evaluation."
            ),
        )

    labels = [
        int(
            event.actual_label
        )
        for event in eligible_events
        if (
            event.actual_label
            is not None
        )
    ]

    if set(
        labels
    ) != {
        0,
        1,
    }:
        return (
            labeled_event_count,
            eligible_event_count,
            excluded_missing_amount_count,
            None,
            (
                "Labeled economic evaluation requires both "
                "legitimate and fraud outcomes."
            ),
        )

    probabilities = [
        event.calibrated_probability
        for event in eligible_events
    ]

    amounts = [
        float(
            event.transaction_amount
        )
        for event in eligible_events
        if (
            event.transaction_amount
            is not None
        )
    ]

    bundle = (
        loaded_policy.bundle
    )

    frozen_metrics = (
        evaluate_policy(
            labels,
            probabilities,
            amounts,
            thresholds=(
                bundle.thresholds
            ),
            cost_assumptions=(
                bundle.cost_assumptions
            ),
        )
    )

    candidate_metrics = (
        evaluate_policy(
            labels,
            probabilities,
            amounts,
            thresholds=(
                candidate_thresholds
            ),
            cost_assumptions=(
                bundle.cost_assumptions
            ),
        )
    )

    (
        frozen_feasible,
        frozen_violations,
    ) = evaluate_policy_constraints(
        frozen_metrics,
        bundle.constraints,
    )

    (
        candidate_feasible,
        candidate_violations,
    ) = evaluate_policy_constraints(
        candidate_metrics,
        bundle.constraints,
    )

    comparison = (
        LabeledPolicyComparison(
            evaluated_event_count=(
                eligible_event_count
            ),
            frozen_metrics=(
                frozen_metrics
            ),
            candidate_metrics=(
                candidate_metrics
            ),
            frozen_feasible=(
                frozen_feasible
            ),
            frozen_constraint_violations=(
                frozen_violations
            ),
            candidate_feasible=(
                candidate_feasible
            ),
            candidate_constraint_violations=(
                candidate_violations
            ),
        )
    )

    return (
        labeled_event_count,
        eligible_event_count,
        excluded_missing_amount_count,
        comparison,
        None,
    )


def simulate_threshold_scenario(
    store: PredictionStore,
    loaded_policy: LoadedPolicy,
    *,
    candidate_thresholds: PolicyThresholds,
    filters: SimulationFilters | None = None,
) -> ThresholdSimulationResult:
    """Compare temporary candidate thresholds with persisted frozen decisions."""

    if not isinstance(
        loaded_policy,
        LoadedPolicy,
    ):
        raise ValueError(
            "loaded_policy must be a LoadedPolicy instance"
        )

    if not isinstance(
        candidate_thresholds,
        PolicyThresholds,
    ):
        raise ValueError(
            "candidate_thresholds must be a PolicyThresholds instance"
        )

    resolved_filters = (
        SimulationFilters()
        if filters is None
        else filters
    )

    _validate_filters(
        resolved_filters
    )

    events = _load_events(
        store,
        resolved_filters,
    )

    _validate_event_contract(
        events,
        loaded_policy,
    )

    frozen_decisions = [
        event.decision
        for event in events
    ]

    if events:
        candidate_decisions = (
            assign_policy_decisions(
                [
                    event
                    .calibrated_probability
                    for event in events
                ],
                candidate_thresholds,
            )
            .tolist()
        )

    else:
        candidate_decisions = []

    frozen_workload = (
        _build_workload(
            frozen_decisions
        )
    )

    candidate_workload = (
        _build_workload(
            candidate_decisions
        )
    )

    changed_decision_count = sum(
        frozen_decision
        != candidate_decision
        for (
            frozen_decision,
            candidate_decision,
        ) in zip(
            frozen_decisions,
            candidate_decisions,
        )
    )

    transitions = (
        _build_transitions(
            frozen_decisions,
            candidate_decisions,
        )
    )

    bundle = (
        loaded_policy.bundle
    )

    (
        labeled_event_count,
        eligible_event_count,
        excluded_missing_amount_count,
        labeled_comparison,
        unavailable_reason,
    ) = _build_labeled_comparison(
        events,
        loaded_policy,
        candidate_thresholds,
    )

    total_events = len(
        events
    )

    return ThresholdSimulationResult(
        total_events=(
            total_events
        ),
        labeled_event_count=(
            labeled_event_count
        ),
        label_coverage_rate=(
            _safe_rate(
                labeled_event_count,
                total_events,
            )
        ),
        economics_eligible_event_count=(
            eligible_event_count
        ),
        economics_eligible_labeled_rate=(
            _safe_rate(
                eligible_event_count,
                labeled_event_count,
            )
        ),
        economics_excluded_missing_amount_count=(
            excluded_missing_amount_count
        ),
        frozen_thresholds=(
            bundle.thresholds
        ),
        candidate_thresholds=(
            candidate_thresholds
        ),
        frozen_workload=(
            frozen_workload
        ),
        candidate_workload=(
            candidate_workload
        ),
        changed_decision_count=(
            changed_decision_count
        ),
        changed_decision_rate=(
            _safe_rate(
                changed_decision_count,
                total_events,
            )
        ),
        transitions=(
            transitions
        ),
        frozen_operational_constraints=(
            _evaluate_operational_constraints(
                frozen_workload,
                bundle.constraints,
            )
        ),
        candidate_operational_constraints=(
            _evaluate_operational_constraints(
                candidate_workload,
                bundle.constraints,
            )
        ),
        labeled_comparison=(
            labeled_comparison
        ),
        labeled_evaluation_unavailable_reason=(
            unavailable_reason
        ),
    )
