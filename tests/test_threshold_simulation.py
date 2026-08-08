"""Tests for Phase 6 read-only threshold simulation."""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from api.logging_db import (
    PredictionEvent,
    PredictionEventCreate,
    PredictionStore,
)
from api.model_loader import LoadedPolicy
from api.prediction_labels import (
    PredictionLabelUpdate,
    record_prediction_labels,
)
from dashboard.simulation import (
    SimulationFilters,
    simulate_threshold_scenario,
)
from src.policy import (
    PolicyConstraints,
    PolicyCostAssumptions,
    PolicyThresholds,
    assign_policy_decisions,
    evaluate_policy,
)


BASE_TIME = datetime(
    2026,
    8,
    8,
    8,
    0,
    tzinfo=timezone.utc,
)

POLICY_SHA256 = (
    "a" * 64
)


@pytest.fixture
def prediction_store(
    tmp_path: Path,
) -> PredictionStore:
    """Create one isolated SQLite scenario store."""

    database_path = (
        tmp_path
        / "threshold-simulation.db"
    )

    store = PredictionStore(
        f"sqlite:///{database_path}"
    )
    store.init_schema()

    try:
        yield store
    finally:
        store.dispose()


@pytest.fixture
def loaded_policy() -> LoadedPolicy:
    """Return a small deterministic frozen-policy contract for simulation."""

    thresholds = (
        PolicyThresholds(
            review_threshold=0.20,
            block_threshold=0.80,
        )
    )

    cost_assumptions = (
        PolicyCostAssumptions(
            cost_per_review=2.0,
            cost_per_legitimate_block=5.0,
            fraud_loss_multiplier=1.0,
            review_fraud_capture_rate=0.50,
        )
    )

    constraints = (
        PolicyConstraints(
            max_review_rate=0.50,
            max_block_rate=0.50,
            max_intervention_rate=0.75,
        )
    )

    bundle = SimpleNamespace(
        baseline_model_version=(
            "baseline-v1"
        ),
        policy_version=(
            "calibrated-policy-v1"
        ),
        calibration_selection=(
            SimpleNamespace(
                selected_method="sigmoid"
            )
        ),
        thresholds=(
            thresholds
        ),
        cost_assumptions=(
            cost_assumptions
        ),
        constraints=(
            constraints
        ),
    )

    return LoadedPolicy(
        bundle=bundle,  # type: ignore[arg-type]
        artifact_path=Path(
            "unused-policy.joblib"
        ),
        artifact_sha256=(
            POLICY_SHA256
        ),
    )


def _event(
    loaded_policy: LoadedPolicy,
    transaction_id: str | int,
    *,
    probability: float,
    hours_after_base: int,
    amount: float | None,
    decision: str | None = None,
    policy_version: str | None = None,
) -> PredictionEventCreate:
    """Return one event matching the supplied frozen policy."""

    bundle = (
        loaded_policy.bundle
    )

    resolved_decision = (
        str(
            assign_policy_decisions(
                [
                    probability
                ],
                bundle.thresholds,
            )[0]
        )
        if decision is None
        else decision
    )

    return PredictionEventCreate(
        transaction_id=(
            transaction_id
        ),
        raw_model_score=(
            probability
        ),
        calibrated_probability=(
            probability
        ),
        decision=(
            resolved_decision
        ),
        transaction_amount=(
            amount
        ),
        model_version=(
            bundle
            .baseline_model_version
        ),
        policy_version=(
            (
                bundle.policy_version
            )
            if policy_version is None
            else policy_version
        ),
        calibration_method=(
            bundle
            .calibration_selection
            .selected_method
        ),
        review_threshold=(
            bundle
            .thresholds
            .review_threshold
        ),
        block_threshold=(
            bundle
            .thresholds
            .block_threshold
        ),
        policy_artifact_sha256=(
            loaded_policy
            .artifact_sha256
        ),
        explanation_version=(
            "shap-explanation-v1"
        ),
        reason_code_version=(
            "reason-codes-v1"
        ),
        reason_codes=(),
        scored_at_utc=(
            BASE_TIME
            + timedelta(
                hours=(
                    hours_after_base
                )
            )
        ),
    )


def _populate_store(
    store: PredictionStore,
    loaded_policy: LoadedPolicy,
    *,
    include_labels: bool,
    missing_amount_transaction: str | None = None,
) -> None:
    """Persist five deterministic events spanning frozen boundaries."""

    amounts: dict[
        str,
        float | None,
    ] = {
        "tx-1": 10.0,
        "tx-2": 20.0,
        "tx-3": 30.0,
        "tx-4": 40.0,
        "tx-5": 50.0,
    }

    if (
        missing_amount_transaction
        is not None
    ):
        amounts[
            missing_amount_transaction
        ] = None

    store.add_events(
        [
            _event(
                loaded_policy,
                "tx-1",
                probability=0.10,
                hours_after_base=1,
                amount=amounts[
                    "tx-1"
                ],
            ),
            _event(
                loaded_policy,
                "tx-2",
                probability=0.20,
                hours_after_base=2,
                amount=amounts[
                    "tx-2"
                ],
            ),
            _event(
                loaded_policy,
                "tx-3",
                probability=0.50,
                hours_after_base=3,
                amount=amounts[
                    "tx-3"
                ],
            ),
            _event(
                loaded_policy,
                "tx-4",
                probability=0.80,
                hours_after_base=4,
                amount=amounts[
                    "tx-4"
                ],
            ),
            _event(
                loaded_policy,
                "tx-5",
                probability=0.90,
                hours_after_base=5,
                amount=amounts[
                    "tx-5"
                ],
            ),
        ]
    )

    if not include_labels:
        return

    labels = {
        "tx-1": 0,
        "tx-2": 0,
        "tx-3": 1,
        "tx-4": 0,
        "tx-5": 1,
    }

    record_prediction_labels(
        store,
        [
            PredictionLabelUpdate(
                transaction_id=(
                    transaction_id
                ),
                actual_label=(
                    actual_label
                ),
                recorded_at_utc=(
                    BASE_TIME
                    + timedelta(
                        hours=6
                    )
                ),
            )
            for (
                transaction_id,
                actual_label,
            ) in labels.items()
        ],
    )


def _database_state(
    store: PredictionStore,
) -> tuple[
    tuple[
        object,
        ...,
    ],
    ...,
]:
    """Return immutable event state for read-only assertions."""

    with store.session_factory() as session:
        events = list(
            session.scalars(
                select(
                    PredictionEvent
                ).order_by(
                    PredictionEvent
                    .prediction_id
                )
            )
        )

    return tuple(
        (
            event.prediction_id,
            event.transaction_id,
            event.calibrated_probability,
            event.decision,
            event.transaction_amount,
            event.model_version,
            event.policy_version,
            event.calibration_method,
            event.review_threshold,
            event.block_threshold,
            event.policy_artifact_sha256,
            event.actual_label,
            event.label_recorded_at_utc,
        )
        for event in events
    )


def test_empty_store_returns_workload_only_result(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Handle an empty monitoring database without inventing fraud metrics."""

    result = (
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                PolicyThresholds(
                    0.30,
                    0.90,
                )
            ),
        )
    )

    assert result.total_events == 0

    assert (
        result.frozen_workload
        .total_events
        == 0
    )
    assert (
        result.candidate_workload
        .total_events
        == 0
    )

    assert (
        result.changed_decision_count
        == 0
    )
    assert (
        result.changed_decision_rate
        == 0.0
    )

    assert result.transitions == ()

    assert result.labeled_event_count == 0
    assert (
        result.labeled_comparison
        is None
    )
    assert (
        result
        .labeled_evaluation_unavailable_reason
        is not None
    )


def test_candidate_thresholds_change_workload_and_transitions(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Compare candidate workload with actual persisted frozen decisions."""

    _populate_store(
        prediction_store,
        loaded_policy,
        include_labels=False,
    )

    result = (
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                PolicyThresholds(
                    0.40,
                    0.90,
                )
            ),
        )
    )

    frozen = (
        result.frozen_workload
    )

    assert frozen.allow_count == 1
    assert frozen.review_count == 2
    assert frozen.block_count == 2
    assert (
        frozen.intervention_count
        == 4
    )

    candidate = (
        result.candidate_workload
    )

    assert candidate.allow_count == 2
    assert candidate.review_count == 2
    assert candidate.block_count == 1
    assert (
        candidate.intervention_count
        == 3
    )

    assert (
        result.changed_decision_count
        == 2
    )
    assert (
        result.changed_decision_rate
        == pytest.approx(
            0.40
        )
    )

    assert [
        (
            item.frozen_decision,
            item.candidate_decision,
            item.event_count,
        )
        for item in result.transitions
    ] == [
        (
            "ALLOW",
            "ALLOW",
            1,
        ),
        (
            "REVIEW",
            "ALLOW",
            1,
        ),
        (
            "REVIEW",
            "REVIEW",
            1,
        ),
        (
            "BLOCK",
            "REVIEW",
            1,
        ),
        (
            "BLOCK",
            "BLOCK",
            1,
        ),
    ]


def test_candidate_equal_to_frozen_reproduces_actual_decisions(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Require the critical frozen-threshold parity invariant."""

    _populate_store(
        prediction_store,
        loaded_policy,
        include_labels=False,
    )

    result = (
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                loaded_policy
                .bundle
                .thresholds
            ),
        )
    )

    assert (
        result.changed_decision_count
        == 0
    )
    assert (
        result.changed_decision_rate
        == 0.0
    )

    assert (
        result.candidate_workload
        == result.frozen_workload
    )


def test_candidate_boundary_semantics_reuse_core_policy_logic(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Treat review and block thresholds as inclusive lower boundaries."""

    _populate_store(
        prediction_store,
        loaded_policy,
        include_labels=False,
    )

    result = (
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                PolicyThresholds(
                    0.20,
                    0.80,
                )
            ),
        )
    )

    assert (
        result.candidate_workload
        .allow_count
        == 1
    )
    assert (
        result.candidate_workload
        .review_count
        == 2
    )
    assert (
        result.candidate_workload
        .block_count
        == 2
    )


def test_unlabeled_scenario_does_not_expose_fraud_metrics(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Keep fraud and economic claims unavailable without labels."""

    _populate_store(
        prediction_store,
        loaded_policy,
        include_labels=False,
    )

    result = (
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                PolicyThresholds(
                    0.40,
                    0.90,
                )
            ),
        )
    )

    assert result.labeled_event_count == 0
    assert (
        result.label_coverage_rate
        == 0.0
    )

    assert (
        result.labeled_comparison
        is None
    )

    assert (
        "No ground-truth labels"
        in (
            result
            .labeled_evaluation_unavailable_reason
            or ""
        )
    )


def test_labeled_metrics_match_direct_core_policy_evaluation(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Reuse the exact Phase 3 policy evaluator for labeled scenarios."""

    _populate_store(
        prediction_store,
        loaded_policy,
        include_labels=True,
    )

    candidate_thresholds = (
        PolicyThresholds(
            0.40,
            0.90,
        )
    )

    result = (
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                candidate_thresholds
            ),
        )
    )

    comparison = (
        result.labeled_comparison
    )

    assert comparison is not None
    assert (
        comparison
        .evaluated_event_count
        == 5
    )

    labels = [
        0,
        0,
        1,
        0,
        1,
    ]
    probabilities = [
        0.10,
        0.20,
        0.50,
        0.80,
        0.90,
    ]
    amounts = [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ]

    expected_frozen = (
        evaluate_policy(
            labels,
            probabilities,
            amounts,
            thresholds=(
                loaded_policy
                .bundle
                .thresholds
            ),
            cost_assumptions=(
                loaded_policy
                .bundle
                .cost_assumptions
            ),
        )
    )

    expected_candidate = (
        evaluate_policy(
            labels,
            probabilities,
            amounts,
            thresholds=(
                candidate_thresholds
            ),
            cost_assumptions=(
                loaded_policy
                .bundle
                .cost_assumptions
            ),
        )
    )

    assert (
        comparison.frozen_metrics
        == expected_frozen
    )
    assert (
        comparison.candidate_metrics
        == expected_candidate
    )

    assert result.labeled_event_count == 5
    assert (
        result.label_coverage_rate
        == pytest.approx(
            1.0
        )
    )
    assert (
        result
        .economics_eligible_event_count
        == 5
    )
    assert (
        result
        .economics_eligible_labeled_rate
        == pytest.approx(
            1.0
        )
    )


def test_missing_amount_is_excluded_from_labeled_economics(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Evaluate only labeled events that also have transaction amounts."""

    _populate_store(
        prediction_store,
        loaded_policy,
        include_labels=True,
        missing_amount_transaction=(
            "tx-3"
        ),
    )

    result = (
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                PolicyThresholds(
                    0.40,
                    0.90,
                )
            ),
        )
    )

    assert result.labeled_event_count == 5
    assert (
        result
        .economics_eligible_event_count
        == 4
    )
    assert (
        result
        .economics_excluded_missing_amount_count
        == 1
    )

    comparison = (
        result.labeled_comparison
    )

    assert comparison is not None

    assert (
        comparison
        .evaluated_event_count
        == 4
    )
    assert (
        comparison
        .candidate_metrics[
            "transaction_count"
        ]
        == 4
    )


def test_one_class_labeled_subset_does_not_claim_policy_performance(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Require both binary outcome classes before fraud evaluation."""

    prediction_store.add_events(
        [
            _event(
                loaded_policy,
                "tx-legit-1",
                probability=0.10,
                hours_after_base=1,
                amount=10.0,
            ),
            _event(
                loaded_policy,
                "tx-legit-2",
                probability=0.50,
                hours_after_base=2,
                amount=20.0,
            ),
        ]
    )

    record_prediction_labels(
        prediction_store,
        [
            PredictionLabelUpdate(
                transaction_id=(
                    "tx-legit-1"
                ),
                actual_label=0,
                recorded_at_utc=(
                    BASE_TIME
                    + timedelta(
                        hours=3
                    )
                ),
            ),
            PredictionLabelUpdate(
                transaction_id=(
                    "tx-legit-2"
                ),
                actual_label=0,
                recorded_at_utc=(
                    BASE_TIME
                    + timedelta(
                        hours=3
                    )
                ),
            ),
        ],
    )

    result = (
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                PolicyThresholds(
                    0.40,
                    0.90,
                )
            ),
        )
    )

    assert (
        result.labeled_comparison
        is None
    )
    assert (
        "requires both legitimate and fraud outcomes"
        in (
            result
            .labeled_evaluation_unavailable_reason
            or ""
        )
    )


def test_operational_constraints_use_all_scenario_events(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Evaluate review/block/intervention constraints without labels."""

    _populate_store(
        prediction_store,
        loaded_policy,
        include_labels=False,
    )

    result = (
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                PolicyThresholds(
                    0.40,
                    0.90,
                )
            ),
        )
    )

    assert (
        result
        .frozen_operational_constraints
        .feasible
        is False
    )
    assert (
        result
        .frozen_operational_constraints
        .violations
        == (
            "max_intervention_rate",
        )
    )

    assert (
        result
        .candidate_operational_constraints
        .feasible
        is True
    )
    assert (
        result
        .candidate_operational_constraints
        .violations
        == ()
    )


def test_time_filter_applies_before_simulation(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Simulate only events inside the requested UTC time range."""

    _populate_store(
        prediction_store,
        loaded_policy,
        include_labels=False,
    )

    result = (
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                PolicyThresholds(
                    0.40,
                    0.90,
                )
            ),
            filters=(
                SimulationFilters(
                    start_utc=(
                        BASE_TIME
                        + timedelta(
                            hours=2
                        )
                    ),
                    end_utc=(
                        BASE_TIME
                        + timedelta(
                            hours=4
                        )
                    ),
                )
            ),
        )
    )

    assert result.total_events == 3

    assert (
        result.frozen_workload
        .review_count
        == 2
    )
    assert (
        result.frozen_workload
        .block_count
        == 1
    )


def test_mismatched_policy_provenance_is_rejected(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Do not mix events from another policy contract into a scenario."""

    prediction_store.add_events(
        [
            _event(
                loaded_policy,
                "tx-mismatch",
                probability=0.50,
                hours_after_base=1,
                amount=10.0,
                policy_version=(
                    "other-policy"
                ),
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "does not match the loaded frozen policy contract"
        ),
    ):
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                PolicyThresholds(
                    0.40,
                    0.90,
                )
            ),
        )


def test_corrupted_persisted_decision_is_rejected(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Require persisted actual decisions to match their frozen thresholds."""

    prediction_store.add_events(
        [
            _event(
                loaded_policy,
                "tx-corrupt",
                probability=0.10,
                hours_after_base=1,
                amount=10.0,
                decision="BLOCK",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "Persisted decisions do not match"
        ),
    ):
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                PolicyThresholds(
                    0.40,
                    0.90,
                )
            ),
        )


@pytest.mark.parametrize(
    "filters",
    [
        SimulationFilters(
            start_utc=datetime(
                2026,
                8,
                8,
                8,
                0,
            )
        ),
        SimulationFilters(
            start_utc=datetime(
                2026,
                8,
                8,
                11,
                0,
                tzinfo=timezone.utc,
            ),
            end_utc=datetime(
                2026,
                8,
                8,
                10,
                0,
                tzinfo=timezone.utc,
            ),
        ),
        SimulationFilters(
            start_utc=datetime(
                2026,
                8,
                8,
                11,
                0,
                tzinfo=timezone(
                    timedelta(
                        hours=3
                    )
                ),
            )
        ),
    ],
)
def test_invalid_simulation_filters_are_rejected(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
    filters: SimulationFilters,
) -> None:
    """Reject naive, reversed, or non-UTC scenario filters."""

    with pytest.raises(
        ValueError
    ):
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds=(
                PolicyThresholds(
                    0.40,
                    0.90,
                )
            ),
            filters=filters,
        )


def test_invalid_candidate_threshold_object_is_rejected(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Require candidates to use the core validated threshold type."""

    with pytest.raises(
        ValueError,
        match=(
            "candidate_thresholds must be a PolicyThresholds instance"
        ),
    ):
        simulate_threshold_scenario(
            prediction_store,
            loaded_policy,
            candidate_thresholds="invalid",  # type: ignore[arg-type]
        )


def test_threshold_simulation_is_read_only(
    prediction_store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Never write candidate decisions or thresholds back to persistence."""

    _populate_store(
        prediction_store,
        loaded_policy,
        include_labels=True,
    )

    before = _database_state(
        prediction_store
    )

    simulate_threshold_scenario(
        prediction_store,
        loaded_policy,
        candidate_thresholds=(
            PolicyThresholds(
                0.40,
                0.90,
            )
        ),
    )

    after = _database_state(
        prediction_store
    )

    assert after == before
