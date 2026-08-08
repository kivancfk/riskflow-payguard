"""Streamlit smoke tests for the Phase 6 threshold simulator."""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from api.config import (
    FROZEN_BASELINE_MODEL_VERSION,
    FROZEN_BLOCK_THRESHOLD,
    FROZEN_CALIBRATION_METHOD,
    FROZEN_EXPLANATION_VERSION,
    FROZEN_POLICY_SHA256,
    FROZEN_POLICY_VERSION,
    FROZEN_REASON_CODE_VERSION,
    FROZEN_REVIEW_THRESHOLD,
)
from api.logging_db import (
    PredictionEvent,
    PredictionEventCreate,
    PredictionStore,
)
from api.prediction_labels import (
    PredictionLabelUpdate,
    record_prediction_labels,
)
from src.policy import (
    PolicyThresholds,
    assign_policy_decisions,
)


BASE_TIME = datetime(
    2026,
    8,
    8,
    8,
    0,
    tzinfo=timezone.utc,
)

FROZEN_THRESHOLDS = (
    PolicyThresholds(
        review_threshold=(
            FROZEN_REVIEW_THRESHOLD
        ),
        block_threshold=(
            FROZEN_BLOCK_THRESHOLD
        ),
    )
)


def _event(
    transaction_id: str,
    *,
    probability: float,
    amount: float,
    hours_after_base: int,
) -> PredictionEventCreate:
    """Return one event matching the exact frozen policy contract."""

    decision = str(
        assign_policy_decisions(
            [
                probability
            ],
            FROZEN_THRESHOLDS,
        )[0]
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
            decision
        ),
        transaction_amount=(
            amount
        ),
        model_version=(
            FROZEN_BASELINE_MODEL_VERSION
        ),
        policy_version=(
            FROZEN_POLICY_VERSION
        ),
        calibration_method=(
            FROZEN_CALIBRATION_METHOD
        ),
        review_threshold=(
            FROZEN_REVIEW_THRESHOLD
        ),
        block_threshold=(
            FROZEN_BLOCK_THRESHOLD
        ),
        policy_artifact_sha256=(
            FROZEN_POLICY_SHA256
        ),
        explanation_version=(
            FROZEN_EXPLANATION_VERSION
        ),
        reason_code_version=(
            FROZEN_REASON_CODE_VERSION
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


def _seed_events(
    database_url: str,
    *,
    include_labels: bool,
) -> None:
    """Persist ALLOW, REVIEW, and BLOCK examples."""

    store = PredictionStore(
        database_url
    )
    store.init_schema()

    try:
        store.add_events(
            [
                _event(
                    "tx-allow",
                    probability=0.05,
                    amount=10.0,
                    hours_after_base=1,
                ),
                _event(
                    "tx-review",
                    probability=0.50,
                    amount=30.0,
                    hours_after_base=2,
                ),
                _event(
                    "tx-block",
                    probability=0.90,
                    amount=40.0,
                    hours_after_base=3,
                ),
            ]
        )

        if include_labels:
            record_prediction_labels(
                store,
                [
                    PredictionLabelUpdate(
                        transaction_id=(
                            "tx-allow"
                        ),
                        actual_label=0,
                        recorded_at_utc=(
                            BASE_TIME
                            + timedelta(
                                hours=4
                            )
                        ),
                    ),
                    PredictionLabelUpdate(
                        transaction_id=(
                            "tx-review"
                        ),
                        actual_label=1,
                        recorded_at_utc=(
                            BASE_TIME
                            + timedelta(
                                hours=4
                            )
                        ),
                    ),
                    PredictionLabelUpdate(
                        transaction_id=(
                            "tx-block"
                        ),
                        actual_label=0,
                        recorded_at_utc=(
                            BASE_TIME
                            + timedelta(
                                hours=4
                            )
                        ),
                    ),
                ],
            )

    finally:
        store.dispose()


def _open_simulator() -> AppTest:
    """Open the dashboard and switch to the simulator view."""

    app_test = (
        AppTest.from_file(
            "dashboard/app.py"
        )
        .run(
            timeout=10
        )
    )

    app_test.sidebar.radio[
        0
    ].set_value(
        "Threshold Simulator"
    )

    return app_test.run(
        timeout=10
    )


def _metric_values(
    app_test: AppTest,
) -> dict[
    str,
    str,
]:
    """Return rendered metric values by label."""

    return {
        metric.label: metric.value
        for metric
        in app_test.metric
    }


def _database_state(
    database_url: str,
) -> tuple[
    tuple[
        object,
        ...,
    ],
    ...,
]:
    """Return immutable persisted state for read-only verification."""

    store = PredictionStore(
        database_url
    )
    store.init_schema()

    try:
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
                event.review_threshold,
                event.block_threshold,
                event.actual_label,
                event.label_recorded_at_utc,
            )
            for event in events
        )

    finally:
        store.dispose()


def test_empty_threshold_simulator_smoke() -> None:
    """Render an empty scenario without inventing fraud metrics."""

    app_test = (
        _open_simulator()
    )

    assert (
        len(
            app_test.exception
        )
        == 0
    )

    assert (
        app_test.header[
            0
        ].value
        == "Threshold Simulator"
    )

    assert len(
        app_test.number_input
    ) == 2

    assert (
        app_test.number_input[
            0
        ].value
        == pytest.approx(
            FROZEN_REVIEW_THRESHOLD
        )
    )

    assert (
        app_test.number_input[
            1
        ].value
        == pytest.approx(
            FROZEN_BLOCK_THRESHOLD
        )
    )

    metrics = (
        _metric_values(
            app_test
        )
    )

    assert (
        metrics[
            "Scenario events"
        ]
        == "0"
    )
    assert (
        metrics[
            "Changed decisions"
        ]
        == "0"
    )

    assert any(
        "No ground-truth labels"
        in info.value
        for info
        in app_test.info
    )


def test_default_candidate_reproduces_frozen_workload(
    runtime_database_url: str,
) -> None:
    """Default candidate thresholds must exactly reproduce frozen decisions."""

    _seed_events(
        runtime_database_url,
        include_labels=False,
    )

    app_test = (
        _open_simulator()
    )

    assert (
        len(
            app_test.exception
        )
        == 0
    )

    metrics = (
        _metric_values(
            app_test
        )
    )

    assert (
        metrics[
            "Scenario events"
        ]
        == "3"
    )
    assert (
        metrics[
            "Changed decisions"
        ]
        == "0"
    )
    assert (
        metrics[
            "Changed-decision rate"
        ]
        == "0.0%"
    )
    assert (
        metrics[
            "Frozen intervention"
        ]
        == "66.7%"
    )
    assert (
        metrics[
            "Candidate intervention"
        ]
        == "66.7%"
    )


def test_candidate_inputs_change_scenario_only(
    runtime_database_url: str,
) -> None:
    """Temporary controls should change scenario workload without persistence."""

    _seed_events(
        runtime_database_url,
        include_labels=False,
    )

    app_test = (
        _open_simulator()
    )

    app_test.number_input[
        0
    ].set_value(
        0.60
    )
    app_test.number_input[
        1
    ].set_value(
        0.95
    )

    app_test = app_test.run(
        timeout=10
    )

    assert (
        len(
            app_test.exception
        )
        == 0
    )

    metrics = (
        _metric_values(
            app_test
        )
    )

    assert (
        metrics[
            "Changed decisions"
        ]
        == "2"
    )
    assert (
        metrics[
            "Changed-decision rate"
        ]
        == "66.7%"
    )
    assert (
        metrics[
            "Frozen intervention"
        ]
        == "66.7%"
    )
    assert (
        metrics[
            "Candidate intervention"
        ]
        == "33.3%"
    )


def test_invalid_candidate_order_is_rejected_in_ui() -> None:
    """Do not construct a scenario when review exceeds block."""

    app_test = (
        _open_simulator()
    )

    app_test.number_input[
        0
    ].set_value(
        0.90
    )
    app_test.number_input[
        1
    ].set_value(
        0.80
    )

    app_test = app_test.run(
        timeout=10
    )

    assert (
        len(
            app_test.exception
        )
        == 0
    )

    assert any(
        (
            "Candidate review threshold must be less than "
            "or equal to the candidate block threshold."
        )
        in error.value
        for error
        in app_test.error
    )


def test_labeled_comparison_appears_only_with_supported_data(
    runtime_database_url: str,
) -> None:
    """Expose fraud/economic metrics when both labeled classes are available."""

    _seed_events(
        runtime_database_url,
        include_labels=True,
    )

    app_test = (
        _open_simulator()
    )

    assert (
        len(
            app_test.exception
        )
        == 0
    )

    metrics = (
        _metric_values(
            app_test
        )
    )

    assert (
        metrics[
            "Labeled events"
        ]
        == "3"
    )
    assert (
        metrics[
            "Economic evaluation events"
        ]
        == "3"
    )
    assert (
        metrics[
            "Missing-amount exclusions"
        ]
        == "0"
    )

    assert any(
        subheader.value
        == (
            "Labeled Fraud / Economic Comparison"
        )
        for subheader
        in app_test.subheader
    )


def test_threshold_simulator_is_read_only(
    runtime_database_url: str,
) -> None:
    """Dashboard candidate thresholds must never modify persisted state."""

    _seed_events(
        runtime_database_url,
        include_labels=True,
    )

    before = (
        _database_state(
            runtime_database_url
        )
    )

    app_test = (
        _open_simulator()
    )

    app_test.number_input[
        0
    ].set_value(
        0.60
    )
    app_test.number_input[
        1
    ].set_value(
        0.95
    )

    app_test = app_test.run(
        timeout=10
    )

    assert (
        len(
            app_test.exception
        )
        == 0
    )

    after = (
        _database_state(
            runtime_database_url
        )
    )

    assert after == before
