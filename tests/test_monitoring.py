"""Tests for Phase 6 read-only prediction monitoring queries."""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import pytest
from sqlalchemy import select

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
from dashboard.monitoring import (
    MonitoringFilters,
    load_monitoring_snapshot,
)


BASE_TIME = datetime(
    2026,
    8,
    8,
    8,
    0,
    tzinfo=timezone.utc,
)


@pytest.fixture
def prediction_store(
    tmp_path: Path,
) -> PredictionStore:
    """Create an isolated SQLite monitoring store."""

    database_path = (
        tmp_path
        / "monitoring.db"
    )

    store = PredictionStore(
        f"sqlite:///{database_path}"
    )
    store.init_schema()

    try:
        yield store
    finally:
        store.dispose()


def _event(
    transaction_id: str | int,
    *,
    decision: str,
    probability: float,
    hours_after_base: int,
    amount: float | None,
    reason_codes: tuple[
        str,
        ...,
    ],
) -> PredictionEventCreate:
    """Return one deterministic persisted prediction event."""

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
        reason_codes=(
            reason_codes
        ),
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
) -> None:
    """Persist a deterministic four-event monitoring dataset."""

    store.add_events(
        [
            _event(
                "tx-allow",
                decision="ALLOW",
                probability=0.05,
                hours_after_base=1,
                amount=10.0,
                reason_codes=(),
            ),
            _event(
                "tx-review-1",
                decision="REVIEW",
                probability=0.40,
                hours_after_base=2,
                amount=20.0,
                reason_codes=(
                    "RISK_A",
                    "RISK_B",
                ),
            ),
            _event(
                "tx-block",
                decision="BLOCK",
                probability=0.95,
                hours_after_base=3,
                amount=30.0,
                reason_codes=(
                    "RISK_A",
                ),
            ),
            _event(
                "tx-review-2",
                decision="REVIEW",
                probability=0.30,
                hours_after_base=4,
                amount=None,
                reason_codes=(
                    "RISK_C",
                ),
            ),
        ]
    )

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
                        hours=5
                    )
                ),
            ),
            PredictionLabelUpdate(
                transaction_id=(
                    "tx-block"
                ),
                actual_label=1,
                recorded_at_utc=(
                    BASE_TIME
                    + timedelta(
                        hours=5
                    )
                ),
            ),
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
    """Return immutable event state for read-only verification."""

    with store.session_factory() as session:
        events = list(
            session.scalars(
                select(
                    PredictionEvent
                ).order_by(
                    PredictionEvent.prediction_id
                )
            )
        )

    return tuple(
        (
            event.prediction_id,
            event.transaction_id,
            event.raw_model_score,
            event.calibrated_probability,
            event.decision,
            event.transaction_amount,
            event.model_version,
            event.policy_version,
            event.calibration_method,
            event.review_threshold,
            event.block_threshold,
            event.policy_artifact_sha256,
            event.explanation_version,
            event.reason_code_version,
            tuple(
                event.reason_codes
            ),
            event.scored_at_utc,
            event.actual_label,
            event.label_recorded_at_utc,
        )
        for event in events
    )


def test_empty_store_returns_zero_monitoring_snapshot(
    prediction_store: PredictionStore,
) -> None:
    """Return useful deterministic monitoring values for an empty store."""

    snapshot = (
        load_monitoring_snapshot(
            prediction_store
        )
    )

    overview = (
        snapshot.overview
    )

    assert overview.total_events == 0

    assert overview.allow_count == 0
    assert overview.review_count == 0
    assert overview.block_count == 0

    assert overview.allow_rate == 0.0
    assert overview.review_rate == 0.0
    assert overview.block_rate == 0.0

    assert (
        overview.intervention_count
        == 0
    )
    assert (
        overview.intervention_rate
        == 0.0
    )

    assert overview.labeled_count == 0
    assert overview.unlabeled_count == 0
    assert (
        overview.label_coverage_rate
        == 0.0
    )

    assert (
        overview
        .average_calibrated_probability
        is None
    )
    assert (
        overview
        .minimum_calibrated_probability
        is None
    )
    assert (
        overview
        .maximum_calibrated_probability
        is None
    )

    assert (
        snapshot.score_observations
        == ()
    )
    assert (
        snapshot.reason_code_frequencies
        == ()
    )
    assert (
        snapshot.recent_flagged_predictions
        == ()
    )
    assert (
        snapshot.policy_provenance
        == ()
    )


def test_monitoring_overview_uses_persisted_actual_decisions(
    prediction_store: PredictionStore,
) -> None:
    """Aggregate stored production decisions without threshold recomputation."""

    _populate_store(
        prediction_store
    )

    snapshot = (
        load_monitoring_snapshot(
            prediction_store
        )
    )

    overview = (
        snapshot.overview
    )

    assert overview.total_events == 4

    assert overview.allow_count == 1
    assert overview.review_count == 2
    assert overview.block_count == 1

    assert overview.allow_rate == pytest.approx(
        0.25
    )
    assert overview.review_rate == pytest.approx(
        0.50
    )
    assert overview.block_rate == pytest.approx(
        0.25
    )

    assert (
        overview.intervention_count
        == 3
    )
    assert (
        overview.intervention_rate
        == pytest.approx(
            0.75
        )
    )

    assert overview.labeled_count == 2
    assert overview.unlabeled_count == 2
    assert (
        overview.label_coverage_rate
        == pytest.approx(
            0.50
        )
    )

    assert (
        overview
        .average_calibrated_probability
        == pytest.approx(
            0.425
        )
    )
    assert (
        overview
        .minimum_calibrated_probability
        == pytest.approx(
            0.05
        )
    )
    assert (
        overview
        .maximum_calibrated_probability
        == pytest.approx(
            0.95
        )
    )


def test_score_observations_are_chronological(
    prediction_store: PredictionStore,
) -> None:
    """Expose score observations in deterministic event-time order."""

    _populate_store(
        prediction_store
    )

    observations = (
        load_monitoring_snapshot(
            prediction_store
        )
        .score_observations
    )

    assert [
        observation.transaction_id
        for observation in observations
    ] == [
        "tx-allow",
        "tx-review-1",
        "tx-block",
        "tx-review-2",
    ]

    assert [
        observation.calibrated_probability
        for observation in observations
    ] == pytest.approx(
        [
            0.05,
            0.40,
            0.95,
            0.30,
        ]
    )

    assert [
        observation.decision
        for observation in observations
    ] == [
        "ALLOW",
        "REVIEW",
        "BLOCK",
        "REVIEW",
    ]


def test_reason_code_frequencies_count_events(
    prediction_store: PredictionStore,
) -> None:
    """Count each reason at most once per persisted event."""

    _populate_store(
        prediction_store
    )

    frequencies = (
        load_monitoring_snapshot(
            prediction_store
        )
        .reason_code_frequencies
    )

    assert [
        (
            item.reason_code,
            item.event_count,
            item.event_rate,
        )
        for item in frequencies
    ] == [
        (
            "RISK_A",
            2,
            pytest.approx(
                0.50
            ),
        ),
        (
            "RISK_B",
            1,
            pytest.approx(
                0.25
            ),
        ),
        (
            "RISK_C",
            1,
            pytest.approx(
                0.25
            ),
        ),
    ]


def test_recent_flagged_predictions_are_newest_first(
    prediction_store: PredictionStore,
) -> None:
    """Return only REVIEW/BLOCK events ordered newest first."""

    _populate_store(
        prediction_store
    )

    recent = (
        load_monitoring_snapshot(
            prediction_store,
            recent_limit=2,
        )
        .recent_flagged_predictions
    )

    assert [
        item.transaction_id
        for item in recent
    ] == [
        "tx-review-2",
        "tx-block",
    ]

    assert [
        item.decision
        for item in recent
    ] == [
        "REVIEW",
        "BLOCK",
    ]

    assert recent[0].actual_label is None
    assert recent[1].actual_label == 1

    assert (
        recent[0].transaction_amount
        is None
    )
    assert (
        recent[1].transaction_amount
        == pytest.approx(
            30.0
        )
    )


def test_policy_provenance_summarizes_stored_contract(
    prediction_store: PredictionStore,
) -> None:
    """Expose the exact frozen provenance stored with prediction events."""

    _populate_store(
        prediction_store
    )

    provenance = (
        load_monitoring_snapshot(
            prediction_store
        )
        .policy_provenance
    )

    assert len(
        provenance
    ) == 1

    contract = (
        provenance[0]
    )

    assert (
        contract.model_version
        == FROZEN_BASELINE_MODEL_VERSION
    )
    assert (
        contract.policy_version
        == FROZEN_POLICY_VERSION
    )
    assert (
        contract.calibration_method
        == FROZEN_CALIBRATION_METHOD
    )
    assert (
        contract.review_threshold
        == FROZEN_REVIEW_THRESHOLD
    )
    assert (
        contract.block_threshold
        == FROZEN_BLOCK_THRESHOLD
    )
    assert (
        contract.policy_artifact_sha256
        == FROZEN_POLICY_SHA256
    )
    assert (
        contract.explanation_version
        == FROZEN_EXPLANATION_VERSION
    )
    assert (
        contract.reason_code_version
        == FROZEN_REASON_CODE_VERSION
    )
    assert contract.event_count == 4


def test_monitoring_filters_apply_before_aggregation(
    prediction_store: PredictionStore,
) -> None:
    """Apply UTC time and decision filters consistently to every view."""

    _populate_store(
        prediction_store
    )

    snapshot = (
        load_monitoring_snapshot(
            prediction_store,
            filters=(
                MonitoringFilters(
                    start_utc=(
                        BASE_TIME
                        + timedelta(
                            hours=2
                        )
                    ),
                    end_utc=(
                        BASE_TIME
                        + timedelta(
                            hours=3
                        )
                    ),
                    decisions=(
                        "REVIEW",
                    ),
                )
            ),
        )
    )

    assert (
        snapshot.overview.total_events
        == 1
    )
    assert (
        snapshot.overview.review_count
        == 1
    )
    assert (
        snapshot.overview.allow_count
        == 0
    )
    assert (
        snapshot.overview.block_count
        == 0
    )

    assert [
        item.transaction_id
        for item in (
            snapshot
            .score_observations
        )
    ] == [
        "tx-review-1",
    ]

    assert [
        item.transaction_id
        for item in (
            snapshot
            .recent_flagged_predictions
        )
    ] == [
        "tx-review-1",
    ]

    assert [
        item.reason_code
        for item in (
            snapshot
            .reason_code_frequencies
        )
    ] == [
        "RISK_A",
        "RISK_B",
    ]

    assert (
        snapshot.policy_provenance[
            0
        ].event_count
        == 1
    )


@pytest.mark.parametrize(
    "filters",
    [
        MonitoringFilters(
            start_utc=datetime(
                2026,
                8,
                8,
                8,
                0,
            )
        ),
        MonitoringFilters(
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
        MonitoringFilters(
            decisions=(
                "UNKNOWN",
            )
        ),
        MonitoringFilters(
            decisions=(
                "REVIEW",
                "REVIEW",
            )
        ),
    ],
)
def test_invalid_monitoring_filters_are_rejected(
    prediction_store: PredictionStore,
    filters: MonitoringFilters,
) -> None:
    """Reject ambiguous or invalid monitoring filter definitions."""

    with pytest.raises(
        ValueError
    ):
        load_monitoring_snapshot(
            prediction_store,
            filters=filters,
        )


def test_non_utc_filter_is_rejected(
    prediction_store: PredictionStore,
) -> None:
    """Require explicit UTC rather than another timezone offset."""

    non_utc = datetime(
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

    with pytest.raises(
        ValueError,
        match="start_utc must use UTC",
    ):
        load_monitoring_snapshot(
            prediction_store,
            filters=(
                MonitoringFilters(
                    start_utc=(
                        non_utc
                    )
                )
            ),
        )


@pytest.mark.parametrize(
    "recent_limit",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_recent_limit_is_rejected(
    prediction_store: PredictionStore,
    recent_limit: object,
) -> None:
    """Require a positive integer recent-event limit."""

    with pytest.raises(
        ValueError,
        match=(
            "recent_limit must be a positive integer"
        ),
    ):
        load_monitoring_snapshot(
            prediction_store,
            recent_limit=recent_limit,  # type: ignore[arg-type]
        )


def test_monitoring_snapshot_is_read_only(
    prediction_store: PredictionStore,
) -> None:
    """Do not mutate persisted decisions, labels, or provenance."""

    _populate_store(
        prediction_store
    )

    before = _database_state(
        prediction_store
    )

    load_monitoring_snapshot(
        prediction_store,
        recent_limit=3,
    )

    after = _database_state(
        prediction_store
    )

    assert after == before
