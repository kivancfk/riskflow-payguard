"""Tests for Phase 6 append-only prediction persistence."""

from __future__ import annotations

import hashlib
from datetime import datetime
from datetime import timezone
from pathlib import Path

import pytest
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import StatementError

from api.config import (
    FROZEN_BASELINE_MODEL_VERSION,
    FROZEN_BLOCK_THRESHOLD,
    FROZEN_CALIBRATION_METHOD,
    FROZEN_EXPLANATION_VERSION,
    FROZEN_POLICY_ARTIFACT,
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


SCORED_AT_UTC = datetime(
    2026,
    8,
    8,
    11,
    30,
    tzinfo=timezone.utc,
)


@pytest.fixture
def prediction_store(
    tmp_path: Path,
) -> PredictionStore:
    """Create an isolated file-backed SQLite prediction store."""

    database_path = (
        tmp_path
        / "predictions.db"
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
    transaction_id: str | int = "txn-001",
    *,
    decision: str = "REVIEW",
    scored_at_utc: datetime = SCORED_AT_UTC,
) -> PredictionEventCreate:
    """Return one deterministic prediction-event payload."""

    return PredictionEventCreate(
        transaction_id=transaction_id,
        raw_model_score=0.42,
        calibrated_probability=0.31,
        decision=decision,
        transaction_amount=125.50,
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
            "TEST_REASON_1",
            "TEST_REASON_2",
        ),
        scored_at_utc=scored_at_utc,
    )


def _all_events(
    store: PredictionStore,
) -> list[PredictionEvent]:
    """Return all persisted prediction events in insertion order."""

    with store.session_factory() as session:
        return list(
            session.scalars(
                select(
                    PredictionEvent
                ).order_by(
                    PredictionEvent.prediction_id
                )
            )
        )


def _event_count(
    store: PredictionStore,
) -> int:
    """Return the number of persisted prediction events."""

    with store.session_factory() as session:
        return int(
            session.scalar(
                select(
                    func.count(
                        PredictionEvent.prediction_id
                    )
                )
            )
            or 0
        )


def _artifact_sha256() -> str:
    """Return the current frozen policy artifact SHA-256."""

    artifact_path = Path(
        FROZEN_POLICY_ARTIFACT
    ).resolve()

    return hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()


def test_prediction_event_round_trip_preserves_frozen_provenance(
    prediction_store: PredictionStore,
) -> None:
    """Persist one event without losing inference or provenance fields."""

    prediction_ids = (
        prediction_store.add_events(
            [
                _event(
                    transaction_id=12345
                )
            ]
        )
    )

    assert len(
        prediction_ids
    ) == 1

    events = _all_events(
        prediction_store
    )

    assert len(events) == 1

    event = events[0]

    assert (
        event.prediction_id
        == prediction_ids[0]
    )
    assert event.transaction_id == 12345

    assert event.raw_model_score == pytest.approx(
        0.42
    )
    assert (
        event.calibrated_probability
        == pytest.approx(
            0.31
        )
    )
    assert event.decision == "REVIEW"
    assert (
        event.transaction_amount
        == pytest.approx(
            125.50
        )
    )

    assert (
        event.model_version
        == FROZEN_BASELINE_MODEL_VERSION
    )
    assert (
        event.policy_version
        == FROZEN_POLICY_VERSION
    )
    assert (
        event.calibration_method
        == FROZEN_CALIBRATION_METHOD
    )

    assert (
        event.review_threshold
        == FROZEN_REVIEW_THRESHOLD
    )
    assert (
        event.block_threshold
        == FROZEN_BLOCK_THRESHOLD
    )

    assert (
        event.policy_artifact_sha256
        == FROZEN_POLICY_SHA256
    )
    assert (
        event.explanation_version
        == FROZEN_EXPLANATION_VERSION
    )
    assert (
        event.reason_code_version
        == FROZEN_REASON_CODE_VERSION
    )

    assert event.reason_codes == [
        "TEST_REASON_1",
        "TEST_REASON_2",
    ]

    assert (
        event.scored_at_utc
        == SCORED_AT_UTC
    )
    assert (
        event.scored_at_utc.tzinfo
        == timezone.utc
    )

    assert event.actual_label is None
    assert (
        event.label_recorded_at_utc
        is None
    )


def test_repeated_transaction_id_creates_distinct_events(
    prediction_store: PredictionStore,
) -> None:
    """Treat repeated scoring as separate immutable monitoring events."""

    prediction_ids = (
        prediction_store.add_events(
            [
                _event(
                    transaction_id=(
                        "same-transaction"
                    )
                ),
                _event(
                    transaction_id=(
                        "same-transaction"
                    )
                ),
            ]
        )
    )

    assert len(
        prediction_ids
    ) == 2
    assert (
        prediction_ids[0]
        != prediction_ids[1]
    )

    events = _all_events(
        prediction_store
    )

    assert len(events) == 2

    assert [
        event.transaction_id
        for event in events
    ] == [
        "same-transaction",
        "same-transaction",
    ]


def test_batch_insert_rolls_back_when_one_event_is_invalid(
    prediction_store: PredictionStore,
) -> None:
    """Do not partially persist a failed multi-event write."""

    with pytest.raises(
        IntegrityError
    ):
        prediction_store.add_events(
            [
                _event(
                    transaction_id="valid"
                ),
                _event(
                    transaction_id="invalid",
                    decision="INVALID",
                ),
            ]
        )

    assert (
        _event_count(
            prediction_store
        )
        == 0
    )


def test_prediction_store_rejects_naive_utc_timestamp(
    prediction_store: PredictionStore,
) -> None:
    """Require explicit timezone information for persisted event timestamps."""

    naive_timestamp = datetime(
        2026,
        8,
        8,
        11,
        30,
    )

    with pytest.raises(
        StatementError
    ) as error:
        prediction_store.add_events(
            [
                _event(
                    scored_at_utc=(
                        naive_timestamp
                    )
                )
            ]
        )

    assert (
        "must be timezone-aware"
        in str(error.value)
    )

    assert (
        _event_count(
            prediction_store
        )
        == 0
    )


def test_prediction_store_does_not_modify_frozen_policy_artifact(
    prediction_store: PredictionStore,
) -> None:
    """Keep the frozen policy bytes unchanged across persistence operations."""

    before_sha256 = (
        _artifact_sha256()
    )

    assert (
        before_sha256
        == FROZEN_POLICY_SHA256
    )

    prediction_store.add_events(
        [
            _event()
        ]
    )

    after_sha256 = (
        _artifact_sha256()
    )

    assert (
        after_sha256
        == before_sha256
    )
