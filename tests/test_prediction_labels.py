"""Tests for Phase 6 ground-truth prediction labeling."""

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


SCORED_AT_UTC = datetime(
    2026,
    8,
    8,
    10,
    0,
    tzinfo=timezone.utc,
)

LABEL_TIME_1 = datetime(
    2026,
    8,
    8,
    12,
    0,
    tzinfo=timezone.utc,
)

LABEL_TIME_2 = datetime(
    2026,
    8,
    8,
    13,
    0,
    tzinfo=timezone.utc,
)


@pytest.fixture
def prediction_store(
    tmp_path: Path,
) -> PredictionStore:
    """Create one isolated SQLite prediction store."""

    database_path = (
        tmp_path
        / "prediction-labels.db"
    )

    store = PredictionStore(
        f"sqlite:///{database_path}"
    )
    store.init_schema()

    try:
        yield store
    finally:
        store.dispose()


def _prediction_event(
    transaction_id: str | int,
) -> PredictionEventCreate:
    """Return one deterministic persisted prediction payload."""

    return PredictionEventCreate(
        transaction_id=transaction_id,
        raw_model_score=0.42,
        calibrated_probability=0.31,
        decision="REVIEW",
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
            "TEST_REASON",
        ),
        scored_at_utc=(
            SCORED_AT_UTC
        ),
    )


def _all_events(
    store: PredictionStore,
) -> list[
    PredictionEvent
]:
    """Return persisted events in insertion order."""

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


def test_label_backfill_updates_all_repeated_transaction_events(
    prediction_store: PredictionStore,
) -> None:
    """Apply one transaction outcome to every scoring event for it."""

    prediction_store.add_events(
        [
            _prediction_event(
                "tx-repeat"
            ),
            _prediction_event(
                "tx-repeat"
            ),
        ]
    )

    results = record_prediction_labels(
        prediction_store,
        [
            PredictionLabelUpdate(
                transaction_id=(
                    "tx-repeat"
                ),
                actual_label=1,
                recorded_at_utc=(
                    LABEL_TIME_1
                ),
            )
        ],
    )

    assert len(results) == 1

    result = results[0]

    assert (
        result.transaction_id
        == "tx-repeat"
    )
    assert result.actual_label == 1
    assert result.matched_events == 2
    assert result.updated_events == 2

    events = _all_events(
        prediction_store
    )

    assert [
        event.actual_label
        for event in events
    ] == [
        1,
        1,
    ]

    assert [
        event.label_recorded_at_utc
        for event in events
    ] == [
        LABEL_TIME_1,
        LABEL_TIME_1,
    ]


def test_same_label_replay_is_idempotent(
    prediction_store: PredictionStore,
) -> None:
    """Do not rewrite timestamps when the same outcome is replayed."""

    prediction_store.add_events(
        [
            _prediction_event(
                "tx-idempotent"
            )
        ]
    )

    first_result = (
        record_prediction_labels(
            prediction_store,
            [
                PredictionLabelUpdate(
                    transaction_id=(
                        "tx-idempotent"
                    ),
                    actual_label=0,
                    recorded_at_utc=(
                        LABEL_TIME_1
                    ),
                )
            ],
        )[0]
    )

    second_result = (
        record_prediction_labels(
            prediction_store,
            [
                PredictionLabelUpdate(
                    transaction_id=(
                        "tx-idempotent"
                    ),
                    actual_label=0,
                    recorded_at_utc=(
                        LABEL_TIME_2
                    ),
                )
            ],
        )[0]
    )

    assert (
        first_result.updated_events
        == 1
    )
    assert (
        second_result.matched_events
        == 1
    )
    assert (
        second_result.updated_events
        == 0
    )

    event = _all_events(
        prediction_store
    )[0]

    assert event.actual_label == 0
    assert (
        event.label_recorded_at_utc
        == LABEL_TIME_1
    )


def test_conflicting_relabel_is_rejected(
    prediction_store: PredictionStore,
) -> None:
    """Refuse to overwrite an established ground-truth outcome."""

    prediction_store.add_events(
        [
            _prediction_event(
                "tx-conflict"
            )
        ]
    )

    record_prediction_labels(
        prediction_store,
        [
            PredictionLabelUpdate(
                transaction_id=(
                    "tx-conflict"
                ),
                actual_label=1,
                recorded_at_utc=(
                    LABEL_TIME_1
                ),
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting ground-truth label"
        ),
    ):
        record_prediction_labels(
            prediction_store,
            [
                PredictionLabelUpdate(
                    transaction_id=(
                        "tx-conflict"
                    ),
                    actual_label=0,
                    recorded_at_utc=(
                        LABEL_TIME_2
                    ),
                )
            ],
        )

    event = _all_events(
        prediction_store
    )[0]

    assert event.actual_label == 1
    assert (
        event.label_recorded_at_utc
        == LABEL_TIME_1
    )


def test_unknown_transaction_is_rejected(
    prediction_store: PredictionStore,
) -> None:
    """Require every label to match at least one persisted event."""

    with pytest.raises(
        LookupError,
        match=(
            "No persisted prediction events found"
        ),
    ):
        record_prediction_labels(
            prediction_store,
            [
                PredictionLabelUpdate(
                    transaction_id=(
                        "tx-missing"
                    ),
                    actual_label=1,
                    recorded_at_utc=(
                        LABEL_TIME_1
                    ),
                )
            ],
        )


def test_label_batch_is_atomic_on_later_failure(
    prediction_store: PredictionStore,
) -> None:
    """Roll back earlier updates if any label in the batch fails."""

    prediction_store.add_events(
        [
            _prediction_event(
                "tx-first"
            ),
            _prediction_event(
                "tx-conflicting"
            ),
        ]
    )

    record_prediction_labels(
        prediction_store,
        [
            PredictionLabelUpdate(
                transaction_id=(
                    "tx-conflicting"
                ),
                actual_label=1,
                recorded_at_utc=(
                    LABEL_TIME_1
                ),
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Conflicting ground-truth label"
        ),
    ):
        record_prediction_labels(
            prediction_store,
            [
                PredictionLabelUpdate(
                    transaction_id=(
                        "tx-first"
                    ),
                    actual_label=0,
                    recorded_at_utc=(
                        LABEL_TIME_2
                    ),
                ),
                PredictionLabelUpdate(
                    transaction_id=(
                        "tx-conflicting"
                    ),
                    actual_label=0,
                    recorded_at_utc=(
                        LABEL_TIME_2
                    ),
                ),
            ],
        )

    events = _all_events(
        prediction_store
    )

    first_event = events[0]
    conflicting_event = (
        events[1]
    )

    assert (
        first_event.actual_label
        is None
    )
    assert (
        first_event
        .label_recorded_at_utc
        is None
    )

    assert (
        conflicting_event.actual_label
        == 1
    )
    assert (
        conflicting_event
        .label_recorded_at_utc
        == LABEL_TIME_1
    )


@pytest.mark.parametrize(
    "invalid_label",
    [
        -1,
        2,
        True,
        1.0,
    ],
)
def test_invalid_binary_labels_are_rejected(
    prediction_store: PredictionStore,
    invalid_label: object,
) -> None:
    """Accept only exact integer binary ground-truth labels."""

    prediction_store.add_events(
        [
            _prediction_event(
                "tx-invalid-label"
            )
        ]
    )

    update = PredictionLabelUpdate(
        transaction_id=(
            "tx-invalid-label"
        ),
        actual_label=invalid_label,  # type: ignore[arg-type]
        recorded_at_utc=(
            LABEL_TIME_1
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "actual_label must be integer 0 or 1"
        ),
    ):
        record_prediction_labels(
            prediction_store,
            [
                update
            ],
        )

    event = _all_events(
        prediction_store
    )[0]

    assert (
        event.actual_label
        is None
    )


def test_non_utc_label_timestamp_is_rejected(
    prediction_store: PredictionStore,
) -> None:
    """Require ground-truth recording timestamps to use UTC."""

    prediction_store.add_events(
        [
            _prediction_event(
                "tx-timezone"
            )
        ]
    )

    non_utc = datetime(
        2026,
        8,
        8,
        15,
        0,
        tzinfo=timezone(
            timedelta(
                hours=3
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "recorded_at_utc must use UTC"
        ),
    ):
        record_prediction_labels(
            prediction_store,
            [
                PredictionLabelUpdate(
                    transaction_id=(
                        "tx-timezone"
                    ),
                    actual_label=1,
                    recorded_at_utc=(
                        non_utc
                    ),
                )
            ],
        )


def test_integer_and_string_transaction_ids_remain_distinct(
    prediction_store: PredictionStore,
) -> None:
    """Do not conflate numerically similar string and integer identifiers."""

    prediction_store.add_events(
        [
            _prediction_event(
                42
            ),
            _prediction_event(
                "42"
            ),
        ]
    )

    integer_result = (
        record_prediction_labels(
            prediction_store,
            [
                PredictionLabelUpdate(
                    transaction_id=42,
                    actual_label=1,
                    recorded_at_utc=(
                        LABEL_TIME_1
                    ),
                )
            ],
        )[0]
    )

    assert (
        integer_result.matched_events
        == 1
    )
    assert (
        integer_result.updated_events
        == 1
    )

    events = _all_events(
        prediction_store
    )

    assert (
        events[0].transaction_id
        == 42
    )
    assert (
        events[0].actual_label
        == 1
    )

    assert (
        events[1].transaction_id
        == "42"
    )
    assert (
        events[1].actual_label
        is None
    )

    string_result = (
        record_prediction_labels(
            prediction_store,
            [
                PredictionLabelUpdate(
                    transaction_id="42",
                    actual_label=0,
                    recorded_at_utc=(
                        LABEL_TIME_2
                    ),
                )
            ],
        )[0]
    )

    assert (
        string_result.matched_events
        == 1
    )
    assert (
        string_result.updated_events
        == 1
    )

    events = _all_events(
        prediction_store
    )

    assert (
        events[0].actual_label
        == 1
    )
    assert (
        events[1].actual_label
        == 0
    )
