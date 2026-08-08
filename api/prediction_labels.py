"""Ground-truth label backfill for persisted prediction events."""

from __future__ import annotations

import json

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone

from sqlalchemy import Text
from sqlalchemy import cast
from sqlalchemy import select
from sqlalchemy.sql.elements import ColumnElement

from api.logging_db import (
    PredictionEvent,
    PredictionStore,
    TransactionIdentifier,
)


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(
        timezone.utc
    )


@dataclass(frozen=True)
class PredictionLabelUpdate:
    """One transaction-level ground-truth label update."""

    transaction_id: TransactionIdentifier
    actual_label: int
    recorded_at_utc: datetime = field(
        default_factory=_utc_now
    )


@dataclass(frozen=True)
class PredictionLabelResult:
    """Summary of one transaction-level label backfill."""

    transaction_id: TransactionIdentifier
    actual_label: int
    matched_events: int
    updated_events: int


def _validate_transaction_id(
    transaction_id: object,
) -> None:
    """Require the same identifier shape accepted by the public API."""

    if isinstance(
        transaction_id,
        bool,
    ):
        raise ValueError(
            "transaction_id must be a string or integer"
        )

    if isinstance(
        transaction_id,
        str,
    ):
        if (
            not transaction_id
            or transaction_id
            != transaction_id.strip()
        ):
            raise ValueError(
                "transaction_id must be a non-empty string "
                "without surrounding whitespace"
            )

        return

    if isinstance(
        transaction_id,
        int,
    ):
        return

    raise ValueError(
        "transaction_id must be a string or integer"
    )


def _validate_actual_label(
    actual_label: object,
) -> None:
    """Require an exact binary integer label."""

    if (
        isinstance(
            actual_label,
            bool,
        )
        or not isinstance(
            actual_label,
            int,
        )
        or actual_label
        not in {
            0,
            1,
        }
    ):
        raise ValueError(
            "actual_label must be integer 0 or 1"
        )


def _validate_recorded_at_utc(
    recorded_at_utc: object,
) -> None:
    """Require a timezone-aware UTC label timestamp."""

    if not isinstance(
        recorded_at_utc,
        datetime,
    ):
        raise ValueError(
            "recorded_at_utc must be a datetime"
        )

    if (
        recorded_at_utc.tzinfo
        is None
        or recorded_at_utc.utcoffset()
        is None
    ):
        raise ValueError(
            "recorded_at_utc must be timezone-aware"
        )

    if (
        recorded_at_utc.utcoffset()
        != timezone.utc.utcoffset(
            recorded_at_utc
        )
    ):
        raise ValueError(
            "recorded_at_utc must use UTC"
        )


def _validate_updates(
    updates: Sequence[
        PredictionLabelUpdate
    ],
) -> None:
    """Validate one non-empty unambiguous label batch."""

    if not updates:
        raise ValueError(
            "updates must contain at least one label update"
        )

    seen_transaction_ids: set[
        tuple[
            type,
            TransactionIdentifier,
        ]
    ] = set()

    for update in updates:
        _validate_transaction_id(
            update.transaction_id
        )
        _validate_actual_label(
            update.actual_label
        )
        _validate_recorded_at_utc(
            update.recorded_at_utc
        )

        transaction_key = (
            type(
                update.transaction_id
            ),
            update.transaction_id,
        )

        if (
            transaction_key
            in seen_transaction_ids
        ):
            raise ValueError(
                "Each transaction_id may appear only once "
                "in one label update batch"
            )

        seen_transaction_ids.add(
            transaction_key
        )


def _transaction_id_predicate(
    transaction_id: TransactionIdentifier,
) -> ColumnElement[bool]:
    """Match the exact typed scalar stored in the JSON identifier column."""

    serialized_transaction_id = (
        json.dumps(
            transaction_id,
            separators=(
                ",",
                ":",
            ),
        )
    )

    return (
        cast(
            PredictionEvent.transaction_id,
            Text,
        )
        == serialized_transaction_id
    )

def record_prediction_labels(
    store: PredictionStore,
    updates: Sequence[
        PredictionLabelUpdate
    ],
) -> tuple[
    PredictionLabelResult,
    ...,
]:
    """Atomically backfill ground truth for persisted prediction events."""

    _validate_updates(
        updates
    )

    results: list[
        PredictionLabelResult
    ] = []

    with store.session_factory() as session:
        try:
            for update in updates:
                events = list(
                    session.scalars(
                        select(
                            PredictionEvent
                        )
                        .where(
                            _transaction_id_predicate(
                                update.transaction_id
                            )
                        )
                        .order_by(
                            PredictionEvent.prediction_id
                        )
                    )
                )

                if not events:
                    raise LookupError(
                        "No persisted prediction events found for "
                        f"transaction_id={update.transaction_id!r}"
                    )

                conflicting_events = [
                    event
                    for event in events
                    if (
                        event.actual_label
                        is not None
                        and event.actual_label
                        != update.actual_label
                    )
                ]

                if conflicting_events:
                    raise ValueError(
                        "Conflicting ground-truth label for "
                        f"transaction_id={update.transaction_id!r}"
                    )

                updated_events = 0

                for event in events:
                    if (
                        event.actual_label
                        is not None
                    ):
                        continue

                    event.actual_label = (
                        update.actual_label
                    )
                    event.label_recorded_at_utc = (
                        update.recorded_at_utc
                    )

                    updated_events += 1

                results.append(
                    PredictionLabelResult(
                        transaction_id=(
                            update.transaction_id
                        ),
                        actual_label=(
                            update.actual_label
                        ),
                        matched_events=len(
                            events
                        ),
                        updated_events=(
                            updated_events
                        ),
                    )
                )

            session.commit()

        except Exception:
            session.rollback()
            raise

    return tuple(
        results
    )
