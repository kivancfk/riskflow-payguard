"""Read-only monitoring queries for persisted Phase 6 prediction events."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone

from sqlalchemy import select

from api.logging_db import (
    PredictionEvent,
    PredictionStore,
    TransactionIdentifier,
)


VALID_DECISIONS = frozenset(
    {
        "ALLOW",
        "REVIEW",
        "BLOCK",
    }
)

INTERVENTION_DECISIONS = frozenset(
    {
        "REVIEW",
        "BLOCK",
    }
)


@dataclass(frozen=True)
class MonitoringFilters:
    """Optional read-only filters applied to persisted prediction events."""

    start_utc: datetime | None = None
    end_utc: datetime | None = None
    decisions: tuple[str, ...] = ()


@dataclass(frozen=True)
class MonitoringOverview:
    """Top-level operational metrics for persisted prediction events."""

    total_events: int

    allow_count: int
    review_count: int
    block_count: int

    allow_rate: float
    review_rate: float
    block_rate: float

    intervention_count: int
    intervention_rate: float

    labeled_count: int
    unlabeled_count: int
    label_coverage_rate: float

    average_calibrated_probability: float | None
    minimum_calibrated_probability: float | None
    maximum_calibrated_probability: float | None


@dataclass(frozen=True)
class ScoreObservation:
    """One persisted calibrated score for monitoring charts."""

    prediction_id: int
    transaction_id: TransactionIdentifier
    scored_at_utc: datetime
    calibrated_probability: float
    decision: str


@dataclass(frozen=True)
class ReasonCodeFrequency:
    """Event-level frequency for one persisted analyst reason code."""

    reason_code: str
    event_count: int
    event_rate: float


@dataclass(frozen=True)
class RecentFlaggedPrediction:
    """One recent persisted REVIEW or BLOCK prediction."""

    prediction_id: int
    transaction_id: TransactionIdentifier
    scored_at_utc: datetime
    calibrated_probability: float
    decision: str
    transaction_amount: float | None
    reason_codes: tuple[str, ...]
    actual_label: int | None


@dataclass(frozen=True)
class PolicyProvenance:
    """One distinct persisted frozen-policy contract."""

    model_version: str
    policy_version: str
    calibration_method: str
    review_threshold: float
    block_threshold: float
    policy_artifact_sha256: str
    explanation_version: str
    reason_code_version: str
    event_count: int


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Complete read-only monitoring dataset for one filter selection."""

    overview: MonitoringOverview
    score_observations: tuple[
        ScoreObservation,
        ...,
    ]
    reason_code_frequencies: tuple[
        ReasonCodeFrequency,
        ...,
    ]
    recent_flagged_predictions: tuple[
        RecentFlaggedPrediction,
        ...,
    ]
    policy_provenance: tuple[
        PolicyProvenance,
        ...,
    ]


def _validate_utc_datetime(
    value: datetime,
    *,
    field_name: str,
) -> None:
    """Require timezone-aware UTC filter timestamps."""

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
    filters: MonitoringFilters,
) -> None:
    """Validate monitoring filters before querying persistence."""

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
        and filters.start_utc > filters.end_utc
    ):
        raise ValueError(
            "start_utc must be earlier than or equal to end_utc"
        )

    invalid_decisions = sorted(
        set(filters.decisions)
        - VALID_DECISIONS
    )

    if invalid_decisions:
        raise ValueError(
            "Unsupported monitoring decisions: "
            + ", ".join(
                invalid_decisions
            )
        )

    if (
        len(filters.decisions)
        != len(
            set(filters.decisions)
        )
    ):
        raise ValueError(
            "Monitoring decisions must not contain duplicates"
        )


def _validate_recent_limit(
    recent_limit: int,
) -> None:
    """Require a positive integer recent-event limit."""

    if (
        isinstance(
            recent_limit,
            bool,
        )
        or not isinstance(
            recent_limit,
            int,
        )
        or recent_limit < 1
    ):
        raise ValueError(
            "recent_limit must be a positive integer"
        )


def _load_events(
    store: PredictionStore,
    filters: MonitoringFilters,
) -> list[
    PredictionEvent
]:
    """Load filtered persisted events in chronological order."""

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

    if filters.decisions:
        statement = statement.where(
            PredictionEvent.decision.in_(
                filters.decisions
            )
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


def _safe_rate(
    numerator: int,
    denominator: int,
) -> float:
    """Return a deterministic zero rate for an empty denominator."""

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def _build_overview(
    events: list[
        PredictionEvent
    ],
) -> MonitoringOverview:
    """Aggregate actual persisted policy outcomes."""

    total_events = len(
        events
    )

    decision_counts = Counter(
        event.decision
        for event in events
    )

    allow_count = decision_counts[
        "ALLOW"
    ]
    review_count = decision_counts[
        "REVIEW"
    ]
    block_count = decision_counts[
        "BLOCK"
    ]

    intervention_count = (
        review_count
        + block_count
    )

    labeled_count = sum(
        event.actual_label
        is not None
        for event in events
    )

    unlabeled_count = (
        total_events
        - labeled_count
    )

    probabilities = [
        event.calibrated_probability
        for event in events
    ]

    if probabilities:
        average_probability = (
            sum(
                probabilities
            )
            / total_events
        )
        minimum_probability = min(
            probabilities
        )
        maximum_probability = max(
            probabilities
        )

    else:
        average_probability = None
        minimum_probability = None
        maximum_probability = None

    return MonitoringOverview(
        total_events=total_events,
        allow_count=allow_count,
        review_count=review_count,
        block_count=block_count,
        allow_rate=_safe_rate(
            allow_count,
            total_events,
        ),
        review_rate=_safe_rate(
            review_count,
            total_events,
        ),
        block_rate=_safe_rate(
            block_count,
            total_events,
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
        labeled_count=labeled_count,
        unlabeled_count=(
            unlabeled_count
        ),
        label_coverage_rate=(
            _safe_rate(
                labeled_count,
                total_events,
            )
        ),
        average_calibrated_probability=(
            average_probability
        ),
        minimum_calibrated_probability=(
            minimum_probability
        ),
        maximum_calibrated_probability=(
            maximum_probability
        ),
    )


def _build_score_observations(
    events: list[
        PredictionEvent
    ],
) -> tuple[
    ScoreObservation,
    ...,
]:
    """Expose persisted calibrated scores without recomputing decisions."""

    return tuple(
        ScoreObservation(
            prediction_id=(
                event.prediction_id
            ),
            transaction_id=(
                event.transaction_id
            ),
            scored_at_utc=(
                event.scored_at_utc
            ),
            calibrated_probability=(
                event.calibrated_probability
            ),
            decision=(
                event.decision
            ),
        )
        for event in events
    )


def _build_reason_code_frequencies(
    events: list[
        PredictionEvent
    ],
) -> tuple[
    ReasonCodeFrequency,
    ...,
]:
    """Count how many filtered events contain each reason code."""

    counts: Counter[str] = (
        Counter()
    )

    for event in events:
        counts.update(
            set(
                event.reason_codes
            )
        )

    total_events = len(
        events
    )

    return tuple(
        ReasonCodeFrequency(
            reason_code=(
                reason_code
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
        for (
            reason_code,
            event_count,
        ) in sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    )


def _build_recent_flagged_predictions(
    events: list[
        PredictionEvent
    ],
    *,
    recent_limit: int,
) -> tuple[
    RecentFlaggedPrediction,
    ...,
]:
    """Return newest persisted REVIEW/BLOCK events."""

    flagged_events = [
        event
        for event in events
        if (
            event.decision
            in INTERVENTION_DECISIONS
        )
    ]

    flagged_events.sort(
        key=lambda event: (
            event.scored_at_utc,
            event.prediction_id,
        ),
        reverse=True,
    )

    return tuple(
        RecentFlaggedPrediction(
            prediction_id=(
                event.prediction_id
            ),
            transaction_id=(
                event.transaction_id
            ),
            scored_at_utc=(
                event.scored_at_utc
            ),
            calibrated_probability=(
                event.calibrated_probability
            ),
            decision=(
                event.decision
            ),
            transaction_amount=(
                event.transaction_amount
            ),
            reason_codes=tuple(
                event.reason_codes
            ),
            actual_label=(
                event.actual_label
            ),
        )
        for event in flagged_events[
            :recent_limit
        ]
    )


def _build_policy_provenance(
    events: list[
        PredictionEvent
    ],
) -> tuple[
    PolicyProvenance,
    ...,
]:
    """Summarize distinct persisted policy contracts."""

    counts: Counter[
        tuple[
            str,
            str,
            str,
            float,
            float,
            str,
            str,
            str,
        ]
    ] = Counter()

    for event in events:
        key = (
            event.model_version,
            event.policy_version,
            event.calibration_method,
            event.review_threshold,
            event.block_threshold,
            event.policy_artifact_sha256,
            event.explanation_version,
            event.reason_code_version,
        )

        counts[
            key
        ] += 1

    provenance = [
        PolicyProvenance(
            model_version=(
                key[0]
            ),
            policy_version=(
                key[1]
            ),
            calibration_method=(
                key[2]
            ),
            review_threshold=(
                key[3]
            ),
            block_threshold=(
                key[4]
            ),
            policy_artifact_sha256=(
                key[5]
            ),
            explanation_version=(
                key[6]
            ),
            reason_code_version=(
                key[7]
            ),
            event_count=(
                event_count
            ),
        )
        for (
            key,
            event_count,
        ) in counts.items()
    ]

    provenance.sort(
        key=lambda item: (
            item.model_version,
            item.policy_version,
            item.calibration_method,
            item.review_threshold,
            item.block_threshold,
            item.policy_artifact_sha256,
            item.explanation_version,
            item.reason_code_version,
        )
    )

    return tuple(
        provenance
    )


def load_monitoring_snapshot(
    store: PredictionStore,
    *,
    filters: MonitoringFilters | None = None,
    recent_limit: int = 20,
) -> MonitoringSnapshot:
    """Return one deterministic read-only monitoring snapshot."""

    resolved_filters = (
        MonitoringFilters()
        if filters is None
        else filters
    )

    _validate_filters(
        resolved_filters
    )
    _validate_recent_limit(
        recent_limit
    )

    events = _load_events(
        store,
        resolved_filters,
    )

    return MonitoringSnapshot(
        overview=(
            _build_overview(
                events
            )
        ),
        score_observations=(
            _build_score_observations(
                events
            )
        ),
        reason_code_frequencies=(
            _build_reason_code_frequencies(
                events
            )
        ),
        recent_flagged_predictions=(
            _build_recent_flagged_predictions(
                events,
                recent_limit=(
                    recent_limit
                ),
            )
        ),
        policy_provenance=(
            _build_policy_provenance(
                events
            )
        ),
    )
