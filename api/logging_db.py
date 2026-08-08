"""Prediction-event persistence for Phase 6 monitoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import TypeDecorator

from api.config import settings


TransactionIdentifier = str | int


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(
        timezone.utc
    )


class UTCDateTime(TypeDecorator):
    """Persist UTC datetimes and restore timezone awareness."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(
        self,
        value: datetime | None,
        dialect: object,
    ) -> datetime | None:
        """Normalize timezone-aware values to naive UTC for storage."""

        if value is None:
            return None

        if value.tzinfo is None:
            raise ValueError(
                "UTC datetime values must be timezone-aware"
            )

        return (
            value.astimezone(
                timezone.utc
            )
            .replace(
                tzinfo=None
            )
        )

    def process_result_value(
        self,
        value: datetime | None,
        dialect: object,
    ) -> datetime | None:
        """Restore persisted timestamps as timezone-aware UTC."""

        if value is None:
            return None

        if value.tzinfo is None:
            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )


class Base(DeclarativeBase):
    """Base class for Phase 6 persistence models."""


class PredictionEvent(Base):
    """One immutable scoring event produced by the frozen policy."""

    __tablename__ = "prediction_events"

    __table_args__ = (
        CheckConstraint(
            "raw_model_score >= 0.0 "
            "AND raw_model_score <= 1.0",
            name="ck_prediction_event_raw_score",
        ),
        CheckConstraint(
            "calibrated_probability >= 0.0 "
            "AND calibrated_probability <= 1.0",
            name="ck_prediction_event_calibrated_probability",
        ),
        CheckConstraint(
            "decision IN ('ALLOW', 'REVIEW', 'BLOCK')",
            name="ck_prediction_event_decision",
        ),
        CheckConstraint(
            "review_threshold >= 0.0 "
            "AND review_threshold <= 1.0",
            name="ck_prediction_event_review_threshold",
        ),
        CheckConstraint(
            "block_threshold >= 0.0 "
            "AND block_threshold <= 1.0",
            name="ck_prediction_event_block_threshold",
        ),
        CheckConstraint(
            "review_threshold <= block_threshold",
            name="ck_prediction_event_threshold_order",
        ),
        CheckConstraint(
            "transaction_amount IS NULL "
            "OR transaction_amount >= 0.0",
            name="ck_prediction_event_transaction_amount",
        ),
        CheckConstraint(
            "actual_label IS NULL "
            "OR actual_label IN (0, 1)",
            name="ck_prediction_event_actual_label",
        ),
    )

    prediction_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    transaction_id: Mapped[
        TransactionIdentifier
    ] = mapped_column(
        JSON,
        nullable=False,
    )

    raw_model_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    calibrated_probability: Mapped[
        float
    ] = mapped_column(
        Float,
        nullable=False,
    )
    decision: Mapped[str] = mapped_column(
        String(6),
        nullable=False,
    )

    transaction_amount: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    model_version: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    policy_version: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )
    calibration_method: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    review_threshold: Mapped[
        float
    ] = mapped_column(
        Float,
        nullable=False,
    )
    block_threshold: Mapped[
        float
    ] = mapped_column(
        Float,
        nullable=False,
    )

    policy_artifact_sha256: Mapped[
        str
    ] = mapped_column(
        String(64),
        nullable=False,
    )

    explanation_version: Mapped[
        str
    ] = mapped_column(
        String,
        nullable=False,
    )
    reason_code_version: Mapped[
        str
    ] = mapped_column(
        String,
        nullable=False,
    )

    reason_codes: Mapped[
        list[str]
    ] = mapped_column(
        JSON,
        nullable=False,
    )

    scored_at_utc: Mapped[
        datetime
    ] = mapped_column(
        UTCDateTime(),
        nullable=False,
    )

    actual_label: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )
    label_recorded_at_utc: Mapped[
        datetime | None
    ] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )


@dataclass(frozen=True)
class PredictionEventCreate:
    """Values required to append one prediction event."""

    transaction_id: TransactionIdentifier

    raw_model_score: float
    calibrated_probability: float
    decision: str

    transaction_amount: float | None

    model_version: str
    policy_version: str
    calibration_method: str

    review_threshold: float
    block_threshold: float

    policy_artifact_sha256: str

    explanation_version: str
    reason_code_version: str

    reason_codes: tuple[str, ...]

    scored_at_utc: datetime = field(
        default_factory=_utc_now
    )


def _create_engine(
    database_url: str,
) -> Engine:
    """Create a SQLAlchemy engine suitable for API and test use."""

    if not isinstance(
        database_url,
        str,
    ):
        raise TypeError(
            "database_url must be a string"
        )

    if not database_url.strip():
        raise ValueError(
            "database_url must not be empty"
        )

    engine_kwargs: dict[
        str,
        object,
    ] = {
        "future": True,
    }

    if database_url.startswith(
        "sqlite"
    ):
        engine_kwargs[
            "connect_args"
        ] = {
            "check_same_thread": False,
        }

        if database_url in {
            "sqlite://",
            "sqlite:///:memory:",
        }:
            engine_kwargs[
                "poolclass"
            ] = StaticPool

    return create_engine(
        database_url,
        **engine_kwargs,
    )


class PredictionStore:
    """Append-only persistence boundary for prediction events."""

    def __init__(
        self,
        database_url: str,
    ) -> None:
        self.engine = _create_engine(
            database_url
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )

    def init_schema(
        self,
    ) -> None:
        """Create Phase 6 tables when they do not already exist."""

        Base.metadata.create_all(
            self.engine
        )

    def add_events(
        self,
        events: Sequence[
            PredictionEventCreate
        ],
    ) -> tuple[int, ...]:
        """Atomically append one or more prediction events."""

        if not events:
            raise ValueError(
                "events must contain at least one prediction event"
            )

        rows = [
            PredictionEvent(
                transaction_id=(
                    event.transaction_id
                ),
                raw_model_score=(
                    event.raw_model_score
                ),
                calibrated_probability=(
                    event.calibrated_probability
                ),
                decision=event.decision,
                transaction_amount=(
                    event.transaction_amount
                ),
                model_version=(
                    event.model_version
                ),
                policy_version=(
                    event.policy_version
                ),
                calibration_method=(
                    event.calibration_method
                ),
                review_threshold=(
                    event.review_threshold
                ),
                block_threshold=(
                    event.block_threshold
                ),
                policy_artifact_sha256=(
                    event.policy_artifact_sha256
                ),
                explanation_version=(
                    event.explanation_version
                ),
                reason_code_version=(
                    event.reason_code_version
                ),
                reason_codes=list(
                    event.reason_codes
                ),
                scored_at_utc=(
                    event.scored_at_utc
                ),
                actual_label=None,
                label_recorded_at_utc=None,
            )
            for event in events
        ]

        with self.session_factory() as session:
            try:
                session.add_all(
                    rows
                )
                session.flush()

                prediction_ids = tuple(
                    row.prediction_id
                    for row in rows
                )

                if any(
                    prediction_id is None
                    for prediction_id
                    in prediction_ids
                ):
                    raise RuntimeError(
                        "Prediction event insert did not return "
                        "all generated identifiers"
                    )

                session.commit()

            except Exception:
                session.rollback()
                raise

        return prediction_ids

    def dispose(
        self,
    ) -> None:
        """Release database-engine resources."""

        self.engine.dispose()


def create_prediction_store(
    database_url: str | None = None,
) -> PredictionStore:
    """Create a prediction store using explicit or configured database URL."""

    resolved_database_url = (
        settings.database_url
        if database_url is None
        else database_url
    )

    return PredictionStore(
        resolved_database_url
    )
