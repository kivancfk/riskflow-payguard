"""Streamlit smoke tests for the Phase 6 monitoring view."""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone

from sqlalchemy import select
from streamlit.testing.v1 import (
    AppTest,
)

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


BASE_TIME = datetime(
    2026,
    8,
    8,
    8,
    0,
    tzinfo=timezone.utc,
)


def _event(
    transaction_id: str,
    *,
    probability: float,
    decision: str,
    amount: float,
    reason_codes: tuple[
        str,
        ...,
    ],
    hours_after_base: int,
) -> PredictionEventCreate:
    """Return one dashboard smoke-test event."""

    return PredictionEventCreate(
        transaction_id=transaction_id,
        raw_model_score=probability,
        calibrated_probability=(
            probability
        ),
        decision=decision,
        transaction_amount=amount,
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
        reason_codes=reason_codes,
        scored_at_utc=(
            BASE_TIME
            + timedelta(
                hours=hours_after_base
            )
        ),
    )


def _seed_monitoring_data(
    database_url: str,
) -> None:
    """Populate the isolated runtime database with monitoring events."""

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
                    decision="ALLOW",
                    amount=10.0,
                    reason_codes=(),
                    hours_after_base=1,
                ),
                _event(
                    "tx-review",
                    probability=0.30,
                    decision="REVIEW",
                    amount=20.0,
                    reason_codes=(
                        "RISK_A",
                    ),
                    hours_after_base=2,
                ),
                _event(
                    "tx-block",
                    probability=0.90,
                    decision="BLOCK",
                    amount=30.0,
                    reason_codes=(
                        "RISK_A",
                        "RISK_B",
                    ),
                    hours_after_base=3,
                ),
            ]
        )

        record_prediction_labels(
            store,
            [
                PredictionLabelUpdate(
                    transaction_id=(
                        "tx-block"
                    ),
                    actual_label=1,
                    recorded_at_utc=(
                        BASE_TIME
                        + timedelta(
                            hours=4
                        )
                    ),
                )
            ],
        )

    finally:
        store.dispose()


def _metric_values(
    app_test: AppTest,
) -> dict[
    str,
    str,
]:
    """Return rendered Streamlit metric values by label."""

    return {
        metric.label: metric.value
        for metric
        in app_test.metric
    }


def test_empty_monitoring_dashboard_smoke() -> None:
    """Render the monitoring view successfully with an empty database."""

    app_test = (
        AppTest.from_file(
            "dashboard/app.py"
        )
        .run(
            timeout=10
        )
    )

    assert (
        len(
            app_test.exception
        )
        == 0
    )

    assert (
        app_test.title[
            0
        ].value
        == "RiskFlow PayGuard"
    )

    assert (
        app_test.header[
            0
        ].value
        == "Monitoring"
    )

    assert any(
        (
            "No prediction events "
            "have been persisted yet"
        )
        in info.value
        for info
        in app_test.info
    )

    metrics = _metric_values(
        app_test
    )

    assert (
        metrics[
            "Prediction events"
        ]
        == "0"
    )
    assert (
        metrics[
            "Intervention rate"
        ]
        == "0.0%"
    )
    assert (
        metrics[
            "Label coverage"
        ]
        == "0.0%"
    )
    assert (
        metrics[
            "Average calibrated probability"
        ]
        == "—"
    )


def test_populated_monitoring_dashboard_smoke(
    runtime_database_url: str,
) -> None:
    """Render actual persisted decisions and label coverage."""

    _seed_monitoring_data(
        runtime_database_url
    )

    app_test = (
        AppTest.from_file(
            "dashboard/app.py"
        )
        .run(
            timeout=10
        )
    )

    assert (
        len(
            app_test.exception
        )
        == 0
    )

    metrics = _metric_values(
        app_test
    )

    assert (
        metrics[
            "Prediction events"
        ]
        == "3"
    )
    assert (
        metrics[
            "Intervention rate"
        ]
        == "66.7%"
    )
    assert (
        metrics[
            "Label coverage"
        ]
        == "33.3%"
    )
    assert (
        metrics[
            "Average calibrated probability"
        ]
        == "0.417"
    )

    assert (
        metrics[
            "Labeled events"
        ]
        == "1"
    )
    assert (
        metrics[
            "Unlabeled events"
        ]
        == "2"
    )
    assert (
        metrics[
            "Coverage"
        ]
        == "33.3%"
    )


def test_monitoring_dashboard_is_read_only(
    runtime_database_url: str,
) -> None:
    """Rendering the Streamlit view must not modify persisted events."""

    _seed_monitoring_data(
        runtime_database_url
    )

    store = PredictionStore(
        runtime_database_url
    )
    store.init_schema()

    try:
        with (
            store.session_factory()
        ) as session:
            before = [
                (
                    event.prediction_id,
                    event.transaction_id,
                    event.decision,
                    event.actual_label,
                    event.label_recorded_at_utc,
                )
                for event
                in session.scalars(
                    select(
                        PredictionEvent
                    ).order_by(
                        PredictionEvent
                        .prediction_id
                    )
                )
            ]

        app_test = (
            AppTest.from_file(
                "dashboard/app.py"
            )
            .run(
            timeout=10
        )
        )

        assert (
            len(
                app_test.exception
            )
            == 0
        )

        with (
            store.session_factory()
        ) as session:
            after = [
                (
                    event.prediction_id,
                    event.transaction_id,
                    event.decision,
                    event.actual_label,
                    event.label_recorded_at_utc,
                )
                for event
                in session.scalars(
                    select(
                        PredictionEvent
                    ).order_by(
                        PredictionEvent
                        .prediction_id
                    )
                )
            ]

    finally:
        store.dispose()

    assert after == before
