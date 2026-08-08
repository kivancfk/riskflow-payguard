"""Integration tests for Phase 6 API prediction persistence."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import api.main as api_main
from api import model_loader
from api.config import (
    FROZEN_BLOCK_THRESHOLD,
    FROZEN_CALIBRATION_METHOD,
    FROZEN_POLICY_SHA256,
    FROZEN_REVIEW_THRESHOLD,
)
from api.logging_db import (
    PredictionEvent,
    PredictionEventCreate,
    PredictionStore,
)
from src.data_processing import AMOUNT_COLUMN
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
)


@pytest.fixture(autouse=True)
def reset_process_policy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep persistence integration tests isolated from loader state."""

    monkeypatch.setattr(
        model_loader,
        "_loaded",
        None,
    )


def _valid_features(
    *,
    transaction_amount: float = 125.50,
) -> dict[
    str,
    object,
]:
    """Return one complete deterministic API feature payload."""

    categorical = set(
        CATEGORICAL_FEATURES
    )

    features: dict[
        str,
        object,
    ] = {
        feature_name: (
            None
            if feature_name
            in categorical
            else 0.0
        )
        for feature_name
        in FEATURE_COLUMNS
    }

    features[
        AMOUNT_COLUMN
    ] = transaction_amount

    return features


def _valid_payload(
    transaction_id: str | int,
    *,
    transaction_amount: float = 125.50,
) -> dict[
    str,
    object,
]:
    """Return one valid transaction request payload."""

    return {
        "transaction_id": (
            transaction_id
        ),
        "features": (
            _valid_features(
                transaction_amount=(
                    transaction_amount
                )
            )
        ),
    }


def _persisted_events(
    client: TestClient,
) -> list[
    PredictionEvent
]:
    """Read all events from the active test prediction store."""

    store = (
        client.app
        .state
        .prediction_store
    )

    with (
        store
        .session_factory()
    ) as session:
        return list(
            session.scalars(
                select(
                    PredictionEvent
                ).order_by(
                    PredictionEvent
                    .prediction_id
                )
            )
        )


def test_predict_persists_exact_successful_prediction() -> None:
    """Persist one successful response with frozen-policy provenance."""

    payload = _valid_payload(
        "tx-persist-single",
        transaction_amount=275.25,
    )

    with TestClient(
        api_main.app
    ) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

        events = _persisted_events(
            client
        )

    assert response.status_code == 200
    assert len(events) == 1

    body = response.json()
    event = events[0]

    assert (
        event.transaction_id
        == body["transaction_id"]
        == "tx-persist-single"
    )

    assert (
        event.raw_model_score
        == pytest.approx(
            body[
                "raw_model_score"
            ]
        )
    )

    assert (
        event.calibrated_probability
        == pytest.approx(
            body[
                "calibrated_probability"
            ]
        )
    )

    assert (
        event.decision
        == body["decision"]
    )

    assert (
        event.transaction_amount
        == pytest.approx(
            275.25
        )
    )

    assert (
        event.model_version
        == body["model_version"]
    )
    assert (
        event.policy_version
        == body["policy_version"]
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
        == body[
            "explanation_version"
        ]
    )
    assert (
        event.reason_code_version
        == body[
            "reason_code_version"
        ]
    )
    assert (
        event.reason_codes
        == body["reason_codes"]
    )

    assert (
        event.scored_at_utc.tzinfo
        is not None
    )

    assert event.actual_label is None
    assert (
        event.label_recorded_at_utc
        is None
    )


def test_batch_predict_persists_all_events_in_one_store_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist one successful batch using one atomic store operation."""

    calls: list[
        tuple[
            PredictionEventCreate,
            ...,
        ]
    ] = []

    original_add_events = (
        PredictionStore.add_events
    )

    def counting_add_events(
        self: PredictionStore,
        events: Sequence[
            PredictionEventCreate
        ],
    ) -> tuple[int, ...]:
        calls.append(
            tuple(events)
        )

        return original_add_events(
            self,
            events,
        )

    monkeypatch.setattr(
        PredictionStore,
        "add_events",
        counting_add_events,
    )

    payload = {
        "transactions": [
            _valid_payload(
                "tx-batch-1",
                transaction_amount=10.0,
            ),
            _valid_payload(
                "tx-batch-2",
                transaction_amount=20.0,
            ),
        ]
    }

    with TestClient(
        api_main.app
    ) as client:
        response = client.post(
            "/batch-predict",
            json=payload,
        )

        events = _persisted_events(
            client
        )

    assert response.status_code == 200

    body = response.json()

    assert len(
        body["predictions"]
    ) == 2

    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert len(events) == 2

    for (
        event,
        prediction,
        expected_amount,
    ) in zip(
        events,
        body["predictions"],
        (
            10.0,
            20.0,
        ),
    ):
        assert (
            event.transaction_id
            == prediction[
                "transaction_id"
            ]
        )

        assert (
            event.raw_model_score
            == pytest.approx(
                prediction[
                    "raw_model_score"
                ]
            )
        )

        assert (
            event.calibrated_probability
            == pytest.approx(
                prediction[
                    "calibrated_probability"
                ]
            )
        )

        assert (
            event.decision
            == prediction[
                "decision"
            ]
        )

        assert (
            event.transaction_amount
            == pytest.approx(
                expected_amount
            )
        )

    assert (
        events[0].scored_at_utc
        == events[1].scored_at_utc
    )


def test_validation_failure_persists_zero_events() -> None:
    """Do not create an event when FastAPI rejects the request."""

    payload = _valid_payload(
        "tx-invalid"
    )

    features = payload[
        "features"
    ]

    assert isinstance(
        features,
        dict,
    )

    del features[
        AMOUNT_COLUMN
    ]

    with TestClient(
        api_main.app
    ) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

        events = _persisted_events(
            client
        )

    assert response.status_code == 422
    assert events == []


def test_inference_failure_persists_zero_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not persist when frozen inference fails before event creation."""

    def failing_predict_transaction(
        *args: object,
        **kwargs: object,
    ) -> object:
        raise RuntimeError(
            "synthetic inference failure"
        )

    monkeypatch.setattr(
        api_main,
        "predict_transaction",
        failing_predict_transaction,
    )

    with TestClient(
        api_main.app,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/predict",
            json=_valid_payload(
                "tx-inference-failure"
            ),
        )

        events = _persisted_events(
            client
        )

    assert response.status_code == 500
    assert events == []


def test_persistence_failure_does_not_return_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not return HTTP 200 when the prediction event cannot be stored."""

    def failing_add_events(
        self: PredictionStore,
        events: Sequence[
            PredictionEventCreate
        ],
    ) -> tuple[int, ...]:
        raise RuntimeError(
            "synthetic persistence failure"
        )

    monkeypatch.setattr(
        PredictionStore,
        "add_events",
        failing_add_events,
    )

    with TestClient(
        api_main.app,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/predict",
            json=_valid_payload(
                "tx-persistence-failure"
            ),
        )

        events = _persisted_events(
            client
        )

    assert response.status_code == 500
    assert events == []
