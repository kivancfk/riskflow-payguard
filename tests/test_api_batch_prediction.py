"""Tests for ordered Phase 5 batch prediction integration."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import model_loader
from api.config import (
    FROZEN_POLICY_SHA256,
    settings,
)
from api.feature_frames import (
    build_batch_feature_frame,
)
from api.main import app
from api.prediction_service import (
    prediction_response_from_record,
)
from api.schemas import (
    BatchPredictRequest,
)
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
)
from src.policy_explanations import (
    predict_policy_with_explanations,
)


@pytest.fixture(autouse=True)
def reset_process_policy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep batch API tests isolated from process-level loader state."""

    monkeypatch.setattr(
        model_loader,
        "_loaded",
        None,
    )


def _valid_features(
    *,
    transaction_amount: float = 10.0,
    product_code: str | None = None,
) -> dict[str, object]:
    """Return one complete deterministic model feature payload."""

    categorical = set(
        CATEGORICAL_FEATURES
    )

    features: dict[
        str,
        object,
    ] = {
        feature_name: (
            None
            if feature_name in categorical
            else 0.0
        )
        for feature_name in FEATURE_COLUMNS
    }

    features[
        "TransactionAmt"
    ] = transaction_amount
    features[
        "ProductCD"
    ] = product_code

    return features


def _transaction_payload(
    transaction_id: str | int,
    *,
    transaction_amount: float = 10.0,
    product_code: str | None = None,
) -> dict[str, object]:
    """Return one complete batch transaction payload."""

    return {
        "transaction_id": (
            transaction_id
        ),
        "features": _valid_features(
            transaction_amount=(
                transaction_amount
            ),
            product_code=(
                product_code
            ),
        ),
    }


def _batch_payload() -> dict[
    str,
    object,
]:
    """Return three deliberately non-sorted transaction rows."""

    return {
        "transactions": [
            _transaction_payload(
                "tx-second",
                transaction_amount=20.0,
            ),
            _transaction_payload(
                "tx-first",
                transaction_amount=10.0,
            ),
            _transaction_payload(
                300,
                transaction_amount=30.0,
            ),
        ],
    }


def test_batch_predict_matches_direct_phase_4_inference() -> None:
    """Require HTTP batch output to equal direct aligned Phase 4 inference."""

    payload = _batch_payload()

    validated_request = (
        BatchPredictRequest
        .model_validate(
            payload
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/batch-predict",
            json=payload,
        )

        bundle = (
            model_loader.get_policy()
        )

        transaction_ids, feature_frame = (
            build_batch_feature_frame(
                validated_request.transactions,
                bundle,
            )
        )

        direct = (
            predict_policy_with_explanations(
                bundle,
                transaction_ids,
                feature_frame,
                frame_name=(
                    "direct batch API parity test"
                ),
            )
        )

    assert response.status_code == 200

    expected_predictions = [
        prediction_response_from_record(
            record
        ).model_dump(
            mode="json"
        )
        for record in direct.explanations
    ]

    assert response.json() == {
        "predictions": (
            expected_predictions
        ),
    }


def test_batch_predict_preserves_input_order() -> None:
    """Preserve transaction order rather than sorting by ID or risk."""

    payload = _batch_payload()

    with TestClient(app) as client:
        response = client.post(
            "/batch-predict",
            json=payload,
        )

    assert response.status_code == 200

    predictions = response.json()[
        "predictions"
    ]

    assert [
        prediction[
            "transaction_id"
        ]
        for prediction in predictions
    ] == [
        "tx-second",
        "tx-first",
        300,
    ]


def test_batch_predict_matches_equivalent_single_requests() -> None:
    """Require each batch row to equal its equivalent /predict response."""

    payload = _batch_payload()

    transactions = payload[
        "transactions"
    ]
    assert isinstance(
        transactions,
        list,
    )

    with TestClient(app) as client:
        batch_response = (
            client.post(
                "/batch-predict",
                json=payload,
            )
        )

        single_responses = [
            client.post(
                "/predict",
                json=transaction,
            )
            for transaction in transactions
        ]

    assert (
        batch_response.status_code
        == 200
    )

    for response in single_responses:
        assert (
            response.status_code
            == 200
        )

    batch_predictions = (
        batch_response.json()[
            "predictions"
        ]
    )

    assert batch_predictions == [
        response.json()
        for response in single_responses
    ]


def test_batch_predict_preserves_mixed_identifier_types() -> None:
    """Preserve string and integer identifiers without coercion."""

    payload = {
        "transactions": [
            _transaction_payload(
                "1001",
                transaction_amount=10.0,
            ),
            _transaction_payload(
                1001,
                transaction_amount=20.0,
            ),
        ],
    }

    with TestClient(app) as client:
        response = client.post(
            "/batch-predict",
            json=payload,
        )

    assert response.status_code == 200

    identifiers = [
        prediction[
            "transaction_id"
        ]
        for prediction in (
            response.json()[
                "predictions"
            ]
        )
    ]

    assert identifiers == [
        "1001",
        1001,
    ]


def test_batch_predict_rejects_empty_batch() -> None:
    """Return HTTP 422 for an empty request batch."""

    with TestClient(app) as client:
        response = client.post(
            "/batch-predict",
            json={
                "transactions": [],
            },
        )

    assert response.status_code == 422


def test_batch_predict_rejects_duplicate_identifiers() -> None:
    """Return HTTP 422 before inference when batch identifiers repeat."""

    payload = {
        "transactions": [
            _transaction_payload(
                "duplicate",
                transaction_amount=10.0,
            ),
            _transaction_payload(
                "duplicate",
                transaction_amount=20.0,
            ),
        ],
    }

    with TestClient(app) as client:
        response = client.post(
            "/batch-predict",
            json=payload,
        )

    assert response.status_code == 422

    assert (
        "unique within a batch"
        in response.text
    )


def test_batch_predict_rejects_invalid_nested_transaction() -> None:
    """Apply the same strict feature validation to every batch row."""

    valid = _transaction_payload(
        "valid"
    )
    invalid = _transaction_payload(
        "invalid"
    )

    invalid_features = invalid[
        "features"
    ]
    assert isinstance(
        invalid_features,
        dict,
    )

    del invalid_features[
        "TransactionAmt"
    ]

    payload = {
        "transactions": [
            valid,
            invalid,
        ],
    }

    with TestClient(app) as client:
        response = client.post(
            "/batch-predict",
            json=payload,
        )

    assert response.status_code == 422


def test_batch_predict_is_deterministic() -> None:
    """Return identical ordered JSON for repeated identical batch requests."""

    payload = _batch_payload()

    with TestClient(app) as client:
        first = client.post(
            "/batch-predict",
            json=payload,
        )
        second = client.post(
            "/batch-predict",
            json=payload,
        )

    assert first.status_code == 200
    assert second.status_code == 200

    assert (
        first.json()
        == second.json()
    )


def test_batch_predict_does_not_modify_frozen_artifact() -> None:
    """Keep the frozen policy artifact unchanged after batch inference."""

    artifact_path = (
        Path(
            settings.policy_path
        )
        .expanduser()
        .resolve()
    )

    before_sha256 = (
        model_loader
        .calculate_policy_sha256(
            artifact_path
        )
    )

    assert (
        before_sha256
        == FROZEN_POLICY_SHA256
    )

    with TestClient(app) as client:
        response = client.post(
            "/batch-predict",
            json=_batch_payload(),
        )

    assert response.status_code == 200

    after_sha256 = (
        model_loader
        .calculate_policy_sha256(
            artifact_path
        )
    )

    assert (
        after_sha256
        == before_sha256
        == FROZEN_POLICY_SHA256
    )
