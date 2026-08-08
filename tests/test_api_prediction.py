"""Tests for the Phase 5 single-transaction prediction endpoint."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import model_loader
from api.config import (
    FROZEN_POLICY_SHA256,
    settings,
)
from api.feature_frames import (
    build_feature_frame,
)
from api.main import app
from api.schemas import (
    TransactionRequest,
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
    """Keep API prediction tests isolated from process-level loader state."""

    monkeypatch.setattr(
        model_loader,
        "_loaded",
        None,
    )


def _valid_features() -> dict[
    str,
    object,
]:
    """Return one complete deterministic API feature payload."""

    categorical = set(
        CATEGORICAL_FEATURES
    )

    return {
        feature_name: (
            None
            if feature_name in categorical
            else 0.0
        )
        for feature_name in FEATURE_COLUMNS
    }


def _valid_payload(
    transaction_id: str | int = "tx-001",
) -> dict[str, object]:
    """Return one complete API prediction request."""

    return {
        "transaction_id": (
            transaction_id
        ),
        "features": _valid_features(),
    }


def test_predict_matches_direct_phase_4_inference() -> None:
    """Require API scoring and explanations to match direct inference exactly."""

    payload = _valid_payload(
        "tx-parity"
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

        bundle = (
            model_loader.get_policy()
        )

        transaction = (
            TransactionRequest
            .model_validate(
                payload
            )
        )

        feature_frame = (
            build_feature_frame(
                transaction,
                bundle,
            )
        )

        direct = (
            predict_policy_with_explanations(
                bundle,
                [
                    transaction
                    .transaction_id,
                ],
                feature_frame,
                frame_name=(
                    "direct API parity test"
                ),
            )
        )

    assert response.status_code == 200

    record = direct.explanations[0]
    body = response.json()

    assert (
        body["transaction_id"]
        == record.transaction_id
    )

    assert (
        body["model_version"]
        == record.model_version
    )
    assert (
        body["policy_version"]
        == record.policy_version
    )
    assert (
        body["explanation_version"]
        == record.explanation_version
    )
    assert (
        body["reason_code_version"]
        == record.reason_code_version
    )

    assert (
        body["raw_model_score"]
        == record.raw_model_score
    )
    assert (
        body[
            "calibrated_probability"
        ]
        == record.calibrated_probability
    )
    assert (
        body["decision"]
        == record.decision
    )

    assert (
        body[
            "top_positive_contributions"
        ]
        == [
            asdict(
                contribution
            )
            for contribution in (
                record
                .top_positive_contributions
            )
        ]
    )

    assert (
        body[
            "top_negative_contributions"
        ]
        == [
            asdict(
                contribution
            )
            for contribution in (
                record
                .top_negative_contributions
            )
        ]
    )

    assert (
        body["reason_codes"]
        == list(
            record.reason_codes
        )
    )

    assert (
        body["reasons"]
        == [
            {
                "code": reason.code,
                "message": reason.message,
            }
            for reason in record.reasons
        ]
    )

    assert body[
        "reconstruction"
    ] == {
        "raw_model_margin": (
            record.raw_model_margin
        ),
        "expected_value_raw": (
            record.expected_value_raw
        ),
        "shap_sum_raw": (
            record.shap_sum_raw
        ),
        "reconstructed_raw_margin": (
            record
            .reconstructed_raw_margin
        ),
        "reconstructed_raw_model_score": (
            record
            .reconstructed_raw_model_score
        ),
        "margin_reconstruction_error": (
            record
            .margin_reconstruction_error
        ),
        "score_reconstruction_error": (
            record
            .score_reconstruction_error
        ),
    }


def test_predict_preserves_integer_transaction_id() -> None:
    """Preserve integer transaction identifiers through HTTP serialization."""

    payload = _valid_payload(
        9001
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 200
    assert (
        response.json()[
            "transaction_id"
        ]
        == 9001
    )


def test_predict_is_deterministic_for_repeated_request() -> None:
    """Return identical model-facing JSON for repeated identical requests."""

    payload = _valid_payload(
        "tx-deterministic"
    )

    with TestClient(app) as client:
        first = client.post(
            "/predict",
            json=payload,
        )
        second = client.post(
            "/predict",
            json=payload,
        )

    assert first.status_code == 200
    assert second.status_code == 200

    assert (
        first.json()
        == second.json()
    )


def test_predict_rejects_absent_feature() -> None:
    """Return HTTP 422 when a required frozen feature key is absent."""

    payload = _valid_payload()

    features = payload[
        "features"
    ]
    assert isinstance(
        features,
        dict,
    )

    del features[
        "TransactionAmt"
    ]

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 422


def test_predict_rejects_extra_feature() -> None:
    """Return HTTP 422 for model features outside the frozen contract."""

    payload = _valid_payload()

    features = payload[
        "features"
    ]
    assert isinstance(
        features,
        dict,
    )

    features[
        "future_feature"
    ] = 1.0

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 422


def test_predict_rejects_wrong_numerical_type() -> None:
    """Return HTTP 422 instead of coercing malformed numerical input."""

    payload = _valid_payload()

    features = payload[
        "features"
    ]
    assert isinstance(
        features,
        dict,
    )

    features[
        "TransactionAmt"
    ] = "10.0"

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 422


def test_predict_does_not_modify_frozen_artifact() -> None:
    """Keep the policy artifact byte-for-byte unchanged after scoring."""

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
            "/predict",
            json=_valid_payload(
                "tx-immutable"
            ),
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


def test_batch_prediction_is_not_registered_yet() -> None:
    """Keep batch integration isolated for the next Phase 5 commit."""

    with TestClient(app) as client:
        response = client.post(
            "/batch-predict",
            json={
                "transactions": [
                    _valid_payload()
                ],
            },
        )

    assert response.status_code == 404
