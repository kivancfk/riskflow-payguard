"""Cross-cutting hardening tests for the frozen Phase 5 API."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import model_loader
from api import prediction_service
from api.config import (
    FROZEN_POLICY_SHA256,
    settings,
)
from api.main import app
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
)
from src.shap_reconstruction import (
    RECONSTRUCTION_ABSOLUTE_TOLERANCE,
)


@pytest.fixture(autouse=True)
def reset_process_policy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep hardening tests isolated from process-level loader state."""

    monkeypatch.setattr(
        model_loader,
        "_loaded",
        None,
    )


def _valid_features(
    *,
    transaction_amount: float | None = 10.0,
    product_code: str | int | float | None = None,
) -> dict[str, object]:
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
    transaction_amount: float | None = 10.0,
    product_code: str | int | float | None = None,
) -> dict[str, object]:
    """Return one complete transaction payload."""

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


def _batch_payload() -> dict[str, object]:
    """Return a deterministic ordered request batch."""

    return {
        "transactions": [
            _transaction_payload(
                "batch-c",
                transaction_amount=30.0,
            ),
            _transaction_payload(
                "batch-a",
                transaction_amount=10.0,
            ),
            _transaction_payload(
                2002,
                transaction_amount=20.0,
            ),
        ],
    }


def test_single_request_uses_phase_4_boundary_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require one explanation-aware inference call per single request."""

    original_predict = (
        prediction_service
        .predict_policy_with_explanations
    )
    call_count = 0

    def counting_predict(
        *args: object,
        **kwargs: object,
    ):
        nonlocal call_count
        call_count += 1

        return original_predict(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        prediction_service,
        "predict_policy_with_explanations",
        counting_predict,
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=_transaction_payload(
                "single-boundary"
            ),
        )

    assert response.status_code == 200
    assert call_count == 1


def test_batch_request_uses_phase_4_boundary_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require one vectorized explanation-aware inference call per batch."""

    original_predict = (
        prediction_service
        .predict_policy_with_explanations
    )
    call_count = 0

    def counting_predict(
        *args: object,
        **kwargs: object,
    ):
        nonlocal call_count
        call_count += 1

        return original_predict(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        prediction_service,
        "predict_policy_with_explanations",
        counting_predict,
    )

    with TestClient(app) as client:
        response = client.post(
            "/batch-predict",
            json=_batch_payload(),
        )

    assert response.status_code == 200
    assert call_count == 1


def test_invalid_single_request_never_reaches_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed single input at validation before model inference."""

    call_count = 0

    def forbidden_predict(
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal call_count
        call_count += 1

        raise AssertionError(
            "Inference boundary must not run "
            "for an invalid request"
        )

    monkeypatch.setattr(
        prediction_service,
        "predict_policy_with_explanations",
        forbidden_predict,
    )

    payload = _transaction_payload(
        "invalid-single"
    )

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
    assert call_count == 0


def test_invalid_batch_never_reaches_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject duplicate batch identifiers before model inference."""

    call_count = 0

    def forbidden_predict(
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal call_count
        call_count += 1

        raise AssertionError(
            "Inference boundary must not run "
            "for an invalid batch"
        )

    monkeypatch.setattr(
        prediction_service,
        "predict_policy_with_explanations",
        forbidden_predict,
    )

    payload = {
        "transactions": [
            _transaction_payload(
                "duplicate"
            ),
            _transaction_payload(
                "duplicate"
            ),
        ],
    }

    with TestClient(app) as client:
        response = client.post(
            "/batch-predict",
            json=payload,
        )

    assert response.status_code == 422
    assert call_count == 0


def test_unseen_and_missing_values_do_not_mutate_encoder() -> None:
    """Accept unseen/missing inputs while keeping frozen vocabularies unchanged."""

    with TestClient(app) as client:
        bundle = (
            model_loader.get_policy()
        )
        encoder = (
            bundle
            .baseline_bundle
            .encoder
        )

        vocabularies_before = (
            copy.deepcopy(
                encoder
                .category_vocabularies
            )
        )

        unseen_response = (
            client.post(
                "/predict",
                json=_transaction_payload(
                    "unknown-category",
                    product_code=(
                        "UNSEEN-FUTURE-CATEGORY"
                    ),
                ),
            )
        )

        missing_response = (
            client.post(
                "/predict",
                json=_transaction_payload(
                    "missing-values",
                    transaction_amount=None,
                    product_code=None,
                ),
            )
        )

        vocabularies_after = (
            copy.deepcopy(
                encoder
                .category_vocabularies
            )
        )

    assert (
        unseen_response.status_code
        == 200
    )
    assert (
        missing_response.status_code
        == 200
    )

    assert (
        vocabularies_after
        == vocabularies_before
    )


def test_single_response_preserves_explanation_invariants() -> None:
    """Require reason and SHAP reconstruction invariants at the HTTP boundary."""

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=_transaction_payload(
                "explanation-invariants",
                transaction_amount=75.0,
            ),
        )

    assert response.status_code == 200

    prediction = response.json()

    assert (
        prediction[
            "reason_codes"
        ]
        == [
            reason[
                "code"
            ]
            for reason in (
                prediction[
                    "reasons"
                ]
            )
        ]
    )

    reconstruction = (
        prediction[
            "reconstruction"
        ]
    )

    assert (
        reconstruction[
            "margin_reconstruction_error"
        ]
        <= RECONSTRUCTION_ABSOLUTE_TOLERANCE
    )
    assert (
        reconstruction[
            "score_reconstruction_error"
        ]
        <= RECONSTRUCTION_ABSOLUTE_TOLERANCE
    )

    assert (
        abs(
            reconstruction[
                "raw_model_margin"
            ]
            - reconstruction[
                "reconstructed_raw_margin"
            ]
        )
        <= RECONSTRUCTION_ABSOLUTE_TOLERANCE
    )

    assert (
        abs(
            prediction[
                "raw_model_score"
            ]
            - reconstruction[
                "reconstructed_raw_model_score"
            ]
        )
        <= RECONSTRUCTION_ABSOLUTE_TOLERANCE
    )

    assert (
        "scored_at"
        not in prediction
    )


def test_batch_response_preserves_explanation_invariants() -> None:
    """Require explanation invariants independently for every batch row."""

    with TestClient(app) as client:
        response = client.post(
            "/batch-predict",
            json=_batch_payload(),
        )

    assert response.status_code == 200

    predictions = response.json()[
        "predictions"
    ]

    assert len(
        predictions
    ) == 3

    for prediction in predictions:
        assert (
            prediction[
                "reason_codes"
            ]
            == [
                reason[
                    "code"
                ]
                for reason in (
                    prediction[
                        "reasons"
                    ]
                )
            ]
        )

        reconstruction = (
            prediction[
                "reconstruction"
            ]
        )

        assert (
            reconstruction[
                "margin_reconstruction_error"
            ]
            <= RECONSTRUCTION_ABSOLUTE_TOLERANCE
        )
        assert (
            reconstruction[
                "score_reconstruction_error"
            ]
            <= RECONSTRUCTION_ABSOLUTE_TOLERANCE
        )

        assert (
            "scored_at"
            not in prediction
        )


def test_mixed_api_workload_preserves_frozen_state() -> None:
    """Keep artifact bytes and categorical vocabularies immutable across traffic."""

    artifact_path = (
        Path(
            settings.policy_path
        )
        .expanduser()
        .resolve()
    )

    sha256_before = (
        model_loader
        .calculate_policy_sha256(
            artifact_path
        )
    )

    assert (
        sha256_before
        == FROZEN_POLICY_SHA256
    )

    with TestClient(app) as client:
        bundle = (
            model_loader.get_policy()
        )
        encoder = (
            bundle
            .baseline_bundle
            .encoder
        )

        vocabularies_before = (
            copy.deepcopy(
                encoder
                .category_vocabularies
            )
        )

        single_response = (
            client.post(
                "/predict",
                json=_transaction_payload(
                    "mixed-single",
                    product_code=(
                        "ANOTHER-UNSEEN-CATEGORY"
                    ),
                ),
            )
        )

        batch_response = (
            client.post(
                "/batch-predict",
                json=_batch_payload(),
            )
        )

        repeated_response = (
            client.post(
                "/predict",
                json=_transaction_payload(
                    "mixed-single",
                    product_code=(
                        "ANOTHER-UNSEEN-CATEGORY"
                    ),
                ),
            )
        )

        vocabularies_after = (
            copy.deepcopy(
                encoder
                .category_vocabularies
            )
        )

    assert (
        single_response.status_code
        == 200
    )
    assert (
        batch_response.status_code
        == 200
    )
    assert (
        repeated_response.status_code
        == 200
    )

    assert (
        single_response.json()
        == repeated_response.json()
    )

    assert (
        vocabularies_after
        == vocabularies_before
    )

    sha256_after = (
        model_loader
        .calculate_policy_sha256(
            artifact_path
        )
    )

    assert (
        sha256_after
        == sha256_before
        == FROZEN_POLICY_SHA256
    )


def test_phase_5_business_route_surface_is_frozen() -> None:
    """Expose only the intended Phase 5 business-facing routes."""

    registered_routes = {
        (
            method,
            route.path,
        )
        for route in app.routes
        for method in (
            getattr(
                route,
                "methods",
                set(),
            )
            or set()
        )
    }

    assert (
        "GET",
        "/health",
    ) in registered_routes

    assert (
        "GET",
        "/model-info",
    ) in registered_routes

    assert (
        "POST",
        "/predict",
    ) in registered_routes

    assert (
        "POST",
        "/batch-predict",
    ) in registered_routes

    assert (
        "POST",
        "/threshold-simulation",
    ) not in registered_routes

    assert (
        "GET",
        "/recent-predictions",
    ) not in registered_routes
