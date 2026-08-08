"""FastAPI lifecycle, health, and model-information tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import model_loader
from api.config import (
    FROZEN_BASELINE_MODEL_VERSION,
    FROZEN_BLOCK_THRESHOLD,
    FROZEN_CALIBRATION_METHOD,
    FROZEN_EXPLANATION_VERSION,
    FROZEN_POLICY_SHA256,
    FROZEN_POLICY_VERSION,
    FROZEN_REASON_CODE_VERSION,
    FROZEN_REVIEW_THRESHOLD,
    settings,
)
from api.main import app
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
)


@pytest.fixture(autouse=True)
def reset_process_policy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep application lifespan tests isolated."""

    monkeypatch.setattr(
        model_loader,
        "_loaded",
        None,
    )


def test_health_reports_loaded_frozen_policy() -> None:
    """Serve health only after successful frozen-policy startup."""

    with TestClient(app) as client:
        response = client.get(
            "/health"
        )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_lifespan_requests_policy_load_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request one policy load for one application lifespan."""

    original_load_policy = (
        model_loader.load_policy
    )
    load_count = 0

    def counting_load_policy(
        policy_path: str | Path,
    ):
        nonlocal load_count

        load_count += 1

        return original_load_policy(
            policy_path
        )

    monkeypatch.setattr(
        model_loader,
        "load_policy",
        counting_load_policy,
    )

    with TestClient(app) as client:
        health_response = client.get(
            "/health"
        )
        info_response = client.get(
            "/model-info"
        )
        second_health_response = (
            client.get(
                "/health"
            )
        )

    assert (
        health_response.status_code
        == 200
    )
    assert (
        info_response.status_code
        == 200
    )
    assert (
        second_health_response.status_code
        == 200
    )

    assert load_count == 1


def test_startup_fails_when_policy_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail application startup rather than serving without a policy."""

    missing_policy = (
        tmp_path
        / "missing-policy.joblib"
    )

    monkeypatch.setattr(
        settings,
        "policy_path",
        missing_policy,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Frozen policy artifact not found",
    ):
        with TestClient(app):
            pass


def test_model_info_matches_frozen_contract() -> None:
    """Expose metadata from the verified frozen inference policy."""

    with TestClient(app) as client:
        response = client.get(
            "/model-info"
        )

    assert response.status_code == 200

    body = response.json()

    assert (
        body[
            "baseline_model_version"
        ]
        == FROZEN_BASELINE_MODEL_VERSION
    )
    assert (
        body["policy_version"]
        == FROZEN_POLICY_VERSION
    )
    assert (
        body["calibration_method"]
        == FROZEN_CALIBRATION_METHOD
    )

    assert (
        body["review_threshold"]
        == FROZEN_REVIEW_THRESHOLD
    )
    assert (
        body["block_threshold"]
        == FROZEN_BLOCK_THRESHOLD
    )

    assert (
        body["explanation_version"]
        == FROZEN_EXPLANATION_VERSION
    )
    assert (
        body["reason_code_version"]
        == FROZEN_REASON_CODE_VERSION
    )

    assert (
        body[
            "policy_artifact_sha256"
        ]
        == FROZEN_POLICY_SHA256
    )

    assert (
        body["feature_names"]
        == list(FEATURE_COLUMNS)
    )

    assert len(
        FEATURE_COLUMNS
    ) == 63
    assert len(
        CATEGORICAL_FEATURES
    ) == 29
    assert len(
        NUMERICAL_FEATURES
    ) == 34

    assert (
        body["feature_count"]
        == 63
    )
    assert (
        body[
            "categorical_feature_count"
        ]
        == 29
    )
    assert (
        body[
            "numerical_feature_count"
        ]
        == 34
    )


def test_information_requests_do_not_modify_policy_artifact() -> None:
    """Keep frozen artifact bytes unchanged across startup and requests."""

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
        assert (
            client.get(
                "/health"
            ).status_code
            == 200
        )

        assert (
            client.get(
                "/model-info"
            ).status_code
            == 200
        )

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


@pytest.mark.parametrize(
    (
        "method",
        "path",
    ),
    (
        (
            "post",
            "/threshold-simulation",
        ),
        (
            "get",
            "/recent-predictions",
        ),
    ),
)
def test_out_of_scope_legacy_routes_are_not_registered(
    method: str,
    path: str,
) -> None:
    """Do not expose legacy database or threshold-simulation APIs."""

    with TestClient(app) as client:
        response = getattr(
            client,
            method,
        )(
            path
        )

    assert response.status_code == 404
