"""Tests for committed deterministic API demonstration payloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import model_loader
from api.config import (
    FROZEN_BASELINE_MODEL_VERSION,
    FROZEN_BLOCK_THRESHOLD,
    FROZEN_EXPLANATION_VERSION,
    FROZEN_POLICY_VERSION,
    FROZEN_REASON_CODE_VERSION,
    FROZEN_REVIEW_THRESHOLD,
)
from api.main import app
from api.schemas import (
    BatchPredictRequest,
    TransactionRequest,
)
from src.features import (
    FEATURE_COLUMNS,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SAMPLE_DIR = (
    ROOT
    / "data"
    / "sample_payloads"
)

SINGLE_PATH = (
    SAMPLE_DIR
    / "predict_single.json"
)

BATCH_PATH = (
    SAMPLE_DIR
    / "predict_batch.json"
)


@pytest.fixture(autouse=True)
def reset_process_policy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep sample-payload API tests isolated from cached policy state."""

    monkeypatch.setattr(
        model_loader,
        "_loaded",
        None,
    )


def _load_json(
    path: Path,
) -> dict[str, object]:
    """Load one committed JSON demonstration payload."""

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _assert_frozen_provenance(
    prediction: dict[str, object],
) -> None:
    """Require one demo response to expose the frozen inference versions."""

    assert (
        prediction["model_version"]
        == FROZEN_BASELINE_MODEL_VERSION
    )
    assert (
        prediction["policy_version"]
        == FROZEN_POLICY_VERSION
    )
    assert (
        prediction["explanation_version"]
        == FROZEN_EXPLANATION_VERSION
    )
    assert (
        prediction["reason_code_version"]
        == FROZEN_REASON_CODE_VERSION
    )


def test_single_sample_matches_contract_and_scores_review() -> None:
    """Require the single committed demo to be a valid REVIEW example."""

    payload = _load_json(
        SINGLE_PATH
    )

    validated = (
        TransactionRequest
        .model_validate(
            payload
        )
    )

    assert (
        validated.transaction_id
        == "demo-higher-value-002"
    )

    raw_features = payload[
        "features"
    ]
    assert isinstance(
        raw_features,
        dict,
    )

    assert list(
        raw_features
    ) == FEATURE_COLUMNS

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert (
        response.status_code
        == 200
    )

    prediction = (
        response.json()
    )

    assert (
        prediction[
            "transaction_id"
        ]
        == "demo-higher-value-002"
    )
    assert (
        prediction["decision"]
        == "REVIEW"
    )

    probability = float(
        prediction[
            "calibrated_probability"
        ]
    )

    assert (
        FROZEN_REVIEW_THRESHOLD
        <= probability
        < FROZEN_BLOCK_THRESHOLD
    )

    _assert_frozen_provenance(
        prediction
    )


def test_batch_sample_matches_contract_and_demo_decisions() -> None:
    """Require the committed batch to remain valid and representative."""

    payload = _load_json(
        BATCH_PATH
    )

    validated = (
        BatchPredictRequest
        .model_validate(
            payload
        )
    )

    expected_ids = [
        "demo-everyday-001",
        "demo-higher-value-002",
        "demo-mobile-identity-004",
    ]

    assert [
        transaction.transaction_id
        for transaction
        in validated.transactions
    ] == expected_ids

    raw_transactions = payload[
        "transactions"
    ]
    assert isinstance(
        raw_transactions,
        list,
    )

    for transaction in (
        raw_transactions
    ):
        assert isinstance(
            transaction,
            dict,
        )

        raw_features = (
            transaction[
                "features"
            ]
        )
        assert isinstance(
            raw_features,
            dict,
        )

        assert list(
            raw_features
        ) == FEATURE_COLUMNS

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/batch-predict",
            json=payload,
        )

    assert (
        response.status_code
        == 200
    )

    predictions = (
        response.json()[
            "predictions"
        ]
    )

    assert [
        prediction[
            "transaction_id"
        ]
        for prediction
        in predictions
    ] == expected_ids

    assert [
        prediction[
            "decision"
        ]
        for prediction
        in predictions
    ] == [
        "ALLOW",
        "REVIEW",
        "ALLOW",
    ]

    for prediction in (
        predictions
    ):
        _assert_frozen_provenance(
            prediction
        )

    everyday_probability = float(
        predictions[
            0
        ][
            "calibrated_probability"
        ]
    )
    review_probability = float(
        predictions[
            1
        ][
            "calibrated_probability"
        ]
    )
    mobile_probability = float(
        predictions[
            2
        ][
            "calibrated_probability"
        ]
    )

    assert (
        everyday_probability
        < FROZEN_REVIEW_THRESHOLD
    )
    assert (
        FROZEN_REVIEW_THRESHOLD
        <= review_probability
        < FROZEN_BLOCK_THRESHOLD
    )
    assert (
        mobile_probability
        < FROZEN_REVIEW_THRESHOLD
    )


def test_single_sample_matches_batch_review_profile() -> None:
    """Keep the standalone REVIEW demo identical to its batch counterpart."""

    single_payload = (
        _load_json(
            SINGLE_PATH
        )
    )
    batch_payload = (
        _load_json(
            BATCH_PATH
        )
    )

    transactions = (
        batch_payload[
            "transactions"
        ]
    )
    assert isinstance(
        transactions,
        list,
    )

    assert (
        single_payload
        == transactions[1]
    )
