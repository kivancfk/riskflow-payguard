"""Tests for strict Phase 5 API request and response schemas."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from api.schemas import (
    AnalystReasonResponse,
    BatchPredictRequest,
    FeatureContributionResponse,
    PredictionResponse,
    ReconstructionResponse,
    TransactionModelFeatures,
    TransactionRequest,
)
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
)


def _valid_features() -> dict[
    str,
    object,
]:
    """Return one complete valid API feature payload."""

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


def _valid_transaction(
    transaction_id: str | int = "tx-001",
) -> dict[str, object]:
    """Return one complete valid transaction request payload."""

    return {
        "transaction_id": transaction_id,
        "features": _valid_features(),
    }


def test_feature_schema_matches_phase_1_contract() -> None:
    """Expose every frozen model feature exactly once."""

    schema_features = tuple(
        TransactionModelFeatures
        .model_fields
        .keys()
    )

    assert schema_features == tuple(
        FEATURE_COLUMNS
    )

    assert len(schema_features) == 63

    assert set(
        CATEGORICAL_FEATURES
    ).union(
        NUMERICAL_FEATURES
    ) == set(
        schema_features
    )

    assert not set(
        CATEGORICAL_FEATURES
    ).intersection(
        NUMERICAL_FEATURES
    )


def test_complete_transaction_request_is_accepted() -> None:
    """Accept a complete strict transaction payload."""

    request = (
        TransactionRequest.model_validate(
            _valid_transaction()
        )
    )

    assert (
        request.transaction_id
        == "tx-001"
    )
    assert (
        set(
            request.features.model_dump()
        )
        == set(FEATURE_COLUMNS)
    )


def test_explicit_missing_feature_is_accepted() -> None:
    """Treat explicit null as missing rather than an absent API field."""

    payload = _valid_transaction()
    payload["features"][
        "ProductCD"
    ] = None
    payload["features"][
        "TransactionAmt"
    ] = None

    request = (
        TransactionRequest.model_validate(
            payload
        )
    )

    assert (
        request.features.ProductCD
        is None
    )
    assert (
        request.features.TransactionAmt
        is None
    )


def test_unknown_categorical_value_is_accepted() -> None:
    """Do not reject categories merely because training did not observe them."""

    payload = _valid_transaction()
    payload["features"][
        "ProductCD"
    ] = "UNSEEN-FUTURE-CATEGORY"

    request = (
        TransactionRequest.model_validate(
            payload
        )
    )

    assert (
        request.features.ProductCD
        == "UNSEEN-FUTURE-CATEGORY"
    )


def test_numeric_categorical_value_is_accepted() -> None:
    """Allow raw numeric values for categoricals such as card fields."""

    payload = _valid_transaction()
    payload["features"][
        "card1"
    ] = 12345

    request = (
        TransactionRequest.model_validate(
            payload
        )
    )

    assert request.features.card1 == 12345


def test_absent_feature_is_rejected() -> None:
    """Require every frozen feature key in the HTTP request."""

    payload = _valid_transaction()
    del payload["features"][
        "TransactionAmt"
    ]

    with pytest.raises(
        ValidationError,
    ):
        TransactionRequest.model_validate(
            payload
        )


def test_extra_feature_is_rejected() -> None:
    """Reject fields outside the frozen model feature contract."""

    payload = _valid_transaction()
    payload["features"][
        "future_engineered_feature"
    ] = 1.0

    with pytest.raises(
        ValidationError,
    ):
        TransactionRequest.model_validate(
            payload
        )


def test_extra_transaction_field_is_rejected() -> None:
    """Reject unexpected request-level metadata."""

    payload = _valid_transaction()
    payload["merchant_name"] = "example"

    with pytest.raises(
        ValidationError,
    ):
        TransactionRequest.model_validate(
            payload
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        "12.5",
        True,
        {},
        [],
    ],
)
def test_invalid_numerical_types_are_rejected(
    invalid_value: object,
) -> None:
    """Do not coerce malformed numerical model inputs."""

    payload = _valid_transaction()
    payload["features"][
        "TransactionAmt"
    ] = invalid_value

    with pytest.raises(
        ValidationError,
    ):
        TransactionRequest.model_validate(
            payload
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_non_finite_numerical_values_are_rejected(
    invalid_value: float,
) -> None:
    """Reject non-finite model inputs before inference."""

    payload = _valid_transaction()
    payload["features"][
        "TransactionAmt"
    ] = invalid_value

    with pytest.raises(
        ValidationError,
    ):
        TransactionRequest.model_validate(
            payload
        )


@pytest.mark.parametrize(
    "invalid_identifier",
    [
        "",
        " ",
        " tx-001",
        "tx-001 ",
        True,
        1.5,
        None,
    ],
)
def test_invalid_transaction_identifiers_are_rejected(
    invalid_identifier: object,
) -> None:
    """Apply the stable string-or-integer identifier contract."""

    payload = _valid_transaction()
    payload[
        "transaction_id"
    ] = invalid_identifier

    with pytest.raises(
        ValidationError,
    ):
        TransactionRequest.model_validate(
            payload
        )


@pytest.mark.parametrize(
    "valid_identifier",
    [
        "tx-001",
        1001,
    ],
)
def test_valid_transaction_identifiers_are_preserved(
    valid_identifier: str | int,
) -> None:
    """Preserve valid request identifiers without coercion."""

    request = (
        TransactionRequest.model_validate(
            _valid_transaction(
                valid_identifier
            )
        )
    )

    assert (
        request.transaction_id
        == valid_identifier
    )
    assert type(
        request.transaction_id
    ) is type(
        valid_identifier
    )


def test_empty_batch_is_rejected() -> None:
    """Require at least one transaction per batch."""

    with pytest.raises(
        ValidationError,
    ):
        BatchPredictRequest.model_validate(
            {
                "transactions": [],
            }
        )


def test_duplicate_batch_identifiers_are_rejected() -> None:
    """Require one unique identifier per batch row."""

    payload = {
        "transactions": [
            _valid_transaction(
                "duplicate"
            ),
            _valid_transaction(
                "duplicate"
            ),
        ],
    }

    with pytest.raises(
        ValidationError,
        match="unique within a batch",
    ):
        BatchPredictRequest.model_validate(
            payload
        )


def test_batch_request_preserves_input_order() -> None:
    """Keep transaction sequence unchanged during validation."""

    batch = (
        BatchPredictRequest.model_validate(
            {
                "transactions": [
                    _valid_transaction(
                        "tx-b"
                    ),
                    _valid_transaction(
                        "tx-a"
                    ),
                    _valid_transaction(
                        42
                    ),
                ],
            }
        )
    )

    assert [
        transaction.transaction_id
        for transaction in batch.transactions
    ] == [
        "tx-b",
        "tx-a",
        42,
    ]


def test_prediction_response_contract_is_typed() -> None:
    """Validate the planned Phase 5 prediction response shape."""

    contribution = (
        FeatureContributionResponse(
            feature="TransactionAmt",
            feature_index=0,
            feature_group="TRANSACTION_AMOUNT",
            direction="INCREASES_SCORE",
            shap_value_raw=0.4,
            absolute_shap_value_raw=0.4,
            value_state="OBSERVED",
            rank=1,
        )
    )

    reason = AnalystReasonResponse(
        code=(
            "TRANSACTION_AMOUNT_OBSERVED_"
            "INCREASES_SCORE"
        ),
        message=(
            "Transaction amount signal increased "
            "the model score."
        ),
    )

    reconstruction = (
        ReconstructionResponse(
            raw_model_margin=0.3,
            expected_value_raw=0.1,
            shap_sum_raw=0.2,
            reconstructed_raw_margin=0.3,
            reconstructed_raw_model_score=0.57,
            margin_reconstruction_error=0.0,
            score_reconstruction_error=0.0,
        )
    )

    response = PredictionResponse(
        transaction_id="tx-001",
        model_version="baseline-v1",
        policy_version="calibrated-policy-v1",
        explanation_version=(
            "shap-explanation-v1"
        ),
        reason_code_version="reason-codes-v1",
        raw_model_score=0.57,
        calibrated_probability=0.41,
        decision="REVIEW",
        top_positive_contributions=[
            contribution
        ],
        top_negative_contributions=[],
        reason_codes=[
            reason.code
        ],
        reasons=[
            reason
        ],
        reconstruction=reconstruction,
    )

    assert (
        response.transaction_id
        == "tx-001"
    )
    assert (
        response.top_positive_contributions[
            0
        ].feature
        == "TransactionAmt"
    )
    assert (
        response.reason_codes
        == [reason.code]
    )
