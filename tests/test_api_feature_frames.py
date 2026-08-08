"""Tests for exact-order Phase 5 API feature-frame construction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.config import (
    FROZEN_POLICY_ARTIFACT,
)
from api.feature_frames import (
    _validate_api_feature_contract,
    build_batch_feature_frame,
    build_feature_frame,
)
from api.schemas import (
    TransactionRequest,
)
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
)
from src.policy_bundle import (
    CalibratedPolicyBundle,
    load_calibrated_policy_bundle,
)


POLICY_PATH = Path(
    FROZEN_POLICY_ARTIFACT
).resolve()


@pytest.fixture(scope="module")
def frozen_bundle() -> CalibratedPolicyBundle:
    """Load the existing trusted frozen policy artifact."""

    return load_calibrated_policy_bundle(
        POLICY_PATH
    )


def _valid_features(
    *,
    transaction_amount: float | None = 10.0,
    product_code: str | None = None,
) -> dict[str, object]:
    """Return one complete model feature payload."""

    categorical = set(
        CATEGORICAL_FEATURES
    )

    payload: dict[
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

    payload[
        "TransactionAmt"
    ] = transaction_amount
    payload[
        "ProductCD"
    ] = product_code

    return payload


def _transaction(
    transaction_id: str | int,
    *,
    transaction_amount: float | None = 10.0,
    product_code: str | None = None,
) -> TransactionRequest:
    """Build one validated transaction request."""

    return (
        TransactionRequest.model_validate(
            {
                "transaction_id": (
                    transaction_id
                ),
                "features": _valid_features(
                    transaction_amount=(
                        transaction_amount
                    ),
                    product_code=product_code,
                ),
            }
        )
    )


def test_frozen_bundle_matches_api_feature_contract(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Cross-check API typing against the actual frozen artifact."""

    baseline = (
        frozen_bundle.baseline_bundle
    )

    assert tuple(
        baseline.feature_columns
    ) == tuple(
        FEATURE_COLUMNS
    )

    assert set(
        baseline.categorical_features
    ) == set(
        CATEGORICAL_FEATURES
    )

    assert set(
        baseline.numerical_features
    ) == set(
        NUMERICAL_FEATURES
    )

    _validate_api_feature_contract(
        frozen_bundle
    )


def test_single_feature_frame_uses_frozen_bundle_order(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Construct one row in exact model-facing feature order."""

    transaction = _transaction(
        "tx-001"
    )

    frame = build_feature_frame(
        transaction,
        frozen_bundle,
    )

    assert frame.shape == (
        1,
        len(
            frozen_bundle
            .baseline_bundle
            .feature_columns
        ),
    )

    assert tuple(
        frame.columns
    ) == tuple(
        frozen_bundle
        .baseline_bundle
        .feature_columns
    )


def test_numerical_values_are_float_model_inputs(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Normalize validated numerical API values to model-safe floats."""

    transaction = _transaction(
        "tx-001",
        transaction_amount=25.0,
    )

    frame = build_feature_frame(
        transaction,
        frozen_bundle,
    )

    assert (
        frame["TransactionAmt"].dtype
        == np.dtype("float64")
    )
    assert (
        frame.loc[
            0,
            "TransactionAmt",
        ]
        == 25.0
    )

    for feature_name in (
        frozen_bundle
        .baseline_bundle
        .numerical_features
    ):
        assert (
            frame[
                feature_name
            ].dtype
            == np.dtype("float64")
        )


def test_explicit_missing_numerical_value_becomes_nan(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Represent explicit missing numerical values for LightGBM as NaN."""

    transaction = _transaction(
        "tx-001",
        transaction_amount=None,
    )

    frame = build_feature_frame(
        transaction,
        frozen_bundle,
    )

    assert pd.isna(
        frame.loc[
            0,
            "TransactionAmt",
        ]
    )


def test_unknown_category_is_preserved_until_frozen_encoder(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Pass unseen categories unchanged to the existing frozen encoder."""

    transaction = _transaction(
        "tx-unknown",
        product_code=(
            "UNSEEN-FUTURE-CATEGORY"
        ),
    )

    frame = build_feature_frame(
        transaction,
        frozen_bundle,
    )

    assert (
        frame.loc[
            0,
            "ProductCD",
        ]
        == "UNSEEN-FUTURE-CATEGORY"
    )

    encoder = (
        frozen_bundle
        .baseline_bundle
        .encoder
    )

    encoded = encoder.transform(
        frame,
        frame_name=(
            "API unknown-category test"
        ),
    )

    assert (
        encoded.loc[
            0,
            "ProductCD",
        ]
        == encoder.unknown_code
    )


def test_missing_category_reaches_frozen_missing_code(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Preserve explicit categorical missingness for the frozen encoder."""

    transaction = _transaction(
        "tx-missing",
        product_code=None,
    )

    frame = build_feature_frame(
        transaction,
        frozen_bundle,
    )

    encoder = (
        frozen_bundle
        .baseline_bundle
        .encoder
    )

    encoded = encoder.transform(
        frame,
        frame_name=(
            "API missing-category test"
        ),
    )

    assert (
        encoded.loc[
            0,
            "ProductCD",
        ]
        == encoder.missing_code
    )


def test_batch_frame_preserves_identifier_and_row_order(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Keep API request order unchanged in identifiers and model rows."""

    transactions = [
        _transaction(
            "tx-second",
            transaction_amount=20.0,
        ),
        _transaction(
            "tx-first",
            transaction_amount=10.0,
        ),
        _transaction(
            300,
            transaction_amount=30.0,
        ),
    ]

    transaction_ids, frame = (
        build_batch_feature_frame(
            transactions,
            frozen_bundle,
        )
    )

    assert transaction_ids == (
        "tx-second",
        "tx-first",
        300,
    )

    assert frame[
        "TransactionAmt"
    ].tolist() == [
        20.0,
        10.0,
        30.0,
    ]


def test_batch_frame_rejects_empty_input(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Protect the frame helper when called outside Pydantic validation."""

    with pytest.raises(
        ValueError,
        match="at least one transaction",
    ):
        build_batch_feature_frame(
            [],
            frozen_bundle,
        )


def test_batch_frame_rejects_duplicate_identifiers(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Keep helper-level uniqueness protection for direct callers."""

    duplicate_transactions = [
        _transaction(
            "duplicate",
        ),
        _transaction(
            "duplicate",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="unique within a batch",
    ):
        build_batch_feature_frame(
            duplicate_transactions,
            frozen_bundle,
        )


def test_frame_builder_detects_frozen_contract_drift(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Fail closed if API and artifact feature names ever diverge."""

    changed_baseline = replace(
        frozen_bundle.baseline_bundle,
        feature_columns=(
            *frozen_bundle
            .baseline_bundle
            .feature_columns[:-1],
            "unexpected_feature",
        ),
    )

    changed_policy = replace(
        frozen_bundle,
        baseline_bundle=(
            changed_baseline
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "API feature names do not match "
            "the frozen model contract"
        ),
    ):
        _validate_api_feature_contract(
            changed_policy
        )
