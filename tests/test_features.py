"""Tests for the compact Phase 1 feature contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    AMOUNT_COLUMN,
    JOIN_KEY,
    TIME_COLUMN,
)
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    IDENTITY_FEATURES,
    MATCH_FEATURES,
    MISSING_CATEGORY,
    NUMERICAL_FEATURES,
    TRANSACTION_FEATURES,
    build_feature_metadata,
    build_features,
)


def _make_transaction() -> pd.DataFrame:
    """Create a small transaction table with the full input contract."""
    data: dict[str, list[object]] = {
        JOIN_KEY: [101, 102, 103],
        TIME_COLUMN: [86_400, 90_000, 176_400],
    }

    for column in TRANSACTION_FEATURES:
        data[column] = [1.0, 2.0, np.nan]

    data[AMOUNT_COLUMN] = [0.0, 99.0, 10.0]
    data["ProductCD"] = ["W", "C", "R"]
    data["card1"] = [1001, 1002, 1003]
    data["card4"] = ["visa", "mastercard", None]
    data["card6"] = ["debit", "credit", None]
    data["P_emaildomain"] = [
        "gmail.com",
        None,
        "outlook.com",
    ]
    data["R_emaildomain"] = [
        None,
        "hotmail.com",
        "gmail.com",
    ]

    for column in MATCH_FEATURES:
        data[column] = ["T", "F", None]

    return pd.DataFrame(data)


def _make_identity() -> pd.DataFrame:
    """Create identity rows in a different order from transactions."""
    data: dict[str, list[object]] = {
        JOIN_KEY: [103, 101],
    }

    for column in IDENTITY_FEATURES:
        data[column] = [3.0, 1.0]

    data["DeviceType"] = ["mobile", "desktop"]
    data["id_12"] = ["Found", "NotFound"]
    data["id_15"] = ["New", None]
    data["id_16"] = ["Found", "NotFound"]
    data["id_28"] = ["New", "Found"]
    data["id_29"] = ["Found", "NotFound"]
    data["id_31"] = ["chrome", "safari"]
    data["id_35"] = ["T", "F"]
    data["id_36"] = ["F", "T"]
    data["id_37"] = ["T", "F"]
    data["id_38"] = ["F", "T"]

    return pd.DataFrame(data)


def test_build_features_matches_declared_contract() -> None:
    transaction = _make_transaction()
    identity = _make_identity()

    features = build_features(
        transaction,
        identity,
        time_origin=86_400,
    )

    assert features.shape == (3, 63)
    assert list(features.columns) == FEATURE_COLUMNS
    assert len(CATEGORICAL_FEATURES) == 29
    assert len(NUMERICAL_FEATURES) == 34
    assert not set(CATEGORICAL_FEATURES) & set(
        NUMERICAL_FEATURES
    )
    assert set(FEATURE_COLUMNS) == (
        set(CATEGORICAL_FEATURES)
        | set(NUMERICAL_FEATURES)
    )


def test_build_features_aligns_identity_by_transaction_id() -> None:
    transaction = _make_transaction()
    identity = _make_identity()

    features = build_features(
        transaction,
        identity,
        time_origin=86_400,
    )

    assert features["DeviceType"].tolist() == [
        "desktop",
        MISSING_CATEGORY,
        "mobile",
    ]

    assert features["identity_availability"].tolist() == [
        "1",
        "0",
        "1",
    ]


def test_build_features_creates_expected_derived_values() -> None:
    transaction = _make_transaction()

    features = build_features(
        transaction,
        identity=None,
        time_origin=86_400,
    )

    assert features["relative_day"].tolist() == [0, 0, 1]
    assert features["relative_hour_cycle"].tolist() == [
        "0",
        "1",
        "1",
    ]

    np.testing.assert_allclose(
        features["transaction_amount_log1p"].to_numpy(),
        np.log1p([0.0, 99.0, 10.0]),
    )

    assert features.attrs["time_origin"] == 86_400.0


def test_build_features_without_identity_uses_missing_values() -> None:
    transaction = _make_transaction()

    features = build_features(
        transaction,
        identity=None,
    )

    assert (
        features["identity_availability"]
        == "0"
    ).all()

    assert (
        features["DeviceType"]
        == MISSING_CATEGORY
    ).all()

    assert features["id_01"].isna().all()


def test_build_features_rejects_duplicate_transaction_keys() -> None:
    transaction = _make_transaction()
    transaction.loc[1, JOIN_KEY] = transaction.loc[0, JOIN_KEY]

    with pytest.raises(
        ValueError,
        match="duplicate TransactionID",
    ):
        build_features(transaction)


def test_build_features_rejects_missing_required_column() -> None:
    transaction = _make_transaction().drop(
        columns=["ProductCD"]
    )

    with pytest.raises(
        ValueError,
        match="ProductCD",
    ):
        build_features(transaction)


def test_build_features_rejects_negative_amount() -> None:
    transaction = _make_transaction()
    transaction.loc[0, AMOUNT_COLUMN] = -1.0

    with pytest.raises(
        ValueError,
        match="must not contain negative values",
    ):
        build_features(transaction)


def test_build_features_rejects_time_before_origin() -> None:
    transaction = _make_transaction()

    with pytest.raises(
        ValueError,
        match="earlier than time_origin",
    ):
        build_features(
            transaction,
            time_origin=90_001,
        )


def test_build_feature_metadata_matches_contract() -> None:
    metadata = build_feature_metadata()

    assert metadata.shape == (63, 5)
    assert metadata["feature_position"].tolist() == list(
        range(1, 64)
    )
    assert metadata["feature_name"].tolist() == FEATURE_COLUMNS
    assert metadata["feature_name"].is_unique

    assert set(metadata["feature_type"]) == {
        "categorical",
        "numerical",
    }

    assert set(metadata["source"]) == {
        "transaction",
        "identity",
        "derived",
    }
