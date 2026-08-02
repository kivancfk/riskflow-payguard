"""Tests for Phase 1 processed-dataset materialization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.materialize_phase1 import (
    TRAIN_OUTPUT,
    _build_labeled_output,
    _build_unlabeled_output,
    _chronological_split_indices,
    _parquet_artifact_summary,
    _split_summary,
    _validate_output_policy,
    _validate_parquet_schema,
    _write_parquet,
)
from src.data_processing import (
    AMOUNT_COLUMN,
    JOIN_KEY,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
)


def _make_feature_frame(
    row_count: int,
) -> pd.DataFrame:
    """Create a minimal frame satisfying the feature contract."""
    data: dict[str, list[object]] = {}

    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL_FEATURES:
            data[column] = ["value"] * row_count
        else:
            data[column] = [0.0] * row_count

    return pd.DataFrame(data)


def test_chronological_split_indices_use_stable_order() -> None:
    transaction = pd.DataFrame(
        {
            TIME_COLUMN: list(reversed(range(20))),
        }
    )

    split_indices = _chronological_split_indices(
        transaction
    )

    assert split_indices["train"].tolist() == list(
        range(19, 5, -1)
    )
    assert split_indices["validation"].tolist() == [
        5,
        4,
        3,
    ]
    assert split_indices["test"].tolist() == [
        2,
        1,
        0,
    ]

    assert len(split_indices["train"]) == 14
    assert len(split_indices["validation"]) == 3
    assert len(split_indices["test"]) == 3


def test_validate_output_policy_rejects_existing_artifact(
    tmp_path: Path,
) -> None:
    existing_path = tmp_path / TRAIN_OUTPUT
    existing_path.write_text(
        "existing",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="Use --overwrite",
    ):
        _validate_output_policy(
            tmp_path,
            overwrite=False,
        )

    _validate_output_policy(
        tmp_path,
        overwrite=True,
    )


def test_build_labeled_output_preserves_requested_order() -> None:
    transaction = pd.DataFrame(
        {
            JOIN_KEY: [101, 102, 103],
            TARGET_COLUMN: [0, 1, 0],
        }
    )
    features = _make_feature_frame(3)

    output = _build_labeled_output(
        transaction,
        features,
        pd.Index([2, 0]),
    )

    assert output.shape == (2, 65)
    assert output[JOIN_KEY].tolist() == [103, 101]
    assert output[TARGET_COLUMN].tolist() == [0, 0]
    assert list(output.columns) == [
        JOIN_KEY,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]


def test_build_unlabeled_output_matches_kaggle_schema() -> None:
    transaction = pd.DataFrame(
        {
            JOIN_KEY: [201, 202],
        }
    )
    features = _make_feature_frame(2)

    output = _build_unlabeled_output(
        transaction,
        features,
    )

    assert output.shape == (2, 64)
    assert output[JOIN_KEY].tolist() == [201, 202]
    assert TARGET_COLUMN not in output.columns
    assert list(output.columns) == [
        JOIN_KEY,
        *FEATURE_COLUMNS,
    ]


def test_parquet_helpers_validate_structure(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "first": [1, 2, 3],
            "second": ["a", "b", "c"],
        }
    )
    path = tmp_path / "artifact.parquet"

    _write_parquet(
        dataframe,
        path,
    )

    _validate_parquet_schema(
        path,
        expected_columns=["first", "second"],
        expected_rows=3,
    )

    summary = _parquet_artifact_summary(path)

    assert summary["filename"] == "artifact.parquet"
    assert summary["row_count"] == 3
    assert summary["column_count"] == 2
    assert summary["size_bytes"] > 0


def test_split_summary_reports_fraud_and_identity() -> None:
    transaction = pd.DataFrame(
        {
            JOIN_KEY: [1, 2, 3],
            TARGET_COLUMN: [0, 1, 1],
            TIME_COLUMN: [100, 200, 300],
            AMOUNT_COLUMN: [10.0, 50.0, 100.0],
        }
    )

    features = _make_feature_frame(3)
    features["identity_availability"] = [
        "0",
        "1",
        "1",
    ]

    summary = _split_summary(
        transaction,
        features,
        pd.Index([0, 1]),
    )

    assert summary["transaction_count"] == 2
    assert summary["fraud_count"] == 1
    assert summary["fraud_rate"] == pytest.approx(0.5)
    assert summary["identity_count"] == 1
    assert summary["identity_coverage"] == pytest.approx(
        0.5
    )
    assert summary["minimum_transaction_dt"] == 100
    assert summary["maximum_transaction_dt"] == 200
    assert summary["total_transaction_amount"] == pytest.approx(
        60.0
    )
    assert summary[
        "fraudulent_transaction_amount"
    ] == pytest.approx(50.0)
