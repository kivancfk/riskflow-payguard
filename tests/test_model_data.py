"""Tests for baseline-model dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_processing import JOIN_KEY, TARGET_COLUMN
from src.model_data import (
    load_baseline_datasets,
    load_validation_dataset,
)


FEATURE_COLUMNS = [
    "category_feature",
    "numeric_feature",
]

CATEGORICAL_FEATURES = [
    "category_feature",
]

NUMERICAL_FEATURES = [
    "numeric_feature",
]

SPLIT_IDS = {
    "train": [101, 102],
    "validation": [201, 202],
    "test": [301, 302],
}


def _write_feature_metadata(
    processed_dir: Path,
) -> None:
    """Write a small valid feature contract."""
    metadata = pd.DataFrame(
        {
            "feature_position": [1, 2],
            "feature_name": FEATURE_COLUMNS,
            "feature_group": ["test", "test"],
            "feature_type": [
                "categorical",
                "numerical",
            ],
            "source": ["transaction", "transaction"],
        }
    )

    metadata.to_csv(
        processed_dir / "feature_metadata.csv",
        index=False,
    )


def _write_manifest(
    processed_dir: Path,
    *,
    feature_columns: list[str] | None = None,
) -> None:
    """Write a small valid dataset manifest."""
    ordered_features = (
        feature_columns
        if feature_columns is not None
        else FEATURE_COLUMNS
    )

    manifest = {
        "schema_version": 1,
        "feature_contract": {
            "feature_count": 2,
            "categorical_feature_count": 1,
            "numerical_feature_count": 1,
            "feature_columns": ordered_features,
            "categorical_features": (
                CATEGORICAL_FEATURES
            ),
            "numerical_features": NUMERICAL_FEATURES,
            "labeled_output_columns": [
                JOIN_KEY,
                TARGET_COLUMN,
                *ordered_features,
            ],
        },
        "split_contract": {
            "splits": {
                split_name: {
                    "transaction_count": 2,
                }
                for split_name in SPLIT_IDS
            }
        },
    }

    (
        processed_dir / "dataset_manifest.json"
    ).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _write_split(
    processed_dir: Path,
    split_name: str,
    *,
    transaction_ids: list[int] | None = None,
    target: list[int] | None = None,
    column_order: list[str] | None = None,
    rename_numeric: bool = False,
) -> None:
    """Write one small labeled Parquet split."""
    dataframe = pd.DataFrame(
        {
            JOIN_KEY: (
                transaction_ids
                if transaction_ids is not None
                else SPLIT_IDS[split_name]
            ),
            TARGET_COLUMN: (
                target
                if target is not None
                else [0, 1]
            ),
            "category_feature": ["a", "b"],
            "numeric_feature": [1.0, 2.0],
        }
    )

    if rename_numeric:
        dataframe = dataframe.rename(
            columns={
                "numeric_feature": (
                    "unexpected_feature"
                )
            }
        )

    if column_order is not None:
        dataframe = dataframe.loc[
            :,
            column_order,
        ]

    dataframe.to_parquet(
        processed_dir / f"{split_name}.parquet",
        index=False,
    )


def _write_valid_artifacts(
    processed_dir: Path,
) -> None:
    """Write a complete valid synthetic dataset."""
    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_feature_metadata(processed_dir)
    _write_manifest(processed_dir)

    for split_name in SPLIT_IDS:
        _write_split(
            processed_dir,
            split_name,
        )


def test_load_baseline_datasets_returns_validated_splits(
    tmp_path: Path,
) -> None:
    _write_valid_artifacts(tmp_path)

    datasets = load_baseline_datasets(tmp_path)

    assert datasets.feature_contract.feature_columns == (
        "category_feature",
        "numeric_feature",
    )
    assert (
        datasets.feature_contract.categorical_features
        == ("category_feature",)
    )
    assert (
        datasets.feature_contract.numerical_features
        == ("numeric_feature",)
    )

    assert datasets.train.features.shape == (2, 2)
    assert datasets.validation.features.shape == (2, 2)
    assert datasets.test.features.shape == (2, 2)

    assert datasets.train.target.tolist() == [0, 1]
    assert str(datasets.train.target.dtype) == "int8"
    assert datasets.test.transaction_ids.tolist() == [
        301,
        302,
    ]


def test_load_baseline_datasets_rejects_missing_artifact(
    tmp_path: Path,
) -> None:
    _write_valid_artifacts(tmp_path)

    (
        tmp_path / "validation.parquet"
    ).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="validation.parquet",
    ):
        load_baseline_datasets(tmp_path)


def test_load_baseline_datasets_rejects_invalid_targets(
    tmp_path: Path,
) -> None:
    _write_valid_artifacts(tmp_path)

    invalid_cases = [
        ([0, 2], "only 0 and 1"),
        ([0, 0], "both classes"),
    ]

    for target, expected_message in invalid_cases:
        _write_split(
            tmp_path,
            "train",
            target=target,
        )

        with pytest.raises(
            ValueError,
            match=expected_message,
        ):
            load_baseline_datasets(tmp_path)


def test_load_baseline_datasets_rejects_contract_mismatch(
    tmp_path: Path,
) -> None:
    _write_valid_artifacts(tmp_path)

    _write_manifest(
        tmp_path,
        feature_columns=[
            "numeric_feature",
            "category_feature",
        ],
    )

    with pytest.raises(
        ValueError,
        match="feature order",
    ):
        load_baseline_datasets(tmp_path)


def test_load_baseline_datasets_rejects_feature_reordering(
    tmp_path: Path,
) -> None:
    _write_valid_artifacts(tmp_path)

    _write_split(
        tmp_path,
        "validation",
        column_order=[
            JOIN_KEY,
            TARGET_COLUMN,
            "numeric_feature",
            "category_feature",
        ],
    )

    with pytest.raises(
        ValueError,
        match="expected labeled feature order",
    ):
        load_baseline_datasets(tmp_path)


def test_load_baseline_datasets_rejects_incompatible_schema(
    tmp_path: Path,
) -> None:
    _write_valid_artifacts(tmp_path)

    _write_split(
        tmp_path,
        "test",
        rename_numeric=True,
    )

    with pytest.raises(
        ValueError,
        match="expected labeled feature order",
    ):
        load_baseline_datasets(tmp_path)


def test_load_baseline_datasets_rejects_split_overlap(
    tmp_path: Path,
) -> None:
    _write_valid_artifacts(tmp_path)

    _write_split(
        tmp_path,
        "test",
        transaction_ids=[101, 302],
    )

    with pytest.raises(
        ValueError,
        match="overlapping TransactionID",
    ):
        load_baseline_datasets(tmp_path)

def test_load_validation_dataset_does_not_require_other_splits(
    tmp_path: Path,
) -> None:
    _write_valid_artifacts(tmp_path)

    (tmp_path / "train.parquet").unlink()
    (tmp_path / "test.parquet").unlink()

    dataset = load_validation_dataset(tmp_path)

    assert dataset.validation.transaction_ids.tolist() == [
        201,
        202,
    ]
    assert dataset.validation.target.tolist() == [0, 1]
    assert dataset.feature_contract.feature_columns == (
        "category_feature",
        "numeric_feature",
    )

