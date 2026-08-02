"""Load and validate the Phase 1 baseline-model datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.data_processing import JOIN_KEY, TARGET_COLUMN


TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"

LABELED_SPLITS: tuple[str, ...] = (
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
)

SPLIT_FILES: dict[str, str] = {
    TRAIN_SPLIT: "train.parquet",
    VALIDATION_SPLIT: "validation.parquet",
    TEST_SPLIT: "test.parquet",
}

FEATURE_METADATA_FILE = "feature_metadata.csv"
DATASET_MANIFEST_FILE = "dataset_manifest.json"

REQUIRED_METADATA_COLUMNS: list[str] = [
    "feature_position",
    "feature_name",
    "feature_group",
    "feature_type",
    "source",
]

ALLOWED_FEATURE_TYPES = {
    "categorical",
    "numerical",
}

PathLike = str | Path


@dataclass(frozen=True)
class FeatureContract:
    """Ordered feature names and types used by the baseline model."""

    feature_columns: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numerical_features: tuple[str, ...]

    @property
    def labeled_columns(self) -> tuple[str, ...]:
        """Return the required labeled Parquet column order."""
        return (
            JOIN_KEY,
            TARGET_COLUMN,
            *self.feature_columns,
        )


@dataclass(frozen=True)
class LabeledDataset:
    """One validated labeled model split."""

    transaction_ids: pd.Series
    features: pd.DataFrame
    target: pd.Series


@dataclass(frozen=True)
class BaselineDatasets:
    """Validated chronological baseline-model datasets."""

    train: LabeledDataset
    validation: LabeledDataset
    test: LabeledDataset
    feature_contract: FeatureContract
    manifest: dict[str, Any]


def _validate_artifact(path: Path) -> None:
    """Require a non-empty file at the expected artifact path."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required Phase 1 artifact not found: {path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Expected a file but found another path: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Required Phase 1 artifact is empty: {path}"
        )


def _load_feature_metadata(path: Path) -> pd.DataFrame:
    """Load and validate the ordered feature metadata."""
    _validate_artifact(path)

    metadata = pd.read_csv(path)

    if list(metadata.columns) != REQUIRED_METADATA_COLUMNS:
        raise ValueError(
            "Feature metadata columns do not match "
            "the expected schema"
        )

    if metadata.empty:
        raise ValueError(
            "Feature metadata must contain at least one feature"
        )

    if metadata.isna().any().any():
        raise ValueError(
            "Feature metadata contains missing contract values"
        )

    try:
        positions = pd.to_numeric(
            metadata["feature_position"],
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Feature positions must be integers"
        ) from error

    expected_positions = list(
        range(1, len(metadata) + 1)
    )

    if positions.tolist() != expected_positions:
        raise ValueError(
            "Feature metadata positions must be ordered "
            "and contiguous from 1"
        )

    if metadata["feature_name"].duplicated().any():
        raise ValueError(
            "Feature metadata contains duplicate feature names"
        )

    feature_types = set(
        metadata["feature_type"].astype(str)
    )

    unexpected_types = sorted(
        feature_types - ALLOWED_FEATURE_TYPES
    )

    if unexpected_types:
        raise ValueError(
            "Feature metadata contains unsupported "
            "feature types: "
            + ", ".join(unexpected_types)
        )

    if feature_types != ALLOWED_FEATURE_TYPES:
        raise ValueError(
            "Feature metadata must contain categorical "
            "and numerical features"
        )

    return metadata


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate the Phase 1 manifest."""
    _validate_artifact(path)

    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Dataset manifest is not valid JSON: {path}"
        ) from error

    if not isinstance(manifest, dict):
        raise ValueError(
            "Dataset manifest must contain a JSON object"
        )

    if manifest.get("schema_version") != 1:
        raise ValueError(
            "Unsupported dataset manifest schema version"
        )

    return manifest


def _require_mapping(
    value: object,
    field_name: str,
) -> dict[str, Any]:
    """Return a required manifest object."""
    if not isinstance(value, dict):
        raise ValueError(
            f"Dataset manifest field must be an object: "
            f"{field_name}"
        )

    return value


def _require_string_list(
    value: object,
    field_name: str,
) -> tuple[str, ...]:
    """Return a required manifest list of unique strings."""
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"Dataset manifest field must be a "
            f"non-empty list: {field_name}"
        )

    if not all(
        isinstance(item, str) and item
        for item in value
    ):
        raise ValueError(
            f"Dataset manifest list must contain "
            f"non-empty strings: {field_name}"
        )

    if len(value) != len(set(value)):
        raise ValueError(
            f"Dataset manifest list contains duplicates: "
            f"{field_name}"
        )

    return tuple(value)


def _build_feature_contract(
    metadata: pd.DataFrame,
    manifest: dict[str, Any],
) -> FeatureContract:
    """Cross-check metadata and manifest feature contracts."""
    manifest_contract = _require_mapping(
        manifest.get("feature_contract"),
        "feature_contract",
    )

    manifest_features = _require_string_list(
        manifest_contract.get("feature_columns"),
        "feature_contract.feature_columns",
    )
    manifest_categorical = _require_string_list(
        manifest_contract.get("categorical_features"),
        "feature_contract.categorical_features",
    )
    manifest_numerical = _require_string_list(
        manifest_contract.get("numerical_features"),
        "feature_contract.numerical_features",
    )

    metadata_features = tuple(
        metadata["feature_name"].astype(str)
    )
    metadata_categorical = tuple(
        metadata.loc[
            metadata["feature_type"].eq("categorical"),
            "feature_name",
        ].astype(str)
    )
    metadata_numerical = tuple(
        metadata.loc[
            metadata["feature_type"].eq("numerical"),
            "feature_name",
        ].astype(str)
    )

    if manifest_features != metadata_features:
        raise ValueError(
            "Manifest feature order does not match "
            "feature metadata"
        )

    if manifest_categorical != metadata_categorical:
        raise ValueError(
            "Manifest categorical features do not match "
            "feature metadata"
        )

    if manifest_numerical != metadata_numerical:
        raise ValueError(
            "Manifest numerical features do not match "
            "feature metadata"
        )

    categorical_set = set(manifest_categorical)
    numerical_set = set(manifest_numerical)

    if categorical_set.intersection(numerical_set):
        raise ValueError(
            "Categorical and numerical feature lists overlap"
        )

    if categorical_set.union(
        numerical_set
    ) != set(manifest_features):
        raise ValueError(
            "Categorical and numerical features do not "
            "cover the full feature contract"
        )

    expected_counts = {
        "feature_count": len(manifest_features),
        "categorical_feature_count": len(
            manifest_categorical
        ),
        "numerical_feature_count": len(
            manifest_numerical
        ),
    }

    for field_name, expected_value in (
        expected_counts.items()
    ):
        if (
            manifest_contract.get(field_name)
            != expected_value
        ):
            raise ValueError(
                f"Manifest {field_name} does not match "
                "the feature contract"
            )

    expected_labeled_columns = (
        JOIN_KEY,
        TARGET_COLUMN,
        *manifest_features,
    )
    manifest_labeled_columns = _require_string_list(
        manifest_contract.get("labeled_output_columns"),
        "feature_contract.labeled_output_columns",
    )

    if (
        manifest_labeled_columns
        != expected_labeled_columns
    ):
        raise ValueError(
            "Manifest labeled output columns do not "
            "match the feature contract"
        )

    return FeatureContract(
        feature_columns=manifest_features,
        categorical_features=manifest_categorical,
        numerical_features=manifest_numerical,
    )


def _expected_split_rows(
    manifest: dict[str, Any],
) -> dict[str, int]:
    """Read required labeled split row counts."""
    split_contract = _require_mapping(
        manifest.get("split_contract"),
        "split_contract",
    )
    split_summaries = _require_mapping(
        split_contract.get("splits"),
        "split_contract.splits",
    )

    expected_rows: dict[str, int] = {}

    for split_name in LABELED_SPLITS:
        summary = _require_mapping(
            split_summaries.get(split_name),
            f"split_contract.splits.{split_name}",
        )
        row_count = summary.get("transaction_count")

        if (
            not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count <= 0
        ):
            raise ValueError(
                "Manifest transaction_count must be "
                f"a positive integer for {split_name}"
            )

        expected_rows[split_name] = row_count

    return expected_rows


def _validate_target(
    target: pd.Series,
    split_name: str,
) -> pd.Series:
    """Require a complete binary target with both classes."""
    if target.isna().any():
        raise ValueError(
            f"{split_name} target contains missing values"
        )

    try:
        numeric_target = pd.to_numeric(
            target,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{split_name} target must be numerical"
        ) from error

    target_values = set(
        numeric_target.unique().tolist()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{split_name} target must contain only 0 and 1"
        )

    if target_values != {0, 1}:
        raise ValueError(
            f"{split_name} target must contain both classes"
        )

    return numeric_target.astype("int8")


def _load_labeled_split(
    path: Path,
    split_name: str,
    contract: FeatureContract,
    expected_rows: int,
) -> LabeledDataset:
    """Load and validate one labeled Parquet split."""
    _validate_artifact(path)

    dataframe = pd.read_parquet(path)

    if dataframe.empty:
        raise ValueError(
            f"{split_name} dataset must not be empty"
        )

    if list(dataframe.columns) != list(
        contract.labeled_columns
    ):
        raise ValueError(
            f"{split_name} columns do not match the "
            "expected labeled feature order"
        )

    if len(dataframe) != expected_rows:
        raise ValueError(
            f"{split_name} contains {len(dataframe):,} rows; "
            f"expected {expected_rows:,}"
        )

    transaction_ids = dataframe[JOIN_KEY]

    if transaction_ids.isna().any():
        raise ValueError(
            f"{split_name} contains missing "
            f"{JOIN_KEY} values"
        )

    if transaction_ids.duplicated().any():
        raise ValueError(
            f"{split_name} contains duplicate "
            f"{JOIN_KEY} values"
        )

    target = _validate_target(
        dataframe[TARGET_COLUMN],
        split_name,
    )

    features = dataframe.loc[
        :,
        list(contract.feature_columns),
    ].copy()

    non_numeric_features = [
        column
        for column in contract.numerical_features
        if not is_numeric_dtype(features[column])
    ]

    if non_numeric_features:
        raise ValueError(
            f"{split_name} numerical features have "
            "non-numerical dtypes: "
            + ", ".join(non_numeric_features)
        )

    return LabeledDataset(
        transaction_ids=transaction_ids.copy(),
        features=features,
        target=target,
    )


def _validate_split_isolation(
    datasets: dict[str, LabeledDataset],
) -> None:
    """Require TransactionID isolation across all splits."""
    combined_ids = pd.concat(
        [
            dataset.transaction_ids
            for dataset in datasets.values()
        ],
        ignore_index=True,
    )

    overlap_count = int(
        combined_ids.duplicated().sum()
    )

    if overlap_count:
        raise ValueError(
            "Labeled splits contain overlapping "
            f"{JOIN_KEY} values: {overlap_count:,}"
        )


def load_baseline_datasets(
    processed_dir: PathLike = Path("data/processed"),
) -> BaselineDatasets:
    """Load validated train, validation, and test data."""
    processed_path = Path(processed_dir).resolve()

    metadata = _load_feature_metadata(
        processed_path / FEATURE_METADATA_FILE
    )
    manifest = _load_manifest(
        processed_path / DATASET_MANIFEST_FILE
    )
    contract = _build_feature_contract(
        metadata,
        manifest,
    )
    expected_rows = _expected_split_rows(manifest)

    loaded_splits = {
        split_name: _load_labeled_split(
            processed_path / SPLIT_FILES[split_name],
            split_name,
            contract,
            expected_rows[split_name],
        )
        for split_name in LABELED_SPLITS
    }

    _validate_split_isolation(loaded_splits)

    return BaselineDatasets(
        train=loaded_splits[TRAIN_SPLIT],
        validation=loaded_splits[VALIDATION_SPLIT],
        test=loaded_splits[TEST_SPLIT],
        feature_contract=contract,
        manifest=manifest,
    )
