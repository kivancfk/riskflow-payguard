"""Chronological data for calibration and policy development."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.model_data import (
    FeatureContract,
    LabeledDataset,
    ValidationDatasetBundle,
    load_validation_dataset,
)


PARTITION_METHOD = "chronological_row_position"
REQUESTED_CALIBRATION_FIT_FRACTION = 0.5
MINIMUM_DEVELOPMENT_ROWS = 4


@dataclass(frozen=True)
class DevelopmentSegmentMetadata:
    """Reproducible summary for one development segment."""

    name: str
    row_count: int
    first_transaction_id: object
    last_transaction_id: object
    negative_count: int
    fraud_count: int
    fraud_rate: float


@dataclass(frozen=True)
class CalibrationDevelopmentMetadata:
    """Metadata describing the chronological partition."""

    partition_method: str
    total_rows: int
    split_index: int
    requested_calibration_fit_fraction: float
    actual_calibration_fit_fraction: float
    calibration_fit: DevelopmentSegmentMetadata
    policy_selection: DevelopmentSegmentMetadata


@dataclass(frozen=True)
class CalibrationDevelopmentData:
    """Validation data isolated for two development purposes."""

    calibration_fit: LabeledDataset
    policy_selection: LabeledDataset
    feature_contract: FeatureContract
    manifest: dict[str, Any]
    metadata: CalibrationDevelopmentMetadata


def _validate_source_dataset(
    validation: LabeledDataset,
    feature_contract: FeatureContract,
) -> pd.Series:
    """Validate alignment and return a normalized target."""

    row_count = len(validation.transaction_ids)

    if row_count < MINIMUM_DEVELOPMENT_ROWS:
        raise ValueError(
            "Validation data must contain at least "
            f"{MINIMUM_DEVELOPMENT_ROWS} rows"
        )

    if len(validation.features) != row_count:
        raise ValueError(
            "Validation feature rows do not align with "
            "TransactionID rows"
        )

    if len(validation.target) != row_count:
        raise ValueError(
            "Validation target rows do not align with "
            "TransactionID rows"
        )

    if validation.transaction_ids.isna().any():
        raise ValueError(
            "Validation data contains missing TransactionID values"
        )

    if validation.transaction_ids.duplicated().any():
        raise ValueError(
            "Validation data contains duplicate TransactionID values"
        )

    if tuple(validation.features.columns) != (
        feature_contract.feature_columns
    ):
        raise ValueError(
            "Validation feature columns do not match "
            "the ordered feature contract"
        )

    if validation.target.isna().any():
        raise ValueError(
            "Validation target contains missing values"
        )

    try:
        numeric_target = pd.to_numeric(
            validation.target,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Validation target must be numerical"
        ) from error

    target_values = set(
        numeric_target.unique().tolist()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "Validation target must contain only 0 and 1"
        )

    return numeric_target.astype("int8").reset_index(
        drop=True
    )


def _slice_dataset(
    validation: LabeledDataset,
    normalized_target: pd.Series,
    start: int,
    stop: int,
) -> LabeledDataset:
    """Copy one contiguous positional slice."""

    return LabeledDataset(
        transaction_ids=(
            validation.transaction_ids.iloc[start:stop]
            .reset_index(drop=True)
            .copy()
        ),
        features=(
            validation.features.iloc[start:stop]
            .reset_index(drop=True)
            .copy()
        ),
        target=(
            normalized_target.iloc[start:stop]
            .reset_index(drop=True)
            .copy()
        ),
    )


def _require_both_classes(
    dataset: LabeledDataset,
    segment_name: str,
) -> None:
    """Require fraudulent and legitimate rows."""

    if set(dataset.target.unique().tolist()) != {0, 1}:
        raise ValueError(
            f"{segment_name} target must contain both classes"
        )


def _python_scalar(value: object) -> object:
    """Convert NumPy-like scalars into Python values."""

    item_method = getattr(value, "item", None)

    if callable(item_method):
        return item_method()

    return value


def _segment_metadata(
    name: str,
    dataset: LabeledDataset,
) -> DevelopmentSegmentMetadata:
    """Build deterministic metadata for one segment."""

    row_count = len(dataset.target)
    fraud_count = int(dataset.target.sum())
    negative_count = row_count - fraud_count

    return DevelopmentSegmentMetadata(
        name=name,
        row_count=row_count,
        first_transaction_id=_python_scalar(
            dataset.transaction_ids.iloc[0]
        ),
        last_transaction_id=_python_scalar(
            dataset.transaction_ids.iloc[-1]
        ),
        negative_count=negative_count,
        fraud_count=fraud_count,
        fraud_rate=fraud_count / row_count,
    )


def partition_calibration_development_data(
    validation_data: ValidationDatasetBundle,
) -> CalibrationDevelopmentData:
    """Split validation into earlier fit and later policy data."""

    validation = validation_data.validation
    normalized_target = _validate_source_dataset(
        validation,
        validation_data.feature_contract,
    )

    total_rows = len(validation.transaction_ids)
    split_index = total_rows // 2

    calibration_fit = _slice_dataset(
        validation,
        normalized_target,
        0,
        split_index,
    )
    policy_selection = _slice_dataset(
        validation,
        normalized_target,
        split_index,
        total_rows,
    )

    _require_both_classes(
        calibration_fit,
        "calibration_fit",
    )
    _require_both_classes(
        policy_selection,
        "policy_selection",
    )

    calibration_fit_metadata = _segment_metadata(
        "calibration_fit",
        calibration_fit,
    )
    policy_selection_metadata = _segment_metadata(
        "policy_selection",
        policy_selection,
    )

    metadata = CalibrationDevelopmentMetadata(
        partition_method=PARTITION_METHOD,
        total_rows=total_rows,
        split_index=split_index,
        requested_calibration_fit_fraction=(
            REQUESTED_CALIBRATION_FIT_FRACTION
        ),
        actual_calibration_fit_fraction=(
            split_index / total_rows
        ),
        calibration_fit=calibration_fit_metadata,
        policy_selection=policy_selection_metadata,
    )

    return CalibrationDevelopmentData(
        calibration_fit=calibration_fit,
        policy_selection=policy_selection,
        feature_contract=validation_data.feature_contract,
        manifest=deepcopy(validation_data.manifest),
        metadata=metadata,
    )


def load_calibration_development_data(
    processed_dir: str | Path = Path("data/processed"),
) -> CalibrationDevelopmentData:
    """Load validation only and create the partition."""

    validation_data = load_validation_dataset(
        processed_dir
    )

    return partition_calibration_development_data(
        validation_data
    )
