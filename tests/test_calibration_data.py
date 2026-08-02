"""Tests for chronological calibration development data."""

from __future__ import annotations

import pandas as pd
import pytest

from src.calibration_data import (
    PARTITION_METHOD,
    partition_calibration_development_data,
)
from src.model_data import (
    FeatureContract,
    LabeledDataset,
    ValidationDatasetBundle,
)


FEATURE_CONTRACT = FeatureContract(
    feature_columns=(
        "category_feature",
        "numeric_feature",
    ),
    categorical_features=("category_feature",),
    numerical_features=("numeric_feature",),
)


def _validation_bundle(
    row_count: int = 8,
    *,
    transaction_ids: list[int] | None = None,
    target: list[int] | None = None,
) -> ValidationDatasetBundle:
    """Create chronological synthetic validation data."""

    ids = (
        transaction_ids
        if transaction_ids is not None
        else list(range(100, 100 + row_count))
    )
    labels = (
        target
        if target is not None
        else [
            index % 2
            for index in range(row_count)
        ]
    )

    validation = LabeledDataset(
        transaction_ids=pd.Series(
            ids,
            name="TransactionID",
        ),
        features=pd.DataFrame(
            {
                "category_feature": [
                    f"category-{index % 3}"
                    for index in range(row_count)
                ],
                "numeric_feature": [
                    float(index)
                    for index in range(row_count)
                ],
            }
        ),
        target=pd.Series(
            labels,
            name="isFraud",
            dtype="int8",
        ),
    )

    return ValidationDatasetBundle(
        validation=validation,
        feature_contract=FEATURE_CONTRACT,
        manifest={"schema_version": 1},
    )


def test_partition_preserves_even_chronology_and_metadata() -> None:
    source = _validation_bundle()

    first = partition_calibration_development_data(
        source
    )
    second = partition_calibration_development_data(
        source
    )

    assert first.calibration_fit.transaction_ids.tolist() == [
        100,
        101,
        102,
        103,
    ]
    assert first.policy_selection.transaction_ids.tolist() == [
        104,
        105,
        106,
        107,
    ]

    assert set(
        first.calibration_fit.transaction_ids
    ).isdisjoint(
        set(first.policy_selection.transaction_ids)
    )

    assert first.calibration_fit.features.columns.tolist() == [
        "category_feature",
        "numeric_feature",
    ]
    assert first.policy_selection.features.columns.tolist() == [
        "category_feature",
        "numeric_feature",
    ]

    assert first.feature_contract is FEATURE_CONTRACT
    assert first.manifest == source.manifest
    assert first.manifest is not source.manifest

    metadata = first.metadata

    assert metadata == second.metadata
    assert metadata.partition_method == PARTITION_METHOD
    assert metadata.total_rows == 8
    assert metadata.split_index == 4

    assert (
        metadata.requested_calibration_fit_fraction
        == 0.5
    )
    assert metadata.actual_calibration_fit_fraction == 0.5

    assert metadata.calibration_fit.row_count == 4
    assert (
        metadata.calibration_fit.first_transaction_id
        == 100
    )
    assert (
        metadata.calibration_fit.last_transaction_id
        == 103
    )
    assert metadata.calibration_fit.negative_count == 2
    assert metadata.calibration_fit.fraud_count == 2
    assert metadata.calibration_fit.fraud_rate == 0.5

    assert metadata.policy_selection.row_count == 4
    assert (
        metadata.policy_selection.first_transaction_id
        == 104
    )
    assert (
        metadata.policy_selection.last_transaction_id
        == 107
    )
    assert metadata.policy_selection.negative_count == 2
    assert metadata.policy_selection.fraud_count == 2
    assert metadata.policy_selection.fraud_rate == 0.5


def test_partition_uses_floor_for_odd_row_count() -> None:
    partition = partition_calibration_development_data(
        _validation_bundle(row_count=7)
    )

    assert len(partition.calibration_fit.target) == 3
    assert len(partition.policy_selection.target) == 4
    assert partition.metadata.split_index == 3

    assert (
        partition.metadata.actual_calibration_fit_fraction
        == pytest.approx(3 / 7)
    )

    assert partition.calibration_fit.transaction_ids.tolist() == [
        100,
        101,
        102,
    ]
    assert partition.policy_selection.transaction_ids.tolist() == [
        103,
        104,
        105,
        106,
    ]


def test_partition_rejects_too_few_rows() -> None:
    with pytest.raises(
        ValueError,
        match="at least 4 rows",
    ):
        partition_calibration_development_data(
            _validation_bundle(row_count=3)
        )


def test_partition_rejects_duplicate_transaction_ids() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate TransactionID",
    ):
        partition_calibration_development_data(
            _validation_bundle(
                transaction_ids=[
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    100,
                ],
            )
        )


def test_partition_rejects_feature_contract_mismatch() -> None:
    source = _validation_bundle()

    validation = LabeledDataset(
        transaction_ids=source.validation.transaction_ids,
        features=source.validation.features.loc[
            :,
            [
                "numeric_feature",
                "category_feature",
            ],
        ],
        target=source.validation.target,
    )

    mismatched = ValidationDatasetBundle(
        validation=validation,
        feature_contract=FEATURE_CONTRACT,
        manifest=source.manifest,
    )

    with pytest.raises(
        ValueError,
        match="ordered feature contract",
    ):
        partition_calibration_development_data(
            mismatched
        )


def test_partition_requires_both_classes_per_segment() -> None:
    source = _validation_bundle(
        target=[
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "calibration_fit target must "
            "contain both classes"
        ),
    ):
        partition_calibration_development_data(
            source
        )


def test_partition_rejects_misaligned_lengths() -> None:
    source = _validation_bundle()

    validation = LabeledDataset(
        transaction_ids=source.validation.transaction_ids,
        features=source.validation.features.iloc[:-1],
        target=source.validation.target,
    )

    misaligned = ValidationDatasetBundle(
        validation=validation,
        feature_contract=FEATURE_CONTRACT,
        manifest=source.manifest,
    )

    with pytest.raises(
        ValueError,
        match="feature rows do not align",
    ):
        partition_calibration_development_data(
            misaligned
        )
