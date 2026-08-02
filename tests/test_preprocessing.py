"""Tests for deterministic categorical preprocessing."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.model_data import FeatureContract
from src.preprocessing import (
    CategoricalEncoder,
    MISSING_CATEGORY_CODE,
    UNKNOWN_CATEGORY_CODE,
)


FEATURE_CONTRACT = FeatureContract(
    feature_columns=(
        "category_a",
        "numeric_feature",
        "category_b",
    ),
    categorical_features=(
        "category_a",
        "category_b",
    ),
    numerical_features=(
        "numeric_feature",
    ),
)


def _training_features() -> pd.DataFrame:
    """Return a small mixed-type training frame."""
    return pd.DataFrame(
        {
            "category_a": [
                "beta",
                "alpha",
                None,
                "alpha",
            ],
            "numeric_feature": [
                1.0,
                np.nan,
                3.0,
                4.0,
            ],
            "category_b": [
                "x",
                "y",
                "x",
                "<MISSING>",
            ],
        }
    )


def test_fit_builds_deterministic_training_vocabularies(
) -> None:
    training = _training_features()

    encoder = CategoricalEncoder.fit(
        training,
        FEATURE_CONTRACT,
    )
    reordered_encoder = CategoricalEncoder.fit(
        training.iloc[::-1].reset_index(drop=True),
        FEATURE_CONTRACT,
    )

    assert encoder.category_vocabularies == {
        "category_a": ("alpha", "beta"),
        "category_b": ("x", "y"),
    }
    assert (
        reordered_encoder.category_vocabularies
        == encoder.category_vocabularies
    )


def test_transform_handles_missing_and_unseen_categories(
) -> None:
    encoder = CategoricalEncoder.fit(
        _training_features(),
        FEATURE_CONTRACT,
    )

    validation = pd.DataFrame(
        {
            "category_a": [
                "alpha",
                "gamma",
                None,
            ],
            "numeric_feature": [
                10.0,
                20.0,
                np.nan,
            ],
            "category_b": [
                "y",
                "unknown",
                "<MISSING>",
            ],
        }
    )

    transformed = encoder.transform(
        validation,
        frame_name="validation features",
    )

    assert transformed["category_a"].tolist() == [
        2,
        UNKNOWN_CATEGORY_CODE,
        MISSING_CATEGORY_CODE,
    ]
    assert transformed["category_b"].tolist() == [
        3,
        UNKNOWN_CATEGORY_CODE,
        MISSING_CATEGORY_CODE,
    ]

    assert transformed[
        "numeric_feature"
    ].iloc[:2].tolist() == [10.0, 20.0]

    assert pd.isna(
        transformed["numeric_feature"].iloc[2]
    )

    assert str(
        transformed["category_a"].dtype
    ) == "int32"
    assert str(
        transformed["category_b"].dtype
    ) == "int32"


def test_transform_preserves_feature_order_and_index(
) -> None:
    training = _training_features()
    training.index = [10, 20, 30, 40]

    encoder = CategoricalEncoder.fit(
        training,
        FEATURE_CONTRACT,
    )
    transformed = encoder.transform(training)

    assert list(transformed.columns) == list(
        FEATURE_CONTRACT.feature_columns
    )
    assert transformed.index.tolist() == [
        10,
        20,
        30,
        40,
    ]


def test_transform_does_not_expand_training_vocabulary(
) -> None:
    encoder = CategoricalEncoder.fit(
        _training_features(),
        FEATURE_CONTRACT,
    )

    original_vocabularies = dict(
        encoder.category_vocabularies
    )

    validation = pd.DataFrame(
        {
            "category_a": ["new_a"],
            "numeric_feature": [5.0],
            "category_b": ["new_b"],
        }
    )

    transformed = encoder.transform(validation)

    assert transformed["category_a"].tolist() == [
        UNKNOWN_CATEGORY_CODE
    ]
    assert transformed["category_b"].tolist() == [
        UNKNOWN_CATEGORY_CODE
    ]
    assert (
        encoder.category_vocabularies
        == original_vocabularies
    )


def test_transform_rejects_incorrect_feature_order(
) -> None:
    encoder = CategoricalEncoder.fit(
        _training_features(),
        FEATURE_CONTRACT,
    )

    reordered = _training_features().loc[
        :,
        [
            "numeric_feature",
            "category_a",
            "category_b",
        ],
    ]

    with pytest.raises(
        ValueError,
        match="expected feature order",
    ):
        encoder.transform(reordered)


def test_encoder_survives_joblib_round_trip(
    tmp_path: Path,
) -> None:
    training = _training_features()

    encoder = CategoricalEncoder.fit(
        training,
        FEATURE_CONTRACT,
    )
    expected = encoder.transform(training)

    artifact_path = tmp_path / "encoder.joblib"

    joblib.dump(
        encoder,
        artifact_path,
    )
    restored = joblib.load(artifact_path)

    actual = restored.transform(training)

    assert (
        restored.category_vocabularies
        == encoder.category_vocabularies
    )
    pd.testing.assert_frame_equal(
        actual,
        expected,
    )
