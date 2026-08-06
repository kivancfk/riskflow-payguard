"""Tests for explanation feature-value state classification."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.explainability import (
    MISSING,
    OBSERVED,
    UNKNOWN_CATEGORY,
)
from src.explanation_values import (
    classify_feature_value_states,
)
from src.model_data import FeatureContract
from src.preprocessing import (
    FIRST_KNOWN_CATEGORY_CODE,
    MISSING_CATEGORY_TOKEN,
    CategoricalEncoder,
)


def _feature_contract() -> FeatureContract:
    """Return a small mixed feature contract."""

    return FeatureContract(
        feature_columns=(
            "category",
            "amount",
            "distance",
        ),
        categorical_features=(
            "category",
        ),
        numerical_features=(
            "amount",
            "distance",
        ),
    )


def _training_features() -> pd.DataFrame:
    """Return deterministic encoder-fit data."""

    return pd.DataFrame(
        {
            "category": [
                "known_a",
                "known_b",
                None,
            ],
            "amount": [
                10.0,
                20.0,
                30.0,
            ],
            "distance": [
                1.0,
                2.0,
                np.nan,
            ],
        }
    )


def _encoder() -> CategoricalEncoder:
    """Return a fitted frozen categorical encoder."""

    return CategoricalEncoder.fit(
        _training_features(),
        _feature_contract(),
    )


def test_classifies_observed_missing_and_unknown_values() -> None:
    features = pd.DataFrame(
        {
            "category": [
                "known_a",
                "unseen",
                None,
            ],
            "amount": [
                100.0,
                np.nan,
                300.0,
            ],
            "distance": [
                pd.NA,
                5.0,
                6.0,
            ],
        },
        index=[
            101,
            102,
            103,
        ],
    )

    states = classify_feature_value_states(
        features,
        _encoder(),
    )

    assert states.index.tolist() == [
        101,
        102,
        103,
    ]
    assert states.columns.tolist() == [
        "category",
        "amount",
        "distance",
    ]

    assert states.to_numpy(
        dtype="object"
    ).tolist() == [
        [
            OBSERVED,
            OBSERVED,
            MISSING,
        ],
        [
            UNKNOWN_CATEGORY,
            MISSING,
            OBSERVED,
        ],
        [
            MISSING,
            OBSERVED,
            OBSERVED,
        ],
    ]


def test_literal_missing_token_follows_encoder_semantics() -> None:
    features = pd.DataFrame(
        {
            "category": [
                MISSING_CATEGORY_TOKEN,
            ],
            "amount": [
                10.0,
            ],
            "distance": [
                1.0,
            ],
        }
    )

    states = classify_feature_value_states(
        features,
        _encoder(),
    )

    assert (
        states.loc[
            0,
            "category",
        ]
        == MISSING
    )


def test_categorical_normalization_matches_encoder() -> None:
    contract = FeatureContract(
        feature_columns=(
            "category",
            "amount",
        ),
        categorical_features=(
            "category",
        ),
        numerical_features=(
            "amount",
        ),
    )
    training = pd.DataFrame(
        {
            "category": [
                1,
                2,
            ],
            "amount": [
                10.0,
                20.0,
            ],
        }
    )
    encoder = CategoricalEncoder.fit(
        training,
        contract,
    )
    features = pd.DataFrame(
        {
            "category": [
                1,
                3,
            ],
            "amount": [
                30.0,
                40.0,
            ],
        }
    )

    states = classify_feature_value_states(
        features,
        encoder,
    )

    assert states[
        "category"
    ].tolist() == [
        OBSERVED,
        UNKNOWN_CATEGORY,
    ]


def test_classification_is_deterministic_and_read_only() -> None:
    encoder = _encoder()
    features = pd.DataFrame(
        {
            "category": [
                "known_b",
                "new_category",
            ],
            "amount": [
                np.nan,
                20.0,
            ],
            "distance": [
                4.0,
                5.0,
            ],
        },
        index=[
            7,
            8,
        ],
    )

    original_features = features.copy(
        deep=True
    )
    original_vocabularies = dict(
        encoder.category_vocabularies
    )

    first = classify_feature_value_states(
        features,
        encoder,
    )
    second = classify_feature_value_states(
        features,
        encoder,
    )

    assert_frame_equal(
        first,
        second,
    )
    assert_frame_equal(
        features,
        original_features,
    )
    assert (
        encoder.category_vocabularies
        == original_vocabularies
    )


def test_numerical_non_missing_values_are_observed() -> None:
    features = pd.DataFrame(
        {
            "category": [
                "known_a",
                "known_b",
            ],
            "amount": [
                0.0,
                -1.0,
            ],
            "distance": [
                float("inf"),
                float("-inf"),
            ],
        }
    )

    states = classify_feature_value_states(
        features,
        _encoder(),
    )

    assert states[
        "amount"
    ].tolist() == [
        OBSERVED,
        OBSERVED,
    ]
    assert states[
        "distance"
    ].tolist() == [
        OBSERVED,
        OBSERVED,
    ]


def test_rejects_non_encoder() -> None:
    with pytest.raises(
        ValueError,
        match="must be a CategoricalEncoder",
    ):
        classify_feature_value_states(
            _training_features(),
            object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    (
        "missing_code",
        "unknown_code",
        "message",
    ),
    [
        (
            0,
            0,
            "must be different",
        ),
        (
            FIRST_KNOWN_CATEGORY_CODE,
            1,
            "reserved below",
        ),
        (
            0,
            FIRST_KNOWN_CATEGORY_CODE,
            "reserved below",
        ),
        (
            True,
            1,
            "reserved below",
        ),
    ],
)
def test_rejects_invalid_reserved_codes(
    missing_code: object,
    unknown_code: object,
    message: str,
) -> None:
    encoder = replace(
        _encoder(),
        missing_code=missing_code,
        unknown_code=unknown_code,
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        classify_feature_value_states(
            _training_features(),
            encoder,
        )


def test_rejects_mismatched_vocabularies() -> None:
    encoder = replace(
        _encoder(),
        category_vocabularies={},
    )

    with pytest.raises(
        ValueError,
        match="vocabularies do not match",
    ):
        classify_feature_value_states(
            _training_features(),
            encoder,
        )


def test_requires_exact_frozen_feature_order() -> None:
    features = _training_features().loc[
        :,
        [
            "amount",
            "category",
            "distance",
        ],
    ]

    with pytest.raises(
        ValueError,
        match="expected feature order",
    ):
        classify_feature_value_states(
            features,
            _encoder(),
        )


def test_rejects_empty_feature_frame() -> None:
    features = _training_features().iloc[
        0:0
    ]

    with pytest.raises(
        ValueError,
        match="at least one row",
    ):
        classify_feature_value_states(
            features,
            _encoder(),
        )
