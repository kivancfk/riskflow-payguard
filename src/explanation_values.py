"""Classify raw feature values for model explanations."""

from __future__ import annotations

from numbers import Integral

import pandas as pd

from src.explainability import (
    MISSING,
    OBSERVED,
    UNKNOWN_CATEGORY,
)
from src.preprocessing import (
    FIRST_KNOWN_CATEGORY_CODE,
    CategoricalEncoder,
)


def _validate_reserved_code(
    value: int,
    *,
    value_name: str,
) -> int:
    """Return a valid encoder-reserved categorical code."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not (
            0
            <= int(value)
            < FIRST_KNOWN_CATEGORY_CODE
        )
    ):
        raise ValueError(
            f"{value_name} must be an integer "
            "reserved below the first known-category code"
        )

    return int(value)


def _validate_encoder(
    encoder: object,
) -> CategoricalEncoder:
    """Validate the frozen encoder state used for classification."""

    if not isinstance(
        encoder,
        CategoricalEncoder,
    ):
        raise ValueError(
            "encoder must be a CategoricalEncoder"
        )

    missing_code = _validate_reserved_code(
        encoder.missing_code,
        value_name="encoder missing_code",
    )
    unknown_code = _validate_reserved_code(
        encoder.unknown_code,
        value_name="encoder unknown_code",
    )

    if missing_code == unknown_code:
        raise ValueError(
            "encoder missing and unknown codes "
            "must be different"
        )

    expected_vocabulary_features = set(
        encoder.categorical_features
    )
    actual_vocabulary_features = set(
        encoder.category_vocabularies
    )

    if (
        actual_vocabulary_features
        != expected_vocabulary_features
    ):
        raise ValueError(
            "encoder vocabularies do not match "
            "the categorical feature contract"
        )

    return encoder


def classify_feature_value_states(
    features: pd.DataFrame,
    encoder: CategoricalEncoder,
    *,
    frame_name: str = "explanation features",
) -> pd.DataFrame:
    """Classify every raw feature value for explanation generation.

    The returned DataFrame preserves the input row index and frozen
    feature order.

    Categorical states are inferred from the encoded missing and unknown
    sentinels produced by the frozen encoder. Numerical values are marked
    missing only when the original raw value is missing.
    """

    validated_encoder = _validate_encoder(
        encoder
    )

    encoded_features = (
        validated_encoder.transform(
            features,
            frame_name=frame_name,
        )
    )

    states = pd.DataFrame(
        OBSERVED,
        index=features.index.copy(),
        columns=list(
            validated_encoder.feature_columns
        ),
        dtype="string",
    )

    for feature_name in (
        validated_encoder.numerical_features
    ):
        missing_mask = features[
            feature_name
        ].isna()

        states.loc[
            missing_mask,
            feature_name,
        ] = MISSING

    for feature_name in (
        validated_encoder.categorical_features
    ):
        encoded_values = encoded_features[
            feature_name
        ]

        missing_mask = encoded_values.eq(
            validated_encoder.missing_code
        )
        unknown_mask = encoded_values.eq(
            validated_encoder.unknown_code
        )

        if (
            missing_mask & unknown_mask
        ).any():
            raise RuntimeError(
                "A categorical value cannot be both "
                "missing and unknown"
            )

        states.loc[
            missing_mask,
            feature_name,
        ] = MISSING
        states.loc[
            unknown_mask,
            feature_name,
        ] = UNKNOWN_CATEGORY

    return states.loc[
        :,
        list(
            validated_encoder.feature_columns
        ),
    ]
