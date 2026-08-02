"""Deterministic categorical preprocessing for fraud models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.model_data import FeatureContract


MISSING_CATEGORY_TOKEN = "<MISSING>"

MISSING_CATEGORY_CODE = 0
UNKNOWN_CATEGORY_CODE = 1
FIRST_KNOWN_CATEGORY_CODE = 2


def _validate_contract(
    contract: FeatureContract,
) -> None:
    """Validate feature-group coverage before fitting."""
    feature_columns = contract.feature_columns
    categorical_features = contract.categorical_features
    numerical_features = contract.numerical_features

    if not feature_columns:
        raise ValueError(
            "Feature contract must contain at least one feature"
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            "Feature contract contains duplicate feature names"
        )

    categorical_set = set(categorical_features)
    numerical_set = set(numerical_features)

    if categorical_set.intersection(numerical_set):
        raise ValueError(
            "Categorical and numerical feature lists overlap"
        )

    if categorical_set.union(
        numerical_set
    ) != set(feature_columns):
        raise ValueError(
            "Categorical and numerical features do not "
            "cover the full feature contract"
        )


def _validate_feature_frame(
    features: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    """Require a non-empty frame in exact contract order."""
    if features.empty:
        raise ValueError(
            f"{frame_name} must contain at least one row"
        )

    if list(features.columns) != list(feature_columns):
        raise ValueError(
            f"{frame_name} columns do not match the "
            "expected feature order"
        )


def _normalize_category_values(
    series: pd.Series,
) -> pd.Series:
    """Normalize categorical values using the Phase 1 token."""
    return (
        series.astype("string")
        .fillna(MISSING_CATEGORY_TOKEN)
    )


@dataclass(frozen=True)
class CategoricalEncoder:
    """Training-only categorical vocabularies and codes."""

    feature_columns: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numerical_features: tuple[str, ...]
    category_vocabularies: dict[
        str,
        tuple[str, ...],
    ]
    missing_code: int = MISSING_CATEGORY_CODE
    unknown_code: int = UNKNOWN_CATEGORY_CODE

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        contract: FeatureContract,
    ) -> "CategoricalEncoder":
        """Learn deterministic vocabularies from training only."""
        _validate_contract(contract)
        _validate_feature_frame(
            features,
            contract.feature_columns,
            frame_name="training features",
        )

        category_vocabularies: dict[
            str,
            tuple[str, ...],
        ] = {}

        for column in contract.categorical_features:
            normalized = _normalize_category_values(
                features[column]
            )

            known_categories = sorted(
                value
                for value in normalized.unique().tolist()
                if value != MISSING_CATEGORY_TOKEN
            )

            category_vocabularies[column] = tuple(
                known_categories
            )

        return cls(
            feature_columns=contract.feature_columns,
            categorical_features=(
                contract.categorical_features
            ),
            numerical_features=contract.numerical_features,
            category_vocabularies=category_vocabularies,
        )

    def transform(
        self,
        features: pd.DataFrame,
        *,
        frame_name: str = "features",
    ) -> pd.DataFrame:
        """Apply fitted mappings without expanding vocabularies."""
        _validate_feature_frame(
            features,
            self.feature_columns,
            frame_name=frame_name,
        )

        expected_vocabulary_columns = set(
            self.categorical_features
        )
        actual_vocabulary_columns = set(
            self.category_vocabularies
        )

        if (
            actual_vocabulary_columns
            != expected_vocabulary_columns
        ):
            raise ValueError(
                "Encoder vocabularies do not match the "
                "categorical feature contract"
            )

        transformed = features.copy()

        for column in self.categorical_features:
            normalized = _normalize_category_values(
                transformed[column]
            )

            vocabulary = self.category_vocabularies[
                column
            ]

            code_lookup = {
                category: code
                for code, category in enumerate(
                    vocabulary,
                    start=FIRST_KNOWN_CATEGORY_CODE,
                )
            }

            encoded = normalized.map(code_lookup)

            missing_mask = normalized.eq(
                MISSING_CATEGORY_TOKEN
            )

            encoded = encoded.fillna(
                self.unknown_code
            )
            encoded.loc[missing_mask] = (
                self.missing_code
            )

            transformed[column] = encoded.astype(
                "int32"
            )

        return transformed.loc[
            :,
            list(self.feature_columns),
        ]
