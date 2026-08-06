"""Tests for native LightGBM SHAP extraction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from pandas.testing import assert_frame_equal

from src.lightgbm_shap import (
    extract_lightgbm_contributions,
)
from src.model_bundle import (
    ModelBundle,
    build_model_bundle,
)
from src.model_data import FeatureContract
from src.preprocessing import (
    CategoricalEncoder,
)


FEATURE_CONTRACT = FeatureContract(
    feature_columns=(
        "category_feature",
        "numeric_feature",
    ),
    categorical_features=(
        "category_feature",
    ),
    numerical_features=(
        "numeric_feature",
    ),
)


def _training_features() -> pd.DataFrame:
    """Return deterministic LightGBM training data."""

    return pd.DataFrame(
        {
            "category_feature": [
                "low",
                "low",
                "medium",
                "medium",
                "high",
                "high",
                "low",
                "medium",
                "high",
                None,
                "low",
                "high",
            ],
            "numeric_feature": [
                0.1,
                0.2,
                0.4,
                0.5,
                0.8,
                0.9,
                0.3,
                0.6,
                1.0,
                0.7,
                0.15,
                1.1,
            ],
        }
    )


def _training_target() -> pd.Series:
    """Return a binary target aligned to training data."""

    return pd.Series(
        [
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
        ],
        dtype="int8",
    )


def _make_bundle() -> ModelBundle:
    """Train and package a tiny deterministic LightGBM model."""

    features = _training_features()
    encoder = CategoricalEncoder.fit(
        features,
        FEATURE_CONTRACT,
    )
    encoded_features = encoder.transform(
        features
    )

    model = LGBMClassifier(
        objective="binary",
        n_estimators=10,
        learning_rate=0.20,
        num_leaves=4,
        min_child_samples=1,
        random_state=7,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(
        encoded_features,
        _training_target(),
        categorical_feature=[
            "category_feature",
        ],
    )

    return build_model_bundle(
        model=model,
        encoder=encoder,
        feature_contract=FEATURE_CONTRACT,
        training_config={
            "n_estimators": 10,
            "seed": 7,
        },
        scale_pos_weight=1.4,
        best_iteration=10,
        validation_metrics={
            "pr_auc": 0.80,
        },
        test_metrics={
            "pr_auc": 0.75,
        },
        dataset_manifest={
            "schema_version": 1,
        },
        model_version="baseline-test-v1",
        created_at_utc=(
            "2026-08-02T15:00:00+00:00"
        ),
    )


def _inference_features() -> pd.DataFrame:
    """Return known, unseen, and missing categories."""

    return pd.DataFrame(
        {
            "category_feature": [
                "low",
                "future",
                None,
            ],
            "numeric_feature": [
                0.2,
                0.75,
                0.5,
            ],
        },
        index=[
            101,
            102,
            103,
        ],
    )


def test_extracts_expected_native_matrix_shape() -> None:
    result = extract_lightgbm_contributions(
        _make_bundle(),
        _inference_features(),
    )

    assert result.feature_names == (
        "category_feature",
        "numeric_feature",
    )
    assert result.row_count == 3
    assert result.feature_count == 2
    assert result.shap_values_raw.shape == (
        3,
        2,
    )
    assert result.expected_values_raw.shape == (
        3,
    )
    assert np.isfinite(
        result.shap_values_raw
    ).all()
    assert np.isfinite(
        result.expected_values_raw
    ).all()


def test_missing_and_unknown_categories_are_supported() -> None:
    result = extract_lightgbm_contributions(
        _make_bundle(),
        _inference_features(),
    )

    assert result.row_count == 3
    assert not np.isnan(
        result.shap_values_raw
    ).any()


def test_extraction_is_deterministic_and_read_only() -> None:
    bundle = _make_bundle()
    features = _inference_features()

    original_features = features.copy(
        deep=True
    )
    original_vocabularies = dict(
        bundle.encoder.category_vocabularies
    )
    probabilities_before = (
        bundle.predict_fraud_probabilities(
            features
        )
    )

    first = extract_lightgbm_contributions(
        bundle,
        features,
    )
    second = extract_lightgbm_contributions(
        bundle,
        features,
    )

    np.testing.assert_array_equal(
        first.shap_values_raw,
        second.shap_values_raw,
    )
    np.testing.assert_array_equal(
        first.expected_values_raw,
        second.expected_values_raw,
    )

    assert_frame_equal(
        features,
        original_features,
    )
    assert (
        bundle.encoder.category_vocabularies
        == original_vocabularies
    )

    probabilities_after = (
        bundle.predict_fraud_probabilities(
            features
        )
    )

    np.testing.assert_array_equal(
        probabilities_after,
        probabilities_before,
    )

    with pytest.raises(
        ValueError,
        match="read-only",
    ):
        first.shap_values_raw[
            0,
            0,
        ] = 0.0

    with pytest.raises(
        ValueError,
        match="read-only",
    ):
        first.expected_values_raw[0] = 0.0


class _FakeBooster:
    """Return a configured native-prediction payload."""

    def __init__(
        self,
        output: object,
    ) -> None:
        self.output = output

    def predict(
        self,
        *_: object,
        **__: object,
    ) -> object:
        return self.output


class _FakeModel:
    """Satisfy the model-bundle contract for failure tests."""

    def __init__(
        self,
        *,
        booster: object | None,
    ) -> None:
        self.feature_name_ = list(
            FEATURE_CONTRACT.feature_columns
        )

        if booster is not None:
            self.booster_ = booster

    def predict_proba(
        self,
        features: pd.DataFrame,
        **_: object,
    ) -> np.ndarray:
        return np.tile(
            np.array(
                [
                    0.5,
                    0.5,
                ]
            ),
            (
                len(features),
                1,
            ),
        )


def test_rejects_model_without_fitted_booster() -> None:
    bundle = replace(
        _make_bundle(),
        model=_FakeModel(
            booster=None
        ),
    )

    with pytest.raises(
        ValueError,
        match="fitted LightGBM booster",
    ):
        extract_lightgbm_contributions(
            bundle,
            _inference_features(),
        )


def test_rejects_unexpected_native_shape() -> None:
    bundle = replace(
        _make_bundle(),
        model=_FakeModel(
            booster=_FakeBooster(
                np.zeros(
                    (
                        3,
                        2,
                    )
                )
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected contribution shape",
    ):
        extract_lightgbm_contributions(
            bundle,
            _inference_features(),
        )


def test_rejects_non_finite_native_output() -> None:
    output = np.zeros(
        (
            3,
            3,
        ),
        dtype="float64",
    )
    output[
        1,
        0,
    ] = np.nan

    bundle = replace(
        _make_bundle(),
        model=_FakeModel(
            booster=_FakeBooster(
                output
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="non-finite contributions",
    ):
        extract_lightgbm_contributions(
            bundle,
            _inference_features(),
        )


def test_requires_exact_frozen_feature_order() -> None:
    reordered_features = (
        _inference_features().loc[
            :,
            [
                "numeric_feature",
                "category_feature",
            ],
        ]
    )

    with pytest.raises(
        ValueError,
        match="expected feature order",
    ):
        extract_lightgbm_contributions(
            _make_bundle(),
            reordered_features,
        )


def test_rejects_empty_feature_frame() -> None:
    features = _inference_features().iloc[
        0:0
    ]

    with pytest.raises(
        ValueError,
        match="at least one row",
    ):
        extract_lightgbm_contributions(
            _make_bundle(),
            features,
        )
