"""Tests for LightGBM SHAP score reconstruction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from pandas.testing import assert_frame_equal

from src.lightgbm_shap import (
    LightGBMContributionMatrix,
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
from src.shap_reconstruction import (
    RECONSTRUCTION_ABSOLUTE_TOLERANCE,
    lightgbm_binary_sigmoid,
    reconstruct_lightgbm_scores,
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
    """Train and package a tiny deterministic model."""

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


def test_reconstructs_native_margins_and_probabilities() -> None:
    bundle = _make_bundle()
    features = _inference_features()

    result = reconstruct_lightgbm_scores(
        bundle,
        features,
    )

    encoded_features = (
        bundle.encoder.transform(
            features
        )
    )
    direct_raw_margins = (
        bundle.model.booster_.predict(
            encoded_features,
            num_iteration=(
                bundle.best_iteration
            ),
            raw_score=True,
            validate_features=True,
        )
    )
    direct_probabilities = (
        bundle.predict_fraud_probabilities(
            features
        )
    )

    assert result.row_count == 3
    assert result.feature_names == (
        FEATURE_CONTRACT.feature_columns
    )
    assert result.absolute_tolerance == (
        RECONSTRUCTION_ABSOLUTE_TOLERANCE
    )

    np.testing.assert_allclose(
        result.raw_model_margins,
        direct_raw_margins,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.raw_model_scores,
        direct_probabilities,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.reconstructed_raw_margins,
        result.raw_model_margins,
        rtol=0.0,
        atol=(
            RECONSTRUCTION_ABSOLUTE_TOLERANCE
        ),
    )
    np.testing.assert_allclose(
        result.reconstructed_raw_model_scores,
        result.raw_model_scores,
        rtol=0.0,
        atol=(
            RECONSTRUCTION_ABSOLUTE_TOLERANCE
        ),
    )

    assert (
        result.max_margin_reconstruction_error
        <= RECONSTRUCTION_ABSOLUTE_TOLERANCE
    )
    assert (
        result.max_score_reconstruction_error
        <= RECONSTRUCTION_ABSOLUTE_TOLERANCE
    )


def test_sigmoid_is_stable_for_extreme_margins() -> None:
    probabilities = (
        lightgbm_binary_sigmoid(
            [
                -1000.0,
                0.0,
                1000.0,
            ]
        )
    )

    assert probabilities.tolist() == [
        0.0,
        0.5,
        1.0,
    ]
    assert not probabilities.flags.writeable


def test_reconstruction_is_deterministic_and_read_only() -> None:
    bundle = _make_bundle()
    features = _inference_features()
    original_features = features.copy(
        deep=True
    )

    first = reconstruct_lightgbm_scores(
        bundle,
        features,
    )
    second = reconstruct_lightgbm_scores(
        bundle,
        features,
    )

    for field_name in (
        "raw_model_margins",
        "raw_model_scores",
        "expected_values_raw",
        "shap_sums_raw",
        "reconstructed_raw_margins",
        "reconstructed_raw_model_scores",
        "margin_reconstruction_errors",
        "score_reconstruction_errors",
    ):
        first_values = getattr(
            first,
            field_name,
        )
        second_values = getattr(
            second,
            field_name,
        )

        np.testing.assert_array_equal(
            first_values,
            second_values,
        )
        assert not first_values.flags.writeable

    assert_frame_equal(
        features,
        original_features,
    )

    with pytest.raises(
        ValueError,
        match="read-only",
    ):
        first.raw_model_margins[0] = 0.0


def test_accepts_precomputed_contributions() -> None:
    bundle = _make_bundle()
    features = _inference_features()
    contributions = (
        extract_lightgbm_contributions(
            bundle,
            features,
        )
    )

    result = reconstruct_lightgbm_scores(
        bundle,
        features,
        contributions=contributions,
    )

    np.testing.assert_allclose(
        result.expected_values_raw,
        contributions.expected_values_raw,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.shap_sums_raw,
        contributions.shap_values_raw.sum(
            axis=1
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_rejects_mismatched_feature_names() -> None:
    bundle = _make_bundle()
    features = _inference_features()
    contributions = (
        extract_lightgbm_contributions(
            bundle,
            features,
        )
    )
    mismatched = LightGBMContributionMatrix(
        feature_names=(
            "numeric_feature",
            "category_feature",
        ),
        shap_values_raw=(
            contributions.shap_values_raw
        ),
        expected_values_raw=(
            contributions.expected_values_raw
        ),
    )

    with pytest.raises(
        ValueError,
        match="feature names do not match",
    ):
        reconstruct_lightgbm_scores(
            bundle,
            features,
            contributions=mismatched,
        )


def test_rejects_misaligned_contribution_rows() -> None:
    bundle = _make_bundle()
    features = _inference_features()
    contributions = (
        extract_lightgbm_contributions(
            bundle,
            features,
        )
    )
    shortened = LightGBMContributionMatrix(
        feature_names=(
            contributions.feature_names
        ),
        shap_values_raw=(
            contributions.shap_values_raw[
                :2
            ]
        ),
        expected_values_raw=(
            contributions.expected_values_raw[
                :2
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="rows do not align",
    ):
        reconstruct_lightgbm_scores(
            bundle,
            features,
            contributions=shortened,
        )


def test_rejects_failed_margin_reconstruction() -> None:
    bundle = _make_bundle()
    features = _inference_features()
    contributions = (
        extract_lightgbm_contributions(
            bundle,
            features,
        )
    )

    altered_expected_values = (
        contributions.expected_values_raw.copy()
    )
    altered_expected_values[0] += 0.01

    altered = LightGBMContributionMatrix(
        feature_names=(
            contributions.feature_names
        ),
        shap_values_raw=(
            contributions.shap_values_raw
        ),
        expected_values_raw=(
            altered_expected_values
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="raw-margin reconstruction exceeded",
    ):
        reconstruct_lightgbm_scores(
            bundle,
            features,
            contributions=altered,
        )


class _ShiftedProbabilityModel:
    """Preserve the booster while returning altered probabilities."""

    def __init__(
        self,
        model: LGBMClassifier,
    ) -> None:
        self._model = model
        self.booster_ = model.booster_
        self.feature_name_ = list(
            model.feature_name_
        )

    def predict_proba(
        self,
        features: pd.DataFrame,
        **keyword_arguments: object,
    ) -> np.ndarray:
        probabilities = (
            self._model.predict_proba(
                features,
                **keyword_arguments,
            )
        ).copy()

        shifted_positive = np.clip(
            probabilities[
                :,
                1,
            ]
            + 0.01,
            0.0,
            1.0,
        )

        probabilities[
            :,
            1,
        ] = shifted_positive
        probabilities[
            :,
            0,
        ] = 1.0 - shifted_positive

        return probabilities


def test_rejects_failed_score_reconstruction() -> None:
    original_bundle = _make_bundle()
    shifted_bundle = replace(
        original_bundle,
        model=_ShiftedProbabilityModel(
            original_bundle.model
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="raw-score reconstruction exceeded",
    ):
        reconstruct_lightgbm_scores(
            shifted_bundle,
            _inference_features(),
        )


@pytest.mark.parametrize(
    "absolute_tolerance",
    [
        0.0,
        -1e-8,
        float("nan"),
        True,
    ],
)
def test_rejects_invalid_tolerance(
    absolute_tolerance: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="absolute_tolerance",
    ):
        reconstruct_lightgbm_scores(
            _make_bundle(),
            _inference_features(),
            absolute_tolerance=(
                absolute_tolerance
            ),  # type: ignore[arg-type]
        )
