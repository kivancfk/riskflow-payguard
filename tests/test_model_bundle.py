"""Tests for versioned PayGuard model bundles."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.model_bundle import (
    MODEL_BUNDLE_SCHEMA_VERSION,
    ModelBundle,
    build_model_bundle,
    load_model_bundle,
    save_model_bundle,
)
from src.model_data import FeatureContract
from src.preprocessing import CategoricalEncoder


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
    """Return a small deterministic training frame."""
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


def _make_bundle(
    *,
    model_version: str = "baseline-test-v1",
) -> ModelBundle:
    """Train and package a tiny LightGBM model."""
    features = _training_features()
    target = _training_target()

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
        target,
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
            "probability_metrics": {
                "pr_auc": 0.80,
            }
        },
        test_metrics={
            "probability_metrics": {
                "pr_auc": 0.75,
            }
        },
        dataset_manifest={
            "schema_version": 1,
        },
        model_version=model_version,
        created_at_utc=(
            "2026-08-02T15:00:00+00:00"
        ),
    )


def _inference_features() -> pd.DataFrame:
    """Return known, missing, and unseen categories."""
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
        }
    )


def test_model_bundle_round_trip_preserves_predictions(
    tmp_path: Path,
) -> None:
    bundle = _make_bundle()
    inference_features = _inference_features()

    expected_probabilities = (
        bundle.predict_fraud_probabilities(
            inference_features
        )
    )

    artifact_path = (
        tmp_path / "payguard.joblib"
    )

    saved_path = save_model_bundle(
        bundle,
        artifact_path,
    )
    restored = load_model_bundle(
        saved_path
    )

    actual_probabilities = (
        restored.predict_fraud_probabilities(
            inference_features
        )
    )

    assert saved_path == artifact_path.resolve()
    assert restored.schema_version == (
        MODEL_BUNDLE_SCHEMA_VERSION
    )
    assert restored.model_version == (
        "baseline-test-v1"
    )
    assert restored.best_iteration == 10
    assert restored.test_metrics == {
        "probability_metrics": {
            "pr_auc": 0.75,
        }
    }
    assert restored.feature_columns == (
        FEATURE_CONTRACT.feature_columns
    )

    np.testing.assert_allclose(
        actual_probabilities,
        expected_probabilities,
        rtol=0.0,
        atol=0.0,
    )

    assert not list(
        tmp_path.glob(
            ".payguard.joblib.*.tmp"
        )
    )


def test_save_rejects_existing_artifact_without_overwrite(
    tmp_path: Path,
) -> None:
    artifact_path = (
        tmp_path / "payguard.joblib"
    )

    save_model_bundle(
        _make_bundle(),
        artifact_path,
    )

    with pytest.raises(
        FileExistsError,
        match="overwrite=True",
    ):
        save_model_bundle(
            _make_bundle(),
            artifact_path,
        )


def test_save_overwrites_existing_artifact(
    tmp_path: Path,
) -> None:
    artifact_path = (
        tmp_path / "payguard.joblib"
    )

    save_model_bundle(
        _make_bundle(
            model_version="baseline-v1"
        ),
        artifact_path,
    )
    save_model_bundle(
        _make_bundle(
            model_version="baseline-v2"
        ),
        artifact_path,
        overwrite=True,
    )

    restored = load_model_bundle(
        artifact_path
    )

    assert restored.model_version == (
        "baseline-v2"
    )


def test_invalid_overwrite_preserves_existing_artifact(
    tmp_path: Path,
) -> None:
    artifact_path = (
        tmp_path / "payguard.joblib"
    )
    original_bundle = _make_bundle()

    save_model_bundle(
        original_bundle,
        artifact_path,
    )

    invalid_bundle = replace(
        original_bundle,
        best_iteration=0,
    )

    with pytest.raises(
        ValueError,
        match="best_iteration",
    ):
        save_model_bundle(
            invalid_bundle,
            artifact_path,
            overwrite=True,
        )

    restored = load_model_bundle(
        artifact_path
    )

    assert restored.model_version == (
        original_bundle.model_version
    )
    assert restored.best_iteration == 10


def test_load_rejects_non_bundle_artifact(
    tmp_path: Path,
) -> None:
    artifact_path = (
        tmp_path / "invalid.joblib"
    )

    joblib.dump(
        {"not": "a model bundle"},
        artifact_path,
    )

    with pytest.raises(
        ValueError,
        match="ModelBundle",
    ):
        load_model_bundle(
            artifact_path
        )


def test_load_rejects_missing_artifact(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):
        load_model_bundle(
            tmp_path / "missing.joblib"
        )


def test_prediction_rejects_wrong_feature_order(
) -> None:
    bundle = _make_bundle()

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
        bundle.predict_fraud_probabilities(
            reordered_features
        )
