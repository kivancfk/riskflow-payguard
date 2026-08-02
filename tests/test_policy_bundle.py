"""Tests for calibrated PayGuard policy bundles."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.calibration import (
    fit_and_select_calibrator,
)
from src.model_bundle import (
    ModelBundle,
    build_model_bundle,
    save_model_bundle,
)
from src.model_data import FeatureContract
from src.policy import (
    PolicyConstraints,
    PolicyCostAssumptions,
    search_policy_thresholds,
)
from src.policy_bundle import (
    POLICY_BUNDLE_SCHEMA_VERSION,
    CalibratedPolicyBundle,
    build_calibrated_policy_bundle,
    load_calibrated_policy_bundle,
    save_calibrated_policy_bundle,
)
from src.preprocessing import CategoricalEncoder


FEATURE_CONTRACT = FeatureContract(
    feature_columns=(
        "category_feature",
        "numeric_feature",
        "TransactionAmt",
    ),
    categorical_features=(
        "category_feature",
    ),
    numerical_features=(
        "numeric_feature",
        "TransactionAmt",
    ),
)


def _training_features() -> pd.DataFrame:
    """Return deterministic mixed-type training data."""

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
            "TransactionAmt": [
                10.0,
                20.0,
                30.0,
                40.0,
                80.0,
                90.0,
                25.0,
                60.0,
                100.0,
                70.0,
                15.0,
                110.0,
            ],
        }
    )


def _training_target() -> pd.Series:
    """Return a target aligned with the training data."""

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


def _make_baseline_bundle() -> ModelBundle:
    """Train and package a tiny baseline model."""

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
        n_estimators=12,
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
            "n_estimators": 12,
            "seed": 7,
        },
        scale_pos_weight=1.4,
        best_iteration=12,
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
        model_version="baseline-test-v1",
        created_at_utc=(
            "2026-08-02T15:00:00+00:00"
        ),
    )


def _development_arrays() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return deterministic calibration and policy arrays."""

    scores = np.repeat(
        np.array(
            [0.1, 0.3, 0.6, 0.9],
            dtype="float64",
        ),
        10,
    )

    target = np.array(
        (
            [1] + [0] * 9
            + [1] * 3 + [0] * 7
            + [1] * 6 + [0] * 4
            + [1] * 9 + [0]
        ),
        dtype="int8",
    )

    amounts = np.linspace(
        10.0,
        400.0,
        len(target),
    )

    return (
        target,
        scores,
        amounts,
    )


def _make_policy_bundle(
    *,
    policy_version: str = (
        "calibrated-policy-test-v1"
    ),
) -> CalibratedPolicyBundle:
    """Build one complete synthetic policy bundle."""

    baseline_bundle = (
        _make_baseline_bundle()
    )
    target, scores, amounts = (
        _development_arrays()
    )

    calibration_selection = (
        fit_and_select_calibrator(
            target,
            scores,
            target,
            scores,
        )
    )

    policy_search_result = (
        search_policy_thresholds(
            target,
            scores,
            amounts,
            cost_assumptions=(
                PolicyCostAssumptions(
                    cost_per_review=2.0,
                    cost_per_legitimate_block=5.0,
                    fraud_loss_multiplier=1.0,
                    review_fraud_capture_rate=0.5,
                )
            ),
            constraints=(
                PolicyConstraints()
            ),
            quantile_count=5,
        )
    )

    return build_calibrated_policy_bundle(
        baseline_bundle=baseline_bundle,
        calibration_selection=(
            calibration_selection
        ),
        policy_search_result=(
            policy_search_result
        ),
        development_metadata={
            "test_evaluation": None,
        },
        calibration_metrics={
            "selected_method": (
                calibration_selection
                .selected_method
            ),
        },
        drift_diagnostics={
            "reference_segment": (
                "calibration_fit"
            ),
        },
        dataset_manifest=(
            baseline_bundle.dataset_manifest
        ),
        policy_version=policy_version,
        created_at_utc=(
            "2026-08-02T16:00:00+00:00"
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
            "TransactionAmt": [
                20.0,
                75.0,
                50.0,
            ],
        }
    )


def test_policy_bundle_round_trip_preserves_inference(
    tmp_path: Path,
) -> None:
    bundle = _make_policy_bundle()
    features = _inference_features()

    expected = bundle.predict(features)

    artifact_path = (
        tmp_path / "policy.joblib"
    )

    saved_path = (
        save_calibrated_policy_bundle(
            bundle,
            artifact_path,
        )
    )
    restored = (
        load_calibrated_policy_bundle(
            saved_path
        )
    )

    actual = restored.predict(features)

    assert saved_path == artifact_path.resolve()
    assert restored.schema_version == (
        POLICY_BUNDLE_SCHEMA_VERSION
    )
    assert restored.policy_version == (
        "calibrated-policy-test-v1"
    )
    assert restored.baseline_model_version == (
        "baseline-test-v1"
    )

    np.testing.assert_allclose(
        actual.raw_scores,
        expected.raw_scores,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual.calibrated_probabilities,
        expected.calibrated_probabilities,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        actual.decisions,
        expected.decisions,
    )

    assert not list(
        tmp_path.glob(
            ".policy.joblib.*.tmp"
        )
    )


def test_policy_save_does_not_modify_baseline_artifact(
    tmp_path: Path,
) -> None:
    baseline_bundle = (
        _make_baseline_bundle()
    )
    baseline_path = (
        tmp_path / "baseline.joblib"
    )

    save_model_bundle(
        baseline_bundle,
        baseline_path,
    )

    baseline_bytes = (
        baseline_path.read_bytes()
    )

    save_calibrated_policy_bundle(
        _make_policy_bundle(),
        tmp_path / "policy.joblib",
    )

    assert baseline_path.read_bytes() == (
        baseline_bytes
    )


def test_policy_save_rejects_existing_artifact(
    tmp_path: Path,
) -> None:
    artifact_path = (
        tmp_path / "policy.joblib"
    )

    save_calibrated_policy_bundle(
        _make_policy_bundle(),
        artifact_path,
    )

    with pytest.raises(
        FileExistsError,
        match="overwrite=True",
    ):
        save_calibrated_policy_bundle(
            _make_policy_bundle(),
            artifact_path,
        )


def test_invalid_overwrite_preserves_policy(
    tmp_path: Path,
) -> None:
    artifact_path = (
        tmp_path / "policy.joblib"
    )
    original = _make_policy_bundle()

    save_calibrated_policy_bundle(
        original,
        artifact_path,
    )

    invalid = replace(
        original,
        baseline_model_version=(
            "incorrect-baseline-version"
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        save_calibrated_policy_bundle(
            invalid,
            artifact_path,
            overwrite=True,
        )

    restored = (
        load_calibrated_policy_bundle(
            artifact_path
        )
    )

    assert restored.policy_version == (
        original.policy_version
    )
    assert restored.baseline_model_version == (
        original.baseline_model_version
    )


def test_load_rejects_non_policy_artifact(
    tmp_path: Path,
) -> None:
    artifact_path = (
        tmp_path / "invalid.joblib"
    )

    joblib.dump(
        {"not": "a policy bundle"},
        artifact_path,
    )

    with pytest.raises(
        ValueError,
        match="CalibratedPolicyBundle",
    ):
        load_calibrated_policy_bundle(
            artifact_path
        )


def test_policy_prediction_rejects_wrong_feature_order(
) -> None:
    bundle = _make_policy_bundle()

    reordered = _inference_features().loc[
        :,
        [
            "numeric_feature",
            "category_feature",
            "TransactionAmt",
        ],
    ]

    with pytest.raises(
        ValueError,
        match="expected feature order",
    ):
        bundle.predict(reordered)
