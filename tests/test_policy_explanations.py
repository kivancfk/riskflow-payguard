"""Tests for calibrated policy predictions with explanations."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from pandas.testing import assert_frame_equal

from src.calibration import (
    fit_and_select_calibrator,
)
from src.explainability import (
    EXPLANATION_VERSION,
    REASON_CODE_VERSION,
)
from src.model_bundle import (
    ModelBundle,
    build_model_bundle,
)
from src.model_data import FeatureContract
from src.policy import (
    PolicyConstraints,
    PolicyCostAssumptions,
    search_policy_thresholds,
)
from src.policy_bundle import (
    CalibratedPolicyBundle,
    build_calibrated_policy_bundle,
)
from src.policy_explanations import (
    predict_policy_with_explanations,
)
from src.preprocessing import (
    CategoricalEncoder,
)
from src.shap_reconstruction import (
    RECONSTRUCTION_ABSOLUTE_TOLERANCE,
)


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
    """Return deterministic mixed model features."""

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


def _make_baseline_bundle() -> ModelBundle:
    """Train and package a tiny deterministic baseline."""

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
            "n_estimators": 12,
            "seed": 7,
        },
        scale_pos_weight=1.4,
        best_iteration=12,
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


def _development_arrays() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return deterministic calibration and threshold data."""

    scores = np.repeat(
        np.array(
            [
                0.1,
                0.3,
                0.6,
                0.9,
            ],
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


def _make_policy_bundle() -> CalibratedPolicyBundle:
    """Build a complete synthetic calibrated policy."""

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
        policy_version=(
            "calibrated-policy-test-v1"
        ),
        created_at_utc=(
            "2026-08-02T16:00:00+00:00"
        ),
    )


def _inference_features() -> pd.DataFrame:
    """Return known, unseen, and missing inputs."""

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
        },
        index=[
            101,
            102,
            103,
        ],
    )


def test_preserves_existing_calibrated_policy_outputs() -> None:
    bundle = _make_policy_bundle()
    features = _inference_features()

    existing = bundle.predict(
        features
    )
    explained = (
        predict_policy_with_explanations(
            bundle,
            [
                9001,
                9002,
                9003,
            ],
            features,
        )
    )

    np.testing.assert_array_equal(
        explained.raw_scores,
        existing.raw_scores,
    )
    np.testing.assert_array_equal(
        explained.calibrated_probabilities,
        existing.calibrated_probabilities,
    )
    np.testing.assert_array_equal(
        explained.decisions,
        existing.decisions,
    )

    assert explained.transaction_ids == (
        9001,
        9002,
        9003,
    )
    assert len(
        explained.explanations
    ) == 3

    for row_position, record in enumerate(
        explained.explanations
    ):
        assert record.model_version == (
            "baseline-test-v1"
        )
        assert record.policy_version == (
            "calibrated-policy-test-v1"
        )
        assert record.explanation_version == (
            EXPLANATION_VERSION
        )
        assert record.reason_code_version == (
            REASON_CODE_VERSION
        )
        assert record.raw_model_score == (
            existing.raw_scores[
                row_position
            ]
        )
        assert (
            record.calibrated_probability
            == existing.calibrated_probabilities[
                row_position
            ]
        )
        assert record.decision == (
            existing.decisions[
                row_position
            ]
        )
        assert (
            record.margin_reconstruction_error
            <= RECONSTRUCTION_ABSOLUTE_TOLERANCE
        )
        assert (
            record.score_reconstruction_error
            <= RECONSTRUCTION_ABSOLUTE_TOLERANCE
        )
        assert record.reason_codes == tuple(
            reason.code
            for reason in record.reasons
        )


def test_generation_is_deterministic_and_read_only() -> None:
    bundle = _make_policy_bundle()
    features = _inference_features()

    original_features = features.copy(
        deep=True
    )
    original_vocabularies = dict(
        bundle.baseline_bundle
        .encoder.category_vocabularies
    )
    policy_before = bundle.predict(
        features
    )

    first = predict_policy_with_explanations(
        bundle,
        [
            "tx-1",
            "tx-2",
            "tx-3",
        ],
        features,
    )
    second = predict_policy_with_explanations(
        bundle,
        [
            "tx-1",
            "tx-2",
            "tx-3",
        ],
        features,
    )

    np.testing.assert_array_equal(
        first.raw_scores,
        second.raw_scores,
    )
    np.testing.assert_array_equal(
        first.calibrated_probabilities,
        second.calibrated_probabilities,
    )
    np.testing.assert_array_equal(
        first.decisions,
        second.decisions,
    )
    assert (
        first.explanations
        == second.explanations
    )

    assert not first.raw_scores.flags.writeable
    assert not (
        first.calibrated_probabilities
        .flags.writeable
    )
    assert not first.decisions.flags.writeable

    assert_frame_equal(
        features,
        original_features,
    )
    assert (
        bundle.baseline_bundle
        .encoder.category_vocabularies
        == original_vocabularies
    )

    policy_after = bundle.predict(
        features
    )

    np.testing.assert_array_equal(
        policy_after.raw_scores,
        policy_before.raw_scores,
    )
    np.testing.assert_array_equal(
        policy_after.calibrated_probabilities,
        policy_before.calibrated_probabilities,
    )
    np.testing.assert_array_equal(
        policy_after.decisions,
        policy_before.decisions,
    )


def test_top_k_is_applied_per_direction() -> None:
    result = predict_policy_with_explanations(
        _make_policy_bundle(),
        [
            1,
            2,
            3,
        ],
        _inference_features(),
        top_k=1,
    )

    for record in result.explanations:
        assert len(
            record.top_positive_contributions
        ) <= 1
        assert len(
            record.top_negative_contributions
        ) <= 1


def test_explanation_record_is_json_compatible() -> None:
    result = predict_policy_with_explanations(
        _make_policy_bundle(),
        [
            "tx-1",
            "tx-2",
            "tx-3",
        ],
        _inference_features(),
    )

    serialized = asdict(
        result.explanations[0]
    )

    json.dumps(
        serialized
    )


@pytest.mark.parametrize(
    (
        "transaction_ids",
        "message",
    ),
    [
        (
            [
                "tx-1",
                "tx-2",
            ],
            "same length",
        ),
        (
            [
                "tx-1",
                "tx-1",
                "tx-3",
            ],
            "must be unique",
        ),
        (
            [
                "tx-1",
                " ",
                "tx-3",
            ],
            "non-empty string",
        ),
        (
            [
                1,
                True,
                3,
            ],
            "string or integer",
        ),
        (
            "tx-1",
            "iterable of identifiers",
        ),
    ],
)
def test_rejects_invalid_transaction_ids(
    transaction_ids: object,
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        predict_policy_with_explanations(
            _make_policy_bundle(),
            transaction_ids,  # type: ignore[arg-type]
            _inference_features(),
        )


def test_rejects_wrong_feature_order() -> None:
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
        predict_policy_with_explanations(
            _make_policy_bundle(),
            [
                1,
                2,
                3,
            ],
            reordered,
        )
