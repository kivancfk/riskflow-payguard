"""Integration checks for frozen Phase 4 model and policy artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.explainability import (
    EXPLANATION_VERSION,
    REASON_CODE_VERSION,
)
from src.model_bundle import (
    ModelBundle,
    load_model_bundle,
)
from src.model_data import (
    load_validation_dataset,
)
from src.policy_bundle import (
    CalibratedPolicyBundle,
    load_calibrated_policy_bundle,
)
from src.policy_explanations import (
    ExplainedPolicyPrediction,
    predict_policy_with_explanations,
)
from src.shap_reconstruction import (
    RECONSTRUCTION_ABSOLUTE_TOLERANCE,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

BASELINE_ARTIFACT = (
    PROJECT_ROOT
    / "models/payguard_baseline.joblib"
)
POLICY_ARTIFACT = (
    PROJECT_ROOT
    / "models/payguard_calibrated_policy.joblib"
)
PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data/processed"
)

VALIDATION_ARTIFACT = (
    PROCESSED_DATA_DIR
    / "validation.parquet"
)
FEATURE_METADATA_ARTIFACT = (
    PROCESSED_DATA_DIR
    / "feature_metadata.csv"
)
DATASET_MANIFEST_ARTIFACT = (
    PROCESSED_DATA_DIR
    / "dataset_manifest.json"
)

EXPECTED_POLICY_SHA256 = (
    "5d53f23719ae891ecc24585393585765"
    "aa7fc0900ab38f95e37f59c18fe6c90f"
)

EXPECTED_BASELINE_VERSION = "baseline-v1"
EXPECTED_POLICY_VERSION = (
    "calibrated-policy-v1"
)
EXPECTED_CALIBRATION_METHOD = "sigmoid"

EXPECTED_REVIEW_THRESHOLD = (
    0.16255069862369795
)
EXPECTED_BLOCK_THRESHOLD = (
    0.8509223095305902
)

EXPECTED_FEATURE_COUNT = 63
EXPECTED_CATEGORICAL_FEATURE_COUNT = 29
EXPECTED_NUMERICAL_FEATURE_COUNT = 34

SAMPLE_ROW_COUNT = 6
SINGLE_ROW_PARITY_COUNT = 3


def _sha256_file(
    path: Path,
) -> str:
    """Return a streaming SHA-256 fingerprint."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def _booster_sha256(
    bundle: ModelBundle,
) -> str:
    """Fingerprint the fitted LightGBM tree representation."""

    booster = getattr(
        bundle.model,
        "booster_",
        None,
    )
    model_to_string = getattr(
        booster,
        "model_to_string",
        None,
    )

    if not callable(
        model_to_string
    ):
        raise AssertionError(
            "Frozen model does not expose "
            "a fitted LightGBM booster"
        )

    model_text = model_to_string()

    return hashlib.sha256(
        model_text.encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class FrozenArtifactContext:
    """Loaded local artifacts and a small validation sample."""

    baseline_bundle: ModelBundle
    policy_bundle: CalibratedPolicyBundle
    transaction_ids: tuple[int, ...]
    features: pd.DataFrame
    baseline_artifact_sha256: str
    policy_artifact_sha256: str


@pytest.fixture(
    scope="module",
)
def frozen_artifacts(
) -> FrozenArtifactContext:
    """Load local frozen artifacts when available."""

    required_paths = (
        BASELINE_ARTIFACT,
        POLICY_ARTIFACT,
        VALIDATION_ARTIFACT,
        FEATURE_METADATA_ARTIFACT,
        DATASET_MANIFEST_ARTIFACT,
    )

    missing_paths = [
        path
        for path in required_paths
        if not path.is_file()
    ]

    if missing_paths:
        relative_paths = ", ".join(
            str(
                path.relative_to(
                    PROJECT_ROOT
                )
            )
            for path in missing_paths
        )

        pytest.skip(
            "Local Phase 4 artifact validation "
            f"requires: {relative_paths}"
        )

    baseline_artifact_sha256 = (
        _sha256_file(
            BASELINE_ARTIFACT
        )
    )
    policy_artifact_sha256 = (
        _sha256_file(
            POLICY_ARTIFACT
        )
    )

    if (
        policy_artifact_sha256
        != EXPECTED_POLICY_SHA256
    ):
        raise AssertionError(
            "Frozen policy artifact SHA-256 "
            "does not match the Phase 3 "
            "completion record"
        )

    baseline_bundle = (
        load_model_bundle(
            BASELINE_ARTIFACT
        )
    )
    policy_bundle = (
        load_calibrated_policy_bundle(
            POLICY_ARTIFACT
        )
    )
    validation_bundle = (
        load_validation_dataset(
            PROCESSED_DATA_DIR
        )
    )

    validation = (
        validation_bundle.validation
    )

    if len(
        validation.features
    ) < SAMPLE_ROW_COUNT:
        raise AssertionError(
            "Validation data does not contain "
            "enough rows for the Phase 4 check"
        )

    transaction_ids = tuple(
        int(transaction_id)
        for transaction_id in (
            validation.transaction_ids
            .iloc[
                :SAMPLE_ROW_COUNT
            ]
            .tolist()
        )
    )

    features = (
        validation.features
        .iloc[
            :SAMPLE_ROW_COUNT
        ]
        .copy(
            deep=True
        )
    )

    return FrozenArtifactContext(
        baseline_bundle=(
            baseline_bundle
        ),
        policy_bundle=(
            policy_bundle
        ),
        transaction_ids=(
            transaction_ids
        ),
        features=features,
        baseline_artifact_sha256=(
            baseline_artifact_sha256
        ),
        policy_artifact_sha256=(
            policy_artifact_sha256
        ),
    )


def _predict_explained(
    context: FrozenArtifactContext,
    *,
    bundle: (
        CalibratedPolicyBundle
        | None
    ) = None,
) -> ExplainedPolicyPrediction:
    """Generate explanations for the fixed validation sample."""

    resolved_bundle = (
        bundle
        if bundle is not None
        else context.policy_bundle
    )

    return predict_policy_with_explanations(
        resolved_bundle,
        context.transaction_ids,
        context.features.copy(
            deep=True
        ),
    )


def test_frozen_artifact_identity_and_configuration(
    frozen_artifacts: FrozenArtifactContext,
) -> None:
    """Require the documented Phase 2 and Phase 3 configuration."""

    context = frozen_artifacts
    baseline = (
        context.baseline_bundle
    )
    policy = (
        context.policy_bundle
    )

    assert (
        context.policy_artifact_sha256
        == EXPECTED_POLICY_SHA256
    )

    assert baseline.model_version == (
        EXPECTED_BASELINE_VERSION
    )
    assert policy.baseline_model_version == (
        EXPECTED_BASELINE_VERSION
    )
    assert policy.policy_version == (
        EXPECTED_POLICY_VERSION
    )

    assert (
        policy.calibration_selection
        .selected_method
        == EXPECTED_CALIBRATION_METHOD
    )

    assert (
        policy.thresholds.review_threshold
        == EXPECTED_REVIEW_THRESHOLD
    )
    assert (
        policy.thresholds.block_threshold
        == EXPECTED_BLOCK_THRESHOLD
    )

    assert len(
        baseline.feature_columns
    ) == EXPECTED_FEATURE_COUNT
    assert len(
        baseline.categorical_features
    ) == (
        EXPECTED_CATEGORICAL_FEATURE_COUNT
    )
    assert len(
        baseline.numerical_features
    ) == (
        EXPECTED_NUMERICAL_FEATURE_COUNT
    )

    assert (
        policy.baseline_bundle
        .feature_columns
        == baseline.feature_columns
    )
    assert (
        policy.baseline_bundle
        .categorical_features
        == baseline.categorical_features
    )
    assert (
        policy.baseline_bundle
        .numerical_features
        == baseline.numerical_features
    )
    assert (
        policy.baseline_bundle
        .best_iteration
        == baseline.best_iteration
    )
    assert (
        policy.baseline_bundle
        .dataset_manifest
        == baseline.dataset_manifest
    )


def test_explanations_preserve_policy_outputs_and_artifacts(
    frozen_artifacts: FrozenArtifactContext,
) -> None:
    """Require prediction parity and unchanged artifact/model state."""

    context = frozen_artifacts

    features_before = (
        context.features.copy(
            deep=True
        )
    )
    baseline_vocabularies_before = (
        copy.deepcopy(
            context.baseline_bundle
            .encoder.category_vocabularies
        )
    )
    embedded_vocabularies_before = (
        copy.deepcopy(
            context.policy_bundle
            .baseline_bundle
            .encoder.category_vocabularies
        )
    )

    baseline_booster_before = (
        _booster_sha256(
            context.baseline_bundle
        )
    )
    embedded_booster_before = (
        _booster_sha256(
            context.policy_bundle
            .baseline_bundle
        )
    )

    reference_policy_prediction = (
        context.policy_bundle.predict(
            context.features
        )
    )
    separate_baseline_scores = (
        context.baseline_bundle
        .predict_fraud_probabilities(
            context.features
        )
    )

    explained = (
        _predict_explained(
            context
        )
    )

    np.testing.assert_array_equal(
        separate_baseline_scores,
        reference_policy_prediction
        .raw_scores,
    )
    np.testing.assert_array_equal(
        explained.raw_scores,
        reference_policy_prediction
        .raw_scores,
    )
    np.testing.assert_array_equal(
        explained.calibrated_probabilities,
        reference_policy_prediction
        .calibrated_probabilities,
    )
    np.testing.assert_array_equal(
        explained.decisions,
        reference_policy_prediction
        .decisions,
    )

    assert explained.transaction_ids == (
        context.transaction_ids
    )

    for row_position, record in enumerate(
        explained.explanations
    ):
        assert record.transaction_id == (
            context.transaction_ids[
                row_position
            ]
        )
        assert record.model_version == (
            EXPECTED_BASELINE_VERSION
        )
        assert record.policy_version == (
            EXPECTED_POLICY_VERSION
        )
        assert (
            record.explanation_version
            == EXPLANATION_VERSION
        )
        assert (
            record.reason_code_version
            == REASON_CODE_VERSION
        )

        assert record.raw_model_score == (
            explained.raw_scores[
                row_position
            ]
        )
        assert (
            record.calibrated_probability
            == explained
            .calibrated_probabilities[
                row_position
            ]
        )
        assert record.decision == (
            explained.decisions[
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

        assert (
            abs(
                (
                    record.expected_value_raw
                    + record.shap_sum_raw
                )
                - record
                .reconstructed_raw_margin
            )
            <= RECONSTRUCTION_ABSOLUTE_TOLERANCE
        )

        assert len(
            record.top_positive_contributions
        ) <= 3
        assert len(
            record.top_negative_contributions
        ) <= 3

        assert len(
            record.reason_codes
        ) == len(
            set(
                record.reason_codes
            )
        )

    assert_frame_equal(
        context.features,
        features_before,
    )
    assert (
        context.baseline_bundle
        .encoder.category_vocabularies
        == baseline_vocabularies_before
    )
    assert (
        context.policy_bundle
        .baseline_bundle
        .encoder.category_vocabularies
        == embedded_vocabularies_before
    )

    assert _booster_sha256(
        context.baseline_bundle
    ) == baseline_booster_before
    assert _booster_sha256(
        context.policy_bundle
        .baseline_bundle
    ) == embedded_booster_before

    assert _sha256_file(
        BASELINE_ARTIFACT
    ) == (
        context.baseline_artifact_sha256
    )
    assert _sha256_file(
        POLICY_ARTIFACT
    ) == (
        context.policy_artifact_sha256
    )


def test_reloaded_artifact_reproduces_explanations(
    frozen_artifacts: FrozenArtifactContext,
) -> None:
    """Require deterministic output after policy artifact reload."""

    context = frozen_artifacts

    first = _predict_explained(
        context
    )

    reloaded_policy = (
        load_calibrated_policy_bundle(
            POLICY_ARTIFACT
        )
    )
    second = _predict_explained(
        context,
        bundle=reloaded_policy,
    )

    np.testing.assert_array_equal(
        second.raw_scores,
        first.raw_scores,
    )
    np.testing.assert_array_equal(
        second.calibrated_probabilities,
        first.calibrated_probabilities,
    )
    np.testing.assert_array_equal(
        second.decisions,
        first.decisions,
    )

    assert (
        second.explanations
        == first.explanations
    )

    first_json = json.dumps(
        [
            asdict(record)
            for record in (
                first.explanations
            )
        ],
        sort_keys=True,
    )
    second_json = json.dumps(
        [
            asdict(record)
            for record in (
                second.explanations
            )
        ],
        sort_keys=True,
    )

    assert second_json == first_json

    assert _sha256_file(
        POLICY_ARTIFACT
    ) == EXPECTED_POLICY_SHA256


def test_batch_and_single_row_explanations_match(
    frozen_artifacts: FrozenArtifactContext,
) -> None:
    """Require per-row explanations to be batch-independent."""

    context = frozen_artifacts

    batch = _predict_explained(
        context
    )

    for row_position in range(
        SINGLE_ROW_PARITY_COUNT
    ):
        transaction_id = (
            context.transaction_ids[
                row_position
            ]
        )
        single_features = (
            context.features
            .iloc[
                [
                    row_position
                ]
            ]
            .copy(
                deep=True
            )
        )

        single = (
            predict_policy_with_explanations(
                context.policy_bundle,
                [
                    transaction_id,
                ],
                single_features,
            )
        )

        np.testing.assert_array_equal(
            single.raw_scores,
            batch.raw_scores[
                row_position
                : row_position + 1
            ],
        )
        np.testing.assert_array_equal(
            single.calibrated_probabilities,
            batch.calibrated_probabilities[
                row_position
                : row_position + 1
            ],
        )
        np.testing.assert_array_equal(
            single.decisions,
            batch.decisions[
                row_position
                : row_position + 1
            ],
        )

        assert single.explanations == (
            batch.explanations[
                row_position
            ],
        )

    assert _sha256_file(
        BASELINE_ARTIFACT
    ) == (
        context.baseline_artifact_sha256
    )
    assert _sha256_file(
        POLICY_ARTIFACT
    ) == (
        context.policy_artifact_sha256
    )
