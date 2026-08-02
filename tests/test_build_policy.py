"""Tests for the validation-only calibrated policy workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.build_policy import (
    PolicyDevelopmentConfig,
    build_development_policy,
    parse_args,
)
from src.calibration_data import (
    CalibrationDevelopmentData,
    CalibrationDevelopmentMetadata,
    DevelopmentSegmentMetadata,
)
from src.model_bundle import ModelBundle
from src.model_data import LabeledDataset
from src.policy import (
    PolicyConstraints,
    PolicyCostAssumptions,
)
from tests.test_policy_bundle import (
    FEATURE_CONTRACT,
    _make_baseline_bundle,
)


def _development_scores() -> np.ndarray:
    """Return scores with known empirical event rates."""

    return np.repeat(
        np.array(
            [0.1, 0.3, 0.6, 0.9],
            dtype="float64",
        ),
        10,
    )


def _development_target() -> pd.Series:
    """Return labels aligned with development scores."""

    return pd.Series(
        (
            [1] + [0] * 9
            + [1] * 3 + [0] * 7
            + [1] * 6 + [0] * 4
            + [1] * 9 + [0]
        ),
        dtype="int8",
    )


def _segment(
    *,
    first_transaction_id: int,
) -> LabeledDataset:
    """Create one 40-row chronological segment."""

    row_count = 40

    return LabeledDataset(
        transaction_ids=pd.Series(
            range(
                first_transaction_id,
                first_transaction_id
                + row_count,
            )
        ),
        features=pd.DataFrame(
            {
                "category_feature": [
                    "low",
                    "medium",
                    "high",
                    None,
                ]
                * 10,
                "numeric_feature": (
                    np.linspace(
                        0.0,
                        1.0,
                        row_count,
                    )
                ),
                "TransactionAmt": (
                    np.linspace(
                        10.0,
                        400.0,
                        row_count,
                    )
                ),
            }
        ),
        target=_development_target(),
    )


def _development_data(
) -> CalibrationDevelopmentData:
    """Create isolated calibration and policy segments."""

    calibration_fit = _segment(
        first_transaction_id=100
    )
    policy_selection = _segment(
        first_transaction_id=200
    )

    calibration_metadata = (
        DevelopmentSegmentMetadata(
            name="calibration_fit",
            row_count=40,
            first_transaction_id=100,
            last_transaction_id=139,
            negative_count=21,
            fraud_count=19,
            fraud_rate=19 / 40,
        )
    )

    policy_metadata = (
        DevelopmentSegmentMetadata(
            name="policy_selection",
            row_count=40,
            first_transaction_id=200,
            last_transaction_id=239,
            negative_count=21,
            fraud_count=19,
            fraud_rate=19 / 40,
        )
    )

    metadata = (
        CalibrationDevelopmentMetadata(
            partition_method=(
                "chronological_row_position"
            ),
            total_rows=80,
            split_index=40,
            requested_calibration_fit_fraction=0.5,
            actual_calibration_fit_fraction=0.5,
            calibration_fit=(
                calibration_metadata
            ),
            policy_selection=(
                policy_metadata
            ),
        )
    )

    return CalibrationDevelopmentData(
        calibration_fit=calibration_fit,
        policy_selection=policy_selection,
        feature_contract=FEATURE_CONTRACT,
        manifest={
            "schema_version": 1,
        },
        metadata=metadata,
    )


def test_development_workflow_builds_without_test_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_bundle = (
        _make_baseline_bundle()
    )
    development_data = (
        _development_data()
    )
    scores = _development_scores()

    def fake_predict(
        self: ModelBundle,
        features: pd.DataFrame,
        *,
        frame_name: str,
    ) -> np.ndarray:
        assert frame_name in {
            "calibration_fit features",
            "policy_selection features",
        }
        assert len(features) == len(scores)
        return scores.copy()

    monkeypatch.setattr(
        ModelBundle,
        "predict_fraud_probabilities",
        fake_predict,
    )

    bundle = build_development_policy(
        baseline_bundle=baseline_bundle,
        development_data=(
            development_data
        ),
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
        policy_version=(
            "calibrated-policy-test-v1"
        ),
        config=PolicyDevelopmentConfig(
            calibration_bin_count=10,
            psi_bin_count=5,
            quantile_count=5,
        ),
    )

    assert bundle.policy_version == (
        "calibrated-policy-test-v1"
    )
    assert bundle.baseline_model_version == (
        baseline_bundle.model_version
    )
    assert bundle.development_metadata[
        "test_evaluation"
    ] is None

    assert bundle.calibration_metrics[
        "selected_method"
    ] in {
        "identity",
        "sigmoid",
        "isotonic",
    }

    assert bundle.drift_diagnostics[
        "reference_segment"
    ] == "calibration_fit"

    assert (
        bundle.policy_search_result
        .feasible_candidate_count
        > 0
    )


def test_cli_requires_explicit_skip_test_flag(
) -> None:
    with pytest.raises(SystemExit):
        parse_args([])

    args = parse_args(
        [
            "--skip-test-evaluation",
        ]
    )

    assert args.skip_test_evaluation
