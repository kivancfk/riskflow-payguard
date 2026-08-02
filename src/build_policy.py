"""Build the calibrated PayGuard policy from validation data only.

Development usage:

    python -m src.build_policy --skip-test-evaluation

This command deliberately loads only the chronological validation
dataset. It never loads or evaluates the test split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.calibration import (
    DEFAULT_SELECTION_ROUND_DIGITS,
    fit_and_select_calibrator,
)
from src.calibration_data import (
    CalibrationDevelopmentData,
    load_calibration_development_data,
)
from src.calibration_metrics import (
    DEFAULT_BIN_COUNT,
    evaluate_calibration,
)
from src.drift import (
    DEFAULT_PSI_BIN_COUNT,
    DEFAULT_PSI_EPSILON,
    compare_categorical_rates,
    evaluate_categorical_rates,
    evaluate_score_drift,
)
from src.model_bundle import (
    ModelBundle,
    load_model_bundle,
)
from src.policy import (
    DEFAULT_OBJECTIVE_ROUND_DIGITS,
    DEFAULT_QUANTILE_COUNT,
    PolicyConstraints,
    PolicyCostAssumptions,
    PolicyThresholds,
    search_policy_thresholds,
)
from src.policy_bundle import (
    CalibratedPolicyBundle,
    build_calibrated_policy_bundle,
    save_calibrated_policy_bundle,
)


DEFAULT_PROCESSED_DIR = Path(
    "data/processed"
)
DEFAULT_BASELINE_ARTIFACT = Path(
    "models/payguard_baseline.joblib"
)
DEFAULT_POLICY_OUTPUT = Path(
    "models/payguard_calibrated_policy.joblib"
)
DEFAULT_POLICY_VERSION = (
    "calibrated-policy-v1"
)

DEFAULT_COST_PER_REVIEW = 2.0
DEFAULT_COST_PER_LEGITIMATE_BLOCK = 5.0
DEFAULT_FRAUD_LOSS_MULTIPLIER = 1.0
DEFAULT_REVIEW_FRAUD_CAPTURE_RATE = 0.50

DEFAULT_MAX_REVIEW_RATE = 0.05
DEFAULT_MAX_BLOCK_RATE = 0.01
DEFAULT_MAX_INTERVENTION_RATE = 0.06

DEFAULT_REFERENCE_REVIEW_THRESHOLD = 0.30
DEFAULT_REFERENCE_BLOCK_THRESHOLD = 0.70


@dataclass(frozen=True)
class PolicyDevelopmentConfig:
    """Deterministic calibration and policy-search configuration."""

    calibration_bin_count: int = DEFAULT_BIN_COUNT
    selection_round_digits: int = (
        DEFAULT_SELECTION_ROUND_DIGITS
    )
    psi_bin_count: int = (
        DEFAULT_PSI_BIN_COUNT
    )
    psi_epsilon: float = (
        DEFAULT_PSI_EPSILON
    )
    quantile_count: int = (
        DEFAULT_QUANTILE_COUNT
    )
    objective_round_digits: int = (
        DEFAULT_OBJECTIVE_ROUND_DIGITS
    )
    reference_review_threshold: float = (
        DEFAULT_REFERENCE_REVIEW_THRESHOLD
    )
    reference_block_threshold: float = (
        DEFAULT_REFERENCE_BLOCK_THRESHOLD
    )


def _validate_feature_contract(
    baseline_bundle: ModelBundle,
    development_data: CalibrationDevelopmentData,
) -> None:
    """Require the frozen baseline feature contract."""

    contract = (
        development_data.feature_contract
    )

    if baseline_bundle.feature_columns != (
        contract.feature_columns
    ):
        raise ValueError(
            "Baseline feature columns do not match "
            "the calibration development data"
        )

    if (
        baseline_bundle.categorical_features
        != contract.categorical_features
    ):
        raise ValueError(
            "Baseline categorical features do not "
            "match calibration development data"
        )

    if (
        baseline_bundle.numerical_features
        != contract.numerical_features
    ):
        raise ValueError(
            "Baseline numerical features do not "
            "match calibration development data"
        )

    if (
        baseline_bundle.dataset_manifest
        != development_data.manifest
    ):
        raise ValueError(
            "Baseline dataset manifest does not match "
            "the calibration development data"
        )


def build_development_policy(
    *,
    baseline_bundle: ModelBundle,
    development_data: CalibrationDevelopmentData,
    cost_assumptions: PolicyCostAssumptions,
    constraints: PolicyConstraints,
    policy_version: str = DEFAULT_POLICY_VERSION,
    config: PolicyDevelopmentConfig | None = None,
) -> CalibratedPolicyBundle:
    """Build the policy using validation development data only."""

    resolved_config = (
        config
        if config is not None
        else PolicyDevelopmentConfig()
    )

    _validate_feature_contract(
        baseline_bundle,
        development_data,
    )

    calibration_fit = (
        development_data.calibration_fit
    )
    policy_selection = (
        development_data.policy_selection
    )

    calibration_fit_raw_scores = (
        baseline_bundle
        .predict_fraud_probabilities(
            calibration_fit.features,
            frame_name=(
                "calibration_fit features"
            ),
        )
    )

    policy_selection_raw_scores = (
        baseline_bundle
        .predict_fraud_probabilities(
            policy_selection.features,
            frame_name=(
                "policy_selection features"
            ),
        )
    )

    calibration_selection = (
        fit_and_select_calibrator(
            calibration_fit.target,
            calibration_fit_raw_scores,
            policy_selection.target,
            policy_selection_raw_scores,
            bin_count=(
                resolved_config
                .calibration_bin_count
            ),
            selection_round_digits=(
                resolved_config
                .selection_round_digits
            ),
        )
    )

    calibration_fit_probabilities = (
        calibration_selection
        .predict_calibrated_probabilities(
            calibration_fit_raw_scores
        )
    )

    policy_selection_probabilities = (
        calibration_selection
        .predict_calibrated_probabilities(
            policy_selection_raw_scores
        )
    )

    calibration_fit_report = (
        evaluate_calibration(
            calibration_fit.target,
            calibration_fit_probabilities,
            bin_count=(
                resolved_config
                .calibration_bin_count
            ),
        )
    )

    policy_selection_report = (
        evaluate_calibration(
            policy_selection.target,
            policy_selection_probabilities,
            bin_count=(
                resolved_config
                .calibration_bin_count
            ),
        )
    )

    score_drift = evaluate_score_drift(
        calibration_fit_raw_scores,
        policy_selection_raw_scores,
        calibration_fit_probabilities,
        policy_selection_probabilities,
        bin_count=(
            resolved_config.psi_bin_count
        ),
        epsilon=(
            resolved_config.psi_epsilon
        ),
    )

    categorical_drift = (
        compare_categorical_rates(
            calibration_fit.features,
            policy_selection.features,
            baseline_bundle.encoder,
            reference_name=(
                "calibration_fit"
            ),
            comparison_name=(
                "policy_selection"
            ),
        )
    )

    reference_categorical_rates = (
        evaluate_categorical_rates(
            calibration_fit.features,
            baseline_bundle.encoder,
            segment_name=(
                "calibration_fit"
            ),
        )
    )

    amount_column = "TransactionAmt"

    if amount_column not in (
        policy_selection.features.columns
    ):
        raise ValueError(
            "Policy selection data must contain "
            "TransactionAmt"
        )

    reference_thresholds = (
        PolicyThresholds(
            review_threshold=(
                resolved_config
                .reference_review_threshold
            ),
            block_threshold=(
                resolved_config
                .reference_block_threshold
            ),
        )
    )

    policy_search_result = (
        search_policy_thresholds(
            policy_selection.target,
            policy_selection_probabilities,
            policy_selection.features[
                amount_column
            ],
            cost_assumptions=(
                cost_assumptions
            ),
            constraints=constraints,
            quantile_count=(
                resolved_config.quantile_count
            ),
            objective_round_digits=(
                resolved_config
                .objective_round_digits
            ),
            reference_thresholds=(
                reference_thresholds
            ),
        )
    )

    candidate_results = [
        asdict(candidate)
        for candidate in (
            calibration_selection
            .candidate_results
        )
    ]

    calibration_metrics: dict[
        str,
        Any,
    ] = {
        "selected_method": (
            calibration_selection
            .selected_method
        ),
        "selection_bin_count": (
            calibration_selection.bin_count
        ),
        "selection_round_digits": (
            calibration_selection
            .selection_round_digits
        ),
        "candidate_results": (
            candidate_results
        ),
        "calibration_fit": (
            calibration_fit_report
        ),
        "policy_selection": (
            policy_selection_report
        ),
    }

    drift_diagnostics: dict[
        str,
        Any,
    ] = {
        "reference_segment": (
            "calibration_fit"
        ),
        "comparison_segment": (
            "policy_selection"
        ),
        "score_drift": score_drift,
        "categorical_rate_comparison": (
            categorical_drift
        ),
        "reference_categorical_rates": (
            reference_categorical_rates
        ),
        "reference_score_summary": {
            "row_count": len(
                calibration_fit_raw_scores
            ),
            "raw_score_minimum": float(
                calibration_fit_raw_scores.min()
            ),
            "raw_score_maximum": float(
                calibration_fit_raw_scores.max()
            ),
            "raw_score_mean": float(
                calibration_fit_raw_scores.mean()
            ),
            "calibrated_probability_minimum": (
                float(
                    calibration_fit_probabilities
                    .min()
                )
            ),
            "calibrated_probability_maximum": (
                float(
                    calibration_fit_probabilities
                    .max()
                )
            ),
            "calibrated_probability_mean": (
                float(
                    calibration_fit_probabilities
                    .mean()
                )
            ),
        },
    }

    development_metadata: dict[
        str,
        Any,
    ] = {
        "partition": asdict(
            development_data.metadata
        ),
        "baseline_model_version": (
            baseline_bundle.model_version
        ),
        "policy_version": policy_version,
        "feature_count": len(
            baseline_bundle.feature_columns
        ),
        "categorical_feature_count": len(
            baseline_bundle
            .categorical_features
        ),
        "numerical_feature_count": len(
            baseline_bundle
            .numerical_features
        ),
        "development_config": asdict(
            resolved_config
        ),
        "test_evaluation": None,
    }

    return build_calibrated_policy_bundle(
        baseline_bundle=baseline_bundle,
        calibration_selection=(
            calibration_selection
        ),
        policy_search_result=(
            policy_search_result
        ),
        development_metadata=(
            development_metadata
        ),
        calibration_metrics=(
            calibration_metrics
        ),
        drift_diagnostics=(
            drift_diagnostics
        ),
        dataset_manifest=(
            development_data.manifest
        ),
        policy_version=policy_version,
    )


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse the validation-only policy-development CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a calibrated fraud policy using "
            "validation development data only."
        )
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )
    parser.add_argument(
        "--baseline-artifact",
        type=Path,
        default=DEFAULT_BASELINE_ARTIFACT,
    )
    parser.add_argument(
        "--policy-output",
        type=Path,
        default=DEFAULT_POLICY_OUTPUT,
    )
    parser.add_argument(
        "--policy-version",
        default=DEFAULT_POLICY_VERSION,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        required=True,
        help=(
            "Required safety flag. This workflow "
            "does not load the chronological test split."
        ),
    )

    parser.add_argument(
        "--cost-per-review",
        type=float,
        default=DEFAULT_COST_PER_REVIEW,
    )
    parser.add_argument(
        "--cost-per-legitimate-block",
        type=float,
        default=(
            DEFAULT_COST_PER_LEGITIMATE_BLOCK
        ),
    )
    parser.add_argument(
        "--fraud-loss-multiplier",
        type=float,
        default=(
            DEFAULT_FRAUD_LOSS_MULTIPLIER
        ),
    )
    parser.add_argument(
        "--review-fraud-capture-rate",
        type=float,
        default=(
            DEFAULT_REVIEW_FRAUD_CAPTURE_RATE
        ),
    )

    parser.add_argument(
        "--max-review-rate",
        type=float,
        default=DEFAULT_MAX_REVIEW_RATE,
    )
    parser.add_argument(
        "--max-block-rate",
        type=float,
        default=DEFAULT_MAX_BLOCK_RATE,
    )
    parser.add_argument(
        "--max-intervention-rate",
        type=float,
        default=(
            DEFAULT_MAX_INTERVENTION_RATE
        ),
    )
    parser.add_argument(
        "--min-review-precision",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-block-precision",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-fraud-recall",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-fraud-amount-capture",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--calibration-bin-count",
        type=int,
        default=DEFAULT_BIN_COUNT,
    )
    parser.add_argument(
        "--psi-bin-count",
        type=int,
        default=DEFAULT_PSI_BIN_COUNT,
    )
    parser.add_argument(
        "--quantile-count",
        type=int,
        default=DEFAULT_QUANTILE_COUNT,
    )

    return parser.parse_args(argv)


def main() -> None:
    """Build and persist the validation-only policy artifact."""

    args = parse_args()

    baseline_bundle = load_model_bundle(
        args.baseline_artifact
    )

    development_data = (
        load_calibration_development_data(
            args.processed_dir
        )
    )

    cost_assumptions = (
        PolicyCostAssumptions(
            cost_per_review=(
                args.cost_per_review
            ),
            cost_per_legitimate_block=(
                args.cost_per_legitimate_block
            ),
            fraud_loss_multiplier=(
                args.fraud_loss_multiplier
            ),
            review_fraud_capture_rate=(
                args.review_fraud_capture_rate
            ),
        )
    )

    constraints = PolicyConstraints(
        max_review_rate=(
            args.max_review_rate
        ),
        max_block_rate=(
            args.max_block_rate
        ),
        max_intervention_rate=(
            args.max_intervention_rate
        ),
        min_review_precision=(
            args.min_review_precision
        ),
        min_block_precision=(
            args.min_block_precision
        ),
        min_fraud_recall=(
            args.min_fraud_recall
        ),
        min_fraud_amount_capture=(
            args.min_fraud_amount_capture
        ),
    )

    config = PolicyDevelopmentConfig(
        calibration_bin_count=(
            args.calibration_bin_count
        ),
        psi_bin_count=(
            args.psi_bin_count
        ),
        quantile_count=(
            args.quantile_count
        ),
    )

    bundle = build_development_policy(
        baseline_bundle=baseline_bundle,
        development_data=(
            development_data
        ),
        cost_assumptions=(
            cost_assumptions
        ),
        constraints=constraints,
        policy_version=(
            args.policy_version
        ),
        config=config,
    )

    output_path = (
        save_calibrated_policy_bundle(
            bundle,
            args.policy_output,
            overwrite=args.overwrite,
        )
    )

    summary = {
        "policy_output": str(
            output_path
        ),
        "policy_version": (
            bundle.policy_version
        ),
        "baseline_model_version": (
            bundle.baseline_model_version
        ),
        "selected_calibration_method": (
            bundle.calibration_selection
            .selected_method
        ),
        "selected_thresholds": asdict(
            bundle.thresholds
        ),
        "cost_assumptions": asdict(
            bundle.cost_assumptions
        ),
        "constraints": asdict(
            bundle.constraints
        ),
        "policy_metrics": (
            bundle.policy_search_result
            .selected_metrics
        ),
        "test_metrics": None,
    }

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
