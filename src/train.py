"""Train and persist the LightGBM fraud baseline.

Usage:
    python -m src.train

The output is a versioned joblib bundle containing every component
required to reproduce inference.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import (
    LGBMClassifier,
    early_stopping,
    log_evaluation,
)

from src.evaluation import evaluate_fraud_model
from src.model_bundle import (
    build_model_bundle,
    save_model_bundle,
)
from src.model_data import (
    BaselineDatasets,
    LabeledDataset,
    load_baseline_datasets,
)
from src.preprocessing import CategoricalEncoder


DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_MODEL_OUTPUT = Path(
    "models/payguard_baseline.joblib"
)
DEFAULT_MODEL_VERSION = "baseline-v1"

DEFAULT_SEED = 42
DEFAULT_N_JOBS = -1
DEFAULT_N_ESTIMATORS = 2_000
DEFAULT_EARLY_STOPPING_ROUNDS = 100
DEFAULT_LOG_PERIOD = 50


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the first LightGBM baseline."""

    seed: int = DEFAULT_SEED
    n_jobs: int = DEFAULT_N_JOBS

    learning_rate: float = 0.03
    n_estimators: int = DEFAULT_N_ESTIMATORS

    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 100

    subsample: float = 0.90
    subsample_freq: int = 1
    colsample_bytree: float = 0.90

    reg_alpha: float = 0.10
    reg_lambda: float = 1.00

    early_stopping_rounds: int = (
        DEFAULT_EARLY_STOPPING_ROUNDS
    )
    log_period: int = DEFAULT_LOG_PERIOD

    def validate(self) -> None:
        """Reject invalid LightGBM configuration values."""
        integer_fields = {
            "seed": self.seed,
            "n_jobs": self.n_jobs,
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample_freq": self.subsample_freq,
            "early_stopping_rounds": (
                self.early_stopping_rounds
            ),
            "log_period": self.log_period,
        }

        for field_name, value in integer_fields.items():
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
            ):
                raise ValueError(
                    f"{field_name} must be an integer"
                )

        if self.seed < 0:
            raise ValueError(
                "seed must not be negative"
            )

        if self.n_jobs == 0:
            raise ValueError(
                "n_jobs must not be zero"
            )

        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators must be positive"
            )

        if self.num_leaves <= 1:
            raise ValueError(
                "num_leaves must be greater than 1"
            )

        if (
            self.max_depth != -1
            and self.max_depth <= 0
        ):
            raise ValueError(
                "max_depth must be -1 or positive"
            )

        if self.min_child_samples <= 0:
            raise ValueError(
                "min_child_samples must be positive"
            )

        if self.subsample_freq < 0:
            raise ValueError(
                "subsample_freq must not be negative"
            )

        if self.early_stopping_rounds <= 0:
            raise ValueError(
                "early_stopping_rounds must be positive"
            )

        if self.log_period < 0:
            raise ValueError(
                "log_period must not be negative"
            )

        positive_float_fields = {
            "learning_rate": self.learning_rate,
        }

        for (
            field_name,
            value,
        ) in positive_float_fields.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(
                    f"{field_name} must be positive "
                    "and finite"
                )

        fraction_fields = {
            "subsample": self.subsample,
            "colsample_bytree": (
                self.colsample_bytree
            ),
        }

        for field_name, value in fraction_fields.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValueError(
                    f"{field_name} must be greater than "
                    "0 and no greater than 1"
                )

        regularization_fields = {
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }

        for (
            field_name,
            value,
        ) in regularization_fields.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"{field_name} must be non-negative "
                    "and finite"
                )


@dataclass(frozen=True)
class BaselineTrainingResult:
    """Fitted baseline model and validation results."""

    model: LGBMClassifier
    encoder: CategoricalEncoder
    config: TrainingConfig

    scale_pos_weight: float
    best_iteration: int

    validation_metrics: dict[str, Any]


def calculate_scale_pos_weight(
    target: pd.Series | np.ndarray,
) -> float:
    """Calculate negative-to-positive training class ratio."""
    try:
        target_values = np.asarray(
            target,
            dtype="float64",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Training target must contain numerical values"
        ) from error

    if target_values.ndim != 1:
        raise ValueError(
            "Training target must be one-dimensional"
        )

    if target_values.size == 0:
        raise ValueError(
            "Training target must not be empty"
        )

    if not np.isfinite(target_values).all():
        raise ValueError(
            "Training target must contain only "
            "finite values"
        )

    if not np.isin(
        target_values,
        [0.0, 1.0],
    ).all():
        raise ValueError(
            "Training target must contain only 0 and 1"
        )

    positive_count = int(
        target_values.sum()
    )
    negative_count = int(
        len(target_values) - positive_count
    )

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Training target must contain both classes"
        )

    return float(
        negative_count / positive_count
    )


def _build_model(
    config: TrainingConfig,
    *,
    scale_pos_weight: float,
) -> LGBMClassifier:
    """Construct the conservative LightGBM baseline."""
    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=config.subsample_freq,
        colsample_bytree=(
            config.colsample_bytree
        ),
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.seed,
        n_jobs=config.n_jobs,
        importance_type="gain",
        scale_pos_weight=scale_pos_weight,
        metric="None",
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _fit_with_validation(
    model: LGBMClassifier,
    *,
    training_features: pd.DataFrame,
    training_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    categorical_features: tuple[str, ...],
    config: TrainingConfig,
) -> None:
    """Fit across supported LightGBM validation APIs."""
    callbacks = [
        early_stopping(
            stopping_rounds=(
                config.early_stopping_rounds
            ),
            first_metric_only=True,
            verbose=False,
        ),
        log_evaluation(
            period=config.log_period
        ),
    ]

    fit_arguments: dict[str, Any] = {
        "eval_names": ["validation"],
        "eval_metric": [
            "average_precision",
            "auc",
            "binary_logloss",
        ],
        "categorical_feature": list(
            categorical_features
        ),
        "callbacks": callbacks,
    }

    fit_parameters = inspect.signature(
        model.fit
    ).parameters

    if (
        "eval_X" in fit_parameters
        and "eval_y" in fit_parameters
    ):
        fit_arguments["eval_X"] = (
            validation_features
        )
        fit_arguments["eval_y"] = (
            validation_target
        )
    else:
        fit_arguments["eval_set"] = [
            (
                validation_features,
                validation_target,
            )
        ]

    model.fit(
        training_features,
        training_target,
        **fit_arguments,
    )


def _predict_positive_class(
    model: LGBMClassifier,
    features: pd.DataFrame,
    *,
    best_iteration: int,
) -> np.ndarray:
    """Return validated positive-class probabilities."""
    probability_matrix = np.asarray(
        model.predict_proba(
            features,
            num_iteration=best_iteration,
            validate_features=True,
        ),
        dtype="float64",
    )

    expected_shape = (
        len(features),
        2,
    )

    if probability_matrix.shape != expected_shape:
        raise RuntimeError(
            "LightGBM returned an unexpected "
            "probability shape"
        )

    fraud_probabilities = probability_matrix[
        :,
        1,
    ]

    if not np.isfinite(
        fraud_probabilities
    ).all():
        raise RuntimeError(
            "LightGBM returned non-finite "
            "fraud probabilities"
        )

    return fraud_probabilities


def train_baseline_model(
    datasets: BaselineDatasets,
    *,
    config: TrainingConfig | None = None,
) -> BaselineTrainingResult:
    """Fit the baseline using train and validation only."""
    resolved_config = (
        config
        if config is not None
        else TrainingConfig()
    )
    resolved_config.validate()

    encoder = CategoricalEncoder.fit(
        datasets.train.features,
        datasets.feature_contract,
    )

    training_features = encoder.transform(
        datasets.train.features,
        frame_name="training features",
    )
    validation_features = encoder.transform(
        datasets.validation.features,
        frame_name="validation features",
    )

    scale_pos_weight = calculate_scale_pos_weight(
        datasets.train.target
    )

    model = _build_model(
        resolved_config,
        scale_pos_weight=scale_pos_weight,
    )

    _fit_with_validation(
        model,
        training_features=training_features,
        training_target=datasets.train.target,
        validation_features=validation_features,
        validation_target=(
            datasets.validation.target
        ),
        categorical_features=(
            datasets.feature_contract
            .categorical_features
        ),
        config=resolved_config,
    )

    best_iteration = int(
        model.best_iteration_
    )

    if best_iteration <= 0:
        raise RuntimeError(
            "LightGBM did not report a valid "
            "best iteration"
        )

    if tuple(model.feature_name_) != (
        datasets.feature_contract.feature_columns
    ):
        raise RuntimeError(
            "LightGBM feature names do not match "
            "the feature contract"
        )

    validation_probabilities = (
        _predict_positive_class(
            model,
            validation_features,
            best_iteration=best_iteration,
        )
    )

    transaction_amounts = None

    if (
        "TransactionAmt"
        in datasets.validation.features.columns
    ):
        transaction_amounts = (
            datasets.validation.features[
                "TransactionAmt"
            ]
        )

    validation_metrics = evaluate_fraud_model(
        datasets.validation.target,
        validation_probabilities,
        transaction_amounts=transaction_amounts,
    )

    return BaselineTrainingResult(
        model=model,
        encoder=encoder,
        config=resolved_config,
        scale_pos_weight=scale_pos_weight,
        best_iteration=best_iteration,
        validation_metrics=validation_metrics,
    )


def evaluate_training_result_on_split(
    result: BaselineTrainingResult,
    dataset: LabeledDataset,
    *,
    split_name: str,
) -> dict[str, Any]:
    """Evaluate a frozen model on one labeled split."""
    if not isinstance(split_name, str) or not split_name.strip():
        raise ValueError(
            "split_name must be a non-empty string"
        )

    encoded_features = result.encoder.transform(
        dataset.features,
        frame_name=f"{split_name.strip()} features",
    )

    probabilities = _predict_positive_class(
        result.model,
        encoded_features,
        best_iteration=result.best_iteration,
    )

    transaction_amounts = None

    if "TransactionAmt" in dataset.features.columns:
        transaction_amounts = dataset.features[
            "TransactionAmt"
        ]

    return evaluate_fraud_model(
        dataset.target,
        probabilities,
        transaction_amounts=transaction_amounts,
    )


def parse_args() -> argparse.Namespace:
    """Parse the temporary in-memory training CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the in-memory "
            "LightGBM fraud baseline."
        )
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=(
            "Directory containing the Phase 1 "
            "processed datasets."
        ),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT,
        help="Destination for the joblib model bundle.",
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help="Version identifier stored in the bundle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing model bundle.",
    )
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help=(
            "Train and save without evaluating the "
            "chronological test split."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Training random seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="LightGBM training thread count.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help=(
            "Maximum number of boosting iterations."
        ),
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=(
            DEFAULT_EARLY_STOPPING_ROUNDS
        ),
        help=(
            "Validation rounds without improvement "
            "before stopping."
        ),
    )
    parser.add_argument(
        "--log-period",
        type=int,
        default=DEFAULT_LOG_PERIOD,
        help=(
            "Boosting-iteration logging period; "
            "use 0 to disable."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the temporary in-memory baseline workflow."""
    args = parse_args()

    datasets = load_baseline_datasets(
        args.processed_dir
    )

    config = TrainingConfig(
        seed=args.seed,
        n_jobs=args.n_jobs,
        n_estimators=args.n_estimators,
        early_stopping_rounds=(
            args.early_stopping_rounds
        ),
        log_period=args.log_period,
    )

    result = train_baseline_model(
        datasets,
        config=config,
    )

    test_metrics = None

    if not args.skip_test_evaluation:
        test_metrics = (
            evaluate_training_result_on_split(
                result,
                datasets.test,
                split_name="test",
            )
        )

    bundle = build_model_bundle(
        model=result.model,
        encoder=result.encoder,
        feature_contract=(
            datasets.feature_contract
        ),
        training_config=asdict(
            result.config
        ),
        scale_pos_weight=(
            result.scale_pos_weight
        ),
        best_iteration=result.best_iteration,
        validation_metrics=(
            result.validation_metrics
        ),
        test_metrics=test_metrics,
        dataset_manifest=datasets.manifest,
        model_version=args.model_version,
    )

    model_output = save_model_bundle(
        bundle,
        args.model_output,
        overwrite=args.overwrite,
    )

    summary = {
        "model_output": str(model_output),
        "model_version": bundle.model_version,
        "training_config": asdict(result.config),
        "scale_pos_weight": (
            result.scale_pos_weight
        ),
        "best_iteration": result.best_iteration,
        "validation_metrics": (
            result.validation_metrics
        ),
        "test_metrics": test_metrics,
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
