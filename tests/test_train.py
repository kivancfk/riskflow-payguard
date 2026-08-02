"""Tests for the LightGBM baseline training pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    JOIN_KEY,
    TARGET_COLUMN,
)
from src.model_data import (
    BaselineDatasets,
    FeatureContract,
    LabeledDataset,
)
from src.preprocessing import (
    UNKNOWN_CATEGORY_CODE,
)
from src.train import (
    TrainingConfig,
    calculate_scale_pos_weight,
    evaluate_training_result_on_split,
    train_baseline_model,
)


FEATURE_CONTRACT = FeatureContract(
    feature_columns=(
        "TransactionAmt",
        "category_feature",
        "signal_feature",
        "noise_feature",
    ),
    categorical_features=(
        "category_feature",
    ),
    numerical_features=(
        "TransactionAmt",
        "signal_feature",
        "noise_feature",
    ),
)


def _make_split(
    *,
    row_count: int,
    start_id: int,
    seed: int,
    include_unseen_category: bool = False,
) -> LabeledDataset:
    """Create a deterministic synthetic fraud split."""
    random = np.random.default_rng(seed)

    signal = random.normal(
        loc=0.0,
        scale=1.0,
        size=row_count,
    )
    noise = random.normal(
        loc=0.0,
        scale=0.30,
        size=row_count,
    )

    categories = np.where(
        signal >= 0.0,
        "high",
        "low",
    ).astype(object)

    fraud_score = (
        signal
        + 0.60 * (categories == "high")
        + noise
    )

    target = (
        fraud_score > 1.20
    ).astype("int8")

    if include_unseen_category:
        categories[:5] = "future"

    transaction_amount = np.exp(
        3.0 + 0.20 * signal
    )

    features = pd.DataFrame(
        {
            "TransactionAmt": (
                transaction_amount
            ),
            "category_feature": categories,
            "signal_feature": signal,
            "noise_feature": noise,
        }
    )

    transaction_ids = pd.Series(
        np.arange(
            start_id,
            start_id + row_count,
        ),
        name=JOIN_KEY,
    )

    target_series = pd.Series(
        target,
        name=TARGET_COLUMN,
        dtype="int8",
    )

    assert set(target_series.unique()) == {
        0,
        1,
    }

    return LabeledDataset(
        transaction_ids=transaction_ids,
        features=features,
        target=target_series,
    )


def _make_datasets() -> BaselineDatasets:
    """Create train, validation, and unused test data."""
    return BaselineDatasets(
        train=_make_split(
            row_count=240,
            start_id=1_000,
            seed=10,
        ),
        validation=_make_split(
            row_count=100,
            start_id=2_000,
            seed=20,
            include_unseen_category=True,
        ),
        test=_make_split(
            row_count=100,
            start_id=3_000,
            seed=30,
        ),
        feature_contract=FEATURE_CONTRACT,
        manifest={"schema_version": 1},
    )


def _training_config() -> TrainingConfig:
    """Return a fast deterministic unit-test config."""
    return TrainingConfig(
        seed=7,
        n_jobs=1,
        learning_rate=0.08,
        n_estimators=100,
        num_leaves=15,
        max_depth=5,
        min_child_samples=5,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        early_stopping_rounds=10,
        log_period=0,
    )


def test_calculate_scale_pos_weight_matches_ratio(
) -> None:
    scale_pos_weight = calculate_scale_pos_weight(
        pd.Series(
            [0, 0, 0, 1],
            dtype="int8",
        )
    )

    assert scale_pos_weight == pytest.approx(
        3.0
    )


def test_train_baseline_model_fits_and_evaluates(
) -> None:
    datasets = _make_datasets()
    config = _training_config()

    result = train_baseline_model(
        datasets,
        config=config,
    )

    assert result.best_iteration > 0
    assert (
        result.best_iteration
        <= config.n_estimators
    )
    assert (
        result.model.best_iteration_
        == result.best_iteration
    )
    assert tuple(
        result.model.feature_name_
    ) == FEATURE_CONTRACT.feature_columns

    expected_weight = (
        (
            datasets.train.target == 0
        ).sum()
        / (
            datasets.train.target == 1
        ).sum()
    )

    assert result.scale_pos_weight == (
        pytest.approx(expected_weight)
    )

    probability_metrics = (
        result.validation_metrics[
            "probability_metrics"
        ]
    )

    assert (
        probability_metrics[
            "transaction_count"
        ]
        == len(datasets.validation.target)
    )
    assert (
        probability_metrics["pr_auc"]
        > probability_metrics["fraud_rate"]
    )
    assert (
        0.0
        <= probability_metrics["roc_auc"]
        <= 1.0
    )


def test_training_encoder_uses_training_only(
) -> None:
    datasets = _make_datasets()

    result = train_baseline_model(
        datasets,
        config=_training_config(),
    )

    vocabulary = (
        result.encoder.category_vocabularies[
            "category_feature"
        ]
    )

    assert "future" not in vocabulary

    encoded_validation = (
        result.encoder.transform(
            datasets.validation.features,
            frame_name="validation features",
        )
    )

    assert encoded_validation[
        "category_feature"
    ].iloc[:5].tolist() == [
        UNKNOWN_CATEGORY_CODE
    ] * 5


def test_training_is_deterministic() -> None:
    datasets = _make_datasets()
    config = _training_config()

    first_result = train_baseline_model(
        datasets,
        config=config,
    )
    second_result = train_baseline_model(
        datasets,
        config=config,
    )

    first_validation = (
        first_result.encoder.transform(
            datasets.validation.features,
            frame_name="validation features",
        )
    )
    second_validation = (
        second_result.encoder.transform(
            datasets.validation.features,
            frame_name="validation features",
        )
    )

    first_probabilities = (
        first_result.model.predict_proba(
            first_validation,
            num_iteration=(
                first_result.best_iteration
            ),
        )[:, 1]
    )
    second_probabilities = (
        second_result.model.predict_proba(
            second_validation,
            num_iteration=(
                second_result.best_iteration
            ),
        )[:, 1]
    )

    np.testing.assert_allclose(
        first_probabilities,
        second_probabilities,
        rtol=0.0,
        atol=0.0,
    )

    assert (
        first_result.best_iteration
        == second_result.best_iteration
    )


def test_training_rejects_invalid_configuration_and_target(
) -> None:
    with pytest.raises(
        ValueError,
        match="n_estimators",
    ):
        TrainingConfig(
            n_estimators=0
        ).validate()

    with pytest.raises(
        ValueError,
        match="both classes",
    ):
        calculate_scale_pos_weight(
            pd.Series(
                [0, 0, 0],
                dtype="int8",
            )
        )


def test_frozen_model_evaluates_test_split() -> None:
    datasets = _make_datasets()

    result = train_baseline_model(
        datasets,
        config=_training_config(),
    )

    metrics = evaluate_training_result_on_split(
        result,
        datasets.test,
        split_name="test",
    )

    probability_metrics = metrics[
        "probability_metrics"
    ]

    assert (
        probability_metrics[
            "transaction_count"
        ]
        == len(datasets.test.target)
    )
    assert (
        probability_metrics["pr_auc"]
        > probability_metrics["fraud_rate"]
    )
    assert len(metrics["review_capacity"]) == 4

