"""Tests for deterministic per-row analyst explanations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from pandas.testing import assert_frame_equal

from src.explainability import (
    INCREASES_SCORE,
    MISSING,
    OBSERVED,
    UNKNOWN_CATEGORY,
)
from src.explanation_engine import (
    ModelSignalExplanation,
    generate_model_signal_explanations,
)
from src.lightgbm_shap import (
    LightGBMContributionMatrix,
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
        "card1",
        "card2",
        "D1",
        "DeviceType",
    ),
    categorical_features=(
        "card1",
        "DeviceType",
    ),
    numerical_features=(
        "card2",
        "D1",
    ),
)


def _training_features() -> pd.DataFrame:
    """Return deterministic mixed model features."""

    return pd.DataFrame(
        {
            "card1": [
                "visa",
                "visa",
                "mastercard",
                "mastercard",
                "visa",
                "mastercard",
                "visa",
                None,
                "mastercard",
                "visa",
                "mastercard",
                "visa",
            ],
            "card2": [
                100.0,
                110.0,
                200.0,
                210.0,
                120.0,
                220.0,
                130.0,
                230.0,
                240.0,
                140.0,
                250.0,
                150.0,
            ],
            "D1": [
                0.1,
                0.2,
                0.8,
                0.9,
                0.3,
                1.0,
                0.4,
                1.1,
                1.2,
                0.5,
                1.3,
                0.6,
            ],
            "DeviceType": [
                "desktop",
                "desktop",
                "mobile",
                "mobile",
                "desktop",
                "mobile",
                "desktop",
                None,
                "mobile",
                "desktop",
                "mobile",
                "desktop",
            ],
        }
    )


def _training_target() -> pd.Series:
    """Return a binary target aligned to training data."""

    return pd.Series(
        [
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
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
        n_estimators=12,
        learning_rate=0.20,
        num_leaves=4,
        min_child_samples=1,
        random_state=11,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(
        encoded_features,
        _training_target(),
        categorical_feature=[
            "card1",
            "DeviceType",
        ],
    )

    return build_model_bundle(
        model=model,
        encoder=encoder,
        feature_contract=FEATURE_CONTRACT,
        training_config={
            "n_estimators": 12,
            "seed": 11,
        },
        scale_pos_weight=1.0,
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


def _inference_features() -> pd.DataFrame:
    """Return known, unknown, and missing values."""

    return pd.DataFrame(
        {
            "card1": [
                "visa",
                "amex",
                None,
            ],
            "card2": [
                115.0,
                260.0,
                175.0,
            ],
            "D1": [
                np.nan,
                1.4,
                0.7,
            ],
            "DeviceType": [
                "desktop",
                "tablet",
                None,
            ],
        },
        index=[
            101,
            102,
            103,
        ],
    )


def _controlled_contributions(
) -> LightGBMContributionMatrix:
    """Return controlled contributions for ordering tests."""

    return LightGBMContributionMatrix(
        feature_names=(
            FEATURE_CONTRACT.feature_columns
        ),
        shap_values_raw=np.array(
            [
                [
                    0.8,
                    0.7,
                    -0.4,
                    -0.2,
                ],
                [
                    -0.6,
                    0.4,
                    0.3,
                    -0.8,
                ],
                [
                    0.5,
                    0.4,
                    0.3,
                    0.2,
                ],
            ],
            dtype="float64",
        ),
        expected_values_raw=np.zeros(
            3,
            dtype="float64",
        ),
    )


def _controlled_states() -> pd.DataFrame:
    """Return aligned controlled value states."""

    return pd.DataFrame(
        [
            [
                OBSERVED,
                OBSERVED,
                MISSING,
                UNKNOWN_CATEGORY,
            ],
            [
                MISSING,
                OBSERVED,
                OBSERVED,
                OBSERVED,
            ],
            [
                OBSERVED,
                OBSERVED,
                OBSERVED,
                OBSERVED,
            ],
        ],
        index=[
            101,
            102,
            103,
        ],
        columns=list(
            FEATURE_CONTRACT.feature_columns
        ),
    )


def test_orders_contributions_and_deduplicates_reasons() -> None:
    explanations = (
        generate_model_signal_explanations(
            _make_bundle(),
            _inference_features(),
            contributions=(
                _controlled_contributions()
            ),
            value_states=(
                _controlled_states()
            ),
        )
    )

    first = explanations[0]

    assert [
        contribution.feature
        for contribution in (
            first.top_positive_contributions
        )
    ] == [
        "card1",
        "card2",
    ]

    assert [
        contribution.feature
        for contribution in (
            first.top_negative_contributions
        )
    ] == [
        "D1",
        "DeviceType",
    ]

    assert first.reason_codes == (
        "PAYMENT_INSTRUMENT_"
        "OBSERVED_INCREASES_SCORE",
        "TIME_DELTA_"
        "MISSING_DECREASES_SCORE",
        "DEVICE_IDENTITY_"
        "UNKNOWN_CATEGORY_DECREASES_SCORE",
    )

    assert (
        first.reasons[0]
        .contribution.feature
        == "card1"
    )

    assert len(
        first.selected_contributions
    ) == 4


def test_default_generation_is_deterministic_and_read_only() -> None:
    bundle = _make_bundle()
    features = _inference_features()

    original_features = features.copy(
        deep=True
    )
    original_vocabularies = dict(
        bundle.encoder.category_vocabularies
    )

    first = generate_model_signal_explanations(
        bundle,
        features,
    )
    second = generate_model_signal_explanations(
        bundle,
        features,
    )

    assert first == second
    assert [
        explanation.row_position
        for explanation in first
    ] == [
        0,
        1,
        2,
    ]

    assert_frame_equal(
        features,
        original_features,
    )
    assert (
        bundle.encoder.category_vocabularies
        == original_vocabularies
    )

    for explanation in first:
        assert len(
            explanation.reason_codes
        ) == len(
            set(
                explanation.reason_codes
            )
        )

        assert (
            explanation.selected_contributions
            == (
                *explanation
                .top_positive_contributions,
                *explanation
                .top_negative_contributions,
            )
        )


def test_one_direction_can_be_empty() -> None:
    explanations = (
        generate_model_signal_explanations(
            _make_bundle(),
            _inference_features(),
            contributions=(
                _controlled_contributions()
            ),
            value_states=(
                _controlled_states()
            ),
        )
    )

    third = explanations[2]

    assert len(
        third.top_positive_contributions
    ) == 3
    assert (
        third.top_negative_contributions
        == ()
    )
    assert all(
        contribution.direction
        == INCREASES_SCORE
        for contribution in (
            third.top_positive_contributions
        )
    )


def test_rejects_mismatched_contribution_features() -> None:
    controlled = _controlled_contributions()

    mismatched = LightGBMContributionMatrix(
        feature_names=(
            "card2",
            "card1",
            "D1",
            "DeviceType",
        ),
        shap_values_raw=(
            controlled.shap_values_raw
        ),
        expected_values_raw=(
            controlled.expected_values_raw
        ),
    )

    with pytest.raises(
        ValueError,
        match="feature names do not match",
    ):
        generate_model_signal_explanations(
            _make_bundle(),
            _inference_features(),
            contributions=mismatched,
            value_states=(
                _controlled_states()
            ),
        )


def test_rejects_misaligned_contribution_rows() -> None:
    controlled = _controlled_contributions()

    shortened = LightGBMContributionMatrix(
        feature_names=(
            controlled.feature_names
        ),
        shap_values_raw=(
            controlled.shap_values_raw[
                :2
            ]
        ),
        expected_values_raw=(
            controlled.expected_values_raw[
                :2
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="rows do not align",
    ):
        generate_model_signal_explanations(
            _make_bundle(),
            _inference_features(),
            contributions=shortened,
            value_states=(
                _controlled_states()
            ),
        )


def test_rejects_misaligned_state_columns() -> None:
    states = _controlled_states().loc[
        :,
        [
            "card2",
            "card1",
            "D1",
            "DeviceType",
        ],
    ]

    with pytest.raises(
        ValueError,
        match="columns do not match",
    ):
        generate_model_signal_explanations(
            _make_bundle(),
            _inference_features(),
            contributions=(
                _controlled_contributions()
            ),
            value_states=states,
        )


def test_rejects_misaligned_state_index() -> None:
    states = _controlled_states().copy()
    states.index = [
        1,
        2,
        3,
    ]

    with pytest.raises(
        ValueError,
        match="index does not align",
    ):
        generate_model_signal_explanations(
            _make_bundle(),
            _inference_features(),
            contributions=(
                _controlled_contributions()
            ),
            value_states=states,
        )


def test_rejects_invalid_value_state() -> None:
    states = _controlled_states().copy()
    states.loc[
        101,
        "card1",
    ] = "UNSUPPORTED"

    with pytest.raises(
        ValueError,
        match="contains an invalid state",
    ):
        generate_model_signal_explanations(
            _make_bundle(),
            _inference_features(),
            contributions=(
                _controlled_contributions()
            ),
            value_states=states,
        )


@pytest.mark.parametrize(
    (
        "contributions",
        "value_states",
        "message",
    ),
    [
        (
            object(),
            _controlled_states(),
            "LightGBMContributionMatrix",
        ),
        (
            _controlled_contributions(),
            object(),
            "pandas DataFrame",
        ),
    ],
)
def test_rejects_invalid_precomputed_inputs(
    contributions: object,
    value_states: object,
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        generate_model_signal_explanations(
            _make_bundle(),
            _inference_features(),
            contributions=(
                contributions
            ),  # type: ignore[arg-type]
            value_states=(
                value_states
            ),  # type: ignore[arg-type]
        )


def test_result_contract_rejects_duplicate_reasons() -> None:
    valid = generate_model_signal_explanations(
        _make_bundle(),
        _inference_features(),
        contributions=(
            _controlled_contributions()
        ),
        value_states=(
            _controlled_states()
        ),
    )[0]

    with pytest.raises(
        ValueError,
        match="unique and follow",
    ):
        ModelSignalExplanation(
            row_position=valid.row_position,
            top_positive_contributions=(
                valid.top_positive_contributions
            ),
            top_negative_contributions=(
                valid.top_negative_contributions
            ),
            reasons=(
                valid.reasons[0],
                *valid.reasons,
            ),
        )


def test_result_contract_is_immutable() -> None:
    explanation = (
        generate_model_signal_explanations(
            _make_bundle(),
            _inference_features(),
            contributions=(
                _controlled_contributions()
            ),
            value_states=(
                _controlled_states()
            ),
        )[0]
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        explanation.row_position = (
            2
        )  # type: ignore[misc]
