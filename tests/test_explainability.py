"""Tests for deterministic model-explanation contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from dataclasses import asdict

import pytest

from src.explainability import (
    DECREASES_SCORE,
    INCREASES_SCORE,
    MISSING,
    OBSERVED,
    UNKNOWN_CATEGORY,
    FeatureContribution,
)


def _valid_arguments() -> dict[str, object]:
    """Return a valid feature-contribution payload."""

    return {
        "feature": "TransactionAmt",
        "feature_index": 4,
        "feature_group": "TRANSACTION_AMOUNT",
        "direction": INCREASES_SCORE,
        "shap_value_raw": 0.375,
        "value_state": OBSERVED,
        "rank": 1,
    }


def test_feature_contribution_normalizes_and_serializes() -> None:
    contribution = FeatureContribution(
        **_valid_arguments()
    )

    assert contribution.feature == "TransactionAmt"
    assert contribution.feature_index == 4
    assert (
        contribution.feature_group
        == "TRANSACTION_AMOUNT"
    )
    assert contribution.direction == INCREASES_SCORE
    assert contribution.shap_value_raw == 0.375
    assert (
        contribution.absolute_shap_value_raw
        == 0.375
    )
    assert contribution.value_state == OBSERVED
    assert contribution.rank == 1

    serialized = asdict(contribution)

    assert serialized == {
        "feature": "TransactionAmt",
        "feature_index": 4,
        "feature_group": "TRANSACTION_AMOUNT",
        "direction": INCREASES_SCORE,
        "shap_value_raw": 0.375,
        "value_state": OBSERVED,
        "rank": 1,
        "absolute_shap_value_raw": 0.375,
    }

    json.dumps(serialized)


def test_negative_contribution_uses_absolute_magnitude() -> None:
    contribution = FeatureContribution(
        feature="D1",
        feature_index=10,
        feature_group="TIME_DELTA",
        direction=DECREASES_SCORE,
        shap_value_raw=-0.25,
        value_state=MISSING,
        rank=2,
    )

    assert contribution.shap_value_raw == -0.25
    assert (
        contribution.absolute_shap_value_raw
        == 0.25
    )
    assert contribution.value_state == MISSING


def test_unknown_category_is_distinct_from_missing() -> None:
    contribution = FeatureContribution(
        feature="DeviceType",
        feature_index=20,
        feature_group="DEVICE_IDENTITY",
        direction=INCREASES_SCORE,
        shap_value_raw=0.1,
        value_state=UNKNOWN_CATEGORY,
        rank=1,
    )

    assert (
        contribution.value_state
        == UNKNOWN_CATEGORY
    )
    assert contribution.value_state != MISSING


def test_feature_contribution_is_immutable() -> None:
    contribution = FeatureContribution(
        **_valid_arguments()
    )

    with pytest.raises(FrozenInstanceError):
        contribution.rank = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        (
            "feature",
            "",
            "non-empty string",
        ),
        (
            "feature_index",
            -1,
            "greater than or equal to 0",
        ),
        (
            "feature_index",
            True,
            "greater than or equal to 0",
        ),
        (
            "feature_group",
            " ",
            "non-empty string",
        ),
        (
            "direction",
            "RAISES_RISK",
            "direction must be",
        ),
        (
            "shap_value_raw",
            float("nan"),
            "finite",
        ),
        (
            "shap_value_raw",
            0.0,
            "zero-contribution tolerance",
        ),
        (
            "value_state",
            "UNSEEN",
            "value_state must be",
        ),
        (
            "rank",
            0,
            "greater than or equal to 1",
        ),
        (
            "rank",
            False,
            "greater than or equal to 1",
        ),
    ],
)
def test_feature_contribution_rejects_invalid_values(
    field_name: str,
    value: object,
    message: str,
) -> None:
    arguments = _valid_arguments()
    arguments[field_name] = value

    with pytest.raises(
        ValueError,
        match=message,
    ):
        FeatureContribution(
            **arguments
        )


@pytest.mark.parametrize(
    ("direction", "shap_value_raw"),
    [
        (
            INCREASES_SCORE,
            -0.1,
        ),
        (
            DECREASES_SCORE,
            0.1,
        ),
    ],
)
def test_direction_must_match_shap_sign(
    direction: str,
    shap_value_raw: float,
) -> None:
    arguments = _valid_arguments()
    arguments["direction"] = direction
    arguments["shap_value_raw"] = (
        shap_value_raw
    )

    with pytest.raises(
        ValueError,
        match="match the sign",
    ):
        FeatureContribution(
            **arguments
        )


@pytest.mark.parametrize(
    "shap_value_raw",
    [
        1e-12,
        -1e-12,
        5e-13,
        -5e-13,
    ],
)
def test_negligible_contributions_are_rejected(
    shap_value_raw: float,
) -> None:
    arguments = _valid_arguments()
    arguments["shap_value_raw"] = (
        shap_value_raw
    )
    arguments["direction"] = (
        INCREASES_SCORE
        if shap_value_raw > 0.0
        else DECREASES_SCORE
    )

    with pytest.raises(
        ValueError,
        match="zero-contribution tolerance",
    ):
        FeatureContribution(
            **arguments
        )
