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
    select_top_contributions,
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


def test_top_contributions_are_ordered_deterministically() -> None:
    arguments = {
        "feature_names": [
            "positive_first",
            "positive_second",
            "positive_smaller",
            "negative_first",
            "negative_second",
            "negative_smaller",
            "zero",
        ],
        "shap_values_raw": [
            0.5,
            0.5,
            0.2,
            -0.9,
            -0.9,
            -0.2,
            0.0,
        ],
        "feature_groups": [
            "GROUP_0",
            "GROUP_1",
            "GROUP_2",
            "GROUP_3",
            "GROUP_4",
            "GROUP_5",
            "GROUP_6",
        ],
        "value_states": [
            OBSERVED,
            MISSING,
            UNKNOWN_CATEGORY,
            OBSERVED,
            MISSING,
            UNKNOWN_CATEGORY,
            OBSERVED,
        ],
        "top_k": 2,
    }

    first = select_top_contributions(
        **arguments
    )
    second = select_top_contributions(
        **arguments
    )

    assert first == second

    assert [
        contribution.feature
        for contribution in (
            first.top_positive_contributions
        )
    ] == [
        "positive_first",
        "positive_second",
    ]

    assert [
        contribution.feature_index
        for contribution in (
            first.top_positive_contributions
        )
    ] == [0, 1]

    assert [
        contribution.rank
        for contribution in (
            first.top_positive_contributions
        )
    ] == [1, 2]

    assert [
        contribution.feature
        for contribution in (
            first.top_negative_contributions
        )
    ] == [
        "negative_first",
        "negative_second",
    ]

    assert [
        contribution.feature_index
        for contribution in (
            first.top_negative_contributions
        )
    ] == [3, 4]

    assert [
        contribution.rank
        for contribution in (
            first.top_negative_contributions
        )
    ] == [1, 2]

    assert (
        first.top_positive_contributions[1]
        .value_state
        == MISSING
    )
    assert (
        first.top_negative_contributions[1]
        .feature_group
        == "GROUP_4"
    )


def test_negligible_values_are_excluded_from_selection() -> None:
    selection = select_top_contributions(
        feature_names=[
            "zero",
            "positive_boundary",
            "negative_boundary",
            "positive_included",
            "negative_included",
        ],
        shap_values_raw=[
            0.0,
            1e-12,
            -1e-12,
            1.1e-12,
            -1.1e-12,
        ],
        feature_groups=[
            "GROUP",
            "GROUP",
            "GROUP",
            "GROUP",
            "GROUP",
        ],
        value_states=[
            OBSERVED,
            OBSERVED,
            OBSERVED,
            OBSERVED,
            OBSERVED,
        ],
    )

    assert [
        contribution.feature
        for contribution in (
            selection.top_positive_contributions
        )
    ] == ["positive_included"]

    assert [
        contribution.feature
        for contribution in (
            selection.top_negative_contributions
        )
    ] == ["negative_included"]


def test_selection_allows_one_direction_to_be_empty() -> None:
    selection = select_top_contributions(
        feature_names=[
            "positive_one",
            "positive_two",
        ],
        shap_values_raw=[
            0.3,
            0.1,
        ],
        feature_groups=[
            "GROUP",
            "GROUP",
        ],
        value_states=[
            OBSERVED,
            MISSING,
        ],
    )

    assert len(
        selection.top_positive_contributions
    ) == 2
    assert (
        selection.top_negative_contributions
        == ()
    )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        True,
    ],
)
def test_selection_rejects_invalid_top_k(
    top_k: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be an integer",
    ):
        select_top_contributions(
            feature_names=["feature"],
            shap_values_raw=[0.1],
            feature_groups=["GROUP"],
            value_states=[OBSERVED],
            top_k=top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    (
        "shap_values_raw",
        "feature_groups",
        "value_states",
    ),
    [
        (
            [0.1],
            ["GROUP", "GROUP"],
            [OBSERVED, OBSERVED],
        ),
        (
            [0.1, 0.2],
            ["GROUP"],
            [OBSERVED, OBSERVED],
        ),
        (
            [0.1, 0.2],
            ["GROUP", "GROUP"],
            [OBSERVED],
        ),
    ],
)
def test_selection_rejects_misaligned_inputs(
    shap_values_raw: list[float],
    feature_groups: list[str],
    value_states: list[str],
) -> None:
    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        select_top_contributions(
            feature_names=[
                "feature_0",
                "feature_1",
            ],
            shap_values_raw=shap_values_raw,
            feature_groups=feature_groups,
            value_states=value_states,
        )


def test_selection_rejects_empty_feature_contract() -> None:
    with pytest.raises(
        ValueError,
        match="feature_names must contain",
    ):
        select_top_contributions(
            feature_names=[],
            shap_values_raw=[],
            feature_groups=[],
            value_states=[],
        )


def test_selection_rejects_duplicate_feature_names() -> None:
    with pytest.raises(
        ValueError,
        match="feature_names must be unique",
    ):
        select_top_contributions(
            feature_names=[
                "duplicate",
                "duplicate",
            ],
            shap_values_raw=[
                0.1,
                -0.1,
            ],
            feature_groups=[
                "GROUP",
                "GROUP",
            ],
            value_states=[
                OBSERVED,
                OBSERVED,
            ],
        )


@pytest.mark.parametrize(
    "shap_value_raw",
    [
        float("nan"),
        float("inf"),
    ],
)
def test_selection_rejects_non_finite_shap_values(
    shap_value_raw: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        select_top_contributions(
            feature_names=["feature"],
            shap_values_raw=[shap_value_raw],
            feature_groups=["GROUP"],
            value_states=[OBSERVED],
        )


@pytest.mark.parametrize(
    (
        "feature_groups",
        "value_states",
        "message",
    ),
    [
        (
            [""],
            [OBSERVED],
            "non-empty string",
        ),
        (
            ["GROUP"],
            ["UNSEEN"],
            "value_states\\[0\\]",
        ),
    ],
)
def test_selection_validates_all_feature_metadata(
    feature_groups: list[str],
    value_states: list[str],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        select_top_contributions(
            feature_names=["feature"],
            shap_values_raw=[0.0],
            feature_groups=feature_groups,
            value_states=value_states,
        )
