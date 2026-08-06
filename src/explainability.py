"""Validated contracts for deterministic model explanations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from numbers import Integral
from typing import Any
from typing import Literal


EXPLANATION_VERSION = "shap-explanation-v1"
REASON_CODE_VERSION = "reason-codes-v1"

SHAP_ZERO_TOLERANCE = 1e-12

INCREASES_SCORE = "INCREASES_SCORE"
DECREASES_SCORE = "DECREASES_SCORE"

OBSERVED = "OBSERVED"
MISSING = "MISSING"
UNKNOWN_CATEGORY = "UNKNOWN_CATEGORY"

ContributionDirection = Literal[
    "INCREASES_SCORE",
    "DECREASES_SCORE",
]

FeatureValueState = Literal[
    "OBSERVED",
    "MISSING",
    "UNKNOWN_CATEGORY",
]


def _validate_non_empty_string(
    value: str,
    *,
    value_name: str,
) -> str:
    """Return a non-empty string."""

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{value_name} must be a non-empty string"
        )

    return value


def _validate_integer(
    value: int,
    *,
    value_name: str,
    minimum: int,
) -> int:
    """Return an integer at or above the requested minimum."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < minimum
    ):
        raise ValueError(
            f"{value_name} must be an integer "
            f"greater than or equal to {minimum}"
        )

    return int(value)


def _validate_finite_number(
    value: float,
    *,
    value_name: str,
) -> float:
    """Return a finite floating-point value."""

    if isinstance(value, bool):
        raise ValueError(
            f"{value_name} must be numerical"
        )

    try:
        normalized_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{value_name} must be numerical"
        ) from error

    if not math.isfinite(normalized_value):
        raise ValueError(
            f"{value_name} must be finite"
        )

    return normalized_value


@dataclass(frozen=True)
class FeatureContribution:
    """One selected feature contribution to the raw model margin."""

    feature: str
    feature_index: int
    feature_group: str
    direction: ContributionDirection
    shap_value_raw: float
    value_state: FeatureValueState
    rank: int
    absolute_shap_value_raw: float = field(
        init=False
    )

    def __post_init__(self) -> None:
        feature = _validate_non_empty_string(
            self.feature,
            value_name="feature",
        )
        feature_index = _validate_integer(
            self.feature_index,
            value_name="feature_index",
            minimum=0,
        )
        feature_group = _validate_non_empty_string(
            self.feature_group,
            value_name="feature_group",
        )

        if self.direction not in {
            INCREASES_SCORE,
            DECREASES_SCORE,
        }:
            raise ValueError(
                "direction must be INCREASES_SCORE "
                "or DECREASES_SCORE"
            )

        if self.value_state not in {
            OBSERVED,
            MISSING,
            UNKNOWN_CATEGORY,
        }:
            raise ValueError(
                "value_state must be OBSERVED, MISSING, "
                "or UNKNOWN_CATEGORY"
            )

        shap_value_raw = _validate_finite_number(
            self.shap_value_raw,
            value_name="shap_value_raw",
        )

        if (
            abs(shap_value_raw)
            <= SHAP_ZERO_TOLERANCE
        ):
            raise ValueError(
                "shap_value_raw must exceed the "
                "zero-contribution tolerance"
            )

        expected_direction = (
            INCREASES_SCORE
            if shap_value_raw > 0.0
            else DECREASES_SCORE
        )

        if self.direction != expected_direction:
            raise ValueError(
                "direction must match the sign of "
                "shap_value_raw"
            )

        rank = _validate_integer(
            self.rank,
            value_name="rank",
            minimum=1,
        )

        object.__setattr__(
            self,
            "feature",
            feature,
        )
        object.__setattr__(
            self,
            "feature_index",
            feature_index,
        )
        object.__setattr__(
            self,
            "feature_group",
            feature_group,
        )
        object.__setattr__(
            self,
            "shap_value_raw",
            shap_value_raw,
        )
        object.__setattr__(
            self,
            "rank",
            rank,
        )
        object.__setattr__(
            self,
            "absolute_shap_value_raw",
            abs(shap_value_raw),
        )

DEFAULT_TOP_CONTRIBUTIONS = 3


@dataclass(frozen=True)
class ContributionSelection:
    """Deterministically selected positive and negative contributions."""

    top_positive_contributions: tuple[
        FeatureContribution,
        ...,
    ]
    top_negative_contributions: tuple[
        FeatureContribution,
        ...,
    ]

    def __post_init__(self) -> None:
        for (
            field_name,
            contributions,
            expected_direction,
        ) in (
            (
                "top_positive_contributions",
                self.top_positive_contributions,
                INCREASES_SCORE,
            ),
            (
                "top_negative_contributions",
                self.top_negative_contributions,
                DECREASES_SCORE,
            ),
        ):
            if not isinstance(contributions, tuple):
                raise ValueError(
                    f"{field_name} must be a tuple"
                )

            for expected_rank, contribution in enumerate(
                contributions,
                start=1,
            ):
                if not isinstance(
                    contribution,
                    FeatureContribution,
                ):
                    raise ValueError(
                        f"{field_name} must contain only "
                        "FeatureContribution instances"
                    )

                if (
                    contribution.direction
                    != expected_direction
                ):
                    raise ValueError(
                        f"{field_name} contains an invalid "
                        "contribution direction"
                    )

                if contribution.rank != expected_rank:
                    raise ValueError(
                        f"{field_name} ranks must be "
                        "contiguous and one-based"
                    )


def _materialize_iterable(
    values: Iterable[Any],
    *,
    value_name: str,
) -> tuple[Any, ...]:
    """Return a non-empty materialized iterable."""

    if isinstance(values, (str, bytes)):
        raise ValueError(
            f"{value_name} must be an iterable of values"
        )

    try:
        normalized_values = tuple(values)
    except TypeError as error:
        raise ValueError(
            f"{value_name} must be an iterable of values"
        ) from error

    if not normalized_values:
        raise ValueError(
            f"{value_name} must contain at least one value"
        )

    return normalized_values


def _build_ranked_contributions(
    candidates: list[
        tuple[
            str,
            int,
            str,
            float,
            FeatureValueState,
        ]
    ],
    *,
    direction: ContributionDirection,
    top_k: int,
) -> tuple[FeatureContribution, ...]:
    """Build one deterministically ranked contribution list."""

    if direction == INCREASES_SCORE:
        ordered_candidates = sorted(
            candidates,
            key=lambda candidate: (
                -candidate[3],
                candidate[1],
                candidate[0],
            ),
        )
    else:
        ordered_candidates = sorted(
            candidates,
            key=lambda candidate: (
                candidate[3],
                candidate[1],
                candidate[0],
            ),
        )

    return tuple(
        FeatureContribution(
            feature=feature,
            feature_index=feature_index,
            feature_group=feature_group,
            direction=direction,
            shap_value_raw=shap_value_raw,
            value_state=value_state,
            rank=rank,
        )
        for rank, (
            feature,
            feature_index,
            feature_group,
            shap_value_raw,
            value_state,
        ) in enumerate(
            ordered_candidates[:top_k],
            start=1,
        )
    )


def select_top_contributions(
    feature_names: Iterable[str],
    shap_values_raw: Iterable[float],
    feature_groups: Iterable[str],
    value_states: Iterable[FeatureValueState],
    *,
    top_k: int = DEFAULT_TOP_CONTRIBUTIONS,
) -> ContributionSelection:
    """Select deterministic positive and negative SHAP contributions.

    Positive contributions use the ordering key:

    ``(-shap_value_raw, feature_index, feature)``

    Negative contributions use the ordering key:

    ``(shap_value_raw, feature_index, feature)``

    Contributions at or below ``SHAP_ZERO_TOLERANCE`` in absolute
    magnitude are excluded.
    """

    normalized_top_k = _validate_integer(
        top_k,
        value_name="top_k",
        minimum=1,
    )

    normalized_feature_names = (
        _materialize_iterable(
            feature_names,
            value_name="feature_names",
        )
    )
    normalized_shap_values = (
        _materialize_iterable(
            shap_values_raw,
            value_name="shap_values_raw",
        )
    )
    normalized_feature_groups = (
        _materialize_iterable(
            feature_groups,
            value_name="feature_groups",
        )
    )
    normalized_value_states = (
        _materialize_iterable(
            value_states,
            value_name="value_states",
        )
    )

    input_lengths = {
        len(normalized_feature_names),
        len(normalized_shap_values),
        len(normalized_feature_groups),
        len(normalized_value_states),
    }

    if len(input_lengths) != 1:
        raise ValueError(
            "feature_names, shap_values_raw, "
            "feature_groups, and value_states "
            "must have the same length"
        )

    validated_feature_names = tuple(
        _validate_non_empty_string(
            feature_name,
            value_name=(
                f"feature_names[{feature_index}]"
            ),
        )
        for feature_index, feature_name in enumerate(
            normalized_feature_names
        )
    )

    if (
        len(set(validated_feature_names))
        != len(validated_feature_names)
    ):
        raise ValueError(
            "feature_names must be unique"
        )

    validated_shap_values = tuple(
        _validate_finite_number(
            shap_value,
            value_name=(
                f"shap_values_raw[{feature_index}]"
            ),
        )
        for feature_index, shap_value in enumerate(
            normalized_shap_values
        )
    )

    validated_feature_groups = tuple(
        _validate_non_empty_string(
            feature_group,
            value_name=(
                f"feature_groups[{feature_index}]"
            ),
        )
        for feature_index, feature_group in enumerate(
            normalized_feature_groups
        )
    )

    validated_value_states: list[
        FeatureValueState
    ] = []

    for feature_index, value_state in enumerate(
        normalized_value_states
    ):
        if value_state not in {
            OBSERVED,
            MISSING,
            UNKNOWN_CATEGORY,
        }:
            raise ValueError(
                f"value_states[{feature_index}] must be "
                "OBSERVED, MISSING, or UNKNOWN_CATEGORY"
            )

        validated_value_states.append(value_state)

    positive_candidates: list[
        tuple[
            str,
            int,
            str,
            float,
            FeatureValueState,
        ]
    ] = []
    negative_candidates: list[
        tuple[
            str,
            int,
            str,
            float,
            FeatureValueState,
        ]
    ] = []

    for feature_index, (
        feature_name,
        shap_value_raw,
        feature_group,
        value_state,
    ) in enumerate(
        zip(
            validated_feature_names,
            validated_shap_values,
            validated_feature_groups,
            validated_value_states,
            strict=True,
        )
    ):
        if (
            abs(shap_value_raw)
            <= SHAP_ZERO_TOLERANCE
        ):
            continue

        candidate = (
            feature_name,
            feature_index,
            feature_group,
            shap_value_raw,
            value_state,
        )

        if shap_value_raw > 0.0:
            positive_candidates.append(candidate)
        else:
            negative_candidates.append(candidate)

    return ContributionSelection(
        top_positive_contributions=(
            _build_ranked_contributions(
                positive_candidates,
                direction=INCREASES_SCORE,
                top_k=normalized_top_k,
            )
        ),
        top_negative_contributions=(
            _build_ranked_contributions(
                negative_candidates,
                direction=DECREASES_SCORE,
                top_k=normalized_top_k,
            )
        ),
    )
