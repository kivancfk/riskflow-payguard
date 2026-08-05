"""Validated contracts for deterministic model explanations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from numbers import Integral
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
