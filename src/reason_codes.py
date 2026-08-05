"""Stable analyst-facing feature groups and reason codes."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.explainability import (
    DECREASES_SCORE,
    INCREASES_SCORE,
    MISSING,
    OBSERVED,
    REASON_CODE_VERSION,
    UNKNOWN_CATEGORY,
    FeatureContribution,
)


TRANSACTION_AMOUNT = "TRANSACTION_AMOUNT"
TRANSACTION_TIME = "TRANSACTION_TIME"
PAYMENT_INSTRUMENT = "PAYMENT_INSTRUMENT"
ADDRESS = "ADDRESS"
DISTANCE = "DISTANCE"
EMAIL_DOMAIN = "EMAIL_DOMAIN"
COUNT_AGGREGATE = "COUNT_AGGREGATE"
TIME_DELTA = "TIME_DELTA"
MATCH_FLAG = "MATCH_FLAG"
ANONYMIZED_BEHAVIOR = "ANONYMIZED_BEHAVIOR"
DEVICE_IDENTITY = "DEVICE_IDENTITY"
OTHER_MODEL_SIGNAL = "OTHER_MODEL_SIGNAL"

FEATURE_GROUPS = frozenset(
    {
        TRANSACTION_AMOUNT,
        TRANSACTION_TIME,
        PAYMENT_INSTRUMENT,
        ADDRESS,
        DISTANCE,
        EMAIL_DOMAIN,
        COUNT_AGGREGATE,
        TIME_DELTA,
        MATCH_FLAG,
        ANONYMIZED_BEHAVIOR,
        DEVICE_IDENTITY,
        OTHER_MODEL_SIGNAL,
    }
)

_EXACT_FEATURE_GROUPS = {
    "TransactionAmt": TRANSACTION_AMOUNT,
    "transaction_amount_log1p": (
        TRANSACTION_AMOUNT
    ),
    "TransactionDT": TRANSACTION_TIME,
    "relative_day": TRANSACTION_TIME,
    "relative_hour_cycle": TRANSACTION_TIME,
    "ProductCD": PAYMENT_INSTRUMENT,
    "addr1": ADDRESS,
    "addr2": ADDRESS,
    "dist1": DISTANCE,
    "dist2": DISTANCE,
    "P_emaildomain": EMAIL_DOMAIN,
    "R_emaildomain": EMAIL_DOMAIN,
    "DeviceType": DEVICE_IDENTITY,
    "DeviceInfo": DEVICE_IDENTITY,
    "identity_availability": DEVICE_IDENTITY,
}

_CARD_PATTERN = re.compile(
    r"card[1-6]"
)
_COUNT_PATTERN = re.compile(
    r"C(?:[1-9]|1[0-4])"
)
_DELTA_PATTERN = re.compile(
    r"D(?:[1-9]|1[0-5])"
)
_MATCH_PATTERN = re.compile(
    r"M[1-9]"
)
_IDENTITY_PATTERN = re.compile(
    r"id_\d{2}"
)
_ANONYMIZED_PATTERN = re.compile(
    r"V([1-9]\d*)"
)

_FEATURE_GROUP_LABELS = {
    TRANSACTION_AMOUNT: "Transaction amount",
    TRANSACTION_TIME: "Transaction timing",
    PAYMENT_INSTRUMENT: "Payment instrument",
    ADDRESS: "Address",
    DISTANCE: "Distance",
    EMAIL_DOMAIN: "Email domain",
    COUNT_AGGREGATE: "Count aggregate",
    TIME_DELTA: "Time delta",
    MATCH_FLAG: "Match flag",
    ANONYMIZED_BEHAVIOR: (
        "Anonymized behavioral"
    ),
    DEVICE_IDENTITY: "Device identity",
    OTHER_MODEL_SIGNAL: "Other model",
}


def _validate_feature_name(
    feature: str,
) -> str:
    """Return a stable non-empty model feature name."""

    if (
        not isinstance(feature, str)
        or not feature
        or feature != feature.strip()
    ):
        raise ValueError(
            "feature must be a non-empty string "
            "without surrounding whitespace"
        )

    return feature


def feature_group_for(
    feature: str,
) -> str:
    """Map one model feature to its stable analyst group."""

    normalized_feature = _validate_feature_name(
        feature
    )

    exact_group = _EXACT_FEATURE_GROUPS.get(
        normalized_feature
    )

    if exact_group is not None:
        return exact_group

    if _CARD_PATTERN.fullmatch(
        normalized_feature
    ):
        return PAYMENT_INSTRUMENT

    if _COUNT_PATTERN.fullmatch(
        normalized_feature
    ):
        return COUNT_AGGREGATE

    if _DELTA_PATTERN.fullmatch(
        normalized_feature
    ):
        return TIME_DELTA

    if _MATCH_PATTERN.fullmatch(
        normalized_feature
    ):
        return MATCH_FLAG

    if _IDENTITY_PATTERN.fullmatch(
        normalized_feature
    ):
        return DEVICE_IDENTITY

    anonymized_match = (
        _ANONYMIZED_PATTERN.fullmatch(
            normalized_feature
        )
    )

    if (
        anonymized_match is not None
        and 1
        <= int(anonymized_match.group(1))
        <= 339
    ):
        return ANONYMIZED_BEHAVIOR

    return OTHER_MODEL_SIGNAL


def _reason_code_for(
    contribution: FeatureContribution,
) -> str:
    """Build the stable reason-code identifier."""

    return "_".join(
        (
            contribution.feature_group,
            contribution.value_state,
            contribution.direction,
        )
    )


def _reason_message_for(
    contribution: FeatureContribution,
) -> str:
    """Build a deterministic non-causal analyst message."""

    label = _FEATURE_GROUP_LABELS[
        contribution.feature_group
    ]

    if contribution.direction == INCREASES_SCORE:
        direction_verb = "increased"
    elif (
        contribution.direction
        == DECREASES_SCORE
    ):
        direction_verb = "decreased"
    else:
        raise ValueError(
            "Unsupported contribution direction"
        )

    if contribution.value_state == OBSERVED:
        return (
            f"{label} signal {direction_verb} "
            "the model score."
        )

    if contribution.value_state == MISSING:
        return (
            f"{label} information was missing "
            f"and {direction_verb} the model score."
        )

    if (
        contribution.value_state
        == UNKNOWN_CATEGORY
    ):
        return (
            f"{label} contained a category not "
            "seen in training and "
            f"{direction_verb} the model score."
        )

    raise ValueError(
        "Unsupported contribution value state"
    )


@dataclass(frozen=True)
class AnalystReason:
    """One stable analyst-facing reason for a model signal."""

    contribution: FeatureContribution
    code: str
    message: str
    reason_code_version: str = (
        REASON_CODE_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.contribution,
            FeatureContribution,
        ):
            raise ValueError(
                "contribution must be a "
                "FeatureContribution instance"
            )

        expected_group = feature_group_for(
            self.contribution.feature
        )

        if (
            self.contribution.feature_group
            != expected_group
        ):
            raise ValueError(
                "contribution feature_group does not "
                "match the stable feature mapping"
            )

        if (
            self.reason_code_version
            != REASON_CODE_VERSION
        ):
            raise ValueError(
                "reason_code_version does not match "
                "the supported version"
            )

        expected_code = _reason_code_for(
            self.contribution
        )

        if self.code != expected_code:
            raise ValueError(
                "code does not match the stable "
                "reason-code grammar"
            )

        expected_message = _reason_message_for(
            self.contribution
        )

        if self.message != expected_message:
            raise ValueError(
                "message does not match the stable "
                "analyst template"
            )


def build_analyst_reason(
    contribution: FeatureContribution,
) -> AnalystReason:
    """Convert one validated contribution to a stable reason."""

    if not isinstance(
        contribution,
        FeatureContribution,
    ):
        raise ValueError(
            "contribution must be a "
            "FeatureContribution instance"
        )

    expected_group = feature_group_for(
        contribution.feature
    )

    if contribution.feature_group != expected_group:
        raise ValueError(
            "contribution feature_group does not "
            "match the stable feature mapping"
        )

    return AnalystReason(
        contribution=contribution,
        code=_reason_code_for(
            contribution
        ),
        message=_reason_message_for(
            contribution
        ),
    )
