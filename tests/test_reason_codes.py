"""Tests for stable analyst-facing reason codes."""

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
    REASON_CODE_VERSION,
    UNKNOWN_CATEGORY,
    FeatureContribution,
)
from src.features import FEATURE_COLUMNS
from src.reason_codes import (
    ADDRESS,
    ANONYMIZED_BEHAVIOR,
    COUNT_AGGREGATE,
    DEVICE_IDENTITY,
    DISTANCE,
    EMAIL_DOMAIN,
    MATCH_FLAG,
    OTHER_MODEL_SIGNAL,
    PAYMENT_INSTRUMENT,
    TIME_DELTA,
    TRANSACTION_AMOUNT,
    TRANSACTION_TIME,
    AnalystReason,
    build_analyst_reason,
    feature_group_for,
)


@pytest.mark.parametrize(
    ("feature", "expected_group"),
    [
        (
            "TransactionAmt",
            TRANSACTION_AMOUNT,
        ),
        (
            "transaction_amount_log1p",
            TRANSACTION_AMOUNT,
        ),
        (
            "TransactionDT",
            TRANSACTION_TIME,
        ),
        (
            "relative_day",
            TRANSACTION_TIME,
        ),
        (
            "relative_hour_cycle",
            TRANSACTION_TIME,
        ),
        (
            "ProductCD",
            PAYMENT_INSTRUMENT,
        ),
        (
            "card1",
            PAYMENT_INSTRUMENT,
        ),
        (
            "card6",
            PAYMENT_INSTRUMENT,
        ),
        (
            "addr1",
            ADDRESS,
        ),
        (
            "addr2",
            ADDRESS,
        ),
        (
            "dist1",
            DISTANCE,
        ),
        (
            "dist2",
            DISTANCE,
        ),
        (
            "P_emaildomain",
            EMAIL_DOMAIN,
        ),
        (
            "R_emaildomain",
            EMAIL_DOMAIN,
        ),
        (
            "C1",
            COUNT_AGGREGATE,
        ),
        (
            "C14",
            COUNT_AGGREGATE,
        ),
        (
            "D1",
            TIME_DELTA,
        ),
        (
            "D15",
            TIME_DELTA,
        ),
        (
            "M1",
            MATCH_FLAG,
        ),
        (
            "M9",
            MATCH_FLAG,
        ),
        (
            "DeviceType",
            DEVICE_IDENTITY,
        ),
        (
            "DeviceInfo",
            DEVICE_IDENTITY,
        ),
        (
            "id_01",
            DEVICE_IDENTITY,
        ),
        (
            "id_38",
            DEVICE_IDENTITY,
        ),
        (
            "identity_availability",
            DEVICE_IDENTITY,
        ),
        (
            "V1",
            ANONYMIZED_BEHAVIOR,
        ),
        (
            "V339",
            ANONYMIZED_BEHAVIOR,
        ),
    ],
)
def test_feature_group_mapping(
    feature: str,
    expected_group: str,
) -> None:
    assert (
        feature_group_for(feature)
        == expected_group
    )


@pytest.mark.parametrize(
    "feature",
    [
        "card7",
        "C15",
        "D16",
        "M10",
        "id_1",
        "V0",
        "V340",
        "custom_feature",
    ],
)
def test_unmapped_features_use_explicit_fallback(
    feature: str,
) -> None:
    assert (
        feature_group_for(feature)
        == OTHER_MODEL_SIGNAL
    )


@pytest.mark.parametrize(
    "feature",
    [
        "",
        " ",
        " card1",
        "card1 ",
    ],
)
def test_feature_group_rejects_invalid_names(
    feature: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="without surrounding whitespace",
    ):
        feature_group_for(feature)


def test_all_frozen_features_have_specialized_groups() -> None:
    assert len(FEATURE_COLUMNS) == 63

    fallback_features = [
        feature
        for feature in FEATURE_COLUMNS
        if (
            feature_group_for(feature)
            == OTHER_MODEL_SIGNAL
        )
    ]

    assert fallback_features == []


def _contribution(
    *,
    feature: str,
    direction: str,
    value_state: str,
    feature_group: str | None = None,
) -> FeatureContribution:
    """Return one contribution for reason-code tests."""

    shap_value_raw = (
        0.25
        if direction == INCREASES_SCORE
        else -0.25
    )

    return FeatureContribution(
        feature=feature,
        feature_index=4,
        feature_group=(
            feature_group
            if feature_group is not None
            else feature_group_for(feature)
        ),
        direction=direction,
        shap_value_raw=shap_value_raw,
        value_state=value_state,
        rank=1,
    )


@pytest.mark.parametrize(
    (
        "contribution",
        "expected_code",
        "expected_message",
    ),
    [
        (
            _contribution(
                feature="TransactionAmt",
                direction=INCREASES_SCORE,
                value_state=OBSERVED,
            ),
            (
                "TRANSACTION_AMOUNT_"
                "OBSERVED_INCREASES_SCORE"
            ),
            (
                "Transaction amount signal increased "
                "the model score."
            ),
        ),
        (
            _contribution(
                feature="D1",
                direction=DECREASES_SCORE,
                value_state=MISSING,
            ),
            (
                "TIME_DELTA_"
                "MISSING_DECREASES_SCORE"
            ),
            (
                "Time delta information was missing "
                "and decreased the model score."
            ),
        ),
        (
            _contribution(
                feature="DeviceType",
                direction=INCREASES_SCORE,
                value_state=UNKNOWN_CATEGORY,
            ),
            (
                "DEVICE_IDENTITY_UNKNOWN_CATEGORY_"
                "INCREASES_SCORE"
            ),
            (
                "Device identity contained a category "
                "not seen in training and increased "
                "the model score."
            ),
        ),
    ],
)
def test_build_analyst_reason_uses_stable_contract(
    contribution: FeatureContribution,
    expected_code: str,
    expected_message: str,
) -> None:
    reason = build_analyst_reason(
        contribution
    )

    assert reason.code == expected_code
    assert reason.message == expected_message
    assert (
        reason.reason_code_version
        == REASON_CODE_VERSION
    )
    assert reason.contribution == contribution


def test_reason_is_immutable_and_json_compatible() -> None:
    reason = build_analyst_reason(
        _contribution(
            feature="P_emaildomain",
            direction=DECREASES_SCORE,
            value_state=OBSERVED,
        )
    )

    with pytest.raises(FrozenInstanceError):
        reason.code = "CHANGED"  # type: ignore[misc]

    json.dumps(
        asdict(reason)
    )


def test_reason_rejects_mismatched_feature_group() -> None:
    contribution = _contribution(
        feature="TransactionAmt",
        feature_group=DEVICE_IDENTITY,
        direction=INCREASES_SCORE,
        value_state=OBSERVED,
    )

    with pytest.raises(
        ValueError,
        match="does not match the stable",
    ):
        build_analyst_reason(
            contribution
        )


def test_reason_constructor_rejects_modified_code() -> None:
    contribution = _contribution(
        feature="addr1",
        direction=INCREASES_SCORE,
        value_state=OBSERVED,
    )
    valid_reason = build_analyst_reason(
        contribution
    )

    with pytest.raises(
        ValueError,
        match="code does not match",
    ):
        AnalystReason(
            contribution=contribution,
            code="ADDRESS_OBSERVED_DECREASES_SCORE",
            message=valid_reason.message,
        )


def test_reason_constructor_rejects_modified_message() -> None:
    contribution = _contribution(
        feature="C1",
        direction=DECREASES_SCORE,
        value_state=MISSING,
    )
    valid_reason = build_analyst_reason(
        contribution
    )

    with pytest.raises(
        ValueError,
        match="message does not match",
    ):
        AnalystReason(
            contribution=contribution,
            code=valid_reason.code,
            message="This feature proves fraud.",
        )


def test_reason_constructor_rejects_modified_version() -> None:
    contribution = _contribution(
        feature="M1",
        direction=INCREASES_SCORE,
        value_state=OBSERVED,
    )
    valid_reason = build_analyst_reason(
        contribution
    )

    with pytest.raises(
        ValueError,
        match="reason_code_version",
    ):
        AnalystReason(
            contribution=contribution,
            code=valid_reason.code,
            message=valid_reason.message,
            reason_code_version="reason-codes-v2",
        )


@pytest.mark.parametrize(
    (
        "direction",
        "value_state",
    ),
    [
        (
            INCREASES_SCORE,
            OBSERVED,
        ),
        (
            DECREASES_SCORE,
            OBSERVED,
        ),
        (
            INCREASES_SCORE,
            MISSING,
        ),
        (
            DECREASES_SCORE,
            MISSING,
        ),
        (
            INCREASES_SCORE,
            UNKNOWN_CATEGORY,
        ),
        (
            DECREASES_SCORE,
            UNKNOWN_CATEGORY,
        ),
    ],
)
def test_reason_messages_avoid_causal_fraud_claims(
    direction: str,
    value_state: str,
) -> None:
    reason = build_analyst_reason(
        _contribution(
            feature="id_01",
            direction=direction,
            value_state=value_state,
        )
    )

    normalized_message = (
        reason.message.lower()
    )

    for prohibited_phrase in (
        "caused",
        "proves",
        "because",
        "is fraudulent",
        "intentional",
    ):
        assert (
            prohibited_phrase
            not in normalized_message
        )
