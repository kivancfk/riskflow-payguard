"""Strict Pydantic contracts for the Phase 5 PayGuard API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from typing import TypeAlias

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import create_model
from pydantic import field_validator
from pydantic import model_validator

from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
)


Decision = Literal[
    "ALLOW",
    "REVIEW",
    "BLOCK",
]

ContributionDirection = Literal[
    "INCREASES_SCORE",
    "DECREASES_SCORE",
]

FeatureValueState = Literal[
    "OBSERVED",
    "MISSING",
    "UNKNOWN_CATEGORY",
]

TransactionIdentifier: TypeAlias = str | int
CategoricalFeatureValue: TypeAlias = (
    str | int | float | None
)
NumericalFeatureValue: TypeAlias = (
    int | float | None
)


class StrictAPIModel(BaseModel):
    """Base model for strict Phase 5 public API contracts."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


def _feature_field_definitions() -> dict[
    str,
    tuple[object, object],
]:
    """Build required typed fields for the frozen Phase 1 feature contract."""

    categorical = set(
        CATEGORICAL_FEATURES
    )
    numerical = set(
        NUMERICAL_FEATURES
    )

    if categorical.intersection(
        numerical
    ):
        raise RuntimeError(
            "Categorical and numerical API feature sets overlap"
        )

    if categorical.union(
        numerical
    ) != set(FEATURE_COLUMNS):
        raise RuntimeError(
            "Categorical and numerical API feature sets do not "
            "cover the complete feature contract"
        )

    definitions: dict[
        str,
        tuple[object, object],
    ] = {}

    for feature_name in FEATURE_COLUMNS:
        if feature_name in categorical:
            annotation = CategoricalFeatureValue
        else:
            annotation = NumericalFeatureValue

        definitions[
            feature_name
        ] = (
            annotation,
            ...,
        )

    return definitions


TransactionModelFeatures = create_model(
    "TransactionModelFeatures",
    __base__=StrictAPIModel,
    **_feature_field_definitions(),
)


class TransactionRequest(StrictAPIModel):
    """One transaction identifier and its complete raw model feature vector."""

    transaction_id: TransactionIdentifier
    features: TransactionModelFeatures

    @field_validator(
        "transaction_id",
        mode="before",
    )
    @classmethod
    def validate_transaction_id(
        cls,
        value: object,
    ) -> object:
        """Apply the stable Phase 4 transaction-identifier contract."""

        if isinstance(value, bool):
            raise ValueError(
                "transaction_id must be a string or integer"
            )

        if isinstance(value, str):
            if (
                not value
                or value != value.strip()
            ):
                raise ValueError(
                    "transaction_id must be a non-empty string "
                    "without surrounding whitespace"
                )

            return value

        if isinstance(value, int):
            return value

        raise ValueError(
            "transaction_id must be a string or integer"
        )


# Temporary compatibility name used by the pre-Phase-5 API scaffold.
TransactionFeatures = TransactionRequest


class BatchPredictRequest(StrictAPIModel):
    """A non-empty ordered batch of unique transactions."""

    transactions: list[
        TransactionRequest
    ] = Field(
        ...,
        min_length=1,
    )

    @model_validator(
        mode="after"
    )
    def validate_unique_transaction_ids(
        self,
    ) -> "BatchPredictRequest":
        """Require unique identifiers within one request batch."""

        transaction_ids = [
            transaction.transaction_id
            for transaction in self.transactions
        ]

        if len(transaction_ids) != len(
            set(transaction_ids)
        ):
            raise ValueError(
                "transaction_id values must be unique within a batch"
            )

        return self


class FeatureContributionResponse(StrictAPIModel):
    """One selected TreeSHAP feature contribution."""

    feature: str
    feature_index: int = Field(
        ...,
        ge=0,
    )
    feature_group: str
    direction: ContributionDirection
    shap_value_raw: float
    absolute_shap_value_raw: float = Field(
        ...,
        ge=0.0,
    )
    value_state: FeatureValueState
    rank: int = Field(
        ...,
        ge=1,
    )


class AnalystReasonResponse(StrictAPIModel):
    """One deterministic analyst-facing reason."""

    code: str
    message: str


class ReconstructionResponse(StrictAPIModel):
    """TreeSHAP reconstruction metadata for the raw LightGBM output."""

    raw_model_margin: float
    expected_value_raw: float
    shap_sum_raw: float
    reconstructed_raw_margin: float
    reconstructed_raw_model_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )
    margin_reconstruction_error: float = Field(
        ...,
        ge=0.0,
    )
    score_reconstruction_error: float = Field(
        ...,
        ge=0.0,
    )


class PredictionResponse(StrictAPIModel):
    """One frozen policy prediction with deterministic explanations."""

    transaction_id: TransactionIdentifier

    model_version: str
    policy_version: str
    explanation_version: str
    reason_code_version: str

    raw_model_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )
    calibrated_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )
    decision: Decision

    top_positive_contributions: list[
        FeatureContributionResponse
    ]
    top_negative_contributions: list[
        FeatureContributionResponse
    ]

    reason_codes: list[str]
    reasons: list[
        AnalystReasonResponse
    ]

    reconstruction: ReconstructionResponse


class BatchPredictResponse(StrictAPIModel):
    """Ordered predictions corresponding one-for-one with batch input."""

    predictions: list[
        PredictionResponse
    ]


class HealthResponse(StrictAPIModel):
    """Application health response."""

    status: Literal["ok"]
    model_loaded: bool


class ModelInfoResponse(StrictAPIModel):
    """Immutable information about the loaded frozen inference policy."""

    baseline_model_version: str
    policy_version: str
    calibration_method: str

    review_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )
    block_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    explanation_version: str
    reason_code_version: str
    policy_artifact_sha256: str

    feature_names: list[str]
    feature_count: int = Field(
        ...,
        ge=1,
    )
    categorical_feature_count: int = Field(
        ...,
        ge=1,
    )
    numerical_feature_count: int = Field(
        ...,
        ge=1,
    )


# ---------------------------------------------------------------------------
# Legacy scaffold schemas
#
# These remain temporarily so api.main can still import before its focused
# Phase 5 refactor in Commit 4. They are not part of the new inference path.
# ---------------------------------------------------------------------------


class RiskFactor(BaseModel):
    """Legacy risk-factor placeholder retained until api.main is refactored."""

    feature: str
    shap_value: float
    direction: Literal[
        "increases_risk",
        "decreases_risk",
    ]


class ThresholdSimulationRequest(BaseModel):
    """Legacy out-of-scope threshold-simulation request."""

    review_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )
    block_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


class ThresholdSimulationResponse(BaseModel):
    """Legacy out-of-scope threshold-simulation response."""

    review_threshold: float
    block_threshold: float
    fraud_caught_rate: float
    false_positive_rate: float
    review_workload_pct: float
    estimated_business_cost: float


class RecentPrediction(BaseModel):
    """Legacy persisted-prediction response."""

    transaction_id: str
    fraud_probability: float
    decision: Decision
    model_version: str
    scored_at: datetime


class RecentPredictionsResponse(BaseModel):
    """Legacy persisted-prediction collection."""

    predictions: list[
        RecentPrediction
    ]
