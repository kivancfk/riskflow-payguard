"""Pydantic schemas for API request and response bodies."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Decision = Literal["ALLOW", "REVIEW", "BLOCK"]


class TransactionFeatures(BaseModel):
    """Minimal transaction payload. Extend with the actual IEEE-CIS features after EDA."""
    transaction_id: str = Field(..., description="Unique merchant-side transaction id")
    transaction_amount: float = Field(..., ge=0)
    product_code: str | None = None
    card_type: str | None = None
    card_issuer_country: str | None = None
    purchaser_email_domain: str | None = None
    device_type: str | None = None
    hour_of_day: int | None = Field(None, ge=0, le=23)


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    decision: Decision
    top_risk_factors: list[RiskFactor]
    model_version: str
    scored_at: datetime


class BatchPredictRequest(BaseModel):
    transactions: list[TransactionFeatures]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResponse]


class ThresholdSimulationRequest(BaseModel):
    review_threshold: float = Field(..., ge=0.0, le=1.0)
    block_threshold: float = Field(..., ge=0.0, le=1.0)


class ThresholdSimulationResponse(BaseModel):
    review_threshold: float
    block_threshold: float
    fraud_caught_rate: float
    false_positive_rate: float
    review_workload_pct: float
    estimated_business_cost: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: datetime | None
    metrics: dict[str, float]
    feature_names: list[str]


class RecentPrediction(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision: Decision
    model_version: str
    scored_at: datetime


class RecentPredictionsResponse(BaseModel):
    predictions: list[RecentPrediction]
