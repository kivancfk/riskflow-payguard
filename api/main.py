"""FastAPI entrypoint for the frozen PayGuard inference service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import model_loader
from api.config import (
    FROZEN_EXPLANATION_VERSION,
    FROZEN_REASON_CODE_VERSION,
    settings,
)
from api.feature_frames import (
    _validate_api_feature_contract,
)
from api.prediction_service import (
    predict_transaction,
    predict_transactions,
)
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load and validate the frozen inference policy before serving requests."""

    bundle = model_loader.load_policy(
        settings.policy_path
    )

    _validate_api_feature_contract(
        bundle
    )

    yield


app = FastAPI(
    title="RiskFlow PayGuard API",
    description=(
        "Frozen payment-fraud scoring, calibrated policy, "
        "and deterministic explanation service."
    ),
    version="0.5.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Report health after successful frozen-policy startup."""

    return HealthResponse(
        status="ok",
        model_loaded=model_loader.is_loaded(),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info() -> ModelInfoResponse:
    """Return immutable metadata for the loaded frozen inference policy."""

    loaded_policy = (
        model_loader.get_loaded_policy()
    )
    bundle = loaded_policy.bundle
    baseline_bundle = (
        bundle.baseline_bundle
    )

    return ModelInfoResponse(
        baseline_model_version=(
            bundle.baseline_model_version
        ),
        policy_version=(
            bundle.policy_version
        ),
        calibration_method=(
            bundle.calibration_selection
            .selected_method
        ),
        review_threshold=(
            bundle.thresholds
            .review_threshold
        ),
        block_threshold=(
            bundle.thresholds
            .block_threshold
        ),
        explanation_version=(
            FROZEN_EXPLANATION_VERSION
        ),
        reason_code_version=(
            FROZEN_REASON_CODE_VERSION
        ),
        policy_artifact_sha256=(
            loaded_policy
            .artifact_sha256
        ),
        feature_names=list(
            baseline_bundle
            .feature_columns
        ),
        feature_count=len(
            baseline_bundle
            .feature_columns
        ),
        categorical_feature_count=len(
            baseline_bundle
            .categorical_features
        ),
        numerical_feature_count=len(
            baseline_bundle
            .numerical_features
        ),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    transaction: TransactionRequest,
) -> PredictionResponse:
    """Score and explain one transaction using the frozen policy."""

    return predict_transaction(
        transaction,
        model_loader.get_policy(),
    )


@app.post(
    "/batch-predict",
    response_model=BatchPredictResponse,
)
def batch_predict(
    request: BatchPredictRequest,
) -> BatchPredictResponse:
    """Score and explain an ordered transaction batch using one inference call."""

    return predict_transactions(
        request.transactions,
        model_loader.get_policy(),
    )
