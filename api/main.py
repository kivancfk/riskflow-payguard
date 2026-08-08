"""FastAPI entrypoint for the frozen PayGuard inference service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request

from api import model_loader
from api.config import (
    FROZEN_EXPLANATION_VERSION,
    FROZEN_REASON_CODE_VERSION,
    settings,
)
from api.feature_frames import (
    _validate_api_feature_contract,
)
from api.logging_db import (
    PredictionStore,
    create_prediction_store,
)
from api.persistence import (
    build_prediction_events,
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
async def lifespan(app: FastAPI):
    """Load frozen inference policy and initialize prediction persistence."""

    bundle = model_loader.load_policy(
        settings.policy_path
    )

    _validate_api_feature_contract(
        bundle
    )

    prediction_store = (
        create_prediction_store()
    )
    prediction_store.init_schema()

    app.state.prediction_store = (
        prediction_store
    )

    try:
        yield

    finally:
        prediction_store.dispose()

        if hasattr(
            app.state,
            "prediction_store",
        ):
            delattr(
                app.state,
                "prediction_store",
            )


app = FastAPI(
    title="RiskFlow PayGuard API",
    description=(
        "Frozen payment-fraud scoring, calibrated policy, "
        "and deterministic explanation service."
    ),
    version="0.6.0",
    lifespan=lifespan,
)


def _get_prediction_store(
    request: Request,
) -> PredictionStore:
    """Return the prediction store initialized by application startup."""

    prediction_store = getattr(
        request.app.state,
        "prediction_store",
        None,
    )

    if prediction_store is None:
        raise RuntimeError(
            "Prediction store is not initialized"
        )

    return prediction_store


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
    request: Request,
) -> PredictionResponse:
    """Score, explain, and persist one frozen-policy prediction."""

    prediction = predict_transaction(
        transaction,
        model_loader.get_policy(),
    )

    events = build_prediction_events(
        [
            transaction
        ],
        [
            prediction
        ],
        model_loader.get_loaded_policy(),
    )

    _get_prediction_store(
        request
    ).add_events(
        events
    )

    return prediction


@app.post(
    "/batch-predict",
    response_model=BatchPredictResponse,
)
def batch_predict(
    request: BatchPredictRequest,
    http_request: Request,
) -> BatchPredictResponse:
    """Score, explain, and atomically persist an ordered transaction batch."""

    response = predict_transactions(
        request.transactions,
        model_loader.get_policy(),
    )

    events = build_prediction_events(
        request.transactions,
        response.predictions,
        model_loader.get_loaded_policy(),
    )

    _get_prediction_store(
        http_request
    ).add_events(
        events
    )

    return response
