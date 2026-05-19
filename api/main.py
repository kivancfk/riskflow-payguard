"""FastAPI service entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from api import model_loader
from api.config import settings
from api.decision_engine import decide
from api.logging_db import SessionLocal, init_db, log_prediction
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RecentPrediction,
    RecentPredictionsResponse,
    RiskFactor,
    ThresholdSimulationRequest,
    ThresholdSimulationResponse,
    TransactionFeatures,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    try:
        model_loader.load_model(settings.model_path)
    except FileNotFoundError as exc:
        # Allow the API to boot without a model so /health still answers honestly.
        print(f"[warn] {exc}")
    yield


app = FastAPI(
    title="RiskFlow PayGuard API",
    description="Real-Time Payment Fraud Risk Scoring System",
    version="0.1.0",
    lifespan=lifespan,
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model_loader.is_loaded())


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if not model_loader.is_loaded():
        raise HTTPException(503, "Model not loaded")
    m = model_loader.get_model()
    return ModelInfoResponse(
        model_version=m.model_version,
        trained_at=None,  # TODO: persist in artifact
        metrics=m.metrics,
        feature_names=m.feature_names,
    )


def _score_one(tx: TransactionFeatures, db: Session) -> PredictionResponse:
    if not model_loader.is_loaded():
        raise HTTPException(503, "Model not loaded")
    m = model_loader.get_model()

    # TODO: real feature pipeline goes here (src.features.build_features)
    # For now this is a placeholder shape — replace once the model is trained.
    raise HTTPException(501, "Scoring pipeline not yet implemented. See src/features.py.")

    # The shape the implementation should return:
    # proba = float(m.model.predict_proba(X)[0, 1])
    # decision = decide(proba, settings.threshold_review, settings.threshold_block)
    # risk_factors = [...]  # from m.explainer
    # log_prediction(db, ...)
    # return PredictionResponse(...)


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionFeatures, db: Session = Depends(get_db)) -> PredictionResponse:
    return _score_one(tx, db)


@app.post("/batch-predict", response_model=BatchPredictResponse)
def batch_predict(
    req: BatchPredictRequest, db: Session = Depends(get_db)
) -> BatchPredictResponse:
    return BatchPredictResponse(predictions=[_score_one(tx, db) for tx in req.transactions])


@app.post("/threshold-simulation", response_model=ThresholdSimulationResponse)
def threshold_simulation(req: ThresholdSimulationRequest) -> ThresholdSimulationResponse:
    """Simulate fraud-caught / FP / review-load / cost for a (review, block) threshold pair.

    TODO: implement against a held-out validation set loaded at startup.
    """
    raise HTTPException(501, "Threshold simulation not yet implemented.")


@app.get("/recent-predictions", response_model=RecentPredictionsResponse)
def recent_predictions(
    limit: int = 100, db: Session = Depends(get_db)
) -> RecentPredictionsResponse:
    """Most recent predictions, newest first. Used by the dashboard's overview page."""
    from api.logging_db import PredictionLog

    limit = max(1, min(limit, 1000))
    rows = (
        db.query(PredictionLog)
        .order_by(PredictionLog.scored_at.desc())
        .limit(limit)
        .all()
    )
    return RecentPredictionsResponse(
        predictions=[
            RecentPrediction(
                transaction_id=r.transaction_id,
                fraud_probability=r.fraud_probability,
                decision=r.decision,  # type: ignore[arg-type]
                model_version=r.model_version,
                scored_at=r.scored_at,
            )
            for r in rows
        ]
    )
