"""Persists every prediction for offline analysis and the dashboard."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.config import settings


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "predictions"

    transaction_id: str = Column(String, primary_key=True)
    fraud_probability: float = Column(Float, nullable=False)
    decision: str = Column(String, nullable=False)
    model_version: str = Column(String, nullable=False)
    features: dict = Column(JSON, nullable=False)
    top_risk_factors: dict = Column(JSON, nullable=False)
    scored_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    actual_label: int | None = Column(Float, nullable=True)  # backfilled later


engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def log_prediction(
    session: Session,
    *,
    transaction_id: str,
    fraud_probability: float,
    decision: str,
    model_version: str,
    features: dict,
    top_risk_factors: list[dict],
) -> None:
    row = PredictionLog(
        transaction_id=transaction_id,
        fraud_probability=fraud_probability,
        decision=decision,
        model_version=model_version,
        features=features,
        top_risk_factors={"factors": top_risk_factors},
    )
    session.merge(row)
    session.commit()
