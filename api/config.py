"""Application settings loaded from environment / .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_path: str = "models/fraud_model.pkl"
    database_url: str = "sqlite:///./predictions.db"

    threshold_review: float = 0.30
    threshold_block: float = 0.70

    cost_per_false_positive: float = 5.0
    cost_per_false_negative: float = 100.0
    cost_per_review: float = 2.0


settings = Settings()
