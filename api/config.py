"""Application configuration for the PayGuard FastAPI service."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


FROZEN_BASELINE_MODEL_VERSION = "baseline-v1"
FROZEN_POLICY_VERSION = "calibrated-policy-v1"
FROZEN_CALIBRATION_METHOD = "sigmoid"

FROZEN_REVIEW_THRESHOLD = 0.16255069862369795
FROZEN_BLOCK_THRESHOLD = 0.8509223095305902

FROZEN_EXPLANATION_VERSION = "shap-explanation-v1"
FROZEN_REASON_CODE_VERSION = "reason-codes-v1"

FROZEN_POLICY_ARTIFACT = Path(
    "models/payguard_calibrated_policy.joblib"
)

FROZEN_POLICY_SHA256 = (
    "5d53f23719ae891ecc24585393585765aa"
    "7fc0900ab38f95e37f59c18fe6c90f"
)


class Settings(BaseSettings):
    """Runtime settings that do not redefine the frozen decision policy."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    policy_path: Path = FROZEN_POLICY_ARTIFACT

    # Retained temporarily because the pre-Phase-5 API scaffold still imports
    # database configuration. Database prediction logging is outside Phase 5
    # and will not participate in the active inference path.
    database_url: str = "sqlite:///./predictions.db"

    @property
    def model_path(self) -> Path:
        """Compatibility alias for the legacy API startup scaffold."""

        return self.policy_path

    @property
    def threshold_review(self) -> float:
        """Compatibility alias for the frozen review threshold."""

        return FROZEN_REVIEW_THRESHOLD

    @property
    def threshold_block(self) -> float:
        """Compatibility alias for the frozen block threshold."""

        return FROZEN_BLOCK_THRESHOLD


settings = Settings()
