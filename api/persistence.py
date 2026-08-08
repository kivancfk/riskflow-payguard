"""Translate frozen API predictions into Phase 6 persistence events."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from datetime import timezone

from api.logging_db import (
    PredictionEventCreate,
)
from api.model_loader import LoadedPolicy
from api.schemas import (
    PredictionResponse,
    TransactionRequest,
)
from src.data_processing import AMOUNT_COLUMN


def build_prediction_events(
    transactions: Sequence[
        TransactionRequest
    ],
    predictions: Sequence[
        PredictionResponse
    ],
    loaded_policy: LoadedPolicy,
) -> tuple[
    PredictionEventCreate,
    ...,
]:
    """Build append-only events from successful frozen predictions."""

    if len(transactions) != len(
        predictions
    ):
        raise RuntimeError(
            "Prediction persistence requires one output "
            "for every input transaction"
        )

    bundle = loaded_policy.bundle

    scored_at_utc = datetime.now(
        timezone.utc
    )

    events: list[
        PredictionEventCreate
    ] = []

    for (
        transaction,
        prediction,
    ) in zip(
        transactions,
        predictions,
    ):
        if (
            transaction.transaction_id
            != prediction.transaction_id
        ):
            raise RuntimeError(
                "Prediction persistence transaction order "
                "does not match inference output"
            )

        raw_transaction_amount = (
            getattr(
                transaction.features,
                AMOUNT_COLUMN,
            )
        )

        transaction_amount = (
            None
            if raw_transaction_amount
            is None
            else float(
                raw_transaction_amount
            )
        )

        events.append(
            PredictionEventCreate(
                transaction_id=(
                    prediction.transaction_id
                ),
                raw_model_score=(
                    prediction.raw_model_score
                ),
                calibrated_probability=(
                    prediction
                    .calibrated_probability
                ),
                decision=(
                    prediction.decision
                ),
                transaction_amount=(
                    transaction_amount
                ),
                model_version=(
                    prediction.model_version
                ),
                policy_version=(
                    prediction.policy_version
                ),
                calibration_method=(
                    bundle
                    .calibration_selection
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
                policy_artifact_sha256=(
                    loaded_policy
                    .artifact_sha256
                ),
                explanation_version=(
                    prediction
                    .explanation_version
                ),
                reason_code_version=(
                    prediction
                    .reason_code_version
                ),
                reason_codes=tuple(
                    prediction.reason_codes
                ),
                scored_at_utc=(
                    scored_at_utc
                ),
            )
        )

    return tuple(
        events
    )
