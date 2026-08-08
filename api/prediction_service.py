"""Single-transaction frozen policy inference for the PayGuard API."""

from __future__ import annotations

from api.feature_frames import (
    build_feature_frame,
)
from api.schemas import (
    AnalystReasonResponse,
    FeatureContributionResponse,
    PredictionResponse,
    ReconstructionResponse,
    TransactionRequest,
)
from src.explainability import (
    FeatureContribution,
)
from src.policy_bundle import (
    CalibratedPolicyBundle,
)
from src.policy_explanations import (
    PolicyExplanationRecord,
    predict_policy_with_explanations,
)
from src.reason_codes import (
    AnalystReason,
)


def _contribution_response(
    contribution: FeatureContribution,
) -> FeatureContributionResponse:
    """Serialize one validated Phase 4 feature contribution."""

    return FeatureContributionResponse(
        feature=contribution.feature,
        feature_index=(
            contribution.feature_index
        ),
        feature_group=(
            contribution.feature_group
        ),
        direction=contribution.direction,
        shap_value_raw=(
            contribution.shap_value_raw
        ),
        absolute_shap_value_raw=(
            contribution
            .absolute_shap_value_raw
        ),
        value_state=(
            contribution.value_state
        ),
        rank=contribution.rank,
    )


def _reason_response(
    reason: AnalystReason,
) -> AnalystReasonResponse:
    """Serialize one deterministic analyst-facing reason."""

    return AnalystReasonResponse(
        code=reason.code,
        message=reason.message,
    )


def prediction_response_from_record(
    record: PolicyExplanationRecord,
) -> PredictionResponse:
    """Serialize one complete Phase 4 prediction record without recomputation."""

    return PredictionResponse(
        transaction_id=(
            record.transaction_id
        ),
        model_version=(
            record.model_version
        ),
        policy_version=(
            record.policy_version
        ),
        explanation_version=(
            record.explanation_version
        ),
        reason_code_version=(
            record.reason_code_version
        ),
        raw_model_score=(
            record.raw_model_score
        ),
        calibrated_probability=(
            record.calibrated_probability
        ),
        decision=record.decision,
        top_positive_contributions=[
            _contribution_response(
                contribution
            )
            for contribution in (
                record
                .top_positive_contributions
            )
        ],
        top_negative_contributions=[
            _contribution_response(
                contribution
            )
            for contribution in (
                record
                .top_negative_contributions
            )
        ],
        reason_codes=list(
            record.reason_codes
        ),
        reasons=[
            _reason_response(
                reason
            )
            for reason in record.reasons
        ],
        reconstruction=(
            ReconstructionResponse(
                raw_model_margin=(
                    record.raw_model_margin
                ),
                expected_value_raw=(
                    record.expected_value_raw
                ),
                shap_sum_raw=(
                    record.shap_sum_raw
                ),
                reconstructed_raw_margin=(
                    record
                    .reconstructed_raw_margin
                ),
                reconstructed_raw_model_score=(
                    record
                    .reconstructed_raw_model_score
                ),
                margin_reconstruction_error=(
                    record
                    .margin_reconstruction_error
                ),
                score_reconstruction_error=(
                    record
                    .score_reconstruction_error
                ),
            )
        ),
    )


def predict_transaction(
    transaction: TransactionRequest,
    bundle: CalibratedPolicyBundle,
) -> PredictionResponse:
    """Run one validated API transaction through the frozen Phase 4 pipeline."""

    feature_frame = build_feature_frame(
        transaction,
        bundle,
    )

    explained_prediction = (
        predict_policy_with_explanations(
            bundle,
            [
                transaction.transaction_id,
            ],
            feature_frame,
            frame_name=(
                "API /predict features"
            ),
        )
    )

    if len(
        explained_prediction.explanations
    ) != 1:
        raise RuntimeError(
            "Single prediction inference returned "
            "an unexpected explanation count"
        )

    record = (
        explained_prediction
        .explanations[0]
    )

    if (
        record.transaction_id
        != transaction.transaction_id
    ):
        raise RuntimeError(
            "Single prediction transaction identifier "
            "does not match the request"
        )

    return prediction_response_from_record(
        record
    )
