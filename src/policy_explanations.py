"""Integrate deterministic explanations with calibrated policy inference."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from src.explainability import (
    DEFAULT_TOP_CONTRIBUTIONS,
    EXPLANATION_VERSION,
    REASON_CODE_VERSION,
    FeatureContribution,
)
from src.explanation_engine import (
    ModelSignalExplanation,
    generate_model_signal_explanations,
)
from src.lightgbm_shap import (
    extract_lightgbm_contributions,
)
from src.policy import (
    ALLOW,
    BLOCK,
    REVIEW,
    PolicyDecision,
)
from src.policy_bundle import (
    CalibratedPolicyBundle,
    _validate_policy_bundle,
)
from src.reason_codes import (
    AnalystReason,
)
from src.shap_reconstruction import (
    RECONSTRUCTION_ABSOLUTE_TOLERANCE,
    reconstruct_lightgbm_scores,
)


TransactionIdentifier = str | int


def _normalize_transaction_identifier(
    value: object,
    *,
    value_name: str,
) -> TransactionIdentifier:
    """Return one stable string or integer transaction identifier."""

    if isinstance(value, bool):
        raise ValueError(
            f"{value_name} must be a string or integer"
        )

    if isinstance(value, Integral):
        return int(value)

    if isinstance(value, str):
        if (
            not value
            or value != value.strip()
        ):
            raise ValueError(
                f"{value_name} must be a non-empty string "
                "without surrounding whitespace"
            )

        return value

    raise ValueError(
        f"{value_name} must be a string or integer"
    )


def _normalize_transaction_ids(
    transaction_ids: Iterable[object],
    *,
    expected_length: int,
) -> tuple[
    TransactionIdentifier,
    ...,
]:
    """Return complete, unique identifiers aligned to inference rows."""

    if isinstance(
        transaction_ids,
        (str, bytes),
    ):
        raise ValueError(
            "transaction_ids must be an iterable "
            "of identifiers"
        )

    try:
        materialized_ids = tuple(
            transaction_ids
        )
    except TypeError as error:
        raise ValueError(
            "transaction_ids must be an iterable "
            "of identifiers"
        ) from error

    if len(materialized_ids) != expected_length:
        raise ValueError(
            "transaction_ids must have the same length "
            "as the feature frame"
        )

    normalized_ids = tuple(
        _normalize_transaction_identifier(
            transaction_id,
            value_name=(
                f"transaction_ids[{row_position}]"
            ),
        )
        for row_position, transaction_id in enumerate(
            materialized_ids
        )
    )

    if len(normalized_ids) != len(
        set(normalized_ids)
    ):
        raise ValueError(
            "transaction_ids must be unique"
        )

    return normalized_ids


def _finite_float(
    value: object,
    *,
    value_name: str,
) -> float:
    """Return one finite floating-point value."""

    if isinstance(value, bool):
        raise ValueError(
            f"{value_name} must be numerical"
        )

    try:
        normalized_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{value_name} must be numerical"
        ) from error

    if not math.isfinite(
        normalized_value
    ):
        raise ValueError(
            f"{value_name} must be finite"
        )

    return normalized_value


def _probability(
    value: object,
    *,
    value_name: str,
) -> float:
    """Return one finite probability."""

    normalized_value = _finite_float(
        value,
        value_name=value_name,
    )

    if not 0.0 <= normalized_value <= 1.0:
        raise ValueError(
            f"{value_name} must be between 0 and 1"
        )

    return normalized_value


def _readonly_probability_vector(
    values: object,
    *,
    value_name: str,
) -> np.ndarray:
    """Return a defensive, finite, read-only probability vector."""

    try:
        vector = np.array(
            values,
            dtype="float64",
            copy=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{value_name} must contain numerical values"
        ) from error

    if vector.ndim != 1:
        raise ValueError(
            f"{value_name} must be one-dimensional"
        )

    if vector.size == 0:
        raise ValueError(
            f"{value_name} must contain at least one value"
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            f"{value_name} must contain only finite values"
        )

    if (
        (vector < 0.0).any()
        or (vector > 1.0).any()
    ):
        raise ValueError(
            f"{value_name} must be between 0 and 1"
        )

    vector.setflags(
        write=False
    )

    return vector


@dataclass(frozen=True)
class PolicyExplanationRecord:
    """One calibrated policy prediction with its model explanation."""

    transaction_id: TransactionIdentifier
    model_version: str
    policy_version: str
    explanation_version: str
    reason_code_version: str

    raw_model_margin: float
    raw_model_score: float
    calibrated_probability: float
    decision: PolicyDecision

    expected_value_raw: float
    shap_sum_raw: float
    reconstructed_raw_margin: float
    reconstructed_raw_model_score: float
    margin_reconstruction_error: float
    score_reconstruction_error: float

    top_positive_contributions: tuple[
        FeatureContribution,
        ...,
    ]
    top_negative_contributions: tuple[
        FeatureContribution,
        ...,
    ]
    reasons: tuple[
        AnalystReason,
        ...,
    ]

    def __post_init__(self) -> None:
        transaction_id = (
            _normalize_transaction_identifier(
                self.transaction_id,
                value_name="transaction_id",
            )
        )

        for value_name, value in (
            (
                "model_version",
                self.model_version,
            ),
            (
                "policy_version",
                self.policy_version,
            ),
        ):
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ValueError(
                    f"{value_name} must be a "
                    "non-empty string"
                )

        if (
            self.explanation_version
            != EXPLANATION_VERSION
        ):
            raise ValueError(
                "explanation_version does not match "
                "the supported version"
            )

        if (
            self.reason_code_version
            != REASON_CODE_VERSION
        ):
            raise ValueError(
                "reason_code_version does not match "
                "the supported version"
            )

        if self.decision not in {
            ALLOW,
            REVIEW,
            BLOCK,
        }:
            raise ValueError(
                "decision must be ALLOW, REVIEW, or BLOCK"
            )

        raw_model_margin = _finite_float(
            self.raw_model_margin,
            value_name="raw_model_margin",
        )
        raw_model_score = _probability(
            self.raw_model_score,
            value_name="raw_model_score",
        )
        calibrated_probability = _probability(
            self.calibrated_probability,
            value_name="calibrated_probability",
        )
        expected_value_raw = _finite_float(
            self.expected_value_raw,
            value_name="expected_value_raw",
        )
        shap_sum_raw = _finite_float(
            self.shap_sum_raw,
            value_name="shap_sum_raw",
        )
        reconstructed_raw_margin = (
            _finite_float(
                self.reconstructed_raw_margin,
                value_name=(
                    "reconstructed_raw_margin"
                ),
            )
        )
        reconstructed_raw_model_score = (
            _probability(
                self.reconstructed_raw_model_score,
                value_name=(
                    "reconstructed_raw_model_score"
                ),
            )
        )
        margin_reconstruction_error = (
            _finite_float(
                self.margin_reconstruction_error,
                value_name=(
                    "margin_reconstruction_error"
                ),
            )
        )
        score_reconstruction_error = (
            _finite_float(
                self.score_reconstruction_error,
                value_name=(
                    "score_reconstruction_error"
                ),
            )
        )

        if margin_reconstruction_error < 0.0:
            raise ValueError(
                "margin_reconstruction_error "
                "must be non-negative"
            )

        if score_reconstruction_error < 0.0:
            raise ValueError(
                "score_reconstruction_error "
                "must be non-negative"
            )

        if (
            abs(
                (
                    expected_value_raw
                    + shap_sum_raw
                )
                - reconstructed_raw_margin
            )
            > RECONSTRUCTION_ABSOLUTE_TOLERANCE
        ):
            raise ValueError(
                "Expected value and SHAP sum do not "
                "reconstruct the raw margin"
            )

        if (
            abs(
                raw_model_margin
                - reconstructed_raw_margin
            )
            - margin_reconstruction_error
            > 1e-15
        ):
            raise ValueError(
                "margin_reconstruction_error does not "
                "match the record values"
            )

        if (
            abs(
                raw_model_score
                - reconstructed_raw_model_score
            )
            - score_reconstruction_error
            > 1e-15
        ):
            raise ValueError(
                "score_reconstruction_error does not "
                "match the record values"
            )

        ModelSignalExplanation(
            row_position=0,
            top_positive_contributions=(
                self.top_positive_contributions
            ),
            top_negative_contributions=(
                self.top_negative_contributions
            ),
            reasons=self.reasons,
        )

        object.__setattr__(
            self,
            "transaction_id",
            transaction_id,
        )
        object.__setattr__(
            self,
            "raw_model_margin",
            raw_model_margin,
        )
        object.__setattr__(
            self,
            "raw_model_score",
            raw_model_score,
        )
        object.__setattr__(
            self,
            "calibrated_probability",
            calibrated_probability,
        )
        object.__setattr__(
            self,
            "expected_value_raw",
            expected_value_raw,
        )
        object.__setattr__(
            self,
            "shap_sum_raw",
            shap_sum_raw,
        )
        object.__setattr__(
            self,
            "reconstructed_raw_margin",
            reconstructed_raw_margin,
        )
        object.__setattr__(
            self,
            "reconstructed_raw_model_score",
            reconstructed_raw_model_score,
        )
        object.__setattr__(
            self,
            "margin_reconstruction_error",
            margin_reconstruction_error,
        )
        object.__setattr__(
            self,
            "score_reconstruction_error",
            score_reconstruction_error,
        )

    @property
    def reason_codes(
        self,
    ) -> tuple[
        str,
        ...,
    ]:
        """Return the stable deduplicated reason codes."""

        return tuple(
            reason.code
            for reason in self.reasons
        )


@dataclass(
    frozen=True,
    eq=False,
)
class ExplainedPolicyPrediction:
    """Batch policy outputs plus aligned explanation records."""

    raw_scores: np.ndarray
    calibrated_probabilities: np.ndarray
    decisions: np.ndarray
    explanations: tuple[
        PolicyExplanationRecord,
        ...,
    ]

    def __post_init__(self) -> None:
        raw_scores = (
            _readonly_probability_vector(
                self.raw_scores,
                value_name="raw_scores",
            )
        )
        calibrated_probabilities = (
            _readonly_probability_vector(
                self.calibrated_probabilities,
                value_name=(
                    "calibrated_probabilities"
                ),
            )
        )

        decisions = np.array(
            self.decisions,
            dtype="object",
            copy=True,
        )

        if decisions.ndim != 1:
            raise ValueError(
                "decisions must be one-dimensional"
            )

        if not isinstance(
            self.explanations,
            tuple,
        ):
            raise ValueError(
                "explanations must be a tuple"
            )

        row_count = len(
            raw_scores
        )

        if len(calibrated_probabilities) != row_count:
            raise ValueError(
                "calibrated_probabilities must align "
                "with raw_scores"
            )

        if len(decisions) != row_count:
            raise ValueError(
                "decisions must align with raw_scores"
            )

        if len(self.explanations) != row_count:
            raise ValueError(
                "explanations must align with raw_scores"
            )

        normalized_decisions: list[
            str
        ] = []

        for row_position, decision in enumerate(
            decisions
        ):
            normalized_decision = str(
                decision
            )

            if normalized_decision not in {
                ALLOW,
                REVIEW,
                BLOCK,
            }:
                raise ValueError(
                    "decisions contains an unsupported "
                    f"value at row {row_position}"
                )

            normalized_decisions.append(
                normalized_decision
            )

        for row_position, explanation in enumerate(
            self.explanations
        ):
            if not isinstance(
                explanation,
                PolicyExplanationRecord,
            ):
                raise ValueError(
                    "explanations must contain only "
                    "PolicyExplanationRecord instances"
                )

            if (
                explanation.raw_model_score
                != raw_scores[row_position]
            ):
                raise ValueError(
                    "Explanation raw score does not "
                    "align with batch output"
                )

            if (
                explanation.calibrated_probability
                != calibrated_probabilities[
                    row_position
                ]
            ):
                raise ValueError(
                    "Explanation calibrated probability "
                    "does not align with batch output"
                )

            if (
                explanation.decision
                != normalized_decisions[
                    row_position
                ]
            ):
                raise ValueError(
                    "Explanation decision does not align "
                    "with batch output"
                )

        normalized_decision_array = np.array(
            normalized_decisions,
            dtype="object",
        )
        normalized_decision_array.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "raw_scores",
            raw_scores,
        )
        object.__setattr__(
            self,
            "calibrated_probabilities",
            calibrated_probabilities,
        )
        object.__setattr__(
            self,
            "decisions",
            normalized_decision_array,
        )

    @property
    def transaction_ids(
        self,
    ) -> tuple[
        TransactionIdentifier,
        ...,
    ]:
        """Return transaction identifiers in prediction order."""

        return tuple(
            explanation.transaction_id
            for explanation in self.explanations
        )


def predict_policy_with_explanations(
    bundle: CalibratedPolicyBundle,
    transaction_ids: Iterable[object],
    features: pd.DataFrame,
    *,
    top_k: int = DEFAULT_TOP_CONTRIBUTIONS,
    frame_name: str = "inference features",
) -> ExplainedPolicyPrediction:
    """Run the frozen calibrated policy and attach explanations."""

    validated_bundle = _validate_policy_bundle(
        bundle
    )

    normalized_transaction_ids = (
        _normalize_transaction_ids(
            transaction_ids,
            expected_length=len(features),
        )
    )

    policy_prediction = (
        validated_bundle.predict(
            features,
            frame_name=frame_name,
        )
    )

    contributions = (
        extract_lightgbm_contributions(
            validated_bundle.baseline_bundle,
            features,
            frame_name=frame_name,
        )
    )

    reconstruction = (
        reconstruct_lightgbm_scores(
            validated_bundle.baseline_bundle,
            features,
            contributions=contributions,
            frame_name=frame_name,
        )
    )

    signal_explanations = (
        generate_model_signal_explanations(
            validated_bundle.baseline_bundle,
            features,
            contributions=contributions,
            top_k=top_k,
            frame_name=frame_name,
        )
    )

    if not np.array_equal(
        reconstruction.raw_model_scores,
        policy_prediction.raw_scores,
    ):
        raise RuntimeError(
            "Explanation raw scores do not match "
            "the calibrated policy prediction"
        )

    if len(signal_explanations) != len(
        policy_prediction.raw_scores
    ):
        raise RuntimeError(
            "Explanation rows do not align with "
            "the calibrated policy prediction"
        )

    records: list[
        PolicyExplanationRecord
    ] = []

    for row_position, signal_explanation in enumerate(
        signal_explanations
    ):
        records.append(
            PolicyExplanationRecord(
                transaction_id=(
                    normalized_transaction_ids[
                        row_position
                    ]
                ),
                model_version=(
                    validated_bundle
                    .baseline_model_version
                ),
                policy_version=(
                    validated_bundle.policy_version
                ),
                explanation_version=(
                    EXPLANATION_VERSION
                ),
                reason_code_version=(
                    REASON_CODE_VERSION
                ),
                raw_model_margin=float(
                    reconstruction.raw_model_margins[
                        row_position
                    ]
                ),
                raw_model_score=float(
                    policy_prediction.raw_scores[
                        row_position
                    ]
                ),
                calibrated_probability=float(
                    policy_prediction
                    .calibrated_probabilities[
                        row_position
                    ]
                ),
                decision=str(
                    policy_prediction.decisions[
                        row_position
                    ]
                ),
                expected_value_raw=float(
                    reconstruction
                    .expected_values_raw[
                        row_position
                    ]
                ),
                shap_sum_raw=float(
                    reconstruction.shap_sums_raw[
                        row_position
                    ]
                ),
                reconstructed_raw_margin=float(
                    reconstruction
                    .reconstructed_raw_margins[
                        row_position
                    ]
                ),
                reconstructed_raw_model_score=float(
                    reconstruction
                    .reconstructed_raw_model_scores[
                        row_position
                    ]
                ),
                margin_reconstruction_error=float(
                    reconstruction
                    .margin_reconstruction_errors[
                        row_position
                    ]
                ),
                score_reconstruction_error=float(
                    reconstruction
                    .score_reconstruction_errors[
                        row_position
                    ]
                ),
                top_positive_contributions=(
                    signal_explanation
                    .top_positive_contributions
                ),
                top_negative_contributions=(
                    signal_explanation
                    .top_negative_contributions
                ),
                reasons=(
                    signal_explanation.reasons
                ),
            )
        )

    return ExplainedPolicyPrediction(
        raw_scores=(
            policy_prediction.raw_scores
        ),
        calibrated_probabilities=(
            policy_prediction
            .calibrated_probabilities
        ),
        decisions=(
            policy_prediction.decisions
        ),
        explanations=tuple(
            records
        ),
    )
