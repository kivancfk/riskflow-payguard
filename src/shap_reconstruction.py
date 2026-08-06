"""Validate LightGBM SHAP raw-margin and score reconstruction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.lightgbm_shap import (
    LightGBMContributionMatrix,
    extract_lightgbm_contributions,
)
from src.model_bundle import (
    ModelBundle,
    _validate_model_bundle,
)


RECONSTRUCTION_ABSOLUTE_TOLERANCE = 1e-8


def _validate_tolerance(
    absolute_tolerance: float,
) -> float:
    """Return a finite, strictly positive absolute tolerance."""

    if isinstance(
        absolute_tolerance,
        bool,
    ):
        raise ValueError(
            "absolute_tolerance must be numerical"
        )

    try:
        normalized_tolerance = float(
            absolute_tolerance
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "absolute_tolerance must be numerical"
        ) from error

    if (
        not math.isfinite(
            normalized_tolerance
        )
        or normalized_tolerance <= 0.0
    ):
        raise ValueError(
            "absolute_tolerance must be finite "
            "and greater than 0"
        )

    return normalized_tolerance


def _readonly_float_vector(
    values: object,
    *,
    value_name: str,
    expected_length: int | None = None,
) -> np.ndarray:
    """Return a defensive, finite, read-only float vector."""

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

    if (
        expected_length is not None
        and len(vector) != expected_length
    ):
        raise ValueError(
            f"{value_name} must contain "
            f"{expected_length} values"
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            f"{value_name} must contain only finite values"
        )

    vector.setflags(
        write=False
    )

    return vector


@dataclass(
    frozen=True,
    eq=False,
)
class LightGBMScoreReconstruction:
    """Validated raw-margin and raw-score reconstruction results."""

    feature_names: tuple[str, ...]
    raw_model_margins: np.ndarray
    raw_model_scores: np.ndarray
    expected_values_raw: np.ndarray
    shap_sums_raw: np.ndarray
    reconstructed_raw_margins: np.ndarray
    reconstructed_raw_model_scores: np.ndarray
    margin_reconstruction_errors: np.ndarray
    score_reconstruction_errors: np.ndarray
    absolute_tolerance: float

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.feature_names,
                tuple,
            )
            or not self.feature_names
            or not all(
                isinstance(feature_name, str)
                and feature_name
                for feature_name in self.feature_names
            )
        ):
            raise ValueError(
                "feature_names must be a non-empty "
                "tuple of non-empty strings"
            )

        if len(self.feature_names) != len(
            set(self.feature_names)
        ):
            raise ValueError(
                "feature_names must be unique"
            )

        absolute_tolerance = (
            _validate_tolerance(
                self.absolute_tolerance
            )
        )

        raw_model_margins = (
            _readonly_float_vector(
                self.raw_model_margins,
                value_name="raw_model_margins",
            )
        )
        row_count = len(
            raw_model_margins
        )

        normalized_vectors = {
            "raw_model_scores": (
                _readonly_float_vector(
                    self.raw_model_scores,
                    value_name="raw_model_scores",
                    expected_length=row_count,
                )
            ),
            "expected_values_raw": (
                _readonly_float_vector(
                    self.expected_values_raw,
                    value_name="expected_values_raw",
                    expected_length=row_count,
                )
            ),
            "shap_sums_raw": (
                _readonly_float_vector(
                    self.shap_sums_raw,
                    value_name="shap_sums_raw",
                    expected_length=row_count,
                )
            ),
            "reconstructed_raw_margins": (
                _readonly_float_vector(
                    self.reconstructed_raw_margins,
                    value_name=(
                        "reconstructed_raw_margins"
                    ),
                    expected_length=row_count,
                )
            ),
            "reconstructed_raw_model_scores": (
                _readonly_float_vector(
                    self.reconstructed_raw_model_scores,
                    value_name=(
                        "reconstructed_raw_model_scores"
                    ),
                    expected_length=row_count,
                )
            ),
            "margin_reconstruction_errors": (
                _readonly_float_vector(
                    self.margin_reconstruction_errors,
                    value_name=(
                        "margin_reconstruction_errors"
                    ),
                    expected_length=row_count,
                )
            ),
            "score_reconstruction_errors": (
                _readonly_float_vector(
                    self.score_reconstruction_errors,
                    value_name=(
                        "score_reconstruction_errors"
                    ),
                    expected_length=row_count,
                )
            ),
        }

        for probability_name in (
            "raw_model_scores",
            "reconstructed_raw_model_scores",
        ):
            probabilities = (
                normalized_vectors[
                    probability_name
                ]
            )

            if (
                (probabilities < 0.0).any()
                or (probabilities > 1.0).any()
            ):
                raise ValueError(
                    f"{probability_name} must be "
                    "between 0 and 1"
                )

        for error_name in (
            "margin_reconstruction_errors",
            "score_reconstruction_errors",
        ):
            errors = normalized_vectors[
                error_name
            ]

            if (errors < 0.0).any():
                raise ValueError(
                    f"{error_name} must be non-negative"
                )

        object.__setattr__(
            self,
            "raw_model_margins",
            raw_model_margins,
        )

        for field_name, vector in (
            normalized_vectors.items()
        ):
            object.__setattr__(
                self,
                field_name,
                vector,
            )

        object.__setattr__(
            self,
            "absolute_tolerance",
            absolute_tolerance,
        )

    @property
    def row_count(self) -> int:
        """Return the number of reconstructed rows."""

        return len(
            self.raw_model_margins
        )

    @property
    def max_margin_reconstruction_error(
        self,
    ) -> float:
        """Return the largest raw-margin reconstruction error."""

        return float(
            self.margin_reconstruction_errors.max()
        )

    @property
    def max_score_reconstruction_error(
        self,
    ) -> float:
        """Return the largest raw-score reconstruction error."""

        return float(
            self.score_reconstruction_errors.max()
        )


def lightgbm_binary_sigmoid(
    raw_margins: object,
) -> np.ndarray:
    """Apply a numerically stable binary logistic sigmoid."""

    margins = _readonly_float_vector(
        raw_margins,
        value_name="raw_margins",
    )

    probabilities = np.empty_like(
        margins
    )

    non_negative_mask = margins >= 0.0

    probabilities[
        non_negative_mask
    ] = 1.0 / (
        1.0
        + np.exp(
            -margins[
                non_negative_mask
            ]
        )
    )

    negative_exponentials = np.exp(
        margins[
            ~non_negative_mask
        ]
    )

    probabilities[
        ~non_negative_mask
    ] = (
        negative_exponentials
        / (
            1.0
            + negative_exponentials
        )
    )

    probabilities.setflags(
        write=False
    )

    return probabilities


def reconstruct_lightgbm_scores(
    bundle: ModelBundle,
    features: pd.DataFrame,
    *,
    contributions: (
        LightGBMContributionMatrix | None
    ) = None,
    frame_name: str = "explanation features",
    absolute_tolerance: float = (
        RECONSTRUCTION_ABSOLUTE_TOLERANCE
    ),
) -> LightGBMScoreReconstruction:
    """Reconstruct and validate raw LightGBM margins and scores."""

    validated_bundle = _validate_model_bundle(
        bundle
    )
    validated_tolerance = (
        _validate_tolerance(
            absolute_tolerance
        )
    )

    resolved_contributions = (
        contributions
        if contributions is not None
        else extract_lightgbm_contributions(
            validated_bundle,
            features,
            frame_name=frame_name,
        )
    )

    if not isinstance(
        resolved_contributions,
        LightGBMContributionMatrix,
    ):
        raise ValueError(
            "contributions must be a "
            "LightGBMContributionMatrix"
        )

    if (
        resolved_contributions.feature_names
        != validated_bundle.feature_columns
    ):
        raise ValueError(
            "Contribution feature names do not match "
            "the frozen model feature contract"
        )

    if (
        resolved_contributions.row_count
        != len(features)
    ):
        raise ValueError(
            "Contribution rows do not align "
            "with the feature frame"
        )

    encoded_features = (
        validated_bundle.encoder.transform(
            features,
            frame_name=frame_name,
        )
    )

    booster = getattr(
        validated_bundle.model,
        "booster_",
        None,
    )
    booster_predict = getattr(
        booster,
        "predict",
        None,
    )

    if not callable(booster_predict):
        raise ValueError(
            "Bundled model does not expose a fitted "
            "LightGBM booster"
        )

    native_raw_margins = (
        _readonly_float_vector(
            booster_predict(
                encoded_features,
                num_iteration=(
                    validated_bundle.best_iteration
                ),
                raw_score=True,
                validate_features=True,
            ),
            value_name="native_raw_margins",
            expected_length=len(
                encoded_features
            ),
        )
    )

    raw_model_scores = (
        _readonly_float_vector(
            validated_bundle
            .predict_fraud_probabilities(
                features,
                frame_name=frame_name,
            ),
            value_name="raw_model_scores",
            expected_length=len(
                encoded_features
            ),
        )
    )

    shap_sums_raw = np.sum(
        resolved_contributions.shap_values_raw,
        axis=1,
        dtype="float64",
    )

    reconstructed_raw_margins = (
        resolved_contributions
        .expected_values_raw
        + shap_sums_raw
    )

    reconstructed_raw_model_scores = (
        lightgbm_binary_sigmoid(
            reconstructed_raw_margins
        )
    )

    margin_errors = np.abs(
        native_raw_margins
        - reconstructed_raw_margins
    )
    score_errors = np.abs(
        raw_model_scores
        - reconstructed_raw_model_scores
    )

    max_margin_error = float(
        margin_errors.max()
    )
    max_score_error = float(
        score_errors.max()
    )

    if max_margin_error > (
        validated_tolerance
    ):
        raise RuntimeError(
            "SHAP raw-margin reconstruction exceeded "
            "the absolute tolerance: "
            f"{max_margin_error} > "
            f"{validated_tolerance}"
        )

    if max_score_error > (
        validated_tolerance
    ):
        raise RuntimeError(
            "SHAP raw-score reconstruction exceeded "
            "the absolute tolerance: "
            f"{max_score_error} > "
            f"{validated_tolerance}"
        )

    return LightGBMScoreReconstruction(
        feature_names=(
            validated_bundle.feature_columns
        ),
        raw_model_margins=(
            native_raw_margins
        ),
        raw_model_scores=(
            raw_model_scores
        ),
        expected_values_raw=(
            resolved_contributions
            .expected_values_raw
        ),
        shap_sums_raw=(
            shap_sums_raw
        ),
        reconstructed_raw_margins=(
            reconstructed_raw_margins
        ),
        reconstructed_raw_model_scores=(
            reconstructed_raw_model_scores
        ),
        margin_reconstruction_errors=(
            margin_errors
        ),
        score_reconstruction_errors=(
            score_errors
        ),
        absolute_tolerance=(
            validated_tolerance
        ),
    )
