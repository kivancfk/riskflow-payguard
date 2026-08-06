"""Native LightGBM SHAP contribution extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model_bundle import (
    ModelBundle,
    _validate_model_bundle,
)


@dataclass(
    frozen=True,
    eq=False,
)
class LightGBMContributionMatrix:
    """Validated native LightGBM contributions for a feature frame."""

    feature_names: tuple[str, ...]
    shap_values_raw: np.ndarray
    expected_values_raw: np.ndarray

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

        try:
            shap_values_raw = np.array(
                self.shap_values_raw,
                dtype="float64",
                copy=True,
            )
            expected_values_raw = np.array(
                self.expected_values_raw,
                dtype="float64",
                copy=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "LightGBM contributions must be numerical"
            ) from error

        if shap_values_raw.ndim != 2:
            raise ValueError(
                "shap_values_raw must be two-dimensional"
            )

        if expected_values_raw.ndim != 1:
            raise ValueError(
                "expected_values_raw must be one-dimensional"
            )

        if shap_values_raw.shape[0] == 0:
            raise ValueError(
                "LightGBM contributions must contain "
                "at least one row"
            )

        if shap_values_raw.shape[1] != len(
            self.feature_names
        ):
            raise ValueError(
                "SHAP feature count does not match "
                "feature_names"
            )

        if len(expected_values_raw) != (
            shap_values_raw.shape[0]
        ):
            raise ValueError(
                "Expected values must align with "
                "SHAP contribution rows"
            )

        if not np.isfinite(
            shap_values_raw
        ).all():
            raise ValueError(
                "shap_values_raw must contain only "
                "finite values"
            )

        if not np.isfinite(
            expected_values_raw
        ).all():
            raise ValueError(
                "expected_values_raw must contain only "
                "finite values"
            )

        shap_values_raw.setflags(
            write=False
        )
        expected_values_raw.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "shap_values_raw",
            shap_values_raw,
        )
        object.__setattr__(
            self,
            "expected_values_raw",
            expected_values_raw,
        )

    @property
    def row_count(self) -> int:
        """Return the number of explained rows."""

        return int(
            self.shap_values_raw.shape[0]
        )

    @property
    def feature_count(self) -> int:
        """Return the number of explained model features."""

        return int(
            self.shap_values_raw.shape[1]
        )


def extract_lightgbm_contributions(
    bundle: ModelBundle,
    features: pd.DataFrame,
    *,
    frame_name: str = "explanation features",
) -> LightGBMContributionMatrix:
    """Extract native LightGBM raw-margin contributions.

    LightGBM returns one column per model feature followed by one
    expected-value column. This function validates and separates those
    components without changing the model, encoder, or input frame.
    """

    validated_bundle = _validate_model_bundle(
        bundle
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

    native_contributions = booster_predict(
        encoded_features,
        num_iteration=(
            validated_bundle.best_iteration
        ),
        pred_contrib=True,
        validate_features=True,
    )

    toarray = getattr(
        native_contributions,
        "toarray",
        None,
    )

    if callable(toarray):
        native_contributions = toarray()

    try:
        contribution_matrix = np.asarray(
            native_contributions,
            dtype="float64",
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "LightGBM returned non-numerical "
            "contributions"
        ) from error

    expected_shape = (
        len(encoded_features),
        len(
            validated_bundle.feature_columns
        )
        + 1,
    )

    if contribution_matrix.shape != (
        expected_shape
    ):
        raise RuntimeError(
            "LightGBM returned an unexpected "
            "contribution shape: "
            f"expected {expected_shape}, "
            f"received {contribution_matrix.shape}"
        )

    if not np.isfinite(
        contribution_matrix
    ).all():
        raise RuntimeError(
            "LightGBM returned non-finite "
            "contributions"
        )

    return LightGBMContributionMatrix(
        feature_names=(
            validated_bundle.feature_columns
        ),
        shap_values_raw=(
            contribution_matrix[
                :,
                :-1,
            ]
        ),
        expected_values_raw=(
            contribution_matrix[
                :,
                -1,
            ]
        ),
    )
