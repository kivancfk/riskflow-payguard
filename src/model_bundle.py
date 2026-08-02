"""Versioned persistence and inference for PayGuard models."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Mapping
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from src.model_data import FeatureContract
from src.preprocessing import CategoricalEncoder


MODEL_BUNDLE_SCHEMA_VERSION = 1

PathLike = str | Path


@dataclass(frozen=True)
class ModelBundle:
    """Complete versioned artifact required for inference."""

    schema_version: int
    model_version: str
    created_at_utc: str

    model: Any
    encoder: CategoricalEncoder

    feature_columns: tuple[str, ...]
    categorical_features: tuple[str, ...]
    numerical_features: tuple[str, ...]

    training_config: dict[str, Any]
    scale_pos_weight: float
    best_iteration: int

    validation_metrics: dict[str, Any]
    dataset_manifest: dict[str, Any]

    def predict_fraud_probabilities(
        self,
        features: pd.DataFrame,
        *,
        frame_name: str = "inference features",
    ) -> np.ndarray:
        """Transform raw features and return fraud probabilities."""
        _validate_model_bundle(self)

        encoded_features = self.encoder.transform(
            features,
            frame_name=frame_name,
        )

        probability_matrix = np.asarray(
            self.model.predict_proba(
                encoded_features,
                num_iteration=self.best_iteration,
                validate_features=True,
            ),
            dtype="float64",
        )

        expected_shape = (
            len(encoded_features),
            2,
        )

        if probability_matrix.shape != expected_shape:
            raise RuntimeError(
                "Model returned an unexpected "
                "probability shape"
            )

        fraud_probabilities = probability_matrix[
            :,
            1,
        ]

        if not np.isfinite(
            fraud_probabilities
        ).all():
            raise RuntimeError(
                "Model returned non-finite "
                "fraud probabilities"
            )

        if (
            (fraud_probabilities < 0.0).any()
            or (fraud_probabilities > 1.0).any()
        ):
            raise RuntimeError(
                "Model returned probabilities outside "
                "the range from 0 to 1"
            )

        return fraud_probabilities


def _validate_model_version(
    model_version: str,
) -> str:
    """Return a non-empty normalized model version."""
    if not isinstance(model_version, str):
        raise ValueError(
            "model_version must be a string"
        )

    normalized_version = model_version.strip()

    if not normalized_version:
        raise ValueError(
            "model_version must not be empty"
        )

    return normalized_version


def _validate_created_at(
    created_at_utc: str,
) -> None:
    """Require an ISO timestamp containing timezone information."""
    if not isinstance(created_at_utc, str):
        raise ValueError(
            "created_at_utc must be a string"
        )

    try:
        parsed_timestamp = datetime.fromisoformat(
            created_at_utc.replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError as error:
        raise ValueError(
            "created_at_utc must be an ISO timestamp"
        ) from error

    if parsed_timestamp.tzinfo is None:
        raise ValueError(
            "created_at_utc must include timezone information"
        )


def _validate_feature_contract(
    bundle: ModelBundle,
) -> None:
    """Validate feature lists across model and encoder."""
    if not bundle.feature_columns:
        raise ValueError(
            "Model bundle must contain feature columns"
        )

    if len(bundle.feature_columns) != len(
        set(bundle.feature_columns)
    ):
        raise ValueError(
            "Model bundle contains duplicate feature names"
        )

    categorical_set = set(
        bundle.categorical_features
    )
    numerical_set = set(
        bundle.numerical_features
    )

    if categorical_set.intersection(
        numerical_set
    ):
        raise ValueError(
            "Categorical and numerical feature lists overlap"
        )

    if categorical_set.union(
        numerical_set
    ) != set(bundle.feature_columns):
        raise ValueError(
            "Categorical and numerical features do not "
            "cover the model feature contract"
        )

    if bundle.encoder.feature_columns != (
        bundle.feature_columns
    ):
        raise ValueError(
            "Encoder feature columns do not match "
            "the model bundle"
        )

    if bundle.encoder.categorical_features != (
        bundle.categorical_features
    ):
        raise ValueError(
            "Encoder categorical features do not match "
            "the model bundle"
        )

    if bundle.encoder.numerical_features != (
        bundle.numerical_features
    ):
        raise ValueError(
            "Encoder numerical features do not match "
            "the model bundle"
        )

    if set(
        bundle.encoder.category_vocabularies
    ) != categorical_set:
        raise ValueError(
            "Encoder vocabularies do not match "
            "the categorical feature contract"
        )

    model_feature_names = getattr(
        bundle.model,
        "feature_name_",
        None,
    )

    if model_feature_names is None:
        raise ValueError(
            "Bundled model does not expose feature names"
        )

    if tuple(model_feature_names) != (
        bundle.feature_columns
    ):
        raise ValueError(
            "Model feature names do not match "
            "the model bundle"
        )


def _validate_model_bundle(
    bundle: object,
) -> ModelBundle:
    """Validate a loaded or newly constructed bundle."""
    if not isinstance(bundle, ModelBundle):
        raise ValueError(
            "Artifact does not contain a ModelBundle"
        )

    if (
        bundle.schema_version
        != MODEL_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported model bundle schema version"
        )

    _validate_model_version(
        bundle.model_version
    )
    _validate_created_at(
        bundle.created_at_utc
    )
    _validate_feature_contract(bundle)

    if not hasattr(
        bundle.model,
        "predict_proba",
    ):
        raise ValueError(
            "Bundled model does not support predict_proba"
        )

    if (
        not isinstance(bundle.best_iteration, int)
        or isinstance(bundle.best_iteration, bool)
        or bundle.best_iteration <= 0
    ):
        raise ValueError(
            "best_iteration must be a positive integer"
        )

    if (
        not isinstance(
            bundle.scale_pos_weight,
            (int, float),
        )
        or isinstance(
            bundle.scale_pos_weight,
            bool,
        )
        or not math.isfinite(
            float(bundle.scale_pos_weight)
        )
        or float(bundle.scale_pos_weight) <= 0.0
    ):
        raise ValueError(
            "scale_pos_weight must be positive and finite"
        )

    mapping_fields = {
        "training_config": bundle.training_config,
        "validation_metrics": (
            bundle.validation_metrics
        ),
        "dataset_manifest": bundle.dataset_manifest,
    }

    for field_name, value in mapping_fields.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"{field_name} must be a dictionary"
            )

    return bundle


def build_model_bundle(
    *,
    model: Any,
    encoder: CategoricalEncoder,
    feature_contract: FeatureContract,
    training_config: Mapping[str, Any],
    scale_pos_weight: float,
    best_iteration: int,
    validation_metrics: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    model_version: str,
    created_at_utc: str | None = None,
) -> ModelBundle:
    """Build and validate the complete inference artifact."""
    resolved_created_at = (
        created_at_utc
        if created_at_utc is not None
        else datetime.now(
            timezone.utc
        ).isoformat()
    )

    bundle = ModelBundle(
        schema_version=(
            MODEL_BUNDLE_SCHEMA_VERSION
        ),
        model_version=_validate_model_version(
            model_version
        ),
        created_at_utc=resolved_created_at,
        model=model,
        encoder=encoder,
        feature_columns=(
            feature_contract.feature_columns
        ),
        categorical_features=(
            feature_contract.categorical_features
        ),
        numerical_features=(
            feature_contract.numerical_features
        ),
        training_config=copy.deepcopy(
            dict(training_config)
        ),
        scale_pos_weight=float(
            scale_pos_weight
        ),
        best_iteration=best_iteration,
        validation_metrics=copy.deepcopy(
            dict(validation_metrics)
        ),
        dataset_manifest=copy.deepcopy(
            dict(dataset_manifest)
        ),
    )

    return _validate_model_bundle(bundle)


def save_model_bundle(
    bundle: ModelBundle,
    output_path: PathLike,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist a validated joblib bundle."""
    validated_bundle = _validate_model_bundle(
        bundle
    )

    resolved_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if resolved_path.exists():
        if not resolved_path.is_file():
            raise IsADirectoryError(
                f"Model output is not a file: "
                f"{resolved_path}"
            )

        if not overwrite:
            raise FileExistsError(
                f"Model artifact already exists: "
                f"{resolved_path}. "
                "Use overwrite=True to replace it."
            )

    temporary_path = (
        resolved_path.parent
        / (
            f".{resolved_path.name}."
            f"{uuid4().hex}.tmp"
        )
    )

    try:
        with temporary_path.open(
            "wb"
        ) as temporary_file:
            joblib.dump(
                validated_bundle,
                temporary_file,
                compress=3,
            )
            temporary_file.flush()
            os.fsync(
                temporary_file.fileno()
            )

        load_model_bundle(temporary_path)

        os.replace(
            temporary_path,
            resolved_path,
        )
    except Exception:
        temporary_path.unlink(
            missing_ok=True
        )
        raise

    return resolved_path


def load_model_bundle(
    artifact_path: PathLike,
) -> ModelBundle:
    """Load and validate a trusted local model bundle."""
    resolved_path = (
        Path(artifact_path)
        .expanduser()
        .resolve()
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {resolved_path}"
        )

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Expected a model file: {resolved_path}"
        )

    if resolved_path.stat().st_size == 0:
        raise ValueError(
            f"Model artifact is empty: {resolved_path}"
        )

    try:
        loaded_artifact = joblib.load(
            resolved_path
        )
    except Exception as error:
        raise ValueError(
            f"Unable to load model artifact: "
            f"{resolved_path}"
        ) from error

    return _validate_model_bundle(
        loaded_artifact
    )
