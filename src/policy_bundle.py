"""Versioned persistence and inference for calibrated fraud policies."""

from __future__ import annotations

import copy
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

from src.calibration import (
    CALIBRATION_METHODS,
    CalibrationSelection,
)
from src.model_bundle import (
    ModelBundle,
    _validate_model_bundle,
)
from src.policy import (
    PolicyConstraints,
    PolicyCostAssumptions,
    PolicySearchResult,
    PolicyThresholds,
    assign_policy_decisions,
)


POLICY_BUNDLE_SCHEMA_VERSION = 1

PathLike = str | Path


@dataclass(frozen=True)
class PolicyPrediction:
    """Raw score, calibrated probability, and policy decision."""

    raw_scores: np.ndarray
    calibrated_probabilities: np.ndarray
    decisions: np.ndarray


@dataclass(frozen=True)
class CalibratedPolicyBundle:
    """Complete frozen artifact required for policy inference."""

    schema_version: int
    policy_version: str
    created_at_utc: str

    baseline_model_version: str
    baseline_bundle: ModelBundle

    calibration_selection: CalibrationSelection
    thresholds: PolicyThresholds
    cost_assumptions: PolicyCostAssumptions
    constraints: PolicyConstraints
    policy_search_result: PolicySearchResult

    development_metadata: dict[str, Any]
    calibration_metrics: dict[str, Any]
    drift_diagnostics: dict[str, Any]
    dataset_manifest: dict[str, Any]

    def predict(
        self,
        features: pd.DataFrame,
        *,
        frame_name: str = "inference features",
    ) -> PolicyPrediction:
        """Score raw features and apply the calibrated policy."""

        _validate_policy_bundle(self)

        raw_scores = (
            self.baseline_bundle
            .predict_fraud_probabilities(
                features,
                frame_name=frame_name,
            )
        )

        calibrated_probabilities = (
            self.calibration_selection
            .predict_calibrated_probabilities(
                raw_scores
            )
        )

        if len(calibrated_probabilities) != len(
            raw_scores
        ):
            raise RuntimeError(
                "Calibrator returned an unexpected "
                "number of probabilities"
            )

        decisions = assign_policy_decisions(
            calibrated_probabilities,
            self.thresholds,
        )

        return PolicyPrediction(
            raw_scores=raw_scores,
            calibrated_probabilities=(
                calibrated_probabilities
            ),
            decisions=decisions,
        )


def _validate_version(
    value: str,
    *,
    value_name: str,
) -> str:
    """Return a non-empty normalized version string."""

    if not isinstance(value, str):
        raise ValueError(
            f"{value_name} must be a string"
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{value_name} must not be empty"
        )

    return normalized_value


def _validate_created_at(
    created_at_utc: str,
) -> None:
    """Require an ISO timestamp with timezone information."""

    if not isinstance(created_at_utc, str):
        raise ValueError(
            "created_at_utc must be a string"
        )

    try:
        timestamp = datetime.fromisoformat(
            created_at_utc.replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError as error:
        raise ValueError(
            "created_at_utc must be an ISO timestamp"
        ) from error

    if timestamp.tzinfo is None:
        raise ValueError(
            "created_at_utc must include timezone information"
        )


def _validate_policy_bundle(
    bundle: object,
) -> CalibratedPolicyBundle:
    """Validate a loaded or newly constructed policy bundle."""

    if not isinstance(
        bundle,
        CalibratedPolicyBundle,
    ):
        raise ValueError(
            "Artifact does not contain a "
            "CalibratedPolicyBundle"
        )

    if (
        bundle.schema_version
        != POLICY_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported calibrated policy "
            "bundle schema version"
        )

    _validate_version(
        bundle.policy_version,
        value_name="policy_version",
    )
    _validate_created_at(
        bundle.created_at_utc
    )

    baseline_bundle = _validate_model_bundle(
        bundle.baseline_bundle
    )

    baseline_model_version = _validate_version(
        bundle.baseline_model_version,
        value_name="baseline_model_version",
    )

    if (
        baseline_model_version
        != baseline_bundle.model_version
    ):
        raise ValueError(
            "baseline_model_version does not match "
            "the embedded baseline bundle"
        )

    if not isinstance(
        bundle.calibration_selection,
        CalibrationSelection,
    ):
        raise ValueError(
            "calibration_selection must be a "
            "CalibrationSelection"
        )

    selected_method = (
        bundle.calibration_selection
        .selected_method
    )

    if selected_method not in CALIBRATION_METHODS:
        raise ValueError(
            "Unsupported selected calibration method"
        )

    calibrator_method = getattr(
        bundle.calibration_selection
        .selected_calibrator,
        "method_name",
        None,
    )

    if calibrator_method != selected_method:
        raise ValueError(
            "Selected calibrator does not match "
            "selected_method"
        )

    if not bundle.calibration_selection.candidate_results:
        raise ValueError(
            "Calibration selection must contain "
            "candidate results"
        )

    if not isinstance(
        bundle.thresholds,
        PolicyThresholds,
    ):
        raise ValueError(
            "thresholds must be a PolicyThresholds"
        )

    if not isinstance(
        bundle.cost_assumptions,
        PolicyCostAssumptions,
    ):
        raise ValueError(
            "cost_assumptions must be a "
            "PolicyCostAssumptions"
        )

    if not isinstance(
        bundle.constraints,
        PolicyConstraints,
    ):
        raise ValueError(
            "constraints must be a PolicyConstraints"
        )

    if not isinstance(
        bundle.policy_search_result,
        PolicySearchResult,
    ):
        raise ValueError(
            "policy_search_result must be a "
            "PolicySearchResult"
        )

    if (
        bundle.thresholds
        != bundle.policy_search_result
        .selected_thresholds
    ):
        raise ValueError(
            "Bundle thresholds do not match "
            "the policy search result"
        )

    if (
        bundle.cost_assumptions
        != bundle.policy_search_result
        .cost_assumptions
    ):
        raise ValueError(
            "Bundle cost assumptions do not match "
            "the policy search result"
        )

    if (
        bundle.constraints
        != bundle.policy_search_result.constraints
    ):
        raise ValueError(
            "Bundle constraints do not match "
            "the policy search result"
        )

    mapping_fields = {
        "development_metadata": (
            bundle.development_metadata
        ),
        "calibration_metrics": (
            bundle.calibration_metrics
        ),
        "drift_diagnostics": (
            bundle.drift_diagnostics
        ),
        "dataset_manifest": (
            bundle.dataset_manifest
        ),
    }

    for field_name, value in mapping_fields.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"{field_name} must be a dictionary"
            )

    if (
        bundle.dataset_manifest
        != baseline_bundle.dataset_manifest
    ):
        raise ValueError(
            "Policy dataset manifest does not match "
            "the baseline bundle manifest"
        )

    return bundle


def build_calibrated_policy_bundle(
    *,
    baseline_bundle: ModelBundle,
    calibration_selection: CalibrationSelection,
    policy_search_result: PolicySearchResult,
    development_metadata: Mapping[str, Any],
    calibration_metrics: Mapping[str, Any],
    drift_diagnostics: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    policy_version: str,
    created_at_utc: str | None = None,
) -> CalibratedPolicyBundle:
    """Build and validate a complete calibrated policy artifact."""

    validated_baseline = _validate_model_bundle(
        baseline_bundle
    )

    resolved_created_at = (
        created_at_utc
        if created_at_utc is not None
        else datetime.now(
            timezone.utc
        ).isoformat()
    )

    bundle = CalibratedPolicyBundle(
        schema_version=(
            POLICY_BUNDLE_SCHEMA_VERSION
        ),
        policy_version=_validate_version(
            policy_version,
            value_name="policy_version",
        ),
        created_at_utc=resolved_created_at,
        baseline_model_version=(
            validated_baseline.model_version
        ),
        baseline_bundle=validated_baseline,
        calibration_selection=(
            calibration_selection
        ),
        thresholds=(
            policy_search_result
            .selected_thresholds
        ),
        cost_assumptions=(
            policy_search_result
            .cost_assumptions
        ),
        constraints=(
            policy_search_result.constraints
        ),
        policy_search_result=(
            policy_search_result
        ),
        development_metadata=copy.deepcopy(
            dict(development_metadata)
        ),
        calibration_metrics=copy.deepcopy(
            dict(calibration_metrics)
        ),
        drift_diagnostics=copy.deepcopy(
            dict(drift_diagnostics)
        ),
        dataset_manifest=copy.deepcopy(
            dict(dataset_manifest)
        ),
    )

    return _validate_policy_bundle(bundle)


def save_calibrated_policy_bundle(
    bundle: CalibratedPolicyBundle,
    output_path: PathLike,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist a validated calibrated policy."""

    validated_bundle = _validate_policy_bundle(
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
                "Policy output is not a file: "
                f"{resolved_path}"
            )

        if not overwrite:
            raise FileExistsError(
                "Policy artifact already exists: "
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

        load_calibrated_policy_bundle(
            temporary_path
        )

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


def load_calibrated_policy_bundle(
    artifact_path: PathLike,
) -> CalibratedPolicyBundle:
    """Load and validate a trusted local policy artifact."""

    resolved_path = (
        Path(artifact_path)
        .expanduser()
        .resolve()
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Policy artifact not found: "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise FileNotFoundError(
            "Expected a policy file: "
            f"{resolved_path}"
        )

    if resolved_path.stat().st_size == 0:
        raise ValueError(
            "Policy artifact is empty: "
            f"{resolved_path}"
        )

    try:
        loaded_artifact = joblib.load(
            resolved_path
        )
    except Exception as error:
        raise ValueError(
            "Unable to load policy artifact: "
            f"{resolved_path}"
        ) from error

    return _validate_policy_bundle(
        loaded_artifact
    )
