"""Load and validate the frozen calibrated PayGuard policy once per process."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from api.config import (
    FROZEN_BASELINE_MODEL_VERSION,
    FROZEN_BLOCK_THRESHOLD,
    FROZEN_CALIBRATION_METHOD,
    FROZEN_EXPLANATION_VERSION,
    FROZEN_POLICY_ARTIFACT,
    FROZEN_POLICY_SHA256,
    FROZEN_POLICY_VERSION,
    FROZEN_REASON_CODE_VERSION,
    FROZEN_REVIEW_THRESHOLD,
)
from src.explainability import (
    EXPLANATION_VERSION,
    REASON_CODE_VERSION,
)
from src.policy_bundle import (
    CalibratedPolicyBundle,
    load_calibrated_policy_bundle,
)


PathLike = str | Path


@dataclass(frozen=True)
class LoadedPolicy:
    """Verified frozen policy retained by the API process."""

    bundle: CalibratedPolicyBundle
    artifact_path: Path
    artifact_sha256: str


_loaded: LoadedPolicy | None = None


def _normalize_expected_sha256(
    expected_sha256: str,
) -> str:
    """Return one validated lowercase SHA-256 digest."""

    if not isinstance(expected_sha256, str):
        raise ValueError(
            "expected_sha256 must be a string"
        )

    normalized = expected_sha256.strip().lower()

    if len(normalized) != 64:
        raise ValueError(
            "expected_sha256 must contain 64 hexadecimal characters"
        )

    if any(
        character not in "0123456789abcdef"
        for character in normalized
    ):
        raise ValueError(
            "expected_sha256 must contain only hexadecimal characters"
        )

    return normalized


def _resolved_path(
    policy_path: PathLike,
) -> Path:
    """Return an absolute policy artifact path."""

    return (
        Path(policy_path)
        .expanduser()
        .resolve()
    )


def _require_policy_file(
    policy_path: Path,
) -> None:
    """Require a non-empty regular policy artifact file."""

    if not policy_path.exists():
        raise FileNotFoundError(
            "Frozen policy artifact not found: "
            f"{policy_path}"
        )

    if not policy_path.is_file():
        raise FileNotFoundError(
            "Expected frozen policy artifact to be a file: "
            f"{policy_path}"
        )

    if policy_path.stat().st_size == 0:
        raise ValueError(
            "Frozen policy artifact is empty: "
            f"{policy_path}"
        )


def calculate_policy_sha256(
    policy_path: PathLike,
) -> str:
    """Calculate the SHA-256 digest of one policy artifact."""

    resolved_path = _resolved_path(
        policy_path
    )
    _require_policy_file(
        resolved_path
    )

    digest = hashlib.sha256()

    with resolved_path.open("rb") as artifact_file:
        for chunk in iter(
            lambda: artifact_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _validate_frozen_policy(
    bundle: CalibratedPolicyBundle,
) -> CalibratedPolicyBundle:
    """Require the loaded bundle to match the complete Phase 5 contract."""

    if not isinstance(
        bundle,
        CalibratedPolicyBundle,
    ):
        raise ValueError(
            "Loaded artifact is not a CalibratedPolicyBundle"
        )

    if (
        bundle.baseline_model_version
        != FROZEN_BASELINE_MODEL_VERSION
    ):
        raise ValueError(
            "Frozen baseline model version mismatch: "
            f"expected {FROZEN_BASELINE_MODEL_VERSION!r}, "
            f"got {bundle.baseline_model_version!r}"
        )

    if (
        bundle.baseline_bundle.model_version
        != FROZEN_BASELINE_MODEL_VERSION
    ):
        raise ValueError(
            "Embedded baseline model version mismatch: "
            f"expected {FROZEN_BASELINE_MODEL_VERSION!r}, "
            f"got {bundle.baseline_bundle.model_version!r}"
        )

    if (
        bundle.policy_version
        != FROZEN_POLICY_VERSION
    ):
        raise ValueError(
            "Frozen policy version mismatch: "
            f"expected {FROZEN_POLICY_VERSION!r}, "
            f"got {bundle.policy_version!r}"
        )

    selected_method = (
        bundle.calibration_selection
        .selected_method
    )

    if (
        selected_method
        != FROZEN_CALIBRATION_METHOD
    ):
        raise ValueError(
            "Frozen calibration method mismatch: "
            f"expected {FROZEN_CALIBRATION_METHOD!r}, "
            f"got {selected_method!r}"
        )

    if (
        bundle.thresholds.review_threshold
        != FROZEN_REVIEW_THRESHOLD
    ):
        raise ValueError(
            "Frozen review threshold mismatch: "
            f"expected {FROZEN_REVIEW_THRESHOLD!r}, "
            f"got {bundle.thresholds.review_threshold!r}"
        )

    if (
        bundle.thresholds.block_threshold
        != FROZEN_BLOCK_THRESHOLD
    ):
        raise ValueError(
            "Frozen block threshold mismatch: "
            f"expected {FROZEN_BLOCK_THRESHOLD!r}, "
            f"got {bundle.thresholds.block_threshold!r}"
        )

    if (
        EXPLANATION_VERSION
        != FROZEN_EXPLANATION_VERSION
    ):
        raise ValueError(
            "Frozen explanation version mismatch: "
            f"expected {FROZEN_EXPLANATION_VERSION!r}, "
            f"got {EXPLANATION_VERSION!r}"
        )

    if (
        REASON_CODE_VERSION
        != FROZEN_REASON_CODE_VERSION
    ):
        raise ValueError(
            "Frozen reason-code version mismatch: "
            f"expected {FROZEN_REASON_CODE_VERSION!r}, "
            f"got {REASON_CODE_VERSION!r}"
        )

    return bundle


def load_policy(
    policy_path: PathLike = FROZEN_POLICY_ARTIFACT,
    *,
    expected_sha256: str = FROZEN_POLICY_SHA256,
) -> CalibratedPolicyBundle:
    """Load, verify, validate, and cache the frozen policy artifact."""

    global _loaded

    resolved_path = _resolved_path(
        policy_path
    )
    normalized_expected_sha256 = (
        _normalize_expected_sha256(
            expected_sha256
        )
    )

    if _loaded is not None:
        if (
            resolved_path
            != _loaded.artifact_path
        ):
            raise RuntimeError(
                "A frozen policy is already loaded from "
                f"{_loaded.artifact_path}; refusing to load "
                f"a different path: {resolved_path}"
            )

        if (
            normalized_expected_sha256
            != _loaded.artifact_sha256
        ):
            raise RuntimeError(
                "A frozen policy is already loaded with SHA-256 "
                f"{_loaded.artifact_sha256}; refusing a different "
                "expected digest"
            )

        return _loaded.bundle

    _require_policy_file(
        resolved_path
    )

    actual_sha256 = calculate_policy_sha256(
        resolved_path
    )

    if (
        actual_sha256
        != normalized_expected_sha256
    ):
        raise ValueError(
            "Frozen policy SHA-256 mismatch: "
            f"expected {normalized_expected_sha256}, "
            f"got {actual_sha256}"
        )

    bundle = load_calibrated_policy_bundle(
        resolved_path
    )

    validated_bundle = _validate_frozen_policy(
        bundle
    )

    _loaded = LoadedPolicy(
        bundle=validated_bundle,
        artifact_path=resolved_path,
        artifact_sha256=actual_sha256,
    )

    return validated_bundle


def get_loaded_policy() -> LoadedPolicy:
    """Return frozen policy metadata retained by the process."""

    if _loaded is None:
        raise RuntimeError(
            "Frozen policy not loaded. "
            "Call load_policy() during application startup."
        )

    return _loaded


def get_policy() -> CalibratedPolicyBundle:
    """Return the loaded calibrated policy bundle."""

    return get_loaded_policy().bundle


def is_loaded() -> bool:
    """Return whether a verified frozen policy is loaded."""

    return _loaded is not None


def load_model(
    model_path: PathLike,
) -> CalibratedPolicyBundle:
    """Temporary compatibility alias for the legacy API scaffold."""

    return load_policy(
        model_path
    )


def get_model() -> CalibratedPolicyBundle:
    """Temporary compatibility alias for the legacy API scaffold."""

    return get_policy()
