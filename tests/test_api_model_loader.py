"""Tests for frozen Phase 5 API policy loading."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from api import model_loader
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
from src.policy_bundle import (
    CalibratedPolicyBundle,
    load_calibrated_policy_bundle,
)


POLICY_PATH = Path(
    FROZEN_POLICY_ARTIFACT
).resolve()


@pytest.fixture(autouse=True)
def reset_process_policy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep loader cache state isolated across tests."""

    monkeypatch.setattr(
        model_loader,
        "_loaded",
        None,
    )


@pytest.fixture(scope="module")
def frozen_bundle() -> CalibratedPolicyBundle:
    """Load the trusted frozen artifact for metadata mutation tests."""

    return load_calibrated_policy_bundle(
        POLICY_PATH
    )


def test_frozen_policy_digest_matches_phase_5_contract() -> None:
    """Require repository artifact bytes to match the frozen digest."""

    assert (
        model_loader.calculate_policy_sha256(
            POLICY_PATH
        )
        == FROZEN_POLICY_SHA256
    )


def test_load_policy_validates_and_caches_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the artifact once and reuse the same validated bundle."""

    original_loader = (
        model_loader.load_calibrated_policy_bundle
    )
    load_count = 0

    def counting_loader(
        artifact_path: str | Path,
    ) -> CalibratedPolicyBundle:
        nonlocal load_count
        load_count += 1

        return original_loader(
            artifact_path
        )

    monkeypatch.setattr(
        model_loader,
        "load_calibrated_policy_bundle",
        counting_loader,
    )

    first = model_loader.load_policy(
        POLICY_PATH
    )
    second = model_loader.load_policy(
        POLICY_PATH
    )

    assert load_count == 1
    assert first is second
    assert model_loader.is_loaded()
    assert model_loader.get_policy() is first

    loaded = model_loader.get_loaded_policy()

    assert loaded.bundle is first
    assert loaded.artifact_path == POLICY_PATH
    assert (
        loaded.artifact_sha256
        == FROZEN_POLICY_SHA256
    )

    assert (
        first.baseline_model_version
        == FROZEN_BASELINE_MODEL_VERSION
    )
    assert (
        first.policy_version
        == FROZEN_POLICY_VERSION
    )
    assert (
        first.calibration_selection.selected_method
        == FROZEN_CALIBRATION_METHOD
    )
    assert (
        first.thresholds.review_threshold
        == FROZEN_REVIEW_THRESHOLD
    )
    assert (
        first.thresholds.block_threshold
        == FROZEN_BLOCK_THRESHOLD
    )

    assert (
        model_loader.calculate_policy_sha256(
            POLICY_PATH
        )
        == FROZEN_POLICY_SHA256
    )


def test_load_policy_rejects_missing_artifact(
    tmp_path: Path,
) -> None:
    """Do not start from an artifact path that does not exist."""

    missing_path = (
        tmp_path
        / "missing-policy.joblib"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Frozen policy artifact not found",
    ):
        model_loader.load_policy(
            missing_path
        )


def test_load_policy_rejects_empty_artifact(
    tmp_path: Path,
) -> None:
    """Do not load a zero-byte artifact."""

    empty_path = (
        tmp_path
        / "empty-policy.joblib"
    )
    empty_path.write_bytes(b"")

    with pytest.raises(
        ValueError,
        match="Frozen policy artifact is empty",
    ):
        model_loader.load_policy(
            empty_path
        )


def test_load_policy_rejects_wrong_sha256() -> None:
    """Reject changed artifact bytes before deserialization."""

    with pytest.raises(
        ValueError,
        match="Frozen policy SHA-256 mismatch",
    ):
        model_loader.load_policy(
            POLICY_PATH,
            expected_sha256="0" * 64,
        )


def test_load_policy_rejects_invalid_artifact(
    tmp_path: Path,
) -> None:
    """Reject bytes that match their digest but are not a valid policy."""

    invalid_path = (
        tmp_path
        / "invalid-policy.joblib"
    )
    invalid_path.write_bytes(
        b"not-a-valid-joblib-policy"
    )

    invalid_sha256 = (
        model_loader.calculate_policy_sha256(
            invalid_path
        )
    )

    with pytest.raises(
        ValueError,
        match="Unable to load policy artifact",
    ):
        model_loader.load_policy(
            invalid_path,
            expected_sha256=invalid_sha256,
        )


def test_frozen_policy_rejects_wrong_baseline_version(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Reject a policy that identifies a different baseline model."""

    changed_bundle = replace(
        frozen_bundle,
        baseline_model_version="baseline-v2",
    )

    with pytest.raises(
        ValueError,
        match="Frozen baseline model version mismatch",
    ):
        model_loader._validate_frozen_policy(
            changed_bundle
        )


def test_frozen_policy_rejects_wrong_policy_version(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Reject a policy with a changed policy version."""

    changed_bundle = replace(
        frozen_bundle,
        policy_version="calibrated-policy-v2",
    )

    with pytest.raises(
        ValueError,
        match="Frozen policy version mismatch",
    ):
        model_loader._validate_frozen_policy(
            changed_bundle
        )


def test_frozen_policy_rejects_wrong_calibration_method(
    frozen_bundle: CalibratedPolicyBundle,
) -> None:
    """Reject a policy using a different calibration method."""

    changed_selection = replace(
        frozen_bundle.calibration_selection,
        selected_method="isotonic",
    )
    changed_bundle = replace(
        frozen_bundle,
        calibration_selection=changed_selection,
    )

    with pytest.raises(
        ValueError,
        match="Frozen calibration method mismatch",
    ):
        model_loader._validate_frozen_policy(
            changed_bundle
        )


@pytest.mark.parametrize(
    (
        "threshold_name",
        "changed_value",
        "expected_message",
    ),
    (
        (
            "review_threshold",
            FROZEN_REVIEW_THRESHOLD + 0.01,
            "Frozen review threshold mismatch",
        ),
        (
            "block_threshold",
            FROZEN_BLOCK_THRESHOLD - 0.01,
            "Frozen block threshold mismatch",
        ),
    ),
)
def test_frozen_policy_rejects_changed_thresholds(
    frozen_bundle: CalibratedPolicyBundle,
    threshold_name: str,
    changed_value: float,
    expected_message: str,
) -> None:
    """Reject any change to frozen review or block boundaries."""

    changed_thresholds = replace(
        frozen_bundle.thresholds,
        **{
            threshold_name: changed_value,
        },
    )
    changed_bundle = replace(
        frozen_bundle,
        thresholds=changed_thresholds,
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        model_loader._validate_frozen_policy(
            changed_bundle
        )


def test_frozen_policy_checks_explanation_version(
    frozen_bundle: CalibratedPolicyBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require the Phase 4 explanation contract to remain frozen."""

    assert (
        model_loader.EXPLANATION_VERSION
        == FROZEN_EXPLANATION_VERSION
    )

    monkeypatch.setattr(
        model_loader,
        "EXPLANATION_VERSION",
        "shap-explanation-v2",
    )

    with pytest.raises(
        ValueError,
        match="Frozen explanation version mismatch",
    ):
        model_loader._validate_frozen_policy(
            frozen_bundle
        )


def test_frozen_policy_checks_reason_code_version(
    frozen_bundle: CalibratedPolicyBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require the Phase 4 reason-code contract to remain frozen."""

    assert (
        model_loader.REASON_CODE_VERSION
        == FROZEN_REASON_CODE_VERSION
    )

    monkeypatch.setattr(
        model_loader,
        "REASON_CODE_VERSION",
        "reason-codes-v2",
    )

    with pytest.raises(
        ValueError,
        match="Frozen reason-code version mismatch",
    ):
        model_loader._validate_frozen_policy(
            frozen_bundle
        )


def test_cached_policy_refuses_different_path(
    tmp_path: Path,
) -> None:
    """Never replace an already loaded in-process policy."""

    model_loader.load_policy(
        POLICY_PATH
    )

    other_path = (
        tmp_path
        / "other-policy.joblib"
    )

    with pytest.raises(
        RuntimeError,
        match="already loaded",
    ):
        model_loader.load_policy(
            other_path
        )


def test_get_policy_requires_startup_load() -> None:
    """Do not expose an uninitialized process policy."""

    assert not model_loader.is_loaded()

    with pytest.raises(
        RuntimeError,
        match="Frozen policy not loaded",
    ):
        model_loader.get_policy()
