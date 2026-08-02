"""Tests for fraud probability calibration and selection."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from src.calibration import (
    CALIBRATION_METHODS,
    IDENTITY_METHOD,
    IdentityCalibrator,
    IsotonicCalibrator,
    SigmoidCalibrator,
    create_calibrator,
    fit_and_select_calibrator,
)


def _development_arrays() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Create scores whose empirical rates equal the scores."""

    scores = np.repeat(
        np.array(
            [0.1, 0.3, 0.6, 0.9],
            dtype="float64",
        ),
        10,
    )

    target = np.array(
        (
            [1] + [0] * 9
            + [1] * 3 + [0] * 7
            + [1] * 6 + [0] * 4
            + [1] * 9 + [0]
        ),
        dtype="int8",
    )

    return target, scores


def test_identity_calibrator_returns_scores_unchanged() -> None:
    calibrator = IdentityCalibrator().fit(
        [0, 1],
        [0.2, 0.8],
    )

    probabilities = (
        calibrator
        .predict_calibrated_probabilities(
            [0.1, 0.5, 0.9]
        )
    )

    np.testing.assert_array_equal(
        probabilities,
        np.array([0.1, 0.5, 0.9]),
    )


@pytest.mark.parametrize(
    "method_name",
    CALIBRATION_METHODS,
)
def test_calibrators_require_fit_before_prediction(
    method_name: str,
) -> None:
    calibrator = create_calibrator(
        method_name
    )

    with pytest.raises(
        RuntimeError,
        match="must be fitted",
    ):
        calibrator.predict_calibrated_probabilities(
            [0.2, 0.8]
        )


def test_sigmoid_calibrator_is_bounded_and_increasing() -> None:
    calibrator = SigmoidCalibrator().fit(
        [0, 0, 0, 1, 1, 1],
        [0.05, 0.15, 0.25, 0.75, 0.85, 0.95],
    )

    probabilities = (
        calibrator
        .predict_calibrated_probabilities(
            [0.0, 0.25, 0.5, 0.75, 1.0]
        )
    )

    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()
    assert (np.diff(probabilities) > 0.0).all()


def test_isotonic_calibrator_is_monotonic_and_clipped() -> None:
    calibrator = IsotonicCalibrator().fit(
        [0, 0, 1, 0, 1, 1],
        [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
    )

    probabilities = (
        calibrator
        .predict_calibrated_probabilities(
            [0.0, 0.15, 0.5, 0.85, 1.0]
        )
    )

    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()
    assert (np.diff(probabilities) >= 0.0).all()

    assert probabilities[0] == pytest.approx(0.0)
    assert probabilities[-1] == pytest.approx(1.0)


def test_create_calibrator_rejects_unknown_method() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported calibration method",
    ):
        create_calibrator("unknown")


def test_selection_evaluates_candidates_and_uses_tie_break() -> None:
    target, scores = _development_arrays()

    selection = fit_and_select_calibrator(
        target,
        scores,
        target,
        scores,
    )

    assert selection.selected_method == (
        IDENTITY_METHOD
    )

    assert [
        candidate.method_name
        for candidate in selection.candidate_results
    ] == list(CALIBRATION_METHODS)

    assert all(
        candidate.eligible
        for candidate in selection.candidate_results
    )

    identity_result = (
        selection.candidate_results[0]
    )
    isotonic_result = (
        selection.candidate_results[2]
    )

    assert identity_result.selection_key is not None
    assert isotonic_result.selection_key is not None

    assert (
        identity_result.selection_key[:3]
        == isotonic_result.selection_key[:3]
    )

    assert (
        identity_result.complexity_rank
        < isotonic_result.complexity_rank
    )

    assert identity_result.metrics is not None
    assert (
        identity_result.metrics[
            "expected_calibration_error"
        ]
        == pytest.approx(0.0)
    )


def test_selection_is_deterministic() -> None:
    target, scores = _development_arrays()

    first = fit_and_select_calibrator(
        target,
        scores,
        target,
        scores,
    )
    second = fit_and_select_calibrator(
        target,
        scores,
        target,
        scores,
    )

    assert first.selected_method == (
        second.selected_method
    )

    assert [
        candidate.method_name
        for candidate in first.candidate_results
    ] == [
        candidate.method_name
        for candidate in second.candidate_results
    ]

    for first_candidate, second_candidate in zip(
        first.candidate_results,
        second.candidate_results,
        strict=True,
    ):
        assert first_candidate.eligible == (
            second_candidate.eligible
        )
        assert (
            first_candidate.disqualification_reason
            == second_candidate.disqualification_reason
        )
        assert first_candidate.selection_key == (
            second_candidate.selection_key
        )

        if first_candidate.metrics is None:
            assert second_candidate.metrics is None
            continue

        assert second_candidate.metrics is not None
        assert first_candidate.metrics.keys() == (
            second_candidate.metrics.keys()
        )

        for metric_name, first_value in (
            first_candidate.metrics.items()
        ):
            second_value = (
                second_candidate.metrics[
                    metric_name
                ]
            )

            if isinstance(first_value, float):
                assert second_value == pytest.approx(
                    first_value,
                    rel=1e-12,
                    abs=1e-15,
                )
            else:
                assert second_value == first_value

    np.testing.assert_allclose(
        first.predict_calibrated_probabilities(
            scores
        ),
        second.predict_calibrated_probabilities(
            scores
        ),
        rtol=1e-12,
        atol=1e-15,
    )


def test_selection_survives_joblib_round_trip(
    tmp_path: Path,
) -> None:
    target, scores = _development_arrays()

    selection = fit_and_select_calibrator(
        target,
        scores,
        target,
        scores,
    )

    artifact_path = (
        tmp_path / "calibration-selection.joblib"
    )

    joblib.dump(
        selection,
        artifact_path,
    )

    loaded_selection = joblib.load(
        artifact_path
    )

    assert loaded_selection.selected_method == (
        selection.selected_method
    )
    assert loaded_selection.candidate_results == (
        selection.candidate_results
    )

    np.testing.assert_allclose(
        loaded_selection
        .predict_calibrated_probabilities(
            scores
        ),
        selection
        .predict_calibrated_probabilities(
            scores
        ),
    )


@pytest.mark.parametrize(
    ("target", "scores", "message"),
    [
        (
            [0, 2],
            [0.2, 0.8],
            "binary values",
        ),
        (
            [0, 0],
            [0.2, 0.8],
            "both binary classes",
        ),
        (
            [0, 1],
            [0.2],
            "same length",
        ),
    ],
)
def test_selection_rejects_invalid_fit_targets(
    target: list[int],
    scores: list[float],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        fit_and_select_calibrator(
            target,
            scores,
            [0, 1],
            [0.2, 0.8],
        )


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (
            [-0.1, 0.8],
            "between 0 and 1",
        ),
        (
            [0.2, np.nan],
            "finite values",
        ),
        (
            [[0.2], [0.8]],
            "one-dimensional",
        ),
    ],
)
def test_selection_rejects_invalid_scores(
    scores: list[float],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        fit_and_select_calibrator(
            [0, 1],
            scores,
            [0, 1],
            [0.2, 0.8],
        )


@pytest.mark.parametrize(
    ("keyword_arguments", "message"),
    [
        (
            {"bin_count": 1},
            "at least 2",
        ),
        (
            {"selection_round_digits": -1},
            "between 0 and 15",
        ),
        (
            {"selection_round_digits": 16},
            "between 0 and 15",
        ),
    ],
)
def test_selection_rejects_invalid_configuration(
    keyword_arguments: dict[str, int],
    message: str,
) -> None:
    target, scores = _development_arrays()

    with pytest.raises(
        ValueError,
        match=message,
    ):
        fit_and_select_calibrator(
            target,
            scores,
            target,
            scores,
            **keyword_arguments,
        )
