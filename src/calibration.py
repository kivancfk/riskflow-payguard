"""Deterministic fraud-probability calibration and selection."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.calibration_metrics import (
    DEFAULT_BIN_COUNT,
    build_reliability_table,
    calculate_calibration_errors,
)
from src.evaluation import evaluate_probability_metrics


IDENTITY_METHOD = "identity"
SIGMOID_METHOD = "sigmoid"
ISOTONIC_METHOD = "isotonic"

CALIBRATION_METHODS: tuple[str, ...] = (
    IDENTITY_METHOD,
    SIGMOID_METHOD,
    ISOTONIC_METHOD,
)

CALIBRATOR_COMPLEXITY: dict[str, int] = {
    method_name: complexity_rank
    for complexity_rank, method_name in enumerate(
        CALIBRATION_METHODS
    )
}

DEFAULT_SELECTION_ROUND_DIGITS = 12
MAX_SELECTION_ROUND_DIGITS = 15

SIGMOID_MAX_ITERATIONS = 1_000
SIGMOID_REGULARIZATION_C = 1_000_000.0
SIGMOID_RANDOM_STATE = 42


def _as_float_vector(
    values: ArrayLike,
    *,
    value_name: str,
) -> np.ndarray:
    """Convert input into a finite one-dimensional float vector."""

    try:
        vector = np.asarray(
            values,
            dtype="float64",
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

    return vector


def _validate_scores(
    raw_scores: ArrayLike,
    *,
    value_name: str = "raw_scores",
) -> np.ndarray:
    """Validate model scores or calibrated probabilities."""

    scores = _as_float_vector(
        raw_scores,
        value_name=value_name,
    )

    if (
        (scores < 0.0).any()
        or (scores > 1.0).any()
    ):
        raise ValueError(
            f"{value_name} must be between 0 and 1"
        )

    return scores


def _validate_fit_inputs(
    y_true: ArrayLike,
    raw_scores: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned binary targets and model scores."""

    target = _as_float_vector(
        y_true,
        value_name="y_true",
    )
    scores = _validate_scores(raw_scores)

    if len(target) != len(scores):
        raise ValueError(
            "y_true and raw_scores must have the same length"
        )

    if not np.isin(
        target,
        [0.0, 1.0],
    ).all():
        raise ValueError(
            "y_true must contain only binary values 0 and 1"
        )

    if set(target.astype("int8").tolist()) != {0, 1}:
        raise ValueError(
            "y_true must contain both binary classes"
        )

    return (
        target.astype("int8"),
        scores,
    )


def _validate_bin_count(bin_count: int) -> int:
    """Return a valid reliability-bin count."""

    if (
        isinstance(bin_count, bool)
        or not isinstance(
            bin_count,
            (int, np.integer),
        )
        or int(bin_count) < 2
    ):
        raise ValueError(
            "bin_count must be an integer of at least 2"
        )

    return int(bin_count)


def _validate_selection_round_digits(
    selection_round_digits: int,
) -> int:
    """Return valid deterministic comparison precision."""

    if (
        isinstance(selection_round_digits, bool)
        or not isinstance(
            selection_round_digits,
            (int, np.integer),
        )
        or not (
            0
            <= int(selection_round_digits)
            <= MAX_SELECTION_ROUND_DIGITS
        )
    ):
        raise ValueError(
            "selection_round_digits must be an integer "
            "between 0 and 15"
        )

    return int(selection_round_digits)


class IdentityCalibrator:
    """Return the frozen model score without modification."""

    method_name = IDENTITY_METHOD

    def __init__(self) -> None:
        self._is_fitted = False

    def fit(
        self,
        y_true: ArrayLike,
        raw_scores: ArrayLike,
    ) -> IdentityCalibrator:
        """Validate development data and mark the identity mapping fitted."""

        _validate_fit_inputs(
            y_true,
            raw_scores,
        )

        self._is_fitted = True

        return self

    def predict_calibrated_probabilities(
        self,
        raw_scores: ArrayLike,
    ) -> np.ndarray:
        """Return a validated copy of the input score."""

        if not self._is_fitted:
            raise RuntimeError(
                "identity calibrator must be fitted "
                "before prediction"
            )

        return _validate_scores(
            raw_scores
        ).copy()


class SigmoidCalibrator:
    """Fit a one-dimensional logistic probability mapping."""

    method_name = SIGMOID_METHOD

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None

    def fit(
        self,
        y_true: ArrayLike,
        raw_scores: ArrayLike,
    ) -> SigmoidCalibrator:
        """Fit sigmoid calibration using calibration-fit data."""

        target, scores = _validate_fit_inputs(
            y_true,
            raw_scores,
        )

        model = LogisticRegression(
            C=SIGMOID_REGULARIZATION_C,
            solver="lbfgs",
            max_iter=SIGMOID_MAX_ITERATIONS,
            random_state=SIGMOID_RANDOM_STATE,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                category=ConvergenceWarning,
            )

            try:
                model.fit(
                    scores.reshape(-1, 1),
                    target,
                )
            except ConvergenceWarning as error:
                raise ValueError(
                    "sigmoid calibration did not converge"
                ) from error

        coefficient = float(
            model.coef_[0, 0]
        )

        if (
            not math.isfinite(coefficient)
            or coefficient <= 0.0
        ):
            raise ValueError(
                "sigmoid calibration must learn "
                "a positive finite slope"
            )

        self._model = model

        return self

    def predict_calibrated_probabilities(
        self,
        raw_scores: ArrayLike,
    ) -> np.ndarray:
        """Apply the fitted sigmoid mapping."""

        if self._model is None:
            raise RuntimeError(
                "sigmoid calibrator must be fitted "
                "before prediction"
            )

        scores = _validate_scores(raw_scores)

        calibrated_probabilities = np.asarray(
            self._model.predict_proba(
                scores.reshape(-1, 1)
            )[:, 1],
            dtype="float64",
        )

        return _validate_scores(
            calibrated_probabilities,
            value_name="calibrated_probabilities",
        )


class IsotonicCalibrator:
    """Fit a monotonic non-parametric probability mapping."""

    method_name = ISOTONIC_METHOD

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None

    def fit(
        self,
        y_true: ArrayLike,
        raw_scores: ArrayLike,
    ) -> IsotonicCalibrator:
        """Fit isotonic calibration using calibration-fit data."""

        target, scores = _validate_fit_inputs(
            y_true,
            raw_scores,
        )

        model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )

        model.fit(
            scores,
            target,
        )

        self._model = model

        return self

    def predict_calibrated_probabilities(
        self,
        raw_scores: ArrayLike,
    ) -> np.ndarray:
        """Apply the fitted monotonic mapping."""

        if self._model is None:
            raise RuntimeError(
                "isotonic calibrator must be fitted "
                "before prediction"
            )

        scores = _validate_scores(raw_scores)

        calibrated_probabilities = np.asarray(
            self._model.predict(scores),
            dtype="float64",
        )

        return _validate_scores(
            calibrated_probabilities,
            value_name="calibrated_probabilities",
        )


ProbabilityCalibrator = (
    IdentityCalibrator
    | SigmoidCalibrator
    | IsotonicCalibrator
)


@dataclass(frozen=True)
class CalibrationCandidateResult:
    """Policy-selection metrics for one calibration candidate."""

    method_name: str
    complexity_rank: int
    eligible: bool
    disqualification_reason: str | None
    metrics: dict[str, float | int] | None
    selection_key: (
        tuple[float, float, float, int]
        | None
    )


@dataclass(frozen=True)
class CalibrationSelection:
    """Selected fitted calibrator and candidate comparison."""

    selected_method: str
    selected_calibrator: ProbabilityCalibrator
    candidate_results: tuple[
        CalibrationCandidateResult,
        ...,
    ]
    bin_count: int
    selection_round_digits: int

    def predict_calibrated_probabilities(
        self,
        raw_scores: ArrayLike,
    ) -> np.ndarray:
        """Apply the selected fitted calibrator."""

        return (
            self.selected_calibrator
            .predict_calibrated_probabilities(
                raw_scores
            )
        )


def create_calibrator(
    method_name: str,
) -> ProbabilityCalibrator:
    """Create one unfitted calibration candidate."""

    if not isinstance(method_name, str):
        raise ValueError(
            "method_name must be a string"
        )

    normalized_method = (
        method_name.strip().lower()
    )

    if normalized_method == IDENTITY_METHOD:
        return IdentityCalibrator()

    if normalized_method == SIGMOID_METHOD:
        return SigmoidCalibrator()

    if normalized_method == ISOTONIC_METHOD:
        return IsotonicCalibrator()

    raise ValueError(
        "Unsupported calibration method: "
        f"{method_name}"
    )


def _evaluate_candidate(
    y_true: np.ndarray,
    calibrated_probabilities: np.ndarray,
    *,
    bin_count: int,
) -> dict[str, float | int]:
    """Calculate candidate metrics on policy-selection data."""

    probability_metrics = (
        evaluate_probability_metrics(
            y_true,
            calibrated_probabilities,
        )
    )

    reliability_table = build_reliability_table(
        y_true,
        calibrated_probabilities,
        bin_count=bin_count,
    )

    calibration_errors = (
        calculate_calibration_errors(
            reliability_table
        )
    )

    return {
        **probability_metrics,
        **calibration_errors,
        "minimum_probability": float(
            calibrated_probabilities.min()
        ),
        "maximum_probability": float(
            calibrated_probabilities.max()
        ),
        "mean_probability": float(
            calibrated_probabilities.mean()
        ),
    }


def _build_selection_key(
    metrics: dict[str, float | int],
    *,
    complexity_rank: int,
    selection_round_digits: int,
) -> tuple[float, float, float, int]:
    """Create the deterministic lexicographic selection key."""

    return (
        round(
            float(metrics["log_loss"]),
            selection_round_digits,
        ),
        round(
            float(metrics["brier_score"]),
            selection_round_digits,
        ),
        round(
            float(
                metrics[
                    "expected_calibration_error"
                ]
            ),
            selection_round_digits,
        ),
        complexity_rank,
    )


def _eligible_selection_key(
    candidate: CalibrationCandidateResult,
) -> tuple[float, float, float, int]:
    """Return the key for an eligible candidate."""

    if candidate.selection_key is None:
        raise RuntimeError(
            "Eligible calibration candidate "
            "does not contain a selection key"
        )

    return candidate.selection_key


def fit_and_select_calibrator(
    calibration_fit_target: ArrayLike,
    calibration_fit_scores: ArrayLike,
    policy_selection_target: ArrayLike,
    policy_selection_scores: ArrayLike,
    *,
    bin_count: int = DEFAULT_BIN_COUNT,
    selection_round_digits: int = (
        DEFAULT_SELECTION_ROUND_DIGITS
    ),
) -> CalibrationSelection:
    """Fit candidates and select using policy-selection data only."""

    fit_target, fit_scores = _validate_fit_inputs(
        calibration_fit_target,
        calibration_fit_scores,
    )

    policy_target, policy_scores = (
        _validate_fit_inputs(
            policy_selection_target,
            policy_selection_scores,
        )
    )

    validated_bin_count = _validate_bin_count(
        bin_count
    )
    validated_round_digits = (
        _validate_selection_round_digits(
            selection_round_digits
        )
    )

    candidate_results: list[
        CalibrationCandidateResult
    ] = []
    fitted_calibrators: dict[
        str,
        ProbabilityCalibrator,
    ] = {}

    for method_name in CALIBRATION_METHODS:
        complexity_rank = (
            CALIBRATOR_COMPLEXITY[
                method_name
            ]
        )
        calibrator = create_calibrator(
            method_name
        )

        try:
            calibrator.fit(
                fit_target,
                fit_scores,
            )

            calibrated_probabilities = (
                calibrator
                .predict_calibrated_probabilities(
                    policy_scores
                )
            )

            metrics = _evaluate_candidate(
                policy_target,
                calibrated_probabilities,
                bin_count=validated_bin_count,
            )

            selection_key = _build_selection_key(
                metrics,
                complexity_rank=complexity_rank,
                selection_round_digits=(
                    validated_round_digits
                ),
            )

            fitted_calibrators[
                method_name
            ] = calibrator

            candidate_results.append(
                CalibrationCandidateResult(
                    method_name=method_name,
                    complexity_rank=(
                        complexity_rank
                    ),
                    eligible=True,
                    disqualification_reason=None,
                    metrics=metrics,
                    selection_key=selection_key,
                )
            )
        except (ValueError, RuntimeError) as error:
            candidate_results.append(
                CalibrationCandidateResult(
                    method_name=method_name,
                    complexity_rank=(
                        complexity_rank
                    ),
                    eligible=False,
                    disqualification_reason=(
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                    metrics=None,
                    selection_key=None,
                )
            )

    eligible_candidates = [
        candidate
        for candidate in candidate_results
        if candidate.eligible
    ]

    if not eligible_candidates:
        raise RuntimeError(
            "No calibration candidate produced "
            "valid policy-selection probabilities"
        )

    selected_candidate = min(
        eligible_candidates,
        key=_eligible_selection_key,
    )

    return CalibrationSelection(
        selected_method=(
            selected_candidate.method_name
        ),
        selected_calibrator=(
            fitted_calibrators[
                selected_candidate.method_name
            ]
        ),
        candidate_results=tuple(
            candidate_results
        ),
        bin_count=validated_bin_count,
        selection_round_digits=(
            validated_round_digits
        ),
    )
