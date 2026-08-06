"""Generate deterministic per-row model-signal explanations."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import pandas as pd

from src.explainability import (
    DEFAULT_TOP_CONTRIBUTIONS,
    MISSING,
    OBSERVED,
    UNKNOWN_CATEGORY,
    ContributionSelection,
    FeatureContribution,
    select_top_contributions,
)
from src.explanation_values import (
    classify_feature_value_states,
)
from src.lightgbm_shap import (
    LightGBMContributionMatrix,
    extract_lightgbm_contributions,
)
from src.model_bundle import (
    ModelBundle,
    _validate_model_bundle,
)
from src.reason_codes import (
    AnalystReason,
    build_analyst_reason,
    feature_group_for,
)


_ALLOWED_VALUE_STATES = {
    OBSERVED,
    MISSING,
    UNKNOWN_CATEGORY,
}


def _deduplicated_reasons(
    selection: ContributionSelection,
) -> tuple[AnalystReason, ...]:
    """Build unique reasons in deterministic contribution order."""

    reasons: list[
        AnalystReason
    ] = []
    seen_codes: set[
        str
    ] = set()

    ordered_contributions = (
        *selection.top_positive_contributions,
        *selection.top_negative_contributions,
    )

    for contribution in ordered_contributions:
        reason = build_analyst_reason(
            contribution
        )

        if reason.code in seen_codes:
            continue

        seen_codes.add(
            reason.code
        )
        reasons.append(
            reason
        )

    return tuple(
        reasons
    )


@dataclass(frozen=True)
class ModelSignalExplanation:
    """Selected contributions and analyst reasons for one row."""

    row_position: int
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
        if (
            isinstance(
                self.row_position,
                bool,
            )
            or not isinstance(
                self.row_position,
                Integral,
            )
            or int(
                self.row_position
            ) < 0
        ):
            raise ValueError(
                "row_position must be an integer "
                "greater than or equal to 0"
            )

        normalized_position = int(
            self.row_position
        )

        selection = ContributionSelection(
            top_positive_contributions=(
                self.top_positive_contributions
            ),
            top_negative_contributions=(
                self.top_negative_contributions
            ),
        )

        if not isinstance(
            self.reasons,
            tuple,
        ):
            raise ValueError(
                "reasons must be a tuple"
            )

        for reason in self.reasons:
            if not isinstance(
                reason,
                AnalystReason,
            ):
                raise ValueError(
                    "reasons must contain only "
                    "AnalystReason instances"
                )

        expected_reasons = (
            _deduplicated_reasons(
                selection
            )
        )

        if self.reasons != expected_reasons:
            raise ValueError(
                "reasons must be unique and follow "
                "positive-before-negative "
                "contribution order"
            )

        object.__setattr__(
            self,
            "row_position",
            normalized_position,
        )

    @property
    def contribution_selection(
        self,
    ) -> ContributionSelection:
        """Return the validated contribution selection."""

        return ContributionSelection(
            top_positive_contributions=(
                self.top_positive_contributions
            ),
            top_negative_contributions=(
                self.top_negative_contributions
            ),
        )

    @property
    def selected_contributions(
        self,
    ) -> tuple[
        FeatureContribution,
        ...,
    ]:
        """Return positive contributions followed by negative ones."""

        return (
            *self.top_positive_contributions,
            *self.top_negative_contributions,
        )

    @property
    def reason_codes(
        self,
    ) -> tuple[
        str,
        ...,
    ]:
        """Return stable reason codes in final output order."""

        return tuple(
            reason.code
            for reason in self.reasons
        )


def _validate_value_states(
    value_states: object,
    features: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """Validate a precomputed value-state frame."""

    if not isinstance(
        value_states,
        pd.DataFrame,
    ):
        raise ValueError(
            "value_states must be a pandas DataFrame"
        )

    if list(
        value_states.columns
    ) != list(
        feature_names
    ):
        raise ValueError(
            "value-state columns do not match "
            "the frozen feature order"
        )

    if not value_states.index.equals(
        features.index
    ):
        raise ValueError(
            "value-state index does not align "
            "with the feature frame"
        )

    for row_position in range(
        len(value_states)
    ):
        for feature_position, feature_name in enumerate(
            feature_names
        ):
            value_state = value_states.iat[
                row_position,
                feature_position,
            ]

            if (
                not isinstance(
                    value_state,
                    str,
                )
                or value_state
                not in _ALLOWED_VALUE_STATES
            ):
                raise ValueError(
                    "value_states contains an invalid "
                    "state at row "
                    f"{row_position}, feature "
                    f"{feature_name}"
                )

    return value_states


def generate_model_signal_explanations(
    bundle: ModelBundle,
    features: pd.DataFrame,
    *,
    contributions: (
        LightGBMContributionMatrix | None
    ) = None,
    value_states: pd.DataFrame | None = None,
    top_k: int = DEFAULT_TOP_CONTRIBUTIONS,
    frame_name: str = "explanation features",
) -> tuple[
    ModelSignalExplanation,
    ...,
]:
    """Generate deterministic contribution and reason records per row."""

    validated_bundle = _validate_model_bundle(
        bundle
    )

    validated_bundle.encoder.transform(
        features,
        frame_name=frame_name,
    )

    if contributions is None:
        resolved_contributions = (
            extract_lightgbm_contributions(
                validated_bundle,
                features,
                frame_name=frame_name,
            )
        )
    else:
        if not isinstance(
            contributions,
            LightGBMContributionMatrix,
        ):
            raise ValueError(
                "contributions must be a "
                "LightGBMContributionMatrix"
            )

        resolved_contributions = contributions

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

    if value_states is None:
        resolved_value_states = (
            classify_feature_value_states(
                features,
                validated_bundle.encoder,
                frame_name=frame_name,
            )
        )
    else:
        resolved_value_states = (
            _validate_value_states(
                value_states,
                features,
                validated_bundle.feature_columns,
            )
        )

    feature_groups = tuple(
        feature_group_for(
            feature_name
        )
        for feature_name in (
            validated_bundle.feature_columns
        )
    )

    explanations: list[
        ModelSignalExplanation
    ] = []

    for row_position in range(
        len(features)
    ):
        selection = select_top_contributions(
            feature_names=(
                validated_bundle.feature_columns
            ),
            shap_values_raw=(
                resolved_contributions
                .shap_values_raw[
                    row_position,
                    :,
                ]
            ),
            feature_groups=feature_groups,
            value_states=tuple(
                resolved_value_states.iloc[
                    row_position
                ].tolist()
            ),
            top_k=top_k,
        )

        explanations.append(
            ModelSignalExplanation(
                row_position=row_position,
                top_positive_contributions=(
                    selection
                    .top_positive_contributions
                ),
                top_negative_contributions=(
                    selection
                    .top_negative_contributions
                ),
                reasons=_deduplicated_reasons(
                    selection
                ),
            )
        )

    return tuple(
        explanations
    )
