"""Streamlit rendering for the Phase 6 monitoring view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api.config import (
    FROZEN_BASELINE_MODEL_VERSION,
    FROZEN_BLOCK_THRESHOLD,
    FROZEN_CALIBRATION_METHOD,
    FROZEN_EXPLANATION_VERSION,
    FROZEN_POLICY_SHA256,
    FROZEN_POLICY_VERSION,
    FROZEN_REASON_CODE_VERSION,
    FROZEN_REVIEW_THRESHOLD,
)
from dashboard.monitoring import (
    MonitoringSnapshot,
)


def _format_rate(
    value: float,
) -> str:
    """Format a fractional rate for dashboard display."""

    return f"{value:.1%}"


def _format_probability(
    value: float | None,
) -> str:
    """Format a probability or return an unavailable marker."""

    if value is None:
        return "—"

    return f"{value:.3f}"


def _configured_policy_frame() -> pd.DataFrame:
    """Return the immutable configured frozen-policy contract."""

    return pd.DataFrame(
        [
            {
                "Model version": (
                    FROZEN_BASELINE_MODEL_VERSION
                ),
                "Policy version": (
                    FROZEN_POLICY_VERSION
                ),
                "Calibration": (
                    FROZEN_CALIBRATION_METHOD
                ),
                "Review threshold": (
                    FROZEN_REVIEW_THRESHOLD
                ),
                "Block threshold": (
                    FROZEN_BLOCK_THRESHOLD
                ),
                "Explanation version": (
                    FROZEN_EXPLANATION_VERSION
                ),
                "Reason-code version": (
                    FROZEN_REASON_CODE_VERSION
                ),
                "Policy SHA-256": (
                    FROZEN_POLICY_SHA256
                ),
                "Persisted events": 0,
            }
        ]
    )


def _persisted_policy_frame(
    snapshot: MonitoringSnapshot,
) -> pd.DataFrame:
    """Return persisted frozen-policy provenance rows."""

    return pd.DataFrame(
        [
            {
                "Model version": (
                    provenance.model_version
                ),
                "Policy version": (
                    provenance.policy_version
                ),
                "Calibration": (
                    provenance.calibration_method
                ),
                "Review threshold": (
                    provenance.review_threshold
                ),
                "Block threshold": (
                    provenance.block_threshold
                ),
                "Explanation version": (
                    provenance.explanation_version
                ),
                "Reason-code version": (
                    provenance.reason_code_version
                ),
                "Policy SHA-256": (
                    provenance
                    .policy_artifact_sha256
                ),
                "Persisted events": (
                    provenance.event_count
                ),
            }
            for provenance
            in snapshot.policy_provenance
        ]
    )


def _decision_distribution_frame(
    snapshot: MonitoringSnapshot,
) -> pd.DataFrame:
    """Return actual persisted decision counts and rates."""

    overview = (
        snapshot.overview
    )

    return pd.DataFrame(
        [
            {
                "Decision": "ALLOW",
                "Events": (
                    overview.allow_count
                ),
                "Rate": (
                    overview.allow_rate
                ),
            },
            {
                "Decision": "REVIEW",
                "Events": (
                    overview.review_count
                ),
                "Rate": (
                    overview.review_rate
                ),
            },
            {
                "Decision": "BLOCK",
                "Events": (
                    overview.block_count
                ),
                "Rate": (
                    overview.block_rate
                ),
            },
        ]
    )


def _score_frame(
    snapshot: MonitoringSnapshot,
) -> pd.DataFrame:
    """Return chronological persisted calibrated-score observations."""

    return pd.DataFrame(
        [
            {
                "Scored at UTC": (
                    observation.scored_at_utc
                ),
                "Prediction ID": (
                    observation.prediction_id
                ),
                "Transaction ID": (
                    observation.transaction_id
                ),
                "Calibrated probability": (
                    observation
                    .calibrated_probability
                ),
                "Decision": (
                    observation.decision
                ),
            }
            for observation
            in snapshot.score_observations
        ]
    )


def _reason_code_frame(
    snapshot: MonitoringSnapshot,
) -> pd.DataFrame:
    """Return event-level reason-code frequencies."""

    return pd.DataFrame(
        [
            {
                "Reason code": (
                    frequency.reason_code
                ),
                "Events": (
                    frequency.event_count
                ),
                "Event rate": (
                    frequency.event_rate
                ),
            }
            for frequency
            in snapshot.reason_code_frequencies
        ]
    )


def _recent_flagged_frame(
    snapshot: MonitoringSnapshot,
) -> pd.DataFrame:
    """Return recent persisted REVIEW/BLOCK predictions."""

    return pd.DataFrame(
        [
            {
                "Prediction ID": (
                    prediction.prediction_id
                ),
                "Transaction ID": (
                    prediction.transaction_id
                ),
                "Scored at UTC": (
                    prediction.scored_at_utc
                ),
                "Calibrated probability": (
                    prediction
                    .calibrated_probability
                ),
                "Decision": (
                    prediction.decision
                ),
                "Transaction amount": (
                    prediction
                    .transaction_amount
                ),
                "Reason codes": (
                    ", ".join(
                        prediction.reason_codes
                    )
                ),
                "Actual label": (
                    prediction.actual_label
                ),
            }
            for prediction
            in (
                snapshot
                .recent_flagged_predictions
            )
        ]
    )


def render_monitoring_view(
    snapshot: MonitoringSnapshot,
) -> None:
    """Render actual persisted frozen-policy monitoring."""

    overview = (
        snapshot.overview
    )

    st.header(
        "Monitoring"
    )
    st.caption(
        "Read-only monitoring of prediction events produced by "
        "the frozen PayGuard policy. Displayed decisions are the "
        "decisions actually persisted by the API."
    )

    if overview.total_events == 0:
        st.info(
            "No prediction events have been persisted yet. "
            "Start the API and score transactions to populate monitoring."
        )

    st.subheader(
        "Frozen Policy Contract"
    )

    if (
        snapshot.policy_provenance
    ):
        policy_frame = (
            _persisted_policy_frame(
                snapshot
            )
        )

        if (
            len(
                snapshot
                .policy_provenance
            )
            > 1
        ):
            st.warning(
                "Multiple persisted policy contracts were found. "
                "Monitoring keeps them separate rather than silently "
                "combining their provenance."
            )

    else:
        st.caption(
            "No event provenance exists yet; showing the configured "
            "immutable frozen contract."
        )

        policy_frame = (
            _configured_policy_frame()
        )

    st.dataframe(
        policy_frame,
        hide_index=True,
        width="stretch",
    )

    st.subheader(
        "Operational Overview"
    )

    metric_columns = (
        st.columns(
            4
        )
    )

    metric_columns[
        0
    ].metric(
        "Prediction events",
        str(
            overview.total_events
        ),
    )

    metric_columns[
        1
    ].metric(
        "Intervention rate",
        _format_rate(
            overview
            .intervention_rate
        ),
    )

    metric_columns[
        2
    ].metric(
        "Label coverage",
        _format_rate(
            overview
            .label_coverage_rate
        ),
    )

    metric_columns[
        3
    ].metric(
        "Average calibrated probability",
        _format_probability(
            overview
            .average_calibrated_probability
        ),
    )

    st.subheader(
        "Actual Decision Distribution"
    )
    st.caption(
        "ALLOW, REVIEW, and BLOCK counts come directly from persisted "
        "production-policy decisions; they are not recomputed from "
        "dashboard controls."
    )

    decision_frame = (
        _decision_distribution_frame(
            snapshot
        )
    )

    st.bar_chart(
        decision_frame.set_index(
            "Decision"
        )[
            "Events"
        ]
    )

    decision_table = (
        decision_frame.copy()
    )
    decision_table[
        "Rate"
    ] = (
        decision_table[
            "Rate"
        ]
        .map(
            _format_rate
        )
    )

    st.dataframe(
        decision_table,
        hide_index=True,
        width="stretch",
    )

    st.subheader(
        "Calibrated Score Monitoring"
    )

    score_frame = (
        _score_frame(
            snapshot
        )
    )

    if score_frame.empty:
        st.caption(
            "No calibrated-score observations are available."
        )

    else:
        st.line_chart(
            score_frame,
            x="Scored at UTC",
            y=(
                "Calibrated probability"
            ),
        )

        st.dataframe(
            score_frame,
            hide_index=True,
            use_container_width=True,
        )

    st.subheader(
        "Reason Code Frequency"
    )

    reason_frame = (
        _reason_code_frame(
            snapshot
        )
    )

    if reason_frame.empty:
        st.caption(
            "No analyst reason codes are available in the "
            "current monitoring selection."
        )

    else:
        st.bar_chart(
            reason_frame.set_index(
                "Reason code"
            )[
                "Events"
            ]
        )

        reason_table = (
            reason_frame.copy()
        )
        reason_table[
            "Event rate"
        ] = (
            reason_table[
                "Event rate"
            ]
            .map(
                _format_rate
            )
        )

        st.dataframe(
            reason_table,
            hide_index=True,
            use_container_width=True,
        )

    st.subheader(
        "Recent Review / Block Queue"
    )

    recent_frame = (
        _recent_flagged_frame(
            snapshot
        )
    )

    if recent_frame.empty:
        st.caption(
            "No REVIEW or BLOCK prediction events are available."
        )

    else:
        st.dataframe(
            recent_frame,
            hide_index=True,
            use_container_width=True,
        )

    st.subheader(
        "Outcome Coverage"
    )

    outcome_columns = (
        st.columns(
            3
        )
    )

    outcome_columns[
        0
    ].metric(
        "Labeled events",
        str(
            overview.labeled_count
        ),
    )

    outcome_columns[
        1
    ].metric(
        "Unlabeled events",
        str(
            overview.unlabeled_count
        ),
    )

    outcome_columns[
        2
    ].metric(
        "Coverage",
        _format_rate(
            overview
            .label_coverage_rate
        ),
    )

    st.caption(
        "Monitoring reports ground-truth coverage only. "
        "Fraud recall, precision, fraud capture, and economic metrics "
        "are not inferred from unlabeled events."
    )
