"""Streamlit rendering for the Phase 6 threshold simulator."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api.logging_db import PredictionStore
from api.model_loader import LoadedPolicy
from dashboard.simulation import (
    LabeledPolicyComparison,
    ThresholdSimulationResult,
    simulate_threshold_scenario,
)
from src.policy import PolicyThresholds


def _format_rate(
    value: float,
) -> str:
    """Format a fractional rate for display."""

    return f"{value:.1%}"


def _format_threshold(
    value: float,
) -> str:
    """Format a threshold without implying policy mutation."""

    return f"{value:.9f}"


def _threshold_frame(
    result: ThresholdSimulationResult,
) -> pd.DataFrame:
    """Return frozen and temporary candidate thresholds."""

    return pd.DataFrame(
        [
            {
                "Policy": "Frozen",
                "Review threshold": (
                    result
                    .frozen_thresholds
                    .review_threshold
                ),
                "Block threshold": (
                    result
                    .frozen_thresholds
                    .block_threshold
                ),
            },
            {
                "Policy": "Candidate",
                "Review threshold": (
                    result
                    .candidate_thresholds
                    .review_threshold
                ),
                "Block threshold": (
                    result
                    .candidate_thresholds
                    .block_threshold
                ),
            },
        ]
    )


def _workload_frame(
    result: ThresholdSimulationResult,
) -> pd.DataFrame:
    """Return frozen-versus-candidate decision workload."""

    frozen = (
        result.frozen_workload
    )
    candidate = (
        result.candidate_workload
    )

    return pd.DataFrame(
        [
            {
                "Decision": "ALLOW",
                "Frozen events": (
                    frozen.allow_count
                ),
                "Frozen rate": (
                    frozen.allow_rate
                ),
                "Candidate events": (
                    candidate.allow_count
                ),
                "Candidate rate": (
                    candidate.allow_rate
                ),
            },
            {
                "Decision": "REVIEW",
                "Frozen events": (
                    frozen.review_count
                ),
                "Frozen rate": (
                    frozen.review_rate
                ),
                "Candidate events": (
                    candidate.review_count
                ),
                "Candidate rate": (
                    candidate.review_rate
                ),
            },
            {
                "Decision": "BLOCK",
                "Frozen events": (
                    frozen.block_count
                ),
                "Frozen rate": (
                    frozen.block_rate
                ),
                "Candidate events": (
                    candidate.block_count
                ),
                "Candidate rate": (
                    candidate.block_rate
                ),
            },
        ]
    )


def _transition_frame(
    result: ThresholdSimulationResult,
) -> pd.DataFrame:
    """Return actual-to-candidate decision transitions."""

    return pd.DataFrame(
        [
            {
                "Frozen decision": (
                    transition
                    .frozen_decision
                ),
                "Candidate decision": (
                    transition
                    .candidate_decision
                ),
                "Events": (
                    transition
                    .event_count
                ),
                "Event rate": (
                    transition
                    .event_rate
                ),
            }
            for transition
            in result.transitions
        ]
    )


def _constraints_frame(
    result: ThresholdSimulationResult,
) -> pd.DataFrame:
    """Return workload-only operational constraint evaluation."""

    frozen = (
        result
        .frozen_operational_constraints
    )
    candidate = (
        result
        .candidate_operational_constraints
    )

    return pd.DataFrame(
        [
            {
                "Policy": "Frozen",
                "Max review rate": (
                    frozen.max_review_rate
                ),
                "Max block rate": (
                    frozen.max_block_rate
                ),
                "Max intervention rate": (
                    frozen
                    .max_intervention_rate
                ),
                "Feasible": (
                    "Yes"
                    if frozen.feasible
                    else "No"
                ),
                "Violations": (
                    ", ".join(
                        frozen.violations
                    )
                    or "None"
                ),
            },
            {
                "Policy": "Candidate",
                "Max review rate": (
                    candidate.max_review_rate
                ),
                "Max block rate": (
                    candidate.max_block_rate
                ),
                "Max intervention rate": (
                    candidate
                    .max_intervention_rate
                ),
                "Feasible": (
                    "Yes"
                    if candidate.feasible
                    else "No"
                ),
                "Violations": (
                    ", ".join(
                        candidate.violations
                    )
                    or "None"
                ),
            },
        ]
    )


def _comparison_frame(
    comparison: LabeledPolicyComparison,
) -> pd.DataFrame:
    """Return selected labeled fraud and development-economics metrics."""

    metric_specs = (
        (
            "Review rate",
            "review_rate",
            "rate",
        ),
        (
            "Block rate",
            "block_rate",
            "rate",
        ),
        (
            "Intervention rate",
            "intervention_rate",
            "rate",
        ),
        (
            "Review precision",
            "review_precision",
            "rate",
        ),
        (
            "Block precision",
            "block_precision",
            "rate",
        ),
        (
            "Fraud recall",
            "fraud_recall",
            "rate",
        ),
        (
            "Expected fraud capture rate",
            "expected_fraud_capture_rate",
            "rate",
        ),
        (
            "Fraud amount capture rate",
            "fraud_amount_capture_rate",
            "rate",
        ),
        (
            "Total cost",
            "total_cost",
            "number",
        ),
        (
            "Average cost per transaction",
            "average_cost_per_transaction",
            "number",
        ),
    )

    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    for (
        display_name,
        metric_name,
        metric_type,
    ) in metric_specs:
        frozen_value = float(
            comparison
            .frozen_metrics[
                metric_name
            ]
        )
        candidate_value = float(
            comparison
            .candidate_metrics[
                metric_name
            ]
        )

        if metric_type == "rate":
            frozen_display = (
                _format_rate(
                    frozen_value
                )
            )
            candidate_display = (
                _format_rate(
                    candidate_value
                )
            )
            delta_display = (
                f"{candidate_value - frozen_value:+.1%}"
            )

        else:
            frozen_display = (
                f"{frozen_value:.2f}"
            )
            candidate_display = (
                f"{candidate_value:.2f}"
            )
            delta_display = (
                f"{candidate_value - frozen_value:+.2f}"
            )

        rows.append(
            {
                "Metric": (
                    display_name
                ),
                "Frozen": (
                    frozen_display
                ),
                "Candidate": (
                    candidate_display
                ),
                "Candidate − Frozen": (
                    delta_display
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _render_labeled_comparison(
    result: ThresholdSimulationResult,
) -> None:
    """Render fraud/economic metrics only when labels support them."""

    st.subheader(
        "Labeled Fraud / Economic Comparison"
    )

    comparison = (
        result.labeled_comparison
    )

    if comparison is None:
        st.info(
            result
            .labeled_evaluation_unavailable_reason
            or (
                "Fraud and economic evaluation "
                "is unavailable."
            )
        )
        st.caption(
            "No fraud-performance or economic claims are "
            "inferred from unlabeled prediction events."
        )
        return

    coverage_columns = (
        st.columns(
            3
        )
    )

    coverage_columns[
        0
    ].metric(
        "Labeled events",
        str(
            result
            .labeled_event_count
        ),
    )

    coverage_columns[
        1
    ].metric(
        "Economic evaluation events",
        str(
            comparison
            .evaluated_event_count
        ),
    )

    coverage_columns[
        2
    ].metric(
        "Missing-amount exclusions",
        str(
            result
            .economics_excluded_missing_amount_count
        ),
    )

    st.caption(
        "Metrics below use only labeled events with transaction "
        "amounts. Cost values use the frozen Phase 3 development "
        "assumptions and are not approved production economics."
    )

    st.dataframe(
        _comparison_frame(
            comparison
        ),
        hide_index=True,
        width="stretch",
    )

    feasibility_frame = (
        pd.DataFrame(
            [
                {
                    "Policy": "Frozen",
                    "Full labeled constraints feasible": (
                        "Yes"
                        if comparison
                        .frozen_feasible
                        else "No"
                    ),
                    "Violations": (
                        ", ".join(
                            comparison
                            .frozen_constraint_violations
                        )
                        or "None"
                    ),
                },
                {
                    "Policy": "Candidate",
                    "Full labeled constraints feasible": (
                        "Yes"
                        if comparison
                        .candidate_feasible
                        else "No"
                    ),
                    "Violations": (
                        ", ".join(
                            comparison
                            .candidate_constraint_violations
                        )
                        or "None"
                    ),
                },
            ]
        )
    )

    st.dataframe(
        feasibility_frame,
        hide_index=True,
        width="stretch",
    )


def render_threshold_simulator(
    store: PredictionStore,
    loaded_policy: LoadedPolicy,
) -> None:
    """Render an ephemeral threshold scenario without policy write-back."""

    bundle = (
        loaded_policy.bundle
    )

    st.header(
        "Threshold Simulator"
    )

    st.warning(
        "Simulation only. Candidate thresholds are temporary dashboard "
        "inputs. They are never saved, applied, promoted, or written "
        "back to the frozen production policy."
    )

    st.caption(
        "Candidate decisions are recomputed only from persisted "
        "calibrated probabilities using the existing policy decision "
        "function. The stored production decisions remain unchanged."
    )

    frozen_columns = (
        st.columns(
            2
        )
    )

    frozen_columns[
        0
    ].metric(
        "Frozen review threshold",
        _format_threshold(
            bundle
            .thresholds
            .review_threshold
        ),
    )

    frozen_columns[
        1
    ].metric(
        "Frozen block threshold",
        _format_threshold(
            bundle
            .thresholds
            .block_threshold
        ),
    )

    st.subheader(
        "Temporary Candidate Thresholds"
    )

    candidate_columns = (
        st.columns(
            2
        )
    )

    candidate_review = (
        candidate_columns[
            0
        ].number_input(
            "Candidate review threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(
                bundle
                .thresholds
                .review_threshold
            ),
            step=0.001,
            format="%.9f",
        )
    )

    candidate_block = (
        candidate_columns[
            1
        ].number_input(
            "Candidate block threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(
                bundle
                .thresholds
                .block_threshold
            ),
            step=0.001,
            format="%.9f",
        )
    )

    if (
        candidate_review
        > candidate_block
    ):
        st.error(
            "Candidate review threshold must be less than "
            "or equal to the candidate block threshold."
        )
        return

    candidate_thresholds = (
        PolicyThresholds(
            review_threshold=(
                candidate_review
            ),
            block_threshold=(
                candidate_block
            ),
        )
    )

    result = (
        simulate_threshold_scenario(
            store,
            loaded_policy,
            candidate_thresholds=(
                candidate_thresholds
            ),
        )
    )

    st.subheader(
        "Scenario Overview"
    )

    overview_columns = (
        st.columns(
            5
        )
    )

    overview_columns[
        0
    ].metric(
        "Scenario events",
        str(
            result.total_events
        ),
    )

    overview_columns[
        1
    ].metric(
        "Changed decisions",
        str(
            result
            .changed_decision_count
        ),
    )

    overview_columns[
        2
    ].metric(
        "Changed-decision rate",
        _format_rate(
            result
            .changed_decision_rate
        ),
    )

    overview_columns[
        3
    ].metric(
        "Frozen intervention",
        _format_rate(
            result
            .frozen_workload
            .intervention_rate
        ),
    )

    overview_columns[
        4
    ].metric(
        "Candidate intervention",
        _format_rate(
            result
            .candidate_workload
            .intervention_rate
        ),
    )

    st.dataframe(
        _threshold_frame(
            result
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader(
        "Workload Comparison"
    )

    workload_frame = (
        _workload_frame(
            result
        )
    )

    workload_table = (
        workload_frame.copy()
    )

    for rate_column in (
        "Frozen rate",
        "Candidate rate",
    ):
        workload_table[
            rate_column
        ] = (
            workload_table[
                rate_column
            ]
            .map(
                _format_rate
            )
        )

    st.dataframe(
        workload_table,
        hide_index=True,
        width="stretch",
    )

    st.subheader(
        "Decision Transitions"
    )

    transition_frame = (
        _transition_frame(
            result
        )
    )

    if transition_frame.empty:
        st.caption(
            "No persisted events are available for transition analysis."
        )

    else:
        transition_table = (
            transition_frame.copy()
        )

        transition_table[
            "Event rate"
        ] = (
            transition_table[
                "Event rate"
            ]
            .map(
                _format_rate
            )
        )

        st.dataframe(
            transition_table,
            hide_index=True,
            width="stretch",
        )

    st.subheader(
        "Operational Constraints"
    )

    st.caption(
        "These review, block, and intervention limits are workload "
        "constraints and can be evaluated on all scenario events "
        "without ground-truth labels."
    )

    st.dataframe(
        _constraints_frame(
            result
        ),
        hide_index=True,
        width="stretch",
    )

    _render_labeled_comparison(
        result
    )
