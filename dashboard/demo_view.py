"""Streamlit rendering for the API-backed PayGuard product demonstration."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api.config import (
    FROZEN_BLOCK_THRESHOLD,
    FROZEN_REVIEW_THRESHOLD,
)
from api.schemas import (
    FeatureContributionResponse,
    PredictionResponse,
)
from dashboard.demo import (
    DemoAPIError,
    DemoDataError,
    load_demo_scenarios,
    submit_demo_prediction,
)


def _format_probability(
    value: float,
) -> str:
    """Format a probability for product-demo display."""

    return f"{value:.1%}"


def _contribution_frame(
    contributions: list[
        FeatureContributionResponse
    ],
) -> pd.DataFrame:
    """Return analyst-friendly TreeSHAP contribution rows."""

    return pd.DataFrame(
        [
            {
                "Rank": contribution.rank,
                "Feature": contribution.feature,
                "Group": contribution.feature_group,
                "Value state": contribution.value_state,
                "SHAP contribution": (
                    contribution.shap_value_raw
                ),
            }
            for contribution
            in contributions
        ]
    )


def _render_prediction(
    prediction: PredictionResponse,
) -> None:
    """Render one validated frozen-policy prediction."""

    st.success(
        "Scored through FastAPI. The successful prediction "
        "was persisted by the API and is now available to Monitoring."
    )

    st.subheader(
        "Decision"
    )

    metric_columns = st.columns(
        4
    )

    metric_columns[
        0
    ].metric(
        "Decision",
        prediction.decision,
    )

    metric_columns[
        1
    ].metric(
        "Calibrated risk probability",
        _format_probability(
            prediction.calibrated_probability
        ),
    )

    metric_columns[
        2
    ].metric(
        "Frozen REVIEW threshold",
        _format_probability(
            FROZEN_REVIEW_THRESHOLD
        ),
    )

    metric_columns[
        3
    ].metric(
        "Frozen BLOCK threshold",
        _format_probability(
            FROZEN_BLOCK_THRESHOLD
        ),
    )

    st.caption(
        "The displayed decision comes directly from the frozen "
        "PayGuard policy returned by the FastAPI service."
    )

    st.subheader(
        "Why the model scored this way"
    )

    st.caption(
        "TreeSHAP contributions and analyst reason codes describe "
        "model signals. They are not causal evidence of fraud. "
        "SHAP values decompose the raw model margin rather than "
        "the calibrated probability."
    )

    contribution_columns = (
        st.columns(
            2
        )
    )

    with contribution_columns[
        0
    ]:
        st.markdown(
            "**Signals increasing model score**"
        )

        positive_frame = (
            _contribution_frame(
                prediction
                .top_positive_contributions
            )
        )

        if positive_frame.empty:
            st.caption(
                "No positive contributions were returned."
            )
        else:
            st.dataframe(
                positive_frame,
                hide_index=True,
                width="stretch",
            )

    with contribution_columns[
        1
    ]:
        st.markdown(
            "**Signals decreasing model score**"
        )

        negative_frame = (
            _contribution_frame(
                prediction
                .top_negative_contributions
            )
        )

        if negative_frame.empty:
            st.caption(
                "No negative contributions were returned."
            )
        else:
            st.dataframe(
                negative_frame,
                hide_index=True,
                width="stretch",
            )

    st.subheader(
        "Analyst reason codes"
    )

    if prediction.reasons:
        reason_frame = pd.DataFrame(
            [
                {
                    "Code": reason.code,
                    "Analyst message": (
                        reason.message
                    ),
                }
                for reason
                in prediction.reasons
            ]
        )

        st.dataframe(
            reason_frame,
            hide_index=True,
            width="stretch",
        )

    else:
        st.caption(
            "No analyst reason codes were returned."
        )

    with st.expander(
        "Prediction provenance and diagnostics"
    ):
        st.json(
            {
                "transaction_id": (
                    prediction.transaction_id
                ),
                "model_version": (
                    prediction.model_version
                ),
                "policy_version": (
                    prediction.policy_version
                ),
                "explanation_version": (
                    prediction
                    .explanation_version
                ),
                "reason_code_version": (
                    prediction
                    .reason_code_version
                ),
                "raw_model_score": (
                    prediction.raw_model_score
                ),
                "calibrated_probability": (
                    prediction
                    .calibrated_probability
                ),
                "decision": (
                    prediction.decision
                ),
                "reconstruction": (
                    prediction
                    .reconstruction
                    .model_dump(
                        mode="json"
                    )
                ),
            }
        )

    st.info(
        "Next step: open Monitoring from the sidebar to see "
        "the persisted decision in the operational dashboard."
    )


def render_product_demo(
    api_base_url: str,
) -> None:
    """Render a guided synthetic transaction through the live API."""

    st.header(
        "Product Demo"
    )

    st.caption(
        "Score a committed synthetic payment transaction through "
        "the real FastAPI service, frozen fraud policy, explanation "
        "layer, and prediction-persistence workflow."
    )

    st.warning(
        "Synthetic demonstration only. These transactions do not "
        "contain ground-truth fraud labels and must not be interpreted "
        "as confirmed legitimate or fraudulent payments."
    )

    try:
        scenarios = (
            load_demo_scenarios()
        )
    except DemoDataError as error:
        st.error(
            str(error)
        )
        return

    scenario_labels = [
        scenario.label
        for scenario
        in scenarios
    ]

    selected_label = (
        st.selectbox(
            "Synthetic transaction",
            scenario_labels,
        )
    )

    scenario = next(
        scenario
        for scenario
        in scenarios
        if (
            scenario.label
            == selected_label
        )
    )

    transaction = (
        scenario.transaction
    )

    transaction_payload = (
        transaction.model_dump(
            mode="json"
        )
    )

    amount = (
        transaction_payload[
            "features"
        ][
            "TransactionAmt"
        ]
    )

    overview_columns = (
        st.columns(
            3
        )
    )

    overview_columns[
        0
    ].metric(
        "Transaction ID",
        str(
            transaction
            .transaction_id
        ),
    )

    overview_columns[
        1
    ].metric(
        "Transaction amount",
        (
            "—"
            if amount is None
            else f"{float(amount):,.2f}"
        ),
    )

    overview_columns[
        2
    ].metric(
        "Feature count",
        str(
            len(
                transaction_payload[
                    "features"
                ]
            )
        ),
    )

    st.caption(
        scenario.description
    )

    normalized_api_url = (
        api_base_url
        .strip()
        .rstrip("/")
    )

    st.caption(
        "FastAPI target: "
        f"`{normalized_api_url}`"
    )

    with st.expander(
        "View strict API request"
    ):
        st.json(
            transaction_payload
        )

    st.caption(
        "No request is sent until Score transaction is selected. "
        "The dashboard does not load or score the model locally."
    )

    if not st.button(
        "Score transaction",
        type="primary",
    ):
        return

    try:
        prediction = (
            submit_demo_prediction(
                transaction,
                api_base_url,
            )
        )
    except DemoAPIError as error:
        st.error(
            str(error)
        )
        return

    _render_prediction(
        prediction
    )
