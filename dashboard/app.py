"""RiskFlow PayGuard product demonstration and operations dashboard."""

from __future__ import annotations

import streamlit as st

from api.config import settings
from api.logging_db import (
    create_prediction_store,
)
from api.model_loader import (
    get_loaded_policy,
    load_policy,
)
from dashboard.demo_view import (
    render_product_demo,
)
from dashboard.monitoring import (
    load_monitoring_snapshot,
)
from dashboard.monitoring_view import (
    render_monitoring_view,
)
from dashboard.simulation_view import (
    render_threshold_simulator,
)


st.set_page_config(
    page_title=(
        "RiskFlow PayGuard Dashboard"
    ),
    layout="wide",
)

st.title(
    "RiskFlow PayGuard"
)
st.caption(
    "Payment fraud-risk scoring, explanations, monitoring, "
    "and read-only threshold simulation."
)

view = st.sidebar.radio(
    "View",
    [
        "Product Demo",
        "Monitoring",
        "Threshold Simulator",
    ],
)


if view == "Product Demo":
    render_product_demo(
        settings.payguard_api_url
    )

else:
    store = (
        create_prediction_store(
            settings.database_url
        )
    )
    store.init_schema()

    try:
        if view == "Monitoring":
            snapshot = (
                load_monitoring_snapshot(
                    store
                )
            )

            render_monitoring_view(
                snapshot
            )

        else:
            load_policy(
                settings.policy_path
            )

            render_threshold_simulator(
                store,
                get_loaded_policy(),
            )

    finally:
        store.dispose()
