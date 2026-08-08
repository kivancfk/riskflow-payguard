"""RiskFlow PayGuard Phase 6 monitoring and simulation dashboard."""

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
    "Monitoring and read-only threshold simulation "
    "around the frozen fraud-scoring policy."
)

view = st.sidebar.radio(
    "View",
    [
        "Monitoring",
        "Threshold Simulator",
    ],
)

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
