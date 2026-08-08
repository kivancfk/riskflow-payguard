"""RiskFlow PayGuard Phase 6 monitoring and simulation dashboard."""

from __future__ import annotations

import streamlit as st

from api.config import settings
from api.logging_db import (
    create_prediction_store,
)
from dashboard.monitoring import (
    load_monitoring_snapshot,
)
from dashboard.monitoring_view import (
    render_monitoring_view,
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
        st.header(
            "Threshold Simulator"
        )
        st.info(
            "The read-only threshold simulator UI will be added "
            "in the next Phase 6 commit. Candidate thresholds will "
            "remain temporary and will never be applied or saved "
            "to the frozen production policy."
        )

finally:
    store.dispose()
