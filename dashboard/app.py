"""RiskFlow PayGuard Dashboard.

Four pages:
    1. Fraud overview        — volume, fraud rate, decisions split, recent flags
    2. Model performance     — ROC, PR, confusion matrix at current threshold
    3. Threshold simulator   — slider over (review, block) thresholds, live
                               fraud-caught / FP cost / review workload chart
    4. Explainability        — global SHAP importance, top-N suspicious txns

Reads from the same predictions DB the RiskFlow PayGuard API writes to.
"""
import streamlit as st

st.set_page_config(page_title="RiskFlow PayGuard Dashboard", layout="wide")

st.title("RiskFlow PayGuard")
st.caption("Real-Time Payment Fraud Risk Scoring — Dashboard scaffold, wire pages up after the model is trained.")

page = st.sidebar.radio(
    "Page",
    ["Fraud overview", "Model performance", "Threshold simulator", "Explainability"],
)

if page == "Fraud overview":
    st.info("TODO: volume, fraud rate, decision split, recent flagged transactions.")
elif page == "Model performance":
    st.info("TODO: ROC curve, PR curve, confusion matrix at current threshold.")
elif page == "Threshold simulator":
    st.info(
        "TODO: sliders for review/block thresholds; live chart of fraud caught, "
        "false-positive cost, and manual-review workload."
    )
elif page == "Explainability":
    st.info("TODO: global SHAP importance + top-N suspicious transactions.")
