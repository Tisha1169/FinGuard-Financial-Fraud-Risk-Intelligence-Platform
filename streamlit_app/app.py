"""FinGuard analyst console - the Phase 10 Streamlit dashboard.

Connects directly to Postgres (Neon in production, local Docker in dev) -
no FastAPI dependency, matching docs/architecture.md's deployment design.

Run locally:
    streamlit run streamlit_app/app.py
"""
import pathlib
import sys

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from streamlit_app.tabs import (  # noqa: E402
    alert_queue,
    case_investigation,
    executive_overview,
    model_monitoring,
    transaction_intelligence,
)

st.set_page_config(page_title="FinGuard — Fraud Risk Intelligence", layout="wide", page_icon="🛡️")

st.title("🛡️ FinGuard — Financial Fraud & Risk Intelligence Platform")
st.caption(
    "All data is synthetic (see docs/data_generation.md). No claim of real-world fraud prevention "
    "or financial impact is made anywhere in this project."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Overview", "Transaction Intelligence", "Alert & Investigation Queue",
    "Case Investigation", "Model & Portfolio Monitoring",
])

with tab1:
    executive_overview.render()
with tab2:
    transaction_intelligence.render()
with tab3:
    alert_queue.render()
with tab4:
    case_investigation.render()
with tab5:
    model_monitoring.render()
