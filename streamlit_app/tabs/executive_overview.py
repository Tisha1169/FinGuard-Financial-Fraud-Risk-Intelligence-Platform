import plotly.graph_objects as go
import streamlit as st

from streamlit_app import db


def render():
    st.header("Executive Risk Overview")
    st.caption(
        "All figures are computed against FinGuard's synthetic dataset. Financial figures "
        "(exposure, estimated loss prevented) are explicit simulations - see docs/risk_scoring.md."
    )

    kpis = db.executive_kpis()

    row1 = st.columns(4)
    row1[0].metric("Transaction volume", f"{kpis['total_transactions']:,}")
    row1[1].metric("Fraud alerts", f"{kpis['total_alerts']:,}")
    row1[2].metric("Confirmed fraud (cases)", f"{kpis['confirmed_fraud']:,}")
    row1[3].metric("True fraud rate", f"{kpis['true_fraud_rate']:.2%}")

    row2 = st.columns(4)
    row2[0].metric("Financial exposure (alerts)", f"${kpis['total_financial_exposure']:,.0f}")
    row2[1].metric("Est. loss prevented", f"${kpis['estimated_loss_prevented']:,.0f}",
                    help="Simulated: 80% recovery rate on confirmed-fraud alert amounts. See docs/risk_scoring.md.")
    fp_rate = kpis["false_positive_rate"]
    row2[2].metric("False-positive rate", f"{fp_rate:.2%}" if fp_rate is not None else "n/a")
    row2[3].metric("Open investigations", f"{kpis['open_investigations']:,}")

    row3 = st.columns(4)
    sla = kpis["sla_compliance_rate"]
    row3[0].metric("SLA compliance", f"{sla:.2%}" if sla is not None else "n/a")
    row3[1].empty()
    row3[2].empty()
    row3[3].empty()

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Transaction volume & fraud trend")
        daily = db.daily_volume_and_fraud()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["day"], y=daily["txn_count"], name="Transactions", yaxis="y", opacity=0.55))
        fig.add_trace(go.Scatter(x=daily["day"], y=daily["fraud_count"], name="Confirmed fraud (labels)",
                                  yaxis="y2", mode="lines+markers", line=dict(color="crimson")))
        fig.update_layout(
            yaxis=dict(title="Transactions/day"),
            yaxis2=dict(title="Fraud/day", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=10, b=10),
            height=380,
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Risk tier distribution")
        tiers = db.risk_tier_distribution()
        order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        colors = {"LOW": "#4c9a63", "MEDIUM": "#d9a441", "HIGH": "#d9743d", "CRITICAL": "#c03b3b"}
        tiers["risk_tier"] = tiers["risk_tier"].astype("category").cat.set_categories(order, ordered=True)
        tiers = tiers.sort_values("risk_tier")
        fig2 = go.Figure(go.Bar(
            x=tiers["n"], y=tiers["risk_tier"].astype(str), orientation="h",
            marker_color=[colors.get(t, "#888") for t in tiers["risk_tier"].astype(str)],
        ))
        fig2.update_layout(margin=dict(t=10, b=10), height=380, xaxis_title="Transactions")
        st.plotly_chart(fig2, width="stretch")

    st.info(
        "This dashboard reflects the current database state, including any actions taken through "
        "the Fraud Alert & Investigation Queue tab. All fraud labels are synthetic (see docs/data_generation.md) - "
        "no claim of real-world fraud prevention is made anywhere in this project.",
        icon="ℹ️",
    )
