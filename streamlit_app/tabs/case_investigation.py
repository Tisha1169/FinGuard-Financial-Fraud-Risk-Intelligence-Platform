import plotly.express as px
import streamlit as st

from streamlit_app import db, model_cache


def render():
    st.header("Case Investigation — Customer 360")
    st.caption("Full context for one case: transaction, customer history, risk evidence, model explanation, and audit trail.")

    case_id = st.number_input("Case ID", min_value=1, step=1, value=1)
    detail = db.case_detail(int(case_id))
    if detail is None:
        st.error(f"Case {case_id} not found.")
        return

    case, txn, risk = detail["case"], detail["transaction"], detail["risk_score"]

    top = st.columns(5)
    top[0].metric("Status", case["status"])
    top[1].metric("Risk tier", case["risk_tier"])
    top[2].metric("Combined score", f"{risk['combined_score']:.3f}" if risk else "n/a")
    top[3].metric("Financial exposure", f"${case['financial_exposure']:,.2f}")
    top[4].metric("Assigned to", case["assigned_investigator"] or "unassigned")

    st.divider()
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Transaction")
        if txn:
            st.write(
                f"**Amount:** ${txn['amount']:,.2f} · **Channel:** {txn['channel']} · "
                f"**Status:** {txn['status']} · **Time:** {txn['transaction_ts']}"
            )
            st.write(
                f"**Merchant:** {txn['merchant_name']} ({txn['mcc_description']}, "
                f"{txn['merchant_risk_category']}) · **Customer home:** {txn['home_city']}, {txn['home_country']} "
                f"({txn['customer_risk_segment']})"
            )

        st.subheader("Risk score breakdown")
        if risk:
            comp_cols = st.columns(4)
            comp_cols[0].metric("ML", f"{risk['ml_component']:.3f}")
            comp_cols[1].metric("Rules", f"{risk['rules_component']:.3f}")
            comp_cols[2].metric("Behavioral", f"{risk['behavioral_component']:.3f}")
            comp_cols[3].metric("Exposure", f"{risk['exposure_component']:.3f}")

        st.subheader("Rules triggered (reason codes)")
        if detail["rules_triggered"]:
            for r in detail["rules_triggered"]:
                with st.expander(f"{r['rule_id']} — {r['severity']} — {r['rule_description']}"):
                    st.json(r["evidence"])
        else:
            st.caption("No rules fired on this transaction.")

        st.subheader("Model explanation (SHAP)")
        explanation = model_cache.explain_transaction(int(txn["transaction_id"])) if txn else None
        if explanation:
            st.caption(f"XGBoost predicted margin: {explanation['predicted_margin']:.3f} (base {explanation['base_value']:.3f})")
            for c in explanation["top_contributors"]:
                direction = "raised" if c["shap_value"] > 0 else "lowered"
                st.write(f"- **{c['feature']}** = {c['value']:.3g} — {direction} the score by {abs(c['shap_value']):.3f}")
        else:
            st.caption(
                "SHAP explanation unavailable for this transaction (it falls outside the cached model's "
                "test-set sample). The risk score breakdown and rules above still explain the decision."
            )

        st.subheader("Recommended action")
        next_actions = detail["valid_next_actions"]
        if not next_actions:
            st.success("Case is closed - no further action required.")
        elif risk and risk["combined_score"] >= 0.8:
            st.warning(f"High combined score ({risk['combined_score']:.2f}) with {len(detail['rules_triggered'])} rule(s) fired - recommend escalating or confirming fraud if evidence holds.")
        else:
            st.info(f"Valid next actions: {', '.join(next_actions)}. Review the evidence above before deciding.")

    with right:
        st.subheader("Customer transaction timeline (last 30)")
        timeline = detail["customer_timeline"]
        if not timeline.empty:
            fig = px.scatter(
                timeline, x="transaction_ts", y="amount", color="is_fraud",
                size="combined_score", hover_data=["risk_tier", "channel", "status"],
                color_discrete_map={True: "crimson", False: "steelblue"},
            )
            fig.update_layout(height=300, margin=dict(t=10, b=10), showlegend=True)
            st.plotly_chart(fig, width="stretch")

        st.subheader("Previous alerts (this customer)")
        prev = detail["previous_alerts"]
        if prev.empty:
            st.caption("No other alerts for this customer.")
        else:
            st.dataframe(
                prev.rename(columns={
                    "alert_id": "Alert", "created_at": "Created", "risk_tier": "Tier",
                    "combined_score": "Score", "financial_exposure": "Exposure",
                    "status": "Status", "resolution": "Resolution",
                }),
                width="stretch", hide_index=True, height=200,
            )

        st.subheader("Audit history")
        if detail["action_history"]:
            for a in detail["action_history"]:
                st.write(f"`{a['performed_at']}` **{a['action_type']}** by {a['performed_by']}" + (f" — {a['notes']}" if a["notes"] else ""))
        else:
            st.caption("No actions logged yet.")
