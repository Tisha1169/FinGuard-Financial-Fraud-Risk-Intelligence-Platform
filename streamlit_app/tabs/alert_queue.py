import streamlit as st

from investigation.case_actions import CaseNotFoundError, assign_case, perform_action
from investigation.config import INVESTIGATORS
from investigation.state_machine import InvalidTransitionError, valid_actions_from
from streamlit_app import db


def render():
    st.header("Fraud Alert & Investigation Queue")
    st.caption(
        "Cases from HIGH/CRITICAL risk-tier alerts, prioritized by combined score. "
        "Actions here write directly to the database (no separate API call in production - "
        "see docs/architecture.md's deployment note) and are validated against the case "
        "state machine (investigation/state_machine.py)."
    )

    col1, col2, col3 = st.columns(3)
    status = col1.selectbox("Status", ["Any", "OPEN", "IN_REVIEW", "ESCALATED", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"])
    risk_tier = col2.selectbox("Risk tier", ["Any", "HIGH", "CRITICAL"])
    investigator_options = ["Any"] + db.investigators()
    investigator = col3.selectbox("Assigned investigator", investigator_options)

    df = db.case_queue(
        status=None if status == "Any" else status,
        risk_tier=None if risk_tier == "Any" else risk_tier,
        investigator=None if investigator == "Any" else investigator,
        limit=200,
    )

    if df.empty:
        st.warning("No cases match these filters.")
        return

    now = db.now_utc()
    df["sla_breached"] = (df["status"].isin(["OPEN", "IN_REVIEW", "ESCALATED"]) & (df["sla_deadline"] < now)) | (
        df["status"] == "CLOSED"
    ) & (df["resolved_at"] > df["sla_deadline"])

    st.dataframe(
        df[["case_id", "risk_tier", "combined_score", "status", "assigned_investigator",
            "financial_exposure", "sla_deadline", "sla_breached", "resolution", "created_at"]]
        .rename(columns={
            "case_id": "Case", "risk_tier": "Tier", "combined_score": "Score", "status": "Status",
            "assigned_investigator": "Investigator", "financial_exposure": "Exposure",
            "sla_deadline": "SLA Deadline", "sla_breached": "SLA Breached", "resolution": "Resolution",
            "created_at": "Created",
        }),
        column_config={
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Exposure": st.column_config.NumberColumn(format="$%.2f"),
        },
        width="stretch", hide_index=True, height=350,
    )

    st.divider()
    st.subheader("Take action on a case")

    # Persist the selected case across reruns (st.rerun() after an action
    # would otherwise snap this back to the queue's top row every time,
    # losing the case the analyst was actually working on).
    if "alert_queue_case_id" not in st.session_state:
        st.session_state["alert_queue_case_id"] = int(df.iloc[0]["case_id"])
    case_id = st.number_input("Case ID", min_value=1, step=1, key="alert_queue_case_id")
    matching = df[df["case_id"] == case_id]
    if matching.empty:
        st.caption("Enter a case ID from the table above.")
        return

    case_row = matching.iloc[0]
    st.write(
        f"**Status:** {case_row['status']} · **Tier:** {case_row['risk_tier']} · "
        f"**Investigator:** {case_row['assigned_investigator'] or 'unassigned'} · "
        f"**Exposure:** ${case_row['financial_exposure']:,.2f}"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        with st.form("assign_form"):
            new_investigator = st.selectbox("Assign to", INVESTIGATORS)
            performed_by = st.text_input("Performed by", value="analyst")
            if st.form_submit_button("Assign"):
                try:
                    assign_case(db.engine(), int(case_id), new_investigator, performed_by)
                    db.clear_case_caches()
                    st.success(f"Case {case_id} assigned to {new_investigator}.")
                    st.rerun()
                except CaseNotFoundError as exc:
                    st.error(str(exc))

    with col_b:
        available_actions = valid_actions_from(case_row["status"])
        with st.form("action_form"):
            if available_actions:
                action_type = st.selectbox("Action", available_actions)
                notes = st.text_area("Notes", placeholder="Optional investigator notes")
                performed_by_action = st.text_input("Performed by ", value="analyst", key="action_performed_by")
                submitted = st.form_submit_button("Apply action")
            else:
                st.selectbox("Action", ["(no valid actions - case is CLOSED)"], disabled=True)
                submitted = False

            if submitted:
                try:
                    result = perform_action(db.engine(), int(case_id), action_type, performed_by_action, notes or None)
                    db.clear_case_caches()
                    st.success(f"Case {case_id}: {result['previous_status']} → {result['new_status']}")
                    st.rerun()
                except CaseNotFoundError as exc:
                    st.error(str(exc))
                except InvalidTransitionError as exc:
                    st.error(str(exc))

    st.caption(f"Open the **Case Investigation** tab and enter case ID {case_id} for the full workbench view.")
