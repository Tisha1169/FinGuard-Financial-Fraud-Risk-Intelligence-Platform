"""Cached data-access layer for the Streamlit app. Talks to Postgres
directly (via database/connection.py, the same module every script in
this project uses) - no FastAPI dependency, matching the production
deployment where Streamlit Community Cloud connects to Neon directly
(see docs/architecture.md's deployment note).

st.cache_data caches query RESULTS (safe across reruns, invalidated by
TTL); st.cache_resource caches the ENGINE and any fitted models (created
once per app process, never re-fit per rerun - re-fitting XGBoost/
Isolation Forest on every Streamlit interaction would make the app
unusably slow).
"""
import datetime
import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine

from database.connection import get_engine

load_dotenv()  # Streamlit doesn't auto-load .env like the CLI scripts do

CACHE_TTL_SECONDS = 60


@st.cache_resource
def engine() -> Engine:
    return get_engine()


def _read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    return pd.read_sql(text(query), engine(), params=params or {})


def _parse_evidence(df_or_rows):
    """Raw SQLAlchemy/pandas queries don't carry JSONB column metadata, so
    `evidence` comes back as a JSON-encoded string rather than a parsed
    object (same issue fixed in api/dependencies.py::parse_rule_row) -
    re-parse it once here so every caller gets real nested data.
    """
    if isinstance(df_or_rows, pd.DataFrame):
        if "evidence" in df_or_rows.columns and len(df_or_rows):
            df_or_rows = df_or_rows.copy()
            df_or_rows["evidence"] = df_or_rows["evidence"].apply(
                lambda e: json.loads(e) if isinstance(e, str) else e
            )
        return df_or_rows
    return [
        {**dict(r), "evidence": json.loads(r["evidence"]) if isinstance(r.get("evidence"), str) else r.get("evidence")}
        for r in df_or_rows
    ]


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def executive_kpis() -> dict:
    with engine().connect() as conn:
        total_txns = conn.execute(text("SELECT COUNT(*) FROM fact_transactions")).scalar()
        total_amount = conn.execute(text("SELECT COALESCE(SUM(amount),0) FROM fact_transactions")).scalar()
        total_alerts = conn.execute(text("SELECT COUNT(*) FROM fraud_alerts")).scalar()
        confirmed_fraud = conn.execute(
            text("SELECT COUNT(*) FROM investigation_cases WHERE resolution = 'CONFIRMED_FRAUD'")
        ).scalar()
        true_fraud_count = conn.execute(text("SELECT COUNT(*) FROM ground_truth_fraud WHERE is_fraud")).scalar()
        total_exposure = conn.execute(text("SELECT COALESCE(SUM(financial_exposure),0) FROM fraud_alerts")).scalar()
        loss_prevented = conn.execute(
            text(
                """
                SELECT COALESCE(SUM(a.financial_exposure) * 0.8, 0)
                FROM fraud_alerts a
                JOIN investigation_cases c ON c.alert_id = a.alert_id
                WHERE c.resolution = 'CONFIRMED_FRAUD'
                """
            )
        ).scalar()
        false_positives = conn.execute(
            text("SELECT COUNT(*) FROM investigation_cases WHERE resolution = 'FALSE_POSITIVE'")
        ).scalar()
        resolved = conn.execute(
            text("SELECT COUNT(*) FROM investigation_cases WHERE status = 'CLOSED'")
        ).scalar()
        open_cases = conn.execute(
            text("SELECT COUNT(*) FROM investigation_cases WHERE status IN ('OPEN','IN_REVIEW','ESCALATED')")
        ).scalar()
        sla_compliant = conn.execute(
            text("SELECT COUNT(*) FROM investigation_cases WHERE status='CLOSED' AND resolved_at <= sla_deadline")
        ).scalar()

    return {
        "total_transactions": total_txns,
        "total_amount": float(total_amount),
        "total_alerts": total_alerts,
        "confirmed_fraud": confirmed_fraud,
        "true_fraud_rate": (true_fraud_count / total_txns) if total_txns else 0,
        "total_financial_exposure": float(total_exposure),
        "estimated_loss_prevented": float(loss_prevented),
        "false_positive_rate": (false_positives / resolved) if resolved else None,
        "open_investigations": open_cases,
        "sla_compliance_rate": (sla_compliant / resolved) if resolved else None,
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def daily_volume_and_fraud() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT date_trunc('day', t.transaction_ts)::date AS day,
               COUNT(*) AS txn_count,
               COUNT(*) FILTER (WHERE g.is_fraud) AS fraud_count,
               SUM(t.amount) AS total_amount
        FROM fact_transactions t
        LEFT JOIN ground_truth_fraud g ON g.transaction_id = t.transaction_id
        GROUP BY 1 ORDER BY 1
        """
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def risk_tier_distribution() -> pd.DataFrame:
    return _read_sql("SELECT risk_tier, COUNT(*) AS n FROM risk_scores GROUP BY 1")


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def transactions_page(
    limit: int = 50, offset: int = 0, risk_tier: str | None = None,
    channel: str | None = None, min_amount: float | None = None,
    customer_id: int | None = None,
) -> pd.DataFrame:
    conditions, params = [], {"limit": limit, "offset": offset}
    if risk_tier:
        conditions.append("r.risk_tier = :risk_tier")
        params["risk_tier"] = risk_tier
    if channel:
        conditions.append("t.channel = :channel")
        params["channel"] = channel
    if min_amount is not None:
        conditions.append("t.amount >= :min_amount")
        params["min_amount"] = min_amount
    if customer_id is not None:
        conditions.append("t.customer_id = :customer_id")
        params["customer_id"] = customer_id
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    return _read_sql(
        f"""
        SELECT t.transaction_id, t.transaction_ts, t.customer_id, t.merchant_id, t.amount,
               t.channel, t.status, r.risk_tier, r.combined_score, r.ml_component,
               r.rules_component, r.behavioral_component
        FROM fact_transactions t
        LEFT JOIN risk_scores r ON r.transaction_id = t.transaction_id
        {where_clause}
        ORDER BY t.transaction_ts DESC
        LIMIT :limit OFFSET :offset
        """,
        params,
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def transaction_rules(transaction_id: int) -> pd.DataFrame:
    df = _read_sql(
        "SELECT rule_id, rule_description, severity, evidence, triggered_at "
        "FROM rules_triggered WHERE transaction_id = :tid ORDER BY triggered_at",
        {"tid": transaction_id},
    )
    return _parse_evidence(df)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def case_queue(
    status: str | None = None, risk_tier: str | None = None,
    investigator: str | None = None, limit: int = 200,
) -> pd.DataFrame:
    conditions, params = [], {"limit": limit}
    if status:
        conditions.append("c.status = :status")
        params["status"] = status
    if risk_tier:
        conditions.append("a.risk_tier = :risk_tier")
        params["risk_tier"] = risk_tier
    if investigator:
        conditions.append("c.assigned_investigator = :investigator")
        params["investigator"] = investigator
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    return _read_sql(
        f"""
        SELECT c.case_id, c.alert_id, c.customer_id, c.status, c.assigned_investigator,
               c.sla_deadline, c.resolution, c.created_at, c.resolved_at,
               a.risk_tier, a.combined_score, a.financial_exposure, a.transaction_id
        FROM investigation_cases c
        JOIN fraud_alerts a ON a.alert_id = c.alert_id
        {where_clause}
        ORDER BY a.combined_score DESC
        LIMIT :limit
        """,
        params,
    )


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def investigators() -> list[str]:
    df = _read_sql("SELECT DISTINCT assigned_investigator FROM investigation_cases WHERE assigned_investigator IS NOT NULL ORDER BY 1")
    return df["assigned_investigator"].tolist()


def case_detail(case_id: int) -> dict | None:
    """Not cached - always fresh, since this is read right after a
    write (assign/action) and must reflect the just-applied change.
    """
    with engine().connect() as conn:
        case_row = conn.execute(
            text(
                """
                SELECT c.case_id, c.alert_id, c.customer_id, c.status, c.assigned_investigator,
                       c.sla_deadline, c.resolution, c.created_at, c.resolved_at,
                       a.risk_tier, a.financial_exposure, a.transaction_id, a.dedup_group_id
                FROM investigation_cases c
                JOIN fraud_alerts a ON a.alert_id = c.alert_id
                WHERE c.case_id = :cid
                """
            ),
            {"cid": case_id},
        ).mappings().first()
        if case_row is None:
            return None
        transaction_id = case_row["transaction_id"]
        customer_id = case_row["customer_id"]

        txn_row = conn.execute(
            text(
                """
                SELECT t.*, cu.risk_segment AS customer_risk_segment, cu.home_city, cu.home_country,
                       m.merchant_name, m.mcc_description, m.risk_category AS merchant_risk_category
                FROM fact_transactions t
                JOIN dim_customer cu ON cu.customer_id = t.customer_id
                JOIN dim_merchant m ON m.merchant_id = t.merchant_id
                WHERE t.transaction_id = :tid
                """
            ),
            {"tid": transaction_id},
        ).mappings().first()

        risk_row = conn.execute(
            text(
                "SELECT ml_component, rules_component, behavioral_component, exposure_component, "
                "combined_score, risk_tier FROM risk_scores WHERE transaction_id = :tid"
            ),
            {"tid": transaction_id},
        ).mappings().first()

        rules_rows = conn.execute(
            text(
                "SELECT rule_id, rule_description, severity, evidence, triggered_at "
                "FROM rules_triggered WHERE transaction_id = :tid ORDER BY triggered_at"
            ),
            {"tid": transaction_id},
        ).mappings().all()

        action_rows = conn.execute(
            text(
                "SELECT action_id, action_type, performed_by, notes, performed_at "
                "FROM investigation_actions WHERE case_id = :cid ORDER BY performed_at"
            ),
            {"cid": case_id},
        ).mappings().all()

        customer_timeline = pd.read_sql(
            text(
                """
                SELECT t.transaction_id, t.transaction_ts, t.amount, t.channel, t.status,
                       r.risk_tier, r.combined_score, g.is_fraud
                FROM fact_transactions t
                LEFT JOIN risk_scores r ON r.transaction_id = t.transaction_id
                LEFT JOIN ground_truth_fraud g ON g.transaction_id = t.transaction_id
                WHERE t.customer_id = :cid
                ORDER BY t.transaction_ts DESC LIMIT 30
                """
            ),
            conn, params={"cid": customer_id},
        )

        previous_alerts = pd.read_sql(
            text(
                """
                SELECT a.alert_id, a.created_at, a.risk_tier, a.combined_score, a.financial_exposure,
                       c.status, c.resolution
                FROM fraud_alerts a
                JOIN investigation_cases c ON c.alert_id = a.alert_id
                WHERE a.customer_id = :cid AND c.case_id != :cid_case
                ORDER BY a.created_at DESC LIMIT 10
                """
            ),
            conn, params={"cid": customer_id, "cid_case": case_id},
        )

    from investigation.state_machine import valid_actions_from

    return {
        "case": dict(case_row),
        "transaction": dict(txn_row) if txn_row else None,
        "risk_score": dict(risk_row) if risk_row else None,
        "rules_triggered": _parse_evidence(rules_rows),
        "action_history": [dict(a) for a in action_rows],
        "valid_next_actions": valid_actions_from(case_row["status"]),
        "customer_timeline": customer_timeline,
        "previous_alerts": previous_alerts,
    }


def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def clear_case_caches() -> None:
    case_queue.clear()
    executive_kpis.clear()
