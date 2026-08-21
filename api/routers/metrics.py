import datetime

import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.dependencies import db_engine
from investigation.metrics import compute_case_metrics

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def get_metrics(engine: Engine = Depends(db_engine)):
    """Portfolio snapshot: detection volume, tier distribution, and live
    operational metrics computed from the CURRENT database state (not a
    cached batch run) - so actions taken through POST /cases/{id}/action
    are reflected immediately, unlike scripts/run_investigation_workflow.py's
    own printed summary, which is a point-in-time snapshot at simulation time.
    """
    with engine.connect() as conn:
        total_transactions = conn.execute(text("SELECT COUNT(*) FROM fact_transactions")).scalar()
        total_alerts = conn.execute(text("SELECT COUNT(*) FROM fraud_alerts")).scalar()
        tier_dist = dict(conn.execute(text("SELECT risk_tier, COUNT(*) FROM risk_scores GROUP BY 1")).fetchall())

        cases_df = pd.read_sql(
            """
            SELECT status, resolution, created_at, resolved_at, sla_deadline, assigned_investigator
            FROM investigation_cases
            """,
            conn,
        )

    if cases_df.empty:
        return {
            "total_transactions": total_transactions, "total_alerts": total_alerts,
            "total_cases": 0, "open_cases": 0, "resolved_cases": 0,
            "sla_compliance_rate": None, "fraud_confirmation_rate": None,
            "false_positive_rate": None, "risk_tier_distribution": tier_dist,
        }

    for col in ["created_at", "resolved_at", "sla_deadline"]:
        cases_df[col] = pd.to_datetime(cases_df[col], utc=True)

    case_metrics = compute_case_metrics(cases_df, now_ts=datetime.datetime.now(datetime.timezone.utc))

    return {
        "total_transactions": total_transactions,
        "total_alerts": total_alerts,
        "total_cases": case_metrics["total_cases"],
        "open_cases": case_metrics["open_cases"],
        "resolved_cases": case_metrics["resolved_cases"],
        "sla_compliance_rate": case_metrics["sla_compliance_rate"],
        "fraud_confirmation_rate": case_metrics["fraud_confirmation_rate"],
        "false_positive_rate": case_metrics["false_positive_rate"],
        "risk_tier_distribution": tier_dist,
    }
