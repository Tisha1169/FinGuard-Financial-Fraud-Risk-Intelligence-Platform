"""Creates one investigation_case per alert (schema grain - see
sql/schema.sql's comment: usually 1:1 with an alert). SLA deadline and
initial status are driven by risk tier: CRITICAL cases are created
directly in ESCALATED status (immediate senior attention, per
docs/architecture.md's "HIGH -> Alert, CRITICAL -> Alert + Escalate"),
HIGH cases start OPEN.
"""
import pandas as pd

from investigation import config


def generate_cases(alerts_df: pd.DataFrame) -> pd.DataFrame:
    cases = alerts_df[["alert_id", "customer_id", "risk_tier", "created_at"]].copy()
    cases["status"] = cases["risk_tier"].map(
        lambda t: "ESCALATED" if t in config.AUTO_ESCALATE_TIERS else "OPEN"
    )
    cases["sla_hours"] = cases["risk_tier"].map(config.SLA_HOURS)
    cases["sla_deadline"] = cases["created_at"] + pd.to_timedelta(cases["sla_hours"], unit="h")
    cases["assigned_investigator"] = None
    cases["resolution"] = None
    cases["resolved_at"] = pd.NaT

    cases = cases.sort_values("alert_id").reset_index(drop=True)
    cases.insert(0, "case_id", range(1, len(cases) + 1))
    # risk_tier is kept for the simulation step (resolution-time modeling)
    # and for by-tier metrics - it's dropped before loading into
    # investigation_cases, which doesn't have that column (risk_tier lives
    # on fraud_alerts; see scripts/run_investigation_workflow.py).
    return cases[[
        "case_id", "alert_id", "customer_id", "status", "assigned_investigator",
        "sla_deadline", "resolution", "created_at", "resolved_at", "risk_tier",
    ]]
