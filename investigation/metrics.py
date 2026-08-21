"""Operational metrics computed from the final investigation_cases state:
SLA compliance, backlog, investigator workload, fraud confirmation rate,
false-positive rate. This is the layer that lets Phase 8 demonstrate
operational thinking, not just detection accuracy.
"""
import pandas as pd


def compute_case_metrics(cases_df: pd.DataFrame, now_ts) -> dict:
    total = len(cases_df)
    open_statuses = {"OPEN", "IN_REVIEW", "ESCALATED"}
    resolved_statuses = {"CLOSED"}

    open_cases = cases_df[cases_df["status"].isin(open_statuses)]
    resolved_cases = cases_df[cases_df["status"].isin(resolved_statuses)]

    # Empty resolved_cases can have an ambiguous/mismatched datetime dtype
    # (no real timestamps to infer tz-awareness from), which breaks
    # comparison against a tz-aware Series - guard explicitly rather than
    # relying on pandas to compare two empty arrays sensibly.
    if len(resolved_cases):
        breached = resolved_cases[resolved_cases["resolved_at"] > resolved_cases["sla_deadline"]]
        resolution_hours = (resolved_cases["resolved_at"] - resolved_cases["created_at"]).dt.total_seconds() / 3600
    else:
        breached = resolved_cases
        resolution_hours = pd.Series([], dtype=float)

    still_open_breached = open_cases[pd.Timestamp(now_ts) > open_cases["sla_deadline"]] if len(open_cases) else open_cases

    confirmed_fraud = resolved_cases[resolved_cases["resolution"] == "CONFIRMED_FRAUD"]
    false_positive = resolved_cases[resolved_cases["resolution"] == "FALSE_POSITIVE"]

    workload = cases_df.groupby("assigned_investigator").size().to_dict()

    return {
        "total_cases": total,
        "open_cases": len(open_cases),
        "resolved_cases": len(resolved_cases),
        "queue_backlog": len(open_cases),
        "sla_compliance_rate": round(1 - (len(breached) / len(resolved_cases)), 4) if len(resolved_cases) else None,
        "resolved_cases_breached_sla": len(breached),
        "open_cases_already_past_sla": len(still_open_breached),
        "avg_resolution_hours": round(float(resolution_hours.mean()), 2) if len(resolved_cases) else None,
        "median_resolution_hours": round(float(resolution_hours.median()), 2) if len(resolved_cases) else None,
        "fraud_confirmation_rate": round(len(confirmed_fraud) / len(resolved_cases), 4) if len(resolved_cases) else None,
        "false_positive_rate": round(len(false_positive) / len(resolved_cases), 4) if len(resolved_cases) else None,
        "investigator_workload": workload,
        "alert_to_case_conversion_rate": 1.0,  # 1:1 by design - see docs/investigation_workflow.md
    }


def compute_metrics_by_tier(cases_df: pd.DataFrame, now_ts) -> dict:
    return {
        tier: compute_case_metrics(group, now_ts)
        for tier, group in cases_df.groupby("risk_tier")
    }
