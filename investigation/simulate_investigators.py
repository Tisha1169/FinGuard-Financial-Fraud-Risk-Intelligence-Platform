"""Simulates investigator activity against the case queue: assignment,
investigation, escalation, and final resolution (confirmed fraud vs.
false positive), each logged as an investigation_actions row plus a
mirrored audit_log entry.

This is a SIMULATION, not real investigator behavior - two explicit,
documented assumptions drive it (see investigation/config.py):
1. INVESTIGATOR_ACCURACY: simulated investigators reach the ground-truth-
   correct conclusion 90% of the time, not 100% - modeling perfect
   investigators would make the false-positive/false-negative operational
   metrics meaningless.
2. Backlog emerges from two effects: cases whose simulated resolution
   time would land after the simulation's "now" instant naturally
   haven't been resolved yet, plus an additional random share of cases
   that would have finished by "now" but are modeled as still stuck in
   the queue (real backlogs have both causes, not just recency).
"""
import random

import numpy as np
import pandas as pd

from investigation import config


def _resolution_hours(rng: np.random.Generator, tier: str) -> float:
    params = config.RESOLUTION_TIME_HOURS_PARAMS[tier]
    return float(rng.lognormal(params["mu"], params["sigma"]))


def simulate(cases_df: pd.DataFrame, alerts_df: pd.DataFrame, ground_truth_df: pd.DataFrame, now_ts) -> tuple:
    """Returns (updated_cases_df, investigation_actions_df, audit_log_df).

    ground_truth_df must have transaction_id, is_fraud.
    alerts_df must have alert_id, transaction_id (to join case -> transaction -> label).
    """
    rng = np.random.default_rng(config.SEED)
    py_rng = random.Random(config.SEED)

    case_to_txn = cases_df[["case_id", "alert_id"]].merge(
        alerts_df[["alert_id", "transaction_id"]], on="alert_id"
    ).merge(ground_truth_df[["transaction_id", "is_fraud"]], on="transaction_id")
    is_fraud_by_case = dict(zip(case_to_txn["case_id"], case_to_txn["is_fraud"]))

    cases = cases_df.copy()
    actions = []
    audit = []

    def log_action(case_id, action_type, performed_by, notes, ts):
        actions.append({
            "case_id": case_id, "action_type": action_type, "performed_by": performed_by,
            "notes": notes, "performed_at": ts,
        })
        audit.append({
            "entity_type": "case", "entity_id": str(case_id), "event_type": action_type,
            "event_payload": {"notes": notes}, "performed_by": performed_by, "performed_at": ts,
        })

    final_status, final_resolution, final_resolved_at = [], [], []
    final_investigator = []

    for row in cases.itertuples():
        investigator = py_rng.choice(config.INVESTIGATORS)
        assign_ts = row.created_at + pd.Timedelta(minutes=int(rng.integers(1, 30)))
        log_action(row.case_id, "ASSIGN", "system", f"auto-assigned to {investigator}", assign_ts)

        investigate_ts = assign_ts + pd.Timedelta(minutes=int(rng.integers(5, 120)))
        log_action(row.case_id, "INVESTIGATE", investigator, "review started", investigate_ts)

        current_status = row.status
        if row.status == "ESCALATED":
            escalate_ts = investigate_ts + pd.Timedelta(minutes=int(rng.integers(1, 15)))
            log_action(row.case_id, "ESCALATE", "system", "CRITICAL tier - auto-escalated on creation", escalate_ts)

        resolution_hours = _resolution_hours(rng, row.risk_tier)
        intended_resolved_at = row.created_at + pd.Timedelta(hours=resolution_hours)

        would_be_resolved_by_now = intended_resolved_at <= now_ts
        stuck_in_backlog = py_rng.random() < config.BACKLOG_SHARE

        if would_be_resolved_by_now and not stuck_in_backlog:
            is_fraud = bool(is_fraud_by_case[row.case_id])
            investigator_correct = py_rng.random() < config.INVESTIGATOR_ACCURACY
            investigator_says_fraud = is_fraud if investigator_correct else not is_fraud

            outcome = "CONFIRMED_FRAUD" if investigator_says_fraud else "FALSE_POSITIVE"
            action_type = "CONFIRM_FRAUD" if investigator_says_fraud else "MARK_FALSE_POSITIVE"
            log_action(row.case_id, action_type, investigator, f"resolution: {outcome}", intended_resolved_at)

            close_ts = intended_resolved_at + pd.Timedelta(minutes=int(rng.integers(1, 10)))
            log_action(row.case_id, "CLOSE", investigator, "case closed", close_ts)

            final_status.append("CLOSED")
            final_resolution.append(outcome)
            final_resolved_at.append(close_ts)
        else:
            # still in the queue as of `now` - keep its current in-progress
            # status (ESCALATED stays ESCALATED; everything else is IN_REVIEW
            # since investigation has started).
            final_status.append("IN_REVIEW" if current_status != "ESCALATED" else "ESCALATED")
            final_resolution.append(None)
            final_resolved_at.append(pd.NaT)

        final_investigator.append(investigator)

    cases["status"] = final_status
    cases["resolution"] = final_resolution
    cases["resolved_at"] = final_resolved_at
    cases["assigned_investigator"] = final_investigator

    return cases, pd.DataFrame(actions), pd.DataFrame(audit)
