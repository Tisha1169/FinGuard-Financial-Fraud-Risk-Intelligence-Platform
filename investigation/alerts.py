"""Generates fraud_alerts from risk_scores: HIGH/CRITICAL tier
transactions become alerts (per docs/architecture.md's risk decision
flow), with alerts for the same customer within a short window collapsed
into one dedup_group_id.
"""
import uuid

import pandas as pd

from investigation import config


def generate_alerts(risk_scores_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    """transactions_df must have transaction_id, customer_id,
    transaction_ts, amount. risk_scores_df must have transaction_id,
    risk_tier, combined_score.
    """
    merged = transactions_df.merge(
        risk_scores_df[["transaction_id", "risk_tier", "combined_score"]], on="transaction_id", how="inner"
    )
    alert_worthy = merged[merged["risk_tier"].isin(config.ALERT_TIERS)].copy()
    alert_worthy = alert_worthy.sort_values(["customer_id", "transaction_ts"]).reset_index(drop=True)

    # Dedup grouping: a new group starts whenever the gap since this
    # customer's previous alert-worthy transaction exceeds the window.
    gap = alert_worthy.groupby("customer_id")["transaction_ts"].diff()
    new_group = gap.isna() | (gap > pd.Timedelta(minutes=config.DEDUP_WINDOW_MINUTES))
    alert_worthy["_group_seq"] = new_group.groupby(alert_worthy["customer_id"]).cumsum()

    group_uids = {}
    def _group_id(customer_id, seq):
        key = (customer_id, seq)
        if key not in group_uids:
            group_uids[key] = f"DEDUP-{uuid.uuid4().hex[:12]}"
        return group_uids[key]

    alert_worthy["dedup_group_id"] = [
        _group_id(c, s) for c, s in zip(alert_worthy["customer_id"], alert_worthy["_group_seq"])
    ]

    alert_worthy["financial_exposure"] = alert_worthy["amount"].round(2)
    alert_worthy["created_at"] = alert_worthy["transaction_ts"]

    result = alert_worthy[[
        "transaction_id", "customer_id", "risk_tier", "combined_score",
        "financial_exposure", "dedup_group_id", "created_at",
    ]].sort_values("created_at").reset_index(drop=True)
    result.insert(0, "alert_id", range(1, len(result) + 1))
    return result
