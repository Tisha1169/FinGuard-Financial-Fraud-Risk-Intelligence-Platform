"""Financial exposure and cost-based threshold trade-off simulation.

Every dollar figure here is explicitly simulated/estimated from the
constants in risk_scoring/config.py - never a claim of real financial
impact. This is a demonstration of HOW a fraud operation would reason
about the trade-off, not a real recovery/loss model.
"""
import numpy as np
import pandas as pd

from risk_scoring import config


def portfolio_impact_at_threshold(df: pd.DataFrame, threshold: float, score_col: str = "combined_score") -> dict:
    """df must have `score_col`, `amount`, and `is_fraud`. Returns the
    simulated financial outcome of alerting everything at or above
    `threshold`.
    """
    alerted = df[score_col] >= threshold
    is_fraud = df["is_fraud"].astype(bool)

    tp = alerted & is_fraud
    fp = alerted & ~is_fraud
    fn = ~alerted & is_fraud

    n_alerts = int(alerted.sum())
    total_investigation_cost = n_alerts * config.INVESTIGATION_COST_USD
    estimated_loss_prevented = float(df.loc[tp, "amount"].sum()) * config.FRAUD_RECOVERY_RATE_IF_CAUGHT
    unprevented_loss = float(df.loc[fn, "amount"].sum())
    net_expected_impact = estimated_loss_prevented - total_investigation_cost - unprevented_loss

    return {
        "threshold": float(threshold),
        "n_alerts": n_alerts,
        "alert_rate": float(alerted.mean()),
        "true_positives": int(tp.sum()),
        "false_positives": int(fp.sum()),
        "false_negatives": int(fn.sum()),
        "precision": float(tp.sum() / n_alerts) if n_alerts else 0.0,
        "recall": float(tp.sum() / is_fraud.sum()) if is_fraud.sum() else 0.0,
        "total_investigation_cost_usd": round(total_investigation_cost, 2),
        "estimated_loss_prevented_usd": round(estimated_loss_prevented, 2),
        "unprevented_loss_usd": round(unprevented_loss, 2),
        "net_expected_impact_usd": round(net_expected_impact, 2),
    }


def threshold_sweep(df: pd.DataFrame, score_col: str = "combined_score", n_points: int = 20) -> pd.DataFrame:
    """Sweeps threshold across the score's own percentile range so the
    sweep is meaningful regardless of the score's scale/distribution.
    """
    percentiles = np.linspace(1, 99, n_points)
    thresholds = np.percentile(df[score_col], percentiles)
    rows = [portfolio_impact_at_threshold(df, t, score_col) for t in sorted(set(thresholds), reverse=True)]
    return pd.DataFrame(rows)


def best_threshold_by_net_impact(sweep_df: pd.DataFrame) -> dict:
    return sweep_df.loc[sweep_df["net_expected_impact_usd"].idxmax()].to_dict()
