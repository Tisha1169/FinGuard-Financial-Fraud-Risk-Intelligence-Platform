"""Phase 8: generates fraud_alerts and investigation_cases from
risk_scores, simulates investigator activity (assignment, investigation,
escalation, resolution), and loads investigation_actions + audit_log.
Reports SLA/operational metrics.

Requires risk_scores to already be populated (python scripts/run_risk_scoring.py).

Usage:
    python scripts/run_investigation_workflow.py
"""
import json
import pathlib
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from database.connection import get_engine  # noqa: E402
from investigation.alerts import generate_alerts  # noqa: E402
from investigation.cases import generate_cases  # noqa: E402
from investigation.metrics import compute_case_metrics, compute_metrics_by_tier  # noqa: E402
from investigation.simulate_investigators import simulate  # noqa: E402

load_dotenv()

ARTIFACTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "artifacts"

TRUNCATE_ORDER = ["audit_log", "investigation_actions", "investigation_cases", "fraud_alerts"]


def main() -> None:
    engine = get_engine()

    print("=" * 70)
    print("1. Loading risk_scores + transactions + ground truth")
    print("=" * 70)
    risk_scores_df = pd.read_sql("SELECT transaction_id, risk_tier, combined_score FROM risk_scores", engine)
    if risk_scores_df.empty:
        raise RuntimeError("risk_scores is empty - run scripts/run_risk_scoring.py first")

    transactions_df = pd.read_sql(
        "SELECT transaction_id, customer_id, transaction_ts, amount FROM fact_transactions", engine
    )
    transactions_df["transaction_ts"] = pd.to_datetime(transactions_df["transaction_ts"], utc=True)
    transactions_df["amount"] = transactions_df["amount"].astype(float)
    ground_truth_df = pd.read_sql("SELECT transaction_id, is_fraud FROM ground_truth_fraud", engine)
    now_ts = transactions_df["transaction_ts"].max()
    print(f"transactions: {len(transactions_df)}, risk_scores: {len(risk_scores_df)}, 'now': {now_ts}")

    print("\n" + "=" * 70)
    print("2. Generating alerts (HIGH/CRITICAL tiers)")
    print("=" * 70)
    alerts_df = generate_alerts(risk_scores_df, transactions_df)
    print(f"alerts generated: {len(alerts_df)}")
    print(f"distinct dedup groups: {alerts_df['dedup_group_id'].nunique()} "
          f"(dedup collapsed {len(alerts_df) - alerts_df['dedup_group_id'].nunique()} alerts)")
    print(alerts_df["risk_tier"].value_counts().to_string())

    print("\n" + "=" * 70)
    print("3. Generating cases (1:1 with alerts)")
    print("=" * 70)
    cases_df = generate_cases(alerts_df)
    print(f"cases generated: {len(cases_df)}")
    print(cases_df["status"].value_counts().to_string())

    print("\n" + "=" * 70)
    print("4. Simulating investigator activity")
    print("=" * 70)
    final_cases_df, actions_df, audit_df = simulate(cases_df, alerts_df, ground_truth_df, now_ts)
    print(f"actions logged: {len(actions_df)}")
    print(f"audit entries: {len(audit_df)}")
    print("final case status distribution:")
    print(final_cases_df["status"].value_counts().to_string())

    print("\n" + "=" * 70)
    print("5. Loading into database")
    print("=" * 70)
    with engine.begin() as conn:
        for table in TRUNCATE_ORDER:
            conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
    print(f"truncated: {', '.join(TRUNCATE_ORDER)}")

    alerts_df.to_sql("fraud_alerts", engine, if_exists="append", index=False, method="multi", chunksize=5000)
    print(f"loaded {len(alerts_df)} rows into fraud_alerts")

    cases_for_db = final_cases_df.drop(columns=["risk_tier", "sla_hours"], errors="ignore")
    cases_for_db.to_sql("investigation_cases", engine, if_exists="append", index=False, method="multi", chunksize=5000)
    print(f"loaded {len(cases_for_db)} rows into investigation_cases")

    actions_df.to_sql("investigation_actions", engine, if_exists="append", index=False, method="multi", chunksize=5000)
    print(f"loaded {len(actions_df)} rows into investigation_actions")

    audit_df_for_db = audit_df.copy()
    audit_df_for_db["event_payload"] = audit_df_for_db["event_payload"].apply(json.dumps)
    audit_df_for_db.to_sql(
        "audit_log", engine, if_exists="append", index=False, method="multi", chunksize=5000,
        dtype={"event_payload": JSONB},
    )
    print(f"loaded {len(audit_df_for_db)} rows into audit_log")

    print("\n" + "=" * 70)
    print("6. Operational metrics (overall)")
    print("=" * 70)
    overall_metrics = compute_case_metrics(final_cases_df, now_ts)
    print(json.dumps(overall_metrics, indent=2, default=str))

    print("\n" + "=" * 70)
    print("7. Operational metrics by tier")
    print("=" * 70)
    tier_metrics = compute_metrics_by_tier(final_cases_df, now_ts)
    print(json.dumps(tier_metrics, indent=2, default=str))

    print("\n" + "=" * 70)
    print("8. Fraud confirmation rate vs. true fraud rate among alerts (sanity check)")
    print("=" * 70)
    alerted_fraud_rate = alerts_df.merge(ground_truth_df, on="transaction_id")["is_fraud"].mean()
    print(f"true fraud rate among alerted transactions: {alerted_fraud_rate:.2%}")
    print(f"simulated fraud confirmation rate (resolved cases): {overall_metrics['fraud_confirmation_rate']:.2%}")
    print("(these should be close, given INVESTIGATOR_ACCURACY=90% - a big gap would signal a simulation bug)")

    print("\n" + "=" * 70)
    print("9. Saving artifacts")
    print("=" * 70)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    output = {
        "now_ts": str(now_ts),
        "n_alerts": len(alerts_df),
        "n_dedup_groups": int(alerts_df["dedup_group_id"].nunique()),
        "n_cases": len(final_cases_df),
        "overall_metrics": overall_metrics,
        "tier_metrics": tier_metrics,
        "alerted_true_fraud_rate": float(alerted_fraud_rate),
    }
    with open(ARTIFACTS_DIR / "phase8_investigation_metrics.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"wrote {ARTIFACTS_DIR / 'phase8_investigation_metrics.json'}")


if __name__ == "__main__":
    main()
