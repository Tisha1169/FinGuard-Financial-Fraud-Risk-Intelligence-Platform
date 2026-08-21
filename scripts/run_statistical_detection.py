"""Computes IQR-based behavioral anomaly scores and Isolation Forest scores
for every transaction, evaluates both against ground truth (diagnostic
only - Isolation Forest is unsupervised and was not fit against labels),
and caches the result to data/behavioral_features.parquet for reuse by
Phase 6 (ML model) and Phase 7 (risk scoring) without recomputing.

Usage:
    python scripts/run_statistical_detection.py
"""
import pathlib
import sys

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from database.connection import get_engine  # noqa: E402
from models.isolation_forest import add_isolation_forest_score  # noqa: E402
from models.statistical import build_statistical_features  # noqa: E402
from rules.batch import build_batch_features  # noqa: E402

load_dotenv()

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

STAT_COLUMNS = [
    "transaction_id", "customer_amount_iqr_score", "merchant_amount_iqr_score",
    "frequency_deviation_zscore", "behavioral_anomaly_score",
]


def evaluate(df: pd.DataFrame, score_col: str, label_col: str = "is_fraud") -> dict:
    y = df[label_col].astype(int)
    scores = df[score_col].fillna(0)
    auc = roc_auc_score(y, scores)

    k = max(1, int(len(df) * 0.02))  # top-2%, matching the known injection rate
    top_k = df.nlargest(k, score_col)
    precision_at_k = top_k[label_col].mean()
    recall_at_k = top_k[label_col].sum() / y.sum()

    return {"auc": round(auc, 4), "precision_at_top2pct": round(precision_at_k, 4), "recall_at_top2pct": round(recall_at_k, 4)}


def main() -> None:
    engine = get_engine()

    print("building rule/velocity features...")
    rules_df = build_batch_features(engine)

    print("building statistical (IQR + frequency deviation) features...")
    stat_df = build_statistical_features(engine)

    combined = rules_df.merge(stat_df[STAT_COLUMNS], on="transaction_id", how="left")

    print("fitting isolation forest...")
    combined = add_isolation_forest_score(combined)

    ground_truth = pd.read_sql("SELECT transaction_id, is_fraud FROM ground_truth_fraud", engine)
    combined = combined.merge(ground_truth, on="transaction_id", how="left")

    print()
    print("evaluation against ground truth (diagnostic - not used for fitting):")
    for col in ["behavioral_anomaly_score", "isolation_forest_score"]:
        metrics = evaluate(combined, col)
        print(f"  {col}: {metrics}")

    output_cols = [
        "transaction_id", "customer_amount_iqr_score", "merchant_amount_iqr_score",
        "frequency_deviation_zscore", "behavioral_anomaly_score", "isolation_forest_score",
    ]
    out_path = DATA_DIR / "behavioral_features.parquet"
    combined[output_cols].to_parquet(out_path, index=False)
    print(f"\nwrote {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
