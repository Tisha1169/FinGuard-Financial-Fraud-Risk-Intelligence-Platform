"""Assembles the full, documented feature matrix for the supervised ML
model - the single source of truth both scripts/train_models.py and the
test suite build from, so "what feature list did we train on" is never
ambiguous.

Design decisions carried over from the Phase 5 audit:
- Identifiers (transaction_id, customer_id, merchant_id, device_id,
  location_id) are kept as columns for joining/error-analysis but are
  NEVER passed to a model as a feature - a synthetic dataset this size
  would let a model overfit to specific entity IDs rather than learning
  generalizable behavior.
- `chargeback_rate_90d` is deliberately EXCLUDED from FEATURE_COLUMNS.
  See docs/model_card.md "chargeback_rate_90d exclusion" for the full
  reasoning: it is not temporally leaking (proven by
  tests/test_ml_pipeline.py::test_chargeback_rate_is_temporally_safe,
  which extends the Phase 3 proof), but its next-day availability is
  unrealistic versus real chargeback reporting lag (weeks to months), so
  including it would overstate real-world achievable performance.
- `isolation_forest_score` IS a feature here, but is computed by a
  train-only-fit Isolation Forest (models/ml_isolation_forest.py) inside
  the training pipeline, never by Phase 5's whole-dataset
  models/isolation_forest.py - see that module's docstring for why reusing
  the Phase 5 scores directly would leak test-period structure into
  training.
"""
import pandas as pd
from sqlalchemy.engine import Engine

from models.statistical import build_statistical_features
from rules.batch import build_batch_features

# Rule IDs fired become one binary feature each - see FEATURE_DOCS below.
RULE_IDS = [
    "R1_VELOCITY", "R2_AMOUNT_SPIKE", "R3_GEO_IMPOSSIBLE", "R4_FAILED_THEN_LARGE",
    "R5_NEW_DEVICE_NEW_LOCATION", "R6_MERCHANT_DEVIATION", "R7_OFF_HOURS",
]
SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

IDENTIFIER_COLUMNS = ["transaction_id", "customer_id", "merchant_id", "device_id", "location_id"]
NON_FEATURE_COLUMNS = IDENTIFIER_COLUMNS + ["transaction_ts", "is_fraud", "fraud_typology", "is_synthetic_label"]

NUMERIC_FEATURE_COLUMNS = [
    "amount",
    "amount_zscore",
    "txn_count_last_10min",
    "txn_count_last_60min",
    "recent_failed_count_15min",
    "distinct_merchants_30d",
    "distinct_devices_30d",
    "customer_amount_iqr_score",
    "merchant_amount_iqr_score",
    "frequency_deviation_zscore",
    "hour",
    "rules_fired_count",
    "max_rule_severity",
]
BOOLEAN_FEATURE_COLUMNS = ["is_new_device", "is_new_location", "is_first_time_at_merchant"] + [
    f"rule_fired_{rid}" for rid in RULE_IDS
]
CATEGORICAL_FEATURE_COLUMNS = ["channel", "status", "customer_risk_segment", "merchant_risk_category"]

# isolation_forest_score is added by the training pipeline after a
# train-only fit - it is not produced by this module, but is documented
# here since it's part of the final feature set fed to the ML models.
DERIVED_AFTER_SPLIT_COLUMNS = ["isolation_forest_score"]

FEATURE_DOCS = {
    "amount": "Raw transaction amount in USD.",
    "amount_zscore": "(amount - customer's 90d avg) / 90d stddev, as of the end of the previous calendar day. NaN if insufficient history (imputed downstream).",
    "txn_count_last_10min": "Count of this customer's transactions in the trailing 10 minutes, strictly before this one.",
    "txn_count_last_60min": "Count of this customer's transactions in the trailing 60 minutes, strictly before this one.",
    "recent_failed_count_15min": "Count of this customer's FAILED-status transactions in the trailing 15 minutes, strictly before this one.",
    "distinct_merchants_30d": "Distinct merchants this customer transacted with in the trailing 30 days, as of the end of the previous calendar day.",
    "distinct_devices_30d": "Distinct devices this customer used in the trailing 30 days, as of the end of the previous calendar day.",
    "customer_amount_iqr_score": "Tukey upper-fence outlier score of amount vs. this customer's trailing 90d amount distribution, as of the immediately preceding transaction. 0 if under MIN_HISTORY_FOR_IQR.",
    "merchant_amount_iqr_score": "Same, but vs. the merchant's own trailing 90d amount distribution.",
    "frequency_deviation_zscore": "Poisson z-score of this customer's trailing 7-day transaction count vs. their expected rate from trailing 90d activity. NaN if under MIN_HISTORY_FOR_FREQUENCY.",
    "hour": "Hour of day (0-23) the transaction occurred, in UTC.",
    "rules_fired_count": "Number of Phase 4 business rules that fired on this transaction (0-7).",
    "max_rule_severity": "Highest severity among fired rules, ordinal-encoded LOW=0 ... CRITICAL=3, or -1 if no rule fired.",
    "is_new_device": "True if this device has never appeared for this customer before.",
    "is_new_location": "True if this location has never appeared for this customer before.",
    "is_first_time_at_merchant": "True if this customer has never transacted at this merchant before.",
    "channel": "Transaction channel: CARD_PRESENT, ECOM, POS, ATM, WALLET.",
    "status": "Transaction outcome: APPROVED, DECLINED, FAILED.",
    "customer_risk_segment": "Static customer risk segment: STANDARD, WATCHLIST, VIP.",
    "merchant_risk_category": "Static merchant risk category: STANDARD, HIGH_RISK.",
    "isolation_forest_score": "Unsupervised anomaly score in [0,1] from a train-only-fit Isolation Forest (see models/ml_isolation_forest.py) - added by the training pipeline, not this module.",
}
for rid in RULE_IDS:
    FEATURE_DOCS[f"rule_fired_{rid}"] = f"True if rule {rid} fired on this transaction (see docs/rules_engine.md)."

FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + BOOLEAN_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS


def _load_rule_firings(engine: Engine) -> pd.DataFrame:
    firings = pd.read_sql("SELECT transaction_id, rule_id, severity FROM rules_triggered", engine)
    if firings.empty:
        return pd.DataFrame(columns=["transaction_id", "rules_fired_count", "max_rule_severity"] + [f"rule_fired_{r}" for r in RULE_IDS])

    firings["severity_ordinal"] = firings["severity"].map(SEVERITY_ORDER)
    agg = firings.groupby("transaction_id").agg(
        rules_fired_count=("rule_id", "count"),
        max_rule_severity=("severity_ordinal", "max"),
    ).reset_index()

    pivot = firings.pivot_table(index="transaction_id", columns="rule_id", values="severity", aggfunc="size", fill_value=0)
    pivot = (pivot > 0).astype(bool)
    for rid in RULE_IDS:
        if rid not in pivot.columns:
            pivot[rid] = False
    pivot = pivot[RULE_IDS].reset_index()
    pivot.columns = ["transaction_id"] + [f"rule_fired_{r}" for r in RULE_IDS]

    return agg.merge(pivot, on="transaction_id", how="outer")


def _load_entity_attributes(engine: Engine) -> pd.DataFrame:
    txns = pd.read_sql(
        """
        SELECT t.transaction_id, t.channel,
               c.risk_segment AS customer_risk_segment, m.risk_category AS merchant_risk_category
        FROM fact_transactions t
        JOIN dim_customer c ON c.customer_id = t.customer_id
        JOIN dim_merchant m ON m.merchant_id = t.merchant_id
        """,
        engine,
    )
    return txns


def build_feature_matrix(engine: Engine) -> pd.DataFrame:
    """Returns one row per transaction with every engineered feature,
    identifiers, the timestamp (for splitting), and the label. Does NOT
    include isolation_forest_score - the training pipeline adds that after
    the chronological split, fitting the forest on the training period only.
    """
    rules_df = build_batch_features(engine)
    stat_df = build_statistical_features(engine)
    rule_firings = _load_rule_firings(engine)
    entity_attrs = _load_entity_attributes(engine)

    stat_cols = ["transaction_id", "customer_amount_iqr_score", "merchant_amount_iqr_score", "frequency_deviation_zscore"]
    df = rules_df.merge(stat_df[stat_cols], on="transaction_id", how="left")
    df = df.merge(rule_firings, on="transaction_id", how="left")
    df = df.merge(entity_attrs, on="transaction_id", how="left")

    df["rules_fired_count"] = df["rules_fired_count"].fillna(0).astype(int)
    df["max_rule_severity"] = df["max_rule_severity"].fillna(-1).astype(int)
    for rid in RULE_IDS:
        df[f"rule_fired_{rid}"] = df[f"rule_fired_{rid}"].fillna(False).astype(bool)

    ground_truth = pd.read_sql("SELECT transaction_id, is_fraud, fraud_typology, is_synthetic_label FROM ground_truth_fraud", engine)
    df = df.merge(ground_truth, on="transaction_id", how="left")

    keep_cols = NON_FEATURE_COLUMNS + FEATURE_COLUMNS
    return df[keep_cols].sort_values("transaction_ts").reset_index(drop=True)
