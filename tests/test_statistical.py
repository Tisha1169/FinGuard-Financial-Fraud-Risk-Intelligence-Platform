"""Tests for the statistical/behavioral anomaly engine (models/statistical.py,
models/isolation_forest.py).

Unit tests exercise the IQR/Tukey-fence and z-score math directly with
hand-built pandas Series (no DB). Integration tests (skipped without a live
DATABASE_URL) validate against real data - including a calibration
regression guard for a real bug caught during development: an unnormalized
Poisson expected-count estimate inflated the frequency-deviation z-score
for ~15% of all transactions instead of the ~0.1-0.5% a well-calibrated
z-score implies.
"""
import numpy as np
import pandas as pd
import pytest

from database.connection import check_connection, get_engine
from models.statistical import (
    MIN_HISTORY_FOR_IQR,
    _iqr_upper_outlier_score,
    _normalize_zscore,
)

# ---------------------------------------------------------------------------
# Unit tests: pure math, no DB
# ---------------------------------------------------------------------------


def test_iqr_score_zero_below_fence():
    amount = pd.Series([50.0])
    q1 = pd.Series([20.0])
    q3 = pd.Series([40.0])
    count = pd.Series([MIN_HISTORY_FOR_IQR])
    score = _iqr_upper_outlier_score(amount, q1, q3, count)
    # fence = 40 + 1.5*20 = 70; 50 < 70 -> no outlier
    assert score.iloc[0] == 0


def test_iqr_score_positive_above_fence():
    amount = pd.Series([100.0])
    q1 = pd.Series([20.0])
    q3 = pd.Series([40.0])
    count = pd.Series([MIN_HISTORY_FOR_IQR])
    score = _iqr_upper_outlier_score(amount, q1, q3, count)
    # fence = 70, IQR = 20 -> score = (100-70)/20 = 1.5
    assert score.iloc[0] == pytest.approx(1.5)


def test_iqr_score_suppressed_below_min_history():
    """The exact bug this test guards against: a thin-history customer
    (few prior transactions) can have a tiny, unstable IQR that turns an
    ordinary transaction into an absurd outlier score. Below
    MIN_HISTORY_FOR_IQR the score must be suppressed to 0 regardless of
    how extreme the raw ratio looks.
    """
    amount = pd.Series([305.74])
    q1 = pd.Series([114.50])
    q3 = pd.Series([114.50])  # IQR effectively 0 from only 2-3 observations
    count = pd.Series([3])
    score = _iqr_upper_outlier_score(amount, q1, q3, count)
    assert score.iloc[0] == 0


def test_iqr_score_zero_when_iqr_is_zero_even_with_enough_history():
    amount = pd.Series([100.0])
    q1 = pd.Series([50.0])
    q3 = pd.Series([50.0])
    count = pd.Series([MIN_HISTORY_FOR_IQR])
    score = _iqr_upper_outlier_score(amount, q1, q3, count)
    assert score.iloc[0] == 0


def test_normalize_zscore_saturates_at_trigger():
    z = pd.Series([0.0, 1.5, 3.0, 5.0, -2.0])
    normalized = _normalize_zscore(z, trigger_z=3.0)
    assert list(normalized) == pytest.approx([0.0, 0.5, 1.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Integration tests: real data
# ---------------------------------------------------------------------------

healthy, _ = check_connection()
pytestmark_db = pytest.mark.skipif(not healthy, reason="requires a live DATABASE_URL (see .env.example)")


@pytestmark_db
def test_frequency_zscore_is_well_calibrated():
    """Regression guard for the elapsed-days bug: with a correctly
    estimated expected count, a z-score >= 3 should be rare (roughly a
    one-tailed p<0.001 event), not routine. The original bug made this
    ~14.7%; this asserts it stays under 2%.
    """
    from models.statistical import build_statistical_features

    engine = get_engine()
    df = build_statistical_features(engine)
    z = df["frequency_deviation_zscore"].dropna()
    assert len(z) > 1000
    share_extreme = (z >= 3).mean()
    assert share_extreme < 0.02, f"frequency z-score miscalibrated: {share_extreme:.2%} exceed z>=3"
    # a well-calibrated z-score should also have roughly unit variance
    assert 0.5 < z.std() < 2.0


@pytestmark_db
def test_customer_iqr_score_excludes_own_transaction():
    """Leakage check: recompute one transaction's IQR fence manually from
    only its strictly-prior history and confirm it matches what
    build_statistical_features produced - i.e. the transaction's own
    amount never entered its own baseline.
    """
    from models.statistical import build_statistical_features
    from rules.batch import load_transactions

    engine = get_engine()
    txns = load_transactions(engine)
    stat_df = build_statistical_features(engine)

    # pick a customer with plenty of history and a mid-history transaction
    counts = txns["customer_id"].value_counts()
    cid = counts[counts >= 30].index[0]
    cust_txns = txns[txns.customer_id == cid].sort_values("transaction_ts").reset_index(drop=True)
    target = cust_txns.iloc[20]

    prior = cust_txns[cust_txns["transaction_ts"] < target["transaction_ts"]]
    prior = prior[prior["transaction_ts"] >= target["transaction_ts"] - pd.Timedelta(days=90)]
    expected_q1 = prior["amount"].quantile(0.25)
    expected_q3 = prior["amount"].quantile(0.75)

    row = stat_df[stat_df["transaction_id"] == target["transaction_id"]].iloc[0]
    assert row["customer_id_amount_q1_asof"] == pytest.approx(expected_q1)
    assert row["customer_id_amount_q3_asof"] == pytest.approx(expected_q3)


@pytestmark_db
def test_behavioral_score_beats_random_baseline():
    from sklearn.metrics import roc_auc_score

    from models.statistical import build_statistical_features

    engine = get_engine()
    df = build_statistical_features(engine)
    ground_truth = pd.read_sql("SELECT transaction_id, is_fraud FROM ground_truth_fraud", engine)
    merged = df.merge(ground_truth, on="transaction_id")

    auc = roc_auc_score(merged["is_fraud"].astype(int), merged["behavioral_anomaly_score"].fillna(0))
    assert auc > 0.6, f"behavioral anomaly score AUC unexpectedly low: {auc:.3f}"


@pytestmark_db
def test_isolation_forest_score_is_bounded_and_separates_fraud():
    from sklearn.metrics import roc_auc_score

    from models.isolation_forest import add_isolation_forest_score
    from models.statistical import build_statistical_features
    from rules.batch import build_batch_features

    engine = get_engine()
    rules_df = build_batch_features(engine)
    stat_df = build_statistical_features(engine)
    combined = rules_df.merge(
        stat_df[["transaction_id", "customer_amount_iqr_score", "merchant_amount_iqr_score", "frequency_deviation_zscore"]],
        on="transaction_id", how="left",
    )
    combined = add_isolation_forest_score(combined)

    assert combined["isolation_forest_score"].between(0, 1).all()

    ground_truth = pd.read_sql("SELECT transaction_id, is_fraud FROM ground_truth_fraud", engine)
    merged = combined.merge(ground_truth, on="transaction_id")
    auc = roc_auc_score(merged["is_fraud"].astype(int), merged["isolation_forest_score"])
    assert auc > 0.8, f"isolation forest AUC unexpectedly low: {auc:.3f}"
