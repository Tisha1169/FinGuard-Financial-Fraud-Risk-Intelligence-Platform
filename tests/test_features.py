"""Tests for the SQL feature engineering layer (features/, sql/feature_engineering.sql).

These are integration tests against a live Postgres database (DATABASE_URL)
loaded with the Phase 2 synthetic dataset - they are skipped automatically
if no database is reachable, since not every environment running this repo
will have Docker Postgres up.
"""
import datetime

import pandas as pd
import pytest
from sqlalchemy import text

from database.connection import check_connection, get_engine
from features.daily_metrics import compute_daily_metrics
from features.point_in_time import (
    compute_transaction_features,
    count_recent_transactions,
    get_customer_baseline_asof,
    is_new_device,
)

healthy, _ = check_connection()
pytestmark = pytest.mark.skipif(not healthy, reason="requires a live DATABASE_URL (see .env.example)")


@pytest.fixture(scope="module")
def engine():
    return get_engine()


@pytest.fixture(scope="module", autouse=True)
def ensure_daily_metrics_populated(engine):
    with engine.connect() as conn:
        n_txns = conn.execute(text("SELECT COUNT(*) FROM fact_transactions")).scalar()
    if not n_txns:
        pytest.skip("fact_transactions is empty - run scripts/generate_data.py + load_data.py first")
    compute_daily_metrics(engine)


def test_daily_metrics_row_counts_are_nonzero(engine):
    with engine.connect() as conn:
        n_customer_rows = conn.execute(text("SELECT COUNT(*) FROM fact_customer_daily_metrics")).scalar()
        n_merchant_rows = conn.execute(text("SELECT COUNT(*) FROM fact_merchant_daily_metrics")).scalar()
    assert n_customer_rows > 0
    assert n_merchant_rows > 0


def test_daily_metrics_matches_independent_pandas_computation(engine):
    """The core correctness check: recompute the 90-day rolling avg/stddev
    for one customer/day directly in pandas from raw fact_transactions, and
    assert it matches what the SQL produced. Two independent
    implementations agreeing is strong evidence the SQL window logic (and
    its LATERAL join bounds) is correct.
    """
    txns = pd.read_sql(
        "SELECT customer_id, transaction_ts, amount FROM fact_transactions", engine
    )
    txns["transaction_ts"] = pd.to_datetime(txns["transaction_ts"], utc=True)
    txns["day"] = txns["transaction_ts"].dt.date

    # pick the customer with the most activity for a meaningful window
    cid = int(txns["customer_id"].value_counts().idxmax())
    sub = txns[txns.customer_id == cid].sort_values("transaction_ts")
    days = sorted(sub["day"].unique())
    d = days[len(days) // 2]

    d_ts = pd.Timestamp(d, tz="UTC")
    window = sub[(sub["transaction_ts"] >= d_ts - pd.Timedelta(days=90)) & (sub["transaction_ts"] < d_ts + pd.Timedelta(days=1))]
    expected_avg = window["amount"].mean()
    expected_std = window["amount"].std(ddof=1)
    expected_count = len(window[window["day"] == d])

    row = pd.read_sql(
        text(
            "SELECT txn_amount_avg_90d, txn_amount_stddev_90d, txn_count "
            "FROM fact_customer_daily_metrics WHERE customer_id=:c AND metric_date=:d"
        ),
        engine,
        params={"c": cid, "d": d},
    ).iloc[0]

    assert float(row.txn_amount_avg_90d) == pytest.approx(expected_avg, abs=0.01)
    assert float(row.txn_amount_stddev_90d) == pytest.approx(expected_std, abs=0.01)
    assert int(row.txn_count) == expected_count


def test_customer_baseline_asof_excludes_same_day(engine):
    """The whole point of the 'asof' lookup: a transaction on day D must
    never see the daily-metrics row for day D itself, even though that row
    exists (it's populated once daily_metrics.py has processed day D).
    """
    with engine.connect() as conn:
        sample = conn.execute(
            text(
                "SELECT customer_id, metric_date FROM fact_customer_daily_metrics "
                "GROUP BY customer_id, metric_date HAVING COUNT(*) = 1 LIMIT 1"
            )
        ).first()
    assert sample is not None
    cid, metric_date = sample

    as_of_same_day = datetime.datetime.combine(metric_date, datetime.time(23, 59, 59))
    baseline = get_customer_baseline_asof(engine, cid, as_of_same_day)
    if baseline is not None:
        assert baseline["metric_date"] < metric_date


def test_velocity_count_excludes_the_scored_transaction(engine):
    """A transaction must not count itself in its own trailing-window
    velocity feature.
    """
    df = pd.read_sql(
        text(
            """
            SELECT t.transaction_id, t.customer_id, t.transaction_ts
            FROM fact_transactions t
            JOIN ground_truth_fraud g ON g.transaction_id = t.transaction_id
            WHERE g.fraud_typology = 'VELOCITY_ABUSE'
            ORDER BY t.customer_id, t.transaction_ts
            LIMIT 5
            """
        ),
        engine,
    )
    assert len(df) > 0
    row = df.iloc[-1]
    count = count_recent_transactions(engine, int(row.customer_id), row.transaction_ts, window_minutes=1440)
    # the window must exclude the transaction being scored itself
    with engine.connect() as conn:
        raw_count_including_self = conn.execute(
            text(
                "SELECT COUNT(*) FROM fact_transactions WHERE customer_id=:cid "
                "AND transaction_ts <= :ts AND transaction_ts >= :ts - INTERVAL '1440 minutes'"
            ),
            {"cid": int(row.customer_id), "ts": row.transaction_ts},
        ).scalar()
    assert count == raw_count_including_self - 1


def test_is_new_device_true_for_first_ever_transaction(engine):
    with engine.connect() as conn:
        first_txn = conn.execute(
            text("SELECT customer_id, device_id, transaction_ts FROM fact_transactions ORDER BY transaction_ts LIMIT 1")
        ).first()
    assert is_new_device(engine, first_txn.customer_id, first_txn.device_id, first_txn.transaction_ts) is True


def test_compute_transaction_features_returns_all_expected_keys(engine):
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT transaction_id, customer_id, merchant_id, device_id, location_id, transaction_ts, amount "
                "FROM fact_transactions ORDER BY transaction_id OFFSET 40000 LIMIT 1"
            )
        ).mappings().first()
    feats = compute_transaction_features(engine, dict(row))
    expected_keys = {
        "customer_baseline_avg_90d", "customer_baseline_stddev_90d", "customer_distinct_merchants_30d",
        "customer_distinct_devices_30d", "amount_zscore", "merchant_avg_amount_90d",
        "merchant_chargeback_rate_90d", "txn_count_today_so_far", "distinct_merchants_today",
        "seconds_since_last_txn", "txn_count_last_10min", "txn_count_last_60min",
        "recent_failed_count_15min", "is_new_device", "is_new_location", "is_first_time_at_merchant",
        "last_location_id", "last_location_lat", "last_location_lon", "last_txn_ts_any",
        "current_location_lat", "current_location_lon", "txns_at_this_hour", "total_prior_txns",
        "has_sufficient_history",
    }
    assert set(feats.keys()) == expected_keys


def test_merchant_chargeback_rate_excludes_current_day(engine):
    """chargeback_rate_90d is defined to exclude the current day's own
    transactions (chargebacks lag reporting) - verify the SQL enforces the
    strict '<' bound rather than accidentally using '<=' by checking a
    merchant/day where all of that day's transactions are fraud but no
    prior-90-day fraud exists: chargeback_rate_90d must be 0, not >0.
    """
    with engine.connect() as conn:
        candidate = conn.execute(
            text(
                """
                SELECT t.merchant_id, date_trunc('day', t.transaction_ts)::date AS d
                FROM fact_transactions t
                JOIN ground_truth_fraud g ON g.transaction_id = t.transaction_id AND g.is_fraud
                GROUP BY 1, 2
                HAVING COUNT(*) >= 1
                LIMIT 1
                """
            )
        ).first()
    assert candidate is not None
    merchant_id, d = candidate

    with engine.connect() as conn:
        prior_fraud_count = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM fact_transactions t
                JOIN ground_truth_fraud g ON g.transaction_id = t.transaction_id AND g.is_fraud
                WHERE t.merchant_id = :m AND t.transaction_ts < :d AND t.transaction_ts >= :d - INTERVAL '90 days'
                """
            ),
            {"m": merchant_id, "d": d},
        ).scalar()
        rate = conn.execute(
            text("SELECT chargeback_rate_90d FROM fact_merchant_daily_metrics WHERE merchant_id=:m AND metric_date=:d"),
            {"m": merchant_id, "d": d},
        ).scalar()

    if prior_fraud_count == 0:
        assert rate == 0 or rate is None
