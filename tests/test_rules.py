"""Tests for the rules engine.

Unit tests (no DB) exercise rules/engine.py directly against hand-built
feature dicts - these check threshold boundaries precisely. Integration
tests (skipped without a live DATABASE_URL) validate the vectorized batch
engine against real data: that it agrees with the single-transaction engine
on the same transactions, and that overall recall/precision against ground
truth land in a sane range.
"""
import datetime

import pandas as pd
import pytest

from database.connection import check_connection, get_engine
from rules import config
from rules.engine import evaluate_transaction, rule_off_hours, rule_unusual_amount, rule_velocity
from rules.severity import severity_for_ratio


# ---------------------------------------------------------------------------
# Unit tests: rules/engine.py against hand-built feature dicts
# ---------------------------------------------------------------------------

BASE_FEATURES = {
    "customer_baseline_avg_90d": 50.0,
    "customer_baseline_stddev_90d": 10.0,
    "customer_distinct_merchants_30d": 5,
    "customer_distinct_devices_30d": 1,
    "amount_zscore": 0.0,
    "merchant_avg_amount_90d": 50.0,
    "merchant_chargeback_rate_90d": 0.01,
    "txn_count_today_so_far": 0,
    "distinct_merchants_today": 0,
    "seconds_since_last_txn": None,
    "txn_count_last_10min": 0,
    "txn_count_last_60min": 0,
    "recent_failed_count_15min": 0,
    "is_new_device": False,
    "is_new_location": False,
    "is_first_time_at_merchant": False,
    "last_location_id": None,
    "last_location_lat": None,
    "last_location_lon": None,
    "last_txn_ts_any": None,
    "current_location_lat": 40.0,
    "current_location_lon": -74.0,
    "txns_at_this_hour": 5,
    "total_prior_txns": 50,
    "has_sufficient_history": True,
}

BASE_TXN = {
    "amount": 50.0,
    "status": "APPROVED",
    "transaction_ts": datetime.datetime(2026, 6, 1, 14, 0, 0),
}


def test_severity_scaling():
    assert severity_for_ratio(1.0, 1.0) == "MEDIUM"
    assert severity_for_ratio(1.5, 1.0) == "HIGH"
    assert severity_for_ratio(2.0, 1.0) == "CRITICAL"
    assert severity_for_ratio(0.5, 1.0) == "LOW"


def test_velocity_rule_does_not_fire_below_threshold():
    features = {**BASE_FEATURES, "txn_count_last_10min": config.VELOCITY_10MIN_TRIGGER - 1}
    assert rule_velocity(BASE_TXN, features) is None


def test_velocity_rule_fires_at_threshold():
    features = {**BASE_FEATURES, "txn_count_last_10min": config.VELOCITY_10MIN_TRIGGER}
    result = rule_velocity(BASE_TXN, features)
    assert result is not None
    assert result["rule_id"] == "R1_VELOCITY"
    assert result["severity"] == "MEDIUM"


def test_unusual_amount_rule_uses_zscore_when_available():
    features = {**BASE_FEATURES, "amount_zscore": config.AMOUNT_ZSCORE_TRIGGER}
    result = rule_unusual_amount(BASE_TXN, features)
    assert result is not None
    assert result["rule_id"] == "R2_AMOUNT_SPIKE"
    assert "zscore" in result["evidence"]


def test_unusual_amount_rule_falls_back_without_stddev():
    features = {**BASE_FEATURES, "amount_zscore": None, "customer_baseline_avg_90d": 10.0}
    txn = {**BASE_TXN, "amount": 10.0 * config.AMOUNT_FALLBACK_MULTIPLIER}
    result = rule_unusual_amount(txn, features)
    assert result is not None
    assert "multiplier" in result["evidence"]


def test_unusual_amount_rule_silent_with_no_history_at_all():
    features = {**BASE_FEATURES, "amount_zscore": None, "customer_baseline_avg_90d": None}
    assert rule_unusual_amount(BASE_TXN, features) is None


def test_off_hours_rule_requires_minimum_history():
    features = {**BASE_FEATURES, "total_prior_txns": config.OFF_HOURS_MIN_PRIOR_TXNS - 1, "txns_at_this_hour": 0}
    assert rule_off_hours(BASE_TXN, features) is None


def test_off_hours_rule_fires_for_never_seen_hour_with_enough_history():
    features = {**BASE_FEATURES, "total_prior_txns": config.OFF_HOURS_MIN_PRIOR_TXNS, "txns_at_this_hour": 0}
    result = rule_off_hours(BASE_TXN, features)
    assert result is not None
    assert result["rule_id"] == "R7_OFF_HOURS"


def test_evaluate_transaction_returns_only_fired_rules():
    # BASE_FEATURES is deliberately "boring" - nothing should fire.
    fired = evaluate_transaction(BASE_TXN, BASE_FEATURES)
    assert fired == []


def test_evaluate_transaction_returns_multiple_firings_when_applicable():
    features = {
        **BASE_FEATURES,
        "txn_count_last_10min": config.VELOCITY_10MIN_TRIGGER,
        "is_new_device": True,
        "is_new_location": True,
    }
    fired = evaluate_transaction(BASE_TXN, features)
    fired_ids = {r["rule_id"] for r in fired}
    assert "R1_VELOCITY" in fired_ids
    assert "R5_NEW_DEVICE_NEW_LOCATION" in fired_ids


# ---------------------------------------------------------------------------
# Integration tests: rules/batch.py against live data
# ---------------------------------------------------------------------------

healthy, _ = check_connection()
pytestmark_db = pytest.mark.skipif(not healthy, reason="requires a live DATABASE_URL (see .env.example)")


@pytestmark_db
def test_batch_rules_have_been_populated():
    engine = get_engine()
    with engine.connect() as conn:
        from sqlalchemy import text
        n = conn.execute(text("SELECT COUNT(*) FROM rules_triggered")).scalar()
    if n == 0:
        pytest.skip("rules_triggered is empty - run scripts/run_rules.py first")
    assert n > 0


@pytestmark_db
def test_batch_engine_matches_single_transaction_engine_on_a_sample():
    """The vectorized batch engine and the per-transaction engine implement
    the same logic two different ways - they must agree. Sample a handful
    of transactions with enough history to be meaningful and cross-check.
    """
    from sqlalchemy import text

    from features.point_in_time import compute_transaction_features
    from rules.batch import apply_rules, build_batch_features

    engine = get_engine()
    batch_df = build_batch_features(engine)
    firings = apply_rules(batch_df)
    batch_fired_ids = firings.groupby("transaction_id")["rule_id"].apply(set).to_dict()

    with engine.connect() as conn:
        sample = conn.execute(
            text(
                "SELECT transaction_id, customer_id, merchant_id, device_id, location_id, transaction_ts, amount, status "
                "FROM fact_transactions ORDER BY transaction_id OFFSET 60000 LIMIT 15"
            )
        ).mappings().all()

    for row in sample:
        txn = dict(row)
        features = compute_transaction_features(engine, txn)
        live_fired = {r["rule_id"] for r in evaluate_transaction(txn, features)}
        batch_fired = batch_fired_ids.get(txn["transaction_id"], set())
        assert live_fired == batch_fired, f"mismatch on transaction {txn['transaction_id']}: live={live_fired} batch={batch_fired}"


@pytestmark_db
def test_rules_engine_recall_and_precision_are_in_expected_range():
    """Sanity bounds on the overall rules-only performance: recall should
    be meaningfully better than random, and precision should be low enough
    to demonstrate why ML + risk scoring is needed on top (rules alone are
    not the final answer). These bounds intentionally document the known,
    expected behavior rather than aiming for perfection.
    """
    from sqlalchemy import text

    engine = get_engine()
    with engine.connect() as conn:
        total_fraud = conn.execute(text("SELECT COUNT(*) FROM ground_truth_fraud WHERE is_fraud")).scalar()
        fraud_caught = conn.execute(
            text(
                "SELECT COUNT(DISTINCT g.transaction_id) FROM ground_truth_fraud g "
                "JOIN rules_triggered r ON r.transaction_id = g.transaction_id WHERE g.is_fraud"
            )
        ).scalar()
        total_flagged = conn.execute(text("SELECT COUNT(DISTINCT transaction_id) FROM rules_triggered")).scalar()

    recall = fraud_caught / total_fraud
    precision = fraud_caught / total_flagged
    assert recall > 0.5, f"rules recall unexpectedly low: {recall:.2%}"
    assert precision < 0.5, f"rules precision unexpectedly high for a rules-only baseline: {precision:.2%}"
