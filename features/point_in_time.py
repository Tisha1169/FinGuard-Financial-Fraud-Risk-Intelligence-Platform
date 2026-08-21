"""Point-in-time feature lookups for a single transaction being scored.

Every function here is deliberately "as-of": it only reads data that would
have been known at the moment the transaction happened. This is what
prevents temporal leakage from creeping into the rules engine (Phase 4)
or the ML model's training features (Phase 6):

- Daily baselines (fact_customer_daily_metrics / fact_merchant_daily_metrics)
  are looked up for the most recent day STRICTLY BEFORE the transaction's
  own day - never the transaction's own day's row, even though that row
  technically exists once daily_metrics.py has run past it.
- Same-day activity (velocity, distinct devices seen today) is computed
  directly from fact_transactions, filtered to strictly before the
  transaction's own timestamp.
"""
from sqlalchemy import text
from sqlalchemy.engine import Engine


def get_customer_baseline_asof(engine: Engine, customer_id: int, as_of_ts) -> dict | None:
    query = text(
        """
        SELECT txn_count, txn_amount_sum, txn_amount_avg_90d, txn_amount_stddev_90d,
               distinct_merchants_30d, distinct_devices_30d, metric_date
        FROM fact_customer_daily_metrics
        WHERE customer_id = :cid AND metric_date < :as_of_date
        ORDER BY metric_date DESC
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"cid": customer_id, "as_of_date": as_of_ts.date()}).mappings().first()
    return dict(row) if row else None


def get_merchant_baseline_asof(engine: Engine, merchant_id: int, as_of_ts) -> dict | None:
    query = text(
        """
        SELECT txn_count, txn_amount_sum, avg_txn_amount_90d, chargeback_rate_90d, metric_date
        FROM fact_merchant_daily_metrics
        WHERE merchant_id = :mid AND metric_date < :as_of_date
        ORDER BY metric_date DESC
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"mid": merchant_id, "as_of_date": as_of_ts.date()}).mappings().first()
    return dict(row) if row else None


def get_intraday_activity(engine: Engine, customer_id: int, as_of_ts) -> dict:
    """Same-day activity strictly before as_of_ts - the intraday complement
    to the (necessarily stale-by-a-day) daily baseline above.
    """
    query = text(
        """
        SELECT
            COUNT(*) AS txn_count_today_so_far,
            COUNT(DISTINCT merchant_id) AS distinct_merchants_today,
            MAX(transaction_ts) AS last_txn_ts
        FROM fact_transactions
        WHERE customer_id = :cid
          AND transaction_ts >= date_trunc('day', CAST(:as_of_ts AS timestamptz))
          AND transaction_ts < :as_of_ts
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"cid": customer_id, "as_of_ts": as_of_ts}).mappings().first()
    return dict(row) if row else {"txn_count_today_so_far": 0, "distinct_merchants_today": 0, "last_txn_ts": None}


def count_recent_transactions(engine: Engine, customer_id: int, as_of_ts, window_minutes: int) -> int:
    """Velocity signal: transactions strictly before as_of_ts within the
    trailing window_minutes. Used by the velocity/card-testing rules.
    """
    query = text(
        """
        SELECT COUNT(*) FROM fact_transactions
        WHERE customer_id = :cid
          AND transaction_ts < :as_of_ts
          AND transaction_ts >= CAST(:as_of_ts AS timestamptz) - (CAST(:window_minutes AS text) || ' minutes')::interval
        """
    )
    with engine.connect() as conn:
        return conn.execute(
            query, {"cid": customer_id, "as_of_ts": as_of_ts, "window_minutes": window_minutes}
        ).scalar()


def is_new_device(engine: Engine, customer_id: int, device_id: int, as_of_ts) -> bool:
    """True if this device has never appeared for this customer strictly
    before as_of_ts.
    """
    query = text(
        """
        SELECT 1 FROM fact_transactions
        WHERE customer_id = :cid AND device_id = :device_id AND transaction_ts < :as_of_ts
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        return conn.execute(query, {"cid": customer_id, "device_id": device_id, "as_of_ts": as_of_ts}).first() is None


def is_new_location(engine: Engine, customer_id: int, location_id: int, as_of_ts) -> bool:
    query = text(
        """
        SELECT 1 FROM fact_transactions
        WHERE customer_id = :cid AND location_id = :location_id AND transaction_ts < :as_of_ts
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        return conn.execute(
            query, {"cid": customer_id, "location_id": location_id, "as_of_ts": as_of_ts}
        ).first() is None


def compute_transaction_features(engine: Engine, transaction: dict) -> dict:
    """Assembles the full point-in-time feature set for one transaction.

    `transaction` must have: customer_id, merchant_id, device_id,
    location_id, transaction_ts, amount.
    """
    cid = transaction["customer_id"]
    as_of_ts = transaction["transaction_ts"]

    customer_baseline = get_customer_baseline_asof(engine, cid, as_of_ts)
    merchant_baseline = get_merchant_baseline_asof(engine, transaction["merchant_id"], as_of_ts)
    intraday = get_intraday_activity(engine, cid, as_of_ts)

    amount_zscore = None
    if customer_baseline and customer_baseline["txn_amount_stddev_90d"]:
        stddev = float(customer_baseline["txn_amount_stddev_90d"])
        if stddev > 0:
            amount_zscore = (float(transaction["amount"]) - float(customer_baseline["txn_amount_avg_90d"])) / stddev

    return {
        "customer_baseline_avg_90d": float(customer_baseline["txn_amount_avg_90d"]) if customer_baseline and customer_baseline["txn_amount_avg_90d"] is not None else None,
        "customer_baseline_stddev_90d": float(customer_baseline["txn_amount_stddev_90d"]) if customer_baseline and customer_baseline["txn_amount_stddev_90d"] is not None else None,
        "customer_distinct_merchants_30d": customer_baseline["distinct_merchants_30d"] if customer_baseline else None,
        "customer_distinct_devices_30d": customer_baseline["distinct_devices_30d"] if customer_baseline else None,
        "amount_zscore": amount_zscore,
        "merchant_avg_amount_90d": float(merchant_baseline["avg_txn_amount_90d"]) if merchant_baseline and merchant_baseline["avg_txn_amount_90d"] is not None else None,
        "merchant_chargeback_rate_90d": float(merchant_baseline["chargeback_rate_90d"]) if merchant_baseline and merchant_baseline["chargeback_rate_90d"] is not None else None,
        "txn_count_today_so_far": intraday["txn_count_today_so_far"],
        "distinct_merchants_today": intraday["distinct_merchants_today"],
        "seconds_since_last_txn": (
            (as_of_ts - intraday["last_txn_ts"]).total_seconds() if intraday["last_txn_ts"] else None
        ),
        "txn_count_last_10min": count_recent_transactions(engine, cid, as_of_ts, 10),
        "txn_count_last_60min": count_recent_transactions(engine, cid, as_of_ts, 60),
        "is_new_device": is_new_device(engine, cid, transaction["device_id"], as_of_ts),
        "is_new_location": is_new_location(engine, cid, transaction["location_id"], as_of_ts),
        "has_sufficient_history": customer_baseline is not None,
    }
