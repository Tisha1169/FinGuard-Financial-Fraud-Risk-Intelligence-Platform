"""Vectorized bulk rule scoring for backfilling rules_triggered across the
whole transaction history.

Why this exists separately from rules/engine.py + features/point_in_time.py:
that pair does ~10 small SQL queries per transaction, which is the right
design for scoring one live transaction (Phase 9's /risk/{id} endpoint) but
would take on the order of an hour to backfill ~100k transactions one at a
time. This module computes the identical point-in-time logic - same
leakage boundaries, same thresholds from rules/config.py, same severity
function - but vectorized over the whole dataset in pandas, the way a real
batch scoring/backfill job would be built. Any threshold or severity change
in rules/config.py or rules/severity.py automatically applies to both paths
since both import from there.
"""
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from data_generation import geo
from rules import config
from rules.severity import severity_for_ratio


def load_transactions(engine: Engine) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT t.transaction_id, t.customer_id, t.merchant_id, t.device_id, t.location_id,
               t.transaction_ts, t.amount, t.status,
               l.latitude AS current_lat, l.longitude AS current_lon
        FROM fact_transactions t
        JOIN dim_location l ON l.location_id = t.location_id
        ORDER BY t.customer_id, t.transaction_ts
        """,
        engine,
    )
    df["transaction_ts"] = pd.to_datetime(df["transaction_ts"], utc=True)
    df["amount"] = df["amount"].astype(float)
    df["day"] = df["transaction_ts"].dt.date
    df["hour"] = df["transaction_ts"].dt.hour
    return df


def _load_daily_metrics(engine: Engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    customer_daily = pd.read_sql(
        "SELECT customer_id, metric_date, txn_amount_avg_90d, txn_amount_stddev_90d, "
        "distinct_merchants_30d, distinct_devices_30d FROM fact_customer_daily_metrics",
        engine,
    )
    merchant_daily = pd.read_sql(
        "SELECT merchant_id, metric_date, avg_txn_amount_90d FROM fact_merchant_daily_metrics", engine
    )
    return customer_daily, merchant_daily


def _asof_join_customer_baseline(txns: pd.DataFrame, customer_daily: pd.DataFrame) -> pd.DataFrame:
    """Joins each transaction to the most recent customer_daily row STRICTLY
    before the transaction's own day, matching
    features.point_in_time.get_customer_baseline_asof's `metric_date < as_of_date`.
    merge_asof only supports <=, so the join key is shifted by one day.
    """
    txns = txns.copy()
    txns["asof_date"] = (pd.to_datetime(txns["day"]) - pd.Timedelta(days=1)).astype("datetime64[ns]")
    customer_daily = customer_daily.copy()
    customer_daily["metric_date"] = pd.to_datetime(customer_daily["metric_date"]).astype("datetime64[ns]")
    customer_daily = customer_daily.sort_values("metric_date")

    merged = pd.merge_asof(
        txns.sort_values("asof_date"),
        customer_daily,
        left_on="asof_date",
        right_on="metric_date",
        by="customer_id",
        direction="backward",
    )
    return merged.sort_values(["customer_id", "transaction_ts"]).reset_index(drop=True)


def _asof_join_merchant_baseline(txns: pd.DataFrame, merchant_daily: pd.DataFrame) -> pd.DataFrame:
    txns = txns.copy()
    merchant_daily = merchant_daily.copy()
    merchant_daily["metric_date"] = pd.to_datetime(merchant_daily["metric_date"]).astype("datetime64[ns]")
    merchant_daily = merchant_daily.sort_values("metric_date")

    merged = pd.merge_asof(
        txns.sort_values("asof_date"),
        merchant_daily,
        left_on="asof_date",
        right_on="metric_date",
        by="merchant_id",
        direction="backward",
        suffixes=("", "_merchant"),
    )
    return merged.sort_values(["customer_id", "transaction_ts"]).reset_index(drop=True)


def build_batch_features(engine: Engine) -> pd.DataFrame:
    txns = load_transactions(engine)
    customer_daily, merchant_daily = _load_daily_metrics(engine)

    df = _asof_join_customer_baseline(txns, customer_daily)
    df = _asof_join_merchant_baseline(df, merchant_daily)

    df["amount_zscore"] = np.where(
        (df["txn_amount_stddev_90d"].notna()) & (df["txn_amount_stddev_90d"] > 0),
        (df["amount"] - df["txn_amount_avg_90d"]) / df["txn_amount_stddev_90d"],
        np.nan,
    )

    grp = df.groupby("customer_id", group_keys=False)
    df["is_failed"] = (df["status"] == "FAILED").astype(int)

    # Velocity: rolling counts over a trailing time window, inclusive of
    # self, then subtract self's own contribution (1) to exclude it -
    # matches features.point_in_time.count_recent_transactions's strict "<".
    indexed = df.set_index("transaction_ts")
    counts_10min = indexed.groupby("customer_id")["transaction_id"].rolling("10min").count()
    counts_60min = indexed.groupby("customer_id")["transaction_id"].rolling("60min").count()
    df["txn_count_last_10min"] = counts_10min.values - 1
    df["txn_count_last_60min"] = counts_60min.values - 1

    failed_rolling = indexed.groupby("customer_id")["is_failed"].rolling("15min").sum()
    df["recent_failed_count_15min"] = failed_rolling.values - df["is_failed"]

    # First-occurrence flags: cumcount()==0 means "never seen before this row".
    df["is_new_device"] = grp.apply(lambda g: g.groupby("device_id").cumcount() == 0, include_groups=False).values
    df["is_new_location"] = grp.apply(lambda g: g.groupby("location_id").cumcount() == 0, include_groups=False).values
    df["is_first_time_at_merchant"] = grp.apply(lambda g: g.groupby("merchant_id").cumcount() == 0, include_groups=False).values

    # Off-hours history: has this customer ever transacted at this hour before,
    # and how many prior transactions do they have at all.
    df["txns_at_this_hour_before"] = grp.apply(lambda g: g.groupby("hour").cumcount(), include_groups=False).values
    df["total_prior_txns"] = grp.cumcount()

    # Previous transaction's location/time, for the geo-impossible rule.
    df["last_lat"] = grp["current_lat"].shift(1)
    df["last_lon"] = grp["current_lon"].shift(1)
    df["last_ts"] = grp["transaction_ts"].shift(1)

    return df.sort_values("transaction_id").reset_index(drop=True)


def _apply_geo_rule(df: pd.DataFrame) -> pd.DataFrame:
    has_prior = df["last_lat"].notna()
    hours = (df["transaction_ts"] - df["last_ts"]).dt.total_seconds() / 3600
    distance_km = pd.Series(np.nan, index=df.index)
    valid = has_prior & (hours > 0)
    distance_km[valid] = [
        geo.haversine_km(a, b, c, d)
        for a, b, c, d in zip(df.loc[valid, "last_lat"], df.loc[valid, "last_lon"], df.loc[valid, "current_lat"], df.loc[valid, "current_lon"])
    ]
    implied_speed = distance_km / hours
    triggered = valid & (distance_km >= config.GEO_MIN_DISTANCE_KM) & (implied_speed >= config.GEO_MAX_PLAUSIBLE_SPEED_KMH)

    out = df.loc[triggered, ["transaction_id"]].copy()
    out["rule_id"] = "R3_GEO_IMPOSSIBLE"
    out["rule_description"] = "Implied travel speed between consecutive transactions exceeds plausible limits"
    out["severity"] = [severity_for_ratio(s, config.GEO_MAX_PLAUSIBLE_SPEED_KMH) for s in implied_speed[triggered]]
    out["evidence"] = [
        {"distance_km": round(dkm, 1), "hours_between_txns": round(h, 2), "implied_speed_kmh": round(sp, 1), "threshold_kmh": config.GEO_MAX_PLAUSIBLE_SPEED_KMH}
        for dkm, h, sp in zip(distance_km[triggered], hours[triggered], implied_speed[triggered])
    ]
    return out


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Runs all 7 rules vectorized and returns a long-format DataFrame of
    firings: [transaction_id, rule_id, rule_description, severity, evidence].
    """
    firings = []

    # R1 velocity
    v10 = df["txn_count_last_10min"] >= config.VELOCITY_10MIN_TRIGGER
    v60 = (~v10) & (df["txn_count_last_60min"] >= config.VELOCITY_60MIN_TRIGGER)
    for mask, col, threshold, key in [
        (v10, "txn_count_last_10min", config.VELOCITY_10MIN_TRIGGER, "txn_count_10min"),
        (v60, "txn_count_last_60min", config.VELOCITY_60MIN_TRIGGER, "txn_count_60min"),
    ]:
        sub = df.loc[mask]
        if len(sub):
            out = sub[["transaction_id"]].copy()
            out["rule_id"] = "R1_VELOCITY"
            out["rule_description"] = "High transaction velocity in a short window"
            out["severity"] = [severity_for_ratio(c, threshold) for c in sub[col]]
            out["evidence"] = [{key: int(c), "threshold": threshold} for c in sub[col]]
            firings.append(out)

    # R2 unusual amount
    has_std = df["amount_zscore"].notna()
    zscore_trigger = has_std & (df["amount_zscore"].abs() >= config.AMOUNT_ZSCORE_TRIGGER)
    sub = df.loc[zscore_trigger]
    if len(sub):
        out = sub[["transaction_id"]].copy()
        out["rule_id"] = "R2_AMOUNT_SPIKE"
        out["rule_description"] = "Transaction amount is a statistical outlier vs. customer baseline"
        out["severity"] = [severity_for_ratio(abs(z), config.AMOUNT_ZSCORE_TRIGGER) for z in sub["amount_zscore"]]
        out["evidence"] = [
            {"amount": float(a), "customer_avg_90d": float(avg) if pd.notna(avg) else None, "zscore": round(float(z), 2), "threshold_zscore": config.AMOUNT_ZSCORE_TRIGGER}
            for a, avg, z in zip(sub["amount"], sub["txn_amount_avg_90d"], sub["amount_zscore"])
        ]
        firings.append(out)

    no_std_but_avg = (~has_std) & df["txn_amount_avg_90d"].notna() & (df["txn_amount_avg_90d"] > 0)
    multiplier = df["amount"] / df["txn_amount_avg_90d"]
    fallback_trigger = no_std_but_avg & (multiplier >= config.AMOUNT_FALLBACK_MULTIPLIER)
    sub = df.loc[fallback_trigger]
    if len(sub):
        out = sub[["transaction_id"]].copy()
        out["rule_id"] = "R2_AMOUNT_SPIKE"
        out["rule_description"] = "Transaction amount is far above customer's limited baseline (insufficient history for z-score)"
        out["severity"] = [severity_for_ratio(m, config.AMOUNT_FALLBACK_MULTIPLIER) for m in multiplier[fallback_trigger]]
        out["evidence"] = [
            {"amount": float(a), "customer_avg_90d": float(avg), "multiplier": round(float(m), 2), "threshold_multiplier": config.AMOUNT_FALLBACK_MULTIPLIER}
            for a, avg, m in zip(sub["amount"], sub["txn_amount_avg_90d"], multiplier[fallback_trigger])
        ]
        firings.append(out)

    # R3 geographic inconsistency
    geo_firings = _apply_geo_rule(df)
    if len(geo_firings):
        firings.append(geo_firings)

    # R4 failed then large
    baseline_ok = df["txn_amount_avg_90d"].notna() & (df["txn_amount_avg_90d"] > 0)
    ft_multiplier = df["amount"] / df["txn_amount_avg_90d"]
    ft_trigger = (
        (df["status"] == "APPROVED")
        & (df["recent_failed_count_15min"] >= config.FAILED_THEN_LARGE_MIN_FAILED)
        & baseline_ok
        & (ft_multiplier >= config.FAILED_THEN_LARGE_AMOUNT_MULTIPLIER)
    )
    sub = df.loc[ft_trigger]
    if len(sub):
        out = sub[["transaction_id"]].copy()
        out["rule_id"] = "R4_FAILED_THEN_LARGE"
        out["rule_description"] = "Approved large transaction preceded by repeated failed attempts"
        out["severity"] = [severity_for_ratio(c, config.FAILED_THEN_LARGE_MIN_FAILED) for c in sub["recent_failed_count_15min"]]
        out["evidence"] = [
            {"recent_failed_count_15min": int(c), "amount": float(a), "customer_avg_90d": float(avg), "multiplier": round(float(m), 2)}
            for c, a, avg, m in zip(sub["recent_failed_count_15min"], sub["amount"], sub["txn_amount_avg_90d"], ft_multiplier[ft_trigger])
        ]
        firings.append(out)

    # R5 new device + new location
    r5_trigger = df["is_new_device"] & df["is_new_location"]
    sub = df.loc[r5_trigger]
    if len(sub):
        out = sub[["transaction_id"]].copy()
        out["rule_id"] = "R5_NEW_DEVICE_NEW_LOCATION"
        out["rule_description"] = "Transaction from a device and location never seen before for this customer"
        out["severity"] = "HIGH"
        out["evidence"] = [{"is_new_device": True, "is_new_location": True}] * len(sub)
        firings.append(out)

    # R6 merchant deviation
    merch_ok = df["avg_txn_amount_90d"].notna() & (df["avg_txn_amount_90d"] > 0)
    merch_multiplier = df["amount"] / df["avg_txn_amount_90d"]
    r6_trigger = merch_ok & (merch_multiplier >= config.MERCHANT_AMOUNT_DEVIATION_MULTIPLIER)
    sub = df.loc[r6_trigger]
    if len(sub):
        out = sub[["transaction_id"]].copy()
        out["rule_id"] = "R6_MERCHANT_DEVIATION"
        out["rule_description"] = "Transaction amount is far above this merchant's own typical transaction size"
        out["severity"] = [severity_for_ratio(m, config.MERCHANT_AMOUNT_DEVIATION_MULTIPLIER) for m in merch_multiplier[r6_trigger]]
        out["evidence"] = [
            {"amount": float(a), "merchant_avg_90d": float(avg), "multiplier": round(float(m), 2)}
            for a, avg, m in zip(sub["amount"], sub["avg_txn_amount_90d"], merch_multiplier[r6_trigger])
        ]
        firings.append(out)

    # R7 off-hours
    r7_trigger = (df["total_prior_txns"] >= config.OFF_HOURS_MIN_PRIOR_TXNS) & (df["txns_at_this_hour_before"] == 0)
    sub = df.loc[r7_trigger]
    if len(sub):
        out = sub[["transaction_id"]].copy()
        out["rule_id"] = "R7_OFF_HOURS"
        out["rule_description"] = "Transaction occurs at an hour of day this customer has never transacted at before"
        out["severity"] = "MEDIUM"
        out["evidence"] = [
            {"transaction_hour": int(h), "txns_at_this_hour_historically": 0, "total_prior_transactions": int(t)}
            for h, t in zip(sub["hour"], sub["total_prior_txns"])
        ]
        firings.append(out)

    if not firings:
        return pd.DataFrame(columns=["transaction_id", "rule_id", "rule_description", "severity", "evidence"])
    return pd.concat(firings, ignore_index=True)
