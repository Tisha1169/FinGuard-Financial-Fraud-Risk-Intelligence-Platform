"""Single-transaction rule engine: given one transaction and its
point-in-time features (from features/point_in_time.py), evaluates all 7
rules and returns the ones that fired.

Used by the live scoring path (the FastAPI /risk/{transaction_id} endpoint,
Phase 9) where scoring one transaction with a handful of small queries is
fine. Bulk backfill scoring instead uses rules/batch.py, a vectorized
implementation of the same logic - see that module's docstring for why
they're kept separate.

Every rule function returns None if it didn't fire, or a dict with
{rule_id, rule_description, severity, evidence} if it did - never a bare
boolean, per the spec: investigators need the reason and the evidence, not
just a flag.
"""
from data_generation import geo
from rules import config
from rules.severity import severity_for_ratio


def rule_velocity(transaction: dict, features: dict) -> dict | None:
    count_10 = features["txn_count_last_10min"]
    count_60 = features["txn_count_last_60min"]
    if count_10 >= config.VELOCITY_10MIN_TRIGGER:
        return {
            "rule_id": "R1_VELOCITY",
            "rule_description": "High transaction velocity in a short window",
            "severity": severity_for_ratio(count_10, config.VELOCITY_10MIN_TRIGGER),
            "evidence": {"txn_count_10min": count_10, "threshold": config.VELOCITY_10MIN_TRIGGER},
        }
    if count_60 >= config.VELOCITY_60MIN_TRIGGER:
        return {
            "rule_id": "R1_VELOCITY",
            "rule_description": "High transaction velocity in a short window",
            "severity": severity_for_ratio(count_60, config.VELOCITY_60MIN_TRIGGER),
            "evidence": {"txn_count_60min": count_60, "threshold": config.VELOCITY_60MIN_TRIGGER},
        }
    return None


def rule_unusual_amount(transaction: dict, features: dict) -> dict | None:
    zscore = features["amount_zscore"]
    if zscore is not None:
        if abs(zscore) >= config.AMOUNT_ZSCORE_TRIGGER:
            return {
                "rule_id": "R2_AMOUNT_SPIKE",
                "rule_description": "Transaction amount is a statistical outlier vs. customer baseline",
                "severity": severity_for_ratio(abs(zscore), config.AMOUNT_ZSCORE_TRIGGER),
                "evidence": {
                    "amount": transaction["amount"],
                    "customer_avg_90d": features["customer_baseline_avg_90d"],
                    "zscore": round(zscore, 2),
                    "threshold_zscore": config.AMOUNT_ZSCORE_TRIGGER,
                },
            }
        return None
    # no stddev available yet (new/thin customer history) - fall back to a
    # raw multiple of the average, if we at least have an average.
    baseline_avg = features["customer_baseline_avg_90d"]
    if baseline_avg and baseline_avg > 0:
        multiplier = transaction["amount"] / baseline_avg
        if multiplier >= config.AMOUNT_FALLBACK_MULTIPLIER:
            return {
                "rule_id": "R2_AMOUNT_SPIKE",
                "rule_description": "Transaction amount is far above customer's limited baseline (insufficient history for z-score)",
                "severity": severity_for_ratio(multiplier, config.AMOUNT_FALLBACK_MULTIPLIER),
                "evidence": {
                    "amount": transaction["amount"],
                    "customer_avg_90d": baseline_avg,
                    "multiplier": round(multiplier, 2),
                    "threshold_multiplier": config.AMOUNT_FALLBACK_MULTIPLIER,
                },
            }
    return None


def rule_geographic_inconsistency(transaction: dict, features: dict) -> dict | None:
    if features["last_location_lat"] is None or features["last_txn_ts_any"] is None:
        return None
    if features["current_location_lat"] is None:
        return None

    hours = (transaction["transaction_ts"] - features["last_txn_ts_any"]).total_seconds() / 3600
    if hours <= 0:
        return None

    distance_km = geo.haversine_km(
        features["last_location_lat"], features["last_location_lon"],
        features["current_location_lat"], features["current_location_lon"],
    )
    if distance_km < config.GEO_MIN_DISTANCE_KM:
        return None

    implied_speed = distance_km / hours
    if implied_speed >= config.GEO_MAX_PLAUSIBLE_SPEED_KMH:
        return {
            "rule_id": "R3_GEO_IMPOSSIBLE",
            "rule_description": "Implied travel speed between consecutive transactions exceeds plausible limits",
            "severity": severity_for_ratio(implied_speed, config.GEO_MAX_PLAUSIBLE_SPEED_KMH),
            "evidence": {
                "distance_km": round(distance_km, 1),
                "hours_between_txns": round(hours, 2),
                "implied_speed_kmh": round(implied_speed, 1),
                "threshold_kmh": config.GEO_MAX_PLAUSIBLE_SPEED_KMH,
            },
        }
    return None


def rule_failed_then_large(transaction: dict, features: dict) -> dict | None:
    if transaction["status"] != "APPROVED":
        return None
    failed_count = features["recent_failed_count_15min"]
    if failed_count < config.FAILED_THEN_LARGE_MIN_FAILED:
        return None
    baseline_avg = features["customer_baseline_avg_90d"]
    if not baseline_avg or baseline_avg <= 0:
        return None
    multiplier = transaction["amount"] / baseline_avg
    if multiplier >= config.FAILED_THEN_LARGE_AMOUNT_MULTIPLIER:
        return {
            "rule_id": "R4_FAILED_THEN_LARGE",
            "rule_description": "Approved large transaction preceded by repeated failed attempts",
            "severity": severity_for_ratio(failed_count, config.FAILED_THEN_LARGE_MIN_FAILED),
            "evidence": {
                "recent_failed_count_15min": failed_count,
                "amount": transaction["amount"],
                "customer_avg_90d": baseline_avg,
                "multiplier": round(multiplier, 2),
            },
        }
    return None


def rule_new_device_new_location(transaction: dict, features: dict) -> dict | None:
    if features["is_new_device"] and features["is_new_location"]:
        return {
            "rule_id": "R5_NEW_DEVICE_NEW_LOCATION",
            "rule_description": "Transaction from a device and location never seen before for this customer",
            "severity": "HIGH",
            "evidence": {"is_new_device": True, "is_new_location": True},
        }
    return None


def rule_merchant_deviation(transaction: dict, features: dict) -> dict | None:
    merchant_avg = features["merchant_avg_amount_90d"]
    if merchant_avg and merchant_avg > 0:
        multiplier = transaction["amount"] / merchant_avg
        if multiplier >= config.MERCHANT_AMOUNT_DEVIATION_MULTIPLIER:
            return {
                "rule_id": "R6_MERCHANT_DEVIATION",
                "rule_description": "Transaction amount is far above this merchant's own typical transaction size",
                "severity": severity_for_ratio(multiplier, config.MERCHANT_AMOUNT_DEVIATION_MULTIPLIER),
                "evidence": {
                    "amount": transaction["amount"],
                    "merchant_avg_90d": merchant_avg,
                    "multiplier": round(multiplier, 2),
                },
            }
    return None


def rule_off_hours(transaction: dict, features: dict) -> dict | None:
    total_prior = features["total_prior_txns"]
    if total_prior < config.OFF_HOURS_MIN_PRIOR_TXNS:
        return None
    if features["txns_at_this_hour"] == 0:
        return {
            "rule_id": "R7_OFF_HOURS",
            "rule_description": "Transaction occurs at an hour of day this customer has never transacted at before",
            "severity": "MEDIUM",
            "evidence": {
                "transaction_hour": transaction["transaction_ts"].hour,
                "txns_at_this_hour_historically": 0,
                "total_prior_transactions": total_prior,
            },
        }
    return None


ALL_RULES = [
    rule_velocity,
    rule_unusual_amount,
    rule_geographic_inconsistency,
    rule_failed_then_large,
    rule_new_device_new_location,
    rule_merchant_deviation,
    rule_off_hours,
]


def evaluate_transaction(transaction: dict, features: dict) -> list[dict]:
    """Runs all rules against one transaction, returns only the ones that fired."""
    # `amount` may arrive as decimal.Decimal when read directly via
    # SQLAlchemy from a NUMERIC column - normalize once here so every rule
    # can safely do float arithmetic against it.
    transaction = {**transaction, "amount": float(transaction["amount"])}
    results = [rule(transaction, features) for rule in ALL_RULES]
    return [r for r in results if r is not None]
