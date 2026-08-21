"""Injects labeled fraud typologies into the transaction stream.

Each function constructs a small burst of transactions that deviates from
the target customer's baseline behavior profile in a specific, realistic
way. Every generated row carries is_fraud=True and a fraud_typology label -
this is the ground truth the rules engine, statistical detector, and ML
model are all evaluated against, and it is explicitly synthetic (see
docs/model_card.md).
"""
import datetime
import random

import numpy as np
import pandas as pd

from data_generation import config, entities, geo

# Approximate transactions produced per fraud "event", used to size how many
# events per typology are needed to hit config.TARGET_FRAUD_TXN_SHARE.
AVG_TXNS_PER_EVENT = {
    "CARD_TESTING": 7,
    "VELOCITY_ABUSE": 9,
    "ACCOUNT_TAKEOVER": 4,
    "UNUSUAL_AMOUNT": 1,
    "GEOGRAPHIC_INCONSISTENCY": 2,
    "FAILED_THEN_LARGE": 4,
    "MERCHANT_BEHAVIOR_DEVIATION": 1,
    "UNUSUAL_TIME_OF_DAY": 1,
    "DEVICE_ANOMALY": 1,
}


def _row(cid, merchant_id, device_id, location_id, ts, amount, channel, card, status, typology):
    return {
        "transaction_uid": entities.new_transaction_uid(),
        "customer_id": cid,
        "merchant_id": merchant_id,
        "device_id": device_id,
        "location_id": location_id,
        "transaction_ts": ts,
        "amount": round(float(amount), 2),
        "currency": "USD",
        "channel": channel,
        "payment_instrument_last4": card,
        "status": status,
        "is_fraud": True,
        "fraud_typology": typology,
    }


def _random_event_dt(rng: random.Random) -> datetime.datetime:
    day_offset = rng.randint(0, (config.END_DATE - config.START_DATE).days - 1)
    day = config.START_DATE + datetime.timedelta(days=day_offset)
    return datetime.datetime.combine(day, datetime.time(rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)))


def card_testing(rng, cid, profile, merchant_ids):
    rows = []
    start = _random_event_dt(rng)
    n = rng.randint(5, 9)
    card = rng.choice(profile["cards"])
    for i in range(n):
        ts = start + datetime.timedelta(seconds=rng.randint(15, 90) * (i + 1))
        merchant = rng.choice(merchant_ids)
        amount = round(rng.uniform(0.5, 5.0), 2)
        status = "APPROVED" if rng.random() < 0.2 else rng.choice(["DECLINED", "FAILED"])
        rows.append(
            _row(cid, merchant, profile["primary_device"], profile["home_location_id"], ts, amount, "ECOM", card, status, "CARD_TESTING")
        )
    return rows


def velocity_abuse(rng, np_rng, cid, profile, merchant_ids):
    rows = []
    start = _random_event_dt(rng)
    n = rng.randint(6, 12)
    card = rng.choice(profile["cards"])
    for i in range(n):
        ts = start + datetime.timedelta(minutes=rng.randint(1, 6) * (i + 1))
        merchant = rng.choice(merchant_ids)
        amount = round(float(np_rng.lognormal(profile["amount_mu"], profile["amount_sigma"])), 2)
        rows.append(
            _row(cid, merchant, profile["primary_device"], profile["home_location_id"], ts, amount, "ECOM", card, "APPROVED", "VELOCITY_ABUSE")
        )
    return rows


def account_takeover(rng, np_rng, cid, profile, merchant_ids, locations_df, next_device_id):
    rows = []
    start = _random_event_dt(rng)
    n = rng.randint(3, 6)
    card = rng.choice(profile["cards"])
    # A device never seen for this customer before, from a location far from home.
    fraud_device_id = next_device_id
    home_loc = locations_df.loc[locations_df["location_id"] == profile["home_location_id"]].iloc[0]
    other_locs = locations_df[locations_df["country"] != home_loc["country"]]
    new_loc = other_locs.sample(random_state=rng.randint(0, 2**31 - 1)).iloc[0] if len(other_locs) else home_loc

    for i in range(n):
        ts = start + datetime.timedelta(minutes=rng.randint(20, 240) * (i + 1))
        merchant = rng.choice(merchant_ids)
        # amounts escalate - typical of an attacker probing then cashing out
        amount = round(float(np_rng.lognormal(profile["amount_mu"] + 1.2, profile["amount_sigma"])), 2)
        rows.append(
            _row(cid, merchant, fraud_device_id, int(new_loc["location_id"]), ts, amount, "ECOM", card, "APPROVED", "ACCOUNT_TAKEOVER")
        )
    return rows, fraud_device_id


def unusual_amount(rng, cid, profile, merchant_ids):
    ts = _random_event_dt(rng)
    merchant = rng.choice(profile["usual_merchants"] + merchant_ids[:5])
    baseline = np.exp(profile["amount_mu"])
    multiplier = rng.uniform(5, 15)
    amount = baseline * multiplier
    card = rng.choice(profile["cards"])
    return [
        _row(cid, merchant, profile["primary_device"], profile["home_location_id"], ts, amount, "ECOM", card, "APPROVED", "UNUSUAL_AMOUNT")
    ]


def geographic_inconsistency(rng, cid, profile, merchant_ids, locations_df):
    home_loc = locations_df.loc[locations_df["location_id"] == profile["home_location_id"]].iloc[0]
    far_locs = locations_df[locations_df["location_id"] != profile["home_location_id"]]
    candidates = []
    for _, loc in far_locs.iterrows():
        dist = geo.haversine_km(home_loc["latitude"], home_loc["longitude"], loc["latitude"], loc["longitude"])
        if dist > 3000:  # far enough that <2h travel is physically impossible
            candidates.append(loc)
    if not candidates:
        return []
    far_loc = rng.choice(candidates)

    ts1 = _random_event_dt(rng)
    gap_minutes = rng.randint(20, 90)
    ts2 = ts1 + datetime.timedelta(minutes=gap_minutes)
    card = rng.choice(profile["cards"])
    m1, m2 = rng.choice(merchant_ids), rng.choice(merchant_ids)
    return [
        _row(cid, m1, profile["primary_device"], profile["home_location_id"], ts1, rng.uniform(20, 150), "CARD_PRESENT", card, "APPROVED", "GEOGRAPHIC_INCONSISTENCY"),
        _row(cid, m2, profile["primary_device"], int(far_loc["location_id"]), ts2, rng.uniform(20, 150), "CARD_PRESENT", card, "APPROVED", "GEOGRAPHIC_INCONSISTENCY"),
    ]


def failed_then_large(rng, cid, profile, merchant_ids):
    rows = []
    start = _random_event_dt(rng)
    n_fail = rng.randint(3, 5)
    card = rng.choice(profile["cards"])
    merchant = rng.choice(merchant_ids)
    for i in range(n_fail):
        ts = start + datetime.timedelta(seconds=rng.randint(20, 60) * (i + 1))
        amount = round(rng.uniform(50, 400), 2)
        rows.append(
            _row(cid, merchant, profile["primary_device"], profile["home_location_id"], ts, amount, "ECOM", card, "FAILED", "FAILED_THEN_LARGE")
        )
    ts_large = rows[-1]["transaction_ts"] + datetime.timedelta(seconds=rng.randint(30, 120))
    baseline = np.exp(profile["amount_mu"])
    amount_large = baseline * rng.uniform(6, 12)
    rows.append(
        _row(cid, merchant, profile["primary_device"], profile["home_location_id"], ts_large, amount_large, "ECOM", card, "APPROVED", "FAILED_THEN_LARGE")
    )
    return rows


def merchant_behavior_deviation(rng, cid, profile, merchants_df):
    high_risk = merchants_df[merchants_df["risk_category"] == "HIGH_RISK"]
    outside_pool = merchants_df[~merchants_df["merchant_id"].isin(profile["usual_merchants"])]
    pool = high_risk if len(high_risk) else outside_pool
    merchant = pool.sample(random_state=rng.randint(0, 2**31 - 1)).iloc[0]
    ts = _random_event_dt(rng)
    baseline = np.exp(profile["amount_mu"])
    amount = baseline * rng.uniform(3, 8)
    card = rng.choice(profile["cards"])
    return [
        _row(cid, int(merchant["merchant_id"]), profile["primary_device"], profile["home_location_id"], ts, amount, "ECOM", card, "APPROVED", "MERCHANT_BEHAVIOR_DEVIATION")
    ]


def unusual_time_of_day(rng, np_rng, cid, profile, merchant_ids):
    day_offset = rng.randint(0, (config.END_DATE - config.START_DATE).days - 1)
    day = config.START_DATE + datetime.timedelta(days=day_offset)
    # pick an hour clearly outside the customer's usual active window
    off_hours = [h for h in range(24) if h < profile["active_start_hour"] - 2 or h > profile["active_end_hour"] + 2]
    hour = rng.choice(off_hours) if off_hours else 3
    ts = datetime.datetime.combine(day, datetime.time(hour, rng.randint(0, 59), rng.randint(0, 59)))
    merchant = rng.choice(profile["usual_merchants"])
    amount = float(np_rng.lognormal(profile["amount_mu"], profile["amount_sigma"]))
    card = rng.choice(profile["cards"])
    return [
        _row(cid, merchant, profile["primary_device"], profile["home_location_id"], ts, amount, "ECOM", card, "APPROVED", "UNUSUAL_TIME_OF_DAY")
    ]


def device_anomaly(rng, np_rng, cid, profile, merchant_ids, next_device_id):
    ts = _random_event_dt(rng)
    merchant = rng.choice(merchant_ids)
    amount = float(np_rng.lognormal(profile["amount_mu"] + 0.5, profile["amount_sigma"]))
    card = rng.choice(profile["cards"])
    row = _row(cid, merchant, next_device_id, profile["home_location_id"], ts, amount, "ECOM", card, "APPROVED", "DEVICE_ANOMALY")
    return [row], next_device_id


def inject_all(rng, np_rng, customers, merchants, locations, profiles, baseline_txn_count, start_device_id):
    """Returns (fraud_transactions_df, extra_devices_df)."""
    target_fraud_txns = baseline_txn_count * config.TARGET_FRAUD_TXN_SHARE
    events_per_typology = {
        typ: max(1, round((target_fraud_txns / len(config.FRAUD_TYPOLOGIES)) / avg))
        for typ, avg in AVG_TXNS_PER_EVENT.items()
    }

    customer_ids = customers["customer_id"].tolist()
    merchant_ids = merchants["merchant_id"].tolist()

    all_rows = []
    extra_devices = []
    next_device_id = start_device_id

    for typology, n_events in events_per_typology.items():
        for _ in range(n_events):
            cid = rng.choice(customer_ids)
            profile = profiles[cid]

            if typology == "CARD_TESTING":
                all_rows.extend(card_testing(rng, cid, profile, merchant_ids))
            elif typology == "VELOCITY_ABUSE":
                all_rows.extend(velocity_abuse(rng, np_rng, cid, profile, merchant_ids))
            elif typology == "ACCOUNT_TAKEOVER":
                rows, used_device_id = account_takeover(rng, np_rng, cid, profile, merchant_ids, locations, next_device_id)
                all_rows.extend(rows)
                extra_devices.append(
                    {"device_id": used_device_id, "device_uid": f"DEV-{used_device_id:06d}", "device_type": "MOBILE", "os": "Android"}
                )
                next_device_id += 1
            elif typology == "UNUSUAL_AMOUNT":
                all_rows.extend(unusual_amount(rng, cid, profile, merchant_ids))
            elif typology == "GEOGRAPHIC_INCONSISTENCY":
                all_rows.extend(geographic_inconsistency(rng, cid, profile, merchant_ids, locations))
            elif typology == "FAILED_THEN_LARGE":
                all_rows.extend(failed_then_large(rng, cid, profile, merchant_ids))
            elif typology == "MERCHANT_BEHAVIOR_DEVIATION":
                all_rows.extend(merchant_behavior_deviation(rng, cid, profile, merchants))
            elif typology == "UNUSUAL_TIME_OF_DAY":
                all_rows.extend(unusual_time_of_day(rng, np_rng, cid, profile, merchant_ids))
            elif typology == "DEVICE_ANOMALY":
                rows, used_device_id = device_anomaly(rng, np_rng, cid, profile, merchant_ids, next_device_id)
                all_rows.extend(rows)
                extra_devices.append(
                    {"device_id": used_device_id, "device_uid": f"DEV-{used_device_id:06d}", "device_type": "MOBILE", "os": "Android"}
                )
                next_device_id += 1

    fraud_df = pd.DataFrame(all_rows)
    devices_df = pd.DataFrame(extra_devices)
    return fraud_df, devices_df
