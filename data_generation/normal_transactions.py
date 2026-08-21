"""Generates legitimate (non-fraud) baseline transactions for every
customer across the simulation window, driven by each customer's behavior
profile from behavior.py.
"""
import datetime
import random

import numpy as np
import pandas as pd

from data_generation import config, entities


def _random_timestamp_on_day(rng: random.Random, day: datetime.date, profile: dict) -> datetime.datetime:
    if rng.random() < 0.92:
        hour = rng.randint(profile["active_start_hour"], profile["active_end_hour"])
    else:
        hour = rng.randint(0, 23)  # occasional off-hours txn is normal too
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return datetime.datetime.combine(day, datetime.time(hour, minute, second))


def generate_normal_transactions(
    rng: random.Random,
    np_rng: np.random.Generator,
    customers: pd.DataFrame,
    profiles: dict,
) -> pd.DataFrame:
    rows = []
    n_days = (config.END_DATE - config.START_DATE).days

    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        profile = profiles[cid]

        for day_offset in range(n_days):
            day = config.START_DATE + datetime.timedelta(days=day_offset)
            n_txns_today = np_rng.poisson(profile["txn_rate_per_day"])
            for _ in range(n_txns_today):
                merchant_id = (
                    rng.choice(profile["usual_merchants"])
                    if rng.random() < 0.85
                    else rng.randint(1, config.NUM_MERCHANTS)
                )
                amount = float(np.round(np_rng.lognormal(profile["amount_mu"], profile["amount_sigma"]), 2))
                amount = min(amount, 5000.0)  # cap extreme lognormal tail

                device_id = (
                    profile["primary_device"]
                    if rng.random() < 0.9
                    else rng.choice(profile["devices"])
                )
                location_id = (
                    profile["home_location_id"] if rng.random() < 0.93 else rng.randint(1, config.NUM_LOCATIONS)
                )
                status = "APPROVED" if rng.random() < 0.97 else rng.choice(["DECLINED", "FAILED"])

                rows.append(
                    {
                        "transaction_uid": entities.new_transaction_uid(),
                        "customer_id": cid,
                        "merchant_id": merchant_id,
                        "device_id": device_id,
                        "location_id": location_id,
                        "transaction_ts": _random_timestamp_on_day(rng, day, profile),
                        "amount": amount,
                        "currency": "USD",
                        "channel": rng.choices(config.CHANNELS, weights=[35, 30, 20, 5, 10])[0],
                        "payment_instrument_last4": rng.choice(profile["cards"]),
                        "status": status,
                        "is_fraud": False,
                        "fraud_typology": None,
                    }
                )
    return pd.DataFrame(rows)
