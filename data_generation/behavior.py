"""Builds a per-customer baseline behavior profile that both the normal
transaction generator and the fraud injector use - the fraud injector's job
is to construct transactions that deviate from this profile, mirroring how
real behavioral fraud signals work.
"""
import random

import numpy as np

from data_generation import config


def build_customer_profiles(
    rng: random.Random,
    np_rng: np.random.Generator,
    customers,
    merchants,
    customer_device_map: dict,
    payment_instruments: dict,
) -> dict:
    profiles = {}
    merchant_ids = merchants["merchant_id"].tolist()

    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        # Typical spend: lognormal so most transactions are small with an
        # occasional larger one - this is the baseline the "unusual amount"
        # typology deviates from.
        amount_mu = np_rng.uniform(2.8, 4.2)      # ln($) -> ~ $16-$67 median
        amount_sigma = np_rng.uniform(0.4, 0.9)

        # Each customer frequents a small, stable pool of merchants.
        pool_size = max(3, int(np_rng.poisson(8)))
        usual_merchants = rng.sample(merchant_ids, min(pool_size, len(merchant_ids)))

        # Active hours: most customers transact in a consistent daily window.
        active_start = int(np_rng.integers(6, 11))
        active_end = int(np_rng.integers(18, 23))

        # Transaction frequency: expected transactions per day.
        txn_rate_per_day = np_rng.uniform(0.3, 1.4)

        profiles[cid] = {
            "amount_mu": amount_mu,
            "amount_sigma": amount_sigma,
            "usual_merchants": usual_merchants,
            "active_start_hour": active_start,
            "active_end_hour": active_end,
            "txn_rate_per_day": txn_rate_per_day,
            "devices": customer_device_map[cid],
            "primary_device": customer_device_map[cid][0],
            "home_location_id": cust["home_location_id"],
            "cards": payment_instruments[cid],
        }
    return profiles
