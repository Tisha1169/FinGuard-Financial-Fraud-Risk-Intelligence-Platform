"""Generates the dimension entities: customers, merchants, devices, locations."""
import random
import uuid

import numpy as np
import pandas as pd
from faker import Faker

from data_generation import config

fake = Faker()


def generate_locations() -> pd.DataFrame:
    cities = config.CITIES[: config.NUM_LOCATIONS]
    rows = [
        {
            "location_id": i + 1,
            "city": city,
            "country": country,
            "latitude": lat,
            "longitude": lon,
        }
        for i, (city, country, lat, lon) in enumerate(cities)
    ]
    return pd.DataFrame(rows)


def generate_customers(rng: random.Random, np_rng: np.random.Generator, locations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(config.NUM_CUSTOMERS):
        home = locations.sample(random_state=rng.randint(0, 2**31 - 1)).iloc[0]
        signup_offset_days = int(np_rng.integers(30, 900))
        risk_roll = rng.random()
        if risk_roll < 0.03:
            segment = "WATCHLIST"
        elif risk_roll < 0.10:
            segment = "VIP"
        else:
            segment = "STANDARD"
        rows.append(
            {
                "customer_id": i + 1,
                "customer_uid": f"CUST-{i + 1:06d}",
                "signup_date": config.START_DATE - pd.Timedelta(days=signup_offset_days),
                "home_country": home["country"],
                "home_city": home["city"],
                "home_location_id": int(home["location_id"]),
                "risk_segment": segment,
                "is_synthetic": True,
            }
        )
    return pd.DataFrame(rows)


def generate_merchants(rng: random.Random, locations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(config.NUM_MERCHANTS):
        mcc_code, mcc_desc = rng.choice(config.MCC_CATEGORIES)
        loc = locations.sample(random_state=rng.randint(0, 2**31 - 1)).iloc[0]
        risk_category = "HIGH_RISK" if rng.random() < 0.08 else "STANDARD"
        rows.append(
            {
                "merchant_id": i + 1,
                "merchant_uid": f"MER-{i + 1:06d}",
                "merchant_name": f"{fake.company()} {mcc_desc.title().replace('_', ' ')}",
                "mcc_code": mcc_code,
                "mcc_description": mcc_desc,
                "merchant_country": loc["country"],
                "merchant_location_id": int(loc["location_id"]),
                "risk_category": risk_category,
                "is_synthetic": True,
            }
        )
    return pd.DataFrame(rows)


def generate_devices(rng: random.Random, np_rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    """Devices are generated per-customer but stored as their own dimension;
    the link to a customer is only observed via fact_transactions.device_id,
    since a device being reused across customers is itself a fraud signal.
    """
    rows = []
    device_id = 1
    customer_device_map = {}
    device_types = ["MOBILE", "DESKTOP", "POS", "ATM"]
    os_by_type = {
        "MOBILE": ["iOS", "Android"],
        "DESKTOP": ["Windows", "macOS", "Linux"],
        "POS": ["EmbeddedOS"],
        "ATM": ["EmbeddedOS"],
    }

    for _, cust in customers.iterrows():
        n_devices = max(1, int(np_rng.poisson(config.DEVICES_PER_CUSTOMER_MEAN)))
        cust_devices = []
        for _ in range(n_devices):
            dtype = rng.choice(device_types[:2])  # customers mostly use mobile/desktop
            rows.append(
                {
                    "device_id": device_id,
                    "device_uid": f"DEV-{device_id:06d}",
                    "device_type": dtype,
                    "os": rng.choice(os_by_type[dtype]),
                }
            )
            cust_devices.append(device_id)
            device_id += 1
        customer_device_map[cust["customer_id"]] = cust_devices

    return pd.DataFrame(rows), customer_device_map


def generate_payment_instruments(rng: random.Random, customers: pd.DataFrame) -> dict:
    """Each customer gets 1-2 card last-4s. Only last 4 digits are ever
    stored, matching PCI-DSS practice - no full PAN anywhere in this system.
    """
    result = {}
    for _, cust in customers.iterrows():
        n_cards = 1 if rng.random() < 0.75 else 2
        result[cust["customer_id"]] = [f"{rng.randint(0, 9999):04d}" for _ in range(n_cards)]
    return result


def new_transaction_uid() -> str:
    return str(uuid.uuid4())
