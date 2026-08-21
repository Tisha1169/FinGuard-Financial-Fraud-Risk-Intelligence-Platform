"""Orchestrates the full synthetic dataset build: entities -> behavior
profiles -> normal transactions -> fraud injection -> assembled tables
matching the Postgres schema in sql/schema.sql.
"""
import random

import numpy as np
import pandas as pd
from faker import Faker

from data_generation import behavior, config, entities, fraud_typologies, normal_transactions


def generate_dataset(seed: int = config.SEED) -> dict:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    Faker.seed(seed)

    locations = entities.generate_locations()
    customers = entities.generate_customers(rng, np_rng, locations)
    merchants = entities.generate_merchants(rng, locations)
    devices, customer_device_map = entities.generate_devices(rng, np_rng, customers)
    payment_instruments = entities.generate_payment_instruments(rng, customers)

    profiles = behavior.build_customer_profiles(
        rng, np_rng, customers, merchants, customer_device_map, payment_instruments
    )

    normal_txns = normal_transactions.generate_normal_transactions(rng, np_rng, customers, profiles)

    next_device_id = int(devices["device_id"].max()) + 1
    fraud_txns, extra_devices = fraud_typologies.inject_all(
        rng, np_rng, customers, merchants, locations, profiles,
        baseline_txn_count=len(normal_txns), start_device_id=next_device_id,
    )

    if len(extra_devices):
        devices = pd.concat([devices, extra_devices], ignore_index=True)

    all_txns = pd.concat([normal_txns, fraud_txns], ignore_index=True)
    all_txns = all_txns.sort_values("transaction_ts").reset_index(drop=True)
    all_txns.insert(0, "transaction_id", range(1, len(all_txns) + 1))

    ground_truth = all_txns[["transaction_id", "is_fraud", "fraud_typology"]].copy()
    ground_truth["is_synthetic_label"] = True

    fact_transactions = all_txns.drop(columns=["is_fraud", "fraud_typology"])

    return {
        "dim_location": locations,
        "dim_customer": customers.drop(columns=["home_location_id"]),
        "dim_merchant": merchants.drop(columns=["merchant_location_id"]),
        "dim_device": devices,
        "fact_transactions": fact_transactions,
        "ground_truth_fraud": ground_truth,
        # kept for the loader to resolve merchant/customer home locations if needed later
        "_customers_with_location": customers,
        "_merchants_with_location": merchants,
    }


def summarize(dataset: dict) -> dict:
    txns = dataset["fact_transactions"]
    gt = dataset["ground_truth_fraud"]
    fraud = gt[gt["is_fraud"]]
    return {
        "customers": len(dataset["dim_customer"]),
        "merchants": len(dataset["dim_merchant"]),
        "devices": len(dataset["dim_device"]),
        "locations": len(dataset["dim_location"]),
        "transactions": len(txns),
        "fraud_transactions": len(fraud),
        "fraud_rate": round(len(fraud) / len(txns), 4) if len(txns) else 0,
        "typology_counts": fraud["fraud_typology"].value_counts().to_dict(),
        "date_range": [str(txns["transaction_ts"].min()), str(txns["transaction_ts"].max())],
    }
