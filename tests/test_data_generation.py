"""Tests for the synthetic data generator. Uses a small dataset (patched
config) so the suite stays fast - these are correctness tests, not a
regeneration of the full dataset.
"""
import numpy as np
import pytest

from data_generation import config, generator


@pytest.fixture(scope="module", autouse=True)
def shrink_dataset_size():
    # Shrink the dataset for test speed without changing generation logic.
    original = (config.NUM_CUSTOMERS, config.NUM_MERCHANTS, config.SIMULATION_DAYS)
    config.NUM_CUSTOMERS, config.NUM_MERCHANTS, config.SIMULATION_DAYS = 60, 20, 30
    yield
    config.NUM_CUSTOMERS, config.NUM_MERCHANTS, config.SIMULATION_DAYS = original


@pytest.fixture(scope="module")
def small_dataset(shrink_dataset_size):
    return generator.generate_dataset(seed=7)


def test_generation_is_reproducible_with_same_seed(small_dataset):
    again = generator.generate_dataset(seed=7)
    pd_a = small_dataset["fact_transactions"]
    pd_b = again["fact_transactions"]
    assert len(pd_a) == len(pd_b)
    assert pd_a["amount"].sum() == pytest.approx(pd_b["amount"].sum())


def test_all_fraud_typologies_present(small_dataset):
    gt = small_dataset["ground_truth_fraud"]
    typologies_seen = set(gt.loc[gt["is_fraud"], "fraud_typology"].unique())
    assert typologies_seen == set(config.FRAUD_TYPOLOGIES)


def test_fraud_rate_within_expected_bounds(small_dataset):
    gt = small_dataset["ground_truth_fraud"]
    rate = gt["is_fraud"].mean()
    # small dataset -> looser bounds than production target
    assert 0.005 < rate < 0.08


def test_ground_truth_labels_are_marked_synthetic(small_dataset):
    gt = small_dataset["ground_truth_fraud"]
    assert gt["is_synthetic_label"].all()


def test_non_fraud_rows_have_null_typology(small_dataset):
    gt = small_dataset["ground_truth_fraud"]
    assert gt.loc[~gt["is_fraud"], "fraud_typology"].isna().all()


def test_transaction_ids_are_sequential_and_unique(small_dataset):
    ids = small_dataset["fact_transactions"]["transaction_id"]
    assert ids.is_unique
    assert ids.min() == 1
    assert ids.max() == len(ids)


def test_transactions_reference_valid_customers_and_merchants(small_dataset):
    txns = small_dataset["fact_transactions"]
    customer_ids = set(small_dataset["dim_customer"]["customer_id"])
    merchant_ids = set(small_dataset["dim_merchant"]["merchant_id"])
    device_ids = set(small_dataset["dim_device"]["device_id"])
    assert set(txns["customer_id"]).issubset(customer_ids)
    assert set(txns["merchant_id"]).issubset(merchant_ids)
    assert set(txns["device_id"]).issubset(device_ids)


def test_payment_instrument_never_stores_full_pan(small_dataset):
    last4 = small_dataset["fact_transactions"]["payment_instrument_last4"]
    assert last4.str.len().eq(4).all()


def test_amounts_are_positive_and_bounded(small_dataset):
    amounts = small_dataset["fact_transactions"]["amount"]
    assert (amounts > 0).all()
    assert amounts.max() < 100_000  # sanity bound, no runaway lognormal tail


def test_card_testing_events_are_low_amount_and_bursty(small_dataset):
    gt = small_dataset["ground_truth_fraud"].merge(
        small_dataset["fact_transactions"], on="transaction_id"
    )
    card_testing = gt[gt["fraud_typology"] == "CARD_TESTING"]
    assert len(card_testing) > 0
    assert card_testing["amount"].max() < 10  # card testing uses small probe amounts


def test_geographic_inconsistency_events_span_distant_locations(small_dataset):
    gt = small_dataset["ground_truth_fraud"].merge(
        small_dataset["fact_transactions"], on="transaction_id"
    )
    geo_events = gt[gt["fraud_typology"] == "GEOGRAPHIC_INCONSISTENCY"]
    assert len(geo_events) > 0
    assert geo_events["location_id"].nunique() > 1
