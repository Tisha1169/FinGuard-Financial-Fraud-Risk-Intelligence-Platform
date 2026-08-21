"""Tests for the Phase 6 supervised ML pipeline.

Unit tests (no DB) cover the metric helpers and preprocessing logic with
hand-built data. Integration tests (skipped without a live DATABASE_URL)
validate the real pipeline: chronological split integrity, train-only
fitting of every learned artifact (imputer, encoder, scaler, Isolation
Forest), the chargeback_rate_90d temporal-safety proof, and end-to-end
model output shape/determinism.
"""
import numpy as np
import pandas as pd
import pytest

from database.connection import check_connection, get_engine
from models.feature_matrix import FEATURE_COLUMNS
from models.ml_metrics import (
    confusion_matrix_at_threshold,
    find_threshold_for_top_k,
    precision_recall_at_top_k,
    precision_recall_f1_at_threshold,
    recall_at_fixed_fpr,
)
from models.ml_preprocessing import FittedPreprocessor
from models.temporal_split import chronological_split, split_summary

# ---------------------------------------------------------------------------
# Unit tests: metric helpers, hand-computed
# ---------------------------------------------------------------------------


def test_precision_recall_f1_at_threshold_hand_computed():
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.7, 0.6, 0.1, 0.1, 0.1]
    # threshold 0.5 -> predicted positive: 0.9,0.8,0.7,0.6 -> indices 0,1,3,4
    # TP: idx0,idx1 (true=1) = 2; FP: idx3,idx4 (true=0) = 2; FN: idx2 (true=1,missed) = 1
    result = precision_recall_f1_at_threshold(y_true, y_score, threshold=0.5)
    assert result["precision"] == pytest.approx(2 / 4)
    assert result["recall"] == pytest.approx(2 / 3)


def test_confusion_matrix_at_threshold_hand_computed():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.1, 0.9, 0.1]
    cm = confusion_matrix_at_threshold(y_true, y_score, threshold=0.5)
    assert cm == {"threshold": 0.5, "tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_precision_recall_at_top_k_hand_computed():
    y_true = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 10 rows, 2 positive
    y_score = [0.95, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # top 20% = top 2 by score: idx0 (0.95, true=1) and idx1 (0.9, true=0)
    result = precision_recall_at_top_k(y_true, y_score, k_frac=0.2)
    assert result["k"] == 2
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)  # caught 1 of 2 positives


def test_find_threshold_for_top_k():
    scores = [0.1, 0.9, 0.5, 0.3, 0.95]  # top 20% of 5 = top 1
    threshold = find_threshold_for_top_k(scores, k_frac=0.2)
    assert threshold == pytest.approx(0.95)


def test_recall_at_fixed_fpr_perfect_separation():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    result = recall_at_fixed_fpr(y_true, y_score, target_fpr=0.5)
    assert result["recall"] == 1.0


# ---------------------------------------------------------------------------
# Unit tests: FittedPreprocessor, train-only fitting with hand-built data
# ---------------------------------------------------------------------------


def _toy_frame(amounts, channels, statuses, segments, categories):
    from models.feature_matrix import RULE_IDS

    n = len(amounts)
    data = {
        "amount": amounts,
        "amount_zscore": [np.nan] * n,
        "txn_count_last_10min": [0] * n,
        "txn_count_last_60min": [0] * n,
        "recent_failed_count_15min": [0] * n,
        "distinct_merchants_30d": [1] * n,
        "distinct_devices_30d": [1] * n,
        "customer_amount_iqr_score": [0.0] * n,
        "merchant_amount_iqr_score": [0.0] * n,
        "frequency_deviation_zscore": [np.nan] * n,
        "hour": [12] * n,
        "rules_fired_count": [0] * n,
        "max_rule_severity": [-1] * n,
        "is_new_device": [False] * n,
        "is_new_location": [False] * n,
        "is_first_time_at_merchant": [False] * n,
        "channel": channels,
        "status": statuses,
        "customer_risk_segment": segments,
        "merchant_risk_category": categories,
    }
    for rid in RULE_IDS:
        data[f"rule_fired_{rid}"] = [False] * n
    return pd.DataFrame(data)


def test_preprocessor_imputes_using_train_median_not_test_median():
    train = _toy_frame(
        amounts=[10.0, 20.0, 30.0], channels=["ECOM"] * 3, statuses=["APPROVED"] * 3,
        segments=["STANDARD"] * 3, categories=["STANDARD"] * 3,
    )
    # amount_zscore is NaN in all training rows except we set one real value
    # to prove the imputer's fitted median comes from train, not test.
    train["amount_zscore"] = [1.0, 2.0, 3.0]

    test = _toy_frame(
        amounts=[999.0], channels=["ECOM"], statuses=["APPROVED"],
        segments=["STANDARD"], categories=["STANDARD"],
    )
    test["amount_zscore"] = [np.nan]  # missing - must be imputed with TRAIN median (2.0), not test's own (undefined)

    pre = FittedPreprocessor().fit(train)
    X_test = pre.transform_for_trees(test)
    amount_zscore_idx = pre.feature_names_.index("amount_zscore")
    assert X_test[0, amount_zscore_idx] == pytest.approx(2.0)  # train median


def test_preprocessor_scaler_fit_on_train_only():
    train = _toy_frame(
        amounts=[10.0, 20.0, 30.0], channels=["ECOM"] * 3, statuses=["APPROVED"] * 3,
        segments=["STANDARD"] * 3, categories=["STANDARD"] * 3,
    )
    pre = FittedPreprocessor().fit(train)
    amount_idx = pre.feature_names_.index("amount")

    # a wildly different test-set amount must NOT shift the scaler's
    # mean/std - it should transform relative to TRAIN's distribution.
    test = _toy_frame(
        amounts=[10000.0], channels=["ECOM"], statuses=["APPROVED"],
        segments=["STANDARD"], categories=["STANDARD"],
    )
    X_test_scaled = pre.transform_for_linear(test)
    # train mean=20, std=~8.16 -> z for 10000 should be huge, proving the
    # scaler wasn't refit to include 10000 in its own distribution.
    assert X_test_scaled[0, amount_idx] > 100


def test_preprocessor_handles_unseen_category_gracefully():
    train = _toy_frame(
        amounts=[10.0, 20.0], channels=["ECOM", "POS"], statuses=["APPROVED"] * 2,
        segments=["STANDARD"] * 2, categories=["STANDARD"] * 2,
    )
    pre = FittedPreprocessor().fit(train)
    test = _toy_frame(
        amounts=[15.0], channels=["ATM"], statuses=["APPROVED"],  # ATM never seen in train
        segments=["STANDARD"], categories=["STANDARD"],
    )
    X_test = pre.transform_for_trees(test)  # must not raise
    assert X_test.shape[0] == 1


# ---------------------------------------------------------------------------
# Integration tests: real data
# ---------------------------------------------------------------------------

healthy, _ = check_connection()
pytestmark_db = pytest.mark.skipif(not healthy, reason="requires a live DATABASE_URL (see .env.example)")


@pytest.fixture(scope="module")
def full_feature_matrix():
    from models.feature_matrix import build_feature_matrix

    engine = get_engine()
    return build_feature_matrix(engine)


@pytestmark_db
def test_chronological_split_has_zero_temporal_overlap(full_feature_matrix):
    train_df, val_df, test_df = chronological_split(full_feature_matrix)
    assert train_df["transaction_ts"].max() <= val_df["transaction_ts"].min()
    assert val_df["transaction_ts"].max() <= test_df["transaction_ts"].min()


@pytestmark_db
def test_chronological_split_proportions_are_approximately_70_15_15(full_feature_matrix):
    train_df, val_df, test_df = chronological_split(full_feature_matrix)
    summary = split_summary(train_df, val_df, test_df)
    assert summary["train_pct"] == pytest.approx(0.70, abs=0.01)
    assert summary["val_pct"] == pytest.approx(0.15, abs=0.01)
    assert summary["test_pct"] == pytest.approx(0.15, abs=0.01)
    assert summary["total_rows"] == len(full_feature_matrix)


@pytestmark_db
def test_chronological_split_is_not_shuffled(full_feature_matrix):
    """Row order within each split must still be time-sorted - proves no
    shuffling occurred anywhere in the split.
    """
    train_df, val_df, test_df = chronological_split(full_feature_matrix)
    for d in [train_df, val_df, test_df]:
        assert d["transaction_ts"].is_monotonic_increasing


@pytestmark_db
def test_isolation_forest_is_not_fit_on_validation_or_test_rows(full_feature_matrix):
    """Direct regression test for the requirement: the forest must be
    fit using exactly the training rows, never anything from val/test.
    """
    from models.ml_isolation_forest import fit_isolation_forest_train_only

    train_df, val_df, test_df = chronological_split(full_feature_matrix)
    artifacts, train_scores = fit_isolation_forest_train_only(train_df)

    assert artifacts["n_rows_fit"] == len(train_df)
    assert artifacts["n_rows_fit"] != len(train_df) + len(val_df) + len(test_df)
    assert len(train_scores) == len(train_df)


@pytestmark_db
def test_isolation_forest_scoring_never_refits_on_val_or_test(full_feature_matrix):
    """Scoring val/test must use the SAME frozen model object (identity
    check), not a freshly fit one - this is what "transform, don't refit"
    actually means at the code level.
    """
    from models.ml_isolation_forest import fit_isolation_forest_train_only, score_with_fitted_forest

    train_df, val_df, test_df = chronological_split(full_feature_matrix)
    artifacts, _ = fit_isolation_forest_train_only(train_df)
    model_before = artifacts["model"]

    _ = score_with_fitted_forest(artifacts, val_df)
    _ = score_with_fitted_forest(artifacts, test_df)

    assert artifacts["model"] is model_before  # same object, never reassigned/refit


@pytestmark_db
def test_isolation_forest_val_test_scores_bounded(full_feature_matrix):
    from models.ml_isolation_forest import fit_isolation_forest_train_only, score_with_fitted_forest

    train_df, val_df, test_df = chronological_split(full_feature_matrix)
    artifacts, _ = fit_isolation_forest_train_only(train_df)
    val_scores = score_with_fitted_forest(artifacts, val_df)
    test_scores = score_with_fitted_forest(artifacts, test_df)

    assert val_scores.between(0, 1).all()
    assert test_scores.between(0, 1).all()


@pytestmark_db
def test_chargeback_rate_is_excluded_from_ml_features():
    """Enforces the documented decision (docs/model_card.md): even though
    chargeback_rate_90d is temporally safe as constructed, it's excluded
    from the primary model due to unrealistic same-day availability vs
    real chargeback reporting lag.
    """
    assert "chargeback_rate_90d" not in FEATURE_COLUMNS


@pytestmark_db
def test_chargeback_rate_temporal_safety_proof():
    """Direct proof (not just the Phase 3 test) that chargeback_rate_90d,
    for a sample of merchant/day rows spanning the full timeline including
    the test period, only ever reflects fraud confirmations from BEFORE
    that day - i.e. it would be safe under the chronological split if it
    were ever used, even though the decision above is to exclude it anyway.
    """
    from sqlalchemy import text

    engine = get_engine()
    with engine.connect() as conn:
        sample = conn.execute(
            text(
                "SELECT merchant_id, metric_date, chargeback_rate_90d FROM fact_merchant_daily_metrics "
                "WHERE chargeback_rate_90d IS NOT NULL ORDER BY random() LIMIT 15"
            )
        ).fetchall()
    assert len(sample) > 0

    with engine.connect() as conn:
        for merchant_id, metric_date, reported_rate in sample:
            recomputed = conn.execute(
                text(
                    """
                    SELECT AVG(CASE WHEN g.is_fraud THEN 1.0 ELSE 0.0 END)
                    FROM fact_transactions t
                    JOIN ground_truth_fraud g ON g.transaction_id = t.transaction_id
                    WHERE t.merchant_id = :m
                      AND t.transaction_ts < :d
                      AND t.transaction_ts >= :d - INTERVAL '90 days'
                    """
                ),
                {"m": merchant_id, "d": metric_date},
            ).scalar()
            # chargeback_rate_90d is stored as NUMERIC(6,4) - tolerance
            # matches that column precision, not floating point noise.
            assert float(recomputed) == pytest.approx(float(reported_rate), abs=5e-5), (
                f"chargeback_rate_90d for merchant {merchant_id} on {metric_date} does not match a "
                f"recomputation using only transactions strictly before that date"
            )


@pytestmark_db
def test_model_prediction_shape_and_types(full_feature_matrix):
    """End-to-end smoke test: fit both models on a small slice and check
    prediction output shape/dtype/range - not a full accuracy check, just
    contract correctness.
    """
    from models.feature_matrix import NUMERIC_FEATURE_COLUMNS
    from models.ml_isolation_forest import fit_isolation_forest_train_only, score_with_fitted_forest
    from models.ml_models import train_logistic_regression, train_xgboost_with_validation_selection

    train_df, val_df, test_df = chronological_split(full_feature_matrix)
    # use a modest slice for speed - this test checks contracts, not accuracy
    train_small = train_df.iloc[-8000:].reset_index(drop=True)
    val_small = val_df.iloc[:2000].reset_index(drop=True)

    iforest_artifacts, train_scores = fit_isolation_forest_train_only(train_small)
    train_small = train_small.copy()
    val_small = val_small.copy()
    train_small["isolation_forest_score"] = train_scores.values
    val_small["isolation_forest_score"] = score_with_fitted_forest(iforest_artifacts, val_small).values

    pre = FittedPreprocessor(numeric_cols=NUMERIC_FEATURE_COLUMNS + ["isolation_forest_score"]).fit(train_small)
    X_train_tree = pre.transform_for_trees(train_small)
    X_val_tree = pre.transform_for_trees(val_small)
    X_train_lin = pre.transform_for_linear(train_small)
    X_val_lin = pre.transform_for_linear(val_small)
    y_train = train_small["is_fraud"].astype(int)
    y_val = val_small["is_fraud"].astype(int)

    logreg = train_logistic_regression(X_train_lin, y_train)
    logreg_proba = logreg.predict_proba(X_val_lin)
    assert logreg_proba.shape == (len(val_small), 2)
    assert np.all((logreg_proba >= 0) & (logreg_proba <= 1))
    assert np.allclose(logreg_proba.sum(axis=1), 1.0)

    xgb_model, _ = train_xgboost_with_validation_selection(X_train_tree, y_train, X_val_tree, y_val)
    xgb_proba = xgb_model.predict_proba(X_val_tree)
    assert xgb_proba.shape == (len(val_small), 2)
    assert np.all((xgb_proba >= 0) & (xgb_proba <= 1))


@pytestmark_db
def test_logistic_regression_is_deterministic(full_feature_matrix):
    """Fixed random_state -> identical coefficients across repeated fits
    on identical data.
    """
    from models.feature_matrix import NUMERIC_FEATURE_COLUMNS
    from models.ml_models import train_logistic_regression

    train_df, _, _ = chronological_split(full_feature_matrix)
    train_small = train_df.iloc[-5000:].reset_index(drop=True)
    pre = FittedPreprocessor(numeric_cols=NUMERIC_FEATURE_COLUMNS + []).fit(train_small)
    # isolation_forest_score not needed for this determinism check - use
    # base numeric cols only via a preprocessor fit without it.
    X_train = pre.transform_for_linear(train_small)
    y_train = train_small["is_fraud"].astype(int)

    model_a = train_logistic_regression(X_train, y_train)
    model_b = train_logistic_regression(X_train, y_train)
    assert np.allclose(model_a.coef_, model_b.coef_)


@pytestmark_db
def test_no_target_or_identifier_leakage_into_feature_columns():
    forbidden = {"is_fraud", "fraud_typology", "is_synthetic_label", "transaction_id", "customer_id", "merchant_id", "device_id", "location_id", "transaction_ts", "chargeback_rate_90d"}
    assert forbidden.isdisjoint(set(FEATURE_COLUMNS))
