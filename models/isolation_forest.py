"""Unsupervised multivariate anomaly detection via Isolation Forest - the
complement to the univariate IQR/rules methods in models/statistical.py
and rules/. Isolation Forest isolates points by recursively splitting on
random features/thresholds; anomalies are, on average, isolated in fewer
splits than normal points, because they don't need many cuts to separate
from the bulk of the data. This lets it catch combinations of features
that are each individually unremarkable but jointly unusual - something no
single-feature rule or IQR check can see.

Being unsupervised, it is fit on the full dataset with no train/test split
in the supervised sense (there's no label to leak) - this is standard
practice for anomaly detection, but it does mean its evaluation against
ground truth here is post-hoc and diagnostic, not a claim of generalization
the way Phase 6's time-aware supervised evaluation will be.

contamination is set to match the known synthetic fraud injection rate
(~2%) purely because we have that luxury in this project - a real
deployment would not know the true fraud rate in advance and would either
use 'auto' or tune contamination against analyst feedback over time. This
is documented as a limitation, not hidden.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_COLUMNS = [
    "amount",
    "amount_zscore",
    "txn_count_last_10min",
    "txn_count_last_60min",
    "recent_failed_count_15min",
    "distinct_merchants_30d",
    "distinct_devices_30d",
    "customer_amount_iqr_score",
    "merchant_amount_iqr_score",
    "frequency_deviation_zscore",
    "hour",
    "is_new_device",
    "is_new_location",
]

CONTAMINATION = 0.02
RANDOM_STATE = 42


def _prepare_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    for col in ["is_new_device", "is_new_location"]:
        X[col] = X[col].astype(int)
    # Isolation Forest can't handle NaN - median-impute, which is a
    # reasonable default for a thin-history customer/merchant (their
    # deviation-based features are legitimately unknown, not zero).
    return X.fillna(X.median(numeric_only=True))


def fit_isolation_forest(df: pd.DataFrame) -> tuple[IsolationForest, pd.Series]:
    """Fits on df[FEATURE_COLUMNS] and returns (model, anomaly_score) where
    anomaly_score is min-max normalized to [0, 1], higher = more anomalous.
    """
    X = _prepare_matrix(df)
    model = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE, n_estimators=200)
    model.fit(X)

    raw_scores = -model.score_samples(X)  # score_samples: higher = more normal, so negate
    normalized = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    return model, pd.Series(normalized, index=df.index, name="isolation_forest_score")


def add_isolation_forest_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _, scores = fit_isolation_forest(df)
    df["isolation_forest_score"] = scores
    return df
