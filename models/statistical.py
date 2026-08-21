"""Statistical/behavioral anomaly detection - the "Statistical Anomaly
Engine" layer, distinct from both the business rules (Phase 4, hard
thresholds) and the supervised ML model (Phase 6, learned from labels).

Two methods, chosen deliberately for complementary strengths/weaknesses:

- **IQR / Tukey fences** (robust to outliers): a customer or merchant with
  even one or two historical whale transactions won't have their mean/
  stddev-based baseline (rules/config.py's z-score approach) dragged
  around, because the IQR method uses the 25th/75th percentile and median
  rather than the mean - a few extreme values don't move a median or
  quartile much. Limitation: needs enough history to estimate quartiles
  meaningfully (thin history -> unreliable fences), and like any
  univariate method it can't see combinations of features that are each
  individually unremarkable but jointly unusual.
- **Isolation Forest** (models/isolation_forest.py, unsupervised,
  multivariate): can catch exactly the multivariate anomalies IQR/rules
  miss (e.g. a normal amount, at a normal hour, but combined with a new
  device AND unusually high velocity). Limitation: it's unsupervised, so
  its notion of "anomalous" is purely statistical rarity, not
  fraud-specific - it will also flag legitimate rare behavior (a customer's
  first big purchase, genuine business travel) with no way to distinguish
  the two without labels.

All rolling statistics here follow the same leakage rule as the rest of
the project: every window is computed inclusive of a transaction, then
shifted by one position within its group so a transaction only ever sees
the state of the window *as of its previous transaction* - never itself.
See docs/feature_engineering.md for the general pattern this extends.
"""
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from rules.batch import load_transactions

# Below this many prior observations, a Q1/Q3 estimate is unstable enough
# that even a small deviation can produce a near-zero IQR and an absurdly
# inflated outlier score (e.g. two prior transactions of $39.00 and $39.02
# gives an IQR of $0.02 - a third transaction of $50 would then look like
# a 500-IQR outlier). Below this threshold the IQR score is suppressed to 0
# rather than trusted.
MIN_HISTORY_FOR_IQR = 10


def _rolling_quantiles_asof(df: pd.DataFrame, group_col: str, value_col: str, window: str) -> pd.DataFrame:
    """Returns a DataFrame keyed by transaction_id with the group's
    trailing-window Q1/median/Q3 of value_col, as observed as of the
    PREVIOUS row in the group (i.e. excluding the row's own value).
    """
    sub = df[["transaction_id", group_col, "transaction_ts", value_col]].sort_values([group_col, "transaction_ts"]).copy()
    indexed = sub.set_index("transaction_ts")
    grp = indexed.groupby(group_col)[value_col]

    sub["q1_incl"] = grp.rolling(window).quantile(0.25).values
    sub["q3_incl"] = grp.rolling(window).quantile(0.75).values
    sub["median_incl"] = grp.rolling(window).median().values
    sub["count_incl"] = grp.rolling(window).count().values

    prefix = f"{group_col}_{value_col}"
    by_group = sub.groupby(group_col)
    sub[f"{prefix}_q1_asof"] = by_group["q1_incl"].shift(1)
    sub[f"{prefix}_q3_asof"] = by_group["q3_incl"].shift(1)
    sub[f"{prefix}_median_asof"] = by_group["median_incl"].shift(1)
    sub[f"{prefix}_count_asof"] = by_group["count_incl"].shift(1)

    return sub[["transaction_id", f"{prefix}_q1_asof", f"{prefix}_q3_asof", f"{prefix}_median_asof", f"{prefix}_count_asof"]]


def _iqr_upper_outlier_score(amount: pd.Series, q1: pd.Series, q3: pd.Series, prior_count: pd.Series) -> pd.Series:
    """Tukey's fence: values beyond Q3 + 1.5*IQR are classified outliers.
    Returns 0 for non-outliers, when IQR is 0/unknown, or when there isn't
    enough prior history to trust the fence (see MIN_HISTORY_FOR_IQR);
    otherwise a positive score scaling with how many IQR-widths beyond the
    fence the value sits.
    """
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    score = (amount - upper_fence) / iqr
    score = score.where(iqr > 0, other=np.nan)
    score = score.where(prior_count >= MIN_HISTORY_FOR_IQR, other=np.nan)
    return score.clip(lower=0).fillna(0)


def add_iqr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds customer_amount_iqr_score and merchant_amount_iqr_score."""
    df = df.copy()

    cust_q = _rolling_quantiles_asof(df, "customer_id", "amount", "90D")
    df = df.merge(cust_q, on="transaction_id", how="left")
    df["customer_amount_iqr_score"] = _iqr_upper_outlier_score(
        df["amount"], df["customer_id_amount_q1_asof"], df["customer_id_amount_q3_asof"], df["customer_id_amount_count_asof"]
    )

    merch_q = _rolling_quantiles_asof(df, "merchant_id", "amount", "90D")
    df = df.merge(merch_q, on="transaction_id", how="left")
    df["merchant_amount_iqr_score"] = _iqr_upper_outlier_score(
        df["amount"], df["merchant_id_amount_q1_asof"], df["merchant_id_amount_q3_asof"], df["merchant_id_amount_count_asof"]
    )

    return df


MIN_HISTORY_FOR_FREQUENCY = 15


def add_frequency_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Compares a customer's short-term (7-day) transaction count to what
    their long-term (90-day) rate would predict - a slower, coarser-grained
    complement to the minute-scale velocity rule (R1), meant to catch a
    customer whose overall activity level has shifted over days/weeks, not
    just minutes.

    A raw ratio (7-day rate / 90-day rate) was tried first and rejected:
    for a customer with a low baseline rate, a week's count is a small-
    sample Poisson draw with high relative variance, so a ratio of 3-4x
    "expected" happens by chance alone far too often to be useful evidence.
    A Poisson z-score ((observed - expected) / sqrt(expected)) accounts for
    that sample-size-dependent variance directly - but this only works if
    "expected" is estimated correctly. The first version of this function
    divided the 90-day trailing count by a fixed 90, which silently
    underestimates a customer's true rate for their first ~90 days of
    activity (a customer active only 30 days has ALL their history in that
    "90-day" window, but dividing by 90 instead of 30 makes their rate look
    3x lower than it is) - verified empirically, this inflated z>=3 to
    14.7% of all transactions instead of the ~0.1% a well-calibrated
    z-score implies. Fixed by dividing by the customer's actual elapsed
    active days (capped at 90), not a fixed window length.
    """
    df = df.sort_values(["customer_id", "transaction_ts"]).copy()
    indexed = df.set_index("transaction_ts")
    grp = indexed.groupby("customer_id")["transaction_id"]

    count_7d_incl = grp.rolling("7D").count().values
    count_90d_incl = grp.rolling("90D").count().values
    df["count_7d_prior"] = count_7d_incl - 1
    df["count_90d_prior"] = count_90d_incl - 1

    first_txn_ts = df.groupby("customer_id")["transaction_ts"].transform("min")
    prior_txn_ts = df.groupby("customer_id")["transaction_ts"].shift(1)
    days_active_asof = (prior_txn_ts - first_txn_ts).dt.total_seconds() / 86400
    days_active_asof = days_active_asof.clip(lower=1, upper=90)

    expected_7d = df["count_90d_prior"] / days_active_asof * 7
    poisson_z = (df["count_7d_prior"] - expected_7d) / np.sqrt(expected_7d.clip(lower=0.5))

    # only meaningful once there's enough history for a stable 90-day rate
    # AND the expected weekly count is large enough for the normal
    # approximation to Poisson variance to be reasonable.
    trustworthy = (df["count_90d_prior"] >= MIN_HISTORY_FOR_FREQUENCY) & (expected_7d >= 1)
    df["frequency_deviation_zscore"] = poisson_z.where(trustworthy, other=np.nan)

    return df.sort_values("transaction_id").reset_index(drop=True)


def _normalize_zscore(z: pd.Series, trigger_z: float) -> pd.Series:
    """Maps a one-sided z-score to [0, 1], reaching 1.0 at trigger_z (a
    z-score of 3 corresponds to roughly p<0.001 for a one-tailed normal
    test - a deliberately conservative bar so this component doesn't
    saturate on ordinary week-to-week variance).
    """
    return (z.clip(lower=0) / trigger_z).clip(upper=1).fillna(0)


def compute_behavioral_anomaly_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combines the IQR and frequency-deviation components into one
    behavioral_anomaly_score in [0, 1] per transaction, using a
    noisy-OR combination (1 - product of (1 - component)): each component
    is treated as independent evidence of anomalous behavior, so any one
    strong signal drives the combined score up, without needing an
    arbitrary weighting scheme. This score is one input into the final
    combined risk score assembled in Phase 7 (risk_scoring/).
    """
    df = df.copy()
    a = (df["customer_amount_iqr_score"].clip(upper=3) / 3).fillna(0)
    b = (df["merchant_amount_iqr_score"].clip(upper=3) / 3).fillna(0)
    c = _normalize_zscore(df["frequency_deviation_zscore"], trigger_z=3.0)

    df["behavioral_anomaly_score"] = 1 - (1 - a) * (1 - b) * (1 - c)
    return df


def build_statistical_features(engine: Engine) -> pd.DataFrame:
    txns = load_transactions(engine)
    txns = add_iqr_features(txns)
    txns = add_frequency_deviation(txns)
    txns = compute_behavioral_anomaly_score(txns)
    return txns
