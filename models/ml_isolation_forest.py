"""Train-only-fit Isolation Forest for use as a supervised-model feature.

This is deliberately separate from Phase 5's models/isolation_forest.py,
which fits on the ENTIRE dataset (correct for that phase's standalone
unsupervised evaluation, where there is no train/test split in the
supervised sense). Reusing those whole-dataset scores as an ML feature
here would leak validation/test-period structure into a value seen at
training time - the forest's splits would have been informed by exactly
the transactions the supervised model is later evaluated on.

The fix: fit the forest on the training period ONLY, then use that frozen
model (plus training-derived imputation medians and score normalization
bounds) to score validation and test - the same "fit on train, transform
on everything" rule applied to every other preprocessing step in this
phase (see models/ml_preprocessing.py).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from models.isolation_forest import CONTAMINATION, FEATURE_COLUMNS, RANDOM_STATE


def _prepare_matrix(df: pd.DataFrame, impute_medians: pd.Series) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    for col in ["is_new_device", "is_new_location"]:
        X[col] = X[col].astype(int)
    return X.fillna(impute_medians)


def fit_isolation_forest_train_only(train_df: pd.DataFrame):
    """Fits on train_df only. Returns a dict of frozen artifacts
    (model, impute_medians, score_min, score_max) plus the resulting
    train-period scores, so nothing downstream needs to touch train_df's
    raw values again.
    """
    raw_medians = train_df[FEATURE_COLUMNS].median(numeric_only=True)
    X_train = _prepare_matrix(train_df, raw_medians)

    model = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE, n_estimators=200)
    model.fit(X_train)

    raw_train_scores = -model.score_samples(X_train)
    score_min, score_max = raw_train_scores.min(), raw_train_scores.max()
    train_scores = (raw_train_scores - score_min) / (score_max - score_min)

    artifacts = {
        "model": model,
        "impute_medians": raw_medians,
        "score_min": score_min,
        "score_max": score_max,
        "n_rows_fit": len(train_df),
    }
    return artifacts, pd.Series(train_scores, index=train_df.index, name="isolation_forest_score")


def score_with_fitted_forest(artifacts: dict, df: pd.DataFrame) -> pd.Series:
    """Scores df using a forest already fit on the training period. Never
    refits - val/test transactions never influence the model or the
    normalization bounds, only get transformed by them.
    """
    X = _prepare_matrix(df, artifacts["impute_medians"])
    raw_scores = -artifacts["model"].score_samples(X)
    denom = artifacts["score_max"] - artifacts["score_min"]
    normalized = (raw_scores - artifacts["score_min"]) / denom
    # val/test can be more or less anomalous than anything seen in
    # training, so their raw scores can fall outside the training min/max -
    # clip to keep the feature in the same [0,1] range the model was
    # trained to expect, rather than letting it silently extrapolate.
    normalized = np.clip(normalized, 0.0, 1.0)
    return pd.Series(normalized, index=df.index, name="isolation_forest_score")
