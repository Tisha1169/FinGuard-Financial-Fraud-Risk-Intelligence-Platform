"""Caches the trained model pipeline for the app's lifetime via
st.cache_resource - retraining XGBoost + Isolation Forest on every
Streamlit rerun (which happens on nearly every user interaction) would
make the app unusable. This costs ~5-10 seconds once per app process
(cold start), matching scripts/train_models.py's own runtime, and reuses
the exact same Phase 6 modules rather than a separate implementation.
"""
import numpy as np
import pandas as pd
import streamlit as st

from models.feature_matrix import NUMERIC_FEATURE_COLUMNS, build_feature_matrix
from models.ml_explain import compute_shap_values, explain_single_prediction, global_importance
from models.ml_isolation_forest import fit_isolation_forest_train_only, score_with_fitted_forest
from models.ml_metrics import full_evaluation
from models.ml_models import train_logistic_regression, train_xgboost_with_validation_selection
from models.ml_preprocessing import FittedPreprocessor
from models.temporal_split import chronological_split, split_summary
from streamlit_app.db import engine


@st.cache_resource(show_spinner="Training fraud detection model (one-time, ~10s)...")
def get_trained_pipeline():
    full_df = build_feature_matrix(engine())
    train_df, val_df, test_df = chronological_split(full_df)

    iforest_artifacts, train_scores = fit_isolation_forest_train_only(train_df)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["isolation_forest_score"] = train_scores.values
    val_df["isolation_forest_score"] = score_with_fitted_forest(iforest_artifacts, val_df).values
    test_df["isolation_forest_score"] = score_with_fitted_forest(iforest_artifacts, test_df).values

    all_numeric_cols = NUMERIC_FEATURE_COLUMNS + ["isolation_forest_score"]
    preprocessor = FittedPreprocessor(numeric_cols=all_numeric_cols).fit(train_df)
    X_train = preprocessor.transform_for_trees(train_df)
    X_val = preprocessor.transform_for_trees(val_df)
    X_test = preprocessor.transform_for_trees(test_df)
    X_train_lin = preprocessor.transform_for_linear(train_df)
    X_val_lin = preprocessor.transform_for_linear(val_df)
    X_test_lin = preprocessor.transform_for_linear(test_df)
    y_train, y_val, y_test = train_df["is_fraud"].astype(int), val_df["is_fraud"].astype(int), test_df["is_fraud"].astype(int)

    logreg = train_logistic_regression(X_train_lin, y_train)
    xgb_model, selection_report = train_xgboost_with_validation_selection(X_train, y_train, X_val, y_val)

    logreg_test_scores = logreg.predict_proba(X_test_lin)[:, 1]
    xgb_test_scores = xgb_model.predict_proba(X_test)[:, 1]
    from models.ml_metrics import find_threshold_for_top_k
    xgb_val_scores = xgb_model.predict_proba(X_val)[:, 1]
    threshold = find_threshold_for_top_k(xgb_val_scores, k_frac=0.02)

    logreg_eval = full_evaluation(y_test, logreg_test_scores, threshold)
    xgb_eval = full_evaluation(y_test, xgb_test_scores, threshold)

    explainer, shap_values, X_sample_df, sample_idx = compute_shap_values(
        xgb_model, X_test, preprocessor.feature_names_, sample_size=1500,
    )
    top_features = global_importance(shap_values, preprocessor.feature_names_, top_n=15)

    return {
        "preprocessor": preprocessor,
        "xgb_model": xgb_model,
        "logreg_model": logreg,
        "threshold": threshold,
        "split_summary": split_summary(train_df, val_df, test_df),
        "selection_report": selection_report,
        "logreg_eval": logreg_eval,
        "xgb_eval": xgb_eval,
        "shap_explainer": explainer,
        "shap_values": shap_values,
        "shap_sample_df": X_sample_df,
        "shap_top_features": top_features,
        "test_df": test_df.reset_index(drop=True),
        "test_scores": xgb_test_scores,
        "y_test": y_test.values,
        "feature_names": preprocessor.feature_names_,
    }


def explain_transaction(transaction_id: int) -> dict | None:
    """Computes a SHAP explanation for one transaction, if it falls in the
    cached model's TEST split (SHAP is only meaningful for the model's
    held-out evaluation; explaining a training-period prediction would be
    explaining an in-sample fit, not a genuine out-of-sample decision).
    Returns None otherwise, and the UI falls back to the rule/risk-score
    breakdown, which is available for every transaction.

    Recomputes SHAP for just this one row rather than reusing the cached
    global sample - the global sample is a fixed random 1,500-row subset
    used for aggregate feature importance, and most looked-up transactions
    won't happen to be in it.
    """
    pipeline = get_trained_pipeline()
    test_df = pipeline["test_df"]
    match = test_df.index[test_df["transaction_id"] == transaction_id]
    if len(match) == 0:
        return None
    row_pos = match[0]
    X_row = pipeline["preprocessor"].transform_for_trees(test_df.iloc[[row_pos]])
    row_shap = pipeline["shap_explainer"].shap_values(pd.DataFrame(X_row, columns=pipeline["feature_names"]))
    base_value = pipeline["shap_explainer"].expected_value

    return explain_single_prediction(
        row_shap, pd.DataFrame(X_row, columns=pipeline["feature_names"]), 0, base_value, top_n=8
    )
