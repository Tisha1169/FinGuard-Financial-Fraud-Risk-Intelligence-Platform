"""Assembles the final combined risk score for every transaction.

Reuses Phase 6's exact machinery (chronological split, train-only-fit
Isolation Forest, train-only-fit preprocessing, train-only-fit XGBoost)
rather than re-deriving it, so the risk score is built on the same
leakage-disciplined foundation the ML evaluation was audited against -
in particular, this deliberately does NOT reuse Phase 5's whole-dataset
Isolation Forest scores (models/isolation_forest.py), for the same reason
Phase 6 didn't: fitting on data that includes the validation/test period
would leak forward-looking structure into a score that's supposed to
represent what a deployed model could have known at the time.

Every transaction gets scored, including the training period - this
mirrors a real deployment, where a model is trained once and then used to
score both the history it learned from and everything after. Training-
period scores are in-sample (the model saw these exact labels during
fitting); validation/test scores are out-of-sample. This is stated
explicitly in docs/risk_scoring.md and is not hidden.
"""
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from models.feature_matrix import NUMERIC_FEATURE_COLUMNS, build_feature_matrix
from models.ml_isolation_forest import fit_isolation_forest_train_only, score_with_fitted_forest
from models.ml_models import train_logistic_regression, train_xgboost_with_validation_selection
from models.ml_preprocessing import FittedPreprocessor
from models.statistical import build_statistical_features
from models.temporal_split import chronological_split
from risk_scoring import components, config


def build_risk_scores(engine: Engine) -> tuple[pd.DataFrame, dict]:
    """Returns (risk_scores_df, diagnostics) where risk_scores_df has one
    row per transaction with all four components, combined_score, and
    risk_tier - ready to load into the risk_scores table.
    """
    full_df = build_feature_matrix(engine)
    train_df, val_df, test_df = chronological_split(full_df)

    # --- Isolation Forest: train-only fit, transform everything ---
    iforest_artifacts, train_iforest_scores = fit_isolation_forest_train_only(train_df)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["isolation_forest_score"] = train_iforest_scores.values
    val_df["isolation_forest_score"] = score_with_fitted_forest(iforest_artifacts, val_df).values
    test_df["isolation_forest_score"] = score_with_fitted_forest(iforest_artifacts, test_df).values

    # --- behavioral_anomaly_score (IQR/frequency) - pure point-in-time
    # statistics, no cross-transaction learning, so it's safe to compute
    # once over the full dataset (unlike Isolation Forest, it was never
    # "fit" on anything - see models/statistical.py) ---
    stat_df = build_statistical_features(engine)
    for d in [train_df, val_df, test_df]:
        d["behavioral_anomaly_score"] = d["transaction_id"].map(
            stat_df.set_index("transaction_id")["behavioral_anomaly_score"]
        ).values

    # --- Preprocessing + XGBoost: train-only fit ---
    all_numeric_cols = NUMERIC_FEATURE_COLUMNS + ["isolation_forest_score"]
    preprocessor = FittedPreprocessor(numeric_cols=all_numeric_cols).fit(train_df)
    X_train = preprocessor.transform_for_trees(train_df)
    X_val = preprocessor.transform_for_trees(val_df)
    X_test = preprocessor.transform_for_trees(test_df)
    y_train = train_df["is_fraud"].astype(int)
    y_val = val_df["is_fraud"].astype(int)

    xgb_model, selection_report = train_xgboost_with_validation_selection(X_train, y_train, X_val, y_val)

    train_df["ml_probability"] = xgb_model.predict_proba(X_train)[:, 1]
    val_df["ml_probability"] = xgb_model.predict_proba(X_val)[:, 1]
    test_df["ml_probability"] = xgb_model.predict_proba(X_test)[:, 1]
    train_df["ml_in_sample"] = True
    val_df["ml_in_sample"] = False
    test_df["ml_in_sample"] = False

    # --- exposure cap: 95th percentile of TRAIN amounts only, frozen ---
    exposure_cap = float(train_df["amount"].quantile(config.EXPOSURE_CAP_PERCENTILE))

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    rules_component = components.compute_rules_component(all_df)
    behavioral_component = components.compute_behavioral_component(all_df)
    exposure_component = components.compute_exposure_component(all_df, exposure_cap)
    ml_component = all_df["ml_probability"]

    combined_score = components.compute_combined_score(ml_component, rules_component, behavioral_component, exposure_component)

    # --- tier cutpoints from VALIDATION-period combined_score quantiles only ---
    val_mask = all_df["transaction_id"].isin(val_df["transaction_id"])
    val_combined = combined_score[val_mask]
    tier_cutpoints = {
        tier: float(val_combined.quantile(q)) for tier, q in config.TIER_QUANTILES.items()
    }
    risk_tier = components.assign_risk_tier(combined_score, tier_cutpoints)

    risk_scores_df = pd.DataFrame({
        "transaction_id": all_df["transaction_id"],
        "ml_component": ml_component.round(5),
        "rules_component": rules_component.round(5),
        "behavioral_component": behavioral_component.round(5),
        "exposure_component": exposure_component.round(5),
        "combined_score": combined_score.round(5),
        "risk_tier": risk_tier,
    })

    diagnostics = {
        "exposure_cap_usd": exposure_cap,
        "tier_cutpoints": tier_cutpoints,
        "xgb_selection_report": selection_report,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "all_df": all_df,  # returned for downstream exposure/threshold analysis (amount, is_fraud, split membership)
        "combined_score": combined_score,
    }
    return risk_scores_df, diagnostics
