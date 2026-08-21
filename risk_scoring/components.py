"""Computes the four normalized [0,1] components of the combined risk
score from a fully-assembled dataframe (feature matrix + ml_probability +
isolation_forest_score already attached by risk_scoring/pipeline.py).
"""
import numpy as np
import pandas as pd

from risk_scoring import config


def compute_rules_component(df: pd.DataFrame) -> pd.Series:
    base = df["max_rule_severity"].map(config.SEVERITY_BASE_SCORE).astype(float)
    extra_rules = (df["rules_fired_count"] - 1).clip(lower=0)
    bonus = (extra_rules * config.MULTI_RULE_BONUS_PER_EXTRA_RULE).clip(upper=config.MULTI_RULE_BONUS_CAP)
    bonus = bonus.where(df["rules_fired_count"] > 0, 0.0)
    return (base + bonus).clip(upper=1.0)


def compute_behavioral_component(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [config.combine_behavioral(a, b) for a, b in zip(df["behavioral_anomaly_score"], df["isolation_forest_score"])],
        index=df.index,
    )


def compute_exposure_component(df: pd.DataFrame, exposure_cap: float) -> pd.Series:
    return (df["amount"] / exposure_cap).clip(upper=1.0)


def compute_combined_score(ml: pd.Series, rules: pd.Series, behavioral: pd.Series, exposure: pd.Series) -> pd.Series:
    w = config.COMPONENT_WEIGHTS
    return w["ml"] * ml + w["rules"] * rules + w["behavioral"] * behavioral + w["exposure"] * exposure


def assign_risk_tier(combined_score: pd.Series, tier_cutpoints: dict) -> pd.Series:
    """tier_cutpoints: {"CRITICAL": score_at_p99, "HIGH": score_at_p95, "MEDIUM": score_at_p80}
    - absolute score thresholds derived from validation quantiles, applied
    to any score (train/val/test alike) via simple comparison, not a
    re-computed quantile (which would make tiers dataset-size-dependent
    and re-leak test data into its own tier boundaries).
    """
    conditions = [
        combined_score >= tier_cutpoints["CRITICAL"],
        combined_score >= tier_cutpoints["HIGH"],
        combined_score >= tier_cutpoints["MEDIUM"],
    ]
    choices = ["CRITICAL", "HIGH", "MEDIUM"]
    return pd.Series(np.select(conditions, choices, default="LOW"), index=combined_score.index)
