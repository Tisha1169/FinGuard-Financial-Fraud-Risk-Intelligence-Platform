"""Risk scoring configuration - weights, tier thresholds, and financial
assumptions. All reasoned defaults, documented here and in
docs/risk_scoring.md, not fit against ground truth (the combined score's
job is to be transparent and explainable, not maximally predictive - the
ML model already exists for that; blending in tunable-to-look-good weights
here would undermine the "justified, documented methodology" requirement).
"""

# Combined score = weighted sum of four [0,1] components. Weights sum to 1.
COMPONENT_WEIGHTS = {
    "ml": 0.40,          # Phase 6 XGBoost probability - the strongest validated signal (test PR-AUC 0.793)
    "rules": 0.20,       # Phase 4 rules - interpretable, investigator-trusted evidence
    "behavioral": 0.20,  # Phase 5 statistical + Isolation Forest anomaly signal
    "exposure": 0.20,    # dollar stakes - raises priority for high-value transactions without letting
                         # size alone dominate (a legitimate $50k purchase can't out-rank real fraud evidence)
}
assert abs(sum(COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9

# Sub-combination of the two Phase 5 anomaly signals into behavioral_component,
# via noisy-OR (documented as a Phase 7 decision in docs/statistical_detection.md).
def combine_behavioral(iqr_frequency_score: float, isolation_forest_score: float) -> float:
    a = max(0.0, min(1.0, iqr_frequency_score))
    b = max(0.0, min(1.0, isolation_forest_score))
    return 1 - (1 - a) * (1 - b)


# rules_component: base severity score (0 if no rule fired) plus a small
# bonus for multiple independent rules firing (corroborating evidence).
SEVERITY_BASE_SCORE = {-1: 0.0, 0: 0.25, 1: 0.50, 2: 0.75, 3: 1.00}  # -1 = no rule fired, 0-3 = LOW..CRITICAL
MULTI_RULE_BONUS_PER_EXTRA_RULE = 0.05
MULTI_RULE_BONUS_CAP = 0.20

# exposure_component: amount / EXPOSURE_CAP, clipped to [0,1]. EXPOSURE_CAP
# is the 95th percentile of TRAIN-period transaction amounts, computed at
# runtime and frozen - never a percentile of the full dataset, to avoid any
# distributional lookahead into validation/test (see docs/risk_scoring.md).
EXPOSURE_CAP_PERCENTILE = 0.95

# Risk tiers: quantile cutpoints on VALIDATION-period combined_score
# distribution (never test), so tier boundaries reflect a realistic
# forward-looking score distribution rather than being reverse-engineered
# from the training data's own scores.
TIER_QUANTILES = {
    "CRITICAL": 0.99,  # top 1%
    "HIGH": 0.95,      # next 4%
    "MEDIUM": 0.80,    # next 15%
    # LOW: remaining 80%
}

# --- Financial simulation assumptions (all explicitly simulated/estimated,
# never claimed as real recoveries - see docs/risk_scoring.md) ---
INVESTIGATION_COST_USD = 12.00          # analyst time per case reviewed
FRAUD_RECOVERY_RATE_IF_CAUGHT = 0.80    # share of transaction amount recoverable if flagged before settlement
