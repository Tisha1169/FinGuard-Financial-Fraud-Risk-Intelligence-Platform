"""Tests for Phase 7 risk scoring: component computation, combination
math, tier assignment, and the financial exposure simulation.

Unit tests cover the pure math with hand-built data. Integration tests
(skipped without a live DATABASE_URL) validate the full pipeline against
real data: tier/fraud-rate separation, the exposure cap's train-only
provenance, and that the Isolation Forest used here is the same
train-only-fit artifact Phase 6 was audited for (not Phase 5's
whole-dataset version).
"""
import numpy as np
import pandas as pd
import pytest

from database.connection import check_connection, get_engine
from risk_scoring import components, config
from risk_scoring.exposure import portfolio_impact_at_threshold, threshold_sweep

# ---------------------------------------------------------------------------
# Unit tests: component math, hand-computed
# ---------------------------------------------------------------------------


def test_component_weights_sum_to_one():
    assert abs(sum(config.COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9


def test_rules_component_zero_when_no_rules_fired():
    df = pd.DataFrame({"max_rule_severity": [-1], "rules_fired_count": [0]})
    result = components.compute_rules_component(df)
    assert result.iloc[0] == 0.0


def test_rules_component_matches_severity_base_score():
    df = pd.DataFrame({"max_rule_severity": [0, 1, 2, 3], "rules_fired_count": [1, 1, 1, 1]})
    result = components.compute_rules_component(df)
    assert list(result) == pytest.approx([0.25, 0.50, 0.75, 1.00])


def test_rules_component_multi_rule_bonus_is_capped():
    df = pd.DataFrame({"max_rule_severity": [3], "rules_fired_count": [10]})
    result = components.compute_rules_component(df)
    # base 1.0 already at max; bonus capped at 0.20 but overall score clipped to 1.0
    assert result.iloc[0] == pytest.approx(1.0)

    df2 = pd.DataFrame({"max_rule_severity": [0], "rules_fired_count": [10]})
    result2 = components.compute_rules_component(df2)
    # base 0.25 + bonus capped at 0.20 = 0.45, not 0.25 + 0.05*9=0.70
    assert result2.iloc[0] == pytest.approx(0.45)


def test_behavioral_component_noisy_or():
    df = pd.DataFrame({"behavioral_anomaly_score": [0.0, 1.0, 0.5], "isolation_forest_score": [0.0, 0.0, 0.5]})
    result = components.compute_behavioral_component(df)
    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(1.0)
    assert result.iloc[2] == pytest.approx(1 - 0.5 * 0.5)  # 0.75


def test_exposure_component_capped_at_one():
    df = pd.DataFrame({"amount": [10.0, 100.0, 1000.0]})
    result = components.compute_exposure_component(df, exposure_cap=100.0)
    assert list(result) == pytest.approx([0.1, 1.0, 1.0])


def test_combined_score_weighted_sum():
    ml = pd.Series([1.0])
    rules = pd.Series([0.0])
    behavioral = pd.Series([0.0])
    exposure = pd.Series([0.0])
    result = components.compute_combined_score(ml, rules, behavioral, exposure)
    assert result.iloc[0] == pytest.approx(config.COMPONENT_WEIGHTS["ml"])


def test_combined_score_all_ones_equals_one():
    ones = pd.Series([1.0])
    result = components.compute_combined_score(ones, ones, ones, ones)
    assert result.iloc[0] == pytest.approx(1.0)


def test_assign_risk_tier_boundaries():
    cutpoints = {"CRITICAL": 0.8, "HIGH": 0.5, "MEDIUM": 0.2}
    scores = pd.Series([0.9, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1])
    tiers = components.assign_risk_tier(scores, cutpoints)
    assert list(tiers) == ["CRITICAL", "CRITICAL", "HIGH", "HIGH", "MEDIUM", "MEDIUM", "LOW"]


# ---------------------------------------------------------------------------
# Unit tests: financial exposure simulation, hand-computed
# ---------------------------------------------------------------------------


def test_portfolio_impact_hand_computed():
    df = pd.DataFrame({
        "combined_score": [0.9, 0.8, 0.3, 0.1],
        "amount": [100.0, 50.0, 200.0, 10.0],
        "is_fraud": [True, False, True, False],
    })
    # threshold 0.5 -> alert idx0 (fraud, TP) and idx1 (not fraud, FP); idx2 is a missed fraud (FN)
    result = portfolio_impact_at_threshold(df, threshold=0.5)
    assert result["n_alerts"] == 2
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["total_investigation_cost_usd"] == pytest.approx(2 * config.INVESTIGATION_COST_USD)
    assert result["estimated_loss_prevented_usd"] == pytest.approx(100.0 * config.FRAUD_RECOVERY_RATE_IF_CAUGHT)
    assert result["unprevented_loss_usd"] == pytest.approx(200.0)  # idx2's amount, missed
    expected_net = (100.0 * config.FRAUD_RECOVERY_RATE_IF_CAUGHT) - (2 * config.INVESTIGATION_COST_USD) - 200.0
    assert result["net_expected_impact_usd"] == pytest.approx(expected_net)


def test_threshold_sweep_returns_one_row_per_distinct_threshold():
    df = pd.DataFrame({
        "combined_score": np.linspace(0, 1, 100),
        "amount": np.full(100, 50.0),
        "is_fraud": [i % 20 == 0 for i in range(100)],
    })
    sweep = threshold_sweep(df, n_points=10)
    assert len(sweep) > 0
    assert len(sweep) <= 10
    assert "net_expected_impact_usd" in sweep.columns


# ---------------------------------------------------------------------------
# Integration tests: real data
# ---------------------------------------------------------------------------

healthy, _ = check_connection()
pytestmark_db = pytest.mark.skipif(not healthy, reason="requires a live DATABASE_URL (see .env.example)")


@pytest.fixture(scope="module")
def risk_scores_result():
    from risk_scoring.pipeline import build_risk_scores

    engine = get_engine()
    return build_risk_scores(engine)


@pytestmark_db
def test_risk_scores_cover_every_transaction(risk_scores_result):
    risk_scores_df, diag = risk_scores_result
    assert len(risk_scores_df) == diag["n_train"] + diag["n_val"] + diag["n_test"]
    assert risk_scores_df["transaction_id"].is_unique


@pytestmark_db
def test_combined_score_bounded_zero_one(risk_scores_result):
    risk_scores_df, _ = risk_scores_result
    assert risk_scores_df["combined_score"].between(0, 1).all()
    for col in ["ml_component", "rules_component", "behavioral_component", "exposure_component"]:
        assert risk_scores_df[col].between(0, 1).all()


@pytestmark_db
def test_risk_tiers_have_valid_labels(risk_scores_result):
    risk_scores_df, _ = risk_scores_result
    assert set(risk_scores_df["risk_tier"].unique()).issubset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


@pytestmark_db
def test_risk_tiers_separate_fraud_monotonically(risk_scores_result):
    """The core sanity check for the whole scoring methodology: fraud rate
    must increase monotonically CRITICAL > HIGH > MEDIUM > LOW. If tiers
    didn't separate fraud this cleanly, the combined score wouldn't be
    doing its job regardless of how principled the weights look on paper.
    """
    risk_scores_df, diag = risk_scores_result
    merged = diag["all_df"].merge(risk_scores_df[["transaction_id", "risk_tier"]], on="transaction_id")
    fraud_rate_by_tier = merged.groupby("risk_tier")["is_fraud"].mean()
    assert fraud_rate_by_tier["CRITICAL"] > fraud_rate_by_tier["HIGH"]
    assert fraud_rate_by_tier["HIGH"] > fraud_rate_by_tier["MEDIUM"]
    assert fraud_rate_by_tier["MEDIUM"] > fraud_rate_by_tier["LOW"]


@pytestmark_db
def test_exposure_cap_uses_train_amounts_only(risk_scores_result):
    """Regression check: the exposure cap must be a statistic of the
    TRAINING period's amounts, not the full dataset (which would leak
    validation/test amount distribution into a component applied to all
    three splits).
    """
    _, diag = risk_scores_result
    all_df = diag["all_df"]
    train_amounts = all_df.iloc[: diag["n_train"]]["amount"]
    expected_cap = float(train_amounts.quantile(config.EXPOSURE_CAP_PERCENTILE))
    assert diag["exposure_cap_usd"] == pytest.approx(expected_cap)


@pytestmark_db
def test_tier_cutpoints_derived_from_validation_only(risk_scores_result):
    """The tier boundaries must come from the validation split's score
    quantiles, not train's or the full population's - otherwise tiers
    would be calibrated against data the model was fit on (train) or
    against test, which the methodology reserves for a single unbiased
    look, not threshold calibration.
    """
    _, diag = risk_scores_result
    all_df = diag["all_df"]
    combined_score = diag["combined_score"]
    val_slice = combined_score.iloc[diag["n_train"]: diag["n_train"] + diag["n_val"]]

    for tier, q in config.TIER_QUANTILES.items():
        expected = float(val_slice.quantile(q))
        assert diag["tier_cutpoints"][tier] == pytest.approx(expected)
