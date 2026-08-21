"""Tests for Phase 8: alert generation, case creation, investigator
simulation, and operational metrics.

Unit tests cover dedup grouping, SLA deadline math, and metric
calculations with hand-built data. Integration tests (skipped without a
live DATABASE_URL) validate the full workflow against real risk_scores:
referential integrity, status-transition validity, and that the
investigator-accuracy simulation produces a fraud confirmation rate
consistent with its documented 90% accuracy assumption.
"""
import pandas as pd
import pytest

from database.connection import check_connection, get_engine
from investigation import config
from investigation.alerts import generate_alerts
from investigation.cases import generate_cases
from investigation.metrics import compute_case_metrics
from investigation.simulate_investigators import simulate

# ---------------------------------------------------------------------------
# Unit tests: alert generation / dedup, hand-built data
# ---------------------------------------------------------------------------


def _txn(tid, cid, ts, amount=100.0):
    return {"transaction_id": tid, "customer_id": cid, "transaction_ts": pd.Timestamp(ts, tz="UTC"), "amount": amount}


def test_only_high_and_critical_tiers_generate_alerts():
    risk_scores = pd.DataFrame({
        "transaction_id": [1, 2, 3, 4],
        "risk_tier": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "combined_score": [0.1, 0.3, 0.6, 0.9],
    })
    txns = pd.DataFrame([_txn(i, 1, "2026-01-01 00:00:00") for i in [1, 2, 3, 4]])
    alerts = generate_alerts(risk_scores, txns)
    assert set(alerts["transaction_id"]) == {3, 4}


def test_dedup_groups_transactions_within_window():
    risk_scores = pd.DataFrame({
        "transaction_id": [1, 2, 3],
        "risk_tier": ["HIGH", "HIGH", "HIGH"],
        "combined_score": [0.6, 0.6, 0.6],
    })
    txns = pd.DataFrame([
        _txn(1, 1, "2026-01-01 00:00:00"),
        _txn(2, 1, "2026-01-01 00:30:00"),  # 30 min later - same group
        _txn(3, 1, "2026-01-01 03:00:00"),  # 2.5h later - new group
    ])
    alerts = generate_alerts(risk_scores, txns)
    groups = alerts.set_index("transaction_id")["dedup_group_id"]
    assert groups[1] == groups[2]
    assert groups[1] != groups[3]


def test_dedup_is_per_customer_not_global():
    risk_scores = pd.DataFrame({"transaction_id": [1, 2], "risk_tier": ["HIGH", "HIGH"], "combined_score": [0.6, 0.6]})
    txns = pd.DataFrame([
        _txn(1, 1, "2026-01-01 00:00:00"),
        _txn(2, 2, "2026-01-01 00:05:00"),  # different customer, same instant
    ])
    alerts = generate_alerts(risk_scores, txns)
    groups = alerts.set_index("transaction_id")["dedup_group_id"]
    assert groups[1] != groups[2]


def test_financial_exposure_equals_transaction_amount():
    risk_scores = pd.DataFrame({"transaction_id": [1], "risk_tier": ["HIGH"], "combined_score": [0.6]})
    txns = pd.DataFrame([_txn(1, 1, "2026-01-01", amount=543.21)])
    alerts = generate_alerts(risk_scores, txns)
    assert alerts.iloc[0]["financial_exposure"] == pytest.approx(543.21)


# ---------------------------------------------------------------------------
# Unit tests: case creation / SLA deadlines
# ---------------------------------------------------------------------------


def test_critical_cases_start_escalated_high_cases_start_open():
    alerts = pd.DataFrame({
        "alert_id": [1, 2], "customer_id": [1, 2], "risk_tier": ["CRITICAL", "HIGH"],
        "created_at": [pd.Timestamp("2026-01-01", tz="UTC")] * 2,
    })
    cases = generate_cases(alerts)
    status_by_alert = cases.set_index("alert_id")["status"]
    assert status_by_alert[1] == "ESCALATED"
    assert status_by_alert[2] == "OPEN"


def test_sla_deadline_matches_configured_hours_per_tier():
    created = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    alerts = pd.DataFrame({
        "alert_id": [1, 2], "customer_id": [1, 2], "risk_tier": ["CRITICAL", "HIGH"],
        "created_at": [created, created],
    })
    cases = generate_cases(alerts)
    deadline_by_alert = cases.set_index("alert_id")["sla_deadline"]
    assert deadline_by_alert[1] == created + pd.Timedelta(hours=config.SLA_HOURS["CRITICAL"])
    assert deadline_by_alert[2] == created + pd.Timedelta(hours=config.SLA_HOURS["HIGH"])


def test_case_ids_are_unique_and_one_per_alert():
    alerts = pd.DataFrame({
        "alert_id": [10, 20, 30], "customer_id": [1, 2, 3], "risk_tier": ["HIGH"] * 3,
        "created_at": [pd.Timestamp("2026-01-01", tz="UTC")] * 3,
    })
    cases = generate_cases(alerts)
    assert cases["case_id"].is_unique
    assert len(cases) == len(alerts)
    assert set(cases["alert_id"]) == set(alerts["alert_id"])


# ---------------------------------------------------------------------------
# Unit tests: metrics, hand-computed
# ---------------------------------------------------------------------------


def test_compute_case_metrics_hand_computed():
    now = pd.Timestamp("2026-01-10", tz="UTC")
    cases = pd.DataFrame({
        "status": ["CLOSED", "CLOSED", "IN_REVIEW", "ESCALATED"],
        "resolution": ["CONFIRMED_FRAUD", "FALSE_POSITIVE", None, None],
        "created_at": [now - pd.Timedelta(hours=10)] * 4,
        "resolved_at": [now - pd.Timedelta(hours=5), now - pd.Timedelta(hours=2), pd.NaT, pd.NaT],
        "sla_deadline": [now - pd.Timedelta(hours=6), now - pd.Timedelta(hours=1), now - pd.Timedelta(hours=1), now + pd.Timedelta(hours=1)],
        "assigned_investigator": ["a", "b", "a", "b"],
    })
    metrics = compute_case_metrics(cases, now)
    assert metrics["total_cases"] == 4
    assert metrics["resolved_cases"] == 2
    assert metrics["open_cases"] == 2
    # case0: resolved at now-5h, deadline now-6h -> resolved AFTER deadline -> breached
    # case1: resolved at now-2h, deadline now-1h -> resolved BEFORE deadline -> compliant
    assert metrics["resolved_cases_breached_sla"] == 1
    assert metrics["sla_compliance_rate"] == pytest.approx(0.5)
    assert metrics["fraud_confirmation_rate"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    # case2 (IN_REVIEW, deadline now-1h) is past due; case3 (ESCALATED, deadline now+1h) is not
    assert metrics["open_cases_already_past_sla"] == 1


def test_compute_case_metrics_handles_zero_resolved_cases():
    now = pd.Timestamp("2026-01-10", tz="UTC")
    cases = pd.DataFrame({
        "status": ["OPEN"], "resolution": [None], "created_at": [now], "resolved_at": [pd.NaT],
        "sla_deadline": [now + pd.Timedelta(hours=1)], "assigned_investigator": ["a"],
    })
    metrics = compute_case_metrics(cases, now)
    assert metrics["sla_compliance_rate"] is None
    assert metrics["avg_resolution_hours"] is None


# ---------------------------------------------------------------------------
# Integration tests: real data
# ---------------------------------------------------------------------------

healthy, _ = check_connection()
pytestmark_db = pytest.mark.skipif(not healthy, reason="requires a live DATABASE_URL (see .env.example)")


@pytest.fixture(scope="module")
def workflow_result():
    engine = get_engine()
    risk_scores_df = pd.read_sql("SELECT transaction_id, risk_tier, combined_score FROM risk_scores", engine)
    if risk_scores_df.empty:
        pytest.skip("risk_scores is empty - run scripts/run_risk_scoring.py first")

    transactions_df = pd.read_sql("SELECT transaction_id, customer_id, transaction_ts, amount FROM fact_transactions", engine)
    transactions_df["transaction_ts"] = pd.to_datetime(transactions_df["transaction_ts"], utc=True)
    transactions_df["amount"] = transactions_df["amount"].astype(float)
    ground_truth_df = pd.read_sql("SELECT transaction_id, is_fraud FROM ground_truth_fraud", engine)
    now_ts = transactions_df["transaction_ts"].max()

    alerts_df = generate_alerts(risk_scores_df, transactions_df)
    cases_df = generate_cases(alerts_df)
    final_cases_df, actions_df, audit_df = simulate(cases_df, alerts_df, ground_truth_df, now_ts)
    return {
        "alerts": alerts_df, "cases": final_cases_df, "actions": actions_df,
        "audit": audit_df, "ground_truth": ground_truth_df, "now_ts": now_ts,
    }


@pytestmark_db
def test_every_case_references_a_valid_alert(workflow_result):
    assert set(workflow_result["cases"]["alert_id"]).issubset(set(workflow_result["alerts"]["alert_id"]))


@pytestmark_db
def test_every_action_references_a_valid_case(workflow_result):
    assert set(workflow_result["actions"]["case_id"]).issubset(set(workflow_result["cases"]["case_id"]))


@pytestmark_db
def test_every_case_has_at_least_assign_and_investigate_actions(workflow_result):
    action_counts = workflow_result["actions"].groupby("case_id")["action_type"].apply(set)
    assert all("ASSIGN" in s and "INVESTIGATE" in s for s in action_counts)


@pytestmark_db
def test_closed_cases_have_a_resolution_and_resolved_at(workflow_result):
    closed = workflow_result["cases"][workflow_result["cases"]["status"] == "CLOSED"]
    assert len(closed) > 0
    assert closed["resolution"].isin(["CONFIRMED_FRAUD", "FALSE_POSITIVE"]).all()
    assert closed["resolved_at"].notna().all()


@pytestmark_db
def test_open_cases_have_no_resolution(workflow_result):
    open_cases = workflow_result["cases"][workflow_result["cases"]["status"] != "CLOSED"]
    assert open_cases["resolution"].isna().all()
    assert open_cases["resolved_at"].isna().all()


@pytestmark_db
def test_resolved_at_never_after_now(workflow_result):
    """A case can't be resolved in the future relative to the simulation's
    'now' instant - this would indicate the backlog logic is broken.
    """
    closed = workflow_result["cases"][workflow_result["cases"]["status"] == "CLOSED"]
    assert (closed["resolved_at"] <= workflow_result["now_ts"]).all()


@pytestmark_db
def test_audit_log_mirrors_every_action(workflow_result):
    assert len(workflow_result["audit"]) == len(workflow_result["actions"])


@pytestmark_db
def test_fraud_confirmation_rate_consistent_with_investigator_accuracy(workflow_result):
    """The core simulation-correctness check: given INVESTIGATOR_ACCURACY,
    the confirmation rate among resolved cases should be close to
    true_fraud_rate * accuracy + (1 - true_fraud_rate) * (1 - accuracy).
    A large deviation would indicate the accuracy simulation isn't wired
    correctly, not just noise.
    """
    from investigation.metrics import compute_case_metrics

    alerts_with_label = workflow_result["alerts"].merge(workflow_result["ground_truth"], on="transaction_id")
    true_fraud_rate = alerts_with_label["is_fraud"].mean()
    expected_confirmation_rate = (
        true_fraud_rate * config.INVESTIGATOR_ACCURACY + (1 - true_fraud_rate) * (1 - config.INVESTIGATOR_ACCURACY)
    )

    metrics = compute_case_metrics(workflow_result["cases"], workflow_result["now_ts"])
    assert metrics["fraud_confirmation_rate"] == pytest.approx(expected_confirmation_rate, abs=0.03)


@pytestmark_db
def test_critical_tier_has_tighter_sla_than_high(workflow_result):
    cases = workflow_result["cases"]
    critical_hours = (cases[cases["risk_tier"] == "CRITICAL"]["sla_deadline"] - cases[cases["risk_tier"] == "CRITICAL"]["created_at"]).dt.total_seconds() / 3600
    high_hours = (cases[cases["risk_tier"] == "HIGH"]["sla_deadline"] - cases[cases["risk_tier"] == "HIGH"]["created_at"]).dt.total_seconds() / 3600
    assert critical_hours.max() < high_hours.min()
