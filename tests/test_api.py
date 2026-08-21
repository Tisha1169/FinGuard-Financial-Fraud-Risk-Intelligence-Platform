"""Integration tests for the FastAPI layer, using FastAPI's TestClient
(in-process, no server process needed). Skipped without a live
DATABASE_URL, like every other DB-dependent test in this suite.

Note: the write-endpoint tests (assign/action) mutate the real local dev
database, same as every other script in this project - the whole dataset
is regenerable (scripts/generate_data.py onward), so this is consistent
with the project's existing "regenerate anytime" approach rather than a
special case. Tests pick a fresh, not-yet-CLOSED case each run to avoid
depending on a specific case's prior state.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from database.connection import check_connection, get_engine

healthy, _ = check_connection()
pytestmark = pytest.mark.skipif(not healthy, reason="requires a live DATABASE_URL (see .env.example)")


@pytest.fixture(scope="module")
def client():
    from api.main import app

    return TestClient(app)


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_transactions_returns_items(client):
    resp = client.get("/transactions?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] > 0
    assert len(body["items"]) == 5


def test_list_transactions_filter_by_risk_tier(client):
    resp = client.get("/transactions?risk_tier=CRITICAL&limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert all(item["risk_tier"] == "CRITICAL" for item in body["items"])


def test_list_transactions_rejects_invalid_risk_tier(client):
    resp = client.get("/transactions?risk_tier=NOT_A_TIER")
    assert resp.status_code == 422


def test_list_transactions_pagination_limit_enforced(client):
    resp = client.get("/transactions?limit=10000")
    assert resp.status_code == 422  # exceeds max limit of 500


def test_list_transactions_filter_by_customer_id(client):
    known_customer = client.get("/transactions?limit=1").json()["items"][0]["customer_id"]
    resp = client.get(f"/transactions?customer_id={known_customer}&limit=50")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] > 0
    assert all(item["customer_id"] == known_customer for item in body["items"])


def test_list_transactions_filter_by_merchant_id(client):
    known_merchant = client.get("/transactions?limit=1").json()["items"][0]["merchant_id"]
    resp = client.get(f"/transactions?merchant_id={known_merchant}&limit=50")
    assert resp.status_code == 200
    body = resp.json()
    assert all(item["merchant_id"] == known_merchant for item in body["items"])


def test_list_transactions_filter_by_min_amount(client):
    resp = client.get("/transactions?min_amount=100&limit=50")
    assert resp.status_code == 200
    body = resp.json()
    assert all(item["amount"] >= 100 for item in body["items"])


def test_list_alerts_only_high_and_critical(client):
    resp = client.get("/alerts?limit=50")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] > 0
    assert all(item["risk_tier"] in {"HIGH", "CRITICAL"} for item in body["items"])


def test_list_alerts_sorted_by_score_descending(client):
    resp = client.get("/alerts?limit=20")
    scores = [item["combined_score"] for item in resp.json()["items"]]
    assert scores == sorted(scores, reverse=True)


def test_list_cases_returns_items(client):
    resp = client.get("/cases?limit=5")
    assert resp.status_code == 200
    assert resp.json()["total"] > 0


def test_list_cases_filter_by_status(client):
    resp = client.get("/cases?status=CLOSED&limit=10")
    body = resp.json()
    assert all(item["status"] == "CLOSED" for item in body["items"])


def test_list_cases_filter_by_assigned_investigator(client):
    known = client.get("/cases?status=CLOSED&limit=1").json()["items"][0]["assigned_investigator"]
    resp = client.get(f"/cases?assigned_investigator={known}&limit=20")
    assert resp.status_code == 200
    body = resp.json()
    assert all(item["assigned_investigator"] == known for item in body["items"])


def test_list_alerts_filter_by_dedup_group_id(client):
    with_dedup = None
    for item in client.get("/alerts?limit=200").json()["items"]:
        if item["dedup_group_id"]:
            with_dedup = item["dedup_group_id"]
            break
    if with_dedup is None:
        pytest.skip("no alert with a dedup_group_id in the current dataset")
    resp = client.get(f"/alerts?dedup_group_id={with_dedup}")
    assert resp.status_code == 200
    body = resp.json()
    assert all(item["dedup_group_id"] == with_dedup for item in body["items"])


def test_list_alerts_filter_by_case_status(client):
    resp = client.get("/alerts?case_status=CLOSED&limit=20")
    assert resp.status_code == 200
    body = resp.json()
    assert all(item["case_status"] == "CLOSED" for item in body["items"])


def test_get_case_detail_has_all_sections(client):
    case_id = client.get("/cases?limit=1").json()["items"][0]["case_id"]
    resp = client.get(f"/cases/{case_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["case"]["case_id"] == case_id
    assert body["transaction"] is not None
    assert body["risk_score"] is not None
    assert "rules_triggered" in body
    assert "action_history" in body
    assert "valid_next_actions" in body


def test_get_case_detail_404_for_missing_case(client):
    resp = client.get("/cases/999999999")
    assert resp.status_code == 404


def test_get_case_detail_evidence_is_parsed_json_not_a_string(client):
    """Regression test: rules_triggered.evidence must come back as a
    nested JSON object, not a JSON-encoded string - caught during manual
    testing where SQLAlchemy's raw text() queries don't know the column
    is JSONB.
    """
    with_rules = None
    resp = client.get("/cases?limit=50")
    for item in resp.json()["items"]:
        detail = client.get(f"/cases/{item['case_id']}").json()
        if detail["rules_triggered"]:
            with_rules = detail
            break
    assert with_rules is not None, "expected at least one case with rules_triggered in the sample"
    for rule in with_rules["rules_triggered"]:
        assert isinstance(rule["evidence"], dict)


def test_get_risk_detail_for_known_transaction(client):
    engine = get_engine()
    with engine.connect() as conn:
        tid = conn.execute(text("SELECT transaction_id FROM risk_scores WHERE risk_tier='CRITICAL' LIMIT 1")).scalar()
    resp = client.get(f"/risk/{tid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == tid
    assert body["risk_tier"] == "CRITICAL"
    assert 0 <= body["combined_score"] <= 1


def test_get_risk_detail_404_for_unscored_transaction(client):
    resp = client.get("/risk/999999999")
    assert resp.status_code == 404


def test_get_metrics_returns_expected_shape(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_transactions"] > 0
    assert body["total_alerts"] > 0
    assert set(body["risk_tier_distribution"].keys()) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def _find_case_in_status(engine, status: str):
    with engine.connect() as conn:
        return conn.execute(text("SELECT case_id FROM investigation_cases WHERE status = :s LIMIT 1"), {"s": status}).scalar()


def _find_any_open_case(engine):
    """Any non-terminal case, preferring OPEN but falling back to
    ESCALATED/IN_REVIEW - the investigator simulation always logs an
    INVESTIGATE action, so no case is ever left in raw OPEN status after
    scripts/run_investigation_workflow.py runs; tests need to work with
    whatever non-terminal status actually exists.
    """
    for status in ["OPEN", "ESCALATED", "IN_REVIEW"]:
        case_id = _find_case_in_status(engine, status)
        if case_id is not None:
            return case_id, status
    return None, None


def test_assign_case_updates_investigator_and_logs_action(client):
    engine = get_engine()
    case_id, _ = _find_any_open_case(engine)
    if case_id is None:
        pytest.skip("no non-terminal case available to test assignment on")

    resp = client.post(f"/cases/{case_id}/assign", json={"investigator": "test.investigator", "performed_by": "pytest"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["assigned_investigator"] == "test.investigator"
    assert body["action"]["action_type"] == "ASSIGN"

    detail = client.get(f"/cases/{case_id}").json()
    assert detail["case"]["assigned_investigator"] == "test.investigator"
    assert any(a["action_type"] == "ASSIGN" and a["performed_by"] == "pytest" for a in detail["action_history"])


def test_action_endpoint_rejects_invalid_transition(client):
    engine = get_engine()
    case_id, status = _find_any_open_case(engine)
    if case_id is None:
        pytest.skip("no non-terminal case available to test invalid transition on")

    # CLOSE is never valid except from CONFIRMED_FRAUD/FALSE_POSITIVE, so
    # it's an invalid transition from any of OPEN/ESCALATED/IN_REVIEW.
    resp = client.post(f"/cases/{case_id}/action", json={"action_type": "CLOSE", "performed_by": "pytest"})
    assert resp.status_code == 409

    # case status must be unchanged after a rejected action
    detail = client.get(f"/cases/{case_id}").json()
    assert detail["case"]["status"] == status


def test_action_endpoint_full_valid_path_reaches_closed(client):
    from investigation.state_machine import valid_actions_from

    engine = get_engine()
    case_id, status = _find_any_open_case(engine)
    if case_id is None:
        pytest.skip("no non-terminal case available for a full resolution path test")

    # Drive the case toward a resolution using whatever action is valid
    # from its current status - don't hardcode INVESTIGATE first, since
    # the case may already be past that step (e.g. ESCALATED).
    if "MARK_FALSE_POSITIVE" not in valid_actions_from(status):
        step = "INVESTIGATE" if "INVESTIGATE" in valid_actions_from(status) else "ESCALATE"
        r1 = client.post(f"/cases/{case_id}/action", json={"action_type": step, "performed_by": "pytest"})
        assert r1.status_code == 200

    r2 = client.post(f"/cases/{case_id}/action", json={"action_type": "MARK_FALSE_POSITIVE", "performed_by": "pytest", "notes": "test resolution"})
    assert r2.status_code == 200
    assert r2.json()["new_status"] == "FALSE_POSITIVE"

    r3 = client.post(f"/cases/{case_id}/action", json={"action_type": "CLOSE", "performed_by": "pytest"})
    assert r3.status_code == 200
    assert r3.json()["new_status"] == "CLOSED"

    detail = client.get(f"/cases/{case_id}").json()
    assert detail["case"]["status"] == "CLOSED"
    assert detail["case"]["resolution"] == "FALSE_POSITIVE"
    assert detail["case"]["resolved_at"] is not None
    assert detail["valid_next_actions"] == []


def test_action_on_nonexistent_case_returns_404(client):
    resp = client.post("/cases/999999999/action", json={"action_type": "INVESTIGATE", "performed_by": "pytest"})
    assert resp.status_code == 404
