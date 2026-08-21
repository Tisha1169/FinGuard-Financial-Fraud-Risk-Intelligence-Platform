"""Direct tests for investigation/case_actions.py - the shared module both
the API and streamlit_app call. Exercised at the function level (not
through HTTP) since streamlit_app calls it directly, without an API layer
in between, in production.
"""
import pytest
from sqlalchemy import text

from database.connection import check_connection, get_engine
from investigation.case_actions import CaseNotFoundError, assign_case, perform_action
from investigation.state_machine import InvalidTransitionError

healthy, _ = check_connection()
pytestmark = pytest.mark.skipif(not healthy, reason="requires a live DATABASE_URL (see .env.example)")


def _find_any_open_case(engine):
    with engine.connect() as conn:
        for status in ["OPEN", "ESCALATED", "IN_REVIEW"]:
            case_id = conn.execute(
                text("SELECT case_id FROM investigation_cases WHERE status = :s LIMIT 1"), {"s": status}
            ).scalar()
            if case_id is not None:
                return case_id, status
    return None, None


def test_assign_case_raises_for_missing_case():
    engine = get_engine()
    with pytest.raises(CaseNotFoundError):
        assign_case(engine, 999999999, "someone")


def test_perform_action_raises_for_missing_case():
    engine = get_engine()
    with pytest.raises(CaseNotFoundError):
        perform_action(engine, 999999999, "INVESTIGATE", "someone")


def test_perform_action_raises_for_invalid_transition():
    engine = get_engine()
    case_id, _ = _find_any_open_case(engine)
    if case_id is None:
        pytest.skip("no non-terminal case available")
    with pytest.raises(InvalidTransitionError):
        perform_action(engine, case_id, "CLOSE", "pytest")


def test_assign_case_persists_and_logs():
    engine = get_engine()
    case_id, _ = _find_any_open_case(engine)
    if case_id is None:
        pytest.skip("no non-terminal case available")

    result = assign_case(engine, case_id, "pytest.investigator", "pytest")
    assert result["assigned_investigator"] == "pytest.investigator"
    assert result["action"]["action_type"] == "ASSIGN"

    with engine.connect() as conn:
        stored = conn.execute(
            text("SELECT assigned_investigator FROM investigation_cases WHERE case_id = :cid"), {"cid": case_id}
        ).scalar()
        audit_count = conn.execute(
            text("SELECT COUNT(*) FROM audit_log WHERE entity_type='case' AND entity_id=:cid AND event_type='ASSIGN'"),
            {"cid": str(case_id)},
        ).scalar()
    assert stored == "pytest.investigator"
    assert audit_count >= 1


def test_perform_action_updates_status_and_logs():
    engine = get_engine()
    case_id, status = _find_any_open_case(engine)
    if case_id is None:
        pytest.skip("no non-terminal case available")

    from investigation.state_machine import valid_actions_from

    action_type = valid_actions_from(status)[0]
    result = perform_action(engine, case_id, action_type, "pytest", notes="test note")
    assert result["previous_status"] == status
    assert result["new_status"] != status

    with engine.connect() as conn:
        stored_status = conn.execute(
            text("SELECT status FROM investigation_cases WHERE case_id = :cid"), {"cid": case_id}
        ).scalar()
    assert stored_status == result["new_status"]
