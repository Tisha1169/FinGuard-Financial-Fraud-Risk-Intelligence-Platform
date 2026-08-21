"""Shared case-mutation logic: assign an investigator, apply a status-
transitioning action. Used by BOTH api/routers/cases.py and
streamlit_app's investigation queue - not duplicated between them.

This matters architecturally: in production, Streamlit talks to Postgres
directly (see docs/architecture.md's deployment note - FastAPI isn't
hosted separately), so the dashboard's case-action buttons can't call the
API over HTTP. Rather than re-implement the same SQL + state-machine
validation a second time inside streamlit_app, both consumers call these
same functions.
"""
import datetime
import json

from sqlalchemy import text
from sqlalchemy.engine import Engine

from investigation.state_machine import InvalidTransitionError, apply_action


class CaseNotFoundError(ValueError):
    pass


def _fetch_case_status(conn, case_id: int) -> str:
    status = conn.execute(
        text("SELECT status FROM investigation_cases WHERE case_id = :cid"), {"cid": case_id}
    ).scalar()
    if status is None:
        raise CaseNotFoundError(f"case {case_id} not found")
    return status


def assign_case(engine: Engine, case_id: int, investigator: str, performed_by: str = "system") -> dict:
    with engine.begin() as conn:
        _fetch_case_status(conn, case_id)  # raises CaseNotFoundError if missing
        now = datetime.datetime.now(datetime.timezone.utc)

        conn.execute(
            text("UPDATE investigation_cases SET assigned_investigator = :inv WHERE case_id = :cid"),
            {"inv": investigator, "cid": case_id},
        )
        action = conn.execute(
            text(
                "INSERT INTO investigation_actions (case_id, action_type, performed_by, notes, performed_at) "
                "VALUES (:cid, 'ASSIGN', :performed_by, :notes, :ts) "
                "RETURNING action_id, action_type, performed_by, notes, performed_at"
            ),
            {"cid": case_id, "performed_by": performed_by, "notes": f"assigned to {investigator}", "ts": now},
        ).mappings().first()
        conn.execute(
            text(
                "INSERT INTO audit_log (entity_type, entity_id, event_type, event_payload, performed_by, performed_at) "
                "VALUES ('case', :cid, 'ASSIGN', CAST(:payload AS jsonb), :performed_by, :ts)"
            ),
            {"cid": str(case_id), "payload": json.dumps({"investigator": investigator}), "performed_by": performed_by, "ts": now},
        )

    return {"case_id": case_id, "assigned_investigator": investigator, "action": dict(action)}


def perform_action(engine: Engine, case_id: int, action_type: str, performed_by: str, notes: str | None = None) -> dict:
    """Raises CaseNotFoundError if the case doesn't exist, or
    InvalidTransitionError if action_type isn't valid from the case's
    current status (see investigation/state_machine.py).
    """
    with engine.begin() as conn:
        current_status = _fetch_case_status(conn, case_id)
        new_status = apply_action(current_status, action_type)  # raises InvalidTransitionError

        now = datetime.datetime.now(datetime.timezone.utc)
        is_resolution = action_type in {"CONFIRM_FRAUD", "MARK_FALSE_POSITIVE"}
        is_close = action_type == "CLOSE"

        set_clauses = ["status = :status"]
        params = {"status": new_status, "cid": case_id}
        if is_resolution:
            set_clauses.append("resolution = :resolution")
            params["resolution"] = new_status
        if is_close:
            set_clauses.append("resolved_at = :resolved_at")
            params["resolved_at"] = now

        conn.execute(text(f"UPDATE investigation_cases SET {', '.join(set_clauses)} WHERE case_id = :cid"), params)

        action = conn.execute(
            text(
                "INSERT INTO investigation_actions (case_id, action_type, performed_by, notes, performed_at) "
                "VALUES (:cid, :action_type, :performed_by, :notes, :ts) "
                "RETURNING action_id, action_type, performed_by, notes, performed_at"
            ),
            {"cid": case_id, "action_type": action_type, "performed_by": performed_by, "notes": notes, "ts": now},
        ).mappings().first()
        conn.execute(
            text(
                "INSERT INTO audit_log (entity_type, entity_id, event_type, event_payload, performed_by, performed_at) "
                "VALUES ('case', :cid, :event_type, CAST(:payload AS jsonb), :performed_by, :ts)"
            ),
            {
                "cid": str(case_id), "event_type": action_type,
                "payload": json.dumps({"previous_status": current_status, "new_status": new_status, "notes": notes}),
                "performed_by": performed_by, "ts": now,
            },
        )

    return {"case_id": case_id, "previous_status": current_status, "new_status": new_status, "action": dict(action)}
