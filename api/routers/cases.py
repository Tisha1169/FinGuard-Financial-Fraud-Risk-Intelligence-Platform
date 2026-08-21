import datetime
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.dependencies import db_engine, parse_rule_row
from api.schemas import ActionRequest, AssignRequest
from investigation.state_machine import InvalidTransitionError, apply_action, valid_actions_from

router = APIRouter(tags=["cases"])


@router.get("/cases")
def list_cases(
    status: Optional[str] = None,
    risk_tier: Optional[str] = Query(None, pattern="^(HIGH|CRITICAL)$"),
    assigned_investigator: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    engine: Engine = Depends(db_engine),
):
    conditions = []
    params: dict = {"limit": limit, "offset": offset}
    if status is not None:
        conditions.append("c.status = :status")
        params["status"] = status
    if risk_tier is not None:
        conditions.append("a.risk_tier = :risk_tier")
        params["risk_tier"] = risk_tier
    if assigned_investigator is not None:
        conditions.append("c.assigned_investigator = :assigned_investigator")
        params["assigned_investigator"] = assigned_investigator

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = text(
        f"""
        SELECT c.case_id, c.alert_id, c.customer_id, c.status, c.assigned_investigator,
               c.sla_deadline, c.resolution, c.created_at, c.resolved_at,
               a.risk_tier, a.financial_exposure
        FROM investigation_cases c
        JOIN fraud_alerts a ON a.alert_id = c.alert_id
        {where_clause}
        ORDER BY c.created_at DESC
        LIMIT :limit OFFSET :offset
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().all()
        total = conn.execute(
            text(f"SELECT COUNT(*) FROM investigation_cases c JOIN fraud_alerts a ON a.alert_id = c.alert_id {where_clause}"),
            params,
        ).scalar()

    return {"total": total, "limit": limit, "offset": offset, "items": [dict(r) for r in rows]}


def _fetch_case_or_404(conn, case_id: int):
    row = conn.execute(
        text(
            """
            SELECT c.case_id, c.alert_id, c.customer_id, c.status, c.assigned_investigator,
                   c.sla_deadline, c.resolution, c.created_at, c.resolved_at,
                   a.risk_tier, a.financial_exposure, a.transaction_id
            FROM investigation_cases c
            JOIN fraud_alerts a ON a.alert_id = c.alert_id
            WHERE c.case_id = :case_id
            """
        ),
        {"case_id": case_id},
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"case {case_id} not found")
    return row


@router.get("/cases/{case_id}")
def get_case_detail(case_id: int, engine: Engine = Depends(db_engine)):
    """Full case detail for the investigation workbench: the case, its
    transaction, customer/merchant context, risk score breakdown, the
    rules that fired (with evidence), and the action history - everything
    an investigator needs on one screen.
    """
    with engine.connect() as conn:
        case_row = _fetch_case_or_404(conn, case_id)
        transaction_id = case_row["transaction_id"]

        txn_row = conn.execute(
            text(
                """
                SELECT t.transaction_id, t.transaction_uid, t.customer_id, t.merchant_id,
                       t.transaction_ts, t.amount, t.currency, t.channel, t.status,
                       c.risk_segment AS customer_risk_segment, m.risk_category AS merchant_risk_category
                FROM fact_transactions t
                JOIN dim_customer c ON c.customer_id = t.customer_id
                JOIN dim_merchant m ON m.merchant_id = t.merchant_id
                WHERE t.transaction_id = :tid
                """
            ),
            {"tid": transaction_id},
        ).mappings().first()

        risk_row = conn.execute(
            text(
                "SELECT ml_component, rules_component, behavioral_component, exposure_component, "
                "combined_score, risk_tier FROM risk_scores WHERE transaction_id = :tid"
            ),
            {"tid": transaction_id},
        ).mappings().first()

        rules_rows = conn.execute(
            text(
                "SELECT rule_id, rule_description, severity, evidence, triggered_at "
                "FROM rules_triggered WHERE transaction_id = :tid ORDER BY triggered_at"
            ),
            {"tid": transaction_id},
        ).mappings().all()

        action_rows = conn.execute(
            text(
                "SELECT action_id, action_type, performed_by, notes, performed_at "
                "FROM investigation_actions WHERE case_id = :cid ORDER BY performed_at"
            ),
            {"cid": case_id},
        ).mappings().all()

    return {
        "case": dict(case_row),
        "transaction": dict(txn_row) if txn_row else None,
        "customer_risk_segment": txn_row["customer_risk_segment"] if txn_row else None,
        "merchant_risk_category": txn_row["merchant_risk_category"] if txn_row else None,
        "risk_score": dict(risk_row) if risk_row else None,
        "rules_triggered": [parse_rule_row(r) for r in rules_rows],
        "action_history": [dict(a) for a in action_rows],
        "valid_next_actions": valid_actions_from(case_row["status"]),
    }


@router.post("/cases/{case_id}/assign")
def assign_case(case_id: int, body: AssignRequest, engine: Engine = Depends(db_engine)):
    with engine.begin() as conn:
        case_row = _fetch_case_or_404(conn, case_id)
        now = datetime.datetime.now(datetime.timezone.utc)

        conn.execute(
            text("UPDATE investigation_cases SET assigned_investigator = :inv WHERE case_id = :cid"),
            {"inv": body.investigator, "cid": case_id},
        )
        result = conn.execute(
            text(
                "INSERT INTO investigation_actions (case_id, action_type, performed_by, notes, performed_at) "
                "VALUES (:cid, 'ASSIGN', :performed_by, :notes, :ts) RETURNING action_id, action_type, performed_by, notes, performed_at"
            ),
            {"cid": case_id, "performed_by": body.performed_by, "notes": f"assigned to {body.investigator}", "ts": now},
        ).mappings().first()
        conn.execute(
            text(
                "INSERT INTO audit_log (entity_type, entity_id, event_type, event_payload, performed_by, performed_at) "
                "VALUES ('case', :cid, 'ASSIGN', CAST(:payload AS jsonb), :performed_by, :ts)"
            ),
            {"cid": str(case_id), "payload": json.dumps({"investigator": body.investigator}), "performed_by": body.performed_by, "ts": now},
        )

    return {"case_id": case_id, "assigned_investigator": body.investigator, "action": dict(result)}


@router.post("/cases/{case_id}/action")
def perform_case_action(case_id: int, body: ActionRequest, engine: Engine = Depends(db_engine)):
    """Applies a status-transitioning action (INVESTIGATE, ESCALATE,
    CONFIRM_FRAUD, MARK_FALSE_POSITIVE, CLOSE) - validated against
    investigation/state_machine.py so an invalid transition (e.g. closing
    an OPEN case directly) is rejected with a clear error rather than
    silently corrupting case state.
    """
    with engine.begin() as conn:
        case_row = _fetch_case_or_404(conn, case_id)
        current_status = case_row["status"]

        try:
            new_status = apply_action(current_status, body.action_type)
        except InvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

        now = datetime.datetime.now(datetime.timezone.utc)
        is_resolution = body.action_type in {"CONFIRM_FRAUD", "MARK_FALSE_POSITIVE"}
        is_close = body.action_type == "CLOSE"

        update_fields = {"status": new_status}
        set_clauses = ["status = :status"]
        if is_resolution:
            set_clauses.append("resolution = :resolution")
            update_fields["resolution"] = new_status  # CONFIRMED_FRAUD or FALSE_POSITIVE
        if is_close:
            set_clauses.append("resolved_at = :resolved_at")
            update_fields["resolved_at"] = now

        update_fields["cid"] = case_id
        conn.execute(
            text(f"UPDATE investigation_cases SET {', '.join(set_clauses)} WHERE case_id = :cid"),
            update_fields,
        )

        result = conn.execute(
            text(
                "INSERT INTO investigation_actions (case_id, action_type, performed_by, notes, performed_at) "
                "VALUES (:cid, :action_type, :performed_by, :notes, :ts) "
                "RETURNING action_id, action_type, performed_by, notes, performed_at"
            ),
            {"cid": case_id, "action_type": body.action_type, "performed_by": body.performed_by, "notes": body.notes, "ts": now},
        ).mappings().first()
        conn.execute(
            text(
                "INSERT INTO audit_log (entity_type, entity_id, event_type, event_payload, performed_by, performed_at) "
                "VALUES ('case', :cid, :event_type, CAST(:payload AS jsonb), :performed_by, :ts)"
            ),
            {
                "cid": str(case_id), "event_type": body.action_type,
                "payload": json.dumps({"previous_status": current_status, "new_status": new_status, "notes": body.notes}),
                "performed_by": body.performed_by, "ts": now,
            },
        )

    return {
        "case_id": case_id, "previous_status": current_status, "new_status": new_status,
        "action": dict(result),
    }
