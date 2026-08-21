from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.dependencies import db_engine

router = APIRouter(tags=["alerts"])


@router.get("/alerts")
def list_alerts(
    risk_tier: Optional[str] = Query(None, pattern="^(HIGH|CRITICAL)$"),
    dedup_group_id: Optional[str] = None,
    case_status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    engine: Engine = Depends(db_engine),
):
    """Prioritized alert list, joined with the case each alert opened
    (1:1 per docs/investigation_workflow.md) so status is visible without
    a second lookup.
    """
    conditions = []
    params: dict = {"limit": limit, "offset": offset}
    if risk_tier is not None:
        conditions.append("a.risk_tier = :risk_tier")
        params["risk_tier"] = risk_tier
    if dedup_group_id is not None:
        conditions.append("a.dedup_group_id = :dedup_group_id")
        params["dedup_group_id"] = dedup_group_id
    if case_status is not None:
        conditions.append("c.status = :case_status")
        params["case_status"] = case_status

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = text(
        f"""
        SELECT a.alert_id, a.transaction_id, a.customer_id, a.risk_tier, a.combined_score,
               a.financial_exposure, a.dedup_group_id, a.created_at,
               c.case_id, c.status AS case_status
        FROM fraud_alerts a
        LEFT JOIN investigation_cases c ON c.alert_id = a.alert_id
        {where_clause}
        ORDER BY a.combined_score DESC, a.created_at DESC
        LIMIT :limit OFFSET :offset
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().all()
        total = conn.execute(
            text(f"SELECT COUNT(*) FROM fraud_alerts a LEFT JOIN investigation_cases c ON c.alert_id = a.alert_id {where_clause}"),
            params,
        ).scalar()

    return {"total": total, "limit": limit, "offset": offset, "items": [dict(r) for r in rows]}
