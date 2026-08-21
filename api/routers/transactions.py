from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.dependencies import db_engine

router = APIRouter(tags=["transactions"])


@router.get("/transactions")
def list_transactions(
    customer_id: Optional[int] = None,
    merchant_id: Optional[int] = None,
    risk_tier: Optional[str] = Query(None, pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$"),
    min_amount: Optional[float] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    engine: Engine = Depends(db_engine),
):
    """Filterable, paginated transaction list, joined with the combined
    risk score for each transaction (if risk scoring has been run).
    """
    conditions = []
    params: dict = {"limit": limit, "offset": offset}
    if customer_id is not None:
        conditions.append("t.customer_id = :customer_id")
        params["customer_id"] = customer_id
    if merchant_id is not None:
        conditions.append("t.merchant_id = :merchant_id")
        params["merchant_id"] = merchant_id
    if risk_tier is not None:
        conditions.append("r.risk_tier = :risk_tier")
        params["risk_tier"] = risk_tier
    if min_amount is not None:
        conditions.append("t.amount >= :min_amount")
        params["min_amount"] = min_amount

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = text(
        f"""
        SELECT t.transaction_id, t.transaction_uid, t.customer_id, t.merchant_id,
               t.transaction_ts, t.amount, t.currency, t.channel, t.status,
               r.risk_tier, r.combined_score
        FROM fact_transactions t
        LEFT JOIN risk_scores r ON r.transaction_id = t.transaction_id
        {where_clause}
        ORDER BY t.transaction_ts DESC
        LIMIT :limit OFFSET :offset
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().all()
        total = conn.execute(
            text(f"SELECT COUNT(*) FROM fact_transactions t LEFT JOIN risk_scores r ON r.transaction_id = t.transaction_id {where_clause}"),
            params,
        ).scalar()

    return {"total": total, "limit": limit, "offset": offset, "items": [dict(r) for r in rows]}
