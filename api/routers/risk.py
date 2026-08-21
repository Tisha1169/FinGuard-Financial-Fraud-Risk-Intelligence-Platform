from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.engine import Engine

from api.dependencies import db_engine, parse_rule_row

router = APIRouter(tags=["risk"])


@router.get("/risk/{transaction_id}")
def get_risk_detail(transaction_id: int, engine: Engine = Depends(db_engine)):
    """The combined risk score breakdown plus the rule evidence behind
    it - "why did this transaction get this score," reading directly from
    risk_scores (Phase 7) and rules_triggered (Phase 4). Does not
    recompute the ML/behavioral components live - those come from the
    batch scoring pipeline (see docs/risk_scoring.md); this endpoint is a
    lookup, not a live scorer.
    """
    with engine.connect() as conn:
        risk_row = conn.execute(
            text(
                "SELECT transaction_id, ml_component, rules_component, behavioral_component, "
                "exposure_component, combined_score, risk_tier FROM risk_scores WHERE transaction_id = :tid"
            ),
            {"tid": transaction_id},
        ).mappings().first()

        if risk_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"no risk score found for transaction {transaction_id} - "
                       f"has scripts/run_risk_scoring.py been run?",
            )

        rules_rows = conn.execute(
            text(
                "SELECT rule_id, rule_description, severity, evidence, triggered_at "
                "FROM rules_triggered WHERE transaction_id = :tid ORDER BY triggered_at"
            ),
            {"tid": transaction_id},
        ).mappings().all()

    return {**dict(risk_row), "rules_triggered": [parse_rule_row(r) for r in rules_rows]}
