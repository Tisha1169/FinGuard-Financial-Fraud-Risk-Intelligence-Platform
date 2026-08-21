"""Pydantic request/response models. Kept close to the actual DB columns
rather than over-abstracted - this is a thin API over the schema, and the
schema (sql/schema.sql) is already the documented source of truth for what
these fields mean.
"""
import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    database: str


class TransactionOut(BaseModel):
    transaction_id: int
    transaction_uid: str
    customer_id: int
    merchant_id: int
    transaction_ts: datetime.datetime
    amount: float
    currency: str
    channel: str
    status: str
    risk_tier: Optional[str] = None
    combined_score: Optional[float] = None


class AlertOut(BaseModel):
    alert_id: int
    transaction_id: int
    customer_id: int
    risk_tier: str
    combined_score: float
    financial_exposure: float
    dedup_group_id: Optional[str] = None
    created_at: datetime.datetime
    case_id: Optional[int] = None
    case_status: Optional[str] = None


class CaseOut(BaseModel):
    case_id: int
    alert_id: int
    customer_id: int
    status: str
    assigned_investigator: Optional[str] = None
    sla_deadline: datetime.datetime
    resolution: Optional[str] = None
    created_at: datetime.datetime
    resolved_at: Optional[datetime.datetime] = None
    risk_tier: Optional[str] = None
    financial_exposure: Optional[float] = None


class RuleEvidence(BaseModel):
    rule_id: str
    rule_description: str
    severity: str
    evidence: dict[str, Any]
    triggered_at: datetime.datetime


class CaseActionOut(BaseModel):
    action_id: int
    action_type: str
    performed_by: str
    notes: Optional[str] = None
    performed_at: datetime.datetime


class CaseDetailOut(BaseModel):
    case: CaseOut
    transaction: TransactionOut
    customer_risk_segment: Optional[str] = None
    merchant_risk_category: Optional[str] = None
    risk_score: Optional[dict[str, Any]] = None
    rules_triggered: list[RuleEvidence] = Field(default_factory=list)
    action_history: list[CaseActionOut] = Field(default_factory=list)
    valid_next_actions: list[str] = Field(default_factory=list)


class RiskDetailOut(BaseModel):
    transaction_id: int
    ml_component: float
    rules_component: float
    behavioral_component: float
    exposure_component: float
    combined_score: float
    risk_tier: str
    rules_triggered: list[RuleEvidence] = Field(default_factory=list)


class AssignRequest(BaseModel):
    investigator: str
    performed_by: str = "system"


class ActionRequest(BaseModel):
    action_type: str
    performed_by: str
    notes: Optional[str] = None


class ActionResponse(BaseModel):
    case_id: int
    previous_status: str
    new_status: str
    action: CaseActionOut


class MetricsOut(BaseModel):
    total_transactions: int
    total_alerts: int
    total_cases: int
    open_cases: int
    resolved_cases: int
    sla_compliance_rate: Optional[float] = None
    fraud_confirmation_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    risk_tier_distribution: dict[str, int]
