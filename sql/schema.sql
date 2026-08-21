-- FinGuard schema
-- Grain of every table is documented inline. This schema targets PostgreSQL
-- (local Docker for dev, Neon for production). Kept deliberately small so a
-- Neon free-tier instance stays well within storage limits.

-- =========================================================================
-- DIMENSIONS
-- =========================================================================

-- Grain: one row per customer.
CREATE TABLE dim_customer (
    customer_id         BIGSERIAL PRIMARY KEY,
    customer_uid        TEXT UNIQUE NOT NULL,          -- stable external id, e.g. "CUST-000001"
    signup_date          DATE NOT NULL,
    home_country          TEXT NOT NULL,
    home_city             TEXT NOT NULL,
    risk_segment          TEXT NOT NULL DEFAULT 'STANDARD', -- STANDARD | WATCHLIST | VIP
    is_synthetic          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Grain: one row per merchant.
CREATE TABLE dim_merchant (
    merchant_id           BIGSERIAL PRIMARY KEY,
    merchant_uid          TEXT UNIQUE NOT NULL,
    merchant_name          TEXT NOT NULL,
    mcc_code                TEXT NOT NULL,               -- merchant category code
    mcc_description         TEXT,
    merchant_country        TEXT NOT NULL,
    risk_category            TEXT NOT NULL DEFAULT 'STANDARD', -- STANDARD | HIGH_RISK
    is_synthetic             BOOLEAN NOT NULL DEFAULT TRUE,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Grain: one row per device fingerprint. A device can be linked to more than
-- one customer over time (shared/reused devices are themselves a fraud
-- signal, e.g. account takeover), so the customer link lives on the fact
-- table, not here.
CREATE TABLE dim_device (
    device_id              BIGSERIAL PRIMARY KEY,
    device_uid              TEXT UNIQUE NOT NULL,
    device_type               TEXT NOT NULL,             -- MOBILE | DESKTOP | POS | ATM
    os                          TEXT,
    first_seen_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Grain: one row per distinct geo point used in transactions (city-level
-- granularity is enough for distance/velocity calculations here).
CREATE TABLE dim_location (
    location_id              BIGSERIAL PRIMARY KEY,
    city                        TEXT NOT NULL,
    country                      TEXT NOT NULL,
    latitude                      NUMERIC(9,6) NOT NULL,
    longitude                      NUMERIC(9,6) NOT NULL,
    UNIQUE (city, country)
);

-- =========================================================================
-- FACTS
-- =========================================================================

-- Grain: one row per transaction. The core fact table everything else joins to.
CREATE TABLE fact_transactions (
    transaction_id           BIGSERIAL PRIMARY KEY,
    transaction_uid            TEXT UNIQUE NOT NULL,
    customer_id                  BIGINT NOT NULL REFERENCES dim_customer(customer_id),
    merchant_id                   BIGINT NOT NULL REFERENCES dim_merchant(merchant_id),
    device_id                      BIGINT REFERENCES dim_device(device_id),
    location_id                     BIGINT REFERENCES dim_location(location_id),
    transaction_ts                    TIMESTAMPTZ NOT NULL,
    amount                              NUMERIC(14,2) NOT NULL,
    currency                             TEXT NOT NULL DEFAULT 'USD',
    channel                                TEXT NOT NULL,   -- CARD_PRESENT | ECOM | POS | ATM | WALLET
    payment_instrument_last4                 TEXT,           -- last 4 digits only, never full PAN
    status                                     TEXT NOT NULL, -- APPROVED | DECLINED | FAILED
    created_at                                   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_txn_customer_ts ON fact_transactions (customer_id, transaction_ts);
CREATE INDEX idx_txn_merchant_ts ON fact_transactions (merchant_id, transaction_ts);
CREATE INDEX idx_txn_ts ON fact_transactions (transaction_ts);

-- Grain: one row per customer per calendar day. Precomputed rolling
-- baselines so the rules/statistical engines don't recompute window
-- aggregates over the full history on every scoring call.
CREATE TABLE fact_customer_daily_metrics (
    customer_id                BIGINT NOT NULL REFERENCES dim_customer(customer_id),
    metric_date                  DATE NOT NULL,
    txn_count                      INT NOT NULL DEFAULT 0,
    txn_amount_sum                  NUMERIC(14,2) NOT NULL DEFAULT 0,
    txn_amount_avg_90d                NUMERIC(14,2),
    txn_amount_stddev_90d               NUMERIC(14,2),
    distinct_merchants_30d                INT,
    distinct_devices_30d                    INT,
    PRIMARY KEY (customer_id, metric_date)
);

-- Grain: one row per merchant per calendar day.
CREATE TABLE fact_merchant_daily_metrics (
    merchant_id                BIGINT NOT NULL REFERENCES dim_merchant(merchant_id),
    metric_date                  DATE NOT NULL,
    txn_count                      INT NOT NULL DEFAULT 0,
    txn_amount_sum                  NUMERIC(14,2) NOT NULL DEFAULT 0,
    avg_txn_amount_90d                NUMERIC(14,2),
    chargeback_rate_90d                 NUMERIC(6,4),
    PRIMARY KEY (merchant_id, metric_date)
);

-- =========================================================================
-- DETECTION OUTPUT
-- =========================================================================

-- Grain: one row per rule firing per transaction (a transaction can trigger
-- multiple rules).
CREATE TABLE rules_triggered (
    rule_trigger_id           BIGSERIAL PRIMARY KEY,
    transaction_id               BIGINT NOT NULL REFERENCES fact_transactions(transaction_id),
    rule_id                        TEXT NOT NULL,          -- e.g. "R1_VELOCITY"
    rule_description                 TEXT NOT NULL,
    severity                           TEXT NOT NULL,       -- LOW | MEDIUM | HIGH | CRITICAL
    evidence                             JSONB NOT NULL,     -- structured evidence, e.g. {"txn_count_5min": 6}
    triggered_at                           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_rules_triggered_txn ON rules_triggered (transaction_id);

-- Grain: one row per transaction scored by ground truth (synthetic labels
-- only). Kept separate from model_predictions so training/eval code can
-- never accidentally join label to itself as a feature.
CREATE TABLE ground_truth_fraud (
    transaction_id            BIGINT PRIMARY KEY REFERENCES fact_transactions(transaction_id),
    is_fraud                    BOOLEAN NOT NULL,
    fraud_typology                TEXT,                    -- e.g. "CARD_TESTING", "ACCOUNT_TAKEOVER"
    is_synthetic_label              BOOLEAN NOT NULL DEFAULT TRUE,
    labeled_at                        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Grain: one row per transaction per model version scored.
CREATE TABLE model_predictions (
    prediction_id             BIGSERIAL PRIMARY KEY,
    transaction_id               BIGINT NOT NULL REFERENCES fact_transactions(transaction_id),
    model_name                     TEXT NOT NULL,           -- "xgboost_v1", "logreg_baseline"
    model_version                    TEXT NOT NULL,
    fraud_probability                  NUMERIC(6,5) NOT NULL,
    top_features                         JSONB,               -- SHAP top contributors, human-readable
    scored_at                              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_predictions_txn ON model_predictions (transaction_id);

-- Grain: one row per transaction — the combined, final risk assessment.
CREATE TABLE risk_scores (
    transaction_id            BIGINT PRIMARY KEY REFERENCES fact_transactions(transaction_id),
    ml_component                 NUMERIC(6,5) NOT NULL,
    rules_component                NUMERIC(6,5) NOT NULL,
    behavioral_component             NUMERIC(6,5) NOT NULL,
    exposure_component                 NUMERIC(6,5) NOT NULL,
    combined_score                       NUMERIC(6,5) NOT NULL,
    risk_tier                              TEXT NOT NULL,     -- LOW | MEDIUM | HIGH | CRITICAL
    scored_at                                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_risk_scores_tier ON risk_scores (risk_tier);

-- =========================================================================
-- ALERTS & CASE MANAGEMENT
-- =========================================================================

-- Grain: one row per alert (post-deduplication; multiple related
-- transactions can roll into one alert).
CREATE TABLE fraud_alerts (
    alert_id                  BIGSERIAL PRIMARY KEY,
    transaction_id                BIGINT NOT NULL REFERENCES fact_transactions(transaction_id),
    customer_id                     BIGINT NOT NULL REFERENCES dim_customer(customer_id),
    risk_tier                         TEXT NOT NULL,
    combined_score                      NUMERIC(6,5) NOT NULL,
    financial_exposure                    NUMERIC(14,2) NOT NULL,
    dedup_group_id                          TEXT,             -- shared id for alerts collapsed together
    created_at                                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_alerts_tier_created ON fraud_alerts (risk_tier, created_at);

-- Grain: one row per case (usually 1:1 with alert, but a case can
-- aggregate multiple alerts for the same customer).
CREATE TABLE investigation_cases (
    case_id                   BIGSERIAL PRIMARY KEY,
    alert_id                     BIGINT NOT NULL REFERENCES fraud_alerts(alert_id),
    customer_id                    BIGINT NOT NULL REFERENCES dim_customer(customer_id),
    status                           TEXT NOT NULL DEFAULT 'OPEN', -- OPEN|IN_REVIEW|ESCALATED|CONFIRMED_FRAUD|FALSE_POSITIVE|CLOSED
    assigned_investigator               TEXT,
    sla_deadline                          TIMESTAMPTZ NOT NULL,
    resolution                              TEXT,
    created_at                                TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at                                 TIMESTAMPTZ
);

CREATE INDEX idx_cases_status ON investigation_cases (status);
CREATE INDEX idx_cases_sla ON investigation_cases (sla_deadline);

-- Grain: one row per investigator action taken on a case.
CREATE TABLE investigation_actions (
    action_id                 BIGSERIAL PRIMARY KEY,
    case_id                       BIGINT NOT NULL REFERENCES investigation_cases(case_id),
    action_type                     TEXT NOT NULL,           -- ASSIGN|INVESTIGATE|ESCALATE|CONFIRM_FRAUD|MARK_FALSE_POSITIVE|CLOSE
    performed_by                       TEXT NOT NULL,
    notes                                 TEXT,
    performed_at                            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_actions_case ON investigation_actions (case_id);

-- =========================================================================
-- MONITORING & AUDIT
-- =========================================================================

-- Grain: one row per model per monitoring period (e.g. per day/week).
CREATE TABLE model_monitoring_metrics (
    metric_id                 BIGSERIAL PRIMARY KEY,
    model_name                    TEXT NOT NULL,
    model_version                    TEXT NOT NULL,
    period_start                        DATE NOT NULL,
    period_end                            DATE NOT NULL,
    precision_at_k                          NUMERIC(6,5),
    recall_at_fpr                             NUMERIC(6,5),
    pr_auc                                      NUMERIC(6,5),
    psi_score                                     NUMERIC(6,5),
    fraud_rate                                      NUMERIC(6,5),
    computed_at                                       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Grain: one row per auditable event, system-wide (append-only).
CREATE TABLE audit_log (
    audit_id                  BIGSERIAL PRIMARY KEY,
    entity_type                   TEXT NOT NULL,            -- 'case' | 'alert' | 'model' | 'rule'
    entity_id                       TEXT NOT NULL,
    event_type                        TEXT NOT NULL,
    event_payload                       JSONB,
    performed_by                          TEXT,
    performed_at                            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_audit_entity ON audit_log (entity_type, entity_id);
