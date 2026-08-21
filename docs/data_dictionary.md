# FinGuard Data Dictionary

Source of truth for schema is [`sql/schema.sql`](../sql/schema.sql). This
document explains grain, keys, and rationale for each table.

## Dimensions

### dim_customer
Grain: one row per customer.
PK: `customer_id`. `customer_uid` is the stable external identifier used by
the rest of the app.
`risk_segment` (STANDARD/WATCHLIST/VIP) is a static customer-level risk
signal, distinct from the per-transaction risk score.

### dim_merchant
Grain: one row per merchant.
PK: `merchant_id`. `mcc_code` (merchant category code) drives merchant
behavioral baselines and some rule evidence (e.g. sudden MCC-mix shift).

### dim_device
Grain: one row per device fingerprint.
PK: `device_id`. Deliberately **not** linked 1:1 to a customer — the
customer↔device relationship is observed per-transaction on
`fact_transactions.device_id`, because a device being reused across
customers is itself a fraud signal (account takeover, device farms).

### dim_location
Grain: one row per distinct (city, country) pair used in transactions.
PK: `location_id`. City-level granularity is sufficient for the
geo-distance/impossible-travel rule; UNIQUE(city, country) prevents
duplicate geo points.

## Facts

### fact_transactions
Grain: one row per transaction — the central fact table.
PK: `transaction_id`. FKs to customer, merchant, device, location.
Indexed on `(customer_id, transaction_ts)` and `(merchant_id, transaction_ts)`
because nearly every downstream engine window-aggregates by customer or
merchant over time.
`payment_instrument_last4` stores only the last 4 digits — no full PAN is
ever stored, matching real-world PCI-DSS practice even in a synthetic
dataset.

### fact_customer_daily_metrics / fact_merchant_daily_metrics
Grain: one row per customer/merchant per day.
Precomputed rolling baselines (90-day avg/stddev, distinct counts) so the
rules and statistical engines aren't recomputing full-history window
aggregates on every transaction scored — this is what a real streaming
fraud system does (precomputed feature store) rather than naive on-the-fly
aggregation.

## Detection Output

### rules_triggered
Grain: one row per rule firing per transaction (many-to-one with
transactions — a single transaction can trip multiple rules).
`evidence` is JSONB holding the structured facts behind the firing (e.g.
`{"txn_count_5min": 6, "threshold": 5}`) so the UI can render a specific
reason, not just a rule name.

### ground_truth_fraud
Grain: one row per labeled transaction.
Kept as its own table, separate from `model_predictions`, specifically so
training/evaluation code cannot accidentally treat the label as a feature
or leak it into scoring. `is_synthetic_label` is always `true` in this
project — there is no real fraud data — and this is stated wherever the
label is surfaced in the UI or docs.

### model_predictions
Grain: one row per (transaction, model version) scored.
Multiple model versions can score the same transaction over time, which is
what `model_monitoring_metrics` compares across.

### risk_scores
Grain: one row per transaction — the final combined assessment.
Stores each normalized component (`ml_component`, `rules_component`,
`behavioral_component`, `exposure_component`) alongside `combined_score` so
every score is explainable by construction, not just by re-running SHAP.

## Alerts & Case Management

### fraud_alerts
Grain: one row per alert, after deduplication. `dedup_group_id` links
alerts that were collapsed from multiple related transactions (e.g. a
velocity burst) into one investigator-facing alert.

### investigation_cases
Grain: one row per case. Usually 1:1 with an alert, but the model allows a
case to aggregate multiple alerts for the same customer via repeated
`alert_id` references at the application layer.
`status` enum: OPEN → IN_REVIEW → (ESCALATED) → CONFIRMED_FRAUD |
FALSE_POSITIVE → CLOSED. `sla_deadline` is set at creation based on risk
tier.

### investigation_actions
Grain: one row per investigator action. Append-only, forms the case-level
audit trail (a scoped subset of the system-wide `audit_log`).

## Monitoring & Audit

### model_monitoring_metrics
Grain: one row per (model, version, period). Tracks precision/recall/PR-AUC
and PSI drift over time so model degradation is visible without re-running
a full evaluation notebook.

### audit_log
Grain: one row per auditable system event, append-only, across all entity
types (`case`, `alert`, `model`, `rule`). This is the compliance-style
trail a real financial institution would require.
