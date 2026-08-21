-- Populates fact_customer_daily_metrics and fact_merchant_daily_metrics
-- from fact_transactions. Idempotent (ON CONFLICT DO UPDATE) so it can be
-- re-run safely after new transactions land.
--
-- Design note on leakage: each row for metric_date `d` is an end-of-day
-- snapshot that INCLUDES all of day d's own transactions - it represents
-- "the customer's baseline as of the close of day d". Consumers (rules
-- engine, ML features) must look up the row for the *most recent day
-- strictly before* the transaction being scored, never for the
-- transaction's own day, or same-day leakage results. See
-- features/point_in_time.py, which enforces this at query time.
--
-- Merchant chargeback_rate_90d is the one exception: it is computed
-- strictly EXCLUDING the current day, because real chargebacks are
-- reported by issuers well after the original transaction - even the
-- end-of-day snapshot for day d should not assume same-day chargeback
-- knowledge.

-- =========================================================================
-- CUSTOMER DAILY METRICS
-- =========================================================================
WITH customer_days AS (
    SELECT DISTINCT customer_id, date_trunc('day', transaction_ts)::date AS metric_date
    FROM fact_transactions
),
customer_daily_counts AS (
    SELECT
        customer_id,
        date_trunc('day', transaction_ts)::date AS metric_date,
        COUNT(*) AS txn_count,
        SUM(amount) AS txn_amount_sum
    FROM fact_transactions
    GROUP BY 1, 2
)
INSERT INTO fact_customer_daily_metrics (
    customer_id, metric_date, txn_count, txn_amount_sum,
    txn_amount_avg_90d, txn_amount_stddev_90d,
    distinct_merchants_30d, distinct_devices_30d
)
SELECT
    d.customer_id,
    d.metric_date,
    c.txn_count,
    c.txn_amount_sum,
    w.avg_amount_90d,
    w.stddev_amount_90d,
    w.distinct_merchants_30d,
    w.distinct_devices_30d
FROM customer_days d
JOIN customer_daily_counts c USING (customer_id, metric_date)
JOIN LATERAL (
    SELECT
        AVG(t.amount) AS avg_amount_90d,
        STDDEV_SAMP(t.amount) AS stddev_amount_90d,
        COUNT(DISTINCT t.merchant_id) FILTER (
            WHERE t.transaction_ts >= d.metric_date - INTERVAL '30 days'
        ) AS distinct_merchants_30d,
        COUNT(DISTINCT t.device_id) FILTER (
            WHERE t.transaction_ts >= d.metric_date - INTERVAL '30 days'
        ) AS distinct_devices_30d
    FROM fact_transactions t
    WHERE t.customer_id = d.customer_id
      AND t.transaction_ts >= d.metric_date - INTERVAL '90 days'
      AND t.transaction_ts < d.metric_date + INTERVAL '1 day'
) w ON true
ON CONFLICT (customer_id, metric_date) DO UPDATE SET
    txn_count = EXCLUDED.txn_count,
    txn_amount_sum = EXCLUDED.txn_amount_sum,
    txn_amount_avg_90d = EXCLUDED.txn_amount_avg_90d,
    txn_amount_stddev_90d = EXCLUDED.txn_amount_stddev_90d,
    distinct_merchants_30d = EXCLUDED.distinct_merchants_30d,
    distinct_devices_30d = EXCLUDED.distinct_devices_30d;

-- =========================================================================
-- MERCHANT DAILY METRICS
-- =========================================================================
WITH merchant_days AS (
    SELECT DISTINCT merchant_id, date_trunc('day', transaction_ts)::date AS metric_date
    FROM fact_transactions
),
merchant_daily_counts AS (
    SELECT
        merchant_id,
        date_trunc('day', transaction_ts)::date AS metric_date,
        COUNT(*) AS txn_count,
        SUM(amount) AS txn_amount_sum
    FROM fact_transactions
    GROUP BY 1, 2
)
INSERT INTO fact_merchant_daily_metrics (
    merchant_id, metric_date, txn_count, txn_amount_sum,
    avg_txn_amount_90d, chargeback_rate_90d
)
SELECT
    d.merchant_id,
    d.metric_date,
    c.txn_count,
    c.txn_amount_sum,
    w.avg_amount_90d,
    w.chargeback_rate_90d
FROM merchant_days d
JOIN merchant_daily_counts c USING (merchant_id, metric_date)
JOIN LATERAL (
    SELECT
        AVG(t.amount) AS avg_amount_90d,
        -- Synthetic proxy: trailing confirmed-fraud rate stands in for a
        -- real chargeback rate, since there is no chargeback-lag process
        -- to simulate. Documented in docs/data_dictionary.md.
        AVG(CASE WHEN g.is_fraud THEN 1.0 ELSE 0.0 END) AS chargeback_rate_90d
    FROM fact_transactions t
    LEFT JOIN ground_truth_fraud g ON g.transaction_id = t.transaction_id
    WHERE t.merchant_id = d.merchant_id
      AND t.transaction_ts >= d.metric_date - INTERVAL '90 days'
      AND t.transaction_ts < d.metric_date  -- excludes current day: chargebacks lag
) w ON true
ON CONFLICT (merchant_id, metric_date) DO UPDATE SET
    txn_count = EXCLUDED.txn_count,
    txn_amount_sum = EXCLUDED.txn_amount_sum,
    avg_txn_amount_90d = EXCLUDED.avg_txn_amount_90d,
    chargeback_rate_90d = EXCLUDED.chargeback_rate_90d;
