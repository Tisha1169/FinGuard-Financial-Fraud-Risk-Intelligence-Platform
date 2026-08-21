# Feature Engineering

Two layers, matching how a real fraud feature store is typically split:

1. **Batch daily rollups** (`fact_customer_daily_metrics`,
   `fact_merchant_daily_metrics`) — computed once per day in SQL, materialized
   in Postgres.
2. **Point-in-time lookups** ([features/point_in_time.py](../features/point_in_time.py)) —
   assembled at scoring time for one specific transaction, combining the
   (necessarily one-day-stale) daily rollup with same-day intraday activity.

## Why split this way

A pure "recompute everything live" approach doesn't scale, and a pure
"only daily batch features" approach misses same-day bursts (card testing,
velocity abuse) entirely — those typologies live entirely inside a single
day. Splitting into a daily batch layer plus an intraday layer is the
standard real-world pattern and is what makes both slow behavioral
baselines (90-day average spend) and fast velocity signals (transactions in
the last 10 minutes) available at scoring time.

## Leakage prevention — the part that actually matters here

This is the most interview-relevant design decision in this phase, so it's
enforced in three separate places, not just documented:

1. **SQL layer** ([sql/feature_engineering.sql](../sql/feature_engineering.sql)) —
   each `fact_customer_daily_metrics` row for day `d` is an end-of-day
   snapshot that *does* include day `d`'s own transactions. That's fine by
   itself, but it means callers must never read the row for the same day
   they're scoring.
2. **Point-in-time layer** ([features/point_in_time.py](../features/point_in_time.py)) —
   `get_customer_baseline_asof()` / `get_merchant_baseline_asof()` explicitly
   query `WHERE metric_date < as_of_date` (strictly less than), never
   `<=`. Same-day activity (`get_intraday_activity`,
   `count_recent_transactions`) is filtered `transaction_ts < as_of_ts`
   (strictly before the transaction being scored), so a transaction never
   counts itself in its own velocity or intraday features.
3. **Merchant chargeback proxy** — `chargeback_rate_90d` additionally
   excludes the *current* day entirely (`< metric_date`, not
   `< metric_date + 1 day`), modeling that real chargebacks are reported by
   issuers well after the original transaction.
4. **Tested, not just asserted** — [tests/test_features.py](../tests/test_features.py)
   directly proves this: e.g. `test_velocity_count_excludes_the_scored_transaction`
   scores a real transaction from an injected `VELOCITY_ABUSE` burst and
   checks the count is exactly one less than a raw SQL count that includes
   the transaction itself.

## Chargeback rate: a documented approximation

There is no chargeback-lag process in this synthetic dataset, so
`fact_merchant_daily_metrics.chargeback_rate_90d` uses the merchant's
trailing 90-day *confirmed-fraud rate* (`ground_truth_fraud.is_fraud`) as a
stand-in. This is a modeling simplification, not a claim that fraud
confirmation and chargebacks are the same signal in the real world.

## Correctness verification

`test_daily_metrics_matches_independent_pandas_computation` recomputes a
90-day trailing average/stddev directly in pandas from raw
`fact_transactions` and asserts it matches the SQL-computed row — two
independent implementations agreeing is stronger evidence of correctness
than either one asserting against itself.

## Running it

```bash
python scripts/compute_features.py
```

Idempotent — re-running after new transactions are loaded upserts
(`ON CONFLICT ... DO UPDATE`) rather than duplicating rows. On the current
dataset (~104k transactions): 66,535 customer-day rows, 17,935 merchant-day
rows, computed in ~6 seconds.
