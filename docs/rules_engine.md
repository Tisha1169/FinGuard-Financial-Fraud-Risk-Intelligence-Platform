# Rules Engine

7 configurable business rules, each returning a rule ID, human-readable
description, severity, and structured evidence - never a bare boolean.
Thresholds live in [rules/config.py](../rules/config.py); severity is a
shared function of how far past threshold the observed value is
([rules/severity.py](../rules/severity.py)), not a per-rule judgment call.

| Rule | Description | Key threshold |
|---|---|---|
| `R1_VELOCITY` | ≥3 transactions in 10 min, or ≥6 in 60 min | `VELOCITY_10MIN_TRIGGER`, `VELOCITY_60MIN_TRIGGER` |
| `R2_AMOUNT_SPIKE` | Amount is a statistical outlier vs. 90-day customer baseline (z-score ≥4, or ≥6x average when there's not enough history for a stddev) | `AMOUNT_ZSCORE_TRIGGER` |
| `R3_GEO_IMPOSSIBLE` | Implied travel speed between consecutive transactions exceeds ~900 km/h | `GEO_MAX_PLAUSIBLE_SPEED_KMH` |
| `R4_FAILED_THEN_LARGE` | ≥2 failed attempts in 15 min followed by an approved transaction ≥4x baseline | `FAILED_THEN_LARGE_MIN_FAILED` |
| `R5_NEW_DEVICE_NEW_LOCATION` | Device and location both never seen before for this customer | — (compound of two point-in-time flags) |
| `R6_MERCHANT_DEVIATION` | Amount ≥3x the merchant's own 90-day average transaction size | `MERCHANT_AMOUNT_DEVIATION_MULTIPLIER` |
| `R7_OFF_HOURS` | Transaction at an hour of day this customer has never transacted at before (given ≥10 prior transactions) | `OFF_HOURS_MIN_PRIOR_TXNS` |

## Two implementations, one set of thresholds

- [rules/engine.py](../rules/engine.py) - single-transaction, dict-based.
  Calls `features/point_in_time.py` (several small queries) for one
  transaction at a time. This is the right shape for live scoring (the
  `/risk/{transaction_id}` API endpoint in Phase 9) where one transaction
  needs an answer in real time.
- [rules/batch.py](../rules/batch.py) - vectorized, pandas-based. Computes
  the identical point-in-time logic (same leakage boundaries, via
  `merge_asof` with a day-1 offset - see [docs/feature_engineering.md](feature_engineering.md))
  across the entire transaction history at once. Scoring all ~104k
  transactions with the row-by-row approach would mean roughly 10 small SQL
  round-trips per transaction (~1M queries, on the order of tens of
  minutes); the vectorized batch version does the same work in **~2.4
  seconds**.

Both import their thresholds from `rules/config.py` and their severity
logic from `rules/severity.py`, so a threshold change applies to both
paths identically. `tests/test_rules.py::test_batch_engine_matches_single_transaction_engine_on_a_sample`
directly verifies the two implementations agree on real transactions - two
independent implementations of the same rule producing the same output is
the strongest evidence they're both actually correct.

## Running it

```bash
python scripts/run_rules.py
```

Idempotent (truncates and rewrites `rules_triggered`). On the current
dataset: 103,651 transactions scored in ~2.4 seconds.

## Evaluation against ground truth (rules alone)

| Metric | Value |
|---|---|
| Recall (fraud transactions with ≥1 rule fired) | 74.0% (1,565 / 2,115) |
| Precision (of all flagged transactions, share that are actually fraud) | 8.4% |

Per-rule precision varies enormously - this is expected and is the central
argument for combining rules with ML rather than shipping a rules-only
system:

| Rule | Flagged | True fraud | Precision |
|---|---|---|---|
| `R4_FAILED_THEN_LARGE` | 53 | 53 | 100.0% |
| `R1_VELOCITY` | 372 | 368 | 98.9% |
| `R2_AMOUNT_SPIKE` | 2,232 | 659 | 29.5% |
| `R6_MERCHANT_DEVIATION` | 4,094 | 700 | 17.1% |
| `R7_OFF_HOURS` | 9,186 | 646 | 7.0% |
| `R5_NEW_DEVICE_NEW_LOCATION` | 1,095 | 60 | 5.5% |
| `R3_GEO_IMPOSSIBLE` | 4,389 | 174 | 4.0% |

**Reading this correctly:** `R1` and `R4` are near-perfect because they
match the exact behavioral signature the corresponding fraud typologies
were constructed with (velocity abuse, failed-then-large) - this is
expected, not evidence the rule generalizes to fraud patterns it wasn't
designed around. `R3` and `R5` are noisy because single-factor rules
(one impossible trip, one new device) are cheap to fire and easy for
legitimate behavior to trigger too (business travel, a new phone) - this
mirrors why production transaction-monitoring systems don't alert on any
single rule firing alone, and is exactly the gap the ML model and combined
risk score (Phases 6-7) are built to close.

## Known limitations

- Thresholds are reasoned defaults, not tuned against a held-out labeled
  set - threshold tuning against the ML model's validation split happens
  in Phase 7 (risk scoring / threshold analysis).
- `R7_OFF_HOURS`'s "zero historical occurrences" condition is stricter
  than a real production rule would use (typically a percentile-of-volume
  threshold) - kept simple here deliberately, with its resulting noisiness
  documented above rather than hidden.
