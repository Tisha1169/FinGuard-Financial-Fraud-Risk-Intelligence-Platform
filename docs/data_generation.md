# Synthetic Data Generation

All data in FinGuard is synthetic. There is no real transaction, customer,
or fraud data anywhere in this project. `ground_truth_fraud.is_synthetic_label`
is `true` for every row, and this is called out again in the model card
(added in Phase 6) wherever the label is used for training or evaluation.

## Why synthetic, and why not just an anonymized public dataset

Public fraud datasets (e.g. IEEE-CIS, PaySim) are useful for calibrating
realistic amount/frequency distributions, but their anonymized/PCA-transformed
fields make it impossible to build a rules engine, geo-distance checks, or a
readable investigation UI on top of them — there's no real merchant, device,
or location semantics to reason about. FinGuard instead generates entities
and transactions directly, with real (if synthetic) attributes, so every
layer of the platform - rules, behavioral baselines, exposure calculation,
case UI - has something meaningful to work with.

## Generation pipeline

`python scripts/generate_data.py` runs:

1. **Entities** ([data_generation/entities.py](../data_generation/entities.py)) —
   1,000 customers, 150 merchants (across 15 MCC categories), ~1,860 devices,
   25 real-world cities (used for haversine distance checks).
2. **Behavior profiles** ([data_generation/behavior.py](../data_generation/behavior.py)) —
   each customer gets a baseline spend distribution (lognormal), a small pool
   of usual merchants, an active-hours window, and a transaction-rate prior.
   This profile is what both the normal-transaction generator and the fraud
   injector reference — fraud is deliberately constructed as a *deviation*
   from it, not an unrelated random draw.
3. **Normal transactions** ([data_generation/normal_transactions.py](../data_generation/normal_transactions.py)) —
   ~101k legitimate transactions over a 120-day window, sampled from each
   customer's profile.
4. **Fraud injection** ([data_generation/fraud_typologies.py](../data_generation/fraud_typologies.py)) —
   9 typologies, each constructed as a small, realistic burst of transactions:

   | Typology | Construction |
   |---|---|
   | `CARD_TESTING` | 5-9 very small ($0.50-$5) probes at different merchants within minutes, high decline rate |
   | `VELOCITY_ABUSE` | 6-12 transactions within ~1 hour, normal-ish amounts |
   | `ACCOUNT_TAKEOVER` | new device + new-country location, escalating amounts over hours |
   | `UNUSUAL_AMOUNT` | single transaction 5-15x the customer's baseline |
   | `GEOGRAPHIC_INCONSISTENCY` | two transactions <90 min apart, >3000km apart (physically impossible travel) |
   | `FAILED_THEN_LARGE` | 3-5 failed attempts followed by one large approved transaction |
   | `MERCHANT_BEHAVIOR_DEVIATION` | transaction at a high-risk / never-used merchant category |
   | `UNUSUAL_TIME_OF_DAY` | transaction well outside the customer's normal active hours |
   | `DEVICE_ANOMALY` | transaction from a device never before associated with the customer |

   Event counts per typology are sized so total fraud transactions land
   around **2% of all transactions** — higher than real-world card fraud
   rates (typically well under 1%), chosen deliberately so the labeled
   fraud set is large enough to train and evaluate a model on. This is a
   known synthetic-data limitation, not a claim about real fraud
   prevalence.

5. **Assembly** ([data_generation/generator.py](../data_generation/generator.py)) —
   merges normal + fraud transactions, sorts by timestamp, assigns
   sequential `transaction_id`s, and splits into the `fact_transactions` /
   `ground_truth_fraud` tables matching [sql/schema.sql](../sql/schema.sql).

All randomness is seeded (`random.Random`, `numpy.random.default_rng`,
`Faker.seed`) — the same seed reproduces byte-identical output, verified by
[tests/test_data_generation.py](../tests/test_data_generation.py).

## Reproducing the dataset

```bash
python scripts/generate_data.py   # writes data/*.parquet + dataset_summary.json
python scripts/load_data.py       # truncates and reloads DATABASE_URL
```

## Current dataset summary (seed=42)

- 1,000 customers, 150 merchants, ~1,866 devices, 25 locations
- 103,651 transactions over a 120-day window
- 2,115 fraud transactions (2.04%), all 9 typologies represented (206-294
  transactions each)
- No full payment card numbers stored anywhere — only last 4 digits

Full numbers are regenerated into `data/dataset_summary.json` on every run.

## Known limitations

- Fraud rate (2%) is elevated relative to real-world card fraud for
  labeled-data volume reasons, stated above.
- Typologies are independently injected; in reality fraud campaigns can
  combine multiple typologies in one event (e.g. account takeover *and*
  velocity abuse together) — out of scope for this version.
- Customer/merchant behavioral profiles are stationary over the 120-day
  window (no seasonal drift) except where a fraud event deliberately
  deviates from them — real customer behavior drifts over time
  independent of fraud, which the model monitoring phase (Phase 12)
  discusses in the context of detecting drift.
