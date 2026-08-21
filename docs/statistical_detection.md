# Statistical / Behavioral Anomaly Engine

Layer 5 of the architecture, distinct from both the business rules (Phase
4, hard thresholds) and the supervised ML model (Phase 6, learned from
labels). Two methods, chosen for complementary strengths:

## Method 1: IQR / Tukey fences (robust, univariate)

For a customer's or merchant's trailing 90-day amount distribution,
`Q3 + 1.5*IQR` marks the classic Tukey upper fence; a transaction's outlier
score is how many IQR-widths past that fence it sits. Chosen over a plain
z-score (mean/stddev) because the median and quartiles aren't dragged
around by one or two whale transactions the way a mean is - a customer
with a single $2,000 purchase in their history won't have every future
transaction's baseline skewed by it the way it would skew a mean/stddev.

**Limitation:** needs enough history to estimate quartiles meaningfully.
See "bugs found" below - this was not just a theoretical concern.

## Method 2: Isolation Forest (unsupervised, multivariate)

Fit across amount, velocity, failed-attempt count, distinct merchant/device
counts, the IQR scores above, frequency deviation, hour, and new-device/
new-location flags. Isolation Forest isolates points via random recursive
splits; anomalies need fewer splits to isolate than normal points, so it
naturally scores multivariate rarity - a transaction that's individually
unremarkable on every single feature but unusual in *combination* (e.g.
median amount, but at 3am, from a new device, right after two failed
attempts) is exactly what a univariate method (rules, IQR) cannot see and
Isolation Forest can.

**Limitation:** it's unsupervised - "anomalous" means statistically rare,
not "fraud." It will flag genuinely rare-but-legitimate behavior (a
customer's first big purchase, real business travel) with no way to tell
the two apart without labels. `contamination=0.02` is set to match the
*known* synthetic injection rate - a real deployment doesn't have this
luxury and would use `'auto'` or tune against analyst feedback over time.

## Combining into one score

`behavioral_anomaly_score` combines the customer IQR score, merchant IQR
score, and frequency-deviation z-score via a noisy-OR
(`1 - Π(1 - component)`): each is treated as independent evidence, so any
one strong signal drives the combined score up without an arbitrary
weighting scheme. This feeds into the final combined risk score in Phase 7.

## Two real bugs found and fixed during development

Both were caught by comparing evaluation metrics that should have been
consistent (a reasonable AUC alongside an unreasonably low precision@top-K
is a tell that something downstream of the ranking is broken) - not by
code review alone.

1. **Unstable quartiles from thin history.** A customer with only 2-3
   prior transactions can have a nearly-zero IQR by chance (e.g. two prior
   transactions of $39.00 and $39.02), turning an ordinary third
   transaction into an apparent 500-IQR outlier. Fixed by requiring
   `MIN_HISTORY_FOR_IQR = 10` prior observations before trusting the fence
   at all (mirrors the same guard already used for `R7_OFF_HOURS` in
   Phase 4).
2. **Miscalibrated frequency z-score from a fixed-window divisor.** The
   first version estimated a customer's expected weekly transaction count
   as `(90-day trailing count) / 90 * 7` - but for a customer only 30 days
   into the simulation, their "90-day" window actually only contains 30
   days of activity, so dividing by 90 underestimated their true rate by
   ~3x and inflated the z-score for everyone early in their history. This
   pushed 14.7% of all transactions to `z >= 3` (a threshold that should
   fire on roughly 0.1-0.5% of transactions for a well-calibrated z-score)
   and saturated `behavioral_anomaly_score` to its maximum for >16% of the
   dataset - which silently wrecked precision at any reasonable alert
   volume despite a still-plausible-looking AUC. Fixed by dividing by the
   customer's actual elapsed active days (capped at 90), not a fixed
   window length. `tests/test_statistical.py::test_frequency_zscore_is_well_calibrated`
   is a permanent regression guard for this.

## Evaluation methodology

Ground truth (`ground_truth_fraud.is_fraud`) is used **only** for the
`evaluate()` step in `scripts/run_statistical_detection.py`, after both the
IQR/frequency features and the Isolation Forest fit are already complete -
confirmed by grepping `models/`, `rules/`, and `features/` for any
reference to `is_fraud`, `ground_truth_fraud`, or `fraud_typology`: there
are none. Isolation Forest is unsupervised and never sees labels at any
point, fit or evaluated. This evaluation is diagnostic (checking whether
the unsupervised/robust-statistics scores happen to correlate with fraud),
not a claim of generalization - that claim is reserved for Phase 6's
time-aware supervised split.

Metrics: AUC-ROC over the full ranking, plus precision/recall at a top-2%
cutoff (matching the known synthetic injection rate, as a proxy for "the
alert volume an investigation team could realistically handle"). All
numbers below were independently re-verified fresh (re-running the full
pipeline against live Postgres) during a Phase 5 audit, and reproduce
exactly - `IsolationForest(random_state=42)` fit twice on identical input
produces bit-identical scores (`max abs diff: 0.0`).

## Evaluation against ground truth (diagnostic)

| Method | AUC-ROC | Precision @ top 2% | Recall @ top 2% |
|---|---|---|---|
| `behavioral_anomaly_score` (IQR + frequency, noisy-OR) | 0.767 | 30.3% | 29.7% |
| `isolation_forest_score` | 0.917 | 46.4% | 45.5% |

### Why Isolation Forest wins: per-typology recall in the top-2% cutoff

| Typology | n | `behavioral_anomaly_score` recall | `isolation_forest_score` recall |
|---|---|---|---|
| CARD_TESTING | 234 | 13.7% | **67.5%** |
| VELOCITY_ABUSE | 206 | 11.7% | **42.2%** |
| ACCOUNT_TAKEOVER | 251 | 22.7% | **35.5%** |
| FAILED_THEN_LARGE | 294 | 62.2% | 89.5% |
| UNUSUAL_AMOUNT | 226 | 80.5% | 84.5% |
| MERCHANT_BEHAVIOR_DEVIATION | 226 | **52.7%** | 48.7% |
| DEVICE_ANOMALY | 226 | 5.8% | 14.6% |
| GEOGRAPHIC_INCONSISTENCY | 226 | 6.6% | 12.8% |
| UNUSUAL_TIME_OF_DAY | 226 | 1.8% | 0.9% |

This is direct evidence for the multivariate-interaction claim, not just an
assertion: `CARD_TESTING` is defined by *many small transactions in a short
window* - a combination of velocity and low amount, neither of which alone
is unusual (small transactions are common; occasional bursts are common).
`behavioral_anomaly_score` has no velocity input at all (it only sees
amount/frequency deviation, not the 10/60-minute counts), so it barely sees
this typology (13.7%). Isolation Forest has `txn_count_last_10min` and
`txn_count_last_60min` as direct inputs and can learn the amount×velocity
combination that defines the typology, hence 67.5%. The same pattern holds
for `VELOCITY_ABUSE` and, more weakly, `ACCOUNT_TAKEOVER` (device+location
change plus escalating amounts).

Where the gap disappears or reverses is equally informative:
`UNUSUAL_AMOUNT` and `FAILED_THEN_LARGE` are both fundamentally univariate
(amount deviation, or amount deviation + a failed-attempt count) - the
IQR/frequency engine is a direct, purpose-built detector for exactly that
shape of signal, so both methods do well and Isolation Forest's edge is
small. `MERCHANT_BEHAVIOR_DEVIATION` is the one typology where the simpler
method actually wins (52.7% vs. 48.7%) - it's injected as a large
transaction at a high-risk or unfamiliar merchant, and `merchant_amount_iqr_score`
targets that directly, while Isolation Forest has to discover the same
signal indirectly through a 13-feature split. `UNUSUAL_TIME_OF_DAY`,
`DEVICE_ANOMALY`, and `GEOGRAPHIC_INCONSISTENCY` are weak for **both**
methods (≤15% recall each) for a specific, identifiable reason: neither
feature set includes a customer-specific "usual hour" signal, a raw
last-location/distance feature, or the new-device-and-new-location
*compound* condition as engineered inputs - `hour`, `is_new_device`, and
`is_new_location` are present individually, but the actual signal (this
specific hour is unusual *for this customer*; this device+location pairing
has never co-occurred) isn't handed to either model as a ready-made
feature. This is exactly what `R7_OFF_HOURS`, `R3_GEO_IMPOSSIBLE`, and
`R5_NEW_DEVICE_NEW_LOCATION` (Phase 4) exist to catch instead - concrete,
typology-by-typology confirmation that no single detection layer covers
everything, which is the actual argument for the hybrid design (see
"Feeding into the hybrid risk engine" below), not just a general claim
about rules-vs-ML tradeoffs.

## Business interpretation

- **Isolation Forest is the stronger standalone anomaly score** (0.917 AUC)
  and should carry more weight than the IQR/frequency score in the
  combined risk score - but it is not a replacement for rules, because it
  misses the exact typologies (`UNUSUAL_TIME_OF_DAY`, `DEVICE_ANOMALY`,
  `GEOGRAPHIC_INCONSISTENCY`) that specific rules (R7, R5, R3) were built
  to catch directly.
- **The behavioral (IQR/frequency) score is not obsolete even though
  Isolation Forest usually beats it** - it wins on `MERCHANT_BEHAVIOR_DEVIATION`
  and is a fully interpretable, auditable number ("this transaction is
  2.3 IQR-widths above the merchant's typical size") in a way a forest's
  anomaly score is not. That interpretability matters directly for the
  investigation workbench (Phase 8): an analyst needs a reason, not just a
  score.
- **Neither unsupervised method is precise enough to alert on alone.** At
  the alert volume investigators could plausibly handle (top 2%),
  Isolation Forest still misses more than half of `CARD_TESTING` and
  `ACCOUNT_TAKEOVER` fraud and the IQR/frequency score misses over 80% of
  four different typologies. This is the direct, typology-level
  justification for combining rules + statistical + ML into one score
  rather than shipping any single layer.

## Feeding into the hybrid risk engine (design only - not built yet)

Per the architecture in `docs/architecture.md`, the eventual combined risk
score is:

```
rules_severity (Phase 4) + behavioral_anomaly_score (Phase 5, IQR/frequency)
  + isolation_forest_score (Phase 5) + ml_probability (Phase 6)
  + financial_exposure (Phase 7) -> combined_score -> risk_tier -> case priority
```

Phase 5's two outputs are two of the several inputs to that score, not the
whole thing - `risk_scores.behavioral_component` (see `sql/schema.sql`) is
sized to hold a single combined behavioral number, so Phase 7 will need a
documented sub-combination of `behavioral_anomaly_score` and
`isolation_forest_score` (e.g. another noisy-OR, or a max, or a learned
weight) before it becomes one component of the final score alongside
`ml_probability` and `rules_severity`. This sub-combination is a Phase 7
design decision and is explicitly out of scope here.

## Known concerns for Phase 6 and beyond

- **`fact_merchant_daily_metrics.chargeback_rate_90d` is label-derived and
  currently unused.** It's computed directly from `ground_truth_fraud.is_fraud`
  (trailing 90 days, excluding the current day - see
  `sql/feature_engineering.sql`). It is *not* used by any rule, by the IQR/
  frequency engine, or by Isolation Forest's `FEATURE_COLUMNS` (verified by
  grep). This is not leakage as implemented - it's a legitimate real-world
  signal (banks do track historical merchant chargeback rates) and the
  trailing/excludes-current-day construction is temporally safe if it's
  ever used as an ML feature in Phase 6. It's flagged here specifically so
  Phase 6 treats it deliberately (documenting it as a target-derived
  feature, expecting it to be unusually predictive, and confirming the
  temporal boundary still holds under the time-aware split) rather than
  including it casually.
- **Baseline "staleness" is inconsistent across features, though
  empirically small.** Three different recency conventions coexist in the
  current feature set, all leakage-safe (none use future data) but with
  different lag: (1) `amount_zscore` and `merchant_avg_amount_90d`
  (Phase 3/4, `rules/batch.py`) are "as of the end of the previous
  calendar day"; (2) `customer_amount_iqr_score` and
  `merchant_amount_iqr_score` (this phase) are "as of the customer's/
  merchant's immediately preceding transaction," which is only precisely
  current when transactions are closely spaced; (3) `txn_count_last_10min`/
  `60min` and `recent_failed_count_15min` are exactly "as of this instant,"
  computed via subtract-self rather than shift. Audited empirically: the
  median gap between a customer's consecutive transactions is 0.81 days,
  97.8% of gaps are under 5 days, and only 0.005% exceed 20 days - so the
  IQR baseline's staleness is negligible in practice for this dataset, but
  the inconsistency is a real design wrinkle worth knowing about before
  building on top of it, and would matter more on a sparser or more
  bursty transaction stream.
- **Isolation Forest's NaN imputation uses the full-dataset median.**
  `models/isolation_forest.py::_prepare_matrix` imputes missing
  deviation-based features (thin-history customers/merchants) with the
  median computed across the *entire* dataset being scored. This is fine
  for the current batch/offline design (fit once over historical data, no
  train/test split, as documented), but would not be appropriate for live
  incremental scoring without freezing a reference median from a fixed
  training window - a Phase 9 (API) concern, not a Phase 5 bug.

## Running it

```bash
python scripts/run_statistical_detection.py
```

Caches results to `data/behavioral_features.parquet` (gitignored, ~4-5
seconds to regenerate) for reuse by Phase 6 (ML model features) and Phase 7
(risk scoring) without recomputing.

## Known limitations

- Isolation Forest is fit on the full dataset with no train/test split, since
  it's unsupervised - its evaluation here is diagnostic, not a claim of
  generalization the way Phase 6's time-aware supervised split will be.
- **Synthetic-data limitations, explicitly** (see
  [docs/data_generation.md](data_generation.md) for the full generator
  design): every number on this page is measured against synthetic,
  independently-injected fraud typologies, not real fraud, and this
  directly shapes the results above:
  - The 2% injected fraud rate is far higher than real-world card fraud
    (typically well under 1%), chosen so there's enough labeled fraud to
    evaluate against - both AUC and precision@top-2% would look different,
    almost certainly worse for precision, at a realistic base rate.
  - Each typology is injected as a clean, internally consistent behavioral
    signature (e.g. `CARD_TESTING` is *always* small amounts in a tight
    burst) - real fraud is noisier and typologies blend together (a real
    account-takeover campaign might also exhibit card-testing behavior
    first). The strong per-typology separation seen in the table above
    (some typologies at >80% recall, others under 15%) is partly an
    artifact of how cleanly each typology was constructed, not necessarily
    how cleanly real fraud separates.
  - Legitimate-but-unusual behavior (genuine business travel, a real first
    large purchase, an actual new phone) is not separately modeled in the
    generator - so precision figures here cannot show what happens when
    real users produce these same statistical signatures without being
    fraud, which is precisely where false positives come from in
    production. Real-world precision at any given recall level is likely
    lower than shown here.
  - `ground_truth_fraud.is_synthetic_label` is `true` for every row in
    this project; no real transaction or fraud data is used anywhere, and
    no claim of real-world fraud prevention or financial loss avoidance is
    made from these numbers.
