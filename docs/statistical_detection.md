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

## Evaluation against ground truth (diagnostic)

| Method | AUC-ROC | Precision @ top 2% | Recall @ top 2% |
|---|---|---|---|
| `behavioral_anomaly_score` (IQR + frequency, noisy-OR) | 0.767 | 30.3% | 29.7% |
| `isolation_forest_score` | 0.917 | 46.4% | 45.5% |

Top-2% is used as the cutoff because it matches the known synthetic fraud
rate - a proxy for "if this were the alert volume an investigation team
could handle." Isolation Forest outperforms the univariate behavioral
score here specifically because several injected typologies (account
takeover, device anomaly) are defined by *combinations* of features
(new device + new location + amount) that a single-feature method can't
fully capture, while Isolation Forest's multivariate splits can.

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
- Both methods are as good as their inputs: they inherit the same
  synthetic-data limitations documented in
  [docs/data_generation.md](data_generation.md) (elevated fraud rate,
  independently-injected typologies).
