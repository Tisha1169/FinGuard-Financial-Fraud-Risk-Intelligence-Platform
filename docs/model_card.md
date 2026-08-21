# Model Card — FinGuard Supervised Fraud Model (Phase 6)

## Model purpose

Predicts the probability that a transaction is fraudulent, as one input
into the combined risk score (Phase 7). Two models are built: a Logistic
Regression baseline (interpretable) and an XGBoost model (primary,
higher-performing). Both are trained on the same leakage-audited feature
set assembled from Phases 3-5.

## Intended use

- As `ml_probability`, one of several inputs (alongside rules severity,
  the Phase 5 behavioral/anomaly scores, and financial exposure) to the
  Phase 7 combined risk score.
- As a portfolio/interview artifact demonstrating correct handling of
  class imbalance, temporal leakage, and evaluation methodology for a
  fraud detection problem.

## Non-intended use

- **Not validated on real transaction data** - trained and evaluated
  entirely on this project's synthetic dataset (see "Synthetic-data
  caveats" below). Do not deploy against real transactions, real
  customers, or use reported metrics as an estimate of real-world
  performance.
- **Not a standalone fraud decision system.** This model's output is one
  signal among several (rules, behavioral anomaly score, Isolation
  Forest, financial exposure) - Phase 7 explicitly exists because no
  single layer, including this one, is precise enough to act on alone
  (see the Phase 5 per-typology evidence in
  [docs/statistical_detection.md](statistical_detection.md), which this
  model's own error analysis below reproduces the same pattern for).
- **Not tuned for a production threshold.** The threshold used in this
  phase's evaluation was selected on validation data purely to make
  precision/recall/F1 reportable at a single operating point - it is not
  a cost-optimized operational threshold. That optimization is Phase 7's
  job.

## Features

37 features after preprocessing (14 numeric including
`isolation_forest_score`, 3 boolean, 7 rule-firing flags, and 13 one-hot
encoded categorical columns from 4 categorical fields). Full feature list
and per-feature documentation: [models/feature_matrix.py](../models/feature_matrix.py)'s
`FEATURE_DOCS` dict, which is the single source of truth (this table is a
copy of it, kept in sync manually).

| Feature | Description |
|---|---|
| `amount` | Raw transaction amount in USD. |
| `amount_zscore` | (amount − customer's 90d avg) / 90d stddev, as of end of previous calendar day. |
| `txn_count_last_10min` / `_60min` | Customer's transaction count in the trailing window, strictly before this transaction. |
| `recent_failed_count_15min` | Customer's FAILED-status transaction count in the trailing 15 minutes. |
| `distinct_merchants_30d` / `distinct_devices_30d` | Customer's distinct merchant/device count in the trailing 30 days, as of end of previous day. |
| `customer_amount_iqr_score` / `merchant_amount_iqr_score` | Tukey upper-fence outlier score vs. the customer's/merchant's own 90d amount distribution, as of the immediately preceding transaction. |
| `frequency_deviation_zscore` | Poisson z-score of 7-day transaction count vs. expected rate from 90d activity. |
| `hour` | Hour of day (UTC), 0-23. |
| `rules_fired_count` / `max_rule_severity` | Count and highest ordinal severity of Phase 4 rules fired on this transaction. |
| `isolation_forest_score` | Unsupervised anomaly score in [0,1], from a **train-only-fit** Isolation Forest (see "Isolation Forest leakage fix" below) - NOT the Phase 5 whole-dataset score. |
| `is_new_device` / `is_new_location` / `is_first_time_at_merchant` | Boolean first-occurrence flags for this customer. |
| `rule_fired_R1_VELOCITY` ... `rule_fired_R7_OFF_HOURS` | One boolean per Phase 4 rule. |
| `channel` (one-hot) | CARD_PRESENT, ECOM, POS, ATM, WALLET. |
| `status` (one-hot) | APPROVED, DECLINED, FAILED. |
| `customer_risk_segment` (one-hot) | STANDARD, WATCHLIST, VIP. |
| `merchant_risk_category` (one-hot) | STANDARD, HIGH_RISK. |

**Identifiers kept for joining/error-analysis but never used as features:**
`transaction_id`, `customer_id`, `merchant_id`, `device_id`, `location_id`,
`transaction_ts`. A synthetic dataset this size would let a model overfit
to specific entity IDs rather than learning generalizable behavior, so
these are excluded from `FEATURE_COLUMNS` entirely
(`tests/test_ml_pipeline.py::test_no_target_or_identifier_leakage_into_feature_columns`
enforces this).

## `chargeback_rate_90d` exclusion — the decision, and the proof behind it

`fact_merchant_daily_metrics.chargeback_rate_90d` is a merchant's trailing
90-day confirmed-fraud rate, deliberately **excluded** from the primary
model's feature set. Two separate questions were investigated:

**Is it temporally leaking?** No.
`tests/test_ml_pipeline.py::test_chargeback_rate_temporal_safety_proof`
recomputes the value independently for a random sample of merchant/day
rows spanning the full timeline (including dates that fall in the test
period), using only `fact_transactions`/`ground_truth_fraud` rows strictly
before that day, and confirms it matches the stored value exactly (within
`NUMERIC(6,4)` column rounding). This holds for every date regardless of
which chronological split it falls into, because the SQL construction
(`sql/feature_engineering.sql`) bounds the window to `[metric_date - 90
days, metric_date)` - strictly in the past relative to the row's own date,
by construction, not by luck.

**Should it be used anyway?** No - excluded, for a different reason: it is
*unrealistically available*. The synthetic construction makes a merchant's
confirmed-fraud history available for scoring the very next calendar day.
Real chargeback reporting lags by weeks to months (issuer disputes,
investigation, network reporting cycles) - using a feature that assumes
next-day availability would make the model look better than any model
could actually perform in production, where a merchant's *true* recent
chargeback rate is largely unknown at scoring time. Including it here
would be methodologically dishonest even though it isn't technically
leaking. `tests/test_ml_pipeline.py::test_chargeback_rate_is_excluded_from_ml_features`
enforces this decision in code, not just in this document.

## Isolation Forest leakage fix

Phase 5's `models/isolation_forest.py` fits on the **entire** dataset -
correct for that phase's standalone unsupervised evaluation (no
supervised train/test split exists there). Reusing those scores directly
as a Phase 6 ML feature would leak validation/test-period structure into
a value seen at training time: the forest's splits would already have
been informed by exactly the transactions the supervised model is later
evaluated on.

Fixed with a dedicated module, [models/ml_isolation_forest.py](../models/ml_isolation_forest.py):
the forest, its imputation medians, and its score-normalization bounds are
all fit on the **training period only**
(`fit_isolation_forest_train_only`); validation and test are scored by
**transforming** through that frozen model
(`score_with_fitted_forest`), never refitting.
`tests/test_ml_pipeline.py::test_isolation_forest_is_not_fit_on_validation_or_test_rows`
and `test_isolation_forest_scoring_never_refits_on_val_or_test` are
regression tests for this specifically (the second checks object identity
- the same fitted model instance is reused for both val and test scoring,
never replaced).

## Training methodology

### Temporal split

Chronological 70/15/15 split by row position on time-sorted data (not
shuffled, no k-fold) - see [models/temporal_split.py](../models/temporal_split.py).
Exact date ranges and row counts are in
[docs/evaluation_report.md](evaluation_report.md). Zero temporal overlap
between splits is enforced and tested
(`test_chronological_split_has_zero_temporal_overlap`).

### Preprocessing - fit on training data only

Numeric imputation (median), categorical one-hot encoding, and feature
scaling (for Logistic Regression only - trees are scale-invariant) are all
fit on the training split exclusively via
[models/ml_preprocessing.py](../models/ml_preprocessing.py)'s
`FittedPreprocessor`, then applied unchanged to validation and test.
Unseen categories in validation/test are handled via
`OneHotEncoder(handle_unknown="ignore")` rather than raising or leaking
information about what categories exist in later periods.

### Imbalance handling: class weights, not SMOTE

Both models use class weighting rather than SMOTE:
- Logistic Regression: `class_weight="balanced"` (inversely proportional
  to class frequency).
- XGBoost: `scale_pos_weight` computed as `n_negative / n_positive` from
  the **training set only** (46.0 in this run - see
  [docs/evaluation_report.md](evaluation_report.md) for the exact split).

**Why not SMOTE:** SMOTE interpolates synthetic minority-class points in
feature space between existing minority examples. Two considerations argue
against it here specifically:
1. The minority class in this project is *already synthetic* - it's not a
   naturally scarce signal that needs augmenting, it's a small number of
   deliberately constructed fraud typologies. Interpolating between, say, a
   `CARD_TESTING` point (many tiny transactions, ECOM) and an
   `ACCOUNT_TAKEOVER` point (new device+location, escalating amounts)
   produces a synthetic feature vector that describes neither typology
   coherently - it would manufacture a fraud "pattern" with no counterpart
   in the injected typologies, undermining the whole point of building the
   typologies deliberately in Phase 2.
2. Class weighting directly reweights the loss function using the real
   (if synthetic) minority examples that exist, without inventing new
   feature-space points at all - a more conservative choice, and one that
   keeps every training example traceable back to an actual injected
   event.

### XGBoost hyperparameter selection

A small, deliberately conservative grid (3 configurations - see
`XGB_CANDIDATE_PARAMS` in [models/ml_models.py](../models/ml_models.py)),
each evaluated by PR-AUC on the **validation** set only. The test set is
never touched during model or hyperparameter selection - see
[docs/evaluation_report.md](evaluation_report.md) for the selected
configuration and the full trial results.

### Threshold selection

The single operating threshold used for precision/recall/F1/confusion-
matrix reporting is selected on **validation** data (top-2% alert-volume
cutoff, matching the convention used throughout Phases 4-5 for
comparability). It is applied unchanged to test for one final,
unbiased evaluation. Cost-based threshold optimization (balancing
false-positive investigation cost against missed-fraud exposure) is
explicitly deferred to Phase 7.

## Evaluation metrics

PR-AUC (primary), ROC-AUC (secondary/reference), precision/recall/F1 at
the validation-selected threshold, recall at a fixed 1% FPR,
precision/recall at a top-2% alert-volume cutoff, and a confusion matrix -
on **both** validation and test. Accuracy is deliberately not reported: at
a ~2% fraud rate, a model that predicts "never fraud" scores ~98% accuracy
while catching zero fraud, making it actively misleading as a headline
metric.

Full results, SHAP findings, and error analysis:
[docs/evaluation_report.md](evaluation_report.md).

## Reproducibility

- `random_state=42` fixed for the split RNG dependencies, Isolation
  Forest, Logistic Regression, and XGBoost.
- `tests/test_ml_pipeline.py::test_logistic_regression_is_deterministic`
  and Phase 5's existing Isolation Forest reproducibility check both
  confirm bit-identical output across repeated fits on identical data.
- Full pipeline reproduction: `python scripts/train_models.py` (writes
  `artifacts/phase6_metrics.json` and `artifacts/xgboost_model.json`,
  both gitignored - regenerate rather than expecting them present).

## Synthetic-data caveats (restated for this model specifically)

Everything in this model card and the evaluation report is measured
against this project's synthetic, independently-injected fraud
typologies (see [docs/data_generation.md](data_generation.md) and the
"Synthetic-data limitations" section of
[docs/statistical_detection.md](statistical_detection.md), which apply
identically here). In particular:

- The elevated 2% fraud rate (vs. real-world rates typically under 1%)
  makes precision figures easier to achieve than they would be at a
  realistic base rate.
- Each fraud typology is a clean, internally consistent behavioral
  signature - real fraud is noisier and typologies blend together. The
  strong per-typology variation in this model's recall (see the error
  analysis in the evaluation report) is partly an artifact of how
  cleanly each typology was constructed.
- Legitimate-but-unusual behavior (genuine business travel, a real first
  large purchase) is not separately modeled, so this evaluation cannot
  show what happens when real users produce these same statistical
  signatures without being fraud - which is exactly where false positives
  come from in production.
- No claim of real-world fraud prevention, financial loss avoidance, or
  performance transfer to any real institution is made anywhere in this
  project.
