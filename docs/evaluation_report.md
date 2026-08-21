# Phase 6 Evaluation Report — Supervised Fraud Model

All results below are from a single, fresh run of `python scripts/train_models.py`
against the current synthetic dataset (103,651 transactions, seed=42
throughout the pipeline). Numbers are copied directly from that run's
output / `artifacts/phase6_metrics.json` - none are hand-edited.

**Synthetic-data warning:** every number on this page is an evaluation on
this project's synthetic, independently-injected fraud typologies (see
[docs/data_generation.md](data_generation.md)), not real transaction data.
No claim is made that these results would transfer to a real financial
institution - see "Synthetic-data caveats" in
[docs/model_card.md](model_card.md) for the specific reasons why real-world
precision in particular is likely lower than shown here.

## A. Split date ranges and row counts

| Split | Rows | % | Fraud rate | Date range |
|---|---|---|---|---|
| Train | 72,555 | 70.0% | 2.13% (1,543) | 2026-04-23 00:11 → 2026-07-15 19:54 |
| Validation | 15,547 | 15.0% | 1.78% (276) | 2026-07-15 19:54 → 2026-08-02 19:59 |
| Test | 15,549 | 15.0% | 1.90% (296) | 2026-08-02 19:59 → 2026-08-21 08:19 |

Chronological, non-shuffled, zero overlap (train's max timestamp ≤
validation's min; validation's max ≤ test's min) - verified by
`tests/test_ml_pipeline.py::test_chronological_split_has_zero_temporal_overlap`.

## B. Feature count and list

**37 features** after preprocessing (14 numeric incl. `isolation_forest_score`,
3 boolean, 7 rule-firing flags, 13 one-hot columns from 4 categorical
fields). Full list and per-feature descriptions: [docs/model_card.md](model_card.md#features).

## C. Logistic Regression results

Threshold selected on validation (top-2% alert volume): **0.9208**.

| Metric | Validation | Test |
|---|---|---|
| PR-AUC | 0.7872 | 0.7067 |
| ROC-AUC | 0.9853 | 0.9793 |
| Precision @ threshold | 0.6516 | 0.6554 |
| Recall @ threshold | 0.7319 | 0.6554 |
| F1 @ threshold | 0.6894 | 0.6554 |
| Recall @ 1% FPR | 0.7717 (actual FPR 0.98%) | 0.6959 (actual FPR 0.97%) |
| Precision @ top 2% | 0.6516 | 0.6290 |
| Recall @ top 2% | 0.7319 | 0.6588 |
| Confusion matrix (TN/FP/FN/TP) | 15163 / 108 / 74 / 202 | 15151 / 102 / 102 / 194 |

## D. XGBoost results

Selected configuration (of 3 validation-scored candidates - see trial
table below): `max_depth=4, n_estimators=200, learning_rate=0.1`,
`scale_pos_weight=46.02` (computed from training data only). Threshold
selected on validation (top-2% alert volume): **0.9190**.

| Metric | Validation | Test |
|---|---|---|
| PR-AUC | 0.8652 | 0.7929 |
| ROC-AUC | 0.9914 | 0.9879 |
| Precision @ threshold | 0.7226 | 0.7103 |
| Recall @ threshold | 0.8116 | 0.6959 |
| F1 @ threshold | 0.7645 | 0.7031 |
| Recall @ 1% FPR | 0.8877 (actual FPR 0.92%) | 0.8041 (actual FPR 0.97%) |
| Precision @ top 2% | 0.7226 | 0.6935 |
| Recall @ top 2% | 0.8116 | 0.7264 |
| Confusion matrix (TN/FP/FN/TP) | 15185 / 86 / 52 / 224 | 15169 / 84 / 90 / 206 |

### XGBoost hyperparameter trials (validation PR-AUC)

| max_depth | n_estimators | learning_rate | val PR-AUC |
|---|---|---|---|
| 3 | 150 | 0.10 | 0.8550 |
| **4** | **200** | **0.10** | **0.8652 (selected)** |
| 5 | 300 | 0.05 | 0.8648 |

The gap between the top two configurations is small (0.0004); this is not
a sensitive choice, which is itself a reasonable sanity check - the result
isn't hinging on a lucky hyperparameter pick.

## Validation → test degradation is expected and real

Both models lose PR-AUC moving from validation to test (Logistic
Regression: 0.787 → 0.707, −10.3%; XGBoost: 0.865 → 0.793, −8.4%). This is
the expected, honest behavior of a genuinely time-aware evaluation - the
model has never seen the test period's data or its specific fraud
instances during training or selection, and fraud patterns (even
synthetic ones, injected with per-event randomness) don't repeat
identically across time. This is presented as-is; the pipeline does not
select a threshold or hyperparameters using test data, so there was no
opportunity to mask this gap.

## E. Top SHAP features (XGBoost, global importance, mean |SHAP value|)

Computed via `shap.TreeExplainer` on a 2,000-row random sample of the test
set - real values from the fitted model, not fabricated.

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `isolation_forest_score` | 1.604 |
| 2 | `channel_ECOM` | 1.356 |
| 3 | `is_first_time_at_merchant` | 1.335 |
| 4 | `hour` | 0.640 |
| 5 | `channel_POS` | 0.541 |
| 6 | `amount` | 0.523 |
| 7 | `channel_CARD_PRESENT` | 0.298 |
| 8 | `amount_zscore` | 0.257 |
| 9 | `channel_WALLET` | 0.228 |
| 10 | `distinct_merchants_30d` | 0.195 |

**Reading this:** `isolation_forest_score` dominates, which is coherent
with Phase 5's finding that Isolation Forest was the stronger standalone
anomaly signal (0.917 AUC vs. 0.767 for the IQR/frequency score) - the
supervised model has effectively learned to lean on the best available
unsupervised signal rather than rediscovering the same multivariate
patterns from scratch. `channel` dominates the next several ranks, which
matches the error analysis below almost exactly: fraud is heavily
concentrated in the `ECOM` channel in this synthetic dataset (most
typologies were constructed as ECOM transactions), so the model has
learned channel as a strong prior. `is_first_time_at_merchant` ranking
third is a genuine, sensible signal (`MERCHANT_BEHAVIOR_DEVIATION` and
`ACCOUNT_TAKEOVER` typologies both involve unfamiliar merchants).

### Individual prediction explanations

**A true positive** (correctly flagged fraud, model margin +5.77 vs. base
−0.006): driven by `is_new_device=1` (SHAP +1.49), `isolation_forest_score=0.386`
(+1.30), `channel_ECOM=1` (+1.02), an unusual `hour=22` (+0.78), and
`is_first_time_at_merchant=1` (+0.71) - a textbook `ACCOUNT_TAKEOVER`-shaped
combination of signals, each contributing independently.

**A false positive** (legitimate transaction incorrectly flagged, margin
+2.63): `isolation_forest_score=0.580` (+2.18) and `is_new_device=1`
(+1.32) dominate, but two features pulled the score *down*:
`rule_fired_R6_MERCHANT_DEVIATION` (−0.85) and `merchant_amount_iqr_score=2.35`
(−0.76) - interesting because a fired rule and an elevated IQR score
usually indicate *more* risk, but here they're pulling against the
prediction, suggesting the model has learned some interaction where this
particular combination is more often benign. This transaction was still
flagged, but the SHAP breakdown shows the model wasn't purely reacting to
"a rule fired" - it weighed several countervailing signals and the new-
device/anomaly-score evidence still won.

**A false negative** (missed fraud, margin −3.40): `is_first_time_at_merchant=0`
(−1.58) and a low `isolation_forest_score=0.244` (−1.20) both argue
strongly for "normal" - this fraud event apparently occurred at a merchant
the customer had used before, with an amount/pattern unremarkable enough
that the anomaly score didn't flag it. This is a legitimate model
limitation, not a bug: some injected fraud (particularly
`GEOGRAPHIC_INCONSISTENCY`, per the segment analysis below) doesn't
present as a multivariate outlier on the features available to this model.

## F. Error analysis (test set, XGBoost @ threshold 0.9190)

Overall: 206 TP, 90 FN, 84 FP, 15,169 TN.

### By channel

| Channel | TP | FN | FP | Recall | FP rate of flagged |
|---|---|---|---|---|---|
| ECOM | 183 | 55 | 82 | 76.9% | 30.9% |
| CARD_PRESENT | 23 | 35 | 2 | **39.7%** | 8.0% |
| ATM / POS / WALLET | 0 | 0 | 0 | n/a (no fraud injected in these channels) | n/a |

The model performs far worse on `CARD_PRESENT` fraud than `ECOM` fraud.
This is not incidental: `GEOGRAPHIC_INCONSISTENCY` (the typology with
consecutive transactions at physically impossible locations) is
constructed using the `CARD_PRESENT` channel in the data generator, and is
exactly the typology this model misses most (see below) - the channel
breakdown and the typology breakdown are describing the same underlying
gap from two angles, not two independent findings.

### By fraud typology (recall)

| Typology | TP | FN | Recall |
|---|---|---|---|
| CARD_TESTING | 17 | 1 | 94.4% |
| FAILED_THEN_LARGE | 24 | 2 | 92.3% |
| DEVICE_ANOMALY | 30 | 3 | 90.9% |
| MERCHANT_BEHAVIOR_DEVIATION | 30 | 5 | 85.7% |
| VELOCITY_ABUSE | 13 | 3 | 81.3% |
| UNUSUAL_AMOUNT | 23 | 7 | 76.7% |
| ACCOUNT_TAKEOVER | 28 | 18 | 60.9% |
| UNUSUAL_TIME_OF_DAY | 18 | 16 | 52.9% |
| **GEOGRAPHIC_INCONSISTENCY** | 23 | 35 | **39.7%** |

**What the model captures well:** typologies with strong, direct feature
support - `CARD_TESTING` and `VELOCITY_ABUSE` (captured by
`txn_count_last_10min`/`_60min` and, per SHAP, heavily by
`isolation_forest_score` which has those same counts as inputs),
`FAILED_THEN_LARGE` (direct rule feature `rule_fired_R4_FAILED_THEN_LARGE`
plus `recent_failed_count_15min`), `DEVICE_ANOMALY` and
`MERCHANT_BEHAVIOR_DEVIATION` (direct boolean/IQR features).

**What it misses:** `GEOGRAPHIC_INCONSISTENCY` by a wide margin (39.7%,
worst of all 9 typologies), and to a lesser extent `UNUSUAL_TIME_OF_DAY`
(52.9%) and `ACCOUNT_TAKEOVER` (60.9%). The reason is identifiable, not
mysterious: **no raw geo-distance or implied-travel-speed feature was
included in `FEATURE_COLUMNS`** - `rule_fired_R3_GEO_IMPOSSIBLE` is
present as a boolean flag, but the model only sees "did the rule fire,"
not the underlying distance/speed magnitude the rule computed. This
exactly mirrors the Phase 5 finding (`docs/statistical_detection.md`) that
neither the IQR/frequency score nor Isolation Forest could see this
typology well either, for the identical reason: none of the engineered
feature sets built through Phase 5 carried a raw geo-distance signal
forward as a numeric feature, only as a rule pass/fail flag with the
underlying evidence discarded. `UNUSUAL_TIME_OF_DAY` shows the same
pattern with `rule_fired_R7_OFF_HOURS`: the model gets a boolean, not "how
unusual is this hour for this specific customer."

**This is a genuine, addressable finding, not a fundamental limitation** -
it directly justifies keeping the rules engine's structured evidence in
the eventual investigation UI (Phase 8) even where the ML model has
weaker coverage, and suggests a concrete Phase 6+ improvement (carrying
`distance_km`/`implied_speed_kmh` from `rules_triggered.evidence` forward
as numeric features) that was intentionally left out of this phase's
scope rather than silently worked around.

### By customer risk segment and merchant risk category

| Segment | TP | FN | Recall |
|---|---|---|---|
| STANDARD | 193 | 79 | 71.0% |
| VIP | 12 | 10 | 54.5% |
| WATCHLIST | 1 | 1 | 50.0% |

| Merchant category | TP | FN | Recall |
|---|---|---|---|
| HIGH_RISK | 35 | 8 | 81.4% |
| STANDARD | 171 | 82 | 67.6% |

VIP and WATCHLIST segments have very small fraud counts in the test set
(22 and 2 total fraud cases respectively) - their recall figures are
directionally informative but not statistically reliable at this sample
size; flagged here rather than treated as a strong conclusion.

## G. Remaining methodological concerns (carried forward + new)

- **`chargeback_rate_90d` exclusion is a judgment call, not a strict
  requirement** - it is proven temporally safe; the exclusion is a
  deliberate choice to avoid overstating achievable performance versus
  real-world reporting lag. A future ablation with it included (clearly
  labeled as an unrealistic best-case) could be informative but was not
  built here.
- **`rules_triggered` evidence is underused.** As found in the error
  analysis above, rule *firing* (boolean) is available to the model but
  the underlying evidence magnitude (distance, speed, z-scores computed
  inside `rules/engine.py`) is not - this is the most concrete lever for
  improving `GEOGRAPHIC_INCONSISTENCY`/`UNUSUAL_TIME_OF_DAY` recall.
- **The 3-configuration XGBoost grid is intentionally small** ("tune
  conservatively" per the phase requirements) - a larger search might
  find a better configuration, but was deliberately not run to avoid any
  appearance of tuning toward a specific result.
- **Small-sample segments** (VIP/WATCHLIST risk segments) should not be
  over-interpreted given the low fraud counts involved.
- All Phase 5 audit concerns (baseline staleness heterogeneity,
  Isolation Forest's batch-only NaN imputation design) still apply and
  are unchanged by this phase.

## Reproduction

```bash
python scripts/train_models.py
```

Deterministic given the current database state (`random_state=42`
throughout); full test suite: `python -m pytest -q` (61 tests, all
passing at the time of this report).
