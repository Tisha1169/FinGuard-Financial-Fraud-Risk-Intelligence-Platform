# Risk Scoring & Financial Exposure (Phase 7)

Combines the outputs of Phases 4-6 into one transparent, explainable
`combined_score` per transaction, assigns a `risk_tier`
(LOW/MEDIUM/HIGH/CRITICAL), and simulates the financial trade-off of
alerting at different score thresholds. This is the layer that turns four
separate detection signals into one number an investigation queue can be
sorted by (Phase 8).

## Why reuse Phase 6's pipeline instead of Phase 5's scores

`risk_scoring/pipeline.py` rebuilds the exact same chronological split,
train-only-fit Isolation Forest, train-only-fit preprocessing, and
train-only-fit XGBoost that Phase 6 used - it does **not** reuse Phase 5's
`models/isolation_forest.py` (which fits on the whole dataset). Using the
whole-dataset version here would reintroduce, in the risk score, exactly
the leakage risk Phase 6's audit fixed for the ML model: a component that
had already "seen" the validation/test period's structure when it was fit.
The `behavioral_anomaly_score` (IQR/frequency) component, by contrast, is
reused directly from Phase 5 - it involves no cross-transaction learning or
label fitting, only point-in-time rolling statistics that are already
leakage-safe by construction regardless of dataset scope (see
[docs/feature_engineering.md](feature_engineering.md)).

Every transaction is scored, including the training period. This mirrors
a real deployment: a model is trained once and then scores both the
history it learned from and everything after. Training-period
`ml_component` values are **in-sample** (the model saw these exact labels
while fitting); validation/test values are **out-of-sample**. This is
not hidden - `risk_scoring/pipeline.py` explicitly tracks this via
`ml_in_sample`.

## The four components

| Component | Weight | Source | Range |
|---|---|---|---|
| `ml_component` | 0.40 | XGBoost fraud probability (Phase 6, retrained here on the same train split) | [0,1] |
| `rules_component` | 0.20 | Base score from highest-severity rule fired (LOW=0.25 ... CRITICAL=1.00), plus up to +0.20 bonus for multiple independent rules firing | [0,1] |
| `behavioral_component` | 0.20 | Noisy-OR of `behavioral_anomaly_score` (IQR/frequency) and `isolation_forest_score` (train-only-fit) | [0,1] |
| `exposure_component` | 0.20 | `amount / exposure_cap`, clipped to 1.0 | [0,1] |

`combined_score = 0.40·ml + 0.20·rules + 0.20·behavioral + 0.20·exposure`
- a weighted sum, chosen over a noisy-OR or product for the *final* blend
specifically because it's the most transparent option: every component's
dollar-and-cents contribution to the final number is directly readable
("this transaction scored 0.72; 0.35 came from the ML model, 0.15 from
rules, 0.12 from behavioral anomaly, 0.10 from its transaction size").
Investigators need that breakdown (Phase 8's Customer 360 view depends on
it), which a noisy-OR combination - correct for combining independent
*evidence of anomaly* in Phase 5, where no single readable number was
needed per component - would obscure.

### Weight justification (reasoned, not fit to data)

Weights were chosen by argument, not by searching for the combination
that maximizes any evaluation metric - optimizing weights against ground
truth here would defeat the purpose of a transparent, explainable score
(that optimization already happened, correctly, inside the ML model in
Phase 6). `ml` gets the largest share (0.40) because it's the most
validated predictive signal (Phase 6 test PR-AUC 0.793). `rules` and
`behavioral` get equal weight (0.20 each) as complementary, independently
useful signals. `exposure` gets a meaningful but bounded 0.20 specifically
so transaction size can raise priority without ever letting size alone
dominate - a large legitimate purchase can add at most 0.20 to the score,
not override genuine fraud evidence from the other three components.

### `exposure_component`: cap chosen from training data only

`exposure_cap` is the 95th percentile of **training-period** transaction
amounts ($124.72 in the current run), frozen and applied unchanged to
validation and test. Computing this from the full dataset (including
future periods) would be a mild form of distributional lookahead -
avoided here for the same reason every other statistic in this pipeline is
fit-on-train-only.
`tests/test_risk_scoring.py::test_exposure_cap_uses_train_amounts_only`
enforces this.

## Risk tiers

Tier cutpoints are **quantiles of the combined-score distribution on the
validation split only** (never test, never the full population) - top 1%
→ CRITICAL, next 4% → HIGH, next 15% → MEDIUM, remaining 80% → LOW. These
are absolute score thresholds once computed, then applied identically to
train/val/test. Using validation (not test) to set the cutpoints follows
the same discipline as Phase 6's threshold selection: test is touched
only for a final, unbiased look.
`tests/test_risk_scoring.py::test_tier_cutpoints_derived_from_validation_only`
is a regression test for this.

### Tier separation (sanity check, full population)

| Tier | Count | Fraud rate | Avg combined score |
|---|---|---|---|
| CRITICAL | 1,240 | 70.6% | 0.937 |
| HIGH | 4,743 | 21.7% | 0.599 |
| MEDIUM | 16,970 | 1.08% | 0.307 |
| LOW | 80,698 | 0.03% | 0.096 |

Fraud rate increases monotonically and dramatically across tiers (0.03% →
70.6%, a >2000x spread from LOW to CRITICAL) - direct evidence the
combined score is doing its job, not just an artifact of the weights
looking reasonable on paper.
`tests/test_risk_scoring.py::test_risk_tiers_separate_fraud_monotonically`
is a permanent regression guard for this.

## Financial exposure simulation

All dollar figures are explicitly simulated from documented assumptions
in [risk_scoring/config.py](../risk_scoring/config.py) - **never a claim
of real financial impact**:

- `INVESTIGATION_COST_USD = $12.00` per case reviewed (simulated analyst
  time).
- `FRAUD_RECOVERY_RATE_IF_CAUGHT = 80%` - the assumed share of a flagged
  fraudulent transaction's amount that's recoverable (vs. lost) if caught
  before settlement.

For a given alert threshold: `estimated_loss_prevented` = (amount of
true-positive alerts) × recovery rate; `unprevented_loss` = full amount of
missed fraud (false negatives); `net_expected_impact` = loss prevented −
total investigation cost − unprevented loss.

### Validation threshold sweep (excerpt)

| Threshold | Alerts | Precision | Recall | Investigation cost | Loss prevented | Net impact |
|---|---|---|---|---|---|---|
| 0.815 (≈CRITICAL cutpoint) | 156 | 73.1% | 41.3% | $1,872 | $22,532 | **$13,637** |
| **0.446 (best)** | **958** | **26.6%** | **92.4%** | **$11,496** | **$27,616** | **$15,452** |
| 0.302 | 1,760 | 15.1% | 96.0% | $21,120 | $27,881 | $6,425 |
| 0.238 | 2,561 | 10.5% | 97.1% | $30,732 | $27,974 | **−$2,978** |

The trade-off is visible and economically coherent: alerting on the
CRITICAL tier alone is too conservative (misses 59% of fraud value,
leaving net impact on the table); lowering the threshold too far adds
investigation cost faster than it recovers additional loss, eventually
turning net impact *negative*. The maximum sits at threshold 0.446 (net
impact $15,452, 92.4% recall at 26.6% precision) - full sweep in
`artifacts/phase7_risk_scoring.json` (gitignored, regenerate via the
script below).

**Why the "best" precision (26.6%) looks low:** this threshold is chosen
to maximize *net dollar impact*, not precision - a $12 investigation cost
is cheap relative to the average fraud transaction's recoverable value, so
it's economically worth investigating many false positives as long as
recall stays high. This is a deliberate, visible trade-off, not an
oversight; a fraud operation with a higher per-case investigation cost or
lower fraud-value transactions would land on a different optimal
threshold, which is exactly what re-running the sweep with updated
`risk_scoring/config.py` constants would show.

### Final unbiased test-set look

Applying the validation-selected threshold (0.446) to the untouched test
split: 960 alerts, 256 true positives (86.5% recall), 704 false positives
(26.7% precision), net expected impact **$15,988** - closely matching
validation's $15,452, evidence the threshold generalizes rather than
having been cherry-picked to validation's specific data.

## Running it

```bash
python scripts/run_risk_scoring.py
```

Rebuilds features, split, Isolation Forest, preprocessing, and XGBoost
from scratch (~10-12 seconds), populates `risk_scores` (truncate +
reload), prints the tier distribution/fraud-rate sanity check and the
full threshold sweep, and writes `artifacts/phase7_risk_scoring.json`.

## Known limitations

- Weights (0.40/0.20/0.20/0.20) are reasoned defaults, not fit or
  cross-validated against any objective - a deliberate choice to keep the
  score's methodology auditable, but it means the weighting itself hasn't
  been empirically validated as optimal.
- The financial constants ($12 investigation cost, 80% recovery rate) are
  illustrative assumptions for demonstrating the trade-off methodology,
  not derived from any real institution's cost structure.
- `net_expected_impact`'s "unprevented loss" assumes 100% of an uncaught
  fraudulent transaction's amount is lost - real chargeback/liability
  rules are more nuanced (card network liability shift rules, partial
  recovery via other means) and are out of scope here.
- All synthetic-data limitations from Phases 2, 5, and 6 apply
  identically to every number on this page - see
  [docs/model_card.md](model_card.md) "Synthetic-data caveats" for the
  full list.
