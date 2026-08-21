# Investigation Workflow (Phase 8)

Turns `risk_scores` into an operational queue: alert generation → case
creation → simulated investigator activity → SLA/operational metrics.
This is the layer that demonstrates fraud *operations* thinking, not just
detection - a topic every target role (fraud risk, transaction monitoring,
financial crime analytics) expects candidates to understand alongside the
modeling.

## Alert generation policy: tiers, not the cost-optimal threshold

Alerts are generated for `HIGH` and `CRITICAL` risk_tier transactions -
following `docs/architecture.md`'s risk decision flow (LOW → log only,
MEDIUM → batch review, HIGH → alert, CRITICAL → alert + escalate), not
Phase 7's cost-optimal threshold (0.446 combined_score) directly. **These
are two different, defensible policies that don't have to coincide**:
Phase 7's threshold sweep found 0.446 as maximizing simulated net dollar
impact, but that value sits *below* the HIGH tier's cutpoint (0.486) - it
would alert on some MEDIUM-tier transactions too. Using the tier system
instead is simpler to explain to a business stakeholder ("we investigate
HIGH and CRITICAL"), aligns with a queue that has a small number of
named priority levels rather than a single continuous cutoff, and is what
the architecture was already designed around in Phase 0. The cost-optimal
threshold remains available in `docs/risk_scoring.md` as a data point for
a future threshold-policy discussion, not silently discarded.

On the current dataset: **5,983 alerts** (4,743 HIGH, 1,240 CRITICAL) out
of 103,651 transactions (5.8%).

## Deduplication

Alerts for the same customer within a 60-minute window (`DEDUP_WINDOW_MINUTES`
in `investigation/config.py`) share one `dedup_group_id` - mirroring how a
real velocity-burst fraud event (many flaggable transactions in minutes)
should be investigated as one event, not many. On the current dataset:
5,983 alerts collapse into **5,295 dedup groups** (688 alerts absorbed
into an existing group rather than starting a new one).

**Note on case grain:** `investigation_cases.alert_id` is a single-value
foreign key (see `sql/schema.sql`'s comment: "usually 1:1 with an alert").
This phase creates one case per alert (not per dedup group) to match that
FK directly; `dedup_group_id` is available on `fraud_alerts` for an
investigator UI (Phase 10) to visually group related cases together, but
case *creation* itself stays 1:1 with alerts, as documented in the schema.

## Case creation and SLA

- **CRITICAL** alerts create a case directly in `ESCALATED` status (auto-
  escalated on creation, matching "Alert + Escalate" in the architecture),
  with a **4-hour** SLA.
- **HIGH** alerts create a case in `OPEN` status, with a **24-hour** SLA.

## Investigator simulation - two documented assumptions

There are no real investigators, so case resolution is simulated with two
explicit, parameterized assumptions (`investigation/config.py`):

1. **`INVESTIGATOR_ACCURACY = 0.90`** - simulated investigators reach the
   ground-truth-correct conclusion 90% of the time, not 100%. This is
   deliberate: a perfect-investigator simulation would make the
   false-positive/false-negative operational metrics meaningless, and
   real investigators do make mistakes (fatigue, ambiguous evidence,
   incomplete information).
2. **Backlog** emerges from two combined effects, not one: (a) cases
   created close enough to the end of the simulation window that their
   randomly-drawn resolution time would land after "now" naturally
   haven't finished yet, and (b) an additional `BACKLOG_SHARE = 12%` of
   cases that *would* have finished by "now" are modeled as still stuck in
   the queue anyway (real backlogs aren't purely a function of recency).

**Verification, not just assertion:** the simulated fraud confirmation
rate should equal `true_fraud_rate × accuracy + (1 - true_fraud_rate) × (1 - accuracy)`
if the accuracy simulation is wired correctly. On the current run: true
fraud rate among alerted transactions is 31.87%, predicting a confirmation
rate of 31.87%×0.90 + 68.13%×0.10 = **35.49%**; the simulation produced
**35.80%** - a match within simulation noise, not just plausible-looking.
`tests/test_investigation_workflow.py::test_fraud_confirmation_rate_consistent_with_investigator_accuracy`
is a permanent regression guard for this (tolerance ±3pp).

## Results (current run)

| Metric | Overall | CRITICAL | HIGH |
|---|---|---|---|
| Total cases | 5,983 | 1,240 | 4,743 |
| Resolved (CLOSED) | 5,271 | 1,095 | 4,176 |
| Open (backlog) | 712 | 145 | 567 |
| SLA compliance (resolved cases) | 93.7% | **86.2%** | **95.7%** |
| Median resolution time | 5.97h | 2.11h | 7.27h |
| Fraud confirmation rate | 35.8% | 67.8% | 27.4% |
| False positive rate | 64.2% | 32.2% | 72.6% |

**A genuine operational insight, not a coincidence:** CRITICAL cases are
resolved *faster* in absolute terms (median 2.11h vs. 7.27h for HIGH) but
have *lower* SLA compliance (86.2% vs. 95.7%). This is real and
intuitive once seen: the 4-hour SLA for CRITICAL is proportionally much
tighter than the 24-hour SLA for HIGH, even though CRITICAL cases get
worked faster in wall-clock terms - a queue can be objectively faster and
still miss its target more often if the target is stricter. This is
exactly the kind of trade-off a fraud operations team has to reason about
when setting SLA policy, and it falls directly out of the simulation
rather than being hand-crafted to demonstrate the point.

Investigator workload is evenly distributed (969-1,059 cases per
investigator across 6 simulated investigators) since assignment is
uniform-random - a real capacity model would weight by investigator
seniority/specialty, out of scope here.

## Financial exposure per alert

`fraud_alerts.financial_exposure` is set to the transaction's own amount
(the money at risk if the alert is genuine fraud) - the same "maximum
exposure" interpretation used throughout Phase 7's exposure component,
kept consistent rather than introducing a second, different definition of
exposure at the alert layer.

## Running it

```bash
python scripts/run_risk_scoring.py        # must run first
python scripts/run_investigation_workflow.py
```

Truncates and reloads `fraud_alerts`, `investigation_cases`,
`investigation_actions`, and `audit_log` (in FK-safe order). Deterministic
given `SEED=42` in `investigation/config.py` - re-running produces
identical output, verified by diffing two fresh runs.

## Known limitations

- Investigator assignment is uniform-random, with no capacity ceiling per
  investigator per day - a real system would need queueing/capacity
  constraints to avoid unrealistic instantaneous assignment.
- `INVESTIGATOR_ACCURACY` and `BACKLOG_SHARE` are illustrative constants,
  not derived from any real fraud operation's performance data.
- All synthetic-data limitations from Phases 2, 5, 6, and 7 apply
  identically here - `is_fraud` ground truth used to simulate investigator
  correctness is itself synthetic.
