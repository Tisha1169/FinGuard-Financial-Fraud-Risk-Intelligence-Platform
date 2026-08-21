# Streamlit Analyst Console (Phase 10)

The primary demo surface: a 5-tab dashboard connecting directly to
Postgres - no FastAPI dependency, matching the production deployment
(Streamlit Community Cloud → Neon, see `docs/architecture.md`'s
deployment note).

## Architecture

- `streamlit_app/db.py` - all read queries, cached with `st.cache_data`
  (60s TTL) so repeated tab switches don't re-hit Postgres unnecessarily.
  Write operations (case assign/action) are **not** cached and explicitly
  invalidate the relevant caches afterward (`db.clear_case_caches()`) so
  the UI reflects a change immediately rather than showing stale data for
  up to 60 seconds.
- `streamlit_app/model_cache.py` - trains the Logistic Regression/XGBoost/
  Isolation Forest pipeline **once per app process** via
  `st.cache_resource` (~10s cold start, matching `scripts/train_models.py`'s
  own runtime). Retraining on every user interaction would make the app
  unusably slow; this mirrors how a real deployment would load a frozen
  model rather than retraining per request.
- `investigation/case_actions.py` (built alongside this phase, shared with
  the API) - the case-mutation logic. Both `api/routers/cases.py` and
  `streamlit_app/tabs/alert_queue.py` call the *same* functions, because in
  production Streamlit talks to Postgres directly and can't call the API
  over HTTP - duplicating the mutation logic in two places was avoided
  from the start rather than fixed after the fact.

## The 5 tabs

1. **Executive Risk Overview** - KPI row (transaction volume, alerts,
   confirmed fraud, financial exposure, estimated loss prevented,
   false-positive rate, open investigations, SLA compliance), transaction/
   fraud trend, risk tier distribution.
2. **Transaction Risk Intelligence** - filterable/searchable transaction
   table joined with risk scores; per-transaction score breakdown and
   rule evidence.
3. **Fraud Alert & Investigation Queue** - HIGH/CRITICAL cases sorted by
   score, with live assign/action forms that write to Postgres and are
   validated against `investigation/state_machine.py`.
4. **Case Investigation - Customer 360** - full workbench: transaction,
   customer/merchant context, risk score breakdown, rules with evidence,
   SHAP model explanation (when available), customer transaction
   timeline, previous alerts, and audit history.
5. **Model & Portfolio Monitoring** - test-set PR-AUC/ROC-AUC/precision/
   recall (XGBoost vs. Logistic Regression), score distribution, global
   SHAP importance, live financial threshold sweep, segment performance
   (by channel/risk segment/merchant category), hyperparameter selection
   trials.

## Verified end-to-end (not just written)

Ran the app locally against live Postgres and drove it through the
browser:
- All 5 tabs render with real data matching the numbers documented in
  `docs/evaluation_report.md` and `docs/risk_scoring.md` exactly (e.g.
  PR-AUC 0.793, `CARD_PRESENT` channel recall 39.66% in the segment
  table - identical to the error-analysis figures in the evaluation
  report).
- Filtering (risk tier, channel, amount) on the Transaction Intelligence
  tab works and updates results correctly.
- Took a live case (3658) through `ESCALATED → IN_REVIEW` via the queue's
  action form; confirmed the write actually persisted in Postgres (not
  just a UI state change) and the audit trail recorded it correctly.
- Confirmed the Customer 360 tab's rule evidence renders as real nested
  JSON, not a stringified blob (same class of bug caught and fixed in
  Phase 9's API - `streamlit_app/db.py::_parse_evidence` applies the
  identical fix for the dashboard's own raw queries).

## A UX bug found and fixed during manual testing

The "Take action on a case" form's Case ID field was recomputed from
`df.iloc[0]["case_id"]` on every render - after `st.rerun()` following a
successful action, this silently reset the field back to the queue's top
row, losing the case the analyst was actually working on (confirmed by
watching it happen: applied `INVESTIGATE` to case 3658, and the form
snapped back to case 2172). Fixed by keying the widget to
`st.session_state["alert_queue_case_id"]`, initialized once rather than
recomputed every render - verified fixed by repeating the same action and
confirming the field stayed on 3658 with its status correctly updated.

## Running it

```bash
streamlit run streamlit_app/app.py
```

Local `.env` is loaded automatically (`streamlit_app/db.py` calls
`load_dotenv()`, since Streamlit doesn't do this itself the way the CLI
scripts do). For Streamlit Community Cloud, set `DATABASE_URL` in the
app's secrets (see `.streamlit/secrets.toml.example`).

## Known limitations

- SHAP explanations in the Case Investigation tab are only available for
  transactions in the cached model's TEST split (out-of-sample) - most
  looked-up cases won't have one, and the UI falls back to the risk score
  breakdown and rule evidence, which are available for every transaction.
- The financial threshold sweep on the Model Monitoring tab is computed
  live across the *full* dataset (all risk-scored transactions), so it
  will differ slightly from the validation-only sweep reported in
  `docs/risk_scoring.md` - stated explicitly in the tab's caption rather
  than presented as the same number.
- 60-second cache TTLs mean KPIs can lag a write by up to a minute on
  tabs other than the one where the write happened (that tab's cache is
  explicitly cleared) - acceptable for a demo, not tuned for real-time
  multi-user use.
