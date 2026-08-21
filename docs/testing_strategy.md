# Testing Strategy (Phase 11)

161 tests, 85% statement coverage across the non-UI codebase (`pytest
--cov`). This document explains what's tested, how, and why the remaining
gaps are gaps by decision, not oversight.

## Coverage by area

| Area | Coverage | Notes |
|---|---|---|
| Data generation (`data_generation/`) | 88-100% | Reproducibility, typology correctness, referential integrity |
| Database / schema (`database/`) | 92% | Connection health-check paths |
| Feature engineering (`features/`) | 100% | Leakage boundaries, independent-pandas cross-check |
| Rules engine (`rules/`) | 85-99% | Batch-vs-live consistency on real data; see "Known gaps" below |
| Statistical/anomaly detection (`models/statistical.py`, `models/isolation_forest.py`) | 100% | Calibration regression guards for two real bugs found in Phase 5 |
| Supervised ML (`models/ml_*.py`) | 83-100% | Split integrity, train-only fitting, SHAP additivity |
| Risk scoring (`risk_scoring/`) | 95-100% | Component math, tier boundaries, financial simulation |
| Investigation workflow (`investigation/`) | 95-100% | State machine, SLA calculations, case mutation logic |
| API (`api/`) | 76-100% | Every endpoint, filter combination, 404/409 error paths |
| Streamlit dashboard (`streamlit_app/`) | 0-36% (pure helpers only) | See "UI testing approach" below |

## Test categories (mapping to the original requirement list)

- **Database access**: `tests/test_database_connection.py`, every
  integration test implicitly (all read/write through `database/connection.py`).
- **Data validation**: `tests/test_data_generation.py` (referential
  integrity, no full PAN storage, amount bounds, reproducibility).
- **Feature generation**: `tests/test_features.py` (SQL rolling stats
  independently cross-checked against pandas), `tests/test_statistical.py`.
- **Rules engine**: `tests/test_rules.py` (unit + batch-vs-live
  consistency on real data), `tests/test_geo.py` (haversine/speed math).
- **Risk scoring**: `tests/test_risk_scoring.py` (component math, tier
  monotonicity against real fraud rates, financial simulation arithmetic).
- **Model prediction**: `tests/test_ml_pipeline.py` (shape/type/range
  contracts, determinism), `tests/test_ml_explain.py` (SHAP additivity -
  the property that proves explanations are real, not fabricated).
- **Threshold logic**: `tests/test_ml_pipeline.py::test_find_threshold_for_top_k`
  and `risk_scoring`'s threshold-sweep tests.
- **Case creation / status transitions**: `tests/test_state_machine.py`
  (14 tests, including a structural check that every non-terminal status
  has at least one valid outgoing action), `tests/test_case_actions.py`.
- **SLA calculations**: `tests/test_investigation_workflow.py` (SLA
  compliance rate computation against hand-built case data).
- **API health**: `tests/test_api.py::test_health_ok` plus every endpoint
  exercised via `TestClient`, including 404s and 409s.
- **Dashboard queries**: `tests/test_streamlit_db.py` (pure-logic parts);
  the SQL queries themselves are the same functions verified end-to-end
  by actually running the dashboard against live data (see
  `docs/dashboard.md`'s "Verified end-to-end" section) - a more direct
  correctness check for a rendering layer than mocking Postgres would be.

## UI testing approach: browser verification, not unit tests

`streamlit_app/tabs/*.py` and `streamlit_app/app.py` show 0% pytest
coverage by design, not neglect. Unit-testing Streamlit's rendering code
would mean mocking `st.dataframe`, `st.plotly_chart`, `st.form`, etc. and
asserting on mock call arguments - which verifies "did the code call the
right Streamlit function," not "does the dashboard actually show correct
data to a user." For this project, the dashboard was instead verified by
**actually running it** against live Postgres and driving it through a
real browser (Phase 10): every tab loaded, filters were exercised, a case
was moved through a real status transition and the write was confirmed
in the database, and the displayed numbers were cross-checked against the
independently-computed figures in `docs/evaluation_report.md` and
`docs/risk_scoring.md`. That is a stronger correctness check for this
specific layer than an assertion-based unit test would be, though it
isn't automatically re-run - a real limitation, listed below.

The parts of `streamlit_app/db.py` that **are** pure logic
(`_parse_evidence`, the JSON-parsing fix for a real bug found during that
manual testing) are unit tested directly (`tests/test_streamlit_db.py`),
since that logic doesn't depend on Streamlit's rendering at all.

## Known gaps (accepted, not hidden)

- **`rules/engine.py` at 85%** - the missing lines are negative/edge
  branches in individual rule functions (e.g. a rule returning `None`
  when a precondition isn't met) that aren't independently unit-tested
  one-by-one; the 99%-covered `rules/batch.py` and the batch-vs-live
  consistency test (`test_rules.py::test_batch_engine_matches_single_transaction_engine_on_a_sample`)
  exercise the same logic against real data, which caught the one real
  bug in this module (the `Decimal`/`float` arithmetic issue, Phase 4).
- **`models/ml_metrics.py` at 83%** - missing lines are in
  `full_evaluation`'s aggregation wrapper, whose individual metric
  functions (`pr_auc`, `precision_recall_f1_at_threshold`,
  `recall_at_fixed_fpr`, `precision_recall_at_top_k`,
  `confusion_matrix_at_threshold`) are each hand-computed and verified
  directly in `tests/test_ml_pipeline.py`.
- **No automated UI regression suite** - re-verifying the dashboard after
  a change currently means re-running it manually in a browser, as done
  in Phase 10. A tool like Playwright could automate this in a future
  iteration; not built here to keep the testing approach proportionate to
  a portfolio project's scope.

## Running the suite

```bash
python -m pytest -q                              # full suite
python -m pytest --cov=. --cov-report=term-missing  # with coverage
```

All DB-dependent tests skip gracefully (not fail) when `DATABASE_URL`
isn't set - confirmed: 91 tests pass standalone with no database, the
remaining 70 skip cleanly.
