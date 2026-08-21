# API (Phase 9)

A local/dev FastAPI service over the same Postgres tables every other
phase's scripts use - clean endpoints, but not deployed separately in
production. See "Deployment" below for why.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + DB connectivity check |
| GET | `/transactions` | Filterable/paginated transaction list, joined with risk score |
| GET | `/alerts` | HIGH/CRITICAL alerts, joined with their case status, sorted by score |
| GET | `/cases` | Filterable/paginated case list (status, risk tier, investigator) |
| GET | `/cases/{case_id}` | Full investigation workbench payload - see below |
| POST | `/cases/{case_id}/assign` | Assign an investigator, logs an ASSIGN action |
| POST | `/cases/{case_id}/action` | Apply a status-transitioning action, validated against the case state machine |
| GET | `/risk/{transaction_id}` | Combined risk score breakdown + the rule evidence behind it |
| GET | `/metrics` | Live portfolio/operational snapshot (not a cached batch run) |

### `GET /cases/{case_id}` - the investigation workbench payload

Everything an investigator needs on one screen in one call: the case, its
transaction, customer/merchant risk context, the full risk score
breakdown, every rule that fired (with structured evidence), the complete
action history, and `valid_next_actions` (from
`investigation/state_machine.py`) so a UI knows which buttons to show
without re-deriving the state machine client-side.

### `POST /cases/{case_id}/action` - state-machine-validated

Applying an action not valid from the case's current status (e.g.
`CLOSE` on an `OPEN` case) returns **409 Conflict** with a message
explaining what statuses that action *is* valid from - verified directly:

```
$ curl -X POST localhost:8000/cases/123/action -d '{"action_type":"CLOSE","performed_by":"a.chen"}'
{"detail":"action CLOSE is not valid from status IN_REVIEW (valid from: ['CONFIRMED_FRAUD', 'FALSE_POSITIVE'])"}
```

The same `investigation/state_machine.py` module used by
`scripts/run_investigation_workflow.py`'s simulation drives this endpoint
- there is exactly one definition of valid transitions in the codebase,
not one for the batch simulation and a separate one for the API.

### `GET /risk/{transaction_id}` - lookup, not a live scorer

Reads directly from `risk_scores` (Phase 7) and `rules_triggered`
(Phase 4) - it does **not** recompute the ML/behavioral components on the
fly for an arbitrary transaction. A true real-time scoring endpoint would
need the single-transaction path already built in `features/point_in_time.py`
and `rules/engine.py` (documented in Phase 4 specifically for this
purpose) wired up to run the trained model per-request; that's a larger
undertaking (loading a frozen model artifact, handling a transaction that
doesn't exist in the DB yet) explicitly out of scope for this phase.

### `GET /metrics` - live, not cached

Computed from the database's current state on every call (via
`investigation/metrics.py`, the same module `scripts/run_investigation_workflow.py`
uses) - so an action taken through `POST /cases/{id}/action` is reflected
immediately. Verified during development: resolving one case via the API
moved `open_cases`/`resolved_cases` in the very next `/metrics` call.

## A real bug found and fixed during manual testing

SQLAlchemy Core's raw `text()` queries don't carry column-type metadata,
so `rules_triggered.evidence` (a JSONB column) came back as a **JSON-
encoded string**, not a parsed object - meaning API responses nested a
stringified blob inside JSON rather than real nested JSON. Caught by
actually curling the endpoint and reading the response, not by assuming
the ORM/driver would handle it. Fixed with `api/dependencies.py::parse_rule_row`,
applied everywhere `rules_triggered` rows are returned
(`/cases/{id}` and `/risk/{id}`), with a regression test
(`tests/test_api.py::test_get_case_detail_evidence_is_parsed_json_not_a_string`).

## Deployment

Per `docs/architecture.md`: **not deployed as a separate service.**
Streamlit Community Cloud (Phase 10) talks to Neon Postgres directly in
production - adding a second hosted service here would be unnecessary
infrastructure for a portfolio deployment. This API exists to demonstrate
a clean service boundary and is exercised by its own test suite; running
it locally is still useful for development/testing without going through
Streamlit.

## Running it locally

```bash
uvicorn api.main:app --reload
```

Interactive docs at `http://localhost:8000/docs` (FastAPI's automatic
OpenAPI UI).

## Testing

`tests/test_api.py` uses FastAPI's `TestClient` (in-process, no server
process needed) against the real local dev database - consistent with
every other script in this project, since the dataset is entirely
regenerable. Write-endpoint tests pick a fresh non-terminal case each run
rather than depending on specific prior case state.

`tests/test_state_machine.py` covers the transition rules in isolation
(no DB) - including a structural sanity check that every non-terminal
status has at least one valid outgoing action (no accidental dead ends).

## Known limitations

- No authentication/authorization - appropriate for a local/dev
  demonstration layer, not for a real deployment.
- `/risk/{transaction_id}` is a lookup against pre-computed batch scores,
  not a live scorer for an arbitrary new transaction (see above).
- No rate limiting or request validation beyond Pydantic's type/pattern
  checks.
