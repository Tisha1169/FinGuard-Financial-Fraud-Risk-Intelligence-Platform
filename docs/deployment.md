# Deployment (Phase 14)

Production target: **Neon PostgreSQL + Streamlit Community Cloud.** No
separate API hosting - Streamlit talks to Neon directly (see
`docs/architecture.md`'s deployment note). This doc covers what's already
verified and exactly what's left to do, since the remaining steps need
your own Neon and Streamlit Cloud accounts - account creation and OAuth
authorization aren't things I do on your behalf.

## Readiness checks completed

- **Dataset size:** current local database is **103 MB total**
  (`fact_transactions` 37 MB, `fact_customer_daily_metrics` 19 MB,
  `risk_scores` 11 MB, rest smaller) - comfortably within Neon's free-tier
  storage allowance. No trimming needed.
- **Clean-install verification:** `requirements.txt` installs from
  scratch into a brand-new virtualenv with no errors (~25s), and every
  package (`pandas`, `xgboost`, `shap`, `streamlit`, `sqlalchemy`,
  `psycopg2-binary`, etc.) imports cleanly afterward - this is exactly
  what Streamlit Community Cloud's build step will do.
- **Secrets handling:** `database/connection.py` already reads
  `DATABASE_URL` from `st.secrets` when the environment variable isn't
  set (built in Phase 1, specifically for this deployment target) -
  no code changes needed to point the app at Neon instead of local
  Docker Postgres.
- **Health check:** `GET /health` (API) and `database.connection.check_connection()`
  (used directly by the dashboard) both return a friendly
  `{"status": "degraded", "database": "<error message>"}` rather than
  crashing if the database is unreachable - verified in
  `tests/test_database_connection.py` and `tests/test_api.py::test_health_ok`.
- **Platform risk flagged:** XGBoost required installing the `libomp`
  system library locally on macOS (its Python wheel doesn't bundle it on
  that platform) - this was a real blocker caught during Phase 6
  development. Streamlit Community Cloud runs on Debian Linux, where
  XGBoost's Linux wheels typically bundle or find `libgomp` without
  extra setup, but **this should be the first thing checked if the
  deployed app's build fails** - see "Troubleshooting" below.

## Step-by-step: what's left (needs your accounts)

### 1. Neon PostgreSQL

1. Create a free account at neon.tech (or sign in) and create a new project.
2. Copy the **pooled** connection string from the Neon dashboard (starts
   `postgresql://...`, includes `?sslmode=require`).
3. Apply the schema and load the dataset against Neon. From your local
   machine, with the Neon connection string as `DATABASE_URL`:
   ```bash
   export DATABASE_URL="<your Neon pooled connection string>"
   python scripts/init_db.py
   python scripts/generate_data.py
   python scripts/load_data.py
   python scripts/compute_features.py
   python scripts/run_rules.py
   python scripts/run_statistical_detection.py
   python scripts/run_risk_scoring.py
   python scripts/run_investigation_workflow.py
   ```
   This reuses the exact same scripts as local development - nothing
   Neon-specific about them, since they only depend on `DATABASE_URL`.
   Expect this to take a few minutes total (most of it is
   `scripts/train_models.py`-equivalent steps re-fitting models; the
   rules/statistical passes are seconds each locally, but network latency
   to Neon will add overhead per query compared to local Docker).
4. If you'd rather I run these steps for you once you have the connection
   string, share it and I can do so - running SQL/scripts against a
   database you've already created and given me credentials for is a
   normal coding task, distinct from creating the Neon account itself.

### 2. Streamlit Community Cloud

1. Sign in to share.streamlit.io with your GitHub account (this repo must
   be pushed to GitHub, which it already is:
   https://github.com/Tisha1169/FinGuard-Financial-Fraud-Risk-Intelligence-Platform).
2. Click "New app", select this repo, branch `main`, main file path
   `streamlit_app/app.py`.
3. Before deploying, open "Advanced settings" → Secrets, and paste:
   ```toml
   DATABASE_URL = "<your Neon pooled connection string>"
   ```
   (matches `.streamlit/secrets.toml.example` in this repo - never commit
   the real secrets file, it's gitignored).
4. Deploy. First load will take ~10-15s longer than subsequent ones,
   since `streamlit_app/model_cache.py` trains the ML pipeline once per
   app instance (`st.cache_resource`) - this is expected, not a bug.

### 3. Verify the deployed app

Once live, check:
- All 5 tabs load without errors.
- Executive Overview KPIs show non-zero numbers (confirms the DB load
  in step 1 succeeded).
- Model & Portfolio Monitoring tab shows PR-AUC ≈0.79 for XGBoost -
  confirms the model trained successfully against the Neon data.
- Take one action in the Fraud Alert & Investigation Queue tab and
  confirm it doesn't error (confirms write access to Neon works, not
  just read).

## Troubleshooting

- **Build fails with an XGBoost/libomp error:** Streamlit Community Cloud
  supports an `apt.txt` or `packages.txt` file at the repo root listing
  system packages to install before the Python build - add `libgomp1` if
  this occurs. Not added preemptively since Debian typically ships it by
  default and adding unnecessary system dependencies is avoided per this
  project's "prefer simple robust architecture" principle - add it only
  if the actual deployed build shows the error.
- **App loads but shows no data:** confirms `DATABASE_URL` in Streamlit
  secrets doesn't match what was loaded in the Neon setup step, or the
  load step 1 wasn't completed against that same Neon database.
- **Slow first load:** expected (model training cold-start) - see step 2.4
  above. If every subsequent load is also slow, Streamlit Cloud may have
  put the app to sleep after inactivity and is cold-starting again
  (free-tier apps sleep after a period of no traffic) - not a bug.

## Local development (unchanged)

```bash
docker compose up -d
python scripts/init_db.py
# ... same pipeline scripts as above, against local DATABASE_URL in .env
uvicorn api.main:app --reload            # optional, local-only
streamlit run streamlit_app/app.py
```
