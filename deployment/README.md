# Deployment

Full deployment guide (Neon + Streamlit Community Cloud, step-by-step,
readiness checks, troubleshooting): see [docs/deployment.md](../docs/deployment.md).

## Local (quick reference)

```bash
docker compose up -d
python scripts/init_db.py
python scripts/generate_data.py && python scripts/load_data.py
python scripts/compute_features.py
python scripts/run_rules.py
python scripts/run_statistical_detection.py
python scripts/run_risk_scoring.py
python scripts/run_investigation_workflow.py
streamlit run streamlit_app/app.py
```

## Production

Neon PostgreSQL + Streamlit Community Cloud. Streamlit talks to Neon
directly - no separate API hosting (see `docs/architecture.md`'s
deployment note). Account setup and secrets configuration are covered in
`docs/deployment.md`.
