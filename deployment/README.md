# Deployment

## Local
`docker compose up -d` starts Postgres. Run `python scripts/init_db.py` to
apply the schema, then run the Streamlit app / API locally against
`DATABASE_URL` in `.env`.

## Production
- **Database:** Neon PostgreSQL (serverless Postgres, free tier). Create a
  project, copy the pooled connection string into Streamlit Community
  Cloud's app secrets as `DATABASE_URL`.
- **App:** Streamlit Community Cloud, pointed at this repo's
  `streamlit_app/app.py` (added in Phase 10). Secrets are configured in the
  Streamlit Cloud dashboard, never committed.
- No separate API deployment: Streamlit talks to Neon directly in
  production. FastAPI (added in Phase 9) is developed and tested locally as
  an architectural layer, documented but not separately hosted.

Full deployment steps and a dataset-sizing plan for Neon's free tier will
be finalized in Phase 14 (`docs/deployment.md`).
