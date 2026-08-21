"""FastAPI application - the local/dev architectural layer documented in
docs/architecture.md. Not deployed separately in production: Streamlit
Community Cloud talks to Neon Postgres directly (see docs/architecture.md
"Deployment note"). This API exists to demonstrate a clean service
boundary and is exercised by its own test suite
(tests/test_api.py) plus, optionally, local Streamlit development.

Run locally:
    uvicorn api.main:app --reload
"""
from fastapi import FastAPI

from api.routers import alerts, cases, health, metrics, risk, transactions

app = FastAPI(
    title="FinGuard API",
    description="Local/dev API for the FinGuard fraud risk intelligence platform.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(transactions.router)
app.include_router(alerts.router)
app.include_router(cases.router)
app.include_router(risk.router)
app.include_router(metrics.router)
