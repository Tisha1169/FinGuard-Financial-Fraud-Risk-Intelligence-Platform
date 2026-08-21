# FinGuard — Financial Fraud & Risk Intelligence Platform

> Status: **Phase 3 — SQL feature engineering.** Not yet resume-ready.
> See [docs/architecture.md](docs/architecture.md) for the full design.

FinGuard simulates a bank/fintech fraud operations platform end to end:
transaction ingestion → hybrid detection (rules + statistics + ML) → a
transparent, explainable risk score and financial exposure figure → a
prioritized investigator case queue with SLA tracking → model/portfolio
monitoring.

It is intentionally **not** a binary "predict fraud with XGBoost" notebook —
the ML model is one input into a larger operational decision system, which
is what fraud risk, transaction monitoring, and financial crime analytics
roles actually look like day to day.

## Why this project

Most fraud portfolio projects stop at a classification model and an AUC
number. FinGuard demonstrates the parts that distinguish a fraud/risk
analytics hire: alert economics (cost of false positives vs. missed fraud),
case investigation workflow, SLA and investigator capacity operations,
explainability an investigator can actually read, and honest, leakage-aware
model evaluation.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full Mermaid
diagram set (system architecture, transaction→investigation flow, schema
ERD, risk decision flow, model monitoring lifecycle).

**Deployment:** Neon PostgreSQL (production database) + Streamlit
Community Cloud (analyst console). No separate API host — Streamlit
connects to Postgres directly in production; FastAPI exists as a documented
architectural layer used locally.

## Tech stack

Python, PostgreSQL, SQLAlchemy, pandas/NumPy/SciPy, scikit-learn, XGBoost,
imbalanced-learn, SHAP, FastAPI, Streamlit, Plotly, Docker, pytest.

## Repository structure

```
data_generation/   synthetic transaction/entity generator
data/              generated datasets (not committed)
sql/               schema.sql, key queries
database/          connection layer
features/          feature engineering (SQL + pandas)
rules/             business rule engine
models/            statistical + ML fraud models
risk_scoring/       combined score + financial exposure
investigation/     case/alert management logic
api/               FastAPI service (local/dev)
streamlit_app/     analyst console
monitoring/        drift, PSI, model performance tracking
tests/             pytest suite
deployment/        Docker, Neon/Streamlit deployment config
docs/              architecture, data dictionary, model card, etc.
scripts/           one-off / setup scripts (e.g. init_db.py)
```

## Local setup

```bash
cp .env.example .env
docker compose up -d
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/init_db.py
python scripts/generate_data.py   # writes data/*.parquet
python scripts/load_data.py       # loads them into DATABASE_URL
python scripts/compute_features.py # populates rolling baseline tables
```

See [docs/data_generation.md](docs/data_generation.md) for how the
synthetic dataset and its 9 fraud typologies are constructed, and
[docs/feature_engineering.md](docs/feature_engineering.md) for the rolling
baseline / leakage-prevention design.

## Data disclosure

All transaction and fraud-label data in this project is **synthetic**,
generated to mimic realistic fraud typologies (card testing, velocity
abuse, account takeover, geographic inconsistency, etc.). No real
transaction or customer data is used. `ground_truth_fraud.is_synthetic_label`
is always `true`. No claims of real-world fraud prevention or financial
savings are made anywhere in this project — all business impact figures are
explicitly simulated/estimated.

## Roadmap

MVP → RESUME-READY → FLAGSHIP milestones and phase-by-phase build log are
tracked as the project progresses; documentation will be filled in as each
phase completes (data dictionary, model card, evaluation report, business
case, deployment guide, interview prep).
