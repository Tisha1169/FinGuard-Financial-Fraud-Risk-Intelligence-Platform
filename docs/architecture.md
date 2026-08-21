# FinGuard Architecture

## System Architecture

```mermaid
flowchart LR
    A[Transaction Stream Simulator] --> B[Validation]
    B --> C[Enrichment / Feature Engineering]
    C --> D[Rules Engine]
    C --> E[Statistical / Behavioral Engine]
    C --> F[ML Fraud Model]
    D --> G[Risk Aggregation & Scoring]
    E --> G
    F --> G
    G --> H[Financial Exposure Calc]
    H --> I[Alert Generation & Dedup]
    I --> J[Case Management]
    J --> K[Investigation API - FastAPI, local/optional]
    K --> L[Streamlit Analyst Console]
    J --> L
    G --> M[Model & Portfolio Monitoring]
    J --> N[Audit Log]

    subgraph Storage
      P[(PostgreSQL - Neon in prod)]
    end
    B --> P
    G --> P
    J --> P
    M --> P
    L -. reads/writes directly .-> P
```

**Deployment note:** production only ships Neon Postgres + Streamlit
Community Cloud. Streamlit talks to Postgres directly for the deployed app.
FastAPI is built and tested as an architectural layer (and documented with
its own endpoints) but is not deployed as a separate service — it runs
locally/in Docker for development and interview demonstration, avoiding an
unnecessary second hosting dependency.

## Transaction → Investigation Flow

```mermaid
sequenceDiagram
    participant Sim as Transaction Simulator
    participant DB as PostgreSQL
    participant Rules as Rules Engine
    participant Stat as Behavioral Engine
    participant ML as ML Model
    participant Score as Risk Aggregator
    participant Alert as Alert/Case Manager
    participant Analyst as Investigator (Streamlit)

    Sim->>DB: insert fact_transactions
    DB->>Rules: fetch txn + customer/merchant baselines
    Rules->>DB: write rules_triggered
    DB->>Stat: fetch rolling metrics
    Stat->>Score: behavioral anomaly score
    DB->>ML: fetch features
    ML->>DB: write model_predictions
    Rules->>Score: rules severity
    Score->>DB: write risk_scores
    Score->>Alert: if tier >= HIGH, generate alert
    Alert->>DB: write fraud_alerts, investigation_cases
    Analyst->>DB: review case, take action
    Analyst->>DB: write investigation_actions, audit_log
```

## Database Schema (entity relationships)

```mermaid
erDiagram
    dim_customer ||--o{ fact_transactions : makes
    dim_merchant ||--o{ fact_transactions : receives
    dim_device ||--o{ fact_transactions : used_in
    dim_location ||--o{ fact_transactions : occurs_at
    fact_transactions ||--o{ rules_triggered : triggers
    fact_transactions ||--o| model_predictions : scored_by
    fact_transactions ||--o| risk_scores : has
    fact_transactions ||--o| ground_truth_fraud : labeled_as
    fact_transactions ||--o{ fraud_alerts : generates
    fraud_alerts ||--o{ investigation_cases : opens
    investigation_cases ||--o{ investigation_actions : logs
    dim_customer ||--o{ fact_customer_daily_metrics : rolls_up_to
    dim_merchant ||--o{ fact_merchant_daily_metrics : rolls_up_to
```

## Risk Decision Flow

```mermaid
flowchart TD
    T[Transaction] --> R{Rules Engine}
    T --> S{Behavioral/Statistical Engine}
    T --> M{ML Model}
    R --> C[Combine: weighted normalized score]
    S --> C
    M --> C
    C --> E[Add Financial Exposure]
    E --> Tier{Risk Tier}
    Tier -->|LOW| Log[Log only, no alert]
    Tier -->|MEDIUM| Watch[Monitor / batch review]
    Tier -->|HIGH| Alert1[Generate Alert]
    Tier -->|CRITICAL| Alert2[Generate Alert + Escalate]
    Alert1 --> Case[Investigation Case]
    Alert2 --> Case
```

## Model & Monitoring Lifecycle

```mermaid
flowchart LR
    Train[Train on historical window] --> Deploy[Score live transactions]
    Deploy --> Monitor[Track score distribution, PSI, fraud rate]
    Monitor --> Drift{Drift detected?}
    Drift -->|No| Deploy
    Drift -->|Yes| Retrain[Retrain / recalibrate threshold]
    Retrain --> Deploy
```
