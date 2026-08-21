"""Shared FastAPI dependencies. Reuses database/connection.py rather than
introducing a second DB-access pattern - the API is a thin read/write
layer over the same Postgres tables every other phase's scripts use.
"""
import json

from sqlalchemy.engine import Engine

from database.connection import get_engine


def db_engine() -> Engine:
    return get_engine()


def parse_rule_row(row: dict) -> dict:
    """SQLAlchemy Core's raw text() queries don't know a column is JSONB,
    so `evidence` comes back as a JSON-encoded string rather than a parsed
    object - re-parse it so API responses nest real JSON, not a stringified
    blob inside the response body.
    """
    out = dict(row)
    if isinstance(out.get("evidence"), str):
        out["evidence"] = json.loads(out["evidence"])
    return out
