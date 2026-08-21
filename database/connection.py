"""Database connection helper shared by the pipeline, API, and Streamlit app.

Reads DATABASE_URL from the environment (Docker Postgres locally, Neon in
production). Streamlit Community Cloud injects this via st.secrets rather
than a real env var, so we check both.
"""
import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    try:
        import streamlit as st

        return st.secrets["DATABASE_URL"]
    except Exception:
        pass

    raise RuntimeError(
        "DATABASE_URL is not set. Copy .env.example to .env for local "
        "development, or set DATABASE_URL in Streamlit secrets for the "
        "deployed app."
    )


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(get_database_url(), pool_pre_ping=True)
    return _engine


def check_connection() -> tuple[bool, str]:
    """Returns (is_healthy, message). Never raises — for use in health checks."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "connected"
    except Exception as exc:
        return False, f"database unreachable: {exc}"
