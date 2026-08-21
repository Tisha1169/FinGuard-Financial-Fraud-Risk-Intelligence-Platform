"""Phase 1 smoke test: DB connection helper behaves correctly with and
without DATABASE_URL set. Does not require a live database.
"""
import os

import pytest

from database.connection import get_database_url


def test_get_database_url_reads_env_var(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")
    assert get_database_url() == "postgresql://u:p@host/db"


def test_get_database_url_raises_when_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        get_database_url()
