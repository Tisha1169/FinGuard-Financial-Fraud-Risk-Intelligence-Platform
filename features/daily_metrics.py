"""Runs sql/feature_engineering.sql to (re)populate the daily rollup
tables from fact_transactions. Idempotent - safe to re-run after new
transactions are loaded.
"""
import pathlib

from sqlalchemy.engine import Engine

SQL_PATH = pathlib.Path(__file__).resolve().parent.parent / "sql" / "feature_engineering.sql"


def compute_daily_metrics(engine: Engine) -> None:
    sql = SQL_PATH.read_text()
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)
