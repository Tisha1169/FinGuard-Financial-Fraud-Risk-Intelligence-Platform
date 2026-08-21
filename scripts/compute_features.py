"""Populates fact_customer_daily_metrics and fact_merchant_daily_metrics
from fact_transactions. Run after scripts/load_data.py.

Usage:
    python scripts/compute_features.py
"""
import pathlib
import sys
import time

from dotenv import load_dotenv
from sqlalchemy import text

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from database.connection import get_engine  # noqa: E402
from features.daily_metrics import compute_daily_metrics  # noqa: E402

load_dotenv()


def main() -> None:
    engine = get_engine()
    start = time.monotonic()
    compute_daily_metrics(engine)
    elapsed = time.monotonic() - start

    with engine.connect() as conn:
        n_customer_rows = conn.execute(text("SELECT COUNT(*) FROM fact_customer_daily_metrics")).scalar()
        n_merchant_rows = conn.execute(text("SELECT COUNT(*) FROM fact_merchant_daily_metrics")).scalar()

    print(f"fact_customer_daily_metrics: {n_customer_rows} rows")
    print(f"fact_merchant_daily_metrics: {n_merchant_rows} rows")
    print(f"computed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
