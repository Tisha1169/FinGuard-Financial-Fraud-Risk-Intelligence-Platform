"""Loads the Parquet files produced by generate_data.py into Postgres,
truncating existing rows first (dev/reset workflow - safe because this is
entirely synthetic data, never run against a database with real records).

Usage:
    python scripts/generate_data.py   # writes data/*.parquet
    python scripts/load_data.py       # loads them into DATABASE_URL
"""
import pathlib
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from database.connection import get_engine  # noqa: E402

load_dotenv()

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# Load order matters: dimensions before facts, facts before ground truth.
LOAD_ORDER = [
    ("dim_location", "dim_location"),
    ("dim_customer", "dim_customer"),
    ("dim_merchant", "dim_merchant"),
    ("dim_device", "dim_device"),
    ("fact_transactions", "fact_transactions"),
    ("ground_truth_fraud", "ground_truth_fraud"),
]

# Truncate in reverse dependency order.
TRUNCATE_ORDER = [t for _, t in reversed(LOAD_ORDER)]


def main() -> None:
    engine = get_engine()

    with engine.begin() as conn:
        for table in TRUNCATE_ORDER:
            conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
        print(f"truncated: {', '.join(TRUNCATE_ORDER)}")

    for file_stem, table in LOAD_ORDER:
        path = DATA_DIR / f"{file_stem}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run scripts/generate_data.py first")
        df = pd.read_parquet(path)
        df.to_sql(table, engine, if_exists="append", index=False, method="multi", chunksize=5000)
        print(f"loaded {len(df)} rows into {table}")


if __name__ == "__main__":
    main()
