"""Applies sql/schema.sql to the database pointed at by DATABASE_URL.

Usage:
    python scripts/init_db.py
"""
import pathlib
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from database.connection import get_engine  # noqa: E402

load_dotenv()

SCHEMA_PATH = pathlib.Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


def main() -> None:
    # Executed as a single multi-statement script (not split on ";") because
    # naively splitting breaks on semicolons that appear inside comments.
    schema_sql = SCHEMA_PATH.read_text()
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(schema_sql)
    print(f"Schema applied from {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
