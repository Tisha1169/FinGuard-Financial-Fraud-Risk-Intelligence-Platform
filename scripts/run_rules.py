"""Backfills rules_triggered for every transaction using the vectorized
batch rule engine (rules/batch.py). Truncates first - safe because this is
entirely derived/synthetic data, re-run any time after new transactions or
feature rows are loaded.

Usage:
    python scripts/run_rules.py
"""
import json
import pathlib
import sys
import time

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from database.connection import get_engine  # noqa: E402
from rules.batch import apply_rules, build_batch_features  # noqa: E402

load_dotenv()


def main() -> None:
    engine = get_engine()
    start = time.monotonic()

    features_df = build_batch_features(engine)
    firings = apply_rules(features_df)
    firings["evidence"] = firings["evidence"].apply(json.dumps)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE rules_triggered RESTART IDENTITY"))

    firings.to_sql(
        "rules_triggered", engine, if_exists="append", index=False,
        method="multi", chunksize=5000, dtype={"evidence": JSONB},
    )

    elapsed = time.monotonic() - start
    print(f"scored {len(features_df)} transactions in {elapsed:.1f}s")
    print(f"rule firings: {len(firings)}")
    print(firings["rule_id"].value_counts().to_string())
    print(f"transactions with >=1 rule firing: {firings['transaction_id'].nunique()} "
          f"({firings['transaction_id'].nunique() / len(features_df):.2%})")


if __name__ == "__main__":
    main()
