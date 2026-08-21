"""Generates the synthetic FinGuard dataset and writes it to data/ as
Parquet (not committed - see .gitignore). Prints a summary for eyeballing.

Usage:
    python scripts/generate_data.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data_generation import generator  # noqa: E402

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    dataset = generator.generate_dataset()
    DATA_DIR.mkdir(exist_ok=True)

    for name in ["dim_location", "dim_customer", "dim_merchant", "dim_device", "fact_transactions", "ground_truth_fraud"]:
        path = DATA_DIR / f"{name}.parquet"
        dataset[name].to_parquet(path, index=False)
        print(f"wrote {path} ({len(dataset[name])} rows)")

    summary = generator.summarize(dataset)
    print(json.dumps(summary, indent=2, default=str))

    summary_path = DATA_DIR / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
