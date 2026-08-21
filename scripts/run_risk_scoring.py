"""Phase 7: builds the combined risk score for every transaction, loads it
into risk_scores, assigns risk tiers, and runs the financial exposure /
threshold trade-off simulation.

Usage:
    python scripts/run_risk_scoring.py
"""
import json
import pathlib
import sys

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import NUMERIC

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from database.connection import get_engine  # noqa: E402
from risk_scoring.exposure import best_threshold_by_net_impact, threshold_sweep  # noqa: E402
from risk_scoring.pipeline import build_risk_scores  # noqa: E402

load_dotenv()

ARTIFACTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "artifacts"


def main() -> None:
    engine = get_engine()

    print("=" * 70)
    print("1. Building combined risk scores")
    print("=" * 70)
    risk_scores_df, diag = build_risk_scores(engine)
    print(f"exposure cap (95th pct of train amounts): ${diag['exposure_cap_usd']:.2f}")
    print(f"tier cutpoints (from validation quantiles): {diag['tier_cutpoints']}")
    print(f"train/val/test rows: {diag['n_train']}/{diag['n_val']}/{diag['n_test']}")
    print("\nrisk tier distribution:")
    print(risk_scores_df["risk_tier"].value_counts().to_string())

    print("\n" + "=" * 70)
    print("2. Loading risk_scores table")
    print("=" * 70)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE risk_scores"))
    risk_scores_df.to_sql(
        "risk_scores", engine, if_exists="append", index=False,
        method="multi", chunksize=5000,
        dtype={c: NUMERIC(6, 5) for c in ["ml_component", "rules_component", "behavioral_component", "exposure_component", "combined_score"]},
    )
    print(f"loaded {len(risk_scores_df)} rows into risk_scores")

    print("\n" + "=" * 70)
    print("3. Risk tier vs. fraud rate (sanity check - tiers should separate fraud)")
    print("=" * 70)
    all_df = diag["all_df"].merge(risk_scores_df[["transaction_id", "risk_tier"]], on="transaction_id")
    tier_fraud_rate = all_df.groupby("risk_tier")["is_fraud"].agg(["mean", "count"])
    print(tier_fraud_rate.to_string())

    print("\n" + "=" * 70)
    print("4. Financial exposure / threshold trade-off (validation only)")
    print("=" * 70)
    # diag["all_df"] is concatenated train+val+test in that order, so the
    # row ranges [0:n_train), [n_train:n_train+n_val), [n_train+n_val:) give
    # exact split membership without re-deriving it from ml_in_sample.
    val_ids = set(diag["all_df"].iloc[diag["n_train"]:diag["n_train"] + diag["n_val"]]["transaction_id"])
    test_ids = set(diag["all_df"].iloc[diag["n_train"] + diag["n_val"]:]["transaction_id"])
    val_scored = all_df[all_df["transaction_id"].isin(val_ids)].merge(
        risk_scores_df[["transaction_id", "combined_score"]], on="transaction_id"
    )
    test_scored = all_df[all_df["transaction_id"].isin(test_ids)].merge(
        risk_scores_df[["transaction_id", "combined_score"]], on="transaction_id"
    )

    val_sweep = threshold_sweep(val_scored)
    print(val_sweep.to_string(index=False))
    best_val = best_threshold_by_net_impact(val_sweep)
    print(f"\nbest validation threshold by net expected impact: {best_val['threshold']:.4f} "
          f"(net impact ${best_val['net_expected_impact_usd']:.2f}, alerts {best_val['n_alerts']}, "
          f"precision {best_val['precision']:.2%}, recall {best_val['recall']:.2%})")

    print("\n" + "=" * 70)
    print("5. Final unbiased look: applying the VALIDATION-selected threshold to TEST")
    print("=" * 70)
    from risk_scoring.exposure import portfolio_impact_at_threshold

    test_result = portfolio_impact_at_threshold(test_scored, best_val["threshold"])
    print(json.dumps(test_result, indent=2))

    print("\n" + "=" * 70)
    print("6. Saving artifacts")
    print("=" * 70)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    output = {
        "exposure_cap_usd": diag["exposure_cap_usd"],
        "tier_cutpoints": diag["tier_cutpoints"],
        "tier_fraud_rate": tier_fraud_rate.reset_index().to_dict(orient="records"),
        "validation_threshold_sweep": val_sweep.to_dict(orient="records"),
        "best_validation_threshold": best_val,
        "test_result_at_validation_threshold": test_result,
    }
    with open(ARTIFACTS_DIR / "phase7_risk_scoring.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"wrote {ARTIFACTS_DIR / 'phase7_risk_scoring.json'}")


if __name__ == "__main__":
    main()
