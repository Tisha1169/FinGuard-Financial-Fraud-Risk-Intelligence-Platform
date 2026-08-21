"""Phase 6 end-to-end training pipeline: feature assembly -> chronological
split -> train-only-fit preprocessing (imputation, encoding, scaling,
Isolation Forest) -> Logistic Regression baseline -> XGBoost primary model
-> validation-based threshold selection -> final unbiased test evaluation
-> SHAP explainability -> error analysis -> artifacts + metrics JSON.

Every "fit" happens on the training split only; validation is used for
model/threshold selection; the test split is touched exactly once, at the
very end, for final reporting. See docs/model_card.md and
docs/evaluation_report.md for the full writeup this script's output feeds.

Usage:
    python scripts/train_models.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from database.connection import get_engine  # noqa: E402
from models.feature_matrix import BOOLEAN_FEATURE_COLUMNS, FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS, build_feature_matrix  # noqa: E402
from models.ml_explain import compute_shap_values, explain_single_prediction, global_importance  # noqa: E402
from models.ml_isolation_forest import fit_isolation_forest_train_only, score_with_fitted_forest  # noqa: E402
from models.ml_metrics import find_threshold_for_top_k, full_evaluation  # noqa: E402
from models.ml_models import train_logistic_regression, train_xgboost_with_validation_selection  # noqa: E402
from models.ml_preprocessing import FittedPreprocessor  # noqa: E402
from models.temporal_split import chronological_split, split_summary  # noqa: E402

load_dotenv()

ARTIFACTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "artifacts"


def error_analysis(df: pd.DataFrame, y_true, y_score, threshold: float, segment_cols: list[str]) -> dict:
    d = df.copy()
    d["y_true"] = np.asarray(y_true)
    d["y_pred"] = (np.asarray(y_score) >= threshold).astype(int)
    d["outcome"] = np.select(
        [
            (d.y_true == 1) & (d.y_pred == 1),
            (d.y_true == 1) & (d.y_pred == 0),
            (d.y_true == 0) & (d.y_pred == 1),
            (d.y_true == 0) & (d.y_pred == 0),
        ],
        ["TP", "FN", "FP", "TN"],
        default="UNKNOWN",
    )

    report = {"overall_counts": d["outcome"].value_counts().to_dict()}
    for col in segment_cols:
        if col not in d.columns:
            continue
        seg = d.groupby(col)["outcome"].value_counts().unstack(fill_value=0)
        for c in ["TP", "FN", "FP", "TN"]:
            if c not in seg.columns:
                seg[c] = 0
        seg["recall"] = seg["TP"] / (seg["TP"] + seg["FN"]).replace(0, np.nan)
        seg["fp_rate_of_flagged"] = seg["FP"] / (seg["FP"] + seg["TP"]).replace(0, np.nan)
        report[col] = seg.to_dict(orient="index")
    return report


def main() -> None:
    engine = get_engine()

    print("=" * 70)
    print("1. Building feature matrix")
    print("=" * 70)
    full_df = build_feature_matrix(engine)
    print(f"total transactions: {len(full_df)}, fraud: {int(full_df['is_fraud'].sum())} "
          f"({full_df['is_fraud'].mean():.2%})")

    print("\n" + "=" * 70)
    print("2. Chronological split (70/15/15, no shuffling)")
    print("=" * 70)
    train_df, val_df, test_df = chronological_split(full_df)
    summary = split_summary(train_df, val_df, test_df)
    print(json.dumps(summary, indent=2))
    for split_name, d in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        rate = d["is_fraud"].mean()
        print(f"  {split_name}: fraud rate {rate:.2%} ({int(d['is_fraud'].sum())} / {len(d)})")

    print("\n" + "=" * 70)
    print("3. Isolation Forest - fit on TRAIN ONLY, transform val/test")
    print("=" * 70)
    iforest_artifacts, train_scores = fit_isolation_forest_train_only(train_df)
    print(f"fit on {iforest_artifacts['n_rows_fit']} training rows (== len(train_df): {iforest_artifacts['n_rows_fit'] == len(train_df)})")
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["isolation_forest_score"] = train_scores.values
    val_df["isolation_forest_score"] = score_with_fitted_forest(iforest_artifacts, val_df).values
    test_df["isolation_forest_score"] = score_with_fitted_forest(iforest_artifacts, test_df).values

    print("\n" + "=" * 70)
    print("4. Preprocessing - fit on TRAIN ONLY (imputation, encoding, scaling)")
    print("=" * 70)
    all_numeric_cols = NUMERIC_FEATURE_COLUMNS + ["isolation_forest_score"]
    preprocessor = FittedPreprocessor(numeric_cols=all_numeric_cols).fit(train_df)
    print(f"feature count after encoding: {len(preprocessor.feature_names_)}")

    X_train_tree = preprocessor.transform_for_trees(train_df)
    X_val_tree = preprocessor.transform_for_trees(val_df)
    X_test_tree = preprocessor.transform_for_trees(test_df)
    X_train_lin = preprocessor.transform_for_linear(train_df)
    X_val_lin = preprocessor.transform_for_linear(val_df)
    X_test_lin = preprocessor.transform_for_linear(test_df)
    y_train, y_val, y_test = train_df["is_fraud"].astype(int), val_df["is_fraud"].astype(int), test_df["is_fraud"].astype(int)

    print("\n" + "=" * 70)
    print("5. Logistic Regression baseline (class_weight='balanced')")
    print("=" * 70)
    logreg = train_logistic_regression(X_train_lin, y_train)
    logreg_val_scores = logreg.predict_proba(X_val_lin)[:, 1]
    logreg_test_scores = logreg.predict_proba(X_test_lin)[:, 1]

    print("\n" + "=" * 70)
    print("6. XGBoost primary model (validation-based conservative tuning)")
    print("=" * 70)
    xgb_model, selection_report = train_xgboost_with_validation_selection(X_train_tree, y_train, X_val_tree, y_val)
    print(json.dumps(selection_report, indent=2))
    xgb_val_scores = xgb_model.predict_proba(X_val_tree)[:, 1]
    xgb_test_scores = xgb_model.predict_proba(X_test_tree)[:, 1]

    print("\n" + "=" * 70)
    print("7. Threshold selection on VALIDATION only (top-2% alert volume)")
    print("=" * 70)
    logreg_threshold = find_threshold_for_top_k(logreg_val_scores, k_frac=0.02)
    xgb_threshold = find_threshold_for_top_k(xgb_val_scores, k_frac=0.02)
    print(f"logreg threshold (from val): {logreg_threshold:.4f}")
    print(f"xgboost threshold (from val): {xgb_threshold:.4f}")

    print("\n" + "=" * 70)
    print("8. Final evaluation - VALIDATION and TEST (test touched once, here)")
    print("=" * 70)
    results = {
        "logreg_validation": full_evaluation(y_val, logreg_val_scores, logreg_threshold),
        "logreg_test": full_evaluation(y_test, logreg_test_scores, logreg_threshold),
        "xgboost_validation": full_evaluation(y_val, xgb_val_scores, xgb_threshold),
        "xgboost_test": full_evaluation(y_test, xgb_test_scores, xgb_threshold),
    }
    for name, res in results.items():
        print(f"\n--- {name} ---")
        print(f"  PR-AUC: {res['pr_auc']:.4f}  ROC-AUC: {res['roc_auc']:.4f}")
        print(f"  @threshold: {res['at_threshold']}")
        print(f"  recall@fixed_fpr: {res['recall_at_fixed_fpr']}")
        print(f"  precision/recall@top2%: {res['precision_recall_at_top_k']}")
        print(f"  confusion matrix: {res['confusion_matrix']}")

    print("\n" + "=" * 70)
    print("9. SHAP explainability (XGBoost, test set sample)")
    print("=" * 70)
    explainer, shap_values, X_sample_df, sample_idx = compute_shap_values(
        xgb_model, X_test_tree, preprocessor.feature_names_, sample_size=2000,
    )
    top_features = global_importance(shap_values, preprocessor.feature_names_, top_n=15)
    print(top_features.to_string(index=False))

    base_value = explainer.expected_value
    y_test_sample = y_test.values[sample_idx]
    xgb_test_scores_sample = xgb_test_scores[sample_idx]
    y_pred_sample = (xgb_test_scores_sample >= xgb_threshold).astype(int)

    examples = {}
    tp_candidates = np.where((y_test_sample == 1) & (y_pred_sample == 1))[0]
    fp_candidates = np.where((y_test_sample == 0) & (y_pred_sample == 1))[0]
    fn_candidates = np.where((y_test_sample == 1) & (y_pred_sample == 0))[0]
    for label, candidates in [("true_positive_example", tp_candidates), ("false_positive_example", fp_candidates), ("false_negative_example", fn_candidates)]:
        if len(candidates) > 0:
            examples[label] = explain_single_prediction(shap_values, X_sample_df, int(candidates[0]), base_value)
    print(json.dumps(examples, indent=2))

    print("\n" + "=" * 70)
    print("10. Error analysis (test set, XGBoost @ threshold)")
    print("=" * 70)
    err_report = error_analysis(
        test_df, y_test, xgb_test_scores, xgb_threshold,
        segment_cols=["channel", "customer_risk_segment", "merchant_risk_category", "fraud_typology"],
    )
    print(json.dumps(err_report, indent=2, default=str))

    print("\n" + "=" * 70)
    print("11. Saving artifacts")
    print("=" * 70)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    xgb_model.save_model(str(ARTIFACTS_DIR / "xgboost_model.json"))

    metrics_output = {
        "split_summary": summary,
        "xgb_selection_report": selection_report,
        "logreg_threshold": logreg_threshold,
        "xgb_threshold": xgb_threshold,
        "results": results,
        "shap_top_features": top_features.to_dict(orient="records"),
        "shap_examples": examples,
        "error_analysis": err_report,
        "feature_count": len(preprocessor.feature_names_),
        "feature_names": preprocessor.feature_names_,
    }
    with open(ARTIFACTS_DIR / "phase6_metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2, default=str)
    print(f"wrote {ARTIFACTS_DIR / 'phase6_metrics.json'}")


if __name__ == "__main__":
    main()
