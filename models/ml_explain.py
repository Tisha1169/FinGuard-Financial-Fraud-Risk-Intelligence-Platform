"""SHAP explainability for the XGBoost model. Global importance ranks
features by mean |SHAP value| across a sample of transactions; individual
explanations attribute one prediction's score to its actual contributing
features - both computed directly from the fitted model and real feature
values, never fabricated or hand-written.
"""
import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: np.ndarray, feature_names: list[str], sample_size: int = 2000, random_state: int = 42):
    """Returns (explainer, shap_values, X_sample_df) for a random sample of
    X (SHAP on tree models is exact but scales with dataset size; a sample
    keeps this fast without changing which features matter).
    """
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    X_sample = X[idx]
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample_df)
    return explainer, shap_values, X_sample_df, idx


def global_importance(shap_values: np.ndarray, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def explain_single_prediction(shap_values: np.ndarray, X_sample_df: pd.DataFrame, row_idx: int, base_value: float, top_n: int = 6) -> dict:
    """Human-readable breakdown of one prediction: its top contributing
    features (by |SHAP value|), each with its actual feature value and
    signed contribution.
    """
    row_shap = shap_values[row_idx]
    row_values = X_sample_df.iloc[row_idx]

    contributions = pd.DataFrame({
        "feature": X_sample_df.columns,
        "value": row_values.values,
        "shap_value": row_shap,
    })
    contributions["abs_shap"] = contributions["shap_value"].abs()
    top = contributions.sort_values("abs_shap", ascending=False).head(top_n)

    return {
        "base_value": float(base_value),
        "predicted_margin": float(base_value + row_shap.sum()),
        "top_contributors": [
            {"feature": r.feature, "value": float(r.value), "shap_value": round(float(r.shap_value), 4)}
            for r in top.itertuples()
        ],
    }
