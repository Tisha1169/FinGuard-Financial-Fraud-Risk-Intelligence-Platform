"""Tests for models/ml_explain.py - the SHAP explainability layer.

The core thing worth verifying isn't styling, it's correctness: SHAP
values must satisfy the additivity property (base_value + sum(shap_values)
== the model's actual raw prediction for that row) - if that didn't hold,
"explanations" would be decorative numbers, not real attributions of the
model's own output. Uses a small XGBoost model fit on synthetic data with
a clear feature-importance ground truth (no DB needed - fast and
self-contained).
"""
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from models.ml_explain import compute_shap_values, explain_single_prediction, global_importance


@pytest.fixture(scope="module")
def toy_model_and_data():
    rng = np.random.RandomState(0)
    n = 500
    # feature_dominant drives the label almost entirely; feature_noise is pure noise -
    # a correct SHAP implementation should rank the dominant feature far above the noise one.
    feature_dominant = rng.uniform(-1, 1, n)
    feature_noise = rng.uniform(-1, 1, n)
    y = (feature_dominant > 0).astype(int)

    X = pd.DataFrame({"feature_dominant": feature_dominant, "feature_noise": feature_noise})
    model = XGBClassifier(max_depth=3, n_estimators=50, random_state=0)
    model.fit(X, y)
    return model, X.values, list(X.columns)


def test_shap_values_have_expected_shape(toy_model_and_data):
    model, X, feature_names = toy_model_and_data
    explainer, shap_values, X_sample_df, idx = compute_shap_values(model, X, feature_names, sample_size=200)
    assert shap_values.shape == (200, len(feature_names))
    assert list(X_sample_df.columns) == feature_names


def test_shap_additivity_matches_actual_model_output(toy_model_and_data):
    """The defining correctness property of SHAP: base_value + sum(shap
    values for a row) must equal the model's own raw margin prediction for
    that exact row - proving the "explanation" is derived from the real
    model, not fabricated or approximated.
    """
    model, X, feature_names = toy_model_and_data
    explainer, shap_values, X_sample_df, idx = compute_shap_values(model, X, feature_names, sample_size=100)
    base_value = explainer.expected_value

    raw_margins = model.predict(X_sample_df, output_margin=True)
    reconstructed = base_value + shap_values.sum(axis=1)
    assert reconstructed == pytest.approx(raw_margins, abs=1e-3)


def test_global_importance_ranks_dominant_feature_first(toy_model_and_data):
    model, X, feature_names = toy_model_and_data
    _, shap_values, _, _ = compute_shap_values(model, X, feature_names, sample_size=300)
    importance = global_importance(shap_values, feature_names, top_n=2)
    assert importance.iloc[0]["feature"] == "feature_dominant"
    assert importance.iloc[0]["mean_abs_shap"] > importance.iloc[1]["mean_abs_shap"]


def test_global_importance_respects_top_n(toy_model_and_data):
    model, X, feature_names = toy_model_and_data
    _, shap_values, _, _ = compute_shap_values(model, X, feature_names, sample_size=100)
    importance = global_importance(shap_values, feature_names, top_n=1)
    assert len(importance) == 1


def test_explain_single_prediction_margin_matches_additivity(toy_model_and_data):
    model, X, feature_names = toy_model_and_data
    explainer, shap_values, X_sample_df, idx = compute_shap_values(model, X, feature_names, sample_size=100)
    base_value = explainer.expected_value

    explanation = explain_single_prediction(shap_values, X_sample_df, row_idx=5, base_value=base_value, top_n=2)
    raw_margin = model.predict(X_sample_df.iloc[[5]], output_margin=True)[0]
    assert explanation["predicted_margin"] == pytest.approx(raw_margin, abs=1e-3)


def test_explain_single_prediction_top_contributors_use_real_feature_values(toy_model_and_data):
    """Each returned contributor's `value` must be the row's ACTUAL feature
    value from the input data, not a placeholder - directly checks against
    the fabrication risk the phase requirements called out.
    """
    model, X, feature_names = toy_model_and_data
    explainer, shap_values, X_sample_df, idx = compute_shap_values(model, X, feature_names, sample_size=50)
    base_value = explainer.expected_value

    row_idx = 3
    explanation = explain_single_prediction(shap_values, X_sample_df, row_idx=row_idx, base_value=base_value, top_n=2)
    actual_row = X_sample_df.iloc[row_idx]
    for contributor in explanation["top_contributors"]:
        assert contributor["value"] == pytest.approx(actual_row[contributor["feature"]])


def test_explain_single_prediction_respects_top_n(toy_model_and_data):
    model, X, feature_names = toy_model_and_data
    explainer, shap_values, X_sample_df, idx = compute_shap_values(model, X, feature_names, sample_size=50)
    explanation = explain_single_prediction(shap_values, X_sample_df, row_idx=0, base_value=explainer.expected_value, top_n=1)
    assert len(explanation["top_contributors"]) == 1
