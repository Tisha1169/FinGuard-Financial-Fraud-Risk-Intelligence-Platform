import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from risk_scoring.exposure import threshold_sweep
from streamlit_app import db, model_cache


def render():
    st.header("Model & Portfolio Monitoring")
    st.caption(
        "Model metrics are computed on the held-out TEST split only (never train/validation - "
        "see docs/evaluation_report.md). All results are evaluated on FinGuard's synthetic dataset; "
        "see docs/model_card.md's synthetic-data caveats before drawing real-world conclusions."
    )

    pipeline = model_cache.get_trained_pipeline()
    xgb_eval, logreg_eval = pipeline["xgb_eval"], pipeline["logreg_eval"]

    st.subheader("Test-set performance (XGBoost vs. Logistic Regression)")
    m = st.columns(6)
    m[0].metric("PR-AUC (XGB)", f"{xgb_eval['pr_auc']:.3f}", f"{xgb_eval['pr_auc']-logreg_eval['pr_auc']:+.3f} vs LogReg")
    m[1].metric("ROC-AUC (XGB)", f"{xgb_eval['roc_auc']:.3f}")
    m[2].metric("Precision @ thresh", f"{xgb_eval['at_threshold']['precision']:.2%}")
    m[3].metric("Recall @ thresh", f"{xgb_eval['at_threshold']['recall']:.2%}")
    m[4].metric("F1 @ thresh", f"{xgb_eval['at_threshold']['f1']:.3f}")
    m[5].metric("Recall @ 1% FPR", f"{xgb_eval['recall_at_fixed_fpr']['recall']:.2%}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score distribution: fraud vs. legitimate (test)")
        test_df = pipeline["test_df"]
        dist_df = pd.DataFrame({"score": pipeline["test_scores"], "is_fraud": pipeline["y_test"]})
        dist_df["label"] = dist_df["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
        fig = px.histogram(dist_df, x="score", color="label", barmode="overlay", nbins=40,
                            color_discrete_map={"Legitimate": "steelblue", "Fraud": "crimson"}, opacity=0.65)
        fig.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Global SHAP feature importance")
        top = pipeline["shap_top_features"]
        fig2 = go.Figure(go.Bar(x=top["mean_abs_shap"], y=top["feature"], orientation="h"))
        fig2.update_layout(height=350, margin=dict(t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, width="stretch")

    st.divider()
    st.subheader("Financial threshold trade-off (validation-derived, current risk_scores)")
    with db.engine().connect() as conn:
        scored = pd.read_sql(
            text(
                "SELECT r.transaction_id, r.combined_score, g.is_fraud, t.amount "
                "FROM risk_scores r JOIN ground_truth_fraud g ON g.transaction_id = r.transaction_id "
                "JOIN fact_transactions t ON t.transaction_id = r.transaction_id"
            ),
            conn,
        )
    sweep = threshold_sweep(scored, n_points=15)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["net_expected_impact_usd"], name="Net expected impact ($)", mode="lines+markers"))
    fig3.update_layout(height=320, margin=dict(t=10, b=10), xaxis_title="Combined score threshold", yaxis_title="Net expected impact (USD)")
    st.plotly_chart(fig3, width="stretch")
    st.caption(
        "Simulated financial impact only (see docs/risk_scoring.md) - not a real recovery/loss model. "
        "Computed live across the full dataset, so this may differ slightly from the validation-only sweep reported in docs/risk_scoring.md."
    )

    st.divider()
    st.subheader("Segment performance (XGBoost, test set)")
    seg_col = st.selectbox("Segment by", ["channel", "customer_risk_segment", "merchant_risk_category"])
    test_df_seg = pipeline["test_df"].copy()
    test_df_seg["score"] = pipeline["test_scores"]
    test_df_seg["y_true"] = pipeline["y_test"]
    threshold = pipeline["threshold"]
    test_df_seg["predicted_fraud"] = (test_df_seg["score"] >= threshold).astype(int)

    seg = test_df_seg.groupby(seg_col).apply(
        lambda g: pd.Series({
            "n": len(g),
            "fraud_count": g["y_true"].sum(),
            "recall": (g[g.y_true == 1]["predicted_fraud"].mean() if (g.y_true == 1).any() else None),
            "precision": (g[g.predicted_fraud == 1]["y_true"].mean() if (g.predicted_fraud == 1).any() else None),
        }),
        include_groups=False,
    ).reset_index()
    st.dataframe(seg, width="stretch", hide_index=True)

    st.divider()
    st.subheader("XGBoost hyperparameter selection (validation PR-AUC)")
    st.dataframe(pd.DataFrame(pipeline["selection_report"]["trials"]), width="stretch", hide_index=True)

    st.info(
        "Drift/PSI monitoring across time periods is a planned future enhancement (Phase 12 in the original "
        "roadmap) - not yet implemented. This tab reflects current-state model performance only.",
        icon="ℹ️",
    )
