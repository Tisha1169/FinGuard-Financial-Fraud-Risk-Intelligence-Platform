"""Model training: Logistic Regression (interpretable baseline) and
XGBoost (primary model). Both use class-weighting for imbalance rather
than SMOTE - see docs/model_card.md "imbalance handling" for the full
reasoning; in short, SMOTE interpolates synthetic minority points in
feature space, which (a) is redundant here since the minority class is
already synthetically constructed rather than naturally scarce, and (b)
risks manufacturing points that don't correspond to any coherent fraud
typology when interpolated across dissimilar fraud events (e.g.
interpolating between a CARD_TESTING point and an ACCOUNT_TAKEOVER point
produces a feature vector that describes neither). Class weights instead
directly reweight the existing, real (if synthetic) minority examples
without inventing new ones.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from models.ml_metrics import pr_auc

RANDOM_STATE = 42


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def compute_scale_pos_weight(y_train) -> float:
    y_train = np.asarray(y_train)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


# A small, deliberately conservative grid - a handful of reasonable
# configurations, not an automated search over dozens of combinations.
# Selected by PR-AUC on the VALIDATION set only; the test set is never
# touched during this selection (see scripts/train_models.py).
XGB_CANDIDATE_PARAMS = [
    {"max_depth": 3, "n_estimators": 150, "learning_rate": 0.1},
    {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.1},
    {"max_depth": 5, "n_estimators": 300, "learning_rate": 0.05},
]


def train_xgboost_with_validation_selection(X_train, y_train, X_val, y_val) -> tuple[XGBClassifier, dict]:
    scale_pos_weight = compute_scale_pos_weight(y_train)

    best_model, best_params, best_val_pr_auc = None, None, -1.0
    trial_results = []

    for params in XGB_CANDIDATE_PARAMS:
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="aucpr",
        )
        model.fit(X_train, y_train)
        val_score = pr_auc(y_val, model.predict_proba(X_val)[:, 1])
        trial_results.append({**params, "val_pr_auc": val_score})

        if val_score > best_val_pr_auc:
            best_model, best_params, best_val_pr_auc = model, params, val_score

    selection_report = {
        "scale_pos_weight": scale_pos_weight,
        "trials": trial_results,
        "selected_params": best_params,
        "selected_val_pr_auc": best_val_pr_auc,
    }
    return best_model, selection_report
