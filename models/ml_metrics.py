"""Metric helpers shared by both models' evaluation. Accuracy is
deliberately not implemented here - with a ~2% fraud rate, "predict
non-fraud always" scores ~98% accuracy while catching zero fraud, so it
would be actively misleading as a headline metric. PR-AUC is treated as
primary; ROC-AUC is reported as a secondary/reference metric only.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def precision_recall_f1_at_threshold(y_true, y_score, threshold: float) -> dict:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def recall_at_fixed_fpr(y_true, y_score, target_fpr: float = 0.01) -> dict:
    """Recall achievable at (at most) the given false-positive rate,
    using the ROC curve. Reports the actual FPR/threshold used, since the
    exact target_fpr may not be hit precisely on discrete data.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return {
        "target_fpr": target_fpr,
        "actual_fpr": float(fpr[idx]),
        "recall": float(tpr[idx]),
        "threshold": float(thresholds[idx]),
    }


def precision_recall_at_top_k(y_true, y_score, k_frac: float = 0.02) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    k = max(1, int(n * k_frac))
    top_k_idx = np.argsort(-y_score)[:k]
    caught = y_true[top_k_idx].sum()
    total_positive = y_true.sum()
    return {
        "k_frac": k_frac,
        "k": k,
        "precision": float(caught / k),
        "recall": float(caught / total_positive) if total_positive > 0 else 0.0,
    }


def confusion_matrix_at_threshold(y_true, y_score, threshold: float) -> dict:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"threshold": threshold, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def score_distribution_summary(y_true, y_score) -> dict:
    y_true = np.asarray(y_true)
    y_score = pd.Series(np.asarray(y_score))
    fraud_scores = y_score[y_true == 1]
    legit_scores = y_score[y_true == 0]
    return {
        "fraud": fraud_scores.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
        "legitimate": legit_scores.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
    }


def find_threshold_for_top_k(y_score, k_frac: float = 0.02) -> float:
    """Threshold that yields approximately the top k_frac of scores flagged."""
    y_score = np.asarray(y_score)
    k = max(1, int(len(y_score) * k_frac))
    sorted_scores = np.sort(y_score)[::-1]
    return float(sorted_scores[k - 1])


def full_evaluation(y_true, y_score, threshold: float, top_k_frac: float = 0.02, target_fpr: float = 0.01) -> dict:
    return {
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
        "at_threshold": precision_recall_f1_at_threshold(y_true, y_score, threshold),
        "recall_at_fixed_fpr": recall_at_fixed_fpr(y_true, y_score, target_fpr),
        "precision_recall_at_top_k": precision_recall_at_top_k(y_true, y_score, top_k_frac),
        "confusion_matrix": confusion_matrix_at_threshold(y_true, y_score, threshold),
        "score_distribution": score_distribution_summary(y_true, y_score),
    }
