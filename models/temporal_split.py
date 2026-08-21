"""Chronological train/validation/test split - no shuffling, no k-fold.

Splits by row position on time-sorted data (not by calendar-day fraction)
so the three sets have exactly the target row-count proportions; the
resulting date ranges are a consequence of the data's actual density, not
an input. This is the standard approach for point-in-time-correct
evaluation: a fraud model that will be deployed forward in time should be
validated the same way it will actually be used - train on the past,
evaluate on data that came strictly after it.
"""
import pandas as pd


def chronological_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15, ts_col: str = "transaction_ts"):
    """Returns (train_df, val_df, test_df). df must already be sorted by
    ts_col (models.feature_matrix.build_feature_matrix guarantees this).
    """
    assert df[ts_col].is_monotonic_increasing, "input must be sorted by transaction_ts before splitting"

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, ts_col: str = "transaction_ts") -> dict:
    def _range(d):
        if len(d) == 0:
            return {"rows": 0, "start": None, "end": None}
        return {"rows": len(d), "start": str(d[ts_col].min()), "end": str(d[ts_col].max())}

    total = len(train_df) + len(val_df) + len(test_df)
    return {
        "train": _range(train_df),
        "validation": _range(val_df),
        "test": _range(test_df),
        "total_rows": total,
        "train_pct": round(len(train_df) / total, 4),
        "val_pct": round(len(val_df) / total, 4),
        "test_pct": round(len(test_df) / total, 4),
    }
