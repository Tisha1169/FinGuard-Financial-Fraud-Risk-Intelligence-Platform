"""Preprocessing (imputation, categorical encoding, scaling) - fit on the
training split ONLY, then applied unchanged to validation and test. This
is the same rule enforced everywhere else in this phase: nothing learned
from data is allowed to see anything past the training period's end.

Two output feature matrices are produced from the same fitted artifacts:
- "tree" matrix (imputed + one-hot encoded, unscaled) for XGBoost, which
  is scale-invariant.
- "scaled" matrix (additionally standardized) for Logistic Regression,
  which needs comparable feature magnitudes for its coefficients and
  regularization to behave sensibly.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.feature_matrix import BOOLEAN_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS


class FittedPreprocessor:
    def __init__(self, numeric_cols: list[str] | None = None):
        # numeric_cols defaults to the static list, but the training
        # pipeline passes NUMERIC_FEATURE_COLUMNS + ["isolation_forest_score"]
        # since that column is computed after the split and isn't part of
        # feature_matrix.py's static list.
        self.numeric_cols = numeric_cols if numeric_cols is not None else list(NUMERIC_FEATURE_COLUMNS)
        self.imputer = SimpleImputer(strategy="median")
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FittedPreprocessor":
        self.imputer.fit(df[self.numeric_cols])
        self.encoder.fit(df[CATEGORICAL_FEATURE_COLUMNS])

        numeric_imputed = self.imputer.transform(df[self.numeric_cols])
        boolean_matrix = df[BOOLEAN_FEATURE_COLUMNS].astype(int).values
        unscaled = np.hstack([numeric_imputed, boolean_matrix])
        self.scaler.fit(unscaled)

        cat_names = list(self.encoder.get_feature_names_out(CATEGORICAL_FEATURE_COLUMNS))
        self.feature_names_ = self.numeric_cols + BOOLEAN_FEATURE_COLUMNS + cat_names
        return self

    def _base_matrix(self, df: pd.DataFrame) -> np.ndarray:
        numeric_imputed = self.imputer.transform(df[self.numeric_cols])
        boolean_matrix = df[BOOLEAN_FEATURE_COLUMNS].astype(int).values
        cat_encoded = self.encoder.transform(df[CATEGORICAL_FEATURE_COLUMNS])
        return np.hstack([numeric_imputed, boolean_matrix, cat_encoded])

    def transform_for_trees(self, df: pd.DataFrame) -> np.ndarray:
        """Unscaled - suitable for XGBoost."""
        return self._base_matrix(df)

    def transform_for_linear(self, df: pd.DataFrame) -> np.ndarray:
        """Scaled - suitable for Logistic Regression. Only the numeric +
        boolean block is standardized (matching what the scaler was fit
        on); one-hot columns are left as 0/1.
        """
        numeric_imputed = self.imputer.transform(df[self.numeric_cols])
        boolean_matrix = df[BOOLEAN_FEATURE_COLUMNS].astype(int).values
        unscaled = np.hstack([numeric_imputed, boolean_matrix])
        scaled = self.scaler.transform(unscaled)
        cat_encoded = self.encoder.transform(df[CATEGORICAL_FEATURE_COLUMNS])
        return np.hstack([scaled, cat_encoded])
