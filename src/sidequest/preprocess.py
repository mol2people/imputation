"""Side-quest preprocessing: training-fitted, per (source, variant).

Order (plan section 5):
  1. drop all-missing raw columns (training data only; names recorded)
  2. numeric columns: median-impute with training medians; add a binary
     missingness indicator for every numeric column that has at least one
     missing training value
  3. categorical demographics (``demo__*``): most-frequent-value imputation
     (training mode) and one-hot encoding over fixed training levels with
     unknown-level handling (unknown -> all-zero row)
  4. drop zero-variance transformed terms (training data only; names recorded)

No scaling.  The fitted object records every training-derived decision so the
verifier can check the drop lists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SQPreprocessor(BaseEstimator, TransformerMixin):
    CAT_PREFIX = "demo__"

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()
        self.input_columns_ = list(X.columns)
        self.all_missing_ = [c for c in X.columns if X[c].isna().all()]
        Xd = X.drop(columns=self.all_missing_)
        self.cat_cols_ = [c for c in Xd.columns if c.startswith(self.CAT_PREFIX)]
        self.num_cols_ = [c for c in Xd.columns if c not in self.cat_cols_]
        self.medians_ = {c: float(Xd[c].median()) for c in self.num_cols_}
        self.missing_cols_ = [c for c in self.num_cols_ if Xd[c].isna().any()]
        self.modes_ = {c: str(Xd[c].dropna().mode().iloc[0])
                       if Xd[c].notna().any() else "" for c in self.cat_cols_}
        self.levels_ = {c: sorted(Xd[c].dropna().astype(str).unique().tolist())
                        for c in self.cat_cols_}

        Xt = self._core(Xd)
        self.zero_variance_ = [c for c in Xt.columns
                               if Xt[c].nunique(dropna=False) <= 1]
        self.feature_names_out_ = [c for c in Xt.columns
                                   if c not in self.zero_variance_]
        return self

    def _core(self, Xd: pd.DataFrame) -> pd.DataFrame:
        parts = {c: Xd[c].astype(float) for c in self.num_cols_}
        for c in self.missing_cols_:
            parts[f"{c}__missing"] = Xd[c].isna().astype(float)
        for c in self.cat_cols_:
            vals = Xd[c].astype("object").where(Xd[c].notna(), self.modes_[c])
            for lv in self.levels_[c]:
                parts[f"{c}={lv}"] = (vals == lv).astype(float)
        return pd.DataFrame(parts, index=Xd.index).fillna(
            {c: self.medians_[c] for c in self.num_cols_})

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        missing = [c for c in self.all_missing_ if c not in X.columns]
        if missing:
            raise KeyError(f"transform input missing columns: {missing}")
        Xt = self._core(X.drop(columns=self.all_missing_))
        return Xt[self.feature_names_out_].to_numpy(dtype=np.float64)

    def get_feature_names_out(self) -> list[str]:
        return list(self.feature_names_out_)
