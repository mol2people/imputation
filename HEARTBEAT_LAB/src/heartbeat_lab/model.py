"""B40 summaries + pooled reference MultiRocket + ridge (alpha=1000).

This follows the strongest full-history arm, with a new split and train-only
fits. It is not a bit-for-bit replay or a claim to reproduce its AUROC.
"""
import os
import sys
from pathlib import Path
from threading import Lock

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import numpy as np
from numba import njit
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import STATS

# numba workqueue cannot run concurrent parallel transforms from app sessions.
_REFERENCE_LOCK = Lock()


def reference_functions():
    """Import the pinned author implementation prepared by 00_reference.py."""
    default = Path(__file__).resolve().parents[2] / "vendor" / "MultiRocket"
    root = Path(os.environ.get("HEARTBEAT_MULTIROCKET", default))
    if not (root / "multirocket" / "multirocket.py").exists():
        raise RuntimeError("Run scripts/00_reference.py first (see README).")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from multirocket.multirocket import fit, transform
    return fit, transform


@njit(cache=True)
def seed_numba(seed):
    # MultiRocket draws biases using numba's RNG, not numpy's Python RNG.
    np.random.seed(seed)


def pooled_features(days, hr, transformed=None):
    """305 static + 18,816 pooled MR columns at the default kernel budget."""
    weekend = days.date.dt.dayofweek.ge(5)
    contrast = days.loc[~weekend, "mean"].mean() - days.loc[weekend, "mean"].mean()
    static = np.r_[days[STATS].mean(), days[STATS].std(ddof=1), contrast,
                   np.isfinite(hr).mean(axis=0)]
    if transformed is None:
        return static.astype(np.float32)
    # A single day has undefined between-day SD. Use the original pooling's
    # zero-SD convention for MR; static NaNs use TRAINING medians downstream.
    sd = transformed.std(axis=0, ddof=1, dtype=np.float64) if len(days) > 1 else np.zeros(transformed.shape[1])
    return np.r_[static, transformed.mean(axis=0, dtype=np.float64), sd].astype(np.float32)


class HeartbeatModel:
    """Fit on a list of (day_table, HR_matrix) pairs, one pair per customer."""

    def __init__(self, alpha=1000.0, seed=20260930, num_features=1250,
                 representation="multirocket", filling="clock"):
        if representation not in {"summary", "multirocket"}:
            raise ValueError("representation must be summary or multirocket")
        if filling not in {"clock", "global"}:
            raise ValueError("filling must be clock or global")
        self.alpha, self.seed, self.num_features = alpha, seed, num_features
        self.representation, self.filling = representation, filling

    def fit(self, customers, y):
        if len(customers) != len(y) or set(y) != {0, 1}:
            raise ValueError("Need one 0/1 label per training customer, with both classes.")
        if any(len(days) < 2 or hr.shape != (len(days), 288) for days, hr in customers):
            raise ValueError("Training needs at least two adequate days per customer and 288 bins/day.")
        if self.representation == "multirocket":
            self._fit_transform(customers)
        features = np.stack([self.features(days, hr) for days, hr in customers])
        # Independent per-column transformations are equivalent to applying
        # these operations separately to the static and MultiRocket blocks.
        self.regression_ = make_pipeline(SimpleImputer(strategy="median"),
                                         VarianceThreshold(), StandardScaler(), Ridge(alpha=self.alpha))
        self.regression_.fit(features, y)
        self.training_days_ = {"min": min(len(d) for d, _ in customers),
                               "max": max(len(d) for d, _ in customers)}
        return self

    def _fit_transform(self, customers):
        fit, _ = reference_functions()
        # One clock bin at a time bounds temporary memory for long histories.
        overall = np.concatenate([hr[np.isfinite(hr)] for _, hr in customers])
        fallback = np.median(overall)
        del overall
        self.fill_ = np.full(288, fallback, dtype=np.float32)
        for b in range(288) if self.filling == "clock" else []:
            values = np.concatenate([hr[:, b][np.isfinite(hr[:, b])] for _, hr in customers])
            if values.size:
                self.fill_[b] = np.median(values)
        rng = np.random.default_rng(self.seed)
        selected = [hr[rng.choice(len(hr), min(4, len(hr)), replace=False)] for _, hr in customers]
        x = np.concatenate(selected)
        x = np.where(np.isfinite(x), x, self.fill_).astype(np.float64)
        with _REFERENCE_LOCK:
            seed_numba(self.seed)
            self.base_ = fit(x, num_features=self.num_features, max_dilations_per_kernel=32)
            seed_numba(self.seed + 1)
            self.diff_ = fit(np.diff(x, axis=1), num_features=self.num_features, max_dilations_per_kernel=32)

    def transform_days(self, hr):
        if self.representation == "summary":
            return None
        _, transform = reference_functions()
        x = np.where(np.isfinite(hr), hr, self.fill_).astype(np.float64)
        # Chunk days so transform work arrays remain modest for long uploads.
        chunks = []
        for start in range(0, len(x), 64):
            batch = x[start:start+64]
            with _REFERENCE_LOCK:
                f = transform(batch, np.diff(batch, axis=1), self.base_, self.diff_, n_features_per_kernel=4)
            chunks.append(np.nan_to_num(f).astype(np.float32))
        return np.concatenate(chunks)

    def features(self, days, hr):
        return pooled_features(days, hr, self.transform_days(hr))

    def predict(self, customers):
        return self.regression_.predict(np.stack([self.features(d, h) for d, h in customers]))

    def timeline(self, days, hr, transformed=None):
        """Daily and cumulative scores, never interpreting a score as probability.

        Cumulative row t uses only rows <= t. Daily singleton summaries are
        outside the training history distribution and are exploratory.
        """
        if len(days) != len(hr) or not days.date.is_monotonic_increasing:
            raise ValueError("Days and bins must align in chronological order.")
        f = self.transform_days(hr) if transformed is None else transformed
        n = len(days)
        observed = np.isfinite(hr)
        stats = days[STATS]
        weekend = days.date.dt.dayofweek.ge(5)
        contrast = (days["mean"].where(~weekend).expanding().mean()
                    - days["mean"].where(weekend).expanding().mean())
        daily = np.column_stack([stats, np.full((n, 9), np.nan), observed])
        cumulative = np.column_stack([
            stats.expanding().mean(), stats.expanding().std(ddof=1), contrast,
            observed.cumsum(axis=0) / np.arange(1, n+1)[:, None]])
        if f is not None:
            # Welford's running moments: each prefix uses only past/current
            # days, without repeatedly scanning years of transformed history.
            means, sds = np.empty_like(f), np.zeros_like(f)
            mean = np.zeros(f.shape[1], dtype=np.float64)
            m2 = np.zeros_like(mean)
            for i, row in enumerate(f):
                delta = row - mean
                mean += delta / (i + 1)
                m2 += delta * (row - mean)
                means[i] = mean
                if i:
                    sds[i] = np.sqrt(np.maximum(m2 / i, 0))
            daily = np.column_stack([daily, f, np.zeros_like(f)])
            cumulative = np.column_stack([cumulative, means, sds])
        out = days[["date"]].copy()
        out["adequate_days_so_far"] = np.arange(1, len(days)+1)
        out["daily_score"] = self.regression_.predict(daily.astype(np.float32))
        out["cumulative_score"] = self.regression_.predict(cumulative.astype(np.float32))
        out["observed_bins"] = np.isfinite(hr).sum(axis=1)
        return out
