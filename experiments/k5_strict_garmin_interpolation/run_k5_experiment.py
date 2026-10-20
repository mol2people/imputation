#!/usr/bin/env python
"""K=5 Disjoint Cross-Validation, Imputation Benchmark, and Production Retraining.

Track A: First 40 adequate days on Strict Garmin Cohort (N = 3,848).
Compares matched 'median' vs 'three_tier' imputation arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

# Threading and numerical environment (8 cores)
os.environ["NUMBA_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import joblib
import numpy as np
import pandas as pd
from numba import njit
from ortools.sat.python import cp_model
from scipy import stats
from scipy.interpolate import PchipInterpolator
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "k5_strict_garmin_interpolation"
TRACK_A_DIR = EXPERIMENT_DIR / "track_40days"
VENDOR_DIR = REPO_ROOT / "experiments" / "temporal_garmin" / "cache" / "vendor"

sys.path.insert(0, str(VENDOR_DIR / "multirocket"))
from multirocket.multirocket import fit as mr_fit, transform as mr_transform

DAY_BINS_PATH = REPO_ROOT / "experiments" / "temporal_garmin" / "cache" / "day_bins.npz"
DAY40_PARQUET = REPO_ROOT / "experiments" / "r1b_dateaware_garmin_2026-09-21" / "cache" / "day40_table.parquet"
FOLDS_REF_PARQUET = REPO_ROOT / "experiments" / "r1b_dateaware_garmin_2026-09-21" / "cache" / "folds.parquet"
COHORT_PARQUET = REPO_ROOT / "experiments" / "artifacts_sq" / "cohort_manifest_model_sources.parquet"

# Constants
SRC = 3
COHORT_SIZE = 3848
DAYS = 40
N_BINS = 288
SEED_BASE = 20261015
ALPHA_GRID = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
MR_NUM_FEATURES_PER_TRANS = 1250
MR_WIDTH = 9408
CHUNK_USERS = 256


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


@njit(cache=True)
def _nb_seed(s: int):
    np.random.seed(s)


def comp_seed(alloc: int, comp: int) -> int:
    return int(np.random.SeedSequence([SEED_BASE, SRC, alloc, comp]).generate_state(1, dtype=np.uint32)[0])


# ============================================================================
# 1. Test Fixtures (§5)
# ============================================================================
def run_imputation_fixtures():
    """Verify exact gap rules against pre-declared test fixtures."""
    log("Running imputation test fixtures (§5)...")
    L = 288
    hr = np.full(L, 75.0, dtype=np.float32)
    mask = np.ones(L, dtype=bool)

    # 1. Day boundary start: bins 0..2 (L=3)
    mask[0:3] = False; hr[0:3] = 0.0
    # 2. Interior L=1: bin 10, bounded by 9 (70.0) and 11 (80.0)
    mask[10] = False; hr[10] = 0.0; hr[9] = 70.0; hr[11] = 80.0
    # 3. Interior L=2 with 4-pt context: bins 20..21
    mask[20:22] = False; hr[20:22] = 0.0
    hr[18] = 70.0; hr[19] = 72.0; hr[22] = 76.0; hr[23] = 78.0
    # 4. Interior L=2 without 4-pt context (bin 29 missing breaks context for bins 31..32)
    mask[29] = False; hr[29] = 0.0
    mask[31:33] = False; hr[31:33] = 0.0
    hr[30] = 70.0; hr[33] = 80.0; hr[34] = 82.0
    # 5. Interior L=12 with 4-pt context: bins 40..51
    mask[40:52] = False; hr[40:52] = 0.0
    hr[38] = 65.0; hr[39] = 68.0; hr[52] = 74.0; hr[53] = 76.0
    # 6. Interior L=13: bins 70..82
    mask[70:83] = False; hr[70:83] = 0.0
    hr[68] = 60.0; hr[69] = 62.0; hr[83] = 70.0; hr[84] = 72.0
    # 7. Day boundary end: bins 285..287 (L=3)
    mask[285:288] = False; hr[285:288] = 0.0

    template = np.full(L, 65.0, dtype=np.float32)
    h_med, h_tier, _ = impute_both_arms(hr[None, None, :], mask[None, None, :], template)

    # Assertions
    # L=1 exact linear midpoint
    assert np.isclose(h_tier[0, 0, 10], 75.0, atol=1e-5), f"L=1 failed: {h_tier[0, 0, 10]}"
    # L=2 PCHIP monotonic within boundary
    assert 72.0 < h_tier[0, 0, 20] < h_tier[0, 0, 21] < 76.0, f"L=2 PCHIP failed: {h_tier[0, 0, 20:22]}"
    # L=2 linear fallback
    assert np.allclose(h_tier[0, 0, 31:33], [70.0 + 10.0 / 3.0, 70.0 + 20.0 / 3.0], atol=1e-5), f"L=2 fallback failed"
    # L=12 PCHIP bounded
    assert 68.0 <= h_tier[0, 0, 40:52].min() and h_tier[0, 0, 40:52].max() <= 74.0, "L=12 PCHIP bounded failed"
    # L=13 median fill
    assert np.all(h_tier[0, 0, 70:83] == 65.0), "L=13 median fill failed"
    # Day start and end boundaries
    assert np.all(h_tier[0, 0, 0:3] == 65.0), "Boundary start fill failed"
    assert np.all(h_tier[0, 0, 285:288] == 65.0), "Boundary end fill failed"
    # Median arm equals template everywhere missing
    assert np.all(h_med[0, 0, ~mask] == 65.0), "Median arm template fill failed"
    # Both arms equal original observed everywhere observed
    assert np.all(h_med[0, 0, mask] == hr[mask]), "Median arm observed values modified"
    assert np.all(h_tier[0, 0, mask] == hr[mask]), "Three-tier arm observed values modified"
    log("All test fixtures passed successfully!")


# ============================================================================
# 2. Imputation Engine (§3.2)
# ============================================================================
def precompute_gap_overlays(hr_raw_2d: np.ndarray, mask_2d: np.ndarray) -> dict:
    """Precompute Tier 1 & Tier 2 interpolation overlays across all days.
    
    hr_raw_2d: (R, 288) float32
    mask_2d:   (R, 288) bool
    """
    R, L = mask_2d.shape
    miss = ~mask_2d
    p = np.zeros((R, L + 2), dtype=bool)
    p[:, 1:-1] = miss
    d = np.diff(p.astype(np.int8), axis=1)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    assert starts.size == ends.size
    rows = (starts // (L + 1)).astype(np.int64)
    sr = (starts % (L + 1)).astype(np.int32)
    er = (ends % (L + 1)).astype(np.int32)
    rlen = (er - sr + 1).astype(np.int32)

    interior = (sr >= 1) & (er <= L - 2)

    # 4-point context: interior, 2 <= L <= 12, s >= 2, e <= L - 3, and all 4 neighbors observed
    cand_pchip = interior & (rlen >= 2) & (rlen <= 12) & (sr >= 2) & (er <= L - 3)
    has_ctx = np.zeros(rlen.size, dtype=bool)
    cp_idx = np.flatnonzero(cand_pchip)
    if cp_idx.size:
        r_c = rows[cp_idx]
        s_c = sr[cp_idx]
        e_c = er[cp_idx]
        ctx_ok = (mask_2d[r_c, s_c - 2] &
                  mask_2d[r_c, s_c - 1] &
                  mask_2d[r_c, e_c + 1] &
                  mask_2d[r_c, e_c + 2])
        has_ctx[cp_idx] = ctx_ok

    # Exposure counts
    total_missing_bins = int(miss.sum())
    tier1_linear_bins = int(rlen[interior & (rlen == 1)].sum())
    tier2_pchip_bins = int(rlen[has_ctx].sum())
    tier2_fb_bins = int(rlen[interior & (rlen >= 2) & (rlen <= 12) & (~has_ctx)].sum())
    tier3_median_bins = total_missing_bins - (tier1_linear_bins + tier2_pchip_bins + tier2_fb_bins)

    exposure = {
        "total_missing_bins": total_missing_bins,
        "tier1_linear_bins": tier1_linear_bins,
        "tier1_linear_pct": float(tier1_linear_bins / total_missing_bins * 100),
        "tier2_pchip_bins": tier2_pchip_bins,
        "tier2_pchip_pct": float(tier2_pchip_bins / total_missing_bins * 100),
        "tier2_fallback_bins": tier2_fb_bins,
        "tier2_fallback_pct": float(tier2_fb_bins / total_missing_bins * 100),
        "tier3_median_bins": tier3_median_bins,
        "tier3_median_pct": float(tier3_median_bins / total_missing_bins * 100),
        "total_missing_runs": int(rlen.size),
        "interior_runs": int(interior.sum()),
        "boundary_runs": int((~interior).sum()),
    }

    # Precompute flat indices and interpolated values
    pos_list = []
    val_list = []

    # 1. Tier 1 linear: L=1 interior
    l1_idx = np.flatnonzero(interior & (rlen == 1))
    if l1_idx.size:
        r1, s1 = rows[l1_idx], sr[l1_idx]
        p_l1 = r1 * L + s1
        v_l1 = 0.5 * (hr_raw_2d[r1, s1 - 1] + hr_raw_2d[r1, s1 + 1])
        pos_list.append(p_l1)
        val_list.append(v_l1)

    # 2. Tier 2 PCHIP: 2 <= L <= 12 with context
    pchip_runs = np.flatnonzero(has_ctx)
    if pchip_runs.size:
        for Lg in range(2, 13):
            ii = pchip_runs[rlen[pchip_runs] == Lg]
            if ii.size == 0:
                continue
            r_ii, s_ii, e_ii = rows[ii], sr[ii], er[ii]
            y = np.stack([hr_raw_2d[r_ii, s_ii - 2], hr_raw_2d[r_ii, s_ii - 1],
                          hr_raw_2d[r_ii, e_ii + 1], hr_raw_2d[r_ii, e_ii + 2]], axis=1).astype(np.float64)
            x_pts = np.array([0.0, 1.0, Lg + 2.0, Lg + 3.0])
            pb = PchipInterpolator(x_pts, y.T, extrapolate=False)
            vals = pb(np.arange(2.0, 2.0 + Lg, dtype=np.float64)).T.astype(np.float32)  # (len(ii), Lg)
            
            p_L = (r_ii[:, None] * L + s_ii[:, None] + np.arange(Lg)[None, :]).ravel()
            pos_list.append(p_L)
            val_list.append(vals.ravel())

    # 3. Tier 2 fallback linear: 2 <= L <= 12 without context
    fb_runs = np.flatnonzero(interior & (rlen >= 2) & (rlen <= 12) & (~has_ctx))
    if fb_runs.size:
        p_fb_list, v_fb_list = [], []
        for idx in fb_runs:
            r_f, s_f, e_f, L_f = rows[idx], sr[idx], er[idx], rlen[idx]
            xL = hr_raw_2d[r_f, s_f - 1]
            xR = hr_raw_2d[r_f, e_f + 1]
            j_steps = np.arange(1, L_f + 1, dtype=np.float32)
            vals = xL + (j_steps / (L_f + 1.0)) * (xR - xL)
            p_fb = r_f * L + s_f + np.arange(L_f)
            p_fb_list.append(p_fb)
            v_fb_list.append(vals)
        pos_list.append(np.concatenate(p_fb_list))
        val_list.append(np.concatenate(v_fb_list))

    if pos_list:
        overlay_pos = np.concatenate(pos_list)
        overlay_val = np.concatenate(val_list)
    else:
        overlay_pos = np.empty(0, dtype=np.int64)
        overlay_val = np.empty(0, dtype=np.float32)

    return {
        "overlay_pos": overlay_pos,
        "overlay_val": overlay_val,
        "exposure": exposure,
    }


def compute_clock_medians(hr_raw: np.ndarray, mask_pu: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Compute 288-bin clock median template strictly on training participants."""
    hr_tr = hr_raw[train_idx]
    mask_tr = mask_pu[train_idx]
    obs = hr_tr[mask_tr]
    overall = float(np.nanmedian(obs)) if obs.size else 75.0
    template = np.full(N_BINS, overall, dtype=np.float32)
    for b in range(N_BINS):
        col_obs = hr_tr[:, :, b][mask_tr[:, :, b]]
        if col_obs.size:
            template[b] = np.float32(np.nanmedian(col_obs))
    template = np.where(np.isfinite(template), template, overall)
    return template


def impute_both_arms(hr_raw: np.ndarray, mask_pu: np.ndarray, template: np.ndarray,
                     overlay_info: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (hr_median, hr_three_tier, exposure)."""
    n, D, L = hr_raw.shape
    R = n * D
    hr_2d = hr_raw.reshape(R, L)
    mask_2d = mask_pu.reshape(R, L)

    # Arm 1: median
    hr_median_2d = np.where(mask_2d, hr_2d, template[None, :]).astype(np.float32)

    # Arm 2: three_tier
    hr_three_tier_2d = hr_median_2d.copy()
    if overlay_info is None:
        overlay_info = precompute_gap_overlays(hr_2d, mask_2d)

    if overlay_info["overlay_pos"].size:
        flat_tier = hr_three_tier_2d.ravel()
        flat_tier[overlay_info["overlay_pos"]] = overlay_info["overlay_val"]
        hr_three_tier_2d = flat_tier.reshape(R, L)

    hr_median = hr_median_2d.reshape(n, D, L)
    hr_three_tier = hr_three_tier_2d.reshape(n, D, L)
    return hr_median, hr_three_tier, overlay_info["exposure"]


# ============================================================================
# 3. Disjoint 5-Fold Partitioning (§3.1)
# ============================================================================
def generate_disjoint_5folds(demographics: pd.DataFrame, seed: int = SEED_BASE) -> pd.DataFrame:
    """Solve balanced 5-way partition using OR-Tools CP-SAT with 8 workers."""
    n = len(demographics)
    k = 5
    log(f"Solving balanced 5-way partition for {n} participants using OR-Tools CP-SAT (seed {seed})...")

    model = cp_model.CpModel()
    x = {(i, j): model.new_bool_var(f"x_{i}_{j}") for i in range(n) for j in range(k)}

    # Each participant assigned to exactly 1 fold
    for i in range(n):
        model.add(sum(x[i, j] for j in range(k)) == 1)

    # Fold sizes: 770, 770, 770, 769, 769
    target_sizes = [770, 770, 770, 769, 769]
    for j in range(k):
        model.add(sum(x[i, j] for i in range(n)) == target_sizes[j])

    # Salutation hard constraint
    sal10_idx = np.flatnonzero(demographics["salutation"].astype(int) == 10).tolist()
    target_sal10 = [277, 277, 277, 277, 276]
    for j in range(k):
        model.add(sum(x[i, j] for i in sal10_idx) == target_sal10[j])

    # Age and BMI deviations
    devs = []
    for col in ["age_band", "bmi_group"]:
        groups = demographics.groupby(col, sort=True).indices
        for c_num, positions in enumerate(groups.values()):
            c_size = len(positions)
            target_in_cell = c_size // k
            for j in range(k):
                diff = model.new_int_var(-c_size, c_size, f"diff_{col}_{c_num}_{j}")
                dev = model.new_int_var(0, c_size, f"dev_{col}_{c_num}_{j}")
                model.add(diff == sum(x[p, j] for p in positions) - target_in_cell)
                model.add_abs_equality(dev, diff)
                devs.append(dev)

    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 10, size=(n, k))
    obj = 1000 * sum(devs) + sum(int(weights[i, j]) * x[i, j] for i in range(n) for j in range(k))
    model.minimize(obj)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = seed
    solver.parameters.max_time_in_seconds = 30.0

    t0 = time.time()
    status = solver.solve(model)
    log(f"CP-SAT solver status: {solver.status_name(status)} in {time.time() - t0:.2f}s")
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), f"Solver failed: {solver.status_name(status)}"

    fold_assign = np.empty(n, dtype=object)
    for i in range(n):
        for j in range(k):
            if solver.value(x[i, j]):
                fold_assign[i] = f"fold_{j}"

    out_df = demographics.copy()
    out_df["fold"] = fold_assign
    return out_df


# ============================================================================
# 4. MultiRocket & B40 Feature Pipeline (§3.3)
# ============================================================================
def build_b40_static(users: np.ndarray, d40: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Compute 305 B40 static summary features."""
    n = len(users)
    g = d40.groupby("user", sort=True)
    stat_cols = [f"d_ch3000_{s}" for s in ("mean", "median", "sd", "vmin", "vmax", "n", "cov_h", "hours")]
    means = g[stat_cols].mean().reindex(users).to_numpy(dtype=np.float32)
    sds = g[stat_cols].std(ddof=1).reindex(users).to_numpy(dtype=np.float32)

    u_pos = pd.Series(np.arange(n), index=users)
    g_pos = d40.user.map(u_pos).to_numpy()
    is_we = d40.d_is_weekend.to_numpy().astype(bool)
    ch_mean = d40.d_ch3000_mean.to_numpy(dtype=np.float32)
    wd_sum = np.bincount(g_pos[~is_we], weights=ch_mean[~is_we], minlength=n)
    wd_cnt = np.bincount(g_pos[~is_we], minlength=n)
    we_sum = np.bincount(g_pos[is_we], weights=ch_mean[is_we], minlength=n)
    we_cnt = np.bincount(g_pos[is_we], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        wd_mean = np.where(wd_cnt > 0, wd_sum / wd_cnt, np.nan)
        we_mean = np.where(we_cnt > 0, we_sum / we_cnt, np.nan)
    weekend_diff = (wd_mean - we_mean).astype(np.float32)[:, None]

    mask_pu = mask.reshape(n, DAYS, N_BINS)
    mask_means = mask_pu.mean(axis=1).astype(np.float32)
    B40 = np.concatenate([means, sds, weekend_diff, mask_means], axis=1)
    assert B40.shape == (n, 305) and np.isfinite(B40).all()
    return B40


def apply_mr_pooled_chunked(mr_transform_fn, params_base, params_diff, hr_filled, mask_pu):
    """Apply MultiRocket transform and pool mean + SD across 40 days."""
    n = hr_filled.shape[0]
    pooled_mean = np.zeros((n, MR_WIDTH), dtype=np.float32)
    pooled_sd = np.zeros((n, MR_WIDTH), dtype=np.float32)
    for s in range(0, n, CHUNK_USERS):
        sz = min(CHUNK_USERS, n - s)
        X = hr_filled[s:s + sz].reshape(sz * DAYS, N_BINS).astype(np.float64)
        X1 = np.diff(X, axis=1)
        F = mr_transform_fn(X, X1, params_base, params_diff, n_features_per_kernel=4)
        F = np.nan_to_num(F).astype(np.float32).reshape(sz, DAYS, MR_WIDTH)
        obs = mask_pu[s:s + sz].any(axis=2)
        F = np.where(obs[:, :, None], F, np.nan)
        with np.errstate(invalid="ignore", all="ignore"):
            m = np.nanmean(F, axis=1, dtype=np.float64)
            sd = np.nanstd(F, axis=1, ddof=1, dtype=np.float64)
        pooled_mean[s:s + sz] = np.nan_to_num(m, nan=0.0).astype(np.float32)
        pooled_sd[s:s + sz] = np.nan_to_num(sd, nan=0.0).astype(np.float32)
        del F
    return np.concatenate([pooled_mean, pooled_sd], axis=1)  # (n, 18816)


# ============================================================================
# 5. Statistical Inference (§3.5)
# ============================================================================
def compute_delong_ci(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 0.05) -> tuple[float, float, tuple[float, float]]:
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true]
    neg = y_score[~y_true]
    m, n = len(neg), len(pos)
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)

    v10 = (np.searchsorted(neg_sorted, pos, side="left") + np.searchsorted(neg_sorted, pos, side="right")) / (2.0 * m)
    v01 = 1.0 - (np.searchsorted(pos_sorted, neg, side="left") + np.searchsorted(pos_sorted, neg, side="right")) / (2.0 * n)

    auc = float(np.mean(v10))
    var_auc = float(np.var(v10, ddof=1) / n + np.var(v01, ddof=1) / m)
    se = float(np.sqrt(var_auc))
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    ci = (float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se)))
    return auc, se, ci


def compute_stratified_bootstrap_ci(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray | None = None,
                                    n_boot: int = 2000, seed: int = SEED_BASE) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    auc_a = np.empty(n_boot, dtype=float)
    auc_b = np.empty(n_boot, dtype=float) if scores_b is not None else None
    diffs = np.empty(n_boot, dtype=float) if scores_b is not None else None

    for b in range(n_boot):
        sample_pos = rng.choice(pos_idx, n_pos, replace=True)
        sample_neg = rng.choice(neg_idx, n_neg, replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        y_b = y_true[sample_idx]

        a_score = roc_auc_score(y_b, scores_a[sample_idx])
        auc_a[b] = a_score
        if scores_b is not None:
            b_score = roc_auc_score(y_b, scores_b[sample_idx])
            auc_b[b] = b_score
            diffs[b] = b_score - a_score

    res = {
        "auc_a_ci": (float(np.percentile(auc_a, 2.5)), float(np.percentile(auc_a, 97.5))),
        "auc_a_mean": float(np.mean(auc_a)),
        "auc_a_se": float(np.std(auc_a, ddof=1)),
    }
    if scores_b is not None:
        res["auc_b_ci"] = (float(np.percentile(auc_b, 2.5)), float(np.percentile(auc_b, 97.5)))
        res["auc_b_mean"] = float(np.mean(auc_b))
        res["diff_ci"] = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
        res["diff_mean"] = float(np.mean(diffs))
        res["diff_se"] = float(np.std(diffs, ddof=1))
    return res


# ============================================================================
# 6. Main Runner (§3.4, §3.5, §3.6)
# ============================================================================
def main():
    t_start_all = time.time()
    log("=== STARTING TRACK A: STRICT GARMIN (40 DAYS) K=5 BENCHMARK ===")
    run_imputation_fixtures()

    TRACK_A_DIR.mkdir(parents=True, exist_ok=True)
    (TRACK_A_DIR / "median").mkdir(exist_ok=True)
    (TRACK_A_DIR / "three_tier").mkdir(exist_ok=True)

    # 1. Load Data
    log("Loading day_bins.npz, day40_table, and demographics...")
    z = np.load(DAY_BINS_PATH)
    hr_raw_flat = z["hr"].astype(np.float32)
    mask_flat = z["mask"].astype(bool)
    pu = z["user"].astype(np.int64)
    users = np.sort(np.unique(pu))
    assert len(users) == COHORT_SIZE

    hr_raw = hr_raw_flat.reshape(COHORT_SIZE, DAYS, N_BINS)
    mask_pu = mask_flat.reshape(COHORT_SIZE, DAYS, N_BINS)

    d40 = pd.read_parquet(DAY40_PARQUET)
    d40 = d40[d40.user.isin(set(users))].sort_values(["user", "date"], kind="stable").reset_index(drop=True)
    b40_static = build_b40_static(users, d40, mask_flat)

    # Demographics
    f_ref = pd.read_parquet(FOLDS_REF_PARQUET)
    demo = f_ref[f_ref.alloc == 0].drop(columns=["alloc", "fold"]).sort_values("user_id").reset_index(drop=True)
    assert set(demo["user_id"]) == set(users)
    y_all = (demo["salutation"].astype(int) == 20).to_numpy(dtype=int)

    # 2. Partitioning
    folds_path = EXPERIMENT_DIR / "folds_5fold_disjoint.parquet"
    if folds_path.exists():
        log(f"Loading existing 5-fold disjoint partition from {folds_path}...")
        df_folds = pd.read_parquet(folds_path)
    else:
        df_folds = generate_disjoint_5folds(demo, seed=SEED_BASE)
        df_folds.to_parquet(folds_path, index=False)
        log(f"Saved 5-fold disjoint partition to {folds_path}.")

    fold_labels = df_folds.set_index("user_id")["fold"].reindex(users).to_numpy()
    assert len(fold_labels) == COHORT_SIZE
    for k in range(5):
        assert (fold_labels == f"fold_{k}").sum() in (769, 770)

    # 3. Precompute Imputation Overlays
    log("Precomputing gap overlays and exposure audit...")
    overlay_info = precompute_gap_overlays(hr_raw.reshape(-1, N_BINS), mask_pu.reshape(-1, N_BINS))
    exposure = overlay_info["exposure"]
    with open(TRACK_A_DIR / "imputation_exposure.json", "w") as f:
        json.dump(exposure, f, indent=2)
    log(f"Exposure audit: total missing bins={exposure['total_missing_bins']}, "
        f"L=1 linear={exposure['tier1_linear_bins']} ({exposure['tier1_linear_pct']:.2f}%), "
        f"2..12 PCHIP={exposure['tier2_pchip_bins']} ({exposure['tier2_pchip_pct']:.2f}%), "
        f"2..12 fallback={exposure['tier2_fallback_bins']} ({exposure['tier2_fallback_pct']:.2f}%), "
        f">12 median={exposure['tier3_median_bins']} ({exposure['tier3_median_pct']:.2f}%).")

    # 4. Cross-Validation Loop
    log("\n=== STARTING 5-FOLD CROSS-VALIDATION SWEEP ===")
    arms = ["median", "three_tier"]
    val_aurocs = {arm: {a: [] for a in ALPHA_GRID} for arm in arms}
    val_preds_per_alpha = {arm: {a: np.zeros(COHORT_SIZE, dtype=float) for a in ALPHA_GRID} for arm in arms}

    for fold_k in range(5):
        t_fold_0 = time.time()
        val_mask = (fold_labels == f"fold_{fold_k}")
        tr_mask = ~val_mask
        tr_idx = np.flatnonzero(tr_mask)
        va_idx = np.flatnonzero(val_mask)
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        log(f"\n--- Fold {fold_k}: Train={len(tr_idx)}, Val={len(va_idx)} ---")

        # Step 4a: Clock median template fit on training participants
        template_k = compute_clock_medians(hr_raw, mask_pu, tr_idx)

        # Step 4b: Impute both arms
        hr_med, hr_tier, _ = impute_both_arms(hr_raw, mask_pu, template_k, overlay_info)

        # Step 4c: Train-day selection for MR fitting (4 days/user; matched seed)
        fit_rng = np.random.default_rng(comp_seed(fold_k, 0))
        fit_days = np.stack([fit_rng.choice(DAYS, 4, replace=False) for _ in range(len(tr_idx))])

        for arm in arms:
            t_arm_0 = time.time()
            hr_arm = hr_med if arm == "median" else hr_tier

            # Fit MultiRocket base and diff kernels on training days
            X_fit = hr_arm[tr_idx[:, None], fit_days].reshape(-1, N_BINS).astype(np.float64)
            X1_fit = np.diff(X_fit, axis=1)

            _nb_seed(comp_seed(fold_k, 1))
            mr_base = mr_fit(X_fit, num_features=MR_NUM_FEATURES_PER_TRANS, max_dilations_per_kernel=32)
            _nb_seed(comp_seed(fold_k, 1) + 1)
            mr_diff = mr_fit(X1_fit, num_features=MR_NUM_FEATURES_PER_TRANS, max_dilations_per_kernel=32)

            # Transform days and pool
            mr_features = apply_mr_pooled_chunked(mr_transform, mr_base, mr_diff, hr_arm, mask_pu)
            X_full = np.concatenate([b40_static, mr_features], axis=1)  # (3848, 19121)

            # Hygiene preprocessing strictly on training
            imp = SimpleImputer(strategy="median").fit(X_full[tr_idx])
            X_imp_tr = imp.transform(X_full[tr_idx])
            X_imp_va = imp.transform(X_full[va_idx])

            vt = VarianceThreshold(0.0).fit(X_imp_tr)
            X_vt_tr = vt.transform(X_imp_tr)
            X_vt_va = vt.transform(X_imp_va)

            scaler = StandardScaler().fit(X_vt_tr)
            X_sc_tr = scaler.transform(X_vt_tr)
            X_sc_va = scaler.transform(X_vt_va)

            # Class weighting
            sample_weights = np.where(y_tr == 1, len(y_tr) / (2.0 * np.sum(y_tr == 1)),
                                      len(y_tr) / (2.0 * np.sum(y_tr == 0)))

            # Sweep alpha
            fold_results = []
            for a in ALPHA_GRID:
                clf = Ridge(alpha=float(a), fit_intercept=True)
                clf.fit(X_sc_tr, y_tr, sample_weight=sample_weights)
                preds = clf.predict(X_sc_va)
                auc = float(roc_auc_score(y_va, preds))
                val_aurocs[arm][a].append(auc)
                val_preds_per_alpha[arm][a][va_idx] = preds
                fold_results.append(f"α={int(a)}: {auc:.4f}")

            log(f"  [{arm:10s}] {', '.join(fold_results)} ({time.time() - t_arm_0:.1f}s)")

        log(f"Fold {fold_k} completed in {time.time() - t_fold_0:.1f}s.")

    # 5. Hyperparameter Selection (§3.4)
    log("\n=== HYPERPARAMETER SELECTION RESULTS ===")
    selected_alpha = {}
    for arm in arms:
        alpha_df = pd.DataFrame({
            "alpha": ALPHA_GRID,
            "mean_val_auroc": [np.mean(val_aurocs[arm][a]) for a in ALPHA_GRID],
            "std_val_auroc": [np.std(val_aurocs[arm][a], ddof=1) for a in ALPHA_GRID],
            "fold_0": [val_aurocs[arm][a][0] for a in ALPHA_GRID],
            "fold_1": [val_aurocs[arm][a][1] for a in ALPHA_GRID],
            "fold_2": [val_aurocs[arm][a][2] for a in ALPHA_GRID],
            "fold_3": [val_aurocs[arm][a][3] for a in ALPHA_GRID],
            "fold_4": [val_aurocs[arm][a][4] for a in ALPHA_GRID],
        })
        alpha_df.to_csv(TRACK_A_DIR / arm / "alpha_grid_results.csv", index=False)
        best_idx = alpha_df["mean_val_auroc"].idxmax()
        best_a = float(alpha_df.loc[best_idx, "alpha"])
        best_auc = float(alpha_df.loc[best_idx, "mean_val_auroc"])
        selected_alpha[arm] = best_a
        log(f"Arm [{arm:10s}] Optimal α* = {best_a:g} (Mean Val AUROC = {best_auc:.4f} ± {alpha_df.loc[best_idx, 'std_val_auroc']:.4f})")

    # 6. Final OOF Evaluation (§3.5)
    log("\n=== COMPUTING FINAL OUT-OF-FOLD (OOF) METRICS (N=3,848) ===")
    oof_scores = {}
    metrics_summary = {}

    for arm in arms:
        a_star = selected_alpha[arm]
        scores = val_preds_per_alpha[arm][a_star]
        oof_scores[arm] = scores

        # DeLong CI
        auc, auc_se, delong_ci = compute_delong_ci(y_all, scores)
        ap = float(average_precision_score(y_all, scores))

        # Platt recalibration
        platt = LogisticRegression(C=1e5, solver="lbfgs")
        platt.fit(scores[:, None], y_all)
        probs = platt.predict_proba(scores[:, None])[:, 1]
        brier = float(brier_score_loss(y_all, probs))

        # Threshold optimization for balanced accuracy
        thresholds = np.linspace(probs.min(), probs.max(), 1000)
        best_bacc, best_thresh = -1.0, 0.5
        for t in thresholds:
            bacc = stats.gmean if False else (
                0.5 * ((probs >= t)[y_all == 1].mean() + (probs < t)[y_all == 0].mean())
            )
            if bacc > best_bacc:
                best_bacc, best_thresh = bacc, t

        preds_bin = (probs >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_all, preds_bin).ravel()
        sens = float(tp / (tp + fn))
        spec = float(tn / (tn + fp))
        acc = float((tp + tn) / len(y_all))

        # Save OOF predictions CSV
        oof_df = df_folds[["user_id", "salutation", "age_band", "bmi_group", "fold"]].copy()
        oof_df["y_true"] = y_all
        oof_df["ridge_margin"] = scores
        oof_df["calibrated_prob"] = probs
        oof_df["pred_binary"] = preds_bin
        oof_df.to_csv(TRACK_A_DIR / arm / "oof_predictions.csv", index=False)

        # Subgroup audits
        subgroups = {}
        for grp_col in ["age_band", "bmi_group"]:
            subgroups[grp_col] = {}
            for g_val, g_idx in df_folds.groupby(grp_col).indices.items():
                if len(np.unique(y_all[g_idx])) > 1:
                    g_auc = float(roc_auc_score(y_all[g_idx], scores[g_idx]))
                else:
                    g_auc = None
                subgroups[grp_col][str(g_val)] = {
                    "n": len(g_idx),
                    "n_class_1": int(np.sum(y_all[g_idx] == 1)),
                    "auroc": g_auc,
                    "accuracy": float(acc if False else (preds_bin[g_idx] == y_all[g_idx]).mean()),
                }

        metrics_summary[arm] = {
            "selected_alpha": a_star,
            "oof_auroc": auc,
            "delong_se": auc_se,
            "delong_95_ci": delong_ci,
            "average_precision": ap,
            "brier_score": brier,
            "platt_coef": float(platt.coef_[0, 0]),
            "platt_intercept": float(platt.intercept_[0]),
            "optimal_threshold": float(best_thresh),
            "balanced_accuracy": float(best_bacc),
            "sensitivity": sens,
            "specificity": spec,
            "accuracy": acc,
            "fold_aurocs": val_aurocs[arm][a_star],
            "fold_auroc_mean": float(np.mean(val_aurocs[arm][a_star])),
            "fold_auroc_std": float(np.std(val_aurocs[arm][a_star], ddof=1)),
            "subgroups": subgroups,
        }

    # Paired Bootstrap Comparison
    log("Computing 2,000-draw paired stratified bootstrap...")
    boot_res = compute_stratified_bootstrap_ci(y_all, oof_scores["median"], oof_scores["three_tier"], n_boot=2000)
    for arm in arms:
        metrics_summary[arm]["bootstrap_95_ci"] = boot_res["auc_a_ci"] if arm == "median" else boot_res["auc_b_ci"]
        with open(TRACK_A_DIR / arm / "metrics_summary.json", "w") as f:
            json.dump(metrics_summary[arm], f, indent=2)

    # Paired fold deltas
    a_med = selected_alpha["median"]
    a_tier = selected_alpha["three_tier"]
    fold_deltas = [tier - med for med, tier in zip(val_aurocs["median"][a_med], val_aurocs["three_tier"][a_tier])]
    oof_delta = metrics_summary["three_tier"]["oof_auroc"] - metrics_summary["median"]["oof_auroc"]

    comp_df = pd.DataFrame({
        "fold": [f"fold_{k}" for k in range(5)] + ["mean_fold", "pooled_oof"],
        "median_auroc": val_aurocs["median"][a_med] + [metrics_summary["median"]["fold_auroc_mean"], metrics_summary["median"]["oof_auroc"]],
        "three_tier_auroc": val_aurocs["three_tier"][a_tier] + [metrics_summary["three_tier"]["fold_auroc_mean"], metrics_summary["three_tier"]["oof_auroc"]],
        "delta_three_tier_minus_median": fold_deltas + [float(np.mean(fold_deltas)), oof_delta],
    })
    comp_df.to_csv(TRACK_A_DIR / "imputation_comparison.csv", index=False)

    log("\n=== FINAL RESULTS SUMMARY (TRACK A - 40 DAYS) ===")
    log(f"Median Arm:     OOF AUROC = {metrics_summary['median']['oof_auroc']:.4f} "
        f"(DeLong 95% CI: [{metrics_summary['median']['delong_95_ci'][0]:.4f}, {metrics_summary['median']['delong_95_ci'][1]:.4f}], "
        f"Bootstrap 95% CI: [{metrics_summary['median']['bootstrap_95_ci'][0]:.4f}, {metrics_summary['median']['bootstrap_95_ci'][1]:.4f}])")
    log(f"Three-Tier Arm: OOF AUROC = {metrics_summary['three_tier']['oof_auroc']:.4f} "
        f"(DeLong 95% CI: [{metrics_summary['three_tier']['delong_95_ci'][0]:.4f}, {metrics_summary['three_tier']['delong_95_ci'][1]:.4f}], "
        f"Bootstrap 95% CI: [{metrics_summary['three_tier']['bootstrap_95_ci'][0]:.4f}, {metrics_summary['three_tier']['bootstrap_95_ci'][1]:.4f}])")
    log(f"Paired Difference (Three-Tier - Median): {oof_delta:+.4f} "
        f"(Bootstrap 95% CI: [{boot_res['diff_ci'][0]:+.4f}, {boot_res['diff_ci'][1]:+.4f}])")

    # 7. Full-Cohort Retraining on K/K ($N=3,848$) (§3.6)
    log("\n=== FULL-COHORT RETRAINING ON 100% OF COHORT (N=3,848) ===")
    template_full = compute_clock_medians(hr_raw, mask_pu, np.arange(COHORT_SIZE))
    hr_med_full, hr_tier_full, _ = impute_both_arms(hr_raw, mask_pu, template_full, overlay_info)

    fit_rng_full = np.random.default_rng(comp_seed(999, 0))
    fit_days_full = np.stack([fit_rng_full.choice(DAYS, 4, replace=False) for _ in range(COHORT_SIZE)])
    full_sample_weights = np.where(y_all == 1, COHORT_SIZE / (2.0 * np.sum(y_all == 1)),
                                   COHORT_SIZE / (2.0 * np.sum(y_all == 0)))

    for arm in arms:
        t_retrain_0 = time.time()
        log(f"Retraining production bundle for [{arm}] with α*={selected_alpha[arm]}...")
        hr_full = hr_med_full if arm == "median" else hr_tier_full

        X_fit_full = hr_full[np.arange(COHORT_SIZE)[:, None], fit_days_full].reshape(-1, N_BINS).astype(np.float64)
        X1_fit_full = np.diff(X_fit_full, axis=1)

        _nb_seed(comp_seed(999, 1))
        mr_base_full = mr_fit(X_fit_full, num_features=MR_NUM_FEATURES_PER_TRANS, max_dilations_per_kernel=32)
        _nb_seed(comp_seed(999, 1) + 1)
        mr_diff_full = mr_fit(X1_fit_full, num_features=MR_NUM_FEATURES_PER_TRANS, max_dilations_per_kernel=32)

        mr_features_full = apply_mr_pooled_chunked(mr_transform, mr_base_full, mr_diff_full, hr_full, mask_pu)
        X_full_cohort = np.concatenate([b40_static, mr_features_full], axis=1)

        imp_full = SimpleImputer(strategy="median").fit(X_full_cohort)
        X_imp_f = imp_full.transform(X_full_cohort)

        vt_full = VarianceThreshold(0.0).fit(X_imp_f)
        X_vt_f = vt_full.transform(X_imp_f)

        scaler_full = StandardScaler().fit(X_vt_f)
        X_sc_f = scaler_full.transform(X_vt_f)

        clf_full = Ridge(alpha=float(selected_alpha[arm]), fit_intercept=True)
        clf_full.fit(X_sc_f, y_all, sample_weight=full_sample_weights)

        # Production bundle
        bundle = {
            "arm": arm,
            "track": "40days",
            "cohort_size": COHORT_SIZE,
            "alpha": selected_alpha[arm],
            "template": template_full,
            "mr_base": mr_base_full,
            "mr_diff": mr_diff_full,
            "imputer": imp_full,
            "variance_threshold": vt_full,
            "scaler": scaler_full,
            "classifier": clf_full,
            "platt_coef": metrics_summary[arm]["platt_coef"],
            "platt_intercept": metrics_summary[arm]["platt_intercept"],
            "optimal_threshold": metrics_summary[arm]["optimal_threshold"],
            "oof_auroc": metrics_summary[arm]["oof_auroc"],
            "brier_score": metrics_summary[arm]["brier_score"],
        }
        bundle_path = TRACK_A_DIR / arm / "production_bundle.joblib"
        joblib.dump(bundle, bundle_path, compress=3)
        log(f"Saved production bundle to {bundle_path} ({time.time() - t_retrain_0:.1f}s).")

    # 8. Markdown Report
    log("\nGenerating comprehensive report (REPORT.md)...")
    report_content = f"""# Strict Garmin Track A (40-Day Budget) Benchmark Report

**Date**: {time.strftime('%Y-%m-%d')}  
**Cohort**: Strict Garmin Single-Source Cohort ($N = 3,848$, Tier 0)  
**Input History**: First 40 adequate days ($153,920$ total participant-days)  
**Evaluation**: $K=5$ disjoint stratified cross-validation (OR-Tools CP-SAT) with zero untested participants.

---

## 1. Executive Summary & Paired Comparison

| Metric | `median` (Clock Medians) | `three_tier` (Linear + PCHIP + Medians) | Paired Difference (`three_tier − median`) |
|---|---:|---:|---:|
| **Optimal Regularization $\\alpha^*$** | `{selected_alpha['median']:g}` | `{selected_alpha['three_tier']:g}` | — |
| **Pooled OOF AUROC** | **{metrics_summary['median']['oof_auroc']:.4f}** | **{metrics_summary['three_tier']['oof_auroc']:.4f}** | **{oof_delta:+.4f}** |
| **DeLong 95% Confidence Interval** | `[{metrics_summary['median']['delong_95_ci'][0]:.4f}, {metrics_summary['median']['delong_95_ci'][1]:.4f}]` | `[{metrics_summary['three_tier']['delong_95_ci'][0]:.4f}, {metrics_summary['three_tier']['delong_95_ci'][1]:.4f}]` | — |
| **Stratified Bootstrap 95% CI (2,000 draws)** | `[{metrics_summary['median']['bootstrap_95_ci'][0]:.4f}, {metrics_summary['median']['bootstrap_95_ci'][1]:.4f}]` | `[{metrics_summary['three_tier']['bootstrap_95_ci'][0]:.4f}, {metrics_summary['three_tier']['bootstrap_95_ci'][1]:.4f}]` | **`[{boot_res['diff_ci'][0]:+.4f}, {boot_res['diff_ci'][1]:+.4f}]`** |
| **Average Precision (AUPRC)** | `{metrics_summary['median']['average_precision']:.4f}` | `{metrics_summary['three_tier']['average_precision']:.4f}` | `{metrics_summary['three_tier']['average_precision'] - metrics_summary['median']['average_precision']:+.4f}` |
| **Brier Score** | `{metrics_summary['median']['brier_score']:.4f}` | `{metrics_summary['three_tier']['brier_score']:.4f}` | `{metrics_summary['three_tier']['brier_score'] - metrics_summary['median']['brier_score']:+.4f}` |
| **Balanced Accuracy (at $\\tau^*$)** | `{metrics_summary['median']['balanced_accuracy']:.4f}` | `{metrics_summary['three_tier']['balanced_accuracy']:.4f}` | `{metrics_summary['three_tier']['balanced_accuracy'] - metrics_summary['median']['balanced_accuracy']:+.4f}` |
| **Sensitivity / Specificity** | `{metrics_summary['median']['sensitivity']:.4f} / {metrics_summary['median']['specificity']:.4f}` | `{metrics_summary['three_tier']['sensitivity']:.4f} / {metrics_summary['three_tier']['specificity']:.4f}` | — |
| **Fold AUROC (Mean $\\pm$ SD)** | `{metrics_summary['median']['fold_auroc_mean']:.4f} ± {metrics_summary['median']['fold_auroc_std']:.4f}` | `{metrics_summary['three_tier']['fold_auroc_mean']:.4f} ± {metrics_summary['three_tier']['fold_auroc_std']:.4f}` | `{np.mean(fold_deltas):+.4f} ± {np.std(fold_deltas, ddof=1):.4f}` |

---

## 2. Imputation Intervention Exposure Audit

Of the **{exposure['total_missing_bins']:,}** total missing bins in the 40-day dataset ({exposure['total_missing_runs']:,} maximal missing runs):
* **Tier 1 (Linear $\\le 5\\text{{ min}}$)**: {exposure['tier1_linear_bins']:,} bins (**{exposure['tier1_linear_pct']:.2f}%** of missing data).
* **Tier 2 (PCHIP $10\\text{{--}}60\\text{{ min}}$)**: {exposure['tier2_pchip_bins']:,} bins (**{exposure['tier2_pchip_pct']:.2f}%**).
* **Tier 2 (Linear Fallback $10\\text{{--}}60\\text{{ min}}$)**: {exposure['tier2_fallback_bins']:,} bins (**{exposure['tier2_fallback_pct']:.2f}%**).
* **Tier 3 (Diurnal Clock Median $>60\\text{{ min}}$ / Boundary)**: {exposure['tier3_median_bins']:,} bins (**{exposure['tier3_median_pct']:.2f}%**).

---

## 3. Fold-by-Fold AUROC Breakdown

| Split | $N_{{\\text{{val}}}}$ | `median` ($\\alpha^*={selected_alpha['median']:g}$) | `three_tier` ($\\alpha^*={selected_alpha['three_tier']:g}$) | Delta (`three_tier − median`) |
|---|---:|---:|---:|---:|
| **Fold 0** | 770 | {val_aurocs['median'][a_med][0]:.4f} | {val_aurocs['three_tier'][a_tier][0]:.4f} | {fold_deltas[0]:+.4f} |
| **Fold 1** | 770 | {val_aurocs['median'][a_med][1]:.4f} | {val_aurocs['three_tier'][a_tier][1]:.4f} | {fold_deltas[1]:+.4f} |
| **Fold 2** | 770 | {val_aurocs['median'][a_med][2]:.4f} | {val_aurocs['three_tier'][a_tier][2]:.4f} | {fold_deltas[2]:+.4f} |
| **Fold 3** | 769 | {val_aurocs['median'][a_med][3]:.4f} | {val_aurocs['three_tier'][a_tier][3]:.4f} | {fold_deltas[3]:+.4f} |
| **Fold 4** | 769 | {val_aurocs['median'][a_med][4]:.4f} | {val_aurocs['three_tier'][a_tier][4]:.4f} | {fold_deltas[4]:+.4f} |
| **Mean $\\pm$ SD** | — | **{metrics_summary['median']['fold_auroc_mean']:.4f} ± {metrics_summary['median']['fold_auroc_std']:.4f}** | **{metrics_summary['three_tier']['fold_auroc_mean']:.4f} ± {metrics_summary['three_tier']['fold_auroc_std']:.4f}** | **{np.mean(fold_deltas):+.4f} ± {np.std(fold_deltas, ddof=1):.4f}** |
| **Pooled OOF** | **3,848** | **{metrics_summary['median']['oof_auroc']:.4f}** | **{metrics_summary['three_tier']['oof_auroc']:.4f}** | **{oof_delta:+.4f}** |

---

## 4. Subgroup Audits on Complete Out-of-Fold Cohort ($N = 3,848$)

### Age Groups

| Age Group | $N$ | Males | `median` AUROC | `three_tier` AUROC | Delta |
|---|---:|---:|---:|---:|---:|
| **20–29** | {metrics_summary['median']['subgroups']['age_band']['20-29']['n']} | {metrics_summary['median']['subgroups']['age_band']['20-29']['n_class_1']} | {metrics_summary['median']['subgroups']['age_band']['20-29']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['age_band']['20-29']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['age_band']['20-29']['auroc'] - metrics_summary['median']['subgroups']['age_band']['20-29']['auroc']:+.4f} |
| **30–39** | {metrics_summary['median']['subgroups']['age_band']['30-39']['n']} | {metrics_summary['median']['subgroups']['age_band']['30-39']['n_class_1']} | {metrics_summary['median']['subgroups']['age_band']['30-39']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['age_band']['30-39']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['age_band']['30-39']['auroc'] - metrics_summary['median']['subgroups']['age_band']['30-39']['auroc']:+.4f} |
| **40–49** | {metrics_summary['median']['subgroups']['age_band']['40-49']['n']} | {metrics_summary['median']['subgroups']['age_band']['40-49']['n_class_1']} | {metrics_summary['median']['subgroups']['age_band']['40-49']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['age_band']['40-49']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['age_band']['40-49']['auroc'] - metrics_summary['median']['subgroups']['age_band']['40-49']['auroc']:+.4f} |
| **50–59** | {metrics_summary['median']['subgroups']['age_band']['50-59']['n']} | {metrics_summary['median']['subgroups']['age_band']['50-59']['n_class_1']} | {metrics_summary['median']['subgroups']['age_band']['50-59']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['age_band']['50-59']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['age_band']['50-59']['auroc'] - metrics_summary['median']['subgroups']['age_band']['50-59']['auroc']:+.4f} |
| **60–69** | {metrics_summary['median']['subgroups']['age_band']['60-69']['n']} | {metrics_summary['median']['subgroups']['age_band']['60-69']['n_class_1']} | {metrics_summary['median']['subgroups']['age_band']['60-69']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['age_band']['60-69']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['age_band']['60-69']['auroc'] - metrics_summary['median']['subgroups']['age_band']['60-69']['auroc']:+.4f} |
| **70+** | {metrics_summary['median']['subgroups']['age_band']['70+']['n']} | {metrics_summary['median']['subgroups']['age_band']['70+']['n_class_1']} | {metrics_summary['median']['subgroups']['age_band']['70+']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['age_band']['70+']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['age_band']['70+']['auroc'] - metrics_summary['median']['subgroups']['age_band']['70+']['auroc']:+.4f} |

### BMI Groups

| BMI Category | $N$ | Males | `median` AUROC | `three_tier` AUROC | Delta |
|---|---:|---:|---:|---:|---:|
| **Normal ($<25$)** | {metrics_summary['median']['subgroups']['bmi_group']['normal']['n']} | {metrics_summary['median']['subgroups']['bmi_group']['normal']['n_class_1']} | {metrics_summary['median']['subgroups']['bmi_group']['normal']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['bmi_group']['normal']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['bmi_group']['normal']['auroc'] - metrics_summary['median']['subgroups']['bmi_group']['normal']['auroc']:+.4f} |
| **Overweight ($25\\text{{--}}30$)** | {metrics_summary['median']['subgroups']['bmi_group']['overweight']['n']} | {metrics_summary['median']['subgroups']['bmi_group']['overweight']['n_class_1']} | {metrics_summary['median']['subgroups']['bmi_group']['overweight']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['bmi_group']['overweight']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['bmi_group']['overweight']['auroc'] - metrics_summary['median']['subgroups']['bmi_group']['overweight']['auroc']:+.4f} |
| **Obese ($\\ge 30$)** | {metrics_summary['median']['subgroups']['bmi_group']['obese']['n']} | {metrics_summary['median']['subgroups']['bmi_group']['obese']['n_class_1']} | {metrics_summary['median']['subgroups']['bmi_group']['obese']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['bmi_group']['obese']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['bmi_group']['obese']['auroc'] - metrics_summary['median']['subgroups']['bmi_group']['obese']['auroc']:+.4f} |
| **Underweight ($<18.5$)** | {metrics_summary['median']['subgroups']['bmi_group']['underweight']['n']} | {metrics_summary['median']['subgroups']['bmi_group']['underweight']['n_class_1']} | {metrics_summary['median']['subgroups']['bmi_group']['underweight']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['bmi_group']['underweight']['auroc']:.4f} | {metrics_summary['three_tier']['subgroups']['bmi_group']['underweight']['auroc'] - metrics_summary['median']['subgroups']['bmi_group']['underweight']['auroc']:+.4f} |

---

## 5. Artifact Verification & Invariants

1. **Partition Integrity**: Every participant in the $N = 3,848$ strict Garmin cohort was evaluated in exactly one validation fold. Zero missing entries, zero duplicate evaluations.
2. **Leakage Prevention**: All imputation medians, MultiRocket transform biases, variance thresholding, and standard scaling were fitted strictly on training participants within each fold.
3. **Matched Experimental Control**: `median` and `three_tier` arms used bit-identical partitions, identical B40 static summaries, and matched initial MultiRocket random seeds.
4. **Production Bundles**: Retrained on 100% of the cohort ($N = 3,848$) under optimal $\\alpha^*$ and saved to:
   - `experiments/k5_strict_garmin_interpolation/track_40days/median/production_bundle.joblib`
   - `experiments/k5_strict_garmin_interpolation/track_40days/three_tier/production_bundle.joblib`
"""
    with open(TRACK_A_DIR / "REPORT.md", "w") as f:
        f.write(report_content)
    with open(EXPERIMENT_DIR / "REPORT.md", "w") as f:
        f.write(report_content)

    # 9. Reproducibility
    repro = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "track": "40days",
        "cohort_size": COHORT_SIZE,
        "n_bins": N_BINS,
        "days": DAYS,
        "total_participant_days": COHORT_SIZE * DAYS,
        "seed_base": SEED_BASE,
        "alpha_grid": ALPHA_GRID,
        "selected_alpha": selected_alpha,
        "elapsed_seconds": round(time.time() - t_start_all, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 20 if sys.platform == "darwin" else 1024), 2),
        "metrics": {
            "median_oof_auroc": metrics_summary["median"]["oof_auroc"],
            "three_tier_oof_auroc": metrics_summary["three_tier"]["oof_auroc"],
            "paired_delta": oof_delta,
            "paired_bootstrap_95_ci": boot_res["diff_ci"],
        }
    }
    with open(EXPERIMENT_DIR / "reproducibility.json", "w") as f:
        json.dump(repro, f, indent=2)

    log(f"\n=== TRACK A COMPLETED SUCCESSFULLY IN {time.time() - t_start_all:.1f}s ===")


if __name__ == "__main__":
    main()
