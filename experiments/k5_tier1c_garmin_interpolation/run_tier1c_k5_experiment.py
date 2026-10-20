#!/usr/bin/env python
"""K=5 Disjoint Cross-Validation, Imputation Benchmark, and Production Retraining.

Cohort: Expanded Garmin Tier 1c (N = 4,410 single-source participants).
Contains:
  - Strict wear stratum (N = 3,848, Tier 0)
  - Intermittent wear & short-history delta stratum (N = 562)
    * 432 intermittent wear profile (max_adequate_weeks_lt_11)
    * 128 short history span (span_lt_13_weeks)
    * 2 low strict days total
Input History: First up to 40 adequate days (min(n_adequate, 40); 174,990 total participant-days).
Compares matched 'median' vs 'three_tier' imputation arms with 8 cores.
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
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "k5_tier1c_garmin_interpolation"
TRACK_DIR = EXPERIMENT_DIR / "track_40days"
VENDOR_DIR = REPO_ROOT / "experiments" / "temporal_garmin" / "cache" / "vendor"

sys.path.insert(0, str(VENDOR_DIR / "multirocket"))
from multirocket.multirocket import fit as mr_fit, transform as mr_transform

sys.path.insert(0, str(REPO_ROOT / "HEARTBEAT_LAB" / "src"))
from heartbeat_lab.data import STATS
from heartbeat_lab.model import pooled_features

MANIFEST_PATH = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "relaxed_garmin_manifest.csv"
PREPARED_STRICT = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "prepared"
PREPARED_DELTA = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "relaxed_cache" / "delta_562"

# Constants
COHORT_SIZE = 4410
N_BINS = 288
SEED_BASE = 20261015
ALPHA_GRID = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
MR_NUM_FEATURES_PER_TRANS = 1250
MR_WIDTH = 9408
CHUNK_DAYS = 64


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


@njit(cache=True)
def _nb_seed(s: int):
    np.random.seed(s)


def comp_seed(alloc: int, comp: int) -> int:
    return int(np.random.SeedSequence([SEED_BASE, 3, alloc, comp]).generate_state(1, dtype=np.uint32)[0])


# ============================================================================
# 1. Imputation Engine
# ============================================================================
def precompute_customer_overlays(customer_hr_list: list[np.ndarray]) -> tuple[list[dict], dict]:
    """Precompute gap overlays and exposure counts across all customer days."""
    customer_overlays = []
    total_missing_bins = 0
    tier1_linear_bins = 0
    tier2_pchip_bins = 0
    tier2_fb_bins = 0
    total_runs = 0

    for hr_user in customer_hr_list:
        R, L = hr_user.shape
        mask = np.isfinite(hr_user)
        miss = ~mask
        p = np.zeros((R, L + 2), dtype=bool)
        p[:, 1:-1] = miss
        d = np.diff(p.astype(np.int8), axis=1)
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1) - 1
        rows = (starts // (L + 1)).astype(np.int64)
        sr = (starts % (L + 1)).astype(np.int32)
        er = (ends % (L + 1)).astype(np.int32)
        rlen = (er - sr + 1).astype(np.int32)

        interior = (sr >= 1) & (er <= L - 2)
        cand_pchip = interior & (rlen >= 2) & (rlen <= 12) & (sr >= 2) & (er <= L - 3)
        has_ctx = np.zeros(rlen.size, dtype=bool)
        cp_idx = np.flatnonzero(cand_pchip)
        if cp_idx.size:
            r_c = rows[cp_idx]
            s_c = sr[cp_idx]
            e_c = er[cp_idx]
            ctx_ok = (mask[r_c, s_c - 2] &
                      mask[r_c, s_c - 1] &
                      mask[r_c, e_c + 1] &
                      mask[r_c, e_c + 2])
            has_ctx[cp_idx] = ctx_ok

        # Exposure accumulations
        n_miss = int(miss.sum())
        n_l1 = int(rlen[interior & (rlen == 1)].sum())
        n_pc = int(rlen[has_ctx].sum())
        n_fb = int(rlen[interior & (rlen >= 2) & (rlen <= 12) & (~has_ctx)].sum())

        total_missing_bins += n_miss
        tier1_linear_bins += n_l1
        tier2_pchip_bins += n_pc
        tier2_fb_bins += n_fb
        total_runs += int(rlen.size)

        pos_list, val_list = [], []
        # L=1 linear
        l1_idx = np.flatnonzero(interior & (rlen == 1))
        if l1_idx.size:
            r1, s1 = rows[l1_idx], sr[l1_idx]
            pos_list.append(r1 * L + s1)
            val_list.append(0.5 * (hr_user[r1, s1 - 1] + hr_user[r1, s1 + 1]))

        # PCHIP
        pc_runs = np.flatnonzero(has_ctx)
        if pc_runs.size:
            for Lg in range(2, 13):
                ii = pc_runs[rlen[pc_runs] == Lg]
                if ii.size == 0:
                    continue
                r_ii, s_ii, e_ii = rows[ii], sr[ii], er[ii]
                y = np.stack([hr_user[r_ii, s_ii - 2], hr_user[r_ii, s_ii - 1],
                              hr_user[r_ii, e_ii + 1], hr_user[r_ii, e_ii + 2]], axis=1).astype(np.float64)
                x_pts = np.array([0.0, 1.0, Lg + 2.0, Lg + 3.0])
                pb = PchipInterpolator(x_pts, y.T, extrapolate=False)
                vals = pb(np.arange(2.0, 2.0 + Lg, dtype=np.float64)).T.astype(np.float32)
                p_L = (r_ii[:, None] * L + s_ii[:, None] + np.arange(Lg)[None, :]).ravel()
                pos_list.append(p_L)
                val_list.append(vals.ravel())

        # Fallback linear
        fb_runs = np.flatnonzero(interior & (rlen >= 2) & (rlen <= 12) & (~has_ctx))
        if fb_runs.size:
            p_fb_list, v_fb_list = [], []
            for idx in fb_runs:
                r_f, s_f, e_f, L_f = rows[idx], sr[idx], er[idx], rlen[idx]
                xL = hr_user[r_f, s_f - 1]
                xR = hr_user[r_f, e_f + 1]
                j_steps = np.arange(1, L_f + 1, dtype=np.float32)
                p_fb_list.append(r_f * L + s_f + np.arange(L_f))
                v_fb_list.append(xL + (j_steps / (L_f + 1.0)) * (xR - xL))
            pos_list.append(np.concatenate(p_fb_list))
            val_list.append(np.concatenate(v_fb_list))

        if pos_list:
            c_pos = np.concatenate(pos_list)
            c_val = np.concatenate(val_list)
        else:
            c_pos = np.empty(0, dtype=np.int64)
            c_val = np.empty(0, dtype=np.float32)

        customer_overlays.append({"pos": c_pos, "val": c_val})

    tier3_median_bins = total_missing_bins - (tier1_linear_bins + tier2_pchip_bins + tier2_fb_bins)
    exposure = {
        "total_missing_bins": total_missing_bins,
        "tier1_linear_bins": tier1_linear_bins,
        "tier1_linear_pct": float(tier1_linear_bins / total_missing_bins * 100) if total_missing_bins else 0.0,
        "tier2_pchip_bins": tier2_pchip_bins,
        "tier2_pchip_pct": float(tier2_pchip_bins / total_missing_bins * 100) if total_missing_bins else 0.0,
        "tier2_fallback_bins": tier2_fb_bins,
        "tier2_fallback_pct": float(tier2_fb_bins / total_missing_bins * 100) if total_missing_bins else 0.0,
        "tier3_median_bins": tier3_median_bins,
        "tier3_median_pct": float(tier3_median_bins / total_missing_bins * 100) if total_missing_bins else 0.0,
        "total_missing_runs": total_runs,
    }
    return customer_overlays, exposure


def compute_training_template(customer_hr_list: list[np.ndarray], train_indices: np.ndarray) -> np.ndarray:
    """Compute 288-bin clock median template strictly on training participants."""
    train_hrs = [customer_hr_list[i] for i in train_indices]
    all_train = np.concatenate(train_hrs, axis=0)
    overall_med = float(np.nanmedian(all_train)) if np.isfinite(all_train).any() else 75.0
    with np.errstate(invalid="ignore", all="ignore"):
        clock_med = np.nanmedian(all_train, axis=0).astype(np.float32)
    clock_med = np.where(np.isfinite(clock_med), clock_med, overall_med)
    return clock_med


def get_customer_imputed_hr(hr_user: np.ndarray, template: np.ndarray, overlay_dict: dict, arm: str) -> np.ndarray:
    """Return imputed HR array for a single customer under specified arm."""
    mask = np.isfinite(hr_user)
    R, L = hr_user.shape
    hr_med = np.where(mask, hr_user, template[None, :]).astype(np.float32)
    if arm == "median":
        return hr_med

    hr_tier = hr_med.copy()
    if overlay_dict["pos"].size:
        flat = hr_tier.ravel()
        flat[overlay_dict["pos"]] = overlay_dict["val"]
        hr_tier = flat.reshape(R, L)
    return hr_tier


# ============================================================================
# 2. Balanced 5-Fold Partitioning (OR-Tools CP-SAT)
# ============================================================================
def solve_tier1c_5folds(manifest_df: pd.DataFrame, seed: int = SEED_BASE) -> pd.DataFrame:
    """Solve balanced 5-way partition for 4,410 participants with exact targets."""
    n = len(manifest_df)
    k = 5
    log(f"Solving balanced 5-way partition for {n} Tier 1c participants using OR-Tools CP-SAT (seed {seed})...")

    model = cp_model.CpModel()
    x = {(i, j): model.new_bool_var(f"x_{i}_{j}") for i in range(n) for j in range(k)}

    # Exactly 1 fold per participant
    for i in range(n):
        model.add(sum(x[i, j] for j in range(k)) == 1)

    # Equal fold size: exactly 882 each
    target_size = n // k
    for j in range(k):
        model.add(sum(x[i, j] for i in range(n)) == target_size)

    # Exact salutation target: 1,594 class 0 -> [319, 319, 319, 319, 318]
    sal10_idx = np.flatnonzero(manifest_df["salutation"].astype(int) == 10).tolist()
    target_sal10 = [319, 319, 319, 319, 318]
    for j in range(k):
        model.add(sum(x[i, j] for i in sal10_idx) == target_sal10[j])

    # Exact delta target: 562 delta -> [113, 113, 112, 112, 112]
    delta_idx = np.flatnonzero(manifest_df["delta_1c_0"] == True).tolist()
    target_delta = [113, 113, 112, 112, 112]
    for j in range(k):
        model.add(sum(x[i, j] for i in delta_idx) == target_delta[j])

    # Age and BMI deviations
    devs = []
    for col in ["age_group", "bmi_group"]:
        groups = manifest_df.groupby(col, sort=True).indices
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

    out_df = manifest_df.copy()
    out_df["fold"] = fold_assign
    return out_df


# ============================================================================
# 3. MultiRocket Pooling Across Variable Day Histories
# ============================================================================
def transform_and_pool_mr(customer_hr_imputed: list[np.ndarray], customer_days_list: list[pd.DataFrame],
                          customer_static_list: list[np.ndarray], mr_base, mr_diff, mr_transform_fn) -> np.ndarray:
    """Transform each customer's days through MultiRocket and pool into 19,121 features."""
    n = len(customer_hr_imputed)
    feature_matrix = np.empty((n, 19121), dtype=np.float32)

    for i in range(n):
        hr_u = customer_hr_imputed[i].astype(np.float64)
        days_u = customer_days_list[i]
        static_u = customer_static_list[i]

        chunks = []
        for start in range(0, len(hr_u), CHUNK_DAYS):
            batch = hr_u[start:start + CHUNK_DAYS]
            f = mr_transform_fn(batch, np.diff(batch, axis=1), mr_base, mr_diff, n_features_per_kernel=4)
            chunks.append(np.nan_to_num(f).astype(np.float32))
        trans = np.concatenate(chunks, axis=0)  # (D_i, 9408)

        # Between-day mean and SD
        mean_feat = trans.mean(axis=0, dtype=np.float64)
        sd_feat = trans.std(axis=0, ddof=1, dtype=np.float64) if len(days_u) > 1 else np.zeros(MR_WIDTH, dtype=np.float64)
        full_feat = np.r_[static_u, mean_feat, sd_feat].astype(np.float32)
        feature_matrix[i] = full_feat

    return feature_matrix


# ============================================================================
# 4. Statistical Inference
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


def compute_stratified_bootstrap_ci(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                                    n_boot: int = 2000, seed: int = SEED_BASE) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    auc_a = np.empty(n_boot, dtype=float)
    auc_b = np.empty(n_boot, dtype=float)
    diffs = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample_pos = rng.choice(pos_idx, n_pos, replace=True)
        sample_neg = rng.choice(neg_idx, n_neg, replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        y_b = y_true[sample_idx]

        a_score = roc_auc_score(y_b, scores_a[sample_idx])
        b_score = roc_auc_score(y_b, scores_b[sample_idx])
        auc_a[b] = a_score
        auc_b[b] = b_score
        diffs[b] = b_score - a_score

    return {
        "auc_a_ci": (float(np.percentile(auc_a, 2.5)), float(np.percentile(auc_a, 97.5))),
        "auc_b_ci": (float(np.percentile(auc_b, 2.5)), float(np.percentile(auc_b, 97.5))),
        "diff_ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        "diff_mean": float(np.mean(diffs)),
        "diff_se": float(np.std(diffs, ddof=1)),
    }


# ============================================================================
# 5. Main Runner
# ============================================================================
def main():
    t_start_all = time.time()
    log("=== STARTING EXPANDED GARMIN (TIER 1C, N=4,410) K=5 BENCHMARK ===")

    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    (TRACK_DIR / "median").mkdir(exist_ok=True)
    (TRACK_DIR / "three_tier").mkdir(exist_ok=True)

    # 1. Load Manifest & Customers
    log("Loading relaxed Garmin manifest and customer caches...")
    manifest = pd.read_csv(MANIFEST_PATH)
    t1c_df = manifest[manifest["tier_1c"] == True].sort_values("user_id").reset_index(drop=True)
    assert len(t1c_df) == COHORT_SIZE

    c_strict = pd.read_csv(PREPARED_STRICT / "customers.csv").set_index("customer_id")
    c_delta = pd.read_csv(PREPARED_DELTA / "customers.csv").set_index("customer_id")

    customer_hr_list = []
    customer_days_list = []
    customer_static_list = []
    total_days = 0

    t0_load = time.time()
    for cid in t1c_df["customer_id"]:
        if cid in c_strict.index:
            path = PREPARED_STRICT / c_strict.loc[cid, "file"]
        else:
            path = PREPARED_DELTA / c_delta.loc[cid, "file"]
        with np.load(path) as npz:
            hr_c = npz["hr"][:40].copy()
            days_c = pd.DataFrame(npz["stats"][:40], columns=STATS)
            days_c["date"] = pd.to_datetime(npz["date"][:40], unit="D")
            static_c = pooled_features(days_c, hr_c, None)

            customer_hr_list.append(hr_c)
            customer_days_list.append(days_c)
            customer_static_list.append(static_c)
            total_days += len(hr_c)

    log(f"Loaded {len(customer_hr_list)} customer histories ({total_days} total days) in {time.time() - t0_load:.2f}s.")

    y_all = t1c_df["y"].to_numpy(dtype=int)
    salutation_all = t1c_df["salutation"].to_numpy(dtype=int)
    is_delta = t1c_df["delta_1c_0"].to_numpy(dtype=bool)

    # 2. Partitioning
    folds_path = EXPERIMENT_DIR / "folds_5fold_disjoint.parquet"
    if folds_path.exists():
        log(f"Loading existing 5-fold partition from {folds_path}...")
        df_folds = pd.read_parquet(folds_path)
    else:
        df_folds = solve_tier1c_5folds(t1c_df, seed=SEED_BASE)
        df_folds.to_parquet(folds_path, index=False)
        log(f"Saved 5-fold disjoint partition to {folds_path}.")

    fold_labels = df_folds["fold"].to_numpy()
    for k in range(5):
        assert (fold_labels == f"fold_{k}").sum() == 882

    # 3. Precompute Gap Overlays & Exposure
    log("Precomputing gap overlays across all 4,410 customer histories...")
    customer_overlays, exposure = precompute_customer_overlays(customer_hr_list)
    with open(TRACK_DIR / "imputation_exposure.json", "w") as f:
        json.dump(exposure, f, indent=2)
    log(f"Exposure audit: total missing bins={exposure['total_missing_bins']}, "
        f"L=1 linear={exposure['tier1_linear_bins']} ({exposure['tier1_linear_pct']:.2f}%), "
        f"2..12 PCHIP={exposure['tier2_pchip_bins']} ({exposure['tier2_pchip_pct']:.2f}%), "
        f"2..12 fallback={exposure['tier2_fallback_bins']} ({exposure['tier2_fallback_pct']:.2f}%), "
        f">12 median={exposure['tier3_median_bins']} ({exposure['tier3_median_pct']:.2f}%).")

    # 4. Cross-Validation Sweep Across 5 Folds
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
        template_k = compute_training_template(customer_hr_list, tr_idx)

        # Step 4b: Sample 4 training days per user for MultiRocket kernel fitting (matched seed)
        fit_rng = np.random.default_rng(comp_seed(fold_k, 0))
        sampled_days_tr = []
        for idx in tr_idx:
            n_d = len(customer_hr_list[idx])
            sampled_days_tr.append(fit_rng.choice(n_d, min(4, n_d), replace=False))

        for arm in arms:
            t_arm_0 = time.time()
            # Impute all customers under current arm and template
            hr_imputed = [get_customer_imputed_hr(customer_hr_list[i], template_k, customer_overlays[i], arm)
                          for i in range(COHORT_SIZE)]

            # Collect MultiRocket fitting days
            fit_slices = [hr_imputed[idx][sampled_days_tr[local_i]] for local_i, idx in enumerate(tr_idx)]
            X_fit = np.concatenate(fit_slices, axis=0).astype(np.float64)
            X1_fit = np.diff(X_fit, axis=1)

            _nb_seed(comp_seed(fold_k, 1))
            mr_base = mr_fit(X_fit, num_features=MR_NUM_FEATURES_PER_TRANS, max_dilations_per_kernel=32)
            _nb_seed(comp_seed(fold_k, 1) + 1)
            mr_diff = mr_fit(X1_fit, num_features=MR_NUM_FEATURES_PER_TRANS, max_dilations_per_kernel=32)

            # Feature transform across cohort
            X_feat = transform_and_pool_mr(hr_imputed, customer_days_list, customer_static_list,
                                           mr_base, mr_diff, mr_transform)

            # Hygiene preprocessing strictly on training
            imp = SimpleImputer(strategy="median").fit(X_feat[tr_idx])
            X_imp_tr = imp.transform(X_feat[tr_idx])
            X_imp_va = imp.transform(X_feat[va_idx])

            vt = VarianceThreshold(0.0).fit(X_imp_tr)
            X_vt_tr = vt.transform(X_imp_tr)
            X_vt_va = vt.transform(X_imp_va)

            scaler = StandardScaler().fit(X_vt_tr)
            X_sc_tr = scaler.transform(X_vt_tr)
            X_sc_va = scaler.transform(X_vt_va)

            # Sample weights
            sample_weights = np.where(y_tr == 1, len(y_tr) / (2.0 * np.sum(y_tr == 1)),
                                      len(y_tr) / (2.0 * np.sum(y_tr == 0)))

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

    # 5. Hyperparameter Selection
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
        alpha_df.to_csv(TRACK_DIR / arm / "alpha_grid_results.csv", index=False)
        best_idx = alpha_df["mean_val_auroc"].idxmax()
        best_a = float(alpha_df.loc[best_idx, "alpha"])
        best_auc = float(alpha_df.loc[best_idx, "mean_val_auroc"])
        selected_alpha[arm] = best_a
        log(f"Arm [{arm:10s}] Optimal α* = {best_a:g} (Mean Val AUROC = {best_auc:.4f} ± {alpha_df.loc[best_idx, 'std_val_auroc']:.4f})")

    # 6. Final OOF Evaluation Across Full Cohort & Strata
    log("\n=== COMPUTING FINAL OUT-OF-FOLD (OOF) METRICS (N=4,410) ===")
    oof_scores = {}
    metrics_summary = {}

    strict_idx = np.flatnonzero(~is_delta)
    delta_idx = np.flatnonzero(is_delta)
    intermittent_idx = np.flatnonzero(t1c_df["fail_reason"] == "max_adequate_weeks_lt_11")
    short_span_idx = np.flatnonzero(t1c_df["fail_reason"] == "span_lt_13_weeks")

    for arm in arms:
        a_star = selected_alpha[arm]
        scores = val_preds_per_alpha[arm][a_star]
        oof_scores[arm] = scores

        # Overall DeLong CI
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
            bacc = 0.5 * ((probs >= t)[y_all == 1].mean() + (probs < t)[y_all == 0].mean())
            if bacc > best_bacc:
                best_bacc, best_thresh = bacc, t

        preds_bin = (probs >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_all, preds_bin).ravel()

        # Stratified stratum scores
        auc_strict = float(roc_auc_score(y_all[strict_idx], scores[strict_idx]))
        auc_delta = float(roc_auc_score(y_all[delta_idx], scores[delta_idx]))
        auc_intermittent = float(roc_auc_score(y_all[intermittent_idx], scores[intermittent_idx]))
        auc_short_span = float(roc_auc_score(y_all[short_span_idx], scores[short_span_idx]))

        # Save OOF predictions CSV
        oof_df = df_folds[["customer_id", "user_id", "salutation", "delta_1c_0", "fail_reason", "age_group", "bmi_group", "fold"]].copy()
        oof_df["y_true"] = y_all
        oof_df["ridge_margin"] = scores
        oof_df["calibrated_prob"] = probs
        oof_df["pred_binary"] = preds_bin
        oof_df.to_csv(TRACK_DIR / arm / "oof_predictions.csv", index=False)

        metrics_summary[arm] = {
            "selected_alpha": a_star,
            "overall_oof_auroc": auc,
            "delong_se": auc_se,
            "delong_95_ci": delong_ci,
            "average_precision": ap,
            "brier_score": brier,
            "optimal_threshold": float(best_thresh),
            "balanced_accuracy": float(best_bacc),
            "sensitivity": float(tp / (tp + fn)),
            "specificity": float(tn / (tn + fp)),
            "accuracy": float((tp + tn) / len(y_all)),
            "fold_aurocs": val_aurocs[arm][a_star],
            "fold_auroc_mean": float(np.mean(val_aurocs[arm][a_star])),
            "fold_auroc_std": float(np.std(val_aurocs[arm][a_star], ddof=1)),
            "strata": {
                "strict_cohort_auroc": auc_strict,
                "delta_cohort_auroc": auc_delta,
                "intermittent_wear_auroc": auc_intermittent,
                "short_span_auroc": auc_short_span,
            }
        }

    # 2,000-draw paired bootstrap
    log("Computing 2,000-draw paired stratified bootstrap...")
    boot_res = compute_stratified_bootstrap_ci(y_all, oof_scores["median"], oof_scores["three_tier"], n_boot=2000)
    for arm in arms:
        metrics_summary[arm]["bootstrap_95_ci"] = boot_res["auc_a_ci"] if arm == "median" else boot_res["auc_b_ci"]
        with open(TRACK_DIR / arm / "metrics_summary.json", "w") as f:
            json.dump(metrics_summary[arm], f, indent=2)

    # Paired fold deltas
    a_med = selected_alpha["median"]
    a_tier = selected_alpha["three_tier"]
    fold_deltas = [tier - med for med, tier in zip(val_aurocs["median"][a_med], val_aurocs["three_tier"][a_tier])]
    oof_delta = metrics_summary["three_tier"]["overall_oof_auroc"] - metrics_summary["median"]["overall_oof_auroc"]

    comp_df = pd.DataFrame({
        "split": [f"fold_{k}" for k in range(5)] + ["mean_fold", "overall_oof", "strict_stratum", "delta_stratum", "intermittent_wear", "short_span"],
        "median_auroc": val_aurocs["median"][a_med] + [
            metrics_summary["median"]["fold_auroc_mean"],
            metrics_summary["median"]["overall_oof_auroc"],
            metrics_summary["median"]["strata"]["strict_cohort_auroc"],
            metrics_summary["median"]["strata"]["delta_cohort_auroc"],
            metrics_summary["median"]["strata"]["intermittent_wear_auroc"],
            metrics_summary["median"]["strata"]["short_span_auroc"],
        ],
        "three_tier_auroc": val_aurocs["three_tier"][a_tier] + [
            metrics_summary["three_tier"]["fold_auroc_mean"],
            metrics_summary["three_tier"]["overall_oof_auroc"],
            metrics_summary["three_tier"]["strata"]["strict_cohort_auroc"],
            metrics_summary["three_tier"]["strata"]["delta_cohort_auroc"],
            metrics_summary["three_tier"]["strata"]["intermittent_wear_auroc"],
            metrics_summary["three_tier"]["strata"]["short_span_auroc"],
        ],
    })
    comp_df["delta_three_tier_minus_median"] = comp_df["three_tier_auroc"] - comp_df["median_auroc"]
    comp_df.to_csv(TRACK_DIR / "imputation_comparison.csv", index=False)

    log("\n=== FINAL RESULTS SUMMARY (TIER 1C - 4,410 USERS) ===")
    log(f"Median Arm:     Overall OOF AUROC = {metrics_summary['median']['overall_oof_auroc']:.4f} "
        f"(Strict={metrics_summary['median']['strata']['strict_cohort_auroc']:.4f}, "
        f"Delta={metrics_summary['median']['strata']['delta_cohort_auroc']:.4f}, "
        f"Intermittent={metrics_summary['median']['strata']['intermittent_wear_auroc']:.4f}, "
        f"ShortSpan={metrics_summary['median']['strata']['short_span_auroc']:.4f})")
    log(f"Three-Tier Arm: Overall OOF AUROC = {metrics_summary['three_tier']['overall_oof_auroc']:.4f} "
        f"(Strict={metrics_summary['three_tier']['strata']['strict_cohort_auroc']:.4f}, "
        f"Delta={metrics_summary['three_tier']['strata']['delta_cohort_auroc']:.4f}, "
        f"Intermittent={metrics_summary['three_tier']['strata']['intermittent_wear_auroc']:.4f}, "
        f"ShortSpan={metrics_summary['three_tier']['strata']['short_span_auroc']:.4f})")
    log(f"Overall Paired Delta: {oof_delta:+.4f} (Bootstrap 95% CI: [{boot_res['diff_ci'][0]:+.4f}, {boot_res['diff_ci'][1]:+.4f}])")
    log(f"Delta Stratum Difference: {metrics_summary['three_tier']['strata']['delta_cohort_auroc'] - metrics_summary['median']['strata']['delta_cohort_auroc']:+.4f}")
    log(f"Intermittent Wear Difference: {metrics_summary['three_tier']['strata']['intermittent_wear_auroc'] - metrics_summary['median']['strata']['intermittent_wear_auroc']:+.4f}")

    # 7. Retrain Production Bundles on 100% of Cohort (N = 4,410)
    log("\n=== FULL-COHORT RETRAINING ON 100% OF COHORT (N=4,410) ===")
    template_full = compute_training_template(customer_hr_list, np.arange(COHORT_SIZE))
    fit_rng_full = np.random.default_rng(comp_seed(999, 0))
    sampled_days_full = [fit_rng_full.choice(len(customer_hr_list[i]), min(4, len(customer_hr_list[i])), replace=False)
                         for i in range(COHORT_SIZE)]
    full_sample_weights = np.where(y_all == 1, COHORT_SIZE / (2.0 * np.sum(y_all == 1)),
                                   COHORT_SIZE / (2.0 * np.sum(y_all == 0)))

    for arm in arms:
        t_retrain_0 = time.time()
        log(f"Retraining production bundle for [{arm}] with α*={selected_alpha[arm]}...")
        hr_full = [get_customer_imputed_hr(customer_hr_list[i], template_full, customer_overlays[i], arm)
                   for i in range(COHORT_SIZE)]

        fit_slices = [hr_full[i][sampled_days_full[i]] for i in range(COHORT_SIZE)]
        X_fit = np.concatenate(fit_slices, axis=0).astype(np.float64)
        X1_fit = np.diff(X_fit, axis=1)

        _nb_seed(comp_seed(999, 1))
        mr_base_f = mr_fit(X_fit, num_features=MR_NUM_FEATURES_PER_TRANS, max_dilations_per_kernel=32)
        _nb_seed(comp_seed(999, 1) + 1)
        mr_diff_f = mr_fit(X1_fit, num_features=MR_NUM_FEATURES_PER_TRANS, max_dilations_per_kernel=32)

        X_feat_full = transform_and_pool_mr(hr_full, customer_days_list, customer_static_list,
                                            mr_base_f, mr_diff_f, mr_transform)

        imp_f = SimpleImputer(strategy="median").fit(X_feat_full)
        X_imp_f = imp_f.transform(X_feat_full)

        vt_f = VarianceThreshold(0.0).fit(X_imp_f)
        X_vt_f = vt_f.transform(X_imp_f)

        scaler_f = StandardScaler().fit(X_vt_f)
        X_sc_f = scaler_f.transform(X_vt_f)

        clf_f = Ridge(alpha=float(selected_alpha[arm]), fit_intercept=True)
        clf_f.fit(X_sc_f, y_all, sample_weight=full_sample_weights)

        bundle = {
            "arm": arm,
            "track": "40days_tier1c",
            "cohort_size": COHORT_SIZE,
            "alpha": selected_alpha[arm],
            "template": template_full,
            "mr_base": mr_base_f,
            "mr_diff": mr_diff_f,
            "imputer": imp_f,
            "variance_threshold": vt_f,
            "scaler": scaler_f,
            "classifier": clf_f,
            "oof_auroc": metrics_summary[arm]["overall_oof_auroc"],
            "strict_auroc": metrics_summary[arm]["strata"]["strict_cohort_auroc"],
            "delta_auroc": metrics_summary[arm]["strata"]["delta_cohort_auroc"],
        }
        bundle_path = TRACK_DIR / arm / "production_bundle.joblib"
        joblib.dump(bundle, bundle_path, compress=3)
        log(f"Saved production bundle to {bundle_path} ({time.time() - t_retrain_0:.1f}s).")

    # 8. Markdown Report
    log("\nGenerating comprehensive report (REPORT.md)...")
    report_content = f"""# Expanded Garmin Cohort (Tier 1c, N=4,410) Benchmark Report

**Date**: {time.strftime('%Y-%m-%d')}  
**Cohort**: Expanded Garmin Single-Source Cohort ($N = 4,410$, Tier 1c)  
**Composition**:
- **Strict Wear Stratum**: $N = 3,848$ participants (Tier 0)
- **Added Delta Stratum**: $N = 562$ participants who failed the strict longitudinal gate:
  * **Intermittent Wear Failure**: $N = 432$ participants (`max_adequate_weeks_lt_11`)
  * **Short History Span**: $N = 128$ participants (`span_lt_13_weeks`)
  * **Low Strict Days Total**: $N = 2$ participants
**Input History**: First up to 40 adequate days (`min(n_adequate, 40)`; $174,990$ total participant-days)  
**Evaluation**: $K=5$ disjoint stratified cross-validation (OR-Tools CP-SAT) with exact stratum balancing ($N_{{\\text{{untested}}}} = 0$).

---

## 1. Executive Summary & Paired Comparison

| Metric / Stratum | `median` (Clock Medians) | `three_tier` (Linear + PCHIP + Medians) | Paired Difference (`three_tier − median`) |
|---|---:|---:|---:|
| **Optimal Regularization $\\alpha^*$** | `{selected_alpha['median']:g}` | `{selected_alpha['three_tier']:g}` | — |
| **Overall Pooled OOF AUROC ($N = 4,410$)** | **{metrics_summary['median']['overall_oof_auroc']:.4f}** | **{metrics_summary['three_tier']['overall_oof_auroc']:.4f}** | **{oof_delta:+.4f}** |
| **DeLong 95% Confidence Interval** | `[{metrics_summary['median']['delong_95_ci'][0]:.4f}, {metrics_summary['median']['delong_95_ci'][1]:.4f}]` | `[{metrics_summary['three_tier']['delong_95_ci'][0]:.4f}, {metrics_summary['three_tier']['delong_95_ci'][1]:.4f}]` | — |
| **Stratified Bootstrap 95% CI (2,000 draws)** | `[{metrics_summary['median']['bootstrap_95_ci'][0]:.4f}, {metrics_summary['median']['bootstrap_95_ci'][1]:.4f}]` | `[{metrics_summary['three_tier']['bootstrap_95_ci'][0]:.4f}, {metrics_summary['three_tier']['bootstrap_95_ci'][1]:.4f}]` | **`[{boot_res['diff_ci'][0]:+.4f}, {boot_res['diff_ci'][1]:+.4f}]`** |
| **Strict Wear Stratum AUROC ($N = 3,848$)** | **{metrics_summary['median']['strata']['strict_cohort_auroc']:.4f}** | **{metrics_summary['three_tier']['strata']['strict_cohort_auroc']:.4f}** | **{metrics_summary['three_tier']['strata']['strict_cohort_auroc'] - metrics_summary['median']['strata']['strict_cohort_auroc']:+.4f}** |
| **Delta Stratum AUROC ($N = 562$)** | **{metrics_summary['median']['strata']['delta_cohort_auroc']:.4f}** | **{metrics_summary['three_tier']['strata']['delta_cohort_auroc']:.4f}** | **{metrics_summary['three_tier']['strata']['delta_cohort_auroc'] - metrics_summary['median']['strata']['delta_cohort_auroc']:+.4f}** |
| **Intermittent Wear Subgroup ($N = 432$)** | **{metrics_summary['median']['strata']['intermittent_wear_auroc']:.4f}** | **{metrics_summary['three_tier']['strata']['intermittent_wear_auroc']:.4f}** | **{metrics_summary['three_tier']['strata']['intermittent_wear_auroc'] - metrics_summary['median']['strata']['intermittent_wear_auroc']:+.4f}** |
| **Short Span Subgroup ($N = 128$)** | **{metrics_summary['median']['strata']['short_span_auroc']:.4f}** | **{metrics_summary['three_tier']['strata']['short_span_auroc']:.4f}** | **{metrics_summary['three_tier']['strata']['short_span_auroc'] - metrics_summary['median']['strata']['short_span_auroc']:+.4f}** |
| **Average Precision (AUPRC)** | `{metrics_summary['median']['average_precision']:.4f}` | `{metrics_summary['three_tier']['average_precision']:.4f}` | `{metrics_summary['three_tier']['average_precision'] - metrics_summary['median']['average_precision']:+.4f}` |
| **Brier Score** | `{metrics_summary['median']['brier_score']:.4f}` | `{metrics_summary['three_tier']['brier_score']:.4f}` | `{metrics_summary['three_tier']['brier_score'] - metrics_summary['median']['brier_score']:+.4f}` |
| **Balanced Accuracy (at $\\tau^*$)** | `{metrics_summary['median']['balanced_accuracy']:.4f}` | `{metrics_summary['three_tier']['balanced_accuracy']:.4f}` | `{metrics_summary['three_tier']['balanced_accuracy'] - metrics_summary['median']['balanced_accuracy']:+.4f}` |
| **Fold AUROC (Mean $\\pm$ SD)** | `{metrics_summary['median']['fold_auroc_mean']:.4f} ± {metrics_summary['median']['fold_auroc_std']:.4f}` | `{metrics_summary['three_tier']['fold_auroc_mean']:.4f} ± {metrics_summary['three_tier']['fold_auroc_std']:.4f}` | `{np.mean(fold_deltas):+.4f} ± {np.std(fold_deltas, ddof=1):.4f}` |

---

## 2. Imputation Exposure Audit on Expanded Cohort

Of the **{exposure['total_missing_bins']:,}** total missing bins across all 174,990 participant-days ({exposure['total_missing_runs']:,} maximal missing runs):
* **Tier 1 (Linear $\\le 5\\text{{ min}}$)**: {exposure['tier1_linear_bins']:,} bins (**{exposure['tier1_linear_pct']:.2f}%**).
* **Tier 2 (PCHIP $10\\text{{--}}60\\text{{ min}}$)**: {exposure['tier2_pchip_bins']:,} bins (**{exposure['tier2_pchip_pct']:.2f}%**).
* **Tier 2 (Linear Fallback $10\\text{{--}}60\\text{{ min}}$)**: {exposure['tier2_fallback_bins']:,} bins (**{exposure['tier2_fallback_pct']:.2f}%**).
* **Tier 3 (Diurnal Clock Median $>60\\text{{ min}}$ / Boundary)**: {exposure['tier3_median_bins']:,} bins (**{exposure['tier3_median_pct']:.2f}%**).

---

## 3. Fold-by-Fold AUROC Breakdown

| Split | $N_{{\\text{{val}}}}$ | `median` ($\\alpha^*={selected_alpha['median']:g}$) | `three_tier` ($\\alpha^*={selected_alpha['three_tier']:g}$) | Delta (`three_tier − median`) |
|---|---:|---:|---:|---:|
| **Fold 0** | 882 | {val_aurocs['median'][a_med][0]:.4f} | {val_aurocs['three_tier'][a_tier][0]:.4f} | {fold_deltas[0]:+.4f} |
| **Fold 1** | 882 | {val_aurocs['median'][a_med][1]:.4f} | {val_aurocs['three_tier'][a_tier][1]:.4f} | {fold_deltas[1]:+.4f} |
| **Fold 2** | 882 | {val_aurocs['median'][a_med][2]:.4f} | {val_aurocs['three_tier'][a_tier][2]:.4f} | {fold_deltas[2]:+.4f} |
| **Fold 3** | 882 | {val_aurocs['median'][a_med][3]:.4f} | {val_aurocs['three_tier'][a_tier][3]:.4f} | {fold_deltas[3]:+.4f} |
| **Fold 4** | 882 | {val_aurocs['median'][a_med][4]:.4f} | {val_aurocs['three_tier'][a_tier][4]:.4f} | {fold_deltas[4]:+.4f} |
| **Mean $\\pm$ SD** | — | **{metrics_summary['median']['fold_auroc_mean']:.4f} ± {metrics_summary['median']['fold_auroc_std']:.4f}** | **{metrics_summary['three_tier']['fold_auroc_mean']:.4f} ± {metrics_summary['three_tier']['fold_auroc_std']:.4f}** | **{np.mean(fold_deltas):+.4f} ± {np.std(fold_deltas, ddof=1):.4f}** |
| **Overall OOF** | **4,410** | **{metrics_summary['median']['overall_oof_auroc']:.4f}** | **{metrics_summary['three_tier']['overall_oof_auroc']:.4f}** | **{oof_delta:+.4f}** |

---

## 4. Key Scientific Insights

1. **Impact of Adding Intermittent & Short Wearers**:
   - In the strict cohort ($N = 3,848$), OOF AUROC was **0.8484**.
   - In the expanded cohort ($N = 4,410$), overall OOF AUROC is **{metrics_summary['median']['overall_oof_auroc']:.4f}**.
   - Within the test sets, the strict stratum scores **{metrics_summary['median']['strata']['strict_cohort_auroc']:.4f}**, while the added delta stratum scores **{metrics_summary['median']['strata']['delta_cohort_auroc']:.4f}**.
   - Specifically, participants with intermittent wear (`max_adequate_weeks_lt_11`) score **{metrics_summary['median']['strata']['intermittent_wear_auroc']:.4f}**, and those with short histories (`span_lt_13_weeks`) score **{metrics_summary['median']['strata']['short_span_auroc']:.4f}**.
2. **Imputation Comparison in Intermittent Wear**:
   - In the intermittent wear subgroup ($N = 432$), the paired `three_tier − median` difference is **{metrics_summary['three_tier']['strata']['intermittent_wear_auroc'] - metrics_summary['median']['strata']['intermittent_wear_auroc']:+.4f}**.
   - In the short history subgroup ($N = 128$), the difference is **{metrics_summary['three_tier']['strata']['short_span_auroc'] - metrics_summary['median']['strata']['short_span_auroc']:+.4f}**.
3. **Reproducibility & Production Models**:
   - Production bundles retrained on 100% of the cohort ($N = 4,410$) are saved to:
     * `experiments/k5_tier1c_garmin_interpolation/track_40days/median/production_bundle.joblib`
     * `experiments/k5_tier1c_garmin_interpolation/track_40days/three_tier/production_bundle.joblib`
"""
    with open(TRACK_DIR / "REPORT.md", "w") as f:
        f.write(report_content)
    with open(EXPERIMENT_DIR / "REPORT.md", "w") as f:
        f.write(report_content)

    repro = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cohort": "tier_1c",
        "cohort_size": COHORT_SIZE,
        "strict_size": int((~is_delta).sum()),
        "delta_size": int(is_delta.sum()),
        "total_participant_days": total_days,
        "seed_base": SEED_BASE,
        "alpha_grid": ALPHA_GRID,
        "selected_alpha": selected_alpha,
        "elapsed_seconds": round(time.time() - t_start_all, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 20 if sys.platform == "darwin" else 1024), 2),
        "metrics": {
            "median_overall_auroc": metrics_summary["median"]["overall_oof_auroc"],
            "three_tier_overall_auroc": metrics_summary["three_tier"]["overall_oof_auroc"],
            "paired_delta": oof_delta,
            "paired_bootstrap_95_ci": boot_res["diff_ci"],
            "median_strict_stratum": metrics_summary["median"]["strata"]["strict_cohort_auroc"],
            "three_tier_strict_stratum": metrics_summary["three_tier"]["strata"]["strict_cohort_auroc"],
            "median_delta_stratum": metrics_summary["median"]["strata"]["delta_cohort_auroc"],
            "three_tier_delta_stratum": metrics_summary["three_tier"]["strata"]["delta_cohort_auroc"],
        }
    }
    with open(EXPERIMENT_DIR / "reproducibility.json", "w") as f:
        json.dump(repro, f, indent=2)

    log(f"\n=== TIER 1C BENCHMARK COMPLETED SUCCESSFULLY IN {time.time() - t_start_all:.1f}s ===")


if __name__ == "__main__":
    main()
