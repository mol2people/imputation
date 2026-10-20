#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
01_run_downstream_benchmark.py

Phase 2 Benchmark Pipeline for monthly_periodicity_downstream:
Evaluates the predictive downstream utility of monthly infradian periodicity
features (R_obs) against static circadian + MultiRocket representations across 8 heads:
  1. H_clock (MultiRocket + static only)
  2. B_clock = H_clock + Q + age (primary controlled baseline)
  3. B_clock + R_obs (primary augmented arm)
  4. B_linear = H_linear + Q + age (filling sensitivity baseline)
  5. B_linear + R_obs (filling sensitivity augmentation)
  6. B_clock without age
  7. B_clock without age + R_obs
  8. Age only

Follows PLAN.md, READINESS.md, and HANDOFF.md specifications:
- Strict train-only fitting for templates, MultiRocket biases/dilations, imputers, and scalers.
- Bit-identical frozen H representations across paired baseline and augmented arms.
- Ridge regression with alpha in {100, 500, 1000, 5000, 10000, 100000} selected on validation AUROC.
- 2,000 participant bootstrap resamples stratified by age_group x salutation for inference.
- Saves results to experiments/monthly_periodicity_downstream/results/downstream_benchmark_metrics.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Thread caps for deterministic performance
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMBA_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ensure repository root and heartbeat_lab are in python path
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(REPO_ROOT / "HEARTBEAT_LAB" / "src"))
sys.path.insert(0, str(REPO_ROOT))

from heartbeat_lab.data import STATS
from heartbeat_lab.model import reference_functions, seed_numba
from experiments.apple_fill_and_mask_by_recording.scripts.transforms import (
    fit_fill_templates,
    interpolate_short_gaps,
)

# Constants & Paths
EXP_DIR = SCRIPT_PATH.parents[1]
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOC_PATH = REPO_ROOT / "experiments/apple_cohort_scaling/artifacts/apple_10_allocations.parquet"
ELIG_PATH = REPO_ROOT / "experiments/female_infradian_cycles_v2/results/window_eligibility.parquet"
CACHE_DIR = REPO_ROOT / "experiments/apple_cohort_scaling/artifacts/cache_5min"

ALPHAS = [100.0, 500.0, 1000.0, 5000.0, 10000.0, 100000.0]
AGE_LEVELS = ["Young Adults", "Middle Ager", "Elderly"]


# -----------------------------------------------------------------------------
# Recording Nuisance & Infradian Fit Helpers (for Q descriptors)
# -----------------------------------------------------------------------------

def build_recording_nuisance_design(dates: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Builds nuisance design for recording outcomes:
      - Intercept
      - Elapsed days linear trend
      - Weekday indicators (Tue-Sun, Mon reference)
    Excludes clock coverage fractions.
    """
    dates = np.asarray(dates, dtype=np.int64)
    n = len(dates)
    col_names = [
        "intercept", "elapsed_days",
        "weekday_tue", "weekday_wed", "weekday_thu",
        "weekday_fri", "weekday_sat", "weekday_sun",
    ]
    cols = [np.ones(n, dtype=np.float64), (dates - dates[0]).astype(np.float64)]
    dow = (dates + 3) % 7
    for d in range(1, 7):
        cols.append((dow == d).astype(np.float64))
    X_raw = np.column_stack(cols)
    
    retained_cols, retained_names = [], []
    Q = np.zeros((n, 0), dtype=np.float64)
    for idx, name in enumerate(col_names):
        v = X_raw[:, idx].copy()
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        if Q.shape[1] > 0:
            v_perp = v - Q @ (Q.T @ v)
            perp_norm = np.linalg.norm(v_perp)
            if perp_norm / v_norm > 1e-4:
                Q = np.column_stack([Q, v_perp / perp_norm])
                retained_cols.append(v)
                retained_names.append(name)
        else:
            Q = (v / v_norm)[:, np.newaxis]
            retained_cols.append(v)
            retained_names.append(name)
    X_retained = np.column_stack(retained_cols)
    proj_resid = np.eye(n, dtype=np.float64) - (Q @ Q.T)
    return X_retained, retained_names, proj_resid


def fit_recording_infradian(
    y: np.ndarray,
    dates: np.ndarray,
    periods: np.ndarray,
    proj_resid: np.ndarray,
) -> Tuple[float, float, float, int, int]:
    """
    Fits harmonic cosinor regression across candidate periods on detrended recording outcome y.
    Returns (A, S, best_T, fit_avail, zero_var).
    """
    total_ss = float(np.sum((y - np.mean(y)) ** 2))
    if np.std(y) < 1e-6:
        return np.nan, np.nan, np.nan, 0, 1
    e_0 = proj_resid @ y
    rss_0 = float(np.sum(e_0 ** 2))
    if rss_0 <= 1e-12 * max(1.0, total_ss) or not np.isfinite(rss_0):
        return np.nan, np.nan, np.nan, 0, 1
        
    t_elapsed = (dates - dates[0]).astype(np.float64)
    best_S = -1.0
    best_T, best_A = np.nan, np.nan
    for T in periods:
        omega = 2.0 * np.pi / T
        C_T = np.column_stack([np.cos(omega * t_elapsed), np.sin(omega * t_elapsed)])
        C_tilde = proj_resid @ C_T
        CTC = C_tilde.T @ C_tilde
        det = CTC[0, 0] * CTC[1, 1] - CTC[0, 1] * CTC[1, 0]
        if det < 1e-12:
            CTC[0, 0] += 1e-8
            CTC[1, 1] += 1e-8
            det = CTC[0, 0] * CTC[1, 1] - CTC[0, 1] * CTC[1, 0]
        inv_CTC = np.array([[CTC[1, 1], -CTC[0, 1]], [-CTC[1, 0], CTC[0, 0]]], dtype=np.float64) / det
        M_T = C_tilde @ inv_CTC
        beta = e_0 @ M_T
        b_c, b_s = beta[0], beta[1]
        delta_rss = b_c ** 2 * CTC[0, 0] + 2.0 * b_c * b_s * CTC[0, 1] + b_s ** 2 * CTC[1, 1]
        s_T = float(np.clip(delta_rss / rss_0, 0.0, 1.0))
        if s_T > best_S:
            best_S = s_T
            best_T = float(T)
            best_A = float(np.sqrt(b_c ** 2 + b_s ** 2))
    return best_A, best_S, best_T, 1, 0


# -----------------------------------------------------------------------------
# Participant Feature Extraction Worker
# -----------------------------------------------------------------------------

def process_participant_data(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts 90-date slices, static circadian features, and Q controls for a participant.
    """
    cid = row["customer_id"]
    sel_dates = np.array(row["w1_selected_dates"])
    start_date = row["w1_start_date"]
    
    cache_path = CACHE_DIR / f"customer_{cid}.npz"
    with np.load(cache_path) as npz:
        dates = npz["date"]
        hr = npz["hr"]
        stats = npz["stats"]
        
    date_to_idx = {d: i for i, d in enumerate(dates)}
    indices = [date_to_idx[d] for d in sel_dates]
    
    hr_sel = hr[indices] # (90, 288)
    stats_sel = stats[indices] # (90, 8)
    
    # 1. Static features B_L(u): 305 cols (8 stats mean, 8 stats SD, 1 weekday contrast, 288 mask means)
    days_df = pd.DataFrame(stats_sel, columns=STATS)
    days_df["date"] = pd.to_datetime(sel_dates, unit="D")
    weekend = days_df["date"].dt.dayofweek.ge(5)
    if (~weekend).any() and weekend.any():
        contrast = days_df.loc[~weekend, "mean"].mean() - days_df.loc[weekend, "mean"].mean()
    else:
        contrast = 0.0
    if np.isnan(contrast):
        contrast = 0.0
        
    static_b_l = np.r_[
        days_df[STATS].mean().to_numpy(),
        days_df[STATS].std(ddof=1).to_numpy() if len(days_df) > 1 else np.zeros(len(STATS)),
        contrast,
        np.isfinite(hr_sel).mean(axis=0),
    ].astype(np.float32)
    
    # 2. Q controls: 22 cols
    valid_mask = np.isfinite(hr_sel)
    p10 = np.empty(len(sel_dates), dtype=np.float64)
    for i in range(len(sel_dates)):
        p10[i] = np.percentile(hr_sel[i, valid_mask[i]], 10)
        
    p10_mean = float(np.mean(p10))
    p10_sd = float(np.std(p10, ddof=1))
    t = (sel_dates - sel_dates[0]).astype(np.float64)
    A_lin = np.column_stack([np.ones(len(t)), t])
    beta_lin, _, _, _ = np.linalg.lstsq(A_lin, p10, rcond=None)
    p10_slope = float(beta_lin[1])
    resids = p10 - (beta_lin[0] + beta_lin[1] * t)
    p10_linear_resid_sd = float(np.sqrt(np.sum(resids ** 2) / (len(t) - 2)))
    
    clock_00_06 = np.mean(valid_mask[:, 0:72], axis=1)
    clock_06_12 = np.mean(valid_mask[:, 72:144], axis=1)
    clock_12_18 = np.mean(valid_mask[:, 144:216], axis=1)
    clock_18_24 = np.mean(valid_mask[:, 216:288], axis=1)
    
    cov_00_06_mean = float(np.mean(clock_00_06))
    cov_06_12_mean = float(np.mean(clock_06_12))
    cov_12_18_mean = float(np.mean(clock_12_18))
    cov_18_24_mean = float(np.mean(clock_18_24))
    
    date_span = float(sel_dates[-1] - sel_dates[0])
    gaps = [int(sel_dates[0] - start_date), int((start_date + 179) - sel_dates[-1])]
    gaps.extend((np.diff(sel_dates) - 1).tolist())
    longest_gap = float(max(gaps))
    
    periods = np.linspace(25.0, 35.0, 21)
    _, _, proj_resid = build_recording_nuisance_design(sel_dates)
    occ_daily = np.mean(valid_mask, axis=1)
    noct_mask = clock_00_06
    
    occ_A, occ_S, occ_T, occ_avail, occ_zvar = fit_recording_infradian(occ_daily, sel_dates, periods, proj_resid)
    noct_A, noct_S, noct_T, noct_avail, noct_zvar = fit_recording_infradian(noct_mask, sel_dates, periods, proj_resid)
    
    hr_fit_avail = 1 if np.isfinite(row["A_i"]) else 0
    hr_zero_var = 0 if np.isfinite(row["A_i"]) else 1
    
    q_vec = np.array([
        p10_mean, p10_sd, p10_slope, p10_linear_resid_sd,
        cov_00_06_mean, cov_06_12_mean, cov_12_18_mean, cov_18_24_mean,
        date_span, longest_gap,
        occ_A, occ_S, occ_T,
        noct_A, noct_S, noct_T,
        float(hr_fit_avail), float(hr_zero_var),
        float(occ_avail), float(occ_zvar),
        float(noct_avail), float(noct_zvar),
    ], dtype=np.float32)
    
    r_vec = np.array([row["A_i"], row["S_i"], row["best_T"]], dtype=np.float32)
    
    return {
        "customer_id": cid,
        "static": static_b_l,
        "q": q_vec,
        "r": r_vec,
        "hr_raw": hr_sel, # (90, 288)
    }


# -----------------------------------------------------------------------------
# MultiRocket Batched Sequence Transformation & Pooling
# -----------------------------------------------------------------------------

def transform_sequence_batched(
    cids: List[str],
    all_x: np.ndarray,
    mr_base,
    mr_diff,
    mr_transform,
    batch_size: int = 2048,
    days_per_user: int = 90,
) -> np.ndarray:
    """
    Batch-transforms sequence arrays via MultiRocket and pools mean + SD across each user's 90 days.
    Returns (N_users, 18816) float32.
    """
    n_users = len(cids)
    n_total_days = len(all_x)
    assert n_total_days == n_users * days_per_user, f"Expected {n_users * days_per_user} days, got {n_total_days}"
    
    # Probe output dimension
    bx0 = np.ascontiguousarray(all_x[:2], dtype=np.float64)
    bd0 = np.diff(bx0, axis=1)
    f_sample = mr_transform(bx0, bd0, mr_base, mr_diff, n_features_per_kernel=4)
    d_feat = f_sample.shape[1] # 9408
    
    day_to_cust_idx = np.repeat(np.arange(n_users), days_per_user)
    sum_mat = np.zeros((n_users, d_feat), dtype=np.float64)
    sum_sq_mat = np.zeros((n_users, d_feat), dtype=np.float64)
    
    for s in range(0, n_total_days, batch_size):
        e = min(s + batch_size, n_total_days)
        bx = np.ascontiguousarray(all_x[s:e], dtype=np.float64)
        bd = np.diff(bx, axis=1)
        f = np.nan_to_num(mr_transform(bx, bd, mr_base, mr_diff, n_features_per_kernel=4)).astype(np.float64)
        batch_cust = day_to_cust_idx[s:e]
        
        u_cust, u_idx = np.unique(batch_cust, return_index=True)
        u_idx = np.r_[u_idx, len(batch_cust)]
        for j, u in enumerate(u_cust):
            f_slice = f[u_idx[j] : u_idx[j + 1]]
            sum_mat[u] += np.sum(f_slice, axis=0)
            sum_sq_mat[u] += np.sum(f_slice ** 2, axis=0)
            
    # Pool across days
    n = float(days_per_user)
    mean_mat = sum_mat / n
    var_mat = (sum_sq_mat - (sum_mat ** 2) / n) / (n - 1.0)
    sd_mat = np.sqrt(np.maximum(0.0, var_mat))
    
    pooled = np.c_[mean_mat, sd_mat].astype(np.float32)
    return pooled


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run downstream benchmark for monthly periodicity.")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test with 200 participants.")
    args = parser.parse_args()
    
    t_start_total = time.time()
    print("=" * 80)
    print("MONTHLY PERIODICITY DOWNSTREAM PREDICTIVE BENCHMARK")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"Python executable: {sys.executable}")
    print(f"Smoke test mode: {args.smoke}")
    print("=" * 80, flush=True)
    
    # -------------------------------------------------------------------------
    # 1. Cohort Merging & Eligibility Filtering
    # -------------------------------------------------------------------------
    print("\n--- 1. Loading Cohort Allocations & Window Eligibility ---")
    alloc_df = pd.read_parquet(ALLOC_PATH)
    elig_df = pd.read_parquet(ELIG_PATH)
    
    alloc_df["customer_id"] = alloc_df["customer_id"].astype(str)
    elig_df["customer_id"] = elig_df["customer_id"].astype(str)
    
    merged_df = pd.merge(
        alloc_df[["customer_id", "user_id", "salutation", "y", "age_group", "bmi_group", "alloc_0"]],
        elig_df[[
            "customer_id", "eligible_w1", "w1_start_date", "w1_actual_span",
            "w1_longest_gap", "w1_selected_dates", "S_i", "best_T", "A_i"
        ]],
        on="customer_id",
        how="inner",
    )
    
    # Filter to eligible participants
    cohort_df = merged_df[merged_df["eligible_w1"] == True].copy()
    cohort_df = cohort_df.sort_values("customer_id").reset_index(drop=True)
    
    if args.smoke:
        print("[SMOKE TEST] Subsampling to first 300 participants (balanced across splits)...")
        sample_cids = []
        for s in ["train", "val", "test"]:
            s_cids = cohort_df[cohort_df["alloc_0"] == s]["customer_id"].head(100).tolist()
            sample_cids.extend(s_cids)
        cohort_df = cohort_df[cohort_df["customer_id"].isin(sample_cids)].reset_index(drop=True)
        
    n_cohort = len(cohort_df)
    print(f"Total eligible cohort: N={n_cohort}")
    print("Allocation 0 split counts:")
    print(cohort_df["alloc_0"].value_counts().sort_index())
    print("Age group breakdown:")
    print(cohort_df["age_group"].value_counts())
    print("Age group x Salutation:")
    print(pd.crosstab(cohort_df["age_group"], cohort_df["salutation"]))
    
    # -------------------------------------------------------------------------
    # 2. Extract Static Features, Q Controls, and Raw Sequences
    # -------------------------------------------------------------------------
    print("\n--- 2. Extracting Static Features, Q Controls, and Raw Sequences ---")
    t0 = time.time()
    records = cohort_df.to_dict("records")
    with ProcessPoolExecutor(max_workers=8) as ex:
        extracted = list(ex.map(process_participant_data, records))
    print(f"Extracted features for {len(extracted)} participants in {time.time() - t0:.2f}s")
    
    ordered_cids = [r["customer_id"] for r in extracted]
    static_mat = np.stack([r["static"] for r in extracted]) # (N, 305)
    q_mat = np.stack([r["q"] for r in extracted])           # (N, 22)
    r_mat = np.stack([r["r"] for r in extracted])           # (N, 3)
    hr_raw_mat = np.concatenate([r["hr_raw"] for r in extracted], axis=0) # (N*90, 288)
    
    # Age block: 3-level one-hot (Young Adults, Middle Ager, Elderly)
    age_dummies = pd.get_dummies(cohort_df["age_group"].astype(pd.CategoricalDtype(categories=AGE_LEVELS)), dtype=np.float32)
    age_mat = age_dummies.to_numpy() # (N, 3)
    
    # Split index arrays
    tr_mask = (cohort_df["alloc_0"] == "train").to_numpy()
    va_mask = (cohort_df["alloc_0"] == "val").to_numpy()
    te_mask = (cohort_df["alloc_0"] == "test").to_numpy()
    
    tr_idx = np.flatnonzero(tr_mask)
    va_idx = np.flatnonzero(va_mask)
    te_idx = np.flatnonzero(te_mask)
    
    y = cohort_df["y"].to_numpy().astype(np.float32)
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
    
    print(f"Static shape: {static_mat.shape}, Q shape: {q_mat.shape}, R shape: {r_mat.shape}, Age shape: {age_mat.shape}")
    print(f"Train N={len(tr_idx)}, Val N={len(va_idx)}, Test N={len(te_idx)}")
    
    # -------------------------------------------------------------------------
    # 3. Fit Fill Templates & Sample Training Days for MultiRocket
    # -------------------------------------------------------------------------
    print("\n--- 3. MultiRocket Fitting & Representation Extraction ---")
    # Train-only templates
    train_hr_list = [extracted[i]["hr_raw"] for i in tr_idx]
    fill_clock, fill_global = fit_fill_templates(train_hr_list)
    print(f"Fitted train-only fill_clock: shape={fill_clock.shape}, fill_global={fill_global:.2f}")
    
    # MultiRocket sampling seed
    msg_sample = b"20261020|apple|ALL|mr_day_sample"
    seed_day_sample = int.from_bytes(hashlib.sha256(msg_sample).digest()[:8], "little")
    rng_day_sample = np.random.default_rng(seed_day_sample)
    
    n_train = len(tr_idx)
    days_per_user = max(1, 4096 // n_train)
    sample_day_rows = []
    for i in tr_idx:
        hr_u = extracted[i]["hr_raw"] # (90, 288)
        pick = rng_day_sample.choice(len(hr_u), min(days_per_user, len(hr_u)), replace=False)
        sample_day_rows.append(hr_u[pick])
    sample_raw = np.concatenate(sample_day_rows, axis=0)
    if len(sample_raw) > 4096:
        sample_raw = sample_raw[rng_day_sample.choice(len(sample_raw), 4096, replace=False)]
    print(f"Sampled {len(sample_raw)} training days for MultiRocket parameter fitting")
    
    # MultiRocket branch seeds
    seed_mr_base = int.from_bytes(hashlib.sha256(b"20261020|apple|ALL|mr_base").digest()[:8], "little") % (2**31 - 1)
    seed_mr_diff = int.from_bytes(hashlib.sha256(b"20261020|apple|ALL|mr_diff").digest()[:8], "little") % (2**31 - 1)
    
    mr_fit, mr_transform = reference_functions()
    
    # ARM 1 & Baseline: H_clock
    print("Fitting MultiRocket parameters on H_clock...")
    sample_mask = np.isfinite(sample_raw).astype(np.uint8)
    sample_x_clock = np.where(sample_mask == 1, sample_raw, fill_clock).astype(np.float64)
    seed_numba(seed_mr_base)
    mr_base_clock = mr_fit(sample_x_clock, num_features=1250, max_dilations_per_kernel=32)
    seed_numba(seed_mr_diff)
    mr_diff_clock = mr_fit(np.diff(sample_x_clock, axis=1), num_features=1250, max_dilations_per_kernel=32)
    
    print("Transforming all participant days for H_clock...")
    all_mask = np.isfinite(hr_raw_mat).astype(np.uint8)
    all_x_clock = np.where(all_mask == 1, hr_raw_mat, fill_clock).astype(np.float64)
    t0_tr = time.time()
    mr_clock_pooled = transform_sequence_batched(
        ordered_cids, all_x_clock, mr_base_clock, mr_diff_clock, mr_transform, batch_size=2048, days_per_user=90
    )
    print(f"H_clock MultiRocket transform completed in {time.time() - t0_tr:.2f}s, shape={mr_clock_pooled.shape}")
    H_clock = np.c_[static_mat, mr_clock_pooled].astype(np.float32)
    
    # ARM 2: H_linear (Filling Sensitivity)
    print("Fitting MultiRocket parameters on H_linear...")
    sample_x_linear = interpolate_short_gaps(sample_raw, clock_template=fill_clock, resolution="5m", k_max=5).astype(np.float64)
    seed_numba(seed_mr_base)
    mr_base_linear = mr_fit(sample_x_linear, num_features=1250, max_dilations_per_kernel=32)
    seed_numba(seed_mr_diff)
    mr_diff_linear = mr_fit(np.diff(sample_x_linear, axis=1), num_features=1250, max_dilations_per_kernel=32)
    
    print("Transforming all participant days for H_linear...")
    all_x_linear = interpolate_short_gaps(hr_raw_mat, clock_template=fill_clock, resolution="5m", k_max=5).astype(np.float64)
    t0_tr = time.time()
    mr_linear_pooled = transform_sequence_batched(
        ordered_cids, all_x_linear, mr_base_linear, mr_diff_linear, mr_transform, batch_size=2048, days_per_user=90
    )
    print(f"H_linear MultiRocket transform completed in {time.time() - t0_tr:.2f}s, shape={mr_linear_pooled.shape}")
    H_linear = np.c_[static_mat, mr_linear_pooled].astype(np.float32)
    
    # -------------------------------------------------------------------------
    # 4. Strict Train-Only Preprocessing of Feature Blocks
    # -------------------------------------------------------------------------
    print("\n--- 4. Preprocessing Feature Blocks Strictly on Training Data ---")
    
    def preprocess_block(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        pipe = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(0.0), StandardScaler())
        X_tr_std = pipe.fit_transform(X[tr_idx])
        X_va_std = pipe.transform(X[va_idx])
        X_te_std = pipe.transform(X[te_idx])
        return X_tr_std, X_va_std, X_te_std, X_tr_std.shape[1]
        
    H_clock_tr, H_clock_va, H_clock_te, d_h_clock = preprocess_block(H_clock)
    H_linear_tr, H_linear_va, H_linear_te, d_h_linear = preprocess_block(H_linear)
    Q_tr, Q_va, Q_te, d_q = preprocess_block(q_mat)
    R_tr, R_va, R_te, d_r = preprocess_block(r_mat)
    age_tr, age_va, age_te, d_age = preprocess_block(age_mat)
    
    print(f"Preprocessed dimensions: H_clock={d_h_clock}, H_linear={d_h_linear}, Q={d_q}, R={d_r}, Age={d_age}")
    
    # -------------------------------------------------------------------------
    # 5. Assemble Eight Heads
    # -------------------------------------------------------------------------
    heads = {
        "H_clock": {
            "tr": H_clock_tr,
            "va": H_clock_va,
            "te": H_clock_te,
        },
        "B_clock": {
            "tr": np.c_[H_clock_tr, Q_tr, age_tr],
            "va": np.c_[H_clock_va, Q_va, age_va],
            "te": np.c_[H_clock_te, Q_te, age_te],
        },
        "B_clock_plus_R_obs": {
            "tr": np.c_[H_clock_tr, Q_tr, age_tr, R_tr],
            "va": np.c_[H_clock_va, Q_va, age_va, R_va],
            "te": np.c_[H_clock_te, Q_te, age_te, R_te],
        },
        "B_linear": {
            "tr": np.c_[H_linear_tr, Q_tr, age_tr],
            "va": np.c_[H_linear_va, Q_va, age_va],
            "te": np.c_[H_linear_te, Q_te, age_te],
        },
        "B_linear_plus_R_obs": {
            "tr": np.c_[H_linear_tr, Q_tr, age_tr, R_tr],
            "va": np.c_[H_linear_va, Q_va, age_va, R_va],
            "te": np.c_[H_linear_te, Q_te, age_te, R_te],
        },
        "B_clock_no_age": {
            "tr": np.c_[H_clock_tr, Q_tr],
            "va": np.c_[H_clock_va, Q_va],
            "te": np.c_[H_clock_te, Q_te],
        },
        "B_clock_no_age_plus_R_obs": {
            "tr": np.c_[H_clock_tr, Q_tr, R_tr],
            "va": np.c_[H_clock_va, Q_va, R_va],
            "te": np.c_[H_clock_te, Q_te, R_te],
        },
        "Age_only": {
            "tr": age_tr,
            "va": age_va,
            "te": age_te,
        },
    }
    
    # -------------------------------------------------------------------------
    # 6. Fit Ridge Regression Heads with Validation Alpha Selection
    # -------------------------------------------------------------------------
    print("\n--- 6. Fitting Ridge Regression Models & Alpha Selection ---")
    fitted_results = {}
    test_scores_dict = {}
    alpha_traces = []
    
    for head_name, data in heads.items():
        X_tr = data["tr"]
        X_va = data["va"]
        X_te = data["te"]
        
        va_aucs = {}
        models = {}
        for alpha in ALPHAS:
            clf = Ridge(alpha=alpha)
            clf.fit(X_tr, y_tr)
            s_va = clf.predict(X_va).astype(np.float64)
            va_auc = float(roc_auc_score(y_va, s_va))
            va_aucs[alpha] = va_auc
            models[alpha] = clf
            alpha_traces.append({
                "head": head_name,
                "alpha": alpha,
                "val_auroc": va_auc,
            })
            
        max_va_auc = max(va_aucs.values())
        # Break near-ties with np.isclose(atol=1e-12) favoring larger alpha
        best_alpha = max([a for a in ALPHAS if np.isclose(va_aucs[a], max_va_auc, atol=1e-12)])
        best_model = models[best_alpha]
        
        s_tr = best_model.predict(X_tr).astype(np.float64)
        s_va = best_model.predict(X_va).astype(np.float64)
        s_te = best_model.predict(X_te).astype(np.float64)
        
        tr_auc = float(roc_auc_score(y_tr, s_tr))
        va_auc = float(roc_auc_score(y_va, s_va))
        te_auc = float(roc_auc_score(y_te, s_te))
        te_ap = float(average_precision_score(y_te, s_te))
        
        test_scores_dict[head_name] = s_te
        fitted_results[head_name] = {
            "selected_alpha": best_alpha,
            "train_auroc": tr_auc,
            "val_auroc": va_auc,
            "test_auroc": te_auc,
            "test_ap": te_ap,
        }
        print(f"Head {head_name:28s}: Selected alpha={best_alpha:7.0f} | Val AUC={va_auc:.5f} | Test AUC={te_auc:.5f} | Test AP={te_ap:.5f}")
        
    alpha_trace_df = pd.DataFrame(alpha_traces)
    alpha_trace_df.to_csv(RESULTS_DIR / "alpha_selection_trace.csv", index=False)
    
    # Save test predictions
    test_cids = [ordered_cids[i] for i in te_idx]
    te_cohort = cohort_df.iloc[te_idx].copy().reset_index(drop=True)
    preds_dict = {
        "customer_id": test_cids,
        "salutation": te_cohort["salutation"].to_numpy(),
        "y": y_te,
        "age_group": te_cohort["age_group"].to_numpy(),
    }
    for h_name, s_te in test_scores_dict.items():
        preds_dict[f"score_{h_name}"] = s_te
    preds_df = pd.DataFrame(preds_dict)
    preds_df.to_parquet(RESULTS_DIR / "test_predictions.parquet", index=False)
    
    # -------------------------------------------------------------------------
    # 7. Within-Age & Training-Age-Weighted Test AUROC
    # -------------------------------------------------------------------------
    print("\n--- 7. Computing Within-Age and Training-Age-Weighted AUROCs ---")
    te_age = te_cohort["age_group"].to_numpy()
    tr_age = cohort_df.iloc[tr_idx]["age_group"].to_numpy()
    
    # Fixed weights from training age proportions
    n_tr_total = float(len(tr_idx))
    w_age = {grp: float(np.sum(tr_age == grp) / n_tr_total) for grp in AGE_LEVELS}
    print(f"Frozen training age weights: Young Adults={w_age['Young Adults']:.4f}, Middle Ager={w_age['Middle Ager']:.4f}, Elderly={w_age['Elderly']:.4f}")
    
    within_age_metrics = {}
    for h_name, s_te in test_scores_dict.items():
        metrics_grp = {}
        weighted_sum = 0.0
        for grp in AGE_LEVELS:
            mask_grp = (te_age == grp)
            y_grp = y_te[mask_grp]
            s_grp = s_te[mask_grp]
            if len(np.unique(y_grp)) > 1:
                auc_grp = float(roc_auc_score(y_grp, s_grp))
            else:
                auc_grp = np.nan
            metrics_grp[grp] = auc_grp
            if not np.isnan(auc_grp):
                weighted_sum += w_age[grp] * auc_grp
        metrics_grp["weighted"] = weighted_sum
        within_age_metrics[h_name] = metrics_grp
        print(f"Head {h_name:28s} | YA: {metrics_grp['Young Adults']:.5f} | MA: {metrics_grp['Middle Ager']:.5f} | Elderly: {metrics_grp['Elderly']:.5f} | Age-Weighted: {metrics_grp['weighted']:.5f}")
        
    # Primary increments on test set
    auc_b_clock = fitted_results["B_clock"]["test_auroc"]
    auc_b_clock_r = fitted_results["B_clock_plus_R_obs"]["test_auroc"]
    delta_clock_obs = auc_b_clock_r - auc_b_clock
    
    auc_b_linear = fitted_results["B_linear"]["test_auroc"]
    auc_b_linear_r = fitted_results["B_linear_plus_R_obs"]["test_auroc"]
    delta_linear_obs = auc_b_linear_r - auc_b_linear
    eta_obs = delta_linear_obs - delta_clock_obs
    
    auc_no_age = fitted_results["B_clock_no_age"]["test_auroc"]
    auc_no_age_r = fitted_results["B_clock_no_age_plus_R_obs"]["test_auroc"]
    delta_no_age_obs = auc_no_age_r - auc_no_age
    
    print("\nPrimary Test Increments:")
    print(f"  delta_clock  = AUROC(B_clock + R_obs) - AUROC(B_clock)   = {delta_clock_obs:+.6f}")
    print(f"  delta_linear = AUROC(B_linear + R_obs) - AUROC(B_linear) = {delta_linear_obs:+.6f}")
    print(f"  eta          = delta_linear - delta_clock                = {eta_obs:+.6f}")
    print(f"  delta_no_age = AUROC(B_no_age + R_obs) - AUROC(B_no_age) = {delta_no_age_obs:+.6f}")
    
    # -------------------------------------------------------------------------
    # 8. Bootstrap Inference (B=2,000 Stratified by Age x Salutation)
    # -------------------------------------------------------------------------
    print("\n--- 8. Running Stratified Participant Bootstrap Inference (B=2,000) ---")
    boot_msg = b"20261020|apple|ALL|evaluation_bootstrap"
    seed_boot = int.from_bytes(hashlib.sha256(boot_msg).digest()[:8], "little")
    rng_boot = np.random.default_rng(seed_boot)
    
    # Identify strata on test set
    strata_indices = []
    strata_labels = []
    for grp in AGE_LEVELS:
        for sal in [10, 20]:
            idx = np.flatnonzero((te_cohort["age_group"] == grp) & (te_cohort["salutation"] == sal))
            strata_indices.append(idx)
            strata_labels.append(f"{grp}|{sal}")
            print(f"  Test Stratum {grp:12s} x salutation {sal}: N={len(idx)}")
            
    B_REPS = 2000 if not args.smoke else 100
    boot_draws = {
        h: {"overall": [], "ya": [], "ma": [], "elderly": [], "weighted": []}
        for h in heads.keys()
    }
    boot_contrasts = {
        "delta_clock": [],
        "delta_linear": [],
        "eta": [],
        "delta_no_age": [],
    }
    
    t0_b = time.time()
    for b in range(B_REPS):
        sampled_pieces = [rng_boot.choice(idx_arr, size=len(idx_arr), replace=True) for idx_arr in strata_indices]
        b_idx = np.concatenate(sampled_pieces)
        y_b = y_te[b_idx]
        age_b = te_age[b_idx]
        
        b_aucs = {}
        for h_name, s_te in test_scores_dict.items():
            s_b = s_te[b_idx]
            auc_all = float(roc_auc_score(y_b, s_b))
            b_aucs[h_name] = auc_all
            boot_draws[h_name]["overall"].append(auc_all)
            
            # Within-age
            w_sum = 0.0
            for grp in AGE_LEVELS:
                mask_grp = (age_b == grp)
                auc_grp = float(roc_auc_score(y_b[mask_grp], s_b[mask_grp]))
                key_grp = "ya" if grp == "Young Adults" else ("ma" if grp == "Middle Ager" else "elderly")
                boot_draws[h_name][key_grp].append(auc_grp)
                w_sum += w_age[grp] * auc_grp
            boot_draws[h_name]["weighted"].append(w_sum)
            
        d_clock_b = b_aucs["B_clock_plus_R_obs"] - b_aucs["B_clock"]
        d_linear_b = b_aucs["B_linear_plus_R_obs"] - b_aucs["B_linear"]
        eta_b = d_linear_b - d_clock_b
        d_no_age_b = b_aucs["B_clock_no_age_plus_R_obs"] - b_aucs["B_clock_no_age"]
        
        boot_contrasts["delta_clock"].append(d_clock_b)
        boot_contrasts["delta_linear"].append(d_linear_b)
        boot_contrasts["eta"].append(eta_b)
        boot_contrasts["delta_no_age"].append(d_no_age_b)
        
    print(f"Completed {B_REPS} bootstrap replicates in {time.time() - t0_b:.2f}s")
    
    # -------------------------------------------------------------------------
    # 9. Format Deliverable Table & Save CSV
    # -------------------------------------------------------------------------
    print("\n--- 9. Compiling Final Benchmark Deliverables ---")
    
    metrics_rows = []
    # 1. Model Heads
    for h_name, res in fitted_results.items():
        draws_overall = np.array(boot_draws[h_name]["overall"])
        ci_95_pw = np.percentile(draws_overall, [2.5, 97.5])
        ci_97_5_bonf = np.percentile(draws_overall, [1.25, 98.75])
        
        metrics_rows.append({
            "category": "head",
            "name": h_name,
            "selected_alpha": res["selected_alpha"],
            "train_auroc": res["train_auroc"],
            "val_auroc": res["val_auroc"],
            "test_auroc": res["test_auroc"],
            "test_ap": res["test_ap"],
            "boot_mean": float(np.mean(draws_overall)),
            "boot_se": float(np.std(draws_overall, ddof=1)),
            "ci_lower_95_pointwise": float(ci_95_pw[0]),
            "ci_upper_95_pointwise": float(ci_95_pw[1]),
            "ci_lower_97_5_bonferroni": float(ci_97_5_bonf[0]),
            "ci_upper_97_5_bonferroni": float(ci_97_5_bonf[1]),
            "within_age_young_adults": within_age_metrics[h_name]["Young Adults"],
            "within_age_middle_ager": within_age_metrics[h_name]["Middle Ager"],
            "within_age_elderly": within_age_metrics[h_name]["Elderly"],
            "age_weighted_auroc": within_age_metrics[h_name]["weighted"],
        })
        
    # 2. Contrasts
    contrasts_meta = [
        ("delta_clock", delta_clock_obs, boot_contrasts["delta_clock"]),
        ("delta_linear", delta_linear_obs, boot_contrasts["delta_linear"]),
        ("eta", eta_obs, boot_contrasts["eta"]),
        ("delta_no_age", delta_no_age_obs, boot_contrasts["delta_no_age"]),
    ]
    for c_name, obs_val, c_draws in contrasts_meta:
        draws_arr = np.array(c_draws)
        ci_95_pw = np.percentile(draws_arr, [2.5, 97.5])
        ci_97_5_bonf = np.percentile(draws_arr, [1.25, 98.75])
        metrics_rows.append({
            "category": "contrast",
            "name": c_name,
            "selected_alpha": np.nan,
            "train_auroc": np.nan,
            "val_auroc": np.nan,
            "test_auroc": obs_val,
            "test_ap": np.nan,
            "boot_mean": float(np.mean(draws_arr)),
            "boot_se": float(np.std(draws_arr, ddof=1)),
            "ci_lower_95_pointwise": float(ci_95_pw[0]),
            "ci_upper_95_pointwise": float(ci_95_pw[1]),
            "ci_lower_97_5_bonferroni": float(ci_97_5_bonf[0]),
            "ci_upper_97_5_bonferroni": float(ci_97_5_bonf[1]),
            "within_age_young_adults": np.nan,
            "within_age_middle_ager": np.nan,
            "within_age_elderly": np.nan,
            "age_weighted_auroc": np.nan,
        })
        
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = RESULTS_DIR / "downstream_benchmark_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved downstream benchmark metrics to {metrics_path}")
    print(metrics_df[["category", "name", "selected_alpha", "test_auroc", "ci_lower_95_pointwise", "ci_upper_95_pointwise"]].to_string())
    
    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY in {time.time() - t_start_total:.2f}s")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
