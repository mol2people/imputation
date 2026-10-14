"""Step 4: Run MultiRocket vs Summary benchmark on Apple Watch pilot cohort (N=1,000).

Evaluates across 5 OR-Tools balanced 70/15/15 allocations:
  1. MultiRocket 5-minute resolution (288 bins/day, clock-median fill)
  2. MultiRocket 15-minute resolution (96 bins/day, clock-median fill)
  3. Static HR Summary baseline (daily means, medians, SDs, day/night contrasts)
All imputations, MultiRocket biases, scaling, and Ridge models are fitted strictly
on training participants inside each allocation fold.
"""
import os
import sys
import time
import json
from pathlib import Path

# Configure low CPU profile (2 threads) to avoid competing with running Garmin job
os.environ["NUMBA_NUM_THREADS"] = os.environ.get("APPLE_THREADS", "2")
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["OMP_NUM_THREADS"] = os.environ.get("APPLE_THREADS", "2")
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("APPLE_THREADS", "2")
os.environ["MKL_NUM_THREADS"] = os.environ.get("APPLE_THREADS", "2")

import numba
numba.set_num_threads(int(os.environ.get("APPLE_THREADS", 2)))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[3]
PILOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "HEARTBEAT_LAB" / "src"))
from heartbeat_lab.data import STATS
from heartbeat_lab.model import reference_functions, seed_numba

ALLOC_PATH = PILOT_DIR / "artifacts" / "pilot_5_allocations.parquet"
CACHE_5M_DIR = PILOT_DIR / "artifacts" / "cache_5min"
CACHE_15M_DIR = PILOT_DIR / "artifacts" / "cache_15min"
OUTPUT_DIR = PILOT_DIR / "artifacts"


def load_cache(cache_dir, customer_ids):
    cust_data = {}
    static_feats = {}
    for cid in customer_ids:
        path = cache_dir / f"customer_{cid}.npz"
        with np.load(path) as npz:
            date_series = pd.to_datetime(npz["date"], unit="D")
            days = pd.DataFrame(npz["stats"], columns=STATS)
            days["date"] = date_series
            hr = npz["hr"].copy()
            cust_data[cid] = (days, hr)
            
            # Static summaries
            weekend = days.date.dt.dayofweek.ge(5)
            contrast = days.loc[~weekend, "mean"].mean() - days.loc[weekend, "mean"].mean()
            static = np.r_[
                days[STATS].mean(),
                days[STATS].std(ddof=1) if len(days) > 1 else np.zeros(len(STATS)),
                contrast,
                np.isfinite(hr).mean(axis=0)
            ].astype(np.float32)
            static_feats[cid] = static
    return cust_data, static_feats


def fit_train_fill(train_cids, cust_data, n_bins):
    train_all = np.concatenate([cust_data[cid][1] for cid in train_cids], axis=0)
    fill = np.nanmedian(train_all, axis=0).astype(np.float32)
    if np.isnan(fill).any():
        fallback = float(np.nanmedian(train_all))
        fill = np.where(np.isnan(fill), fallback, fill)
    return fill


def transform_customer(static, days, hr, fill_vec, mr_base, mr_diff, mr_transform):
    x = np.where(np.isfinite(hr), hr, fill_vec).astype(np.float64)
    chunks = []
    for start in range(0, len(x), 64):
        batch = x[start:start+64]
        f = mr_transform(batch, np.diff(batch, axis=1), mr_base, mr_diff, n_features_per_kernel=4)
        chunks.append(np.nan_to_num(f).astype(np.float32))
    transformed = np.concatenate(chunks)
    sd = transformed.std(axis=0, ddof=1, dtype=np.float64) if len(days) > 1 else np.zeros(transformed.shape[1], dtype=np.float64)
    return np.r_[static, transformed.mean(axis=0, dtype=np.float64), sd].astype(np.float32)


def evaluate_arm(alloc_idx, train_cids, val_cids, test_cids, y_tr, y_va, y_te, cust_data, static_feats, mr_fit, mr_transform, n_bins, alpha=1000.0):
    t0 = time.time()
    # 1. Fit clock median fill
    fill_vec = fit_train_fill(train_cids, cust_data, n_bins)
    
    # 2. Fit MultiRocket biases on 4 sample days per train customer
    seed = 20261014 + alloc_idx * 10
    rng = np.random.default_rng(seed)
    sample_days = []
    for cid in train_cids:
        _, hr = cust_data[cid]
        n_days = len(hr)
        pick = rng.choice(n_days, min(4, n_days), replace=False)
        sample_days.append(hr[pick])
    sample_x = np.concatenate(sample_days)
    sample_x = np.where(np.isfinite(sample_x), sample_x, fill_vec).astype(np.float64)
    
    seed_numba(seed)
    mr_base = mr_fit(sample_x, num_features=1250, max_dilations_per_kernel=32)
    seed_numba(seed + 1)
    mr_diff = mr_fit(np.diff(sample_x, axis=1), num_features=1250, max_dilations_per_kernel=32)
    
    # 3. Transform features
    X_tr = np.stack([transform_customer(static_feats[cid], cust_data[cid][0], cust_data[cid][1], fill_vec, mr_base, mr_diff, mr_transform) for cid in train_cids])
    X_va = np.stack([transform_customer(static_feats[cid], cust_data[cid][0], cust_data[cid][1], fill_vec, mr_base, mr_diff, mr_transform) for cid in val_cids])
    X_te = np.stack([transform_customer(static_feats[cid], cust_data[cid][0], cust_data[cid][1], fill_vec, mr_base, mr_diff, mr_transform) for cid in test_cids])
    
    # 4. Fit Ridge
    pipe = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(0.0), StandardScaler(), Ridge(alpha=alpha))
    pipe.fit(X_tr, y_tr)
    
    val_auroc = roc_auc_score(y_va, pipe.predict(X_va))
    test_auroc = roc_auc_score(y_te, pipe.predict(X_te))
    dt = time.time() - t0
    return val_auroc, test_auroc, dt


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading allocations from {ALLOC_PATH}...")
    alloc_df = pd.read_parquet(ALLOC_PATH)
    all_cids = alloc_df["customer_id"].astype(str).tolist()
    
    print("\nLoading 5-minute cache (288 bins/day)...")
    cust_5m, static_5m = load_cache(CACHE_5M_DIR, all_cids)
    
    print("Loading 15-minute cache (96 bins/day)...")
    cust_15m, static_15m = load_cache(CACHE_15M_DIR, all_cids)
    
    mr_fit, mr_transform = reference_functions()
    
    results = []
    print("\nStarting benchmark across 5 OR-Tools allocations (1,000 Apple users)...")
    
    for a in range(5):
        col = f"alloc_{a}"
        tr_mask = alloc_df[col] == "train"
        va_mask = alloc_df[col] == "val"
        te_mask = alloc_df[col] == "test"
        
        tr_cids = alloc_df.loc[tr_mask, "customer_id"].astype(str).tolist()
        va_cids = alloc_df.loc[va_mask, "customer_id"].astype(str).tolist()
        te_cids = alloc_df.loc[te_mask, "customer_id"].astype(str).tolist()
        
        y_tr = alloc_df.loc[tr_mask, "y"].to_numpy(dtype=int)
        y_va = alloc_df.loc[va_mask, "y"].to_numpy(dtype=int)
        y_te = alloc_df.loc[te_mask, "y"].to_numpy(dtype=int)
        
        # Arm 1: MultiRocket 5-minute (288 bins)
        va_mr5, te_mr5, t_mr5 = evaluate_arm(a, tr_cids, va_cids, te_cids, y_tr, y_va, y_te, cust_5m, static_5m, mr_fit, mr_transform, n_bins=288)
        
        # Arm 2: MultiRocket 15-minute (96 bins)
        va_mr15, te_mr15, t_mr15 = evaluate_arm(a, tr_cids, va_cids, te_cids, y_tr, y_va, y_te, cust_15m, static_15m, mr_fit, mr_transform, n_bins=96)
        
        # Arm 3: Static Summary baseline (using 5m static features)
        t0_sm = time.time()
        X_tr_sm = np.stack([static_5m[cid] for cid in tr_cids])
        X_va_sm = np.stack([static_5m[cid] for cid in va_cids])
        X_te_sm = np.stack([static_5m[cid] for cid in te_cids])
        pipe_sm = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(0.0), StandardScaler(), Ridge(alpha=1000.0))
        pipe_sm.fit(X_tr_sm, y_tr)
        va_sm = roc_auc_score(y_va, pipe_sm.predict(X_va_sm))
        te_sm = roc_auc_score(y_te, pipe_sm.predict(X_te_sm))
        t_sm = time.time() - t0_sm
        
        print(f"Alloc {a} | MR 5m Test: {te_mr5:.4f} ({t_mr5:.1f}s) | MR 15m Test: {te_mr15:.4f} ({t_mr15:.1f}s) | Summary Test: {te_sm:.4f}")
        
        results.append({
            "alloc": a,
            "mr_5m_val_auroc": va_mr5,
            "mr_5m_test_auroc": te_mr5,
            "mr_5m_time_sec": t_mr5,
            "mr_15m_val_auroc": va_mr15,
            "mr_15m_test_auroc": te_mr15,
            "mr_15m_time_sec": t_mr15,
            "summary_val_auroc": va_sm,
            "summary_test_auroc": te_sm,
            "summary_time_sec": t_sm,
        })
        
    res_df = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "pilot_benchmark_results.csv"
    res_df.to_csv(out_csv, index=False)
    
    print("\n================ FINAL APPLE PILOT BENCHMARK (5 Allocations) ================")
    print(f"MultiRocket 5-min (288 bins):  Test AUROC = {res_df['mr_5m_test_auroc'].mean():.4f} +/- {res_df['mr_5m_test_auroc'].std(ddof=1):.4f} (Val: {res_df['mr_5m_val_auroc'].mean():.4f})")
    print(f"MultiRocket 15-min (96 bins):  Test AUROC = {res_df['mr_15m_test_auroc'].mean():.4f} +/- {res_df['mr_15m_test_auroc'].std(ddof=1):.4f} (Val: {res_df['mr_15m_val_auroc'].mean():.4f})")
    print(f"Summary Baseline:              Test AUROC = {res_df['summary_test_auroc'].mean():.4f} +/- {res_df['summary_test_auroc'].std(ddof=1):.4f} (Val: {res_df['summary_val_auroc'].mean():.4f})")
    print(f"Paired MultiRocket Gain (5m vs Summary):  {res_df['mr_5m_test_auroc'].mean() - res_df['summary_test_auroc'].mean():+.4f}")
    print(f"Resolution Gain (15m vs 5m):              {res_df['mr_15m_test_auroc'].mean() - res_df['mr_5m_test_auroc'].mean():+.4f}")
    print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    main()
