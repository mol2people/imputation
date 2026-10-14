"""Step 4: Run MultiRocket temporal binning benchmark on Garmin cohort (N=1,000) using 8 cores.

Evaluates across 5 OR-Tools balanced 70/15/15 allocations:
  1. MultiRocket 1-minute resolution (1,440 bins/day, clock-median fill)
  2. MultiRocket 5-minute resolution (288 bins/day, clock-median fill)
  3. MultiRocket 15-minute resolution (96 bins/day, clock-median fill)
  4. Static HR Summary baseline
All imputations, MultiRocket biases, scaling, and Ridge regressions are fitted
strictly on training participants inside each allocation fold.
"""
import os
import sys
import time
import json
from pathlib import Path

# Configure threading to 8 cores as requested
os.environ["NUMBA_NUM_THREADS"] = "8"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import numba
numba.set_num_threads(8)

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

ALLOC_PATH = PILOT_DIR / "artifacts" / "garmin_binning_5_allocations.parquet"
CACHE_1M_DIR = PILOT_DIR / "artifacts" / "cache_1min"
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


def evaluate_arm(alloc_idx, train_cids, val_cids, test_cids, y_tr, y_va, y_te, test_is_delta, cust_data, static_feats, mr_fit, mr_transform, n_bins, alpha=1000.0):
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
    
    score_va = pipe.predict(X_va)
    score_te = pipe.predict(X_te)
    
    val_auroc = roc_auc_score(y_va, score_va)
    test_auroc = roc_auc_score(y_te, score_te)
    
    test_auroc_strict = roc_auc_score(y_te[~test_is_delta], score_te[~test_is_delta])
    test_auroc_delta = roc_auc_score(y_te[test_is_delta], score_te[test_is_delta]) if test_is_delta.sum() > 0 else float("nan")
    
    dt = time.time() - t0
    return {
        "val_auroc": val_auroc,
        "test_auroc": test_auroc,
        "test_strict_auroc": test_auroc_strict,
        "test_delta_auroc": test_auroc_delta,
        "time_sec": dt,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "garmin_binning_benchmark_results.csv"
    
    print(f"Active Numba threads: {numba.get_num_threads()}")
    print(f"Loading allocations from {ALLOC_PATH}...")
    alloc_df = pd.read_parquet(ALLOC_PATH)
    all_cids = alloc_df["customer_id"].astype(str).tolist()
    
    print("\nLoading 1-minute cache (1,440 bins/day)...")
    cust_1m, static_1m = load_cache(CACHE_1M_DIR, all_cids)
    
    print("Loading 5-minute cache (288 bins/day)...")
    cust_5m, static_5m = load_cache(CACHE_5M_DIR, all_cids)
    
    print("Loading 15-minute cache (96 bins/day)...")
    cust_15m, static_15m = load_cache(CACHE_15M_DIR, all_cids)
    
    mr_fit, mr_transform = reference_functions()
    
    results = []
    print("\nStarting Garmin temporal binning benchmark across 5 allocations on 8 cores...")
    
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
        
        test_is_delta = alloc_df.loc[te_mask, "delta_1c_0"].to_numpy(dtype=bool)
        
        # Arm 1: 1-minute (1440 bins)
        r_1m = evaluate_arm(a, tr_cids, va_cids, te_cids, y_tr, y_va, y_te, test_is_delta, cust_1m, static_1m, mr_fit, mr_transform, n_bins=1440)
        
        # Arm 2: 5-minute (288 bins)
        r_5m = evaluate_arm(a, tr_cids, va_cids, te_cids, y_tr, y_va, y_te, test_is_delta, cust_5m, static_5m, mr_fit, mr_transform, n_bins=288)
        
        # Arm 3: 15-minute (96 bins)
        r_15m = evaluate_arm(a, tr_cids, va_cids, te_cids, y_tr, y_va, y_te, test_is_delta, cust_15m, static_15m, mr_fit, mr_transform, n_bins=96)
        
        # Arm 4: Summary baseline
        t0_sm = time.time()
        X_tr_sm = np.stack([static_5m[cid] for cid in tr_cids])
        X_va_sm = np.stack([static_5m[cid] for cid in va_cids])
        X_te_sm = np.stack([static_5m[cid] for cid in te_cids])
        pipe_sm = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(0.0), StandardScaler(), Ridge(alpha=1000.0))
        pipe_sm.fit(X_tr_sm, y_tr)
        va_sm = roc_auc_score(y_va, pipe_sm.predict(X_va_sm))
        te_sm = roc_auc_score(y_te, pipe_sm.predict(X_te_sm))
        t_sm = time.time() - t0_sm
        
        print(f"Alloc {a} | 1m Test: {r_1m['test_auroc']:.4f} ({r_1m['time_sec']:.1f}s) | 5m Test: {r_5m['test_auroc']:.4f} ({r_5m['time_sec']:.1f}s) | 15m Test: {r_15m['test_auroc']:.4f} ({r_15m['time_sec']:.1f}s) | Summary: {te_sm:.4f}", flush=True)
        
        results.append({
            "alloc": a,
            "mr_1m_test": r_1m["test_auroc"],
            "mr_1m_strict": r_1m["test_strict_auroc"],
            "mr_1m_delta": r_1m["test_delta_auroc"],
            "mr_1m_time": r_1m["time_sec"],
            "mr_5m_test": r_5m["test_auroc"],
            "mr_5m_strict": r_5m["test_strict_auroc"],
            "mr_5m_delta": r_5m["test_delta_auroc"],
            "mr_5m_time": r_5m["time_sec"],
            "mr_15m_test": r_15m["test_auroc"],
            "mr_15m_strict": r_15m["test_strict_auroc"],
            "mr_15m_delta": r_15m["test_delta_auroc"],
            "mr_15m_time": r_15m["time_sec"],
            "summary_test": te_sm,
            "summary_time": t_sm,
        })
        pd.DataFrame(results).to_csv(out_csv, index=False)
        
    res_df = pd.DataFrame(results)
    print("\n================ FINAL GARMIN TEMPORAL BINNING BENCHMARK ================")
    print(f"MultiRocket 1-min (1,440 bins): Test AUROC = {res_df['mr_1m_test'].mean():.4f} +/- {res_df['mr_1m_test'].std(ddof=1):.4f} (Strict: {res_df['mr_1m_strict'].mean():.4f}, Delta: {res_df['mr_1m_delta'].mean():.4f}) | Mean Time: {res_df['mr_1m_time'].mean():.1f}s")
    print(f"MultiRocket 5-min (288 bins):   Test AUROC = {res_df['mr_5m_test'].mean():.4f} +/- {res_df['mr_5m_test'].std(ddof=1):.4f} (Strict: {res_df['mr_5m_strict'].mean():.4f}, Delta: {res_df['mr_5m_delta'].mean():.4f}) | Mean Time: {res_df['mr_5m_time'].mean():.1f}s")
    print(f"MultiRocket 15-min (96 bins):   Test AUROC = {res_df['mr_15m_test'].mean():.4f} +/- {res_df['mr_15m_test'].std(ddof=1):.4f} (Strict: {res_df['mr_15m_strict'].mean():.4f}, Delta: {res_df['mr_15m_delta'].mean():.4f}) | Mean Time: {res_df['mr_15m_time'].mean():.1f}s")
    print(f"Summary Baseline:               Test AUROC = {res_df['summary_test'].mean():.4f} +/- {res_df['summary_test'].std(ddof=1):.4f}")
    print(f"\nDelta (1-min vs 5-min baseline):  {res_df['mr_1m_test'].mean() - res_df['mr_5m_test'].mean():+.4f}")
    print(f"Delta (15-min vs 5-min baseline): {res_df['mr_15m_test'].mean() - res_df['mr_5m_test'].mean():+.4f}")
    print(f"Results saved to {out_csv}")


if __name__ == "__main__":
    main()
