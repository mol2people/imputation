"""Run 10-allocation 70/15/15 cross-validation on Tier 1c Garmin cohort (4,410 users).

All preprocessing, filling, MultiRocket bias quantiles, scaling, and Ridge regression
are fitted strictly on training participants inside each allocation.
Evaluates overall test AUROC, strict stratum AUROC, and wear-fragmented delta stratum AUROC.
"""
import os
import sys
import time
import json
from pathlib import Path

# Configure threading to 7 cores before numba/mkl/openblas initialization
os.environ["NUMBA_NUM_THREADS"] = "7"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["OMP_NUM_THREADS"] = "7"
os.environ["OPENBLAS_NUM_THREADS"] = "7"
os.environ["MKL_NUM_THREADS"] = "7"

import numba
numba.set_num_threads(7)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "HEARTBEAT_LAB" / "src"))
from heartbeat_lab.data import STATS
from heartbeat_lab.model import reference_functions, seed_numba, pooled_features

ALLOC_PATH = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "allocations_tier_1c" / "tier1c_10_allocations.parquet"
PREPARED_STRICT = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "prepared"
PREPARED_DELTA = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "relaxed_cache" / "delta_562"
OUTPUT_DIR = REPO_ROOT / "HEARTBEAT_LAB" / "artifacts" / "tier1c_cv_results"


def load_all_tier1c_data(alloc_df):
    print("Loading data for all 4,410 participants...")
    t0 = time.time()
    
    strict_cust = pd.read_csv(PREPARED_STRICT / "customers.csv", dtype={"customer_id": str}).set_index("customer_id")
    delta_cust = pd.read_csv(PREPARED_DELTA / "customers.csv", dtype={"customer_id": str}).set_index("customer_id")
    
    customers_data = {}
    static_features = {}
    for cid in alloc_df["customer_id"].astype(str):
        if cid in strict_cust.index:
            fname = strict_cust.loc[cid, "file"]
            path = PREPARED_STRICT / fname
        elif cid in delta_cust.index:
            fname = delta_cust.loc[cid, "file"]
            path = PREPARED_DELTA / fname
        else:
            raise KeyError(f"Customer {cid} not found in prepared caches!")
            
        with np.load(path) as npz:
            date_series = pd.to_datetime(npz["date"], unit="D")
            days = pd.DataFrame(npz["stats"], columns=STATS)
            days["date"] = date_series
            hr = npz["hr"].copy()
            customers_data[cid] = (days, hr)
            static_features[cid] = pooled_features(days, hr, None)
            
    print(f"Loaded {len(customers_data)} customer histories and computed static summaries in {time.time() - t0:.2f}s.")
    return customers_data, static_features


def fit_train_fill(train_cids, customers_data, filling="clock"):
    train_all = np.concatenate([customers_data[cid][1] for cid in train_cids], axis=0)
    if filling == "global":
        fallback = float(np.nanmedian(train_all))
        return np.full(288, fallback, dtype=np.float32)
    else:
        # Clock median across all train days per bin
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


def run_allocation(alloc_idx, alloc_df, customers_data, static_features, mr_fit, mr_transform, filling="clock", alpha=1000.0):
    t0_alloc = time.time()
    col = f"alloc_{alloc_idx}"
    train_mask = alloc_df[col] == "train"
    val_mask = alloc_df[col] == "val"
    test_mask = alloc_df[col] == "test"
    
    train_cids = alloc_df.loc[train_mask, "customer_id"].astype(str).tolist()
    val_cids = alloc_df.loc[val_mask, "customer_id"].astype(str).tolist()
    test_cids = alloc_df.loc[test_mask, "customer_id"].astype(str).tolist()
    
    y_tr = alloc_df.loc[train_mask, "y"].to_numpy(dtype=int)
    y_va = alloc_df.loc[val_mask, "y"].to_numpy(dtype=int)
    y_te = alloc_df.loc[test_mask, "y"].to_numpy(dtype=int)
    
    test_is_delta = alloc_df.loc[test_mask, "delta_1c_0"].to_numpy(dtype=bool)
    
    # 1. Fit train-only fill
    fill_vec = fit_train_fill(train_cids, customers_data, filling=filling)
    
    # 2. Fit MultiRocket biases on 4 seeded days per training customer
    seed = 20261014 + alloc_idx * 10
    rng = np.random.default_rng(seed)
    
    sample_days = []
    for cid in train_cids:
        _, hr = customers_data[cid]
        n_days = len(hr)
        pick = rng.choice(n_days, min(4, n_days), replace=False)
        sample_days.append(hr[pick])
    sample_x = np.concatenate(sample_days)
    sample_x = np.where(np.isfinite(sample_x), sample_x, fill_vec).astype(np.float64)
    
    seed_numba(seed)
    mr_base = mr_fit(sample_x, num_features=1250, max_dilations_per_kernel=32)
    seed_numba(seed + 1)
    mr_diff = mr_fit(np.diff(sample_x, axis=1), num_features=1250, max_dilations_per_kernel=32)
    
    # 3. Transform and pool features
    print(f"  Alloc {alloc_idx}: Transforming features (MR + Summary)...", flush=True)
    t_xf_start = time.time()
    
    X_tr_mr = [transform_customer(static_features[cid], customers_data[cid][0], customers_data[cid][1], fill_vec, mr_base, mr_diff, mr_transform) for cid in train_cids]
    X_va_mr = [transform_customer(static_features[cid], customers_data[cid][0], customers_data[cid][1], fill_vec, mr_base, mr_diff, mr_transform) for cid in val_cids]
    X_te_mr = [transform_customer(static_features[cid], customers_data[cid][0], customers_data[cid][1], fill_vec, mr_base, mr_diff, mr_transform) for cid in test_cids]
    
    X_tr_mr, X_va_mr, X_te_mr = np.stack(X_tr_mr), np.stack(X_va_mr), np.stack(X_te_mr)
    X_tr_sm = np.stack([static_features[cid] for cid in train_cids])
    X_va_sm = np.stack([static_features[cid] for cid in val_cids])
    X_te_sm = np.stack([static_features[cid] for cid in test_cids])
    
    t_xf_sec = time.time() - t_xf_start
    print(f"  Alloc {alloc_idx}: MultiRocket feature transform took {t_xf_sec:.1f}s", flush=True)
    
    # 4. Fit Ridge models with train-only hygiene
    pipe_mr = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(0.0), StandardScaler(), Ridge(alpha=alpha))
    pipe_mr.fit(X_tr_mr, y_tr)
    
    pipe_sm = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(0.0), StandardScaler(), Ridge(alpha=alpha))
    pipe_sm.fit(X_tr_sm, y_tr)
    
    # 5. Evaluate
    score_va_mr = pipe_mr.predict(X_va_mr)
    score_te_mr = pipe_mr.predict(X_te_mr)
    
    score_va_sm = pipe_sm.predict(X_va_sm)
    score_te_sm = pipe_sm.predict(X_te_sm)
    
    val_auroc_mr = roc_auc_score(y_va, score_va_mr)
    test_auroc_mr = roc_auc_score(y_te, score_te_mr)
    
    test_auroc_strict_mr = roc_auc_score(y_te[~test_is_delta], score_te_mr[~test_is_delta])
    test_auroc_delta_mr = roc_auc_score(y_te[test_is_delta], score_te_mr[test_is_delta])
    
    val_auroc_sm = roc_auc_score(y_va, score_va_sm)
    test_auroc_sm = roc_auc_score(y_te, score_te_sm)
    
    t_sec = time.time() - t0_alloc
    print(f"  Alloc {alloc_idx} in {t_sec:.1f}s | MR Test: {test_auroc_mr:.4f} (Strict: {test_auroc_strict_mr:.4f}, Delta: {test_auroc_delta_mr:.4f}) | Summary Test: {test_auroc_sm:.4f}", flush=True)
    
    return {
        "alloc": alloc_idx,
        "mr_val_auroc": val_auroc_mr,
        "mr_test_auroc": test_auroc_mr,
        "mr_test_strict_auroc": test_auroc_strict_mr,
        "mr_test_delta_auroc": test_auroc_delta_mr,
        "sm_val_auroc": val_auroc_sm,
        "sm_test_auroc": test_auroc_sm,
        "time_sec": t_sec,
        "n_test_total": len(y_te),
        "n_test_strict": int((~test_is_delta).sum()),
        "n_test_delta": int(test_is_delta.sum()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "tier1c_10_alloc_results.csv"
    
    print(f"Active Numba threads: {numba.get_num_threads()}")
    print(f"Reading allocations from {ALLOC_PATH}...")
    alloc_df = pd.read_parquet(ALLOC_PATH)
    
    customers_data, static_features = load_all_tier1c_data(alloc_df)
    mr_fit, mr_transform = reference_functions()
    
    print("\nStarting 10-allocation cross-validation on Tier 1c (4,410 participants)...")
    results = []
    
    for a in range(10):
        res = run_allocation(a, alloc_df, customers_data, static_features, mr_fit, mr_transform, filling="clock", alpha=1000.0)
        results.append(res)
        # Checkpoint to CSV after each allocation
        pd.DataFrame(results).to_csv(out_csv, index=False)
        
    res_df = pd.DataFrame(results)
    
    print("\n================ FINAL TIER 1C RESULTS (10 Allocations) ================")
    print(f"MR Test AUROC (Overall 4,410 cohort): {res_df['mr_test_auroc'].mean():.4f} +/- {res_df['mr_test_auroc'].std(ddof=1):.4f}")
    print(f"MR Test AUROC (Strict Stratum ~3,848): {res_df['mr_test_strict_auroc'].mean():.4f} +/- {res_df['mr_test_strict_auroc'].std(ddof=1):.4f}")
    print(f"MR Test AUROC (Delta Stratum 562 wear-fail): {res_df['mr_test_delta_auroc'].mean():.4f} +/- {res_df['mr_test_delta_auroc'].std(ddof=1):.4f}")
    print(f"Summary Test AUROC (Baseline):         {res_df['sm_test_auroc'].mean():.4f} +/- {res_df['sm_test_auroc'].std(ddof=1):.4f}")
    print(f"Paired Gain (MR - Summary):            {res_df['mr_test_auroc'].mean() - res_df['sm_test_auroc'].mean():+.4f}")
    print(f"Saved full metrics table to {out_csv}")


if __name__ == "__main__":
    main()
