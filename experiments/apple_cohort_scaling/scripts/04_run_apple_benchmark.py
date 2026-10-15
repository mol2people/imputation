"""Step 4: High-throughput factorial benchmark across 10 allocations on the full Apple cohort.

Optimizations:
  - Vectorized MultiRocket day-batching (batch_size=2048) with online moment accumulation
  - Robust candidate alpha grid: [100.0, 500.0, 1000.0, 5000.0, 10000.0]
  - Strict allocation isolation: Fills, bias quantiles, scaling, and models fit strictly on train
Evaluates:
  - 10 allocations
  - 2 history budgets: All available adequate days vs First up to 40 adequate days
  - 2 resolutions: 5-minute (288 bins) vs 15-minute (96 bins)
  - 2 representations: MultiRocket + Summary vs Summary baseline
  - 2 regularization rules: Fixed alpha=1000.0 vs Validation-selected alpha
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "8"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import sys
import time
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_DIR / "artifacts"

sys.path.insert(0, str(REPO_ROOT / "HEARTBEAT_LAB" / "src"))
from heartbeat_lab.data import STATS
from heartbeat_lab.model import reference_functions, seed_numba

ALLOC_PATH = OUTPUT_DIR / "apple_10_allocations.parquet"
CACHE_5M_DIR = OUTPUT_DIR / "cache_5min"
CACHE_15M_DIR = OUTPUT_DIR / "cache_15min"
ALPHAS = [100.0, 500.0, 1000.0, 5000.0, 10000.0]


def find_optimal_threshold(y_val, y_val_score):
    unique_thresh = np.unique(y_val_score)
    n_pos = np.sum(y_val == 1)
    n_neg = np.sum(y_val == 0)

    pos_sorted = np.sort(y_val_score[y_val == 1])
    neg_sorted = np.sort(y_val_score[y_val == 0])

    tp = len(pos_sorted) - np.searchsorted(pos_sorted, unique_thresh, side='left')
    tn = np.searchsorted(neg_sorted, unique_thresh, side='left')

    bal_acc = 0.5 * (tp / n_pos + tn / n_neg)
    max_acc = np.max(bal_acc)
    # Break exact ties by choosing the larger threshold
    best_indices = np.flatnonzero(np.isclose(bal_acc, max_acc, atol=1e-12))
    best_thresh = float(unique_thresh[best_indices[-1]])
    return best_thresh, float(max_acc)


def load_cache_data(cache_dir, customer_ids):
    print(f"Loading cache from {cache_dir.name} for {len(customer_ids)} customers...")
    t0 = time.time()
    cust_data = {}
    for cid in customer_ids:
        path = cache_dir / f"customer_{cid}.npz"
        with np.load(path) as npz:
            date_series = pd.to_datetime(npz["date"], unit="D")
            days = pd.DataFrame(npz["stats"], columns=STATS)
            days["date"] = date_series
            hr = npz["hr"].copy()
            cust_data[cid] = (days, hr)
    print(f"Loaded {len(cust_data)} customers in {time.time() - t0:.1f}s.")
    return cust_data


def compute_static_features(days, hr):
    weekend = days.date.dt.dayofweek.ge(5)
    if (~weekend).any() and weekend.any():
        contrast = days.loc[~weekend, "mean"].mean() - days.loc[weekend, "mean"].mean()
    else:
        contrast = 0.0
    if np.isnan(contrast):
        contrast = 0.0

    static = np.r_[
        days[STATS].mean().to_numpy(),
        days[STATS].std(ddof=1).to_numpy() if len(days) > 1 else np.zeros(len(STATS)),
        contrast,
        np.isfinite(hr).mean(axis=0)
    ].astype(np.float32)
    return static


def fit_train_fill(train_cids, cust_data_budget, n_bins):
    train_all = np.concatenate([cust_data_budget[cid][1] for cid in train_cids], axis=0)
    fill = np.nanmedian(train_all, axis=0).astype(np.float32)
    if np.isnan(fill).any():
        fallback = float(np.nanmedian(train_all))
        fill = np.where(np.isnan(fill), fallback, fill)
    return fill


def transform_cohort_batched(cids, cust_data_dict, static_dict, fill_vec, mr_base, mr_diff, mr_transform, batch_size=2048):
    c_days = [cust_data_dict[cid][1] for cid in cids]
    cust_counts = [len(hr) for hr in c_days]
    all_x = np.concatenate(c_days, axis=0)
    all_x = np.where(np.isfinite(all_x), all_x, fill_vec).astype(np.float64)
    all_diff = np.diff(all_x, axis=1)

    day_to_cust_idx = np.repeat(np.arange(len(cids)), cust_counts)

    # Probe dimension
    f_sample = mr_transform(all_x[:2], all_diff[:2], mr_base, mr_diff, n_features_per_kernel=4)
    d_feat = f_sample.shape[1]

    sum_mat = np.zeros((len(cids), d_feat), dtype=np.float64)
    sum_sq_mat = np.zeros((len(cids), d_feat), dtype=np.float64)

    n_days = len(all_x)
    for s in range(0, n_days, batch_size):
        e = min(s + batch_size, n_days)
        bx = all_x[s:e]
        bd = all_diff[s:e]
        f = np.nan_to_num(mr_transform(bx, bd, mr_base, mr_diff, n_features_per_kernel=4)).astype(np.float64)
        batch_cust = day_to_cust_idx[s:e]

        u_cust, u_idx = np.unique(batch_cust, return_index=True)
        u_idx = np.r_[u_idx, len(batch_cust)]
        for j, u in enumerate(u_cust):
            f_slice = f[u_idx[j]:u_idx[j+1]]
            sum_mat[u] += np.sum(f_slice, axis=0)
            sum_sq_mat[u] += np.sum(f_slice ** 2, axis=0)

    feats = []
    for i, cid in enumerate(cids):
        n = cust_counts[i]
        mean_v = sum_mat[i] / n
        var_v = (sum_sq_mat[i] - (sum_mat[i] ** 2) / n) / (n - 1) if n > 1 else np.zeros(d_feat)
        sd_v = np.sqrt(np.maximum(0.0, var_v))
        feats.append(np.r_[static_dict[cid], mean_v, sd_v].astype(np.float32))

    return np.stack(feats)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading allocations from {ALLOC_PATH}...")
    alloc_df = pd.read_parquet(ALLOC_PATH)
    all_cids = alloc_df["customer_id"].astype(str).tolist()

    # 1. Preload 5m and 15m caches
    cust_5m_all = load_cache_data(CACHE_5M_DIR, all_cids)
    cust_15m_all = load_cache_data(CACHE_15M_DIR, all_cids)

    # Create 40-day truncated history views
    print("Preparing 40-day truncated history views...")
    cust_5m_40d = {cid: (days.iloc[:40].copy(), hr[:40].copy()) for cid, (days, hr) in cust_5m_all.items()}
    cust_15m_40d = {cid: (days.iloc[:40].copy(), hr[:40].copy()) for cid, (days, hr) in cust_15m_all.items()}

    # Precompute static summary features
    print("Precomputing static summary features...")
    static_5m_all = {cid: compute_static_features(days, hr) for cid, (days, hr) in cust_5m_all.items()}
    static_5m_40d = {cid: compute_static_features(days, hr) for cid, (days, hr) in cust_5m_40d.items()}
    static_15m_all = {cid: compute_static_features(days, hr) for cid, (days, hr) in cust_15m_all.items()}
    static_15m_40d = {cid: compute_static_features(days, hr) for cid, (days, hr) in cust_15m_40d.items()}

    mr_fit, mr_transform = reference_functions()

    benchmark_metrics = []
    alpha_selection_records = []
    test_prediction_chunks = []

    n_allocations = 10
    print(f"\n================ STARTING OPTIMIZED FULL APPLE BENCHMARK ({n_allocations} ALLOCATIONS) ================\n")

    for a in range(n_allocations):
        t0_alloc = time.time()
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

        ordered_cids = tr_cids + va_cids + te_cids
        n_tr = len(tr_cids)
        n_va = len(va_cids)

        print(f"\n--- Allocation {a} (Train={n_tr}, Val={n_va}, Test={len(te_cids)}) ---")

        # Factorial combinations of (budget, resolution)
        configs = [
            ("all_days", "5m", 288, cust_5m_all, static_5m_all),
            ("all_days", "15m", 96, cust_15m_all, static_15m_all),
            ("up_to_40d", "5m", 288, cust_5m_40d, static_5m_40d),
            ("up_to_40d", "15m", 96, cust_15m_40d, static_15m_40d),
        ]

        for budget, res, n_bins, cust_data_dict, static_dict in configs:
            t0_extract = time.time()
            # 1. Fit clock-bin median on training participants
            fill_vec = fit_train_fill(tr_cids, cust_data_dict, n_bins)

            # 2. Sample <= 4,096 training days for MultiRocket bias fitting
            seed_base = 20261014 + a * 100 + (0 if budget == "all_days" else 10) + (0 if res == "5m" else 1)
            rng = np.random.default_rng(seed_base)

            n_target_sample = 4096
            days_per_user = max(1, n_target_sample // len(tr_cids))
            sample_days = []
            for cid in tr_cids:
                _, hr = cust_data_dict[cid]
                n_d = len(hr)
                pick = rng.choice(n_d, min(days_per_user, n_d), replace=False)
                sample_days.append(hr[pick])
            sample_x = np.concatenate(sample_days)
            if len(sample_x) > 4096:
                sample_x = sample_x[rng.choice(len(sample_x), 4096, replace=False)]
            sample_x = np.where(np.isfinite(sample_x), sample_x, fill_vec).astype(np.float64)

            seed_numba(seed_base)
            mr_base = mr_fit(sample_x, num_features=1250, max_dilations_per_kernel=32)
            seed_numba(seed_base + 1)
            mr_diff = mr_fit(np.diff(sample_x, axis=1), num_features=1250, max_dilations_per_kernel=32)

            # 3. High-throughput bulk batched transform
            X_all_mr = transform_cohort_batched(ordered_cids, cust_data_dict, static_dict, fill_vec, mr_base, mr_diff, mr_transform, batch_size=2048)
            X_tr_mr = X_all_mr[:n_tr]
            X_va_mr = X_all_mr[n_tr:n_tr + n_va]
            X_te_mr = X_all_mr[n_tr + n_va:]

            # Preprocessing fit strictly on training features
            pipe_prep = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(0.0), StandardScaler())
            X_tr_std = pipe_prep.fit_transform(X_tr_mr)
            X_va_std = pipe_prep.transform(X_va_mr)
            X_te_std = pipe_prep.transform(X_te_mr)
            t_extract = time.time() - t0_extract

            # 4. Fit candidate alphas on training features
            t0_fit = time.time()
            val_aurocs = {}
            fitted_models = {}
            for alpha in ALPHAS:
                clf = Ridge(alpha=alpha)
                clf.fit(X_tr_std, y_tr)
                va_scores = clf.predict(X_va_std)
                va_auc = roc_auc_score(y_va, va_scores)
                val_aurocs[alpha] = va_auc
                fitted_models[alpha] = (clf, va_scores)
            t_fit = time.time() - t0_fit

            # Select best alpha on validation AUROC (tie-break: larger alpha)
            max_auc = max(val_aurocs.values())
            best_alpha = max([alp for alp, auc in val_aurocs.items() if np.isclose(auc, max_auc, atol=1e-12)])

            for alpha, val_auc in val_aurocs.items():
                alpha_selection_records.append({
                    "allocation": a,
                    "budget": budget,
                    "resolution": res,
                    "representation": "multirocket",
                    "alpha_candidate": alpha,
                    "val_auroc": val_auc,
                    "is_selected": bool(alpha == best_alpha),
                })

            # Evaluate both fixed alpha=1000 and selected alpha
            t0_eval = time.time()
            for rule_name, alpha_val in [("ridge_fixed", 1000.0), ("ridge_selected", best_alpha)]:
                clf, va_scores = fitted_models[alpha_val]
                th, va_bal = find_optimal_threshold(y_va, va_scores)

                te_scores = clf.predict(X_te_std)
                te_auc = roc_auc_score(y_te, te_scores)
                te_bal = balanced_accuracy_score(y_te, (te_scores >= th).astype(int))

                benchmark_metrics.append({
                    "allocation": a,
                    "budget": budget,
                    "resolution": res,
                    "representation": "multirocket",
                    "classifier": rule_name,
                    "alpha": alpha_val,
                    "threshold": th,
                    "val_auroc": val_aurocs[alpha_val],
                    "test_auroc": te_auc,
                    "val_bal_acc": va_bal,
                    "test_bal_acc": te_bal,
                    "time_extract_sec": t_extract,
                    "time_fit_sec": t_fit,
                    "time_eval_sec": time.time() - t0_eval,
                })

                pred_df = pd.DataFrame({
                    "allocation": a,
                    "budget": budget,
                    "resolution": res,
                    "representation": "multirocket",
                    "classifier": rule_name,
                    "alpha": alpha_val,
                    "threshold": th,
                    "customer_id": te_cids,
                    "y_true": y_te,
                    "y_score": te_scores.astype(np.float32),
                })
                test_prediction_chunks.append(pred_df)

            print(f"Alloc {a} | MR {res} {budget:10s} | Test AUROC: Fixed={benchmark_metrics[-2]['test_auroc']:.4f}, Sel(α={int(best_alpha)})={benchmark_metrics[-1]['test_auroc']:.4f} (extract: {t_extract:.1f}s)")

        # Summary baselines (All days and 40d)
        for budget, static_dict in [("all_days", static_5m_all), ("up_to_40d", static_5m_40d)]:
            t0_extract_sm = time.time()
            X_tr_sm = np.stack([static_dict[cid] for cid in tr_cids])
            X_va_sm = np.stack([static_dict[cid] for cid in va_cids])
            X_te_sm = np.stack([static_dict[cid] for cid in te_cids])

            pipe_sm = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(0.0), StandardScaler())
            X_tr_sm_std = pipe_sm.fit_transform(X_tr_sm)
            X_va_sm_std = pipe_sm.transform(X_va_sm)
            X_te_sm_std = pipe_sm.transform(X_te_sm)
            t_extract_sm = time.time() - t0_extract_sm

            t0_fit_sm = time.time()
            val_aurocs_sm = {}
            fitted_models_sm = {}
            for alpha in ALPHAS:
                clf = Ridge(alpha=alpha)
                clf.fit(X_tr_sm_std, y_tr)
                va_scores = clf.predict(X_va_sm_std)
                va_auc = roc_auc_score(y_va, va_scores)
                val_aurocs_sm[alpha] = va_auc
                fitted_models_sm[alpha] = (clf, va_scores)
            t_fit_sm = time.time() - t0_fit_sm

            max_auc_sm = max(val_aurocs_sm.values())
            best_alpha_sm = max([alp for alp, auc in val_aurocs_sm.items() if np.isclose(auc, max_auc_sm, atol=1e-12)])

            for alpha, val_auc in val_aurocs_sm.items():
                alpha_selection_records.append({
                    "allocation": a,
                    "budget": budget,
                    "resolution": "5m",
                    "representation": "summary",
                    "alpha_candidate": alpha,
                    "val_auroc": val_auc,
                    "is_selected": bool(alpha == best_alpha_sm),
                })

            t0_eval_sm = time.time()
            for rule_name, alpha_val in [("ridge_fixed", 1000.0), ("ridge_selected", best_alpha_sm)]:
                clf, va_scores = fitted_models_sm[alpha_val]
                th, va_bal = find_optimal_threshold(y_va, va_scores)

                te_scores = clf.predict(X_te_sm_std)
                te_auc = roc_auc_score(y_te, te_scores)
                te_bal = balanced_accuracy_score(y_te, (te_scores >= th).astype(int))

                benchmark_metrics.append({
                    "allocation": a,
                    "budget": budget,
                    "resolution": "5m",
                    "representation": "summary",
                    "classifier": rule_name,
                    "alpha": alpha_val,
                    "threshold": th,
                    "val_auroc": val_aurocs_sm[alpha_val],
                    "test_auroc": te_auc,
                    "val_bal_acc": va_bal,
                    "test_bal_acc": te_bal,
                    "time_extract_sec": t_extract_sm,
                    "time_fit_sec": t_fit_sm,
                    "time_eval_sec": time.time() - t0_eval_sm,
                })

                pred_df = pd.DataFrame({
                    "allocation": a,
                    "budget": budget,
                    "resolution": "5m",
                    "representation": "summary",
                    "classifier": rule_name,
                    "alpha": alpha_val,
                    "threshold": th,
                    "customer_id": te_cids,
                    "y_true": y_te,
                    "y_score": te_scores.astype(np.float32),
                })
                test_prediction_chunks.append(pred_df)

            print(f"Alloc {a} | Summary {budget:10s}    | Test AUROC: Fixed={benchmark_metrics[-2]['test_auroc']:.4f}, Sel(α={int(best_alpha_sm)})={benchmark_metrics[-1]['test_auroc']:.4f}")

        dt_alloc = time.time() - t0_alloc
        print(f"Alloc {a} total elapsed time: {dt_alloc:.1f}s ({dt_alloc/60:.2f} min)")

    # Save outputs
    print("\nSaving complete benchmark artifacts...")
    metrics_df = pd.DataFrame(benchmark_metrics)
    metrics_csv = OUTPUT_DIR / "apple_benchmark_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved {len(metrics_df)} metric rows to {metrics_csv}")

    alpha_df = pd.DataFrame(alpha_selection_records)
    alpha_csv = OUTPUT_DIR / "alpha_selection.csv"
    alpha_df.to_csv(alpha_csv, index=False)
    print(f"Saved {len(alpha_df)} alpha selection records to {alpha_csv}")

    all_preds_df = pd.concat(test_prediction_chunks, ignore_index=True)
    preds_parquet = OUTPUT_DIR / "scaling_predictions_test.parquet"
    all_preds_df.to_parquet(preds_parquet, index=False)
    print(f"Saved {len(all_preds_df)} test prediction rows to {preds_parquet}")

    print("\n================ BENCHMARK RUN COMPLETED ================")


if __name__ == "__main__":
    main()
