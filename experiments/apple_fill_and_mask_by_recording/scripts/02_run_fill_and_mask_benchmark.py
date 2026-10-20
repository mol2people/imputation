"""Stage 1 Production Benchmark Script: Apple Watch Fill and Mask by Recording.

Evaluates 8 feature arms across allocations 0, 1, 2 at 5m (L=288) and 15m (L=96)
resolutions under all_days and up_to_40d history budgets.

Arms:
  1. H_clock: Baseline clock-bin median filling (reference).
  2. H_global: Scalar global median filling.
  3. H_clean_diff: Clock filling + piecewise clean difference (zeroed at missing).
  4. H_short_interp: Span-matched short-gap (<=30m) linear interpolation + clock fallback.
  5. M_static: Static observation-mask mean + SD (576 / 192 cols, no HR values).
  6. M_temporal: MultiRocket on binary mask M and Delta M plus static mask block.
  7. H_plus_M_concat: Joint single-penalty Ridge over concatenated HR and mask blocks.
  8. H_plus_M_stack: Validation-tuned 21-point convex blend over standardized Ridge scores.

Features strict train-only preprocessing, exact seed alignment, allocation 0 reference
parity gate, and checkpointing after each allocation.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Threading configurations for reproducible and fast BLAS/Numba execution
os.environ["NUMBA_NUM_THREADS"] = "12"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
from numba import njit, prange
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ensure repository root and heartbeat_lab are in python path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "HEARTBEAT_LAB" / "src"))
from heartbeat_lab.data import STATS
from heartbeat_lab.model import reference_functions, seed_numba

sys.path.insert(0, str(REPO_ROOT))
from experiments.apple_fill_and_mask_by_recording.scripts.transforms import (
    compute_clean_diff,
    compute_mask_stats,
    fit_fill_templates,
    interpolate_short_gaps,
)

# Paths
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXPERIMENT_DIR / "results"
SCALING_ARTIFACTS_DIR = REPO_ROOT / "experiments" / "apple_cohort_scaling" / "artifacts"
ALLOC_PATH = SCALING_ARTIFACTS_DIR / "apple_10_allocations.parquet"
REF_PRED_PATH = SCALING_ARTIFACTS_DIR / "scaling_predictions_test.parquet"
CACHE_5M_DIR = SCALING_ARTIFACTS_DIR / "cache_5min"
CACHE_15M_DIR = SCALING_ARTIFACTS_DIR / "cache_15min"

# Regularization grids
ALPHAS_ORIG = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
ALPHAS_EXT = [100.0, 500.0, 1000.0, 5000.0, 10000.0, 100000.0]


@njit(fastmath=True, parallel=True, cache=True)
def mr_transform_diff_only(input_length: int, X1: np.ndarray, parameters1, n_features_per_kernel: int = 4) -> np.ndarray:
    """Exact single-stream MultiRocket transform for difference series X1 (used by H_clean_diff)."""
    num_examples = X1.shape[0]
    dilations1, num_features_per_dilation1, biases1 = parameters1

    indices = np.array((
        0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 1, 6, 0, 1, 7, 0, 1, 8,
        0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 2, 6, 0, 2, 7, 0, 2, 8, 0, 3, 4,
        0, 3, 5, 0, 3, 6, 0, 3, 7, 0, 3, 8, 0, 4, 5, 0, 4, 6, 0, 4, 7,
        0, 4, 8, 0, 5, 6, 0, 5, 7, 0, 5, 8, 0, 6, 7, 0, 6, 8, 0, 7, 8,
        1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8, 1, 3, 4,
        1, 3, 5, 1, 3, 6, 1, 3, 7, 1, 3, 8, 1, 4, 5, 1, 4, 6, 1, 4, 7,
        1, 4, 8, 1, 5, 6, 1, 5, 7, 1, 5, 8, 1, 6, 7, 1, 6, 8, 1, 7, 8,
        2, 3, 4, 2, 3, 5, 2, 3, 6, 2, 3, 7, 2, 3, 8, 2, 4, 5, 2, 4, 6,
        2, 4, 7, 2, 4, 8, 2, 5, 6, 2, 5, 7, 2, 5, 8, 2, 6, 7, 2, 6, 8,
        2, 7, 8, 3, 4, 5, 3, 4, 6, 3, 4, 7, 3, 4, 8, 3, 5, 6, 3, 5, 7,
        3, 5, 8, 3, 6, 7, 3, 6, 8, 3, 7, 8, 4, 5, 6, 4, 5, 7, 4, 5, 8,
        4, 6, 7, 4, 6, 8, 4, 7, 8, 5, 6, 7, 5, 6, 8, 5, 7, 8, 6, 7, 8
    ), dtype=np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations1 = len(dilations1)
    num_features1 = num_kernels * np.sum(num_features_per_dilation1)
    features = np.zeros((num_examples, num_features1 * n_features_per_kernel), dtype=np.float32)

    for example_index in prange(num_examples):
        _X1 = X1[example_index]
        A1 = -_X1
        G1 = _X1 + _X1 + _X1

        feature_index_start = 0

        for dilation_index in range(num_dilations1):
            _padding0 = dilation_index % 2
            dilation = dilations1[dilation_index]
            padding = ((9 - 1) * dilation) // 2
            num_features_this_dilation = num_features_per_dilation1[dilation_index]

            C_alpha = np.zeros(input_length - 1, dtype=np.float32)
            C_alpha[:] = A1

            C_gamma = np.zeros((9, input_length - 1), dtype=np.float32)
            C_gamma[9 // 2] = G1

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):
                C_alpha[-end:] = C_alpha[-end:] + A1[:end]
                C_gamma[gamma_index, -end:] = G1[:end]
                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):
                C_alpha[:-start] = C_alpha[:-start] + A1[start:]
                C_gamma[gamma_index, :-start] = G1[start:]
                start += dilation

            for kernel_index in range(num_kernels):
                feature_index_end = feature_index_start + num_features_this_dilation
                _padding1 = (_padding0 + kernel_index) % 2
                index_0, index_1, index_2 = indices[kernel_index]
                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]
                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0
                        for j in range(C.shape[0]):
                            if C[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += C[j] + _bias
                            elif C[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = C.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / C.shape[0]
                        end = end + num_features1
                        features[example_index, end] = max_stretch
                        end = end + num_features1
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features1
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1
                else:
                    _c = C[padding:-padding]
                    for feature_count in range(num_features_this_dilation):
                        feature_index = feature_index_start + feature_count
                        _bias = biases1[feature_index]
                        ppv = 0
                        last_val = 0
                        max_stretch = 0.0
                        mean_index = 0
                        mean = 0
                        for j in range(_c.shape[0]):
                            if _c[j] > _bias:
                                ppv += 1
                                mean_index += j
                                mean += _c[j] + _bias
                            elif _c[j] < _bias:
                                stretch = j - last_val
                                if stretch > max_stretch:
                                    max_stretch = stretch
                                last_val = j
                        stretch = _c.shape[0] - 1 - last_val
                        if stretch > max_stretch:
                            max_stretch = stretch

                        end = feature_index
                        features[example_index, end] = ppv / _c.shape[0]
                        end = end + num_features1
                        features[example_index, end] = max_stretch
                        end = end + num_features1
                        features[example_index, end] = mean / ppv if ppv > 0 else 0
                        end = end + num_features1
                        features[example_index, end] = mean_index / ppv if ppv > 0 else -1

                feature_index_start = feature_index_end

    return features


def load_cache_data(cache_dir: Path, customer_ids: list[str]) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Preload per-customer daily stats and raw HR grids."""
    print(f"Loading cache from {cache_dir.name} for {len(customer_ids)} customers...", flush=True)
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
    print(f"Loaded {len(cust_data)} customers in {time.time() - t0:.1f}s.", flush=True)
    return cust_data


def compute_static_features_b_l(days: pd.DataFrame, hr: np.ndarray) -> np.ndarray:
    """Compute B_L(u): 305 cols (5m) or 113 cols (15m)."""
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
        np.isfinite(hr).mean(axis=0),
    ].astype(np.float32)
    return static


def compute_static_features_mask(hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute S(M)_L(u) (mean and Bessel SD, 2L cols) and S(M)_SD,L(u) (SD only, L cols)."""
    mask = np.isfinite(hr).astype(np.uint8)
    mean_m, sd_m = compute_mask_stats(mask)
    s_m_full = np.r_[mean_m, sd_m].astype(np.float32)
    s_m_sd = sd_m.astype(np.float32)
    return s_m_full, s_m_sd


def transform_sequence_batched(
    cids: list[str],
    cust_counts: list[int],
    all_x: np.ndarray,
    all_diff: np.ndarray | None,
    mr_base,
    mr_diff,
    mr_transform,
    batch_size: int = 2048,
) -> np.ndarray:
    """Batch-transform sequence arrays via MultiRocket and pool mean + SD across participant days."""
    day_to_cust_idx = np.repeat(np.arange(len(cids)), cust_counts)

    # Probe output dimension
    bx0 = np.ascontiguousarray(all_x[:2], dtype=np.float64)
    bd0 = np.diff(bx0, axis=1) if all_diff is None else np.ascontiguousarray(all_diff[:2], dtype=np.float64)
    f_sample = mr_transform(bx0, bd0, mr_base, mr_diff, n_features_per_kernel=4)
    d_feat = f_sample.shape[1]

    sum_mat = np.zeros((len(cids), d_feat), dtype=np.float64)
    sum_sq_mat = np.zeros((len(cids), d_feat), dtype=np.float64)

    n_days = len(all_x)
    for s in range(0, n_days, batch_size):
        e = min(s + batch_size, n_days)
        bx = np.ascontiguousarray(all_x[s:e], dtype=np.float64)
        bd = np.diff(bx, axis=1) if all_diff is None else np.ascontiguousarray(all_diff[s:e], dtype=np.float64)
        f = np.nan_to_num(mr_transform(bx, bd, mr_base, mr_diff, n_features_per_kernel=4)).astype(np.float64)
        batch_cust = day_to_cust_idx[s:e]

        u_cust, u_idx = np.unique(batch_cust, return_index=True)
        u_idx = np.r_[u_idx, len(batch_cust)]
        for j, u in enumerate(u_cust):
            f_slice = f[u_idx[j] : u_idx[j + 1]]
            sum_mat[u] += np.sum(f_slice, axis=0)
            sum_sq_mat[u] += np.sum(f_slice**2, axis=0)

    seq_feats = []
    for i in range(len(cids)):
        n = cust_counts[i]
        mean_v = sum_mat[i] / n
        var_v = (sum_sq_mat[i] - (sum_mat[i] ** 2) / n) / (n - 1) if n > 1 else np.zeros(d_feat)
        sd_v = np.sqrt(np.maximum(0.0, var_v))
        seq_feats.append(np.r_[mean_v, sd_v].astype(np.float32))

    return np.stack(seq_feats)


def transform_diff_only_batched(
    cids: list[str],
    cust_counts: list[int],
    input_length: int,
    all_diff: np.ndarray,
    mr_diff,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch-transform only the difference stream via MultiRocket and return (mean_mat, sd_mat)."""
    day_to_cust_idx = np.repeat(np.arange(len(cids)), cust_counts)

    bd0 = np.ascontiguousarray(all_diff[:2], dtype=np.float64)
    f_sample = mr_transform_diff_only(input_length, bd0, mr_diff, n_features_per_kernel=4)
    d_feat = f_sample.shape[1]

    sum_mat = np.zeros((len(cids), d_feat), dtype=np.float64)
    sum_sq_mat = np.zeros((len(cids), d_feat), dtype=np.float64)

    n_days = len(all_diff)
    for s in range(0, n_days, batch_size):
        e = min(s + batch_size, n_days)
        bd = np.ascontiguousarray(all_diff[s:e], dtype=np.float64)
        f = np.nan_to_num(mr_transform_diff_only(input_length, bd, mr_diff, n_features_per_kernel=4)).astype(np.float64)
        batch_cust = day_to_cust_idx[s:e]

        u_cust, u_idx = np.unique(batch_cust, return_index=True)
        u_idx = np.r_[u_idx, len(batch_cust)]
        for j, u in enumerate(u_cust):
            f_slice = f[u_idx[j] : u_idx[j + 1]]
            sum_mat[u] += np.sum(f_slice, axis=0)
            sum_sq_mat[u] += np.sum(f_slice**2, axis=0)

    mean_list = []
    sd_list = []
    for i in range(len(cids)):
        n = cust_counts[i]
        mean_v = sum_mat[i] / n
        var_v = (sum_sq_mat[i] - (sum_mat[i] ** 2) / n) / (n - 1) if n > 1 else np.zeros(d_feat)
        sd_v = np.sqrt(np.maximum(0.0, var_v))
        mean_list.append(mean_v.astype(np.float32))
        sd_list.append(sd_v.astype(np.float32))

    return np.stack(mean_list), np.stack(sd_list)


def evaluate_feature_arm(
    X_full: np.ndarray,
    n_tr: int,
    n_va: int,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    y_te: np.ndarray,
    alphas_to_fit: list[float],
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """Standardize strictly on train, fit candidate alphas, and evaluate prediction scores.

    Returns:
        metrics_by_rule: {penalty_type: {metric_name: value}}
        test_scores_by_rule: {penalty_type: te_scores}
        val_scores_by_rule: {penalty_type: va_scores}
        train_scores_by_rule: {penalty_type: tr_scores}
        selected_alpha_by_rule: {penalty_type: alpha}
    """
    X_tr = X_full[:n_tr]
    X_va = X_full[n_tr : n_tr + n_va]
    X_te = X_full[n_tr + n_va :]

    pipe_prep = make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(0.0), StandardScaler())
    X_tr_std = pipe_prep.fit_transform(X_tr)
    X_va_std = pipe_prep.transform(X_va)
    X_te_std = pipe_prep.transform(X_te)

    val_aurocs = {}
    fitted_models = {}
    tr_scores_dict = {}
    va_scores_dict = {}
    te_scores_dict = {}

    for alpha in alphas_to_fit:
        clf = Ridge(alpha=alpha)
        clf.fit(X_tr_std, y_tr)
        s_tr = clf.predict(X_tr_std).astype(np.float32)
        s_va = clf.predict(X_va_std).astype(np.float32)
        s_te = clf.predict(X_te_std).astype(np.float32)

        va_auc = roc_auc_score(y_va, s_va)
        val_aurocs[alpha] = va_auc
        fitted_models[alpha] = clf
        tr_scores_dict[alpha] = s_tr
        va_scores_dict[alpha] = s_va
        te_scores_dict[alpha] = s_te

    # Selection rules:
    # 1. ridge_selected_ext: search over all alphas in ALPHAS_EXT
    max_auc_ext = max([val_aurocs[a] for a in ALPHAS_EXT])
    best_alpha_ext = max([a for a in ALPHAS_EXT if np.isclose(val_aurocs[a], max_auc_ext, atol=1e-12)])

    # 2. ridge_selected: search over alphas in ALPHAS_ORIG (<= 10000)
    max_auc_orig = max([val_aurocs[a] for a in ALPHAS_ORIG])
    best_alpha_orig = max([a for a in ALPHAS_ORIG if np.isclose(val_aurocs[a], max_auc_orig, atol=1e-12)])

    # 3. ridge_fixed: 1000.0
    alpha_fixed = 1000.0

    rules = [
        ("ridge_selected_ext", best_alpha_ext),
        ("ridge_selected", best_alpha_orig),
        ("ridge_fixed", alpha_fixed),
    ]

    metrics_by_rule = {}
    test_scores_by_rule = {}
    val_scores_by_rule = {}
    train_scores_by_rule = {}
    selected_alpha_by_rule = {}

    for rule_name, alpha_val in rules:
        s_tr = tr_scores_dict[alpha_val]
        s_va = va_scores_dict[alpha_val]
        s_te = te_scores_dict[alpha_val]

        tr_auc = roc_auc_score(y_tr, s_tr)
        va_auc = val_aurocs[alpha_val]
        te_auc = roc_auc_score(y_te, s_te)
        te_ap = average_precision_score(y_te, s_te)

        metrics_by_rule[rule_name] = {
            "alpha": alpha_val,
            "weight": np.nan,
            "train_auc": tr_auc,
            "val_auc": va_auc,
            "test_auc": te_auc,
            "test_ap": te_ap,
        }
        test_scores_by_rule[rule_name] = s_te
        val_scores_by_rule[rule_name] = s_va
        train_scores_by_rule[rule_name] = s_tr
        selected_alpha_by_rule[rule_name] = alpha_val

    return metrics_by_rule, test_scores_by_rule, val_scores_by_rule, train_scores_by_rule, selected_alpha_by_rule


def evaluate_stacking(
    s_tr_hr: np.ndarray,
    s_va_hr: np.ndarray,
    s_te_hr: np.ndarray,
    s_tr_m: np.ndarray,
    s_va_m: np.ndarray,
    s_te_m: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    y_te: np.ndarray,
) -> tuple[dict[str, float], float, np.ndarray]:
    """Standardize validation streams and search 21-point convex blend w in [0.0, 1.0]."""
    mu_va_hr = np.mean(s_va_hr)
    sd_va_hr = np.std(s_va_hr, ddof=1)
    s_va_hr_std = (s_va_hr - mu_va_hr) / sd_va_hr if sd_va_hr > 0 else np.zeros_like(s_va_hr)
    s_te_hr_std = (s_te_hr - mu_va_hr) / sd_va_hr if sd_va_hr > 0 else np.zeros_like(s_te_hr)
    s_tr_hr_std = (s_tr_hr - mu_va_hr) / sd_va_hr if sd_va_hr > 0 else np.zeros_like(s_tr_hr)

    mu_va_m = np.mean(s_va_m)
    sd_va_m = np.std(s_va_m, ddof=1)
    s_va_m_std = (s_va_m - mu_va_m) / sd_va_m if sd_va_m > 0 else np.zeros_like(s_va_m)
    s_te_m_std = (s_te_m - mu_va_m) / sd_va_m if sd_va_m > 0 else np.zeros_like(s_te_m)
    s_tr_m_std = (s_tr_m - mu_va_m) / sd_va_m if sd_va_m > 0 else np.zeros_like(s_tr_m)

    val_aucs_w = {}
    for w_val in np.linspace(0.0, 1.0, 21):
        w_round = round(float(w_val), 2)
        s_va_w = (1.0 - w_round) * s_va_hr_std + w_round * s_va_m_std
        val_aucs_w[w_round] = roc_auc_score(y_va, s_va_w)

    max_auc_w = max(val_aucs_w.values())
    # Break near-ties by choosing the smallest w (favoring H_clock)
    best_w = min([w for w, auc in val_aucs_w.items() if np.isclose(auc, max_auc_w, atol=1e-12)])

    s_tr_stack = (1.0 - best_w) * s_tr_hr_std + best_w * s_tr_m_std
    s_va_stack = (1.0 - best_w) * s_va_hr_std + best_w * s_va_m_std
    s_te_stack = (1.0 - best_w) * s_te_hr_std + best_w * s_te_m_std

    tr_auc = roc_auc_score(y_tr, s_tr_stack)
    va_auc = roc_auc_score(y_va, s_va_stack)
    te_auc = roc_auc_score(y_te, s_te_stack)
    te_ap = average_precision_score(y_te, s_te_stack)

    metrics = {
        "alpha": np.nan,
        "weight": best_w,
        "train_auc": tr_auc,
        "val_auc": va_auc,
        "test_auc": te_auc,
        "test_ap": te_ap,
    }
    return metrics, best_w, s_te_stack


def verify_parity_gate(
    ref_df: pd.DataFrame,
    allocation: int,
    budget: str,
    resolution: str,
    te_cids: list[str],
    scores_by_rule: dict[str, np.ndarray],
    y_te: np.ndarray,
):
    """Enforce exact Allocation 0 parity gate on H_clock."""
    ref_sub = ref_df[
        (ref_df["allocation"] == allocation)
        & (ref_df["budget"] == budget)
        & (ref_df["resolution"] == resolution)
        & (ref_df["representation"] == "multirocket")
    ]
    if len(ref_sub) == 0:
        print(f"Warning: No reference records found for alloc {allocation} {budget} {resolution}")
        return

    for rule in ["ridge_fixed", "ridge_selected"]:
        te_scores = scores_by_rule[rule]
        te_auc = roc_auc_score(y_te, te_scores)

        ref_rule = ref_sub[ref_sub["classifier"] == rule].copy()
        ref_rule["customer_id"] = ref_rule["customer_id"].astype(str)
        cur_pred = pd.DataFrame({"customer_id": [str(c) for c in te_cids], "score": te_scores})
        merged = cur_pred.merge(ref_rule, on="customer_id")

        if len(merged) != len(te_cids):
            raise AssertionError(f"Parity Gate mismatch in record count: expected {len(te_cids)}, got {len(merged)}")

        delta_score = float(np.max(np.abs(merged["score"] - merged["y_score"])))
        ref_auc = float(roc_auc_score(merged["y_true"], merged["y_score"]))
        delta_auc = float(abs(te_auc - ref_auc))

        print(
            f"  [PARITY CHECK] Alloc {allocation} | {budget} {resolution} | {rule:15s} | "
            f"Test AUC: cur={te_auc:.6f}, ref={ref_auc:.6f}, Delta AUC={delta_auc:.2e}, Max |Delta score|={delta_score:.2e}"
        )
        assert delta_score <= 1e-5, f"Parity Gate FAILED: max |delta_score| = {delta_score:.2e} > 1e-5"
        assert delta_auc <= 1e-10, f"Parity Gate FAILED: |delta_auc| = {delta_auc:.2e} > 1e-10"


def checkpoint_results(
    benchmark_metrics: list[dict],
    alpha_and_weight_selection: list[dict],
    test_prediction_chunks: list[pd.DataFrame],
):
    """Write intermediate or final benchmark outputs to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(benchmark_metrics)
    metrics_cols = [
        "allocation",
        "resolution",
        "budget",
        "arm",
        "penalty_type",
        "alpha",
        "weight",
        "train_auc",
        "val_auc",
        "test_auc",
        "test_ap",
        "runtime_s",
    ]
    metrics_df = metrics_df[metrics_cols].drop_duplicates(
        subset=["allocation", "resolution", "budget", "arm", "penalty_type"], keep="last"
    )
    metrics_df.to_csv(RESULTS_DIR / "benchmark_metrics.csv", index=False)

    alpha_df = pd.DataFrame(alpha_and_weight_selection)
    alpha_cols = [
        "allocation",
        "resolution",
        "budget",
        "arm",
        "penalty_type",
        "selected_alpha",
        "selected_weight",
        "val_auc",
    ]
    alpha_df = alpha_df[alpha_cols].drop_duplicates(
        subset=["allocation", "resolution", "budget", "arm", "penalty_type"], keep="last"
    )
    alpha_df.to_csv(RESULTS_DIR / "alpha_and_weight_selection.csv", index=False)

    if test_prediction_chunks:
        preds_df = pd.concat(test_prediction_chunks, ignore_index=True)
        pred_cols = ["user_id", "y", "allocation", "resolution", "budget", "arm", "penalty_type", "score"]
        if "customer_id" in preds_df.columns and "user_id" not in preds_df.columns:
            preds_df["user_id"] = preds_df["customer_id"]
        preds_df = preds_df[pred_cols].drop_duplicates(
            subset=["user_id", "allocation", "resolution", "budget", "arm", "penalty_type"], keep="last"
        )
        preds_df.to_parquet(RESULTS_DIR / "test_predictions.parquet", index=False)

    print(
        f"[CHECKPOINT] Wrote {len(metrics_df)} metric rows, {len(alpha_df)} selection rows, "
        f"and {len(preds_df) if test_prediction_chunks else 0} predictions to {RESULTS_DIR.name}/.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Apple Watch Fill and Mask Benchmark Runner")
    parser.add_argument("--allocations", type=str, default="0,1,2", help="Comma-separated allocation indices (e.g. '0,1,2')")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test on 50 participants")
    parser.add_argument("--resolutions", type=str, default="5m,15m", help="Comma-separated resolutions ('5m,15m')")
    parser.add_argument("--budgets", type=str, default="all_days,up_to_40d", help="Comma-separated budgets ('all_days,up_to_40d')")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for MultiRocket transforms")
    args = parser.parse_args()

    allocations = [int(x.strip()) for x in args.allocations.split(",") if x.strip()]
    resolutions = [x.strip() for x in args.resolutions.split(",") if x.strip()]
    budgets = [x.strip() for x in args.budgets.split(",") if x.strip()]

    print("================================================================================", flush=True)
    print("STARTING STAGE 1 APPLE BENCHMARK ENGINE", flush=True)
    print(f"Allocations: {allocations} | Resolutions: {resolutions} | Budgets: {budgets} | Smoke: {args.smoke}", flush=True)
    print("================================================================================", flush=True)

    # 1. Load allocations table
    alloc_df = pd.read_parquet(ALLOC_PATH)
    if args.smoke:
        print("[SMOKE MODE] Subsetting to first 60 participants for rapid pipeline test...", flush=True)
        alloc_df = alloc_df.iloc[:60].copy()

    all_cids = alloc_df["customer_id"].astype(str).tolist()

    # 2. Load caches
    cust_5m_all = load_cache_data(CACHE_5M_DIR, all_cids) if "5m" in resolutions else {}
    cust_15m_all = load_cache_data(CACHE_15M_DIR, all_cids) if "15m" in resolutions else {}

    # 3. Truncated views
    cust_5m_40d = {cid: (days.iloc[:40].copy(), hr[:40].copy()) for cid, (days, hr) in cust_5m_all.items()}
    cust_15m_40d = {cid: (days.iloc[:40].copy(), hr[:40].copy()) for cid, (days, hr) in cust_15m_all.items()}

    # 4. Precompute static feature blocks
    print("Precomputing static feature blocks B_L and S(M)_L...", flush=True)
    static_b_5m_all = {cid: compute_static_features_b_l(days, hr) for cid, (days, hr) in cust_5m_all.items()}
    static_b_5m_40d = {cid: compute_static_features_b_l(days, hr) for cid, (days, hr) in cust_5m_40d.items()}
    static_b_15m_all = {cid: compute_static_features_b_l(days, hr) for cid, (days, hr) in cust_15m_all.items()}
    static_b_15m_40d = {cid: compute_static_features_b_l(days, hr) for cid, (days, hr) in cust_15m_40d.items()}

    static_sm_5m_all = {cid: compute_static_features_mask(hr) for cid, (_, hr) in cust_5m_all.items()}
    static_sm_5m_40d = {cid: compute_static_features_mask(hr) for cid, (_, hr) in cust_5m_40d.items()}
    static_sm_15m_all = {cid: compute_static_features_mask(hr) for cid, (_, hr) in cust_15m_all.items()}
    static_sm_15m_40d = {cid: compute_static_features_mask(hr) for cid, (_, hr) in cust_15m_40d.items()}

    # Reference functions
    mr_fit, mr_transform = reference_functions()
    ref_df = pd.read_parquet(REF_PRED_PATH) if REF_PRED_PATH.exists() and not args.smoke else None

    metrics_file = RESULTS_DIR / "benchmark_metrics.csv"
    alpha_file = RESULTS_DIR / "alpha_and_weight_selection.csv"
    preds_file = RESULTS_DIR / "test_predictions.parquet"

    benchmark_metrics = []
    alpha_and_weight_selection = []
    test_prediction_chunks = []

    if metrics_file.exists() and not args.smoke:
        metrics_df = pd.read_csv(metrics_file)
        benchmark_metrics = metrics_df.to_dict("records")
        print(f"Loaded {len(benchmark_metrics)} existing metrics from {metrics_file.name}", flush=True)
    if alpha_file.exists() and not args.smoke:
        alpha_df = pd.read_csv(alpha_file)
        alpha_and_weight_selection = alpha_df.to_dict("records")
        print(f"Loaded {len(alpha_and_weight_selection)} existing selection records from {alpha_file.name}", flush=True)
    if preds_file.exists() and not args.smoke:
        preds_df = pd.read_parquet(preds_file)
        test_prediction_chunks = [preds_df]
        print(f"Loaded {len(preds_df)} existing test predictions from {preds_file.name}", flush=True)

    for a in allocations:
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
        n_te = len(te_cids)

        print(f"\n================================================================================", flush=True)
        print(f">>> ALLOCATION {a} (Train={n_tr}, Val={n_va}, Test={n_te})", flush=True)
        print(f"================================================================================", flush=True)

        for budget in budgets:
            for res in resolutions:
                t0_cfg = time.time()
                if not args.smoke:
                    existing_cfg = [
                        m for m in benchmark_metrics
                        if m["allocation"] == a and m["budget"] == budget and m["resolution"] == res
                    ]
                    if len(existing_cfg) >= 24:
                        print(f"--- Config Allocation {a} | Budget: {budget} | Resolution: {res} already complete ({len(existing_cfg)} rows). Skipping. ---", flush=True)
                        continue

                print(f"\n--- Running Config: Allocation {a} | Budget: {budget} | Resolution: {res} ---", flush=True)

                # Select active dictionaries
                if res == "5m":
                    cust_dict = cust_5m_all if budget == "all_days" else cust_5m_40d
                    static_b_dict = static_b_5m_all if budget == "all_days" else static_b_5m_40d
                    static_sm_dict = static_sm_5m_all if budget == "all_days" else static_sm_5m_40d
                else:
                    cust_dict = cust_15m_all if budget == "all_days" else cust_15m_40d
                    static_b_dict = static_b_15m_all if budget == "all_days" else static_b_15m_40d
                    static_sm_dict = static_sm_15m_all if budget == "all_days" else static_sm_15m_40d

                # Pre-stack static arrays for ordered cohort
                static_B_mat = np.stack([static_b_dict[cid] for cid in ordered_cids])
                static_SM_full_mat = np.stack([static_sm_dict[cid][0] for cid in ordered_cids])
                static_SM_sd_mat = np.stack([static_sm_dict[cid][1] for cid in ordered_cids])

                # 1. Train-only fill templates
                train_hr_list = [cust_dict[cid][1] for cid in tr_cids]
                fill_clock, fill_global = fit_fill_templates(train_hr_list)

                # 2. Sample <= 4,096 training days using exact seed formula
                seed_base = 20261014 + 100 * a + (10 if budget == "up_to_40d" else 0) + (1 if res == "15m" else 0)
                rng = np.random.default_rng(seed_base)

                n_target_sample = 4096 if not args.smoke else min(128, len(train_hr_list) * 2)
                days_per_user = max(1, n_target_sample // len(tr_cids))
                sample_days = []
                for cid in tr_cids:
                    _, hr = cust_dict[cid]
                    n_d = len(hr)
                    pick = rng.choice(n_d, min(days_per_user, n_d), replace=False)
                    sample_days.append(hr[pick])
                sample_raw = np.concatenate(sample_days)
                if len(sample_raw) > n_target_sample:
                    sample_raw = sample_raw[rng.choice(len(sample_raw), n_target_sample, replace=False)]

                sample_mask = np.isfinite(sample_raw).astype(np.uint8)

                # Prepare raw sequence arrays for full cohort
                all_hr_raw = np.concatenate([cust_dict[cid][1] for cid in ordered_cids], axis=0)
                all_mask = np.isfinite(all_hr_raw).astype(np.uint8)
                cust_counts = [len(cust_dict[cid][1]) for cid in ordered_cids]

                # Tracking for combination arms
                seq_clock_saved = None
                seq_m_temporal_saved = None

                scores_tr_saved = {}  # {arm: {penalty_type: s_tr}}
                scores_va_saved = {}  # {arm: {penalty_type: s_va}}
                scores_te_saved = {}  # {arm: {penalty_type: s_te}}

                # -------------------------------------------------------------
                # ARM 1: H_clock (Baseline Clock Filling)
                # -------------------------------------------------------------
                t0_arm = time.time()
                sample_x_clock = np.where(sample_mask == 1, sample_raw, fill_clock).astype(np.float64)
                seed_numba(seed_base)
                mr_base_clock = mr_fit(sample_x_clock, num_features=1250, max_dilations_per_kernel=32)
                seed_numba(seed_base + 1)
                mr_diff_clock = mr_fit(np.diff(sample_x_clock, axis=1), num_features=1250, max_dilations_per_kernel=32)

                all_x_clock = np.where(all_mask == 1, all_hr_raw, fill_clock)

                seq_clock = transform_sequence_batched(
                    ordered_cids, cust_counts, all_x_clock, None, mr_base_clock, mr_diff_clock, mr_transform, args.batch_size
                )
                del all_x_clock
                seq_clock_saved = seq_clock.copy()

                X_full_clock = np.hstack([static_B_mat, seq_clock])
                metrics_h_clock, te_s_clock, va_s_clock, tr_s_clock, sel_alpha_clock = evaluate_feature_arm(
                    X_full_clock, n_tr, n_va, y_tr, y_va, y_te, ALPHAS_EXT
                )
                del X_full_clock, seq_clock
                t_arm_clock = time.time() - t0_arm

                scores_tr_saved["H_clock"] = tr_s_clock
                scores_va_saved["H_clock"] = va_s_clock
                scores_te_saved["H_clock"] = te_s_clock

                # Record metrics and predictions
                for rule_name, m_dict in metrics_h_clock.items():
                    benchmark_metrics.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_clock",
                        "penalty_type": rule_name,
                        "alpha": m_dict["alpha"],
                        "weight": m_dict["weight"],
                        "train_auc": m_dict["train_auc"],
                        "val_auc": m_dict["val_auc"],
                        "test_auc": m_dict["test_auc"],
                        "test_ap": m_dict["test_ap"],
                        "runtime_s": t_arm_clock,
                    })
                    alpha_and_weight_selection.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_clock",
                        "penalty_type": rule_name,
                        "selected_alpha": sel_alpha_clock[rule_name],
                        "selected_weight": np.nan,
                        "val_auc": m_dict["val_auc"],
                    })
                    test_prediction_chunks.append(pd.DataFrame({
                        "user_id": te_cids,
                        "y": y_te,
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_clock",
                        "penalty_type": rule_name,
                        "score": te_s_clock[rule_name],
                    }))

                print(
                    f"Arm H_clock: primary Test AUC={metrics_h_clock['ridge_selected_ext']['test_auc']:.4f} "
                    f"(Fixed={metrics_h_clock['ridge_fixed']['test_auc']:.4f}, Sel={metrics_h_clock['ridge_selected']['test_auc']:.4f}) in {t_arm_clock:.1f}s",
                    flush=True,
                )

                # Allocation 0 Parity Gate
                if a == 0 and ref_df is not None and not args.smoke:
                    verify_parity_gate(ref_df, a, budget, res, te_cids, te_s_clock, y_te)

                # -------------------------------------------------------------
                # ARM 2: H_global (Global Median Filling)
                # -------------------------------------------------------------
                t0_arm = time.time()
                sample_x_glob = np.where(sample_mask == 1, sample_raw, fill_global).astype(np.float64)
                seed_numba(seed_base)
                mr_base_glob = mr_fit(sample_x_glob, num_features=1250, max_dilations_per_kernel=32)
                seed_numba(seed_base + 1)
                mr_diff_glob = mr_fit(np.diff(sample_x_glob, axis=1), num_features=1250, max_dilations_per_kernel=32)

                all_x_glob = np.where(all_mask == 1, all_hr_raw, fill_global)

                seq_glob = transform_sequence_batched(
                    ordered_cids, cust_counts, all_x_glob, None, mr_base_glob, mr_diff_glob, mr_transform, args.batch_size
                )
                X_full_glob = np.hstack([static_B_mat, seq_glob])
                del all_x_glob, seq_glob

                metrics_h_glob, te_s_glob, va_s_glob, tr_s_glob, sel_alpha_glob = evaluate_feature_arm(
                    X_full_glob, n_tr, n_va, y_tr, y_va, y_te, ALPHAS_EXT
                )
                del X_full_glob
                t_arm_glob = time.time() - t0_arm

                for rule_name, m_dict in metrics_h_glob.items():
                    benchmark_metrics.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_global",
                        "penalty_type": rule_name,
                        "alpha": m_dict["alpha"],
                        "weight": m_dict["weight"],
                        "train_auc": m_dict["train_auc"],
                        "val_auc": m_dict["val_auc"],
                        "test_auc": m_dict["test_auc"],
                        "test_ap": m_dict["test_ap"],
                        "runtime_s": t_arm_glob,
                    })
                    alpha_and_weight_selection.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_global",
                        "penalty_type": rule_name,
                        "selected_alpha": sel_alpha_glob[rule_name],
                        "selected_weight": np.nan,
                        "val_auc": m_dict["val_auc"],
                    })
                    test_prediction_chunks.append(pd.DataFrame({
                        "user_id": te_cids,
                        "y": y_te,
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_global",
                        "penalty_type": rule_name,
                        "score": te_s_glob[rule_name],
                    }))

                print(f"Arm H_global: primary Test AUC={metrics_h_glob['ridge_selected_ext']['test_auc']:.4f} in {t_arm_glob:.1f}s", flush=True)

                # -------------------------------------------------------------
                # ARM 3: H_clean_diff (Clock filling + Piecewise clean difference)
                # -------------------------------------------------------------
                t0_arm = time.time()
                sample_diff_clean = compute_clean_diff(sample_raw, sample_mask).astype(np.float64)
                seed_numba(seed_base + 1)
                mr_diff_clean = mr_fit(sample_diff_clean, num_features=1250, max_dilations_per_kernel=32)

                all_diff_clean = compute_clean_diff(all_hr_raw, all_mask)
                mean_diff_clean, sd_diff_clean = transform_diff_only_batched(
                    ordered_cids, cust_counts, all_hr_raw.shape[1], all_diff_clean, mr_diff_clean, args.batch_size
                )
                del all_diff_clean
                d_half = mean_diff_clean.shape[1]
                seq_clean = np.hstack([
                    seq_clock_saved[:, :d_half],
                    mean_diff_clean,
                    seq_clock_saved[:, 2 * d_half : 3 * d_half],
                    sd_diff_clean,
                ])
                del mean_diff_clean, sd_diff_clean
                X_full_clean = np.hstack([static_B_mat, seq_clean])
                del seq_clean

                metrics_h_clean, te_s_clean, va_s_clean, tr_s_clean, sel_alpha_clean = evaluate_feature_arm(
                    X_full_clean, n_tr, n_va, y_tr, y_va, y_te, ALPHAS_EXT
                )
                del X_full_clean
                t_arm_clean = time.time() - t0_arm

                for rule_name, m_dict in metrics_h_clean.items():
                    benchmark_metrics.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_clean_diff",
                        "penalty_type": rule_name,
                        "alpha": m_dict["alpha"],
                        "weight": m_dict["weight"],
                        "train_auc": m_dict["train_auc"],
                        "val_auc": m_dict["val_auc"],
                        "test_auc": m_dict["test_auc"],
                        "test_ap": m_dict["test_ap"],
                        "runtime_s": t_arm_clean,
                    })
                    alpha_and_weight_selection.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_clean_diff",
                        "penalty_type": rule_name,
                        "selected_alpha": sel_alpha_clean[rule_name],
                        "selected_weight": np.nan,
                        "val_auc": m_dict["val_auc"],
                    })
                    test_prediction_chunks.append(pd.DataFrame({
                        "user_id": te_cids,
                        "y": y_te,
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_clean_diff",
                        "penalty_type": rule_name,
                        "score": te_s_clean[rule_name],
                    }))

                print(f"Arm H_clean_diff: primary Test AUC={metrics_h_clean['ridge_selected_ext']['test_auc']:.4f} in {t_arm_clean:.1f}s", flush=True)

                # -------------------------------------------------------------
                # ARM 4: H_short_interp (Span-matched short-gap linear interpolation)
                # -------------------------------------------------------------
                t0_arm = time.time()
                sample_x_interp = interpolate_short_gaps(
                    sample_raw, mask=sample_mask, clock_template=fill_clock, resolution=res
                ).astype(np.float64)
                seed_numba(seed_base)
                mr_base_interp = mr_fit(sample_x_interp, num_features=1250, max_dilations_per_kernel=32)
                seed_numba(seed_base + 1)
                mr_diff_interp = mr_fit(np.diff(sample_x_interp, axis=1), num_features=1250, max_dilations_per_kernel=32)

                all_x_interp = interpolate_short_gaps(
                    all_hr_raw, mask=all_mask, clock_template=fill_clock, resolution=res
                )

                seq_interp = transform_sequence_batched(
                    ordered_cids, cust_counts, all_x_interp, None, mr_base_interp, mr_diff_interp, mr_transform, args.batch_size
                )
                X_full_interp = np.hstack([static_B_mat, seq_interp])
                del all_x_interp, seq_interp

                metrics_h_interp, te_s_interp, va_s_interp, tr_s_interp, sel_alpha_interp = evaluate_feature_arm(
                    X_full_interp, n_tr, n_va, y_tr, y_va, y_te, ALPHAS_EXT
                )
                del X_full_interp
                t_arm_interp = time.time() - t0_arm

                for rule_name, m_dict in metrics_h_interp.items():
                    benchmark_metrics.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_short_interp",
                        "penalty_type": rule_name,
                        "alpha": m_dict["alpha"],
                        "weight": m_dict["weight"],
                        "train_auc": m_dict["train_auc"],
                        "val_auc": m_dict["val_auc"],
                        "test_auc": m_dict["test_auc"],
                        "test_ap": m_dict["test_ap"],
                        "runtime_s": t_arm_interp,
                    })
                    alpha_and_weight_selection.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_short_interp",
                        "penalty_type": rule_name,
                        "selected_alpha": sel_alpha_interp[rule_name],
                        "selected_weight": np.nan,
                        "val_auc": m_dict["val_auc"],
                    })
                    test_prediction_chunks.append(pd.DataFrame({
                        "user_id": te_cids,
                        "y": y_te,
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_short_interp",
                        "penalty_type": rule_name,
                        "score": te_s_interp[rule_name],
                    }))

                print(f"Arm H_short_interp: primary Test AUC={metrics_h_interp['ridge_selected_ext']['test_auc']:.4f} in {t_arm_interp:.1f}s", flush=True)

                # -------------------------------------------------------------
                # ARM 5: M_static (Static Observation-Mask Baseline)
                # -------------------------------------------------------------
                t0_arm = time.time()
                X_full_m_static = static_SM_full_mat.copy()

                metrics_m_stat, te_s_m_stat, va_s_m_stat, tr_s_m_stat, sel_alpha_m_stat = evaluate_feature_arm(
                    X_full_m_static, n_tr, n_va, y_tr, y_va, y_te, ALPHAS_EXT
                )
                del X_full_m_static
                t_arm_m_stat = time.time() - t0_arm

                for rule_name, m_dict in metrics_m_stat.items():
                    benchmark_metrics.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "M_static",
                        "penalty_type": rule_name,
                        "alpha": m_dict["alpha"],
                        "weight": m_dict["weight"],
                        "train_auc": m_dict["train_auc"],
                        "val_auc": m_dict["val_auc"],
                        "test_auc": m_dict["test_auc"],
                        "test_ap": m_dict["test_ap"],
                        "runtime_s": t_arm_m_stat,
                    })
                    alpha_and_weight_selection.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "M_static",
                        "penalty_type": rule_name,
                        "selected_alpha": sel_alpha_m_stat[rule_name],
                        "selected_weight": np.nan,
                        "val_auc": m_dict["val_auc"],
                    })
                    test_prediction_chunks.append(pd.DataFrame({
                        "user_id": te_cids,
                        "y": y_te,
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "M_static",
                        "penalty_type": rule_name,
                        "score": te_s_m_stat[rule_name],
                    }))

                print(f"Arm M_static: primary Test AUC={metrics_m_stat['ridge_selected_ext']['test_auc']:.4f} in {t_arm_m_stat:.1f}s", flush=True)

                # -------------------------------------------------------------
                # ARM 6: M_temporal (MultiRocket Mask + Delta Mask + Static Mask)
                # -------------------------------------------------------------
                t0_arm = time.time()
                sample_m = sample_mask.astype(np.float64)
                sample_diff_m = np.diff(sample_m, axis=1).astype(np.float64)
                seed_numba(seed_base + 50)
                mr_base_m = mr_fit(sample_m, num_features=1250, max_dilations_per_kernel=32)
                seed_numba(seed_base + 51)
                mr_diff_m = mr_fit(sample_diff_m, num_features=1250, max_dilations_per_kernel=32)

                seq_m_temporal = transform_sequence_batched(
                    ordered_cids, cust_counts, all_mask, None, mr_base_m, mr_diff_m, mr_transform, args.batch_size
                )
                seq_m_temporal_saved = seq_m_temporal.copy()

                X_full_m_temp = np.hstack([static_SM_full_mat, seq_m_temporal])

                metrics_m_temp, te_s_m_temp, va_s_m_temp, tr_s_m_temp, sel_alpha_m_temp = evaluate_feature_arm(
                    X_full_m_temp, n_tr, n_va, y_tr, y_va, y_te, ALPHAS_EXT
                )
                del X_full_m_temp, seq_m_temporal
                t_arm_m_temp = time.time() - t0_arm

                scores_tr_saved["M_temporal"] = tr_s_m_temp
                scores_va_saved["M_temporal"] = va_s_m_temp
                scores_te_saved["M_temporal"] = te_s_m_temp

                for rule_name, m_dict in metrics_m_temp.items():
                    benchmark_metrics.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "M_temporal",
                        "penalty_type": rule_name,
                        "alpha": m_dict["alpha"],
                        "weight": m_dict["weight"],
                        "train_auc": m_dict["train_auc"],
                        "val_auc": m_dict["val_auc"],
                        "test_auc": m_dict["test_auc"],
                        "test_ap": m_dict["test_ap"],
                        "runtime_s": t_arm_m_temp,
                    })
                    alpha_and_weight_selection.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "M_temporal",
                        "penalty_type": rule_name,
                        "selected_alpha": sel_alpha_m_temp[rule_name],
                        "selected_weight": np.nan,
                        "val_auc": m_dict["val_auc"],
                    })
                    test_prediction_chunks.append(pd.DataFrame({
                        "user_id": te_cids,
                        "y": y_te,
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "M_temporal",
                        "penalty_type": rule_name,
                        "score": te_s_m_temp[rule_name],
                    }))

                print(f"Arm M_temporal: primary Test AUC={metrics_m_temp['ridge_selected_ext']['test_auc']:.4f} in {t_arm_m_temp:.1f}s", flush=True)

                # -------------------------------------------------------------
                # ARM 7: H_plus_M_concat (Concatenated Feature Blocks)
                # -------------------------------------------------------------
                t0_arm = time.time()
                # Reusing seq_clock_saved and seq_m_temporal_saved without re-running mr_transform!
                X_full_concat = np.hstack([static_B_mat, static_SM_sd_mat, seq_clock_saved, seq_m_temporal_saved])
                del seq_clock_saved, seq_m_temporal_saved

                metrics_concat, te_s_concat, va_s_concat, tr_s_concat, sel_alpha_concat = evaluate_feature_arm(
                    X_full_concat, n_tr, n_va, y_tr, y_va, y_te, ALPHAS_EXT
                )
                del X_full_concat
                t_arm_concat = time.time() - t0_arm

                for rule_name, m_dict in metrics_concat.items():
                    benchmark_metrics.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_plus_M_concat",
                        "penalty_type": rule_name,
                        "alpha": m_dict["alpha"],
                        "weight": m_dict["weight"],
                        "train_auc": m_dict["train_auc"],
                        "val_auc": m_dict["val_auc"],
                        "test_auc": m_dict["test_auc"],
                        "test_ap": m_dict["test_ap"],
                        "runtime_s": t_arm_concat,
                    })
                    alpha_and_weight_selection.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_plus_M_concat",
                        "penalty_type": rule_name,
                        "selected_alpha": sel_alpha_concat[rule_name],
                        "selected_weight": np.nan,
                        "val_auc": m_dict["val_auc"],
                    })
                    test_prediction_chunks.append(pd.DataFrame({
                        "user_id": te_cids,
                        "y": y_te,
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_plus_M_concat",
                        "penalty_type": rule_name,
                        "score": te_s_concat[rule_name],
                    }))

                print(f"Arm H_plus_M_concat: primary Test AUC={metrics_concat['ridge_selected_ext']['test_auc']:.4f} in {t_arm_concat:.1f}s", flush=True)

                # -------------------------------------------------------------
                # ARM 8: H_plus_M_stack (21-point Convex Blend of Ridge Scores)
                # -------------------------------------------------------------
                t0_arm = time.time()
                for rule_name in ["ridge_selected_ext", "ridge_selected", "ridge_fixed"]:
                    s_tr_hr = scores_tr_saved["H_clock"][rule_name]
                    s_va_hr = scores_va_saved["H_clock"][rule_name]
                    s_te_hr = scores_te_saved["H_clock"][rule_name]

                    s_tr_m = scores_tr_saved["M_temporal"][rule_name]
                    s_va_m = scores_va_saved["M_temporal"][rule_name]
                    s_te_m = scores_te_saved["M_temporal"][rule_name]

                    m_stack, best_w, s_te_stack = evaluate_stacking(
                        s_tr_hr, s_va_hr, s_te_hr, s_tr_m, s_va_m, s_te_m, y_tr, y_va, y_te
                    )
                    t_arm_stack = time.time() - t0_arm

                    benchmark_metrics.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_plus_M_stack",
                        "penalty_type": rule_name,
                        "alpha": np.nan,
                        "weight": best_w,
                        "train_auc": m_stack["train_auc"],
                        "val_auc": m_stack["val_auc"],
                        "test_auc": m_stack["test_auc"],
                        "test_ap": m_stack["test_ap"],
                        "runtime_s": t_arm_stack,
                    })
                    alpha_and_weight_selection.append({
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_plus_M_stack",
                        "penalty_type": rule_name,
                        "selected_alpha": np.nan,
                        "selected_weight": best_w,
                        "val_auc": m_stack["val_auc"],
                    })
                    test_prediction_chunks.append(pd.DataFrame({
                        "user_id": te_cids,
                        "y": y_te,
                        "allocation": a,
                        "resolution": res,
                        "budget": budget,
                        "arm": "H_plus_M_stack",
                        "penalty_type": rule_name,
                        "score": s_te_stack,
                    }))

                print(
                    f"Arm H_plus_M_stack: primary Test AUC={benchmark_metrics[-3]['test_auc']:.4f} "
                    f"(weight w*={benchmark_metrics[-3]['weight']:.2f}) in {t_arm_stack:.1f}s",
                    flush=True,
                )

                del all_hr_raw, all_mask, static_B_mat, static_SM_full_mat, static_SM_sd_mat
                print(f"--- Config Finished in {time.time() - t0_cfg:.1f}s ---", flush=True)
                if not args.smoke:
                    checkpoint_results(benchmark_metrics, alpha_and_weight_selection, test_prediction_chunks)

        print(f"\nAllocation {a} completed in {time.time() - t0_alloc:.1f}s.", flush=True)
        # Checkpoint results after each allocation
        if not args.smoke:
            checkpoint_results(benchmark_metrics, alpha_and_weight_selection, test_prediction_chunks)

    print("\n================================================================================", flush=True)
    print("STAGE 1 BENCHMARK EXECUTION COMPLETE FOR ALL REQUESTED ALLOCATIONS!", flush=True)
    print("================================================================================", flush=True)


if __name__ == "__main__":
    main()
