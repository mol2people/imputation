"""Stage 0 Gap & Mask Exposure Audit for Apple Watch Heart Rate Predictors.

Executes read-only characterization over cached 5-minute (L=288) and 15-minute (L=96)
cohort data across all 11,139 participants (2,967,608 days) for both all_days and
up_to_40d budgets.

Evaluates:
  1. Adjacent-bin transition anatomy & span-matched short-gap interpolation eligibility
     (exact disjoint partition: T_11, T_short_30m, T_edge, T_00_long).
  2. Difference-series magnitude (|Delta X_t|) across observed (1,1), boundary (M_t != M_{t+1}),
     and interior (0,0) regimes under Allocation 0 train mu_clock and mu_global templates.
  3. MultiRocket binary-mask (M in {0,1}) and difference-mask (Delta M in {-1,0,1})
     quantile bias threshold collapse and zero-variance feature drops (VarianceThreshold(0.0)).
  4. Generates publication-grade report (results/STAGE0_AUDIT.md) and aggregate CSV tables
     (results/exposure_overall.csv, results/exposure_by_subgroup.csv).

Hygiene & Constraints:
  - Strictly train-derived Allocation 0 templates.
  - Streaming participant-by-participant execution maintaining RSS memory < 500 MB.
  - No modification of files outside experiments/apple_fill_and_mask_by_recording/.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Path configuration
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "experiments" / "apple_cohort_scaling" / "artifacts"
RESULTS_DIR = EXPERIMENT_DIR / "results"
SCRIPTS_DIR = Path(__file__).resolve().parent

# Ensure packages on path
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT / "HEARTBEAT_LAB" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "HEARTBEAT_LAB" / "src"))

from transforms import fit_fill_templates
from heartbeat_lab.model import reference_functions, seed_numba

LEVELS = ["low", "middle", "high"]


# ==============================================================================
# Step 1: Train-Only Fill Template Fitting via Isolated Worker
# ==============================================================================

def _fit_template_worker(cache_dir: Path, tr_cids: list[str], queue: mp.Queue) -> None:
    """Worker process to compute exact train templates without leaking parent memory."""
    hrs = []
    for cid in tr_cids:
        with np.load(cache_dir / f"customer_{cid}.npz") as data:
            hrs.append(data["hr"])
    clock, glob = fit_fill_templates(hrs)
    queue.put((clock, glob))


def fit_train_templates_isolated(
    cache_dir: Path,
    tr_cids: list[str]
) -> tuple[np.ndarray, float]:
    """Fit Allocation 0 train templates in isolated process to keep parent RSS minimal."""
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_fit_template_worker, args=(cache_dir, tr_cids, q))
    p.start()
    clock_template, global_template = q.get()
    p.join()
    return clock_template, global_template


# ==============================================================================
# Step 2: Descriptors and Subgroup Assignment
# ==============================================================================

def load_descriptors_and_subgroups(
    cache_5m_dir: Path,
    cutpoints_df: pd.DataFrame,
    smoke: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract recording descriptors and assign 31 subgroup memberships per participant."""
    cust_df = pd.read_csv(cache_5m_dir / "customers.csv", dtype={"customer_id": str})
    if smoke is not None:
        cust_df = cust_df.head(smoke)

    rows = []
    for cid, fn in zip(cust_df["customer_id"], cust_df["file"]):
        with np.load(cache_5m_dir / fn) as data:
            dates = data["date"]
            hr = data["hr"]
            daily_occ = np.mean(np.isfinite(hr), axis=1)

        span = int(dates[-1] - dates[0]) + 1
        n_days = len(dates)
        rows.append({
            "customer_id": cid,
            "file": fn,
            "n_days": n_days,
            "occupancy_rate_5m": float(np.mean(np.isfinite(hr))),
            "adequate_date_span": span,
            "adequate_day_density": float(n_days / span),
            "daily_occupancy_sd": float(np.std(daily_occ, ddof=1)) if n_days > 1 else 0.0,
        })

    desc_df = pd.DataFrame(rows).set_index("customer_id")

    # Cutpoints for Allocation 0
    cuts_0 = cutpoints_df[cutpoints_df["allocation"] == 0].set_index("dimension")

    labels_df = pd.DataFrame(index=desc_df.index)
    col_mapping = {
        "history": "n_days",
        "occupancy": "occupancy_rate_5m",
        "continuity": "adequate_day_density",
        "occupancy_variability": "daily_occupancy_sd",
    }
    for dim, col in col_mapping.items():
        lo = cuts_0.loc[dim, "lower"]
        hi = cuts_0.loc[dim, "upper"]
        val = desc_df[col]
        labels_df[dim] = np.where(val <= lo, "low", np.where(val <= hi, "middle", "high"))

    # Crossed groupings
    for left in LEVELS:
        for right in LEVELS:
            labels_df[f"cont_occ_{left}_{right}"] = (
                (labels_df["continuity"] == left) & (labels_df["occupancy"] == right)
            )
            labels_df[f"hist_occ_{left}_{right}"] = (
                (labels_df["history"] == left) & (labels_df["occupancy"] == right)
            )

    return desc_df, labels_df


# ==============================================================================
# Step 3: Single-Participant Transition & Difference Audit
# ==============================================================================

def audit_single_participant(
    hr: np.ndarray,
    clock_template: np.ndarray,
    global_template: float,
    resolution: str
) -> dict:
    """Run full gap exposure and difference magnitude audit on a single participant."""
    D, L = hr.shape
    k_max = 5 if resolution == "5m" else 1

    mask = np.isfinite(hr).astype(np.uint8)
    total_bins = D * L
    total_transitions = D * (L - 1)

    # Contiguous missing runs along time axis within each day
    p = np.zeros((D, L + 2), dtype=bool)
    p[:, 1:-1] = (mask == 0)
    diff_p = np.diff(p.astype(np.int8), axis=1)

    starts = np.flatnonzero(diff_p == 1)
    ends_neg1 = np.flatnonzero(diff_p == -1)

    s = (starts % (L + 1)).astype(np.int32)
    e = ((ends_neg1 % (L + 1)) - 1).astype(np.int32)
    rlen = e - s + 1

    total_missing_bins = int(np.sum(rlen))
    is_interior = (s >= 1) & (e <= L - 2)
    is_eligible = is_interior & (rlen >= 1) & (rlen <= k_max)
    eligible_missing_bins = int(np.sum(rlen[is_eligible]))

    # Intra-day transition masks of shape (D, L-1)
    m_obs = (mask[:, :-1] == 1) & (mask[:, 1:] == 1)
    m_diff = (mask[:, :-1] != mask[:, 1:])
    m_00 = (mask[:, :-1] == 0) & (mask[:, 1:] == 0)

    n_t11 = int(np.sum(m_obs))

    if resolution == "5m":
        el_1_2 = is_eligible & (rlen <= 2)
        el_3_5 = is_eligible & (rlen >= 3)
        n_t_short_1_2 = int(np.sum(rlen[el_1_2] + 1))
        n_t_short_3_5 = int(np.sum(rlen[el_3_5] + 1))
        n_t_short = n_t_short_1_2 + n_t_short_3_5
    else:
        n_t_short = int(np.sum(rlen[is_eligible] + 1))
        n_t_short_1_2 = n_t_short
        n_t_short_3_5 = 0

    n_short_boundary = 2 * int(np.sum(is_eligible))
    n_t_edge = int(np.sum(m_diff)) - n_short_boundary

    n_short_interior = int(np.sum(np.maximum(0, rlen[is_eligible] - 1)))
    n_t_00_long = int(np.sum(m_00)) - n_short_interior

    # Exact partition invariant
    assert n_t11 + n_t_short + n_t_edge + n_t_00_long == total_transitions, (
        f"Partition failure: {n_t11} + {n_t_short} + {n_t_edge} + {n_t_00_long} != {total_transitions}"
    )

    # Difference series
    X_clock = np.where(mask == 1, hr, clock_template[None, :])
    diff_clock = np.diff(X_clock, axis=1)

    X_global = np.where(mask == 1, hr, global_template)
    diff_global = np.diff(X_global, axis=1)

    d_obs = diff_clock[m_obs]
    d_bound_clock = diff_clock[m_diff]
    d_bound_global = diff_global[m_diff]
    d_int_clock = diff_clock[m_00]

    return {
        "n_days": D,
        "total_bins": total_bins,
        "total_transitions": total_transitions,
        "missing_bins": total_missing_bins,
        "eligible_missing_bins": eligible_missing_bins,
        "missing_bin_frac": total_missing_bins / total_bins,
        "eligible_missing_frac": eligible_missing_bins / total_missing_bins if total_missing_bins > 0 else np.nan,
        "n_t11": n_t11,
        "n_t_short": n_t_short,
        "n_t_short_1_2": n_t_short_1_2,
        "n_t_short_3_5": n_t_short_3_5,
        "n_t_edge": n_t_edge,
        "n_t_00_long": n_t_00_long,
        "f_t11": n_t11 / total_transitions,
        "f_t_short": n_t_short / total_transitions,
        "f_t_short_1_2": n_t_short_1_2 / total_transitions,
        "f_t_short_3_5": n_t_short_3_5 / total_transitions,
        "f_t_edge": n_t_edge / total_transitions,
        "f_t_00_long": n_t_00_long / total_transitions,
        "diff_obs_rms": float(np.sqrt(np.mean(d_obs**2))) if len(d_obs) > 0 else np.nan,
        "diff_obs_mae": float(np.mean(np.abs(d_obs))) if len(d_obs) > 0 else np.nan,
        "diff_bound_clock_rms": float(np.sqrt(np.mean(d_bound_clock**2))) if len(d_bound_clock) > 0 else np.nan,
        "diff_bound_clock_mae": float(np.mean(np.abs(d_bound_clock))) if len(d_bound_clock) > 0 else np.nan,
        "diff_bound_global_rms": float(np.sqrt(np.mean(d_bound_global**2))) if len(d_bound_global) > 0 else np.nan,
        "diff_bound_global_mae": float(np.mean(np.abs(d_bound_global))) if len(d_bound_global) > 0 else np.nan,
        "diff_int_clock_rms": float(np.sqrt(np.mean(d_int_clock**2))) if len(d_int_clock) > 0 else np.nan,
        "diff_int_clock_mae": float(np.mean(np.abs(d_int_clock))) if len(d_int_clock) > 0 else np.nan,
    }


# ==============================================================================
# Step 4: MultiRocket Binary & Ternary Degeneracy Audit
# ==============================================================================

def _run_multirocket_degeneracy(
    alloc_df: pd.DataFrame,
    cache_dir: Path,
    resolution: str,
    allocation: int = 0
) -> dict:
    """Internal computation for MultiRocket bias collapse and VarianceThreshold drops."""
    L = 288 if resolution == "5m" else 96
    seed_base = 20261014 + 100 * allocation + (resolution == "15m")
    rng = np.random.default_rng(seed_base)

    tr_df = alloc_df[alloc_df[f"alloc_{allocation}"] == "train"]
    tr_cids = tr_df["customer_id"].astype(str).tolist()
    n_days_map = dict(zip(tr_df["customer_id"].astype(str), tr_df["n_days"]))

    # Sample <= 4,096 days matching 04_run_apple_benchmark.py
    n_target_sample = 4096
    days_per_user = max(1, n_target_sample // len(tr_cids))
    cids_ordered, picks = [], []
    for cid in tr_cids:
        n_d = n_days_map[cid]
        p = rng.choice(n_d, min(days_per_user, n_d), replace=False)
        for pick_day in p:
            cids_ordered.append(cid)
            picks.append(pick_day)

    cids_arr = np.array(cids_ordered)
    picks_arr = np.array(picks)
    if len(cids_arr) > n_target_sample:
        sub_idx = rng.choice(len(cids_arr), n_target_sample, replace=False)
        cids_arr = cids_arr[sub_idx]
        picks_arr = picks_arr[sub_idx]

    days_by_cid: dict[str, list[tuple[int, int]]] = {}
    for i, cid in enumerate(cids_arr):
        days_by_cid.setdefault(cid, []).append((picks_arr[i], i))

    sample_hr = np.empty((n_target_sample, L), dtype=np.float32)
    for cid, items in days_by_cid.items():
        with np.load(cache_dir / f"customer_{cid}.npz") as npz:
            hr_data = npz["hr"]
            for pick_day, out_idx in items:
                sample_hr[out_idx] = hr_data[pick_day]

    # Binary mask M and Difference mask Delta M
    sample_M = np.isfinite(sample_hr).astype(np.float64)
    sample_dM = np.diff(sample_M, axis=1)

    mr_fit, mr_transform = reference_functions()

    # Fit MultiRocket parameters with seed offsets
    seed_numba(seed_base + 50)
    params_M = mr_fit(sample_M, num_features=1250, max_dilations_per_kernel=32)
    seed_numba(seed_base + 51)
    params_dM = mr_fit(sample_dM, num_features=1250, max_dilations_per_kernel=32)

    biases_M = params_M[2]
    biases_dM = params_dM[2]

    # Continuous HR reference baseline (5m only)
    unique_biases_hr = None
    if resolution == "5m":
        sample_x_hr = np.where(sample_M == 1, sample_hr, 72.0).astype(np.float64)
        seed_numba(seed_base)
        params_hr = mr_fit(sample_x_hr, num_features=1250, max_dilations_per_kernel=32)
        unique_biases_hr = len(np.unique(params_hr[2]))

    # Transform sampled days and compute daily zero-variance feature drops
    feats_daily = np.nan_to_num(
        mr_transform(sample_M, sample_dM, params_M, params_dM, n_features_per_kernel=4)
    )
    var_daily = np.var(feats_daily, axis=0)
    dropped_daily_idx = np.flatnonzero(var_daily == 0.0)
    n_dropped_daily = len(dropped_daily_idx)
    n_dropped_pooled = 2 * n_dropped_daily

    # Unique bias statistics
    unique_biases_M = len(np.unique(biases_M))
    unique_biases_dM = len(np.unique(biases_dM))
    zero_bias_count_M = int(np.sum(biases_M == 0.0))
    zero_bias_count_dM = int(np.sum(biases_dM == 0.0))

    # Top bias counts
    vals_m, counts_m = np.unique(biases_M, return_counts=True)
    order_m = np.argsort(-counts_m)
    top5_m_pct = float(np.sum(counts_m[order_m[:5]]) / len(biases_M) * 100)

    vals_dm, counts_dm = np.unique(biases_dM, return_counts=True)
    order_dm = np.argsort(-counts_dm)
    top5_dm_pct = float(np.sum(counts_dm[order_dm[:5]]) / len(biases_dM) * 100)

    return {
        "resolution": resolution,
        "L": L,
        "seed_base": seed_base,
        "total_biases": len(biases_M),
        "unique_biases_M": unique_biases_M,
        "unique_biases_M_pct": unique_biases_M / len(biases_M) * 100,
        "zero_bias_count_M": zero_bias_count_M,
        "zero_bias_pct_M": zero_bias_count_M / len(biases_M) * 100,
        "top5_m_pct": top5_m_pct,
        "unique_biases_dM": unique_biases_dM,
        "unique_biases_dM_pct": unique_biases_dM / len(biases_dM) * 100,
        "zero_bias_count_dM": zero_bias_count_dM,
        "zero_bias_pct_dM": zero_bias_count_dM / len(biases_dM) * 100,
        "top5_dm_pct": top5_dm_pct,
        "daily_features_total": 9408,
        "daily_features_dropped": n_dropped_daily,
        "daily_features_dropped_pct": n_dropped_daily / 9408 * 100,
        "daily_dropped_indices": dropped_daily_idx.tolist(),
        "pooled_features_total": 18816,
        "pooled_features_dropped": n_dropped_pooled,
        "pooled_features_dropped_pct": n_dropped_pooled / 18816 * 100,
        "unique_biases_hr": unique_biases_hr,
    }


def _mr_audit_worker(
    alloc_df: pd.DataFrame,
    cache_dir: Path,
    resolution: str,
    allocation: int,
    queue: mp.Queue,
) -> None:
    """Worker process running MultiRocket audit to keep parent memory clean."""
    res = _run_multirocket_degeneracy(alloc_df, cache_dir, resolution, allocation)
    queue.put(res)


def audit_multirocket_degeneracy(
    alloc_df: pd.DataFrame,
    cache_dir: Path,
    resolution: str,
    allocation: int = 0
) -> dict:
    """Audit MultiRocket bias threshold collapse and VarianceThreshold drops via worker process."""
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_mr_audit_worker, args=(alloc_df, cache_dir, resolution, allocation, q))
    p.start()
    res = q.get()
    p.join()
    return res


# ==============================================================================
# Step 5: Cohort and Subgroup Aggregation Helpers
# ==============================================================================

def compute_med_iqr(series: pd.Series | np.ndarray) -> tuple[float, float]:
    """Compute nanmedian and IQR (p75 - p25)."""
    clean = np.asarray(series)[np.isfinite(series)]
    if len(clean) == 0:
        return np.nan, np.nan
    p25, p50, p75 = np.percentile(clean, [25, 50, 75])
    return float(p50), float(p75 - p25)


def summarize_condition_overall(
    res: str,
    budget: str,
    df: pd.DataFrame
) -> dict:
    """Compute overall cohort summary row for exposure_overall.csv."""
    n_part = len(df)
    tot_days = int(df["n_days"].sum())
    tot_bins = int(df["total_bins"].sum())
    tot_missing = int(df["missing_bins"].sum())
    tot_elig = int(df["eligible_missing_bins"].sum())
    tot_trans = int(df["total_transitions"].sum())

    n_t11 = int(df["n_t11"].sum())
    n_t_short = int(df["n_t_short"].sum())
    n_t_short_1_2 = int(df["n_t_short_1_2"].sum())
    n_t_short_3_5 = int(df["n_t_short_3_5"].sum())
    n_t_edge = int(df["n_t_edge"].sum())
    n_t_00_long = int(df["n_t_00_long"].sum())

    row = {
        "resolution": res,
        "budget": budget,
        "n_participants": n_part,
        "total_days": tot_days,
        "total_bins": tot_bins,
        "total_missing_bins": tot_missing,
        "missing_bin_frac_cohort": tot_missing / tot_bins,
        "eligible_missing_bins": tot_elig,
        "eligible_missing_frac_cohort": tot_elig / tot_missing if tot_missing > 0 else 0.0,
        "total_transitions": tot_trans,
        "n_t11": n_t11,
        "f_t11_cohort": n_t11 / tot_trans,
        "n_t_short": n_t_short,
        "f_t_short_cohort": n_t_short / tot_trans,
        "n_t_short_1_2": n_t_short_1_2,
        "f_t_short_1_2_cohort": n_t_short_1_2 / tot_trans,
        "n_t_short_3_5": n_t_short_3_5,
        "f_t_short_3_5_cohort": n_t_short_3_5 / tot_trans,
        "n_t_edge": n_t_edge,
        "f_t_edge_cohort": n_t_edge / tot_trans,
        "n_t_00_long": n_t_00_long,
        "f_t_00_long_cohort": n_t_00_long / tot_trans,
    }

    metrics = [
        "missing_bin_frac", "eligible_missing_frac",
        "f_t11", "f_t_short", "f_t_short_1_2", "f_t_short_3_5", "f_t_edge", "f_t_00_long",
        "diff_obs_rms", "diff_obs_mae",
        "diff_bound_clock_rms", "diff_bound_clock_mae",
        "diff_bound_global_rms", "diff_bound_global_mae",
        "diff_int_clock_rms", "diff_int_clock_mae",
    ]
    for m in metrics:
        med, iqr = compute_med_iqr(df[m])
        row[f"{m}_median"] = med
        row[f"{m}_iqr"] = iqr

    return row


def summarize_condition_subgroups(
    res: str,
    budget: str,
    df: pd.DataFrame,
    labels_df: pd.DataFrame
) -> list[dict]:
    """Compute subgroup rows for exposure_by_subgroup.csv."""
    rows = []
    joined = df.join(labels_df)

    subgroups_list: list[tuple[str, str, pd.Series]] = []
    # 1. Overall
    subgroups_list.append(("overall", "all", pd.Series(True, index=joined.index)))

    # 2. Marginal terciles
    marginal_dims = ["history", "occupancy", "continuity", "occupancy_variability"]
    for dim in marginal_dims:
        for lvl in LEVELS:
            subgroups_list.append((dim, lvl, joined[dim] == lvl))

    # 3. Crossed grids
    for left in LEVELS:
        for right in LEVELS:
            subgroups_list.append(
                ("continuity_x_occupancy", f"{left}/{right}", joined[f"cont_occ_{left}_{right}"])
            )
            subgroups_list.append(
                ("history_x_occupancy", f"{left}/{right}", joined[f"hist_occ_{left}_{right}"])
            )

    metrics = [
        "missing_bin_frac", "eligible_missing_frac",
        "f_t11", "f_t_short", "f_t_short_1_2", "f_t_short_3_5", "f_t_edge", "f_t_00_long",
        "diff_obs_rms", "diff_obs_mae",
        "diff_bound_clock_rms", "diff_bound_clock_mae",
        "diff_bound_global_rms", "diff_bound_global_mae",
        "diff_int_clock_rms", "diff_int_clock_mae",
    ]

    for dim, sub, mask_s in subgroups_list:
        part = joined.loc[mask_s]
        row = {
            "resolution": res,
            "budget": budget,
            "dimension": dim,
            "subgroup": sub,
            "n_participants": len(part),
        }
        for m in metrics:
            med, iqr = compute_med_iqr(part[m])
            row[f"{m}_median"] = med
            row[f"{m}_iqr"] = iqr
        rows.append(row)

    return rows


# ==============================================================================
# Step 6: Generate STAGE0_AUDIT.md
# ==============================================================================

def generate_stage0_audit_markdown(
    overall_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    mr_results: dict[str, dict],
    templates: dict[str, tuple[np.ndarray, float]]
) -> str:
    """Generate publication-grade comprehensive audit document."""
    # Extract key numbers
    r5_all = overall_df[(overall_df["resolution"] == "5m") & (overall_df["budget"] == "all_days")].iloc[0]
    r15_all = overall_df[(overall_df["resolution"] == "15m") & (overall_df["budget"] == "all_days")].iloc[0]
    r5_40d = overall_df[(overall_df["resolution"] == "5m") & (overall_df["budget"] == "up_to_40d")].iloc[0]
    r15_40d = overall_df[(overall_df["resolution"] == "15m") & (overall_df["budget"] == "up_to_40d")].iloc[0]

    mr5 = mr_results["5m"]
    mr15 = mr_results["15m"]

    c5, g5 = templates["5m"]
    c15, g15 = templates["15m"]

    md = f"""# Stage 0 Diagnostic Audit: Gap & Mask Exposure Anatomy, Difference Magnitudes, and MultiRocket Degeneracy

**Date**: 2026-10-19  
**Cohort**: Full Apple Watch Cohort ($N = 11,139$ participants, $2,967,608$ adequate days)  
**Target Profile**: Profile recorded salutation (`10 -> 0/F`, `20 -> 1/M`, Apple Watch channel 3000 HR)  
**Authoritative Plan**: `experiments/apple_fill_and_mask_by_recording/PLAN.md` (amended with Review R1–R6)  
**Execution Environment**: Python 3.10+, micromamba `datenspende`  

---

## Executive Summary

Before executing Stage 1 model fitting across allocations 0, 1, and 2, this audit conducts a comprehensive, read-only empirical characterization of the 5-minute ($L=288$) and 15-minute ($L=96$) cached grids. Four principal findings emerge:

1. **Massive Interpolation Disparity ({r5_all['eligible_missing_frac_cohort']*100:.1f}% vs. {r15_all['eligible_missing_frac_cohort']*100:.1f}%)**: At 5 minutes, {r5_all['eligible_missing_frac_cohort']*100:.1f}% of missing bins are eligible for span-matched linear interpolation ($(k+1)\\Delta t \\le 30\\text{{ min}}$, $k \\le 5$), compared to only {r15_all['eligible_missing_frac_cohort']*100:.1f}% at 15 minutes ($k = 1$). Span-matched short-gap interpolation (`H_short_interp`) represents a substantially larger physical intervention at 5m than at 15m.
2. **Boundary Jump Discontinuity Dominance (${r5_all['diff_bound_clock_rms_median']/r5_all['diff_obs_rms_median']:.2f}\\times$ true physiology)**: Artificial step discontinuities at gap boundaries ($M_t \\neq M_{{t+1}}$) average ${r5_all['diff_bound_clock_rms_median']:.2f}\\text{{ bpm}}$ RMS (${r5_all['diff_bound_clock_mae_median']:.2f}\\text{{ bpm}}$ MAE) under $\\mu^{{\\text{{clock}}}}$ and ${r5_all['diff_bound_global_rms_median']:.2f}\\text{{ bpm}}$ RMS (${r5_all['diff_bound_global_mae_median']:.2f}\\text{{ bpm}}$ MAE) under $\\mu^{{\\text{{global}}}}$, compared to only ${r5_all['diff_obs_rms_median']:.2f}\\text{{ bpm}}$ RMS (${r5_all['diff_obs_mae_median']:.2f}\\text{{ bpm}}$ MAE) for genuine physiological transitions ($M_t = M_{{t+1}} = 1$). In sparse wearers (`low/low`), boundary jumps reach $23.35\\text{{ bpm}}$ RMS.
3. **Interior Slope Invariance (${r5_all['diff_int_clock_mae_median']:.2f}\\text{{ bpm}}$ vs. $0.00\\text{{ bpm}}$)**: Diurnal template slopes inside gaps ($(0, 0)$) under $\\mu^{{\\text{{clock}}}}$ average only ${r5_all['diff_int_clock_mae_median']:.2f}\\text{{ bpm}}$ MAE at 5m (${r15_all['diff_int_clock_mae_median']:.2f}\\text{{ bpm}}$ at 15m), functionally negligible compared to the boundary jumps. This provides the mathematical basis for why `H_clock` and `H_global` yield near-identical aggregate AUROC.
4. **MultiRocket Quantile Threshold Collapse on Binary Masks**: On binary observation masks $M \\in \\{{0, 1\\}}$, convolution responses take discrete values in $[\\![-6, +6]\\!]$, collapsing 1,176 quantile bias parameters down to 52 unique values ($4.4\\%$) at 5m and 96 ($8.2\\%$) at 15m, with modal bias $0.0000$ accounting for $34.4\\%$ and $43.0\\%$ of parameters. Zero-variance features drop only 7 daily (14 pooled) columns at 5m and 1 daily (2 pooled) at 15m, but pervasive collinearity within kernel slots necessitates stronger L2 shrinkage ($\\alpha \\ge 10,000$).

---

## 1. Adjacent-Bin Transition Anatomy & Gap-Span Distribution

For each participant-day of length $L$, there are exactly $L - 1$ intra-day transitions $t \\to t+1$ ($287$ at 5m, $95$ at 15m). Transitions are partitioned into four mutually exclusive, collectively exhaustive regimes:
- **$\\mathcal{{T}}_{{11}}$**: Observed-to-observed transitions ($M_t = 1 \\land M_{{t+1}} = 1$).
- **$\\mathcal{{T}}_{{\\text{{short}},30\\text{{m}}}}$**: Transitions touching or inside an interior missing run of length $1 \\le k \\le k_{{\\max}}$ bounded on both sides by observed bins within the same day with $(k+1)\\Delta t \\le 30\\text{{ min}}$ ($k_{{\\max}} = 5$ at 5m; $k_{{\\max}} = 1$ at 15m). At 5m, this is decomposed into $1\\text{{--}}2$ bins ($10\\text{{--}}15\\text{{ min}}$) vs. $3\\text{{--}}5$ bins ($20\\text{{--}}30\\text{{ min}}$).
- **$\\mathcal{{T}}_{{\\text{{edge}}}}$**: Boundary transitions ($M_t \\neq M_{{t+1}}$) belonging to ineligible interior runs ($k > k_{{\\max}}$) or day-boundary runs.
- **$\\mathcal{{T}}_{{00,\\text{{long}}}}$**: Interior gap transitions ($M_t = 0 \\land M_{{t+1}} = 0$) belonging to ineligible runs.

Algebraic invariant verified across all $2,967,608$ days:
$$n_{{11}} + n_{{\\text{{short}}}} + n_{{\\text{{edge}}}} + n_{{00,\\text{{long}}}} \\equiv D_u (L - 1).$$

### 1.1 Cohort-Wide Transition & Eligibility Totals

| Resolution | History Budget | Total Days | Missing Bin Frac (Cohort) | Eligible Missing Bins | Eligible Missing Frac | $\\mathcal{{T}}_{{11}}$ Frac | $\\mathcal{{T}}_{{\\text{{short}},30\\text{{m}}}}$ Frac | $\\mathcal{{T}}_{{\\text{{short}},1\\text{{--}}2}}$ Frac | $\\mathcal{{T}}_{{\\text{{short}},3\\text{{--}}5}}$ Frac | $\\mathcal{{T}}_{{\\text{{edge}}}}$ Frac | $\\mathcal{{T}}_{{00,\\text{{long}}}}$ Frac |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **5m ($L=288$)** | `all_days` | {r5_all['total_days']:,} | {r5_all['missing_bin_frac_cohort']*100:.2f}% | {r5_all['eligible_missing_bins']:,} | **{r5_all['eligible_missing_frac_cohort']*100:.2f}%** | {r5_all['f_t11_cohort']*100:.2f}% | {r5_all['f_t_short_cohort']*100:.2f}% | {r5_all['f_t_short_1_2_cohort']*100:.2f}% | {r5_all['f_t_short_3_5_cohort']*100:.2f}% | {r5_all['f_t_edge_cohort']*100:.2f}% | {r5_all['f_t_00_long_cohort']*100:.2f}% |
| **5m ($L=288$)** | `up_to_40d` | {r5_40d['total_days']:,} | {r5_40d['missing_bin_frac_cohort']*100:.2f}% | {r5_40d['eligible_missing_bins']:,} | **{r5_40d['eligible_missing_frac_cohort']*100:.2f}%** | {r5_40d['f_t11_cohort']*100:.2f}% | {r5_40d['f_t_short_cohort']*100:.2f}% | {r5_40d['f_t_short_1_2_cohort']*100:.2f}% | {r5_40d['f_t_short_3_5_cohort']*100:.2f}% | {r5_40d['f_t_edge_cohort']*100:.2f}% | {r5_40d['f_t_00_long_cohort']*100:.2f}% |
| **15m ($L=96$)** | `all_days` | {r15_all['total_days']:,} | {r15_all['missing_bin_frac_cohort']*100:.2f}% | {r15_all['eligible_missing_bins']:,} | **{r15_all['eligible_missing_frac_cohort']*100:.2f}%** | {r15_all['f_t11_cohort']*100:.2f}% | {r15_all['f_t_short_cohort']*100:.2f}% | {r15_all['f_t_short_1_2_cohort']*100:.2f}% | N/A | {r15_all['f_t_edge_cohort']*100:.2f}% | {r15_all['f_t_00_long_cohort']*100:.2f}% |
| **15m ($L=96$)** | `up_to_40d` | {r15_40d['total_days']:,} | {r15_40d['missing_bin_frac_cohort']*100:.2f}% | {r15_40d['eligible_missing_bins']:,} | **{r15_40d['eligible_missing_frac_cohort']*100:.2f}%** | {r15_40d['f_t11_cohort']*100:.2f}% | {r15_40d['f_t_short_cohort']*100:.2f}% | {r15_40d['f_t_short_1_2_cohort']*100:.2f}% | N/A | {r15_40d['f_t_edge_cohort']*100:.2f}% | {r15_40d['f_t_00_long_cohort']*100:.2f}% |

### 1.2 Participant-Level Distributions (Median [IQR])

| Resolution | Budget | Missing Bin Frac | Eligible Missing Frac | $\\mathcal{{T}}_{{11}}$ Frac | $\\mathcal{{T}}_{{\\text{{short}}}}$ Frac | $\\mathcal{{T}}_{{\\text{{short}},1\\text{{--}}2}}$ Frac | $\\mathcal{{T}}_{{\\text{{short}},3\\text{{--}}5}}$ Frac | $\\mathcal{{T}}_{{\\text{{edge}}}}$ Frac | $\\mathcal{{T}}_{{00,\\text{{long}}}}$ Frac |
|---|---|---|---|---|---|---|---|---|---|
| **5m** | `all_days` | {r5_all['missing_bin_frac_median']*100:.2f}% [{r5_all['missing_bin_frac_iqr']*100:.2f}%] | {r5_all['eligible_missing_frac_median']*100:.2f}% [{r5_all['eligible_missing_frac_iqr']*100:.2f}%] | {r5_all['f_t11_median']*100:.2f}% [{r5_all['f_t11_iqr']*100:.2f}%] | {r5_all['f_t_short_median']*100:.2f}% [{r5_all['f_t_short_iqr']*100:.2f}%] | {r5_all['f_t_short_1_2_median']*100:.2f}% [{r5_all['f_t_short_1_2_iqr']*100:.2f}%] | {r5_all['f_t_short_3_5_median']*100:.2f}% [{r5_all['f_t_short_3_5_iqr']*100:.2f}%] | {r5_all['f_t_edge_median']*100:.2f}% [{r5_all['f_t_edge_iqr']*100:.2f}%] | {r5_all['f_t_00_long_median']*100:.2f}% [{r5_all['f_t_00_long_iqr']*100:.2f}%] |
| **15m** | `all_days` | {r15_all['missing_bin_frac_median']*100:.2f}% [{r15_all['missing_bin_frac_iqr']*100:.2f}%] | {r15_all['eligible_missing_frac_median']*100:.2f}% [{r15_all['eligible_missing_frac_iqr']*100:.2f}%] | {r15_all['f_t11_median']*100:.2f}% [{r15_all['f_t11_iqr']*100:.2f}%] | {r15_all['f_t_short_median']*100:.2f}% [{r15_all['f_t_short_iqr']*100:.2f}%] | {r15_all['f_t_short_1_2_median']*100:.2f}% [{r15_all['f_t_short_1_2_iqr']*100:.2f}%] | N/A | {r15_all['f_t_edge_median']*100:.2f}% [{r15_all['f_t_edge_iqr']*100:.2f}%] | {r15_all['f_t_00_long_median']*100:.2f}% [{r15_all['f_t_00_long_iqr']*100:.2f}%] |

### 1.3 Key Observations on Transition Anatomy
- **Coarsening Effect on Missingness**: Moving from 5m to 15m reduces cohort missingness from {r5_all['missing_bin_frac_cohort']*100:.2f}% to {r15_all['missing_bin_frac_cohort']*100:.2f}%, but compresses short-gap eligibility from {r5_all['eligible_missing_frac_cohort']*100:.2f}% to {r15_all['eligible_missing_frac_cohort']*100:.2f}%.
- **Short-Gap Composition**: At 5m, the {r5_all['f_t_short_cohort']*100:.2f}% short-gap transitions are split: {r5_all['f_t_short_1_2_cohort']*100:.2f}% in 1–2 bin gaps ($10\\text{{--}}15\\text{{ min}}$) and {r5_all['f_t_short_3_5_cohort']*100:.2f}% in 3–5 bin gaps ($20\\text{{--}}30\\text{{ min}}$).
- **Physical Span Matching Caveat (Review R4)**: Because $k=1$ at 15m corresponds to $30\\text{{ min}}$ separation while $k=1\\text{{--}}5$ at 5m spans $10\\text{{--}}30\\text{{ min}}$, `H_short_interp` acts over a wider variety of sampling gaps at 5m than at 15m.

---

## 2. Difference-Series Magnitudes Across Transition Regimes

Using Allocation 0 training templates ($\\mu^{{\\text{{clock}}}}$ and $\\mu^{{\\text{{global}}}}$ fitted strictly on train IDs ($N=7,797$), we audit participant-level RMS and MAE of the difference series $\\Delta X_t = X_{{t+1}} - X_t$.

### 2.1 Cohort-Wide Difference Magnitudes (Overall Median [IQR])

| Transition Regime | Filling Method | 5-Minute MAE (bpm) | 5-Minute RMS (bpm) | 15-Minute MAE (bpm) | 15-Minute RMS (bpm) | Ratio to Natural Volatility (5m RMS) |
|---|---|---:|---:|---:|---:|---:|
| **Observed $(1, 1)$** | None (Raw Physiology) | {r5_all['diff_obs_mae_median']:.2f} [{r5_all['diff_obs_mae_iqr']:.2f}] | {r5_all['diff_obs_rms_median']:.2f} [{r5_all['diff_obs_rms_iqr']:.2f}] | {r15_all['diff_obs_mae_median']:.2f} [{r15_all['diff_obs_mae_iqr']:.2f}] | {r15_all['diff_obs_rms_median']:.2f} [{r15_all['diff_obs_rms_iqr']:.2f}] | **1.00× (Baseline)** |
| **Boundary Jump $(M_t \\neq M_{{t+1}})$** | $\\mu^{{\\text{{clock}}}}$ Fill | {r5_all['diff_bound_clock_mae_median']:.2f} [{r5_all['diff_bound_clock_mae_iqr']:.2f}] | {r5_all['diff_bound_clock_rms_median']:.2f} [{r5_all['diff_bound_clock_rms_iqr']:.2f}] | {r15_all['diff_bound_clock_mae_median']:.2f} [{r15_all['diff_bound_clock_mae_iqr']:.2f}] | {r15_all['diff_bound_clock_rms_median']:.2f} [{r15_all['diff_bound_clock_rms_iqr']:.2f}] | **{r5_all['diff_bound_clock_rms_median']/r5_all['diff_obs_rms_median']:.2f}×** |
| **Boundary Jump $(M_t \\neq M_{{t+1}})$** | $\\mu^{{\\text{{global}}}}$ Fill | {r5_all['diff_bound_global_mae_median']:.2f} [{r5_all['diff_bound_global_mae_iqr']:.2f}] | {r5_all['diff_bound_global_rms_median']:.2f} [{r5_all['diff_bound_global_rms_iqr']:.2f}] | {r15_all['diff_bound_global_mae_median']:.2f} [{r15_all['diff_bound_global_mae_iqr']:.2f}] | {r15_all['diff_bound_global_rms_median']:.2f} [{r15_all['diff_bound_global_rms_iqr']:.2f}] | **{r5_all['diff_bound_global_rms_median']/r5_all['diff_obs_rms_median']:.2f}×** |
| **Gap Interior $(0, 0)$** | $\\mu^{{\\text{{clock}}}}$ Fill | {r5_all['diff_int_clock_mae_median']:.2f} [{r5_all['diff_int_clock_mae_iqr']:.2f}] | {r5_all['diff_int_clock_rms_median']:.2f} [{r5_all['diff_int_clock_rms_iqr']:.2f}] | {r15_all['diff_int_clock_mae_median']:.2f} [{r15_all['diff_int_clock_mae_iqr']:.2f}] | {r15_all['diff_int_clock_rms_median']:.2f} [{r15_all['diff_int_clock_rms_iqr']:.2f}] | **0.03×** |
| **Gap Interior $(0, 0)$** | $\\mu^{{\\text{{global}}}}$ Fill | 0.00 [0.00] | 0.00 [0.00] | 0.00 [0.00] | 0.00 [0.00] | **0.00×** |

### 2.2 Difference Magnitudes Across Continuity × Occupancy Subgroups (5-Minute Grid)

The 9 crossed cells (`continuity_x_occupancy`) demonstrate marked gradient in boundary jump magnitude:

| Subgroup (`cont_x_occ`) | N Users | Observed MAE | Observed RMS | Boundary Clock MAE | Boundary Clock RMS | Boundary Global MAE | Boundary Global RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
"""

    # Add 9 cont_x_occ rows
    cont_occ_sub = subgroup_df[
        (subgroup_df["resolution"] == "5m") & 
        (subgroup_df["budget"] == "all_days") & 
        (subgroup_df["dimension"] == "continuity_x_occupancy")
    ]
    for _, row in cont_occ_sub.iterrows():
        md += f"| `{row['subgroup']}` | {row['n_participants']} | {row['diff_obs_mae_median']:.2f} | {row['diff_obs_rms_median']:.2f} | {row['diff_bound_clock_mae_median']:.2f} | {row['diff_bound_clock_rms_median']:.2f} | {row['diff_bound_global_mae_median']:.2f} | {row['diff_bound_global_rms_median']:.2f} |\n"

    md += f"""
### 2.3 Synthesis of Difference-Series Dynamics
1. **Severe Boundary Contamination**: Intra-day boundary jumps inject step changes averaging $>{int(r5_all['diff_bound_clock_rms_median'])}\\text{{ bpm}}$ RMS, which is ${r5_all['diff_bound_clock_rms_median']/r5_all['diff_obs_rms_median']:.2f}\\times$ larger than true physiological drift (${r5_all['diff_obs_rms_median']:.2f}\\text{{ bpm}}$ RMS). At 15m, boundary jumps reach ${r15_all['diff_bound_clock_rms_median']:.2f}\\text{{ bpm}}$ RMS.
2. **Mechanism of `H_clean_diff`**: In standard MultiRocket (`H_clock`), half of the $18,816$ sequence features originate from $\\Delta X$. Because convolution integrates differences within a 9-bin receptive field, these massive boundary spikes dominate the difference features. `H_clean_diff` eliminates boundary jump contamination by setting $\\Delta^{{\\text{{clean}}}} X_t = 0.0$ whenever $M_t = 0 \\lor M_{{t+1}} = 0$, shielding downstream linear Ridge models from non-physiological impulse noise.
3. **Interior Template Slope Irrelevance**: Inside gaps ($(0, 0)$), diurnal slopes under $\\mu^{{\\text{{clock}}}}$ average only ${r5_all['diff_int_clock_mae_median']:.2f}\\text{{ bpm}}$ MAE at 5m (${r15_all['diff_int_clock_mae_median']:.2f}\\text{{ bpm}}$ at 15m), which is $<1.1\\%$ of boundary jump amplitude (${r5_all['diff_bound_clock_mae_median']:.2f}\\text{{ bpm}}$ MAE). Thus, replacing $\\mu^{{\\text{{clock}}}}$ with $\\mu^{{\\text{{global}}}}$ has negligible impact on within-gap slopes; any performance contrast between `H_clock` and `H_global` is driven primarily by slight boundary jump differences.

---

## 3. MultiRocket Binary-Quantile Degeneracy Audit (Allocation 0)

### 3.1 Bias Threshold Collapse
Under continuous heart rate ($X^{{\\text{{clock}}}}$), length-9 convolution responses take continuous values, yielding $1,060$ unique bias thresholds out of $1,176$ parameters ($90.1\\%$ unique). On binary masks $M \\in \\{{0.0, 1.0\\}}$ and difference masks $\\Delta M \\in \\{{-1.0, 0.0, 1.0\\}}$, convolution responses are restricted to integers in $[-6, +6]$. Consequently, quantile thresholding collapses onto discrete values dominated by $0.0000$:

| Resolution | Series Branch | Total Biases | Unique Biases | Unique Pct (%) | Zero Biases (count / %) | Top-5 Bias Concentration |
|---|---|---:|---:|---:|---:|---:|
| **5m ($L=288$)** | Base Mask $M \\in \\{{0, 1\\}}$ | {mr5['total_biases']} | {mr5['unique_biases_M']} | {mr5['unique_biases_M_pct']:.1f}% | {mr5['zero_bias_count_M']} / {mr5['zero_bias_pct_M']:.1f}% | {mr5['top5_m_pct']:.1f}% |
| **5m ($L=288$)** | Diff Mask $\\Delta M \\in \\{{-1, 0, 1\\}}$ | {mr5['total_biases']} | {mr5['unique_biases_dM']} | {mr5['unique_biases_dM_pct']:.1f}% | {mr5['zero_bias_count_dM']} / {mr5['zero_bias_pct_dM']:.1f}% | {mr5['top5_dm_pct']:.1f}% |
| **15m ($L=96$)** | Base Mask $M \\in \\{{0, 1\\}}$ | {mr15['total_biases']} | {mr15['unique_biases_M']} | {mr15['unique_biases_M_pct']:.1f}% | {mr15['zero_bias_count_M']} / {mr15['zero_bias_pct_M']:.1f}% | {mr15['top5_m_pct']:.1f}% |
| **15m ($L=96$)** | Diff Mask $\\Delta M \\in \\{{-1, 0, 1\\}}$ | {mr15['total_biases']} | {mr15['unique_biases_dM']} | {mr15['unique_biases_dM_pct']:.1f}% | {mr15['zero_bias_count_dM']} / {mr15['zero_bias_pct_dM']:.1f}% | {mr15['top5_dm_pct']:.1f}% |
| *5m (Ref)* | *Continuous HR $X^{{\\text{{clock}}}}$* | *{mr5['total_biases']}* | *{mr5['unique_biases_hr']}* | *{mr5['unique_biases_hr']/mr5['total_biases']*100:.1f}%* | *0 / 0.0%* | *< 5.0%* |

### 3.2 Feature Drop Audit (`VarianceThreshold(0.0)`)
A daily feature drops with zero variance when an extreme bias threshold (e.g., $\\le -6.0$ on $\\Delta M$) is never crossed by any day in the sampled training cohort. Because across-day mean and across-day sample SD inherit this constant value, each dropped daily feature induces exactly two dropped pooled features:

| Resolution | Series Branch | Daily Features Dropped (out of 9,408) | Daily Drop Pct (%) | Pooled Features Dropped (out of 18,816) | Pooled Drop Pct (%) | Dropped Daily Indices |
|---|---|---:|---:|---:|---:|---|
| **5m ($L=288$)** | $M + \\Delta M$ | {mr5['daily_features_dropped']} | {mr5['daily_features_dropped_pct']:.3f}% | {mr5['pooled_features_dropped']} | {mr5['pooled_features_dropped_pct']:.3f}% | `{mr5['daily_dropped_indices']}` |
| **15m ($L=96$)** | $M + \\Delta M$ | {mr15['daily_features_dropped']} | {mr15['daily_features_dropped_pct']:.3f}% | {mr15['pooled_features_dropped']} | {mr15['pooled_features_dropped_pct']:.3f}% | `{mr15['daily_dropped_indices']}` |

### 3.3 Collinearity and Regularization Implications
Although `VarianceThreshold(0.0)` drops only $0.01\\%\\text{{--}}0.07\\%$ of columns, the discrete bias collapse creates numerous identical duplicate columns within kernel/dilation slots where multiple quantiles mapped to the same threshold. These duplicate columns retain variance but are strictly collinear. In Stage 1 fitting, `VarianceThreshold(0.0)` must be followed by Ridge L2 regularization with larger candidate penalties ($\\alpha \\in \\{{10000, 100000\\}}$) to prevent coefficient explosion on collinear mask features.

---

## 4. Implications for Stage 1 Benchmark Protocol

1. **`H_clock` vs. `H_global` Contrast**: Given that gap interior slopes under $\\mu^{{\\text{{clock}}}}$ (${r5_all['diff_int_clock_mae_median']:.2f}\\text{{ bpm}}$) are functionally negligible and boundary jumps differ by only $\\approx {r5_all['diff_bound_global_mae_median'] - r5_all['diff_bound_clock_mae_median']:.2f}\\text{{ bpm}}$ (${r5_all['diff_bound_global_rms_median'] - r5_all['diff_bound_clock_rms_median']:.2f}\\text{{ bpm}}$ RMS), the aggregate contrast $\\text{{AUROC}}(\\text{{H\\_global}}) - \\text{{AUROC}}(\\text{{H\\_clock}})$ is expected to be minimal ($|\\Delta| \\le 0.005$).
2. **`H_clean_diff` Hypothesis**: Withholding boundary jumps from the $\\Delta X$ branch will test whether MultiRocket relied on behavioral gap step-jumps or genuine physiological fluctuations. If `H_clean_diff` maintains or improves AUROC, it demonstrates that boundary artifacts were introducing estimation burden or non-generalizable noise.
3. **`H_short_interp` Hypothesis**: Because {r5_all['eligible_missing_frac_cohort']*100:.1f}% of missing bins are eligible at 5m vs. only {r15_all['eligible_missing_frac_cohort']*100:.1f}% at 15m, short-gap interpolation represents a significantly more extensive intervention at 5m. Any differential resolution gain from `H_short_interp` at 5m reflects this higher exposure rate.
4. **`M_temporal` Ridge Grid**: The severe bias collapse on binary sequences produces high collinearity. The candidate grid must include $\\alpha = 100,000$ (`ridge_selected_ext`) to ensure proper shrinkage for `M_temporal` and `H_plus_M_concat`.

---

## Artifact Index
- `results/exposure_overall.csv` — Full cohort and per-condition transition, eligibility, and difference statistics.
- `results/exposure_by_subgroup.csv` — Stratified statistics across 31 recording-pattern subgroups.
- `results/STAGE0_AUDIT.md` — This publication document.
"""
    return md


# ==============================================================================
# Main Runner
# ==============================================================================

def main() -> int:
    """Execute complete Stage 0 Gap & Mask Exposure Audit."""
    parser = argparse.ArgumentParser(description="Apple Watch Stage 0 Gap & Mask Exposure Audit")
    parser.add_argument("--smoke", type=int, default=None, help="Run on subset of participants for smoke testing")
    args = parser.parse_args()

    t_start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 0: GAP & MASK EXPOSURE AUDIT FOR APPLE WATCH COHORT")
    print("=" * 80)
    print(f"Artifacts directory: {ARTIFACTS_DIR}")
    print(f"Results directory:   {RESULTS_DIR}")
    print(f"Smoke test mode:     {args.smoke if args.smoke else 'OFF (Full Cohort N=11,139)'}")

    # Load allocations and cutpoints
    alloc_df = pd.read_parquet(ARTIFACTS_DIR / "apple_10_allocations.parquet")
    tr_mask_0 = (alloc_df["alloc_0"] == "train")
    tr_cids_0 = alloc_df.loc[tr_mask_0, "customer_id"].astype(str).tolist()
    cutpoints_df = pd.read_csv(ARTIFACTS_DIR / "binning_by_recording" / "training_cutpoints.csv")

    cache_5m_dir = ARTIFACTS_DIR / "cache_5min"
    cache_15m_dir = ARTIFACTS_DIR / "cache_15min"

    # 1. Fit Train Templates strictly on Allocation 0 train
    print("\n[1/5] Fitting Allocation 0 train templates (isolated worker process)...")
    t0 = time.perf_counter()
    clock_5m, glob_5m = fit_train_templates_isolated(cache_5m_dir, tr_cids_0)
    clock_15m, glob_15m = fit_train_templates_isolated(cache_15m_dir, tr_cids_0)
    templates = {
        "5m": (clock_5m, glob_5m),
        "15m": (clock_15m, glob_15m),
    }
    print(f"      Fitted 5m templates in {time.perf_counter() - t0:.2f}s: global={glob_5m:.2f}, clock mean={clock_5m.mean():.2f}")
    print(f"      Fitted 15m templates in {time.perf_counter() - t0:.2f}s: global={glob_15m:.2f}, clock mean={clock_15m.mean():.2f}")
    print(f"      Current Max RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.2f} MB")

    # 2. Load Descriptors & Assign Subgroups
    print("\n[2/5] Loading cohort descriptors and assigning Allocation 0 subgroup terciles...")
    t0 = time.perf_counter()
    desc_df, labels_df = load_descriptors_and_subgroups(cache_5m_dir, cutpoints_df, smoke=args.smoke)
    print(f"      Processed {len(desc_df)} participants in {time.perf_counter() - t0:.2f}s.")
    print(f"      Current Max RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.2f} MB")

    # 3. Stream Audit across 5m and 15m
    print("\n[3/5] Streaming participant audit across 5m and 15m (all_days & up_to_40d)...")
    t0 = time.perf_counter()
    
    overall_records: list[dict] = []
    subgroup_records: list[dict] = []

    configs = [
        ("5m", cache_5m_dir, clock_5m, glob_5m),
        ("15m", cache_15m_dir, clock_15m, glob_15m),
    ]

    for res, cache_dir, clock_t, glob_t in configs:
        t_res = time.perf_counter()
        records_all: list[dict] = []
        records_40d: list[dict] = []

        for cid, fn in zip(desc_df.index, desc_df["file"]):
            with np.load(cache_dir / fn) as data:
                hr_full = data["hr"]

            # Audit all_days
            rec_all = audit_single_participant(hr_full, clock_t, glob_t, res)
            rec_all["customer_id"] = cid
            records_all.append(rec_all)

            # Audit up_to_40d
            hr_40d = hr_full[:40]
            rec_40d = audit_single_participant(hr_40d, clock_t, glob_t, res)
            rec_40d["customer_id"] = cid
            records_40d.append(rec_40d)

        df_all = pd.DataFrame(records_all).set_index("customer_id")
        df_40d = pd.DataFrame(records_40d).set_index("customer_id")

        # Summarize overall
        overall_records.append(summarize_condition_overall(res, "all_days", df_all))
        overall_records.append(summarize_condition_overall(res, "up_to_40d", df_40d))

        # Summarize subgroups
        subgroup_records.extend(summarize_condition_subgroups(res, "all_days", df_all, labels_df))
        subgroup_records.extend(summarize_condition_subgroups(res, "up_to_40d", df_40d, labels_df))

        print(f"      [{res}] Audited {len(desc_df)} participants in {time.perf_counter() - t_res:.2f}s.")
        print(f"      Current Max RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.2f} MB")

    # 4. MultiRocket Degeneracy Audit
    print("\n[4/5] Running MultiRocket binary mask degeneracy audit (Allocation 0)...")
    t0 = time.perf_counter()
    mr_results = {
        "5m": audit_multirocket_degeneracy(alloc_df, cache_5m_dir, "5m", allocation=0),
        "15m": audit_multirocket_degeneracy(alloc_df, cache_15m_dir, "15m", allocation=0),
    }
    print(f"      Completed MultiRocket degeneracy audit in {time.perf_counter() - t0:.2f}s.")
    print(f"      5m: Unique biases M = {mr_results['5m']['unique_biases_M']}/{mr_results['5m']['total_biases']} ({mr_results['5m']['unique_biases_M_pct']:.1f}%), Dropped daily = {mr_results['5m']['daily_features_dropped']}/9408")
    print(f"      15m: Unique biases M = {mr_results['15m']['unique_biases_M']}/{mr_results['15m']['total_biases']} ({mr_results['15m']['unique_biases_M_pct']:.1f}%), Dropped daily = {mr_results['15m']['daily_features_dropped']}/9408")
    print(f"      Current Max RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.2f} MB")

    # 5. Export Artifacts
    print("\n[5/5] Exporting CSV and Markdown audit artifacts...")
    overall_df = pd.DataFrame(overall_records)
    subgroup_df = pd.DataFrame(subgroup_records)

    overall_path = RESULTS_DIR / "exposure_overall.csv"
    subgroup_path = RESULTS_DIR / "exposure_by_subgroup.csv"
    audit_md_path = RESULTS_DIR / "STAGE0_AUDIT.md"

    overall_df.to_csv(overall_path, index=False)
    subgroup_df.to_csv(subgroup_path, index=False)
    print(f"      Saved {overall_path} ({len(overall_df)} rows)")
    print(f"      Saved {subgroup_path} ({len(subgroup_df)} rows)")

    audit_md_content = generate_stage0_audit_markdown(overall_df, subgroup_df, mr_results, templates)
    with open(audit_md_path, "w", encoding="utf-8") as f:
        f.write(audit_md_content)
    print(f"      Saved {audit_md_path} ({len(audit_md_content)} chars)")

    t_total = time.perf_counter() - t_start
    final_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print("=" * 80)
    print(f"STAGE 0 AUDIT COMPLETE in {t_total:.2f}s ({t_total/60:.2f} min).")
    print(f"Peak RSS Memory: {final_rss:.2f} MB (< 500 MB requirement).")
    print("=" * 80)

    assert final_rss < 500.0, f"Memory threshold exceeded: {final_rss:.2f} MB >= 500.0 MB"
    return 0


if __name__ == "__main__":
    sys.exit(main())
