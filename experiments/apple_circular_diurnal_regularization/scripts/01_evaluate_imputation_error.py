"""Milestone 2: Masked Imputation Error Benchmark for Circular Diurnal Regularization.

Evaluates reconstruction fidelity (MAE, RMSE) and boundary step jump (Delta_boundary)
across gap lengths (short 1-3 bins, medium 6-12 bins, long 24-48 bins) and
participant occupancy tiers (low <50%, medium 50-80%, high >80%) comparing:
  1. Raw user clock median: H_clock
  2. Circular Gaussian smoothed template sigma=1: H_circ_gauss_s1 (5 min)
  3. Circular Gaussian smoothed template sigma=2: H_circ_gauss_s2 (10 min)
  4. Circular Gaussian smoothed template sigma=4: H_circ_gauss_s4 (20 min)
  5. Short linear interpolation baseline: H_linear

Cohort: 200 participants sampled from Allocation 0 test cohort (N=1,671)
stratified across occupancy tiers using deterministic seed 20261020.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
REPO_ROOT = PROJECT_DIR.parent.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from circular_smoothing import circular_gaussian_smooth

ALLOC_PATH = REPO_ROOT / "experiments" / "apple_cohort_scaling" / "artifacts" / "apple_10_allocations.parquet"
CACHE_5M_DIR = REPO_ROOT / "experiments" / "apple_cohort_scaling" / "artifacts" / "cache_5min"

SEED = 20261020
N_PARTICIPANTS = 200
GAPS_PER_TIER = 20


def compute_cohort_occupancy(alloc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall occupancy rate for Allocation 0 test participants."""
    test_df = alloc_df[alloc_df["alloc_0"] == "test"].copy()
    test_df = test_df.sort_values("customer_id").reset_index(drop=True)
    cids = test_df["customer_id"].astype(str).tolist()

    print(f"Computing occupancy for {len(cids)} Allocation 0 test participants...")
    t0 = time.time()
    occupancies = []
    total_valid_bins = []
    total_bins_list = []

    for cid in cids:
        npz_path = CACHE_5M_DIR / f"customer_{cid}.npz"
        with np.load(npz_path) as npz:
            hr = npz["hr"]
            valid = np.count_nonzero(np.isfinite(hr))
            occupancies.append(valid / hr.size)
            total_valid_bins.append(valid)
            total_bins_list.append(hr.size)

    test_df["occupancy"] = occupancies
    test_df["valid_bins"] = total_valid_bins
    test_df["total_bins"] = total_bins_list

    # Stratify into three occupancy tiers:
    # Low (<50%), Medium (50-80%), High (>80%)
    conditions = [
        test_df["occupancy"] < 0.50,
        (test_df["occupancy"] >= 0.50) & (test_df["occupancy"] <= 0.80),
        test_df["occupancy"] > 0.80,
    ]
    choices = ["low", "medium", "high"]
    test_df["occupancy_tier"] = np.select(conditions, choices, default="unknown")

    t1 = time.time()
    print(f"Occupancy computed in {t1 - t0:.2f}s.")
    print("Test cohort tier breakdown:")
    print(test_df["occupancy_tier"].value_counts())
    return test_df


def sample_stratified_participants(test_df: pd.DataFrame, n_sample: int = 200, seed: int = SEED) -> pd.DataFrame:
    """Sample n_sample participants stratified across occupancy tiers."""
    sampled_df, _ = train_test_split(
        test_df,
        train_size=n_sample,
        stratify=test_df["occupancy_tier"],
        random_state=seed,
    )
    sampled_df = sampled_df.sort_values("customer_id").reset_index(drop=True)
    print(f"\nSampled {len(sampled_df)} participants stratified by occupancy tier:")
    print(sampled_df["occupancy_tier"].value_counts())
    return sampled_df


def find_eligible_gaps_for_participant(
    hr: np.ndarray,
    rng: np.random.Generator,
    gaps_per_tier: int = GAPS_PER_TIER,
) -> list[tuple[str, int, int, int, int]]:
    """Identify contiguous observed segments and sample artificial gaps.

    Returns:
        List of tuples: (gap_tier, gap_length_k, day_idx, start_bin, end_bin)
    """
    finite_mask = np.isfinite(hr)
    D, L = hr.shape

    # Find all contiguous observed runs in each day
    segments = []
    for d in range(D):
        m = finite_mask[d]
        if not m.any():
            continue
        # Pad with 0 at both boundaries to detect edges
        p = np.pad(m.astype(np.int8), (1, 1), mode="constant")
        diff = np.diff(p)
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        for s, e in zip(starts, ends):
            length = e - s
            # Minimum length needed for gap of length 1 with boundary bins is 1 + 2 = 3
            if length >= 3:
                # s is start bin, e-1 is end bin in hr[d]
                segments.append((d, s, e - 1, length))

    if not segments:
        return []

    segments_arr = np.array(segments)  # (N, 4): d, s, e, length

    gap_specs = [
        ("short", 1, 3),    # 1-3 bins (5-15 min)
        ("medium", 6, 12),  # 6-12 bins (30-60 min)
        ("long", 24, 48),   # 24-48 bins (2-4 hours)
    ]

    sampled_gaps = []

    for tier_name, k_min, k_max in gap_specs:
        # Segment must accommodate at least k_min bins plus 2 boundary bins (left and right)
        valid_segs = segments_arr[segments_arr[:, 3] >= k_min + 2]
        if len(valid_segs) == 0:
            continue

        # Sample segments (without replacement if enough, otherwise with replacement)
        if len(valid_segs) >= gaps_per_tier:
            chosen_indices = rng.choice(len(valid_segs), size=gaps_per_tier, replace=False)
        else:
            chosen_indices = rng.choice(len(valid_segs), size=gaps_per_tier, replace=True)

        for s_idx in chosen_indices:
            d, s, e, length = valid_segs[s_idx]
            # Maximum gap length that fits in this segment leaving boundary bins
            max_possible_k = min(k_max, length - 2)
            if max_possible_k < k_min:
                continue
            k = int(rng.integers(k_min, max_possible_k + 1))
            # Start bin must be strictly between s+1 and e-k
            t_start = int(rng.integers(s + 1, e - k + 1))
            t_end = t_start + k - 1
            sampled_gaps.append((tier_name, k, d, t_start, t_end))

    return sampled_gaps


def run_imputation_benchmark() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete Milestone 2 evaluation protocol."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print("STARTING MILESTONE 2: MASKED IMPUTATION ERROR BENCHMARK")
    print(f"Allocations file: {ALLOC_PATH}")
    print(f"Cache directory: {CACHE_5M_DIR}")
    print("======================================================================")

    # 1. Load Allocations and compute occupancy
    alloc_df = pd.read_parquet(ALLOC_PATH)
    test_df = compute_cohort_occupancy(alloc_df)

    # 2. Stratified sampling of 200 participants
    sampled_cohort = sample_stratified_participants(test_df, n_sample=N_PARTICIPANTS, seed=SEED)
    cohort_save_path = RESULTS_DIR / "imputation_sampled_cohort.csv"
    sampled_cohort[["customer_id", "salutation", "y", "age_group", "bmi_group", "n_days", "occupancy", "occupancy_tier"]].to_csv(
        cohort_save_path, index=False
    )
    print(f"Saved sampled cohort metadata to: {cohort_save_path}")

    # 3. Benchmark evaluation
    print("\nEvaluating masked imputation across 200 participants...")
    t0_bench = time.time()
    gap_records = []

    for p_idx, (_, row) in enumerate(sampled_cohort.iterrows()):
        cid = str(row["customer_id"])
        occ_tier = row["occupancy_tier"]
        user_occ = float(row["occupancy"])

        npz_path = CACHE_5M_DIR / f"customer_{cid}.npz"
        with np.load(npz_path) as npz:
            hr = npz["hr"]  # shape (D, 288)

        # 3a. Compute user clock diurnal template mu_u^clock (median across days)
        with np.errstate(invalid="ignore"):
            t_clock = np.nanmedian(hr, axis=0).astype(np.float64)

        # Fallback for any all-NaN clock bins
        if np.isnan(t_clock).any():
            valid_hr = hr[np.isfinite(hr)]
            med_glob = float(np.nanmedian(valid_hr)) if valid_hr.size > 0 else 60.0
            t_clock = np.where(np.isnan(t_clock), med_glob, t_clock)

        # 3b. Compute circular Gaussian smoothed templates
        t_s1 = circular_gaussian_smooth(t_clock, sigma=1.0, nan_policy="normalize")
        t_s2 = circular_gaussian_smooth(t_clock, sigma=2.0, nan_policy="normalize")
        t_s4 = circular_gaussian_smooth(t_clock, sigma=4.0, nan_policy="normalize")

        templates = {
            "H_clock": t_clock,
            "H_circ_gauss_s1": t_s1,
            "H_circ_gauss_s2": t_s2,
            "H_circ_gauss_s4": t_s4,
        }

        # 3c. Deterministic RNG for this participant
        rng = np.random.default_rng(SEED + int(cid))
        gaps = find_eligible_gaps_for_participant(hr, rng, gaps_per_tier=GAPS_PER_TIER)

        for gap_tier, k, d, t_start, t_end in gaps:
            x_true = hr[d, t_start : t_end + 1].astype(np.float64)
            x_left = float(hr[d, t_start - 1])
            x_right = float(hr[d, t_end + 1])

            # Evaluate 4 template methods
            for m_name, tmpl in templates.items():
                x_fill = tmpl[t_start : t_end + 1]
                diff = x_fill - x_true
                mae = float(np.mean(np.abs(diff)))
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                boundary_jump = float(np.abs(x_fill[0] - x_left))

                gap_records.append({
                    "customer_id": cid,
                    "occupancy_tier": occ_tier,
                    "occupancy": user_occ,
                    "gap_tier": gap_tier,
                    "gap_length_bins": k,
                    "method": m_name,
                    "mae_bpm": mae,
                    "rmse_bpm": rmse,
                    "boundary_jump_bpm": boundary_jump,
                })

            # Evaluate linear interpolation baseline
            # x_linear[j] = x_left + (j + 1) / (k + 1) * (x_right - x_left) for j in 0..k-1
            steps = (np.arange(1, k + 1, dtype=np.float64)) / (k + 1.0)
            x_linear = x_left + steps * (x_right - x_left)
            diff_lin = x_linear - x_true
            mae_lin = float(np.mean(np.abs(diff_lin)))
            rmse_lin = float(np.sqrt(np.mean(diff_lin ** 2)))
            boundary_jump_lin = float(np.abs(x_linear[0] - x_left))

            gap_records.append({
                "customer_id": cid,
                "occupancy_tier": occ_tier,
                "occupancy": user_occ,
                "gap_tier": gap_tier,
                "gap_length_bins": k,
                "method": "H_linear",
                "mae_bpm": mae_lin,
                "rmse_bpm": rmse_lin,
                "boundary_jump_bpm": boundary_jump_lin,
            })

    t_bench = time.time() - t0_bench
    per_gap_df = pd.DataFrame(gap_records)
    print(f"Benchmark completed in {t_bench:.2f}s.")
    print(f"Total evaluated gap instances: {len(per_gap_df) // 5} gaps x 5 methods = {len(per_gap_df)} records.")

    # Save per-gap granular dataset
    per_gap_path = RESULTS_DIR / "imputation_benchmark_per_gap.parquet"
    per_gap_df.to_parquet(per_gap_path, index=False)
    print(f"Saved granular per-gap data to: {per_gap_path}")

    # 4. Aggregate metrics
    # Slices: overall, by gap_tier, by occupancy_tier, by gap_tier x occupancy_tier
    summary_rows = []

    def compute_summary_row(sub_df: pd.DataFrame, slice_name: str, gap_t: str, occ_t: str) -> list[dict]:
        rows = []
        for method, m_df in sub_df.groupby("method"):
            n_eval = len(m_df)
            mae_m = float(m_df["mae_bpm"].mean())
            mae_s = float(m_df["mae_bpm"].std(ddof=1))
            mae_sem = mae_s / np.sqrt(n_eval) if n_eval > 1 else 0.0

            rmse_m = float(m_df["rmse_bpm"].mean())
            rmse_s = float(m_df["rmse_bpm"].std(ddof=1))
            rmse_sem = rmse_s / np.sqrt(n_eval) if n_eval > 1 else 0.0

            bj_m = float(m_df["boundary_jump_bpm"].mean())
            bj_s = float(m_df["boundary_jump_bpm"].std(ddof=1))
            bj_sem = bj_s / np.sqrt(n_eval) if n_eval > 1 else 0.0

            rows.append({
                "slice": slice_name,
                "gap_tier": gap_t,
                "occupancy_tier": occ_t,
                "method": method,
                "n_gaps": n_eval,
                "mae_mean": mae_m,
                "mae_std": mae_s,
                "mae_sem": mae_sem,
                "rmse_mean": rmse_m,
                "rmse_std": rmse_s,
                "rmse_sem": rmse_sem,
                "boundary_jump_mean": bj_m,
                "boundary_jump_std": bj_s,
                "boundary_jump_sem": bj_sem,
            })
        return rows

    # Slice 1: Overall
    summary_rows.extend(compute_summary_row(per_gap_df, "overall", "all", "all"))

    # Slice 2: By gap tier
    for gt in ["short", "medium", "long"]:
        gt_df = per_gap_df[per_gap_df["gap_tier"] == gt]
        summary_rows.extend(compute_summary_row(gt_df, "by_gap_tier", gt, "all"))

    # Slice 3: By occupancy tier
    for ot in ["low", "medium", "high"]:
        ot_df = per_gap_df[per_gap_df["occupancy_tier"] == ot]
        summary_rows.extend(compute_summary_row(ot_df, "by_occupancy_tier", "all", ot))

    # Slice 4: By gap tier x occupancy tier
    for gt in ["short", "medium", "long"]:
        for ot in ["low", "medium", "high"]:
            cell_df = per_gap_df[(per_gap_df["gap_tier"] == gt) & (per_gap_df["occupancy_tier"] == ot)]
            if len(cell_df) > 0:
                summary_rows.extend(compute_summary_row(cell_df, "by_gap_and_occupancy", gt, ot))

    summary_df = pd.DataFrame(summary_rows)

    metrics_csv_path = RESULTS_DIR / "imputation_benchmark_metrics.csv"
    summary_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved aggregated metrics table to: {metrics_csv_path}")

    # Print high-level summaries
    print("\n======================================================================")
    print("OVERALL SUMMARY TABLE:")
    print("======================================================================")
    overall_disp = summary_df[summary_df["slice"] == "overall"][
        ["method", "n_gaps", "mae_mean", "rmse_mean", "boundary_jump_mean"]
    ].sort_values("mae_mean")
    print(overall_disp.to_string(index=False))

    print("\n======================================================================")
    print("SUMMARY BY GAP TIER:")
    print("======================================================================")
    by_gap_disp = summary_df[summary_df["slice"] == "by_gap_tier"][
        ["gap_tier", "method", "mae_mean", "rmse_mean", "boundary_jump_mean"]
    ]
    print(by_gap_disp.to_string(index=False))

    print("\n======================================================================")
    print("SUMMARY BY OCCUPANCY TIER:")
    print("======================================================================")
    by_occ_disp = summary_df[summary_df["slice"] == "by_occupancy_tier"][
        ["occupancy_tier", "method", "mae_mean", "rmse_mean", "boundary_jump_mean"]
    ]
    print(by_occ_disp.to_string(index=False))

    # Determine optimal sigma among circular Gaussian smoothed templates
    circ_methods = ["H_circ_gauss_s1", "H_circ_gauss_s2", "H_circ_gauss_s4"]
    overall_circ = summary_df[(summary_df["slice"] == "overall") & (summary_df["method"].isin(circ_methods))]
    best_mae_row = overall_circ.loc[overall_circ["mae_mean"].idxmin()]
    best_bj_row = overall_circ.loc[overall_circ["boundary_jump_mean"].idxmin()]

    print("\n======================================================================")
    print("REGULARIZATION EVALUATION:")
    print(f"Optimal template by Reconstruction MAE: {best_mae_row['method']} (MAE = {best_mae_row['mae_mean']:.4f} bpm)")
    print(f"Optimal template by Boundary Step Jump: {best_bj_row['method']} (Jump = {best_bj_row['boundary_jump_mean']:.4f} bpm)")
    print("======================================================================")

    return summary_df, per_gap_df, sampled_cohort


if __name__ == "__main__":
    run_imputation_benchmark()
