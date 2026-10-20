#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
01_extract_and_calibrate_infradian.py

Full Extraction & Calibration Pipeline for female_infradian_cycles_v2:
1. 180-Day Window Normalization:
   - For all N=11,139 Apple Watch participants, identify first qualifying window
     [start, start+180) partitioned into six 30-day blocks with >=15 qualified dates
     (>=120 valid 5m bins).
   - Uniformly select exactly 15 dates per block using deterministic seed 20261020
     and SHA-256("20261020|apple|{pid}|dates_block_{b}") -> uint64 little-endian.
   - Yields exactly 90 observations across 180 days for all N=4,631 eligible participants.
2. Dependence-Aware Null Calibration (B=199):
   - Daily hr_p10 (10th percentile of observed bins on each selected date).
   - Nonperiodic baseline: intercept, elapsed days linear trend, weekday indicators (Tue-Sun, Mon ref),
     four 6-hour clock coverage fractions (00-06, 06-12, 12-18, 18-24).
   - Stationary calendar-time AR(1) daily residual parameters: correlation phi^{|t-s|}, 0 <= phi < 1, and sigma^2.
   - B=199 parametric bootstrap replicates: epsilon_boot ~ AR(1) on exact calendar timestamps.
   - Searched infradian score S_i = max_{T in [25, 35]} (RSS_0 - RSS_T)/RSS_0.
   - Batched matrix multiplication for all 199 replicates.
   - Excess score E_i = S_i - mean(S_null), empirical p_i = (1 + sum(S_null >= S_i))/(B+1),
     amplitude A_i = sqrt(beta_c^2 + beta_s^2) at best-fit T.
3. Second-Window Temporal Persistence:
   - For N=2,324 two-window eligible participants (second window >=180 days after first window start),
     evaluate persistence of frozen first-window period and phase to predict second-window HR.
4. Output:
   - experiments/female_infradian_cycles_v2/results/window_eligibility.parquet
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Modular Import from 00_verify_v2_methods.py
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

VERIFY_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "00_verify_v2_methods.py")
if not os.path.exists(VERIFY_SCRIPT_PATH):
    raise FileNotFoundError(f"Verification script not found at {VERIFY_SCRIPT_PATH}")

spec = importlib.util.spec_from_file_location("verify_v2_methods", VERIFY_SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load module spec from {VERIFY_SCRIPT_PATH}")
verify_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_mod)

estimate_calendar_ar1 = verify_mod.estimate_calendar_ar1
simulate_calendar_ar1 = verify_mod.simulate_calendar_ar1
build_nonperiodic_baseline = verify_mod.build_nonperiodic_baseline
precompute_harmonic_operators = verify_mod.precompute_harmonic_operators
compute_infradian_scores_batch = verify_mod.compute_infradian_scores_batch


# -----------------------------------------------------------------------------
# Window Search & Sampling Logic
# -----------------------------------------------------------------------------

def find_qualifying_windows(qual_dates: np.ndarray, k_per_block: int = 15) -> Tuple[Optional[int], Optional[int]]:
    """
    Finds the earliest qualifying calendar window [start, start + 180)
    partitioned into six 30-day blocks with >= k qualified dates,
    and the earliest nonoverlapping second window (start_2 >= start_1 + 180).
    """
    n = len(qual_dates)
    if n < 6 * k_per_block:
        return None, None
        
    first_w: Optional[int] = None
    second_w: Optional[int] = None
    
    for start in qual_dates:
        if qual_dates[-1] < start + 150:
            break
            
        qualifies = True
        for j in range(6):
            b_start = start + 30 * j
            b_end = start + 30 * (j + 1)
            cnt = np.searchsorted(qual_dates, b_end) - np.searchsorted(qual_dates, b_start)
            if cnt < k_per_block:
                qualifies = False
                break
                
        if qualifies:
            if first_w is None:
                first_w = int(start)
            elif start >= first_w + 180:
                second_w = int(start)
                break
                
    return first_w, second_w


def select_window_dates(
    qual_dates: np.ndarray,
    start_date: int,
    cid: int,
    k_per_block: int = 15,
) -> np.ndarray:
    """
    Uniformly selects exactly k_per_block dates per 30-day block
    using the deterministic SHA-256 seeding formula:
    seed = uint64 from first 8 bytes of SHA-256("20261020|apple|{pid}|dates_block_{b}").
    """
    selected_blocks = []
    for b in range(6):
        b_start = start_date + 30 * b
        b_end = start_date + 30 * (b + 1)
        in_block = qual_dates[(qual_dates >= b_start) & (qual_dates < b_end)]
        in_block = np.sort(in_block)
        
        msg = f"20261020|apple|{cid}|dates_block_{b}".encode("utf-8")
        digest = hashlib.sha256(msg).digest()
        seed_u64 = int.from_bytes(digest[:8], byteorder="little")
        rng = np.random.default_rng(seed_u64)
        
        chosen = rng.choice(in_block, size=k_per_block, replace=False)
        selected_blocks.append(np.sort(chosen))
        
    return np.concatenate(selected_blocks)


def extract_features_on_dates(
    hr_cache: np.ndarray,
    all_dates: np.ndarray,
    selected_dates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Extracts hr_p10 and four clock coverage fractions on the 90 selected dates.
    """
    # Map dates to indices in hr_cache
    date_to_idx = {d: idx for idx, d in enumerate(all_dates)}
    indices = [date_to_idx[d] for d in selected_dates]
    
    hr_sel = hr_cache[indices] # Shape (90, 288)
    valid_mask = np.isfinite(hr_sel)
    
    n_days = len(selected_dates)
    hr_p10 = np.empty(n_days, dtype=np.float64)
    clock_fracs = np.empty((n_days, 4), dtype=np.float64)
    
    for i in range(n_days):
        v = valid_mask[i]
        vals = hr_sel[i, v]
        hr_p10[i] = np.percentile(vals, 10)
        
        # 4 clock intervals: 0-71, 72-143, 144-215, 216-287
        clock_fracs[i, 0] = np.sum(v[0:72]) / 72.0
        clock_fracs[i, 1] = np.sum(v[72:144]) / 72.0
        clock_fracs[i, 2] = np.sum(v[144:216]) / 72.0
        clock_fracs[i, 3] = np.sum(v[216:288]) / 72.0
        
    mean_occ = float(np.mean(np.sum(valid_mask, axis=1) / 288.0))
    mean_night_cov = float(np.mean(clock_fracs[:, 0]))
    
    return hr_p10, clock_fracs, mean_occ, mean_night_cov


# -----------------------------------------------------------------------------
# Participant Processing Worker
# -----------------------------------------------------------------------------

def process_participant(
    cid: int,
    meta_row: Dict[str, Any],
    cache_path: str,
    periods: np.ndarray,
    b_replicates: int = 199,
) -> Dict[str, Any]:
    """
    Processes a single participant through the complete v2 pipeline.
    """
    record: Dict[str, Any] = {
        "customer_id": cid,
        "user_id": meta_row.get("user_id", cid),
        "y": int(meta_row["y"]),
        "salutation": int(meta_row["salutation"]),
        "age_group": str(meta_row["age_group"]),
        "bmi_group": str(meta_row["bmi_group"]),
        "n_total_days": 0,
        "n_adequate_days": 0,
        "eligible_w1": False,
        "eligible_w2": False,
        "w1_start_date": None,
        "w1_actual_span": None,
        "w1_longest_gap": None,
        "w1_selected_dates": None,
        "w1_mean_occ_5m": np.nan,
        "w1_mean_clock_night": np.nan,
        "S_i": np.nan,
        "best_T": np.nan,
        "A_i": np.nan,
        "beta_c": np.nan,
        "beta_s": np.nan,
        "phi_hat": np.nan,
        "sigma2_hat": np.nan,
        "mean_S_null": np.nan,
        "E_i": np.nan,
        "p_i": np.nan,
        "w2_start_date": None,
        "w2_actual_span": None,
        "w2_longest_gap": None,
        "w2_selected_dates": None,
        "w2_mse_base": np.nan,
        "w2_mse_full": np.nan,
        "w2_delta_mse": np.nan,
        "w2_r2_persist": np.nan,
        "w2_corr_persist": np.nan,
        "w2_S_i": np.nan,
        "w2_best_T": np.nan,
        "delta_T_w1_w2": np.nan,
    }
    
    if not os.path.exists(cache_path):
        return record
        
    try:
        with np.load(cache_path) as npz:
            hr = npz["hr"]
            dates = npz["date"]
            
        record["n_total_days"] = int(len(dates))
        valid_bins = np.isfinite(hr).sum(axis=1)
        qual_mask = (valid_bins >= 120)
        qual_dates = dates[qual_mask]
        record["n_adequate_days"] = int(len(qual_dates))
        
        w1_start, w2_start = find_qualifying_windows(qual_dates, k_per_block=15)
        if w1_start is None:
            return record
            
        record["eligible_w1"] = True
        record["w1_start_date"] = int(w1_start)
        if w2_start is not None:
            record["eligible_w2"] = True
            record["w2_start_date"] = int(w2_start)
        
        # 1. Uniformly sample 15 dates per block for Window 1
        w1_dates = select_window_dates(qual_dates, w1_start, cid, k_per_block=15)
        record["w1_actual_span"] = int(w1_dates[-1] - w1_dates[0])
        record["w1_longest_gap"] = int(np.max(np.diff(w1_dates)))
        record["w1_selected_dates"] = w1_dates.tolist()
        
        # 2. Extract Window 1 features
        w1_hr_p10, w1_clock, w1_occ, w1_night_cov = extract_features_on_dates(
            hr, dates, w1_dates
        )
        record["w1_mean_occ_5m"] = w1_occ
        record["w1_mean_clock_night"] = w1_night_cov
        
        # Check for zero baseline RSS (e.g. constant signal)
        if np.std(w1_hr_p10) < 1e-6:
            # Handle zero variance as undefined according to PLAN.md
            if w2_start is not None:
                w2_dates = select_window_dates(qual_dates, w2_start, cid, k_per_block=15)
                record["w2_actual_span"] = int(w2_dates[-1] - w2_dates[0])
                record["w2_longest_gap"] = int(np.max(np.diff(w2_dates)))
                record["w2_selected_dates"] = w2_dates.tolist()
            return record

        # 3. Fit Window 1 nonperiodic baseline
        X_base, base_names, proj_resid = build_nonperiodic_baseline(w1_dates, w1_clock)
        e_0 = proj_resid @ w1_hr_p10
        rss_0 = float(np.sum(e_0 ** 2))
        
        if rss_0 < 1e-10:
            if w2_start is not None:
                w2_dates = select_window_dates(qual_dates, w2_start, cid, k_per_block=15)
                record["w2_actual_span"] = int(w2_dates[-1] - w2_dates[0])
                record["w2_longest_gap"] = int(np.max(np.diff(w2_dates)))
                record["w2_selected_dates"] = w2_dates.tolist()
            return record
            
        # 4. Stationary calendar-time AR(1) estimation
        phi_hat, sigma2_hat = estimate_calendar_ar1(e_0, w1_dates)
        record["phi_hat"] = phi_hat
        record["sigma2_hat"] = sigma2_hat
        
        # 5. Precompute harmonic operators & compute observed scores
        M_stack, CTC_stack, _ = precompute_harmonic_operators(w1_dates, proj_resid, periods)
        obs_max, obs_by_t = compute_infradian_scores_batch(e_0[np.newaxis, :], proj_resid, M_stack, CTC_stack)
        
        S_i = float(obs_max[0])
        best_t_idx = int(np.argmax(obs_by_t[0]))
        best_T = float(periods[best_t_idx])
        
        b_c = float((e_0 @ M_stack[:, 2 * best_t_idx]))
        b_s = float((e_0 @ M_stack[:, 2 * best_t_idx + 1]))
        A_i = float(np.sqrt(b_c ** 2 + b_s ** 2))
        
        record["S_i"] = S_i
        record["best_T"] = best_T
        record["A_i"] = A_i
        record["beta_c"] = b_c
        record["beta_s"] = b_s
        
        # 6. Null Calibration via B=199 AR(1) parametric bootstrap
        boot_msg = f"20261020|apple|{cid}|null_bootstrap".encode("utf-8")
        boot_seed = int.from_bytes(hashlib.sha256(boot_msg).digest()[:8], "little")
        rng_boot = np.random.default_rng(boot_seed)
        
        eps_boot = simulate_calendar_ar1(phi_hat, sigma2_hat, w1_dates, b_replicates, rng_boot)
        null_max_scores, _ = compute_infradian_scores_batch(eps_boot, proj_resid, M_stack, CTC_stack)
        
        mean_s_null = float(np.mean(null_max_scores))
        E_i = float(S_i - mean_s_null)
        p_i = float((1.0 + np.sum(null_max_scores >= S_i)) / (b_replicates + 1))
        
        record["mean_S_null"] = mean_s_null
        record["E_i"] = E_i
        record["p_i"] = p_i
        
        # 7. Second-Window Temporal Persistence
        if w2_start is not None:
            record["eligible_w2"] = True
            record["w2_start_date"] = int(w2_start)
            
            w2_dates = select_window_dates(qual_dates, w2_start, cid, k_per_block=15)
            record["w2_actual_span"] = int(w2_dates[-1] - w2_dates[0])
            record["w2_longest_gap"] = int(np.max(np.diff(w2_dates)))
            record["w2_selected_dates"] = w2_dates.tolist()
            
            w2_hr_p10, w2_clock, _, _ = extract_features_on_dates(hr, dates, w2_dates)
            
            # Construct Window 1 fitted model coefficients:
            # Baseline beta_0 = (X_base^T X_base)^{-1} X_base^T w1_hr_p10
            beta_0 = np.linalg.lstsq(X_base, w1_hr_p10, rcond=None)[0]
            
            # Predict Window 2 using frozen Window 1 coefficients:
            # Construct matching X_base on Window 2 dates:
            # Regressors: intercept, elapsed days from w1_start (w2_dates - w1_start), weekdays, clock
            w2_cols = []
            for name in base_names:
                if name == "intercept":
                    w2_cols.append(np.ones(len(w2_dates), dtype=np.float64))
                elif name == "elapsed_days":
                    w2_cols.append((w2_dates - w1_start).astype(np.float64))
                elif name.startswith("weekday_"):
                    # dow = (dates + 3) % 7
                    dow = (w2_dates + 3) % 7
                    day_map = {"weekday_tue": 1, "weekday_wed": 2, "weekday_thu": 3,
                               "weekday_fri": 4, "weekday_sat": 5, "weekday_sun": 6}
                    d_val = day_map.get(name, 0)
                    w2_cols.append((dow == d_val).astype(np.float64))
                elif name == "clock_00_06":
                    w2_cols.append(w2_clock[:, 0])
                elif name == "clock_06_12":
                    w2_cols.append(w2_clock[:, 1])
                elif name == "clock_12_18":
                    w2_cols.append(w2_clock[:, 2])
                elif name == "clock_18_24":
                    w2_cols.append(w2_clock[:, 3])
                else:
                    w2_cols.append(np.zeros(len(w2_dates), dtype=np.float64))
                    
            X_base_w2 = np.column_stack(w2_cols)
            y_base_pred_w2 = X_base_w2 @ beta_0
            
            # Harmonic prediction using frozen best_T, b_c, b_s from Window 1:
            omega_frozen = 2.0 * np.pi / best_T
            t_w2_elapsed = (w2_dates - w1_start).astype(np.float64)
            h_pred_w2 = b_c * np.cos(omega_frozen * t_w2_elapsed) + b_s * np.sin(omega_frozen * t_w2_elapsed)
            y_full_pred_w2 = y_base_pred_w2 + h_pred_w2
            
            # Error metrics on Window 2:
            mse_base = float(np.mean((w2_hr_p10 - y_base_pred_w2) ** 2))
            mse_full = float(np.mean((w2_hr_p10 - y_full_pred_w2) ** 2))
            delta_mse = mse_base - mse_full
            r2_persist = float(delta_mse / max(mse_base, 1e-10))
            
            # Correlation between residual from baseline and predicted harmonic component
            resid_w2 = w2_hr_p10 - y_base_pred_w2
            if np.std(resid_w2) > 1e-6 and np.std(h_pred_w2) > 1e-6:
                corr_persist = float(np.corrcoef(resid_w2, h_pred_w2)[0, 1])
            else:
                corr_persist = 0.0
                
            record["w2_mse_base"] = mse_base
            record["w2_mse_full"] = mse_full
            record["w2_delta_mse"] = delta_mse
            record["w2_r2_persist"] = r2_persist
            record["w2_corr_persist"] = corr_persist
            
            # Re-fit Window 2 independently for period stability comparison:
            X_base_2, _, proj_2 = build_nonperiodic_baseline(w2_dates, w2_clock)
            e_0_2 = proj_2 @ w2_hr_p10
            M_2, CTC_2, _ = precompute_harmonic_operators(w2_dates, proj_2, periods)
            w2_obs_max, w2_obs_by_t = compute_infradian_scores_batch(e_0_2[np.newaxis, :], proj_2, M_2, CTC_2)
            
            w2_S_i = float(w2_obs_max[0])
            w2_best_T = float(periods[int(np.argmax(w2_obs_by_t[0]))])
            
            record["w2_S_i"] = w2_S_i
            record["w2_best_T"] = w2_best_T
            record["delta_T_w1_w2"] = float(abs(best_T - w2_best_T))

    except Exception as e:
        record["error"] = str(e)
        
    return record


# Top-level helper for multiprocessing
def _worker_wrapper(args):
    cid, meta_dict, cache_path, periods, b_reps = args
    return process_participant(cid, meta_dict, cache_path, periods, b_reps)


def main() -> int:
    t_start = time.perf_counter()
    print("=" * 80)
    print("01_extract_and_calibrate_infradian.py")
    print("Longitudinal Exposure Normalization, Null Calibration & Persistence Pipeline")
    print("=" * 80)
    
    # 1. Load Allocations and Manifest
    alloc_path = os.path.join(
        PROJECT_ROOT,
        "experiments/apple_cohort_scaling/artifacts/apple_10_allocations.parquet"
    )
    cache_dir = os.path.join(
        PROJECT_ROOT,
        "experiments/apple_cohort_scaling/artifacts/cache_5min"
    )
    results_dir = os.path.join(
        PROJECT_ROOT,
        "experiments/female_infradian_cycles_v2/results"
    )
    os.makedirs(results_dir, exist_ok=True)
    out_parquet = os.path.join(results_dir, "window_eligibility.parquet")
    
    if not os.path.exists(alloc_path):
        raise FileNotFoundError(f"Allocations file not found: {alloc_path}")
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
        
    alloc = pd.read_parquet(alloc_path)
    n_cohort = len(alloc)
    print(f"Loaded demographic allocations for {n_cohort} Apple Watch participants.")
    
    # Candidate periods: 21 points in [25.0, 35.0] days with step 0.5
    periods = np.arange(25.0, 35.5, 0.5)
    b_replicates = 199
    
    # Prepare participant tasks
    tasks = []
    for _, row in alloc.iterrows():
        cid = int(row["customer_id"])
        c_path = os.path.join(cache_dir, f"customer_{cid}.npz")
        meta = {
            "customer_id": cid,
            "user_id": int(row.get("user_id", cid)),
            "y": int(row["y"]),
            "salutation": int(row.get("salutation", 10 if row["y"] == 0 else 20)),
            "age_group": str(row["age_group"]),
            "bmi_group": str(row["bmi_group"]),
        }
        tasks.append((cid, meta, c_path, periods, b_replicates))
        
    print(f"Prepared {len(tasks)} participant tasks. Starting execution...")
    
    # Run processing with ProcessPoolExecutor
    n_cpus = os.cpu_count() or 8
    max_workers = min(n_cpus, 10)
    print(f"Using {max_workers} worker processes.")
    
    results: List[Dict[str, Any]] = []
    done_count = 0
    t0 = time.perf_counter()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(_worker_wrapper, tasks, chunksize=50):
            results.append(res)
            done_count += 1
            if done_count % 1000 == 0 or done_count == n_cohort:
                rate = done_count / max(time.perf_counter() - t0, 0.01)
                print(f"  Processed {done_count}/{n_cohort} participants ({rate:.1f} part/s)...")
                
    total_elapsed = time.perf_counter() - t_start
    print(f"Extraction and calibration completed in {total_elapsed:.1f} s ({total_elapsed/60:.2f} min).")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Verification checks
    n_w1 = df["eligible_w1"].sum()
    n_w2 = df["eligible_w2"].sum()
    print("-" * 80)
    print(f"Window 1 Eligible (N=4,631 target): {n_w1}")
    print(f"Window 2 Eligible (N=2,324 target): {n_w2}")
    
    print("\nWindow 1 by Age and Salutation:")
    print(df[df["eligible_w1"]].groupby(["age_group", "y"]).size())
    
    print("\nWindow 2 by Age and Salutation:")
    print(df[df["eligible_w2"]].groupby(["age_group", "y"]).size())
    
    assert n_w1 == 4631, f"Expected 4631 w1 eligible, got {n_w1}"
    assert n_w2 == 2324, f"Expected 2324 w2 eligible, got {n_w2}"
    
    # Save output parquet
    df.to_parquet(out_parquet, index=False)
    print(f"\nSaved results to {out_parquet}")
    print(f"File size: {os.path.getsize(out_parquet) / 1024:.1f} KB")
    print("=" * 80)
    print("[SUCCESS] Milestone 1 script 01 execution complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
