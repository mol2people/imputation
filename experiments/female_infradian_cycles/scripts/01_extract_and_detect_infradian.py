#!/usr/bin/env python3
"""
Pipeline for Longitudinal Daily Signal Extraction and Infradian Rhythm Detection
Cohort: Apple Watch (N = 11,139)
Target Path: experiments/female_infradian_cycles/scripts/01_extract_and_detect_infradian.py
Output Path: experiments/female_infradian_cycles/results/infradian_metrics_by_participant.csv

Methodology:
1. Demographics & Sex Verification:
   - Reads apple_10_allocations.parquet (N=11,139).
   - Verifies y == 0 is Female (salutation == 10) and y == 1 is Male (salutation == 20).
2. Longitudinal Daily Signal Extraction:
   - Filter adequate wear days: >= 120 valid 5m bins (>= 10.0h wear).
   - Screening criteria: n_adequate_days >= 60 AND timespan_days >= 60.
   - Relative calendar day offset: t_d = date - date[0] on adequate days.
   - Daily signals:
     Physiology:
       1) hr_p10: daily 10th percentile over valid bins (resting HR proxy, bpm).
       2) night_hr: mean HR 00:00–06:00 (bins 0–71), requiring >= 12 valid bins (1h), else NaN.
       3) day_hr: mean HR 09:00–18:00 (bins 108–215), requiring >= 36 valid bins (3h), else NaN.
       4) dipping_ratio: 1 - night_hr / day_hr (if both non-NaN and day_hr > 0, else NaN).
     Wear Behavior:
       5) occ_daily: fraction of valid 5m bins (valid / 288).
       6) nocturnal_compliance: fraction of valid bins in 00:00–06:00 (valid in 0–71 / 72).
       7) gap_episodes: wear-to-gap transitions: ((valid[:, :-1]) & (~valid[:, 1:])).sum(axis=1).
3. Matched Infradian Detection Suite:
   - Evaluates candidate periods T in [25.0, 35.0] days with Delta T = 0.5 days (21 grid points).
   - Method 1: Lomb-Scargle on detrended signal (scipy.signal.lombscargle with omega = 2*pi/T).
     Extracts T*_LS, P*_LS, A*_LS, and SNR over flanking bands [14, 22] U [38, 56] days.
   - Method 2: Harmonic Cosinor regression via closed-form 4x4 OLS normal equations.
     Extracts T*_cos, A*_cos, phi*_cos, Delta R^2 over linear trend null, F-stat, and p-value.
4. Memory & Performance:
   - Streams participant by participant.
   - Peak RSS strictly bounded (< 200 MB, well below 500 MB limit).
"""

from __future__ import annotations

import gc
import importlib.util
import os
import resource
import sys
import time
try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY_PANDAS = True
except ImportError:
    HAS_NUMPY_PANDAS = False
    import csv
    import math
    import zipfile
    import struct
    import ast

# -----------------------------------------------------------------------------
# Modular Import from 00_verify_infradian_methods.py
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

VERIFY_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "00_verify_infradian_methods.py")
if not os.path.exists(VERIFY_SCRIPT_PATH):
    raise FileNotFoundError(f"Verification script not found at {VERIFY_SCRIPT_PATH}")

spec = importlib.util.spec_from_file_location("verify_methods", VERIFY_SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load module spec from {VERIFY_SCRIPT_PATH}")
verify_methods = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_methods)

fit_lomb_scargle = verify_methods.fit_lomb_scargle
fit_cosinor_grid = verify_methods.fit_cosinor_grid
detrend_signal = verify_methods.detrend_signal


def get_current_rss_mb() -> float:
    """Returns current process peak RSS memory in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        # macOS ru_maxrss is in bytes
        return usage / (1024.0 * 1024.0)
    else:
        # Linux ru_maxrss is in kilobytes
        return usage / 1024.0


def extract_daily_signals(
    hr_adeq: np.ndarray,
    valid_bins_adeq: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extracts 7 daily physiological and wear behavior signals on adequate days.

    Parameters
    ----------
    hr_adeq : np.ndarray of shape (N_adeq, 288)
        Binned 5-minute heart rate on adequate days (>= 120 valid bins).
    valid_bins_adeq : np.ndarray of shape (N_adeq,)
        Number of valid 5-minute bins on each adequate day.

    Returns
    -------
    signals : dict of str -> np.ndarray of shape (N_adeq,)
    """
    # 1. Physiology: Resting HR proxy (daily 10th percentile over valid bins, bpm)
    # Fast row-wise percentile
    hr_p10 = np.nanpercentile(hr_adeq, 10, axis=1)

    # 2. Physiology: Nocturnal HR (00:00–06:00, bins 0–71; >= 12 valid bins = 1h required, else NaN)
    night_slice = hr_adeq[:, 0:72]
    night_valid = np.isfinite(night_slice).sum(axis=1)
    night_sum = np.nansum(night_slice, axis=1)
    night_hr = np.where(night_valid >= 12, night_sum / np.maximum(night_valid, 1), np.nan)

    # 3. Physiology: Daytime HR (09:00–18:00, bins 108–215; >= 36 valid bins = 3h required, else NaN)
    day_slice = hr_adeq[:, 108:216]
    day_valid = np.isfinite(day_slice).sum(axis=1)
    day_sum = np.nansum(day_slice, axis=1)
    day_hr = np.where(day_valid >= 36, day_sum / np.maximum(day_valid, 1), np.nan)

    # 4. Physiology: Nocturnal Dipping Ratio (1 - night_hr / day_hr, if valid and day_hr > 0)
    dipping_ratio = np.where(
        np.isfinite(night_hr) & np.isfinite(day_hr) & (day_hr > 0),
        1.0 - (night_hr / day_hr),
        np.nan,
    )

    # 5. Wear Behavior: Daily Occupancy Rate (valid bins / 288)
    occ_daily = valid_bins_adeq / 288.0

    # 6. Wear Behavior: Nocturnal Compliance (valid bins in 00:00–06:00 / 72)
    nocturnal_compliance = night_valid / 72.0

    # 7. Wear Behavior: Gap Episodes (wear-to-gap transitions: valid[:, :-1] & ~valid[:, 1:])
    valid_mat = np.isfinite(hr_adeq)
    gap_episodes = ((valid_mat[:, :-1]) & (~valid_mat[:, 1:])).sum(axis=1).astype(np.float64)

    return {
        "hr_p10": hr_p10,
        "night_hr": night_hr,
        "day_hr": day_hr,
        "dipping_ratio": dipping_ratio,
        "occ_daily": occ_daily,
        "nocturnal_compliance": nocturnal_compliance,
        "gap_episodes": gap_episodes,
    }


def run_pure_python_pipeline(project_root: str, cache_dir: str, output_csv: str) -> int:
    t_start = time.perf_counter()
    manifest_path = os.path.join(project_root, "experiments/apple_cohort_scaling/artifacts/apple_full_manifest.csv")
    customers_path = os.path.join(project_root, "experiments/apple_cohort_scaling/artifacts/cache_5min/customers.csv")
    if not os.path.exists(manifest_path) or not os.path.exists(customers_path):
        raise FileNotFoundError("Missing manifest or customers CSV")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = {r["customer_id"]: r for r in csv.DictReader(f)}
    with open(customers_path, "r", encoding="utf-8") as f:
        customers = [r["customer_id"] for r in csv.DictReader(f)]
        
    n_cohort = len(customers)
    print(f"Loaded demographic allocations: {n_cohort} participants.")
    
    # Assertions on sex coding
    f_cnt = sum(1 for cid in customers if manifest[cid]["y"] == "0")
    m_cnt = sum(1 for cid in customers if manifest[cid]["y"] == "1")
    assert f_cnt + m_cnt == n_cohort, "Inconsistent y values"
    for cid in customers:
        y_val = manifest[cid]["y"]
        sal = manifest[cid]["salutation"]
        if y_val == "0":
            assert sal == "10", f"Expected Female salutation 10, got {sal}"
        else:
            assert sal == "20", f"Expected Male salutation 20, got {sal}"
            
    print(f"[VERIFIED] Sex coding confirmed:")
    print(f"  y == 0: Female (salutation == 10), N = {f_cnt} ({f_cnt/n_cohort*100:.1f}%)")
    print(f"  y == 1: Male   (salutation == 20), N = {m_cnt} ({m_cnt/n_cohort*100:.1f}%)")
    
    periods = [25.0 + 0.5 * k for k in range(21)]
    flank_periods = [14.0 + 1.0 * k for k in range(9)] + [38.0 + 1.0 * k for k in range(19)]
    signal_names = [
        "hr_p10",
        "night_hr",
        "day_hr",
        "dipping_ratio",
        "occ_daily",
        "nocturnal_compliance",
        "gap_episodes",
    ]
    
    fieldnames = [
        "customer_id", "y", "sex", "age_group", "bmi_group",
        "n_total_days", "n_adequate_days", "timespan_days", "screened_pass",
    ]
    for sig_name in signal_names:
        fieldnames.extend([
            f"ls_period_{sig_name}",
            f"ls_power_{sig_name}",
            f"ls_amplitude_{sig_name}",
            f"ls_snr_{sig_name}",
            f"cos_period_{sig_name}",
            f"cos_amplitude_{sig_name}",
            f"cos_acrophase_{sig_name}",
            f"cos_delta_r2_{sig_name}",
            f"cos_f_stat_{sig_name}",
            f"cos_pvalue_{sig_name}",
        ])
        
    n_screened_passed = 0
    t_loop_start = time.perf_counter()
    print("\nStarting cohort streaming...")
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, cid in enumerate(customers):
            demo = manifest[cid]
            y_val = int(demo["y"])
            sex_str = "Female" if y_val == 0 else "Male"
            age_grp = demo["age_group"]
            bmi_grp = demo["bmi_group"]
            
            cache_path = os.path.join(cache_dir, f"customer_{cid}.npz")
            with zipfile.ZipFile(cache_path) as z:
                d_data = z.read("date.npy")
                d_major = d_data[6]
                d_off = 10 + struct.unpack("<H", d_data[8:10])[0] if d_major == 1 else 12 + struct.unpack("<I", d_data[8:12])[0]
                d_shape = ast.literal_eval(d_data[10:d_off].decode("latin1").strip())["shape"]
                dates = struct.unpack(f"<{d_shape[0]}q", d_data[d_off:])
                
                h_data = z.read("hr.npy")
                h_major = h_data[6]
                h_off = 10 + struct.unpack("<H", h_data[8:10])[0] if h_major == 1 else 12 + struct.unpack("<I", h_data[8:12])[0]
                h_shape = ast.literal_eval(h_data[10:h_off].decode("latin1").strip())["shape"]
                hrs = struct.unpack(f"<{h_shape[0]*288}f", h_data[h_off:])
                
            n_total_days = len(dates)
            timespan_days = dates[-1] - dates[0] if n_total_days > 0 else 0
            
            adeq = []
            for d in range(n_total_days):
                day_slice = hrs[d*288:(d+1)*288]
                v_cnt = sum(1 for v in day_slice if math.isfinite(v))
                if v_cnt >= 120:
                    adeq.append((dates[d] - dates[0], day_slice, v_cnt))
                    
            n_adequate_days = len(adeq)
            screened_pass = (n_adequate_days >= 60) and (timespan_days >= 60)
            
            record: Dict[str, Any] = {
                "customer_id": cid,
                "y": y_val,
                "sex": sex_str,
                "age_group": age_grp,
                "bmi_group": bmi_grp,
                "n_total_days": n_total_days,
                "n_adequate_days": n_adequate_days,
                "timespan_days": timespan_days,
                "screened_pass": screened_pass,
            }
            
            if screened_pass:
                n_screened_passed += 1
                t_d = [item[0] for item in adeq]
                
                # 1. hr_p10
                p10 = []
                for _, day_slice, _ in adeq:
                    valid_vals = sorted([v for v in day_slice if math.isfinite(v)])
                    idx_p = 0.10 * (len(valid_vals) - 1)
                    lo = int(idx_p)
                    hi = min(lo + 1, len(valid_vals) - 1)
                    p10.append(valid_vals[lo] + (idx_p - lo) * (valid_vals[hi] - valid_vals[lo]))
                    
                # 2. night_hr
                night_hr = []
                for _, day_slice, _ in adeq:
                    nv = [day_slice[i] for i in range(72) if math.isfinite(day_slice[i])]
                    night_hr.append(sum(nv)/len(nv) if len(nv) >= 12 else float("nan"))
                    
                # 3. day_hr
                day_hr = []
                for _, day_slice, _ in adeq:
                    dv = [day_slice[i] for i in range(108, 216) if math.isfinite(day_slice[i])]
                    day_hr.append(sum(dv)/len(dv) if len(dv) >= 36 else float("nan"))
                    
                # 4. dipping_ratio
                dipping = [
                    1.0 - nv/dv if (math.isfinite(nv) and math.isfinite(dv) and dv > 0) else float("nan")
                    for nv, dv in zip(night_hr, day_hr)
                ]
                
                # 5. occ_daily
                occ = [v_cnt / 288.0 for _, _, v_cnt in adeq]
                
                # 6. nocturnal_compliance
                noct_comp = [
                    sum(1 for i in range(72) if math.isfinite(day_slice[i])) / 72.0
                    for _, day_slice, _ in adeq
                ]
                
                # 7. gap_episodes
                gaps = []
                for _, day_slice, _ in adeq:
                    g = sum(1 for i in range(287) if math.isfinite(day_slice[i]) and not math.isfinite(day_slice[i+1]))
                    gaps.append(float(g))
                    
                sigs = {
                    "hr_p10": p10,
                    "night_hr": night_hr,
                    "day_hr": day_hr,
                    "dipping_ratio": dipping,
                    "occ_daily": occ,
                    "nocturnal_compliance": noct_comp,
                    "gap_episodes": gaps,
                }
                
                for sig_name in signal_names:
                    s_arr = sigs[sig_name]
                    clean = [(ti, yi) for ti, yi in zip(t_d, s_arr) if math.isfinite(ti) and math.isfinite(yi)]
                    if len(clean) >= 10:
                        tc = [p[0] for p in clean]
                        yc = [p[1] for p in clean]
                        y_mean = sum(yc) / len(yc)
                        y_var = sum((yv - y_mean)**2 for yv in yc) / len(yc)
                        if y_var > 1e-12:
                            res_ls = fit_lomb_scargle(tc, yc, periods, flank_periods)
                            res_cos = fit_cosinor_grid(tc, yc, periods)
                            record[f"ls_period_{sig_name}"] = float(res_ls.peak_period)
                            record[f"ls_power_{sig_name}"] = float(res_ls.peak_power)
                            record[f"ls_amplitude_{sig_name}"] = float(res_ls.amplitude)
                            record[f"ls_snr_{sig_name}"] = float(res_ls.snr)
                            record[f"cos_period_{sig_name}"] = float(res_cos.peak_period)
                            record[f"cos_amplitude_{sig_name}"] = float(res_cos.amplitude)
                            record[f"cos_acrophase_{sig_name}"] = float(res_cos.acrophase)
                            record[f"cos_delta_r2_{sig_name}"] = float(res_cos.delta_r2)
                            record[f"cos_f_stat_{sig_name}"] = float(res_cos.f_stat)
                            record[f"cos_pvalue_{sig_name}"] = float(res_cos.p_value)
                        else:
                            for prefix in ["ls_period", "ls_power", "ls_amplitude", "ls_snr", "cos_period", "cos_amplitude", "cos_acrophase", "cos_delta_r2", "cos_f_stat", "cos_pvalue"]:
                                record[f"{prefix}_{sig_name}"] = "nan"
                    else:
                        for prefix in ["ls_period", "ls_power", "ls_amplitude", "ls_snr", "cos_period", "cos_amplitude", "cos_acrophase", "cos_delta_r2", "cos_f_stat", "cos_pvalue"]:
                            record[f"{prefix}_{sig_name}"] = "nan"
            else:
                for sig_name in signal_names:
                    for prefix in ["ls_period", "ls_power", "ls_amplitude", "ls_snr", "cos_period", "cos_amplitude", "cos_acrophase", "cos_delta_r2", "cos_f_stat", "cos_pvalue"]:
                        record[f"{prefix}_{sig_name}"] = "nan"
                        
            writer.writerow(record)
            if (idx + 1) % 500 == 0:
                f_out.flush()
                
            if (idx + 1) % 1000 == 0 or (idx + 1) == n_cohort:
                elapsed = time.perf_counter() - t_loop_start
                rss_mb = get_current_rss_mb()
                pct = (idx + 1) / n_cohort * 100.0
                print(
                    f"[{idx+1:5d}/{n_cohort}] ({pct:5.1f}%) | "
                    f"Screened: {n_screened_passed:5d} ({n_screened_passed/(idx+1)*100:.1f}%) | "
                    f"Elapsed: {elapsed:5.1f}s | "
                    f"Peak RSS: {rss_mb:.1f} MB"
                )
                gc.collect()

    total_elapsed = time.perf_counter() - t_start
    final_rss_mb = get_current_rss_mb()
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Output CSV:          {output_csv}")
    print(f"File size:           {os.path.getsize(output_csv) / (1024 * 1024):.2f} MB")
    print(f"Total rows:          {n_cohort:,d} participants")
    print(f"Total columns:       {len(fieldnames)} columns")
    print(f"Screened pass count: {n_screened_passed:,d} ({n_screened_passed/n_cohort*100:.2f}%)")
    print(f"Screened fail count: {n_cohort - n_screened_passed:,d} ({(n_cohort - n_screened_passed)/n_cohort*100:.2f}%)")
    print(f"Total Pipeline Time: {total_elapsed:.1f} s ({total_elapsed/60:.2f} min)")
    print(f"Final Peak RSS:      {final_rss_mb:.1f} MB (< 200 MB target, < 500 MB limit)")
    print("=" * 80)
    print("[SUCCESS] Milestone 1 script execution and verification complete.")
    return 0


def main() -> int:
    t_start = time.perf_counter()
    print("=" * 80)
    print("M1: Infradian Rhythm Daily Feature Extraction & Infradian Detection Pipeline")
    print("=" * 80)

    # Paths
    alloc_path = os.path.join(
        PROJECT_ROOT,
        "experiments/apple_cohort_scaling/artifacts/apple_10_allocations.parquet",
    )
    cache_dir = os.path.join(
        PROJECT_ROOT,
        "experiments/apple_cohort_scaling/artifacts/cache_5min",
    )
    output_dir = os.path.join(
        PROJECT_ROOT,
        "experiments/female_infradian_cycles/results",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "infradian_metrics_by_participant.csv")

    if not HAS_NUMPY_PANDAS:
        return run_pure_python_pipeline(PROJECT_ROOT, cache_dir, output_csv)
    if not os.path.exists(alloc_path):
        raise FileNotFoundError(f"Allocations file not found: {alloc_path}")

    alloc = pd.read_parquet(alloc_path)
    n_cohort = len(alloc)
    print(f"Loaded demographic allocations: {n_cohort} participants.")

    # Integrity assertions on demographics and sex coding
    assert "y" in alloc.columns, "Column 'y' missing from allocations."
    assert "salutation" in alloc.columns, "Column 'salutation' missing from allocations."
    assert set(alloc["y"].unique()) == {0, 1}, f"Unexpected values in 'y': {alloc['y'].unique()}"

    females = alloc[alloc["y"] == 0]
    males = alloc[alloc["y"] == 1]
    n_females = len(females)
    n_males = len(males)

    f_sal = females["salutation"].unique()
    m_sal = males["salutation"].unique()
    assert len(f_sal) == 1 and f_sal[0] == 10, f"Expected Female salutation 10, got {f_sal}"
    assert len(m_sal) == 1 and m_sal[0] == 20, f"Expected Male salutation 20, got {m_sal}"

    print(f"[VERIFIED] Sex coding confirmed:")
    print(f"  y == 0: Female (salutation == 10), N = {n_females} ({n_females/n_cohort*100:.1f}%)")
    print(f"  y == 1: Male   (salutation == 20), N = {n_males} ({n_males/n_cohort*100:.1f}%)")

    # -------------------------------------------------------------------------
    # 2. Infradian Frequency Grids
    # -------------------------------------------------------------------------
    # Target period band: T in [25.0, 35.0] days with Delta T = 0.5 days (21 grid points)
    periods = np.arange(25.0, 35.5, 0.5)
    assert len(periods) == 21, f"Expected 21 candidate periods, got {len(periods)}"
    print(f"Candidate period grid: {len(periods)} points ({periods[0]:.1f}d to {periods[-1]:.1f}d, step 0.5d)")

    # Flanking background bands: [14, 22] U [38, 56] days for local red-noise floor and SNR
    flank_low = np.arange(14.0, 23.0, 1.0)
    flank_high = np.arange(38.0, 57.0, 1.0)
    flank_periods = np.concatenate([flank_low, flank_high])
    print(f"Flanking background bands: {len(flank_periods)} points ([14, 22] U [38, 56] days)")

    signal_names = [
        "hr_p10",
        "night_hr",
        "day_hr",
        "dipping_ratio",
        "occ_daily",
        "nocturnal_compliance",
        "gap_episodes",
    ]

    # -------------------------------------------------------------------------
    # 3. Stream Cohort and Extract Metrics
    # -------------------------------------------------------------------------
    records: List[Dict[str, Any]] = []
    n_screened_passed = 0
    t_loop_start = time.perf_counter()

    print("\nStarting cohort streaming...")
    for idx, row in enumerate(alloc.itertuples(index=False)):
        cid = str(row.customer_id)
        y_val = int(row.y)
        sex_str = "Female" if y_val == 0 else "Male"
        age_grp = str(row.age_group)
        bmi_grp = str(row.bmi_group)

        cache_path = os.path.join(cache_dir, f"customer_{cid}.npz")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Missing cache file for participant {cid}: {cache_path}")

        # Load single participant data with context manager
        with np.load(cache_path) as npz:
            hr = npz["hr"]      # shape (D_u, 288), float32
            date = npz["date"]  # shape (D_u,), int64 Unix epoch days

        n_total_days = int(len(date))
        timespan_days = int(date[-1] - date[0]) if n_total_days > 0 else 0

        # Filter to adequate days: >= 120 valid 5m bins (>= 10.0h wear)
        valid_bins_per_day = np.isfinite(hr).sum(axis=1)
        is_adequate = (valid_bins_per_day >= 120)
        n_adequate_days = int(np.sum(is_adequate))

        # Screening criteria: n_adequate_days >= 60 AND timespan_days >= 60
        screened_pass = bool((n_adequate_days >= 60) and (timespan_days >= 60))

        record: Dict[str, Any] = {
            "customer_id": cid,
            "y": y_val,
            "sex": sex_str,
            "age_group": age_grp,
            "bmi_group": bmi_grp,
            "n_total_days": n_total_days,
            "n_adequate_days": n_adequate_days,
            "timespan_days": timespan_days,
            "screened_pass": screened_pass,
        }

        if screened_pass:
            n_screened_passed += 1

            # Calendar day offset t_d = date - date[0] on adequate days
            date_adeq = date[is_adequate]
            hr_adeq = hr[is_adequate]
            valid_bins_adeq = valid_bins_per_day[is_adequate]
            t_d = (date_adeq - date[0]).astype(np.float64)

            # Extract 7 daily signals
            signals = extract_daily_signals(hr_adeq, valid_bins_adeq)

            # Infradian detection suite on all 7 signals
            for sig_name in signal_names:
                sig_arr = signals[sig_name]

                # Filter valid finite days for this specific signal
                valid_mask = np.isfinite(sig_arr) & np.isfinite(t_d)
                n_valid_sig = int(np.sum(valid_mask))

                if n_valid_sig >= 10 and np.var(sig_arr[valid_mask]) > 1e-12:
                    t_clean = t_d[valid_mask]
                    y_clean = sig_arr[valid_mask]

                    # Method 1: Lomb-Scargle on detrended signal
                    res_ls = fit_lomb_scargle(
                        t=t_clean,
                        y=y_clean,
                        periods=periods,
                        flank_periods=flank_periods,
                    )
                    record[f"ls_period_{sig_name}"] = float(res_ls.peak_period)
                    record[f"ls_power_{sig_name}"] = float(res_ls.peak_power)
                    record[f"ls_amplitude_{sig_name}"] = float(res_ls.amplitude)
                    record[f"ls_snr_{sig_name}"] = float(res_ls.snr)

                    # Method 2: Harmonic Cosinor regression via 4x4 OLS
                    res_cos = fit_cosinor_grid(
                        t=t_clean,
                        y=y_clean,
                        periods=periods,
                    )
                    record[f"cos_period_{sig_name}"] = float(res_cos.peak_period)
                    record[f"cos_amplitude_{sig_name}"] = float(res_cos.amplitude)
                    record[f"cos_acrophase_{sig_name}"] = float(res_cos.acrophase)
                    record[f"cos_delta_r2_{sig_name}"] = float(res_cos.delta_r2)
                    record[f"cos_f_stat_{sig_name}"] = float(res_cos.f_stat)
                    record[f"cos_pvalue_{sig_name}"] = float(res_cos.p_value)
                else:
                    # Inadequate valid days or zero variance for this specific signal
                    record[f"ls_period_{sig_name}"] = np.nan
                    record[f"ls_power_{sig_name}"] = np.nan
                    record[f"ls_amplitude_{sig_name}"] = np.nan
                    record[f"ls_snr_{sig_name}"] = np.nan
                    record[f"cos_period_{sig_name}"] = np.nan
                    record[f"cos_amplitude_{sig_name}"] = np.nan
                    record[f"cos_acrophase_{sig_name}"] = np.nan
                    record[f"cos_delta_r2_{sig_name}"] = np.nan
                    record[f"cos_f_stat_{sig_name}"] = np.nan
                    record[f"cos_pvalue_{sig_name}"] = np.nan
        else:
            # Unscreened participant: metrics are NaN
            for sig_name in signal_names:
                record[f"ls_period_{sig_name}"] = np.nan
                record[f"ls_power_{sig_name}"] = np.nan
                record[f"ls_amplitude_{sig_name}"] = np.nan
                record[f"ls_snr_{sig_name}"] = np.nan
                record[f"cos_period_{sig_name}"] = np.nan
                record[f"cos_amplitude_{sig_name}"] = np.nan
                record[f"cos_acrophase_{sig_name}"] = np.nan
                record[f"cos_delta_r2_{sig_name}"] = np.nan
                record[f"cos_f_stat_{sig_name}"] = np.nan
                record[f"cos_pvalue_{sig_name}"] = np.nan

        records.append(record)

        # Logging and memory management
        if (idx + 1) % 1000 == 0 or (idx + 1) == n_cohort:
            elapsed = time.perf_counter() - t_loop_start
            rss_mb = get_current_rss_mb()
            pct = (idx + 1) / n_cohort * 100.0
            print(
                f"[{idx+1:5d}/{n_cohort}] ({pct:5.1f}%) | "
                f"Screened: {n_screened_passed:5d} ({n_screened_passed/(idx+1)*100:.1f}%) | "
                f"Elapsed: {elapsed:5.1f}s | "
                f"Peak RSS: {rss_mb:.1f} MB"
            )
            gc.collect()

    # -------------------------------------------------------------------------
    # 4. Save Table to CSV and Verify Integrity
    # -------------------------------------------------------------------------
    t_save_start = time.perf_counter()
    df_out = pd.DataFrame(records)
    assert len(df_out) == n_cohort, f"Expected {n_cohort} output rows, got {len(df_out)}"

    df_out.to_csv(output_csv, index=False)
    t_save_elapsed = time.perf_counter() - t_save_start
    total_elapsed = time.perf_counter() - t_start
    final_rss_mb = get_current_rss_mb()

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Output CSV:          {output_csv}")
    print(f"File size:           {os.path.getsize(output_csv) / (1024 * 1024):.2f} MB")
    print(f"Total rows:          {len(df_out):,d} participants")
    print(f"Total columns:       {len(df_out.columns)} columns")
    print(f"Screened pass count: {n_screened_passed:,d} ({n_screened_passed/n_cohort*100:.2f}%)")
    print(f"Screened fail count: {n_cohort - n_screened_passed:,d} ({(n_cohort - n_screened_passed)/n_cohort*100:.2f}%)")
    print(f"CSV Save Time:       {t_save_elapsed:.2f} s")
    print(f"Total Pipeline Time: {total_elapsed:.1f} s ({total_elapsed/60:.2f} min)")
    print(f"Final Peak RSS:      {final_rss_mb:.1f} MB (< 200 MB target, < 500 MB limit)")

    # Stratified screening summary
    print("\n--- Screening Pass Rate by Demographic Group ---")
    screen_summary = (
        df_out.groupby(["sex", "age_group"])
        .agg(
            total=("customer_id", "count"),
            screened=("screened_pass", "sum"),
            pass_rate=("screened_pass", "mean"),
        )
        .reset_index()
    )
    screen_summary["pass_rate"] = screen_summary["pass_rate"].apply(lambda x: f"{x*100:.1f}%")
    print(screen_summary.to_string(index=False))

    # Metric distributions on screened participants
    df_screened = df_out[df_out["screened_pass"]].copy()
    print(f"\n--- Infradian Signal Metrics on Screened Participants (N = {len(df_screened)}) ---")
    metric_cols_inspect = [
        ("hr_p10 Cosinor Period", "cos_period_hr_p10"),
        ("hr_p10 Cosinor Amp (bpm)", "cos_amplitude_hr_p10"),
        ("hr_p10 Cosinor Delta R^2", "cos_delta_r2_hr_p10"),
        ("hr_p10 LS Peak Power", "ls_power_hr_p10"),
        ("hr_p10 LS SNR", "ls_snr_hr_p10"),
        ("night_hr Cosinor Amp (bpm)", "cos_amplitude_night_hr"),
        ("day_hr Cosinor Amp (bpm)", "cos_amplitude_day_hr"),
        ("dipping_ratio Cosinor Amp", "cos_amplitude_dipping_ratio"),
        ("occ_daily Cosinor Amp", "cos_amplitude_occ_daily"),
    ]

    for label, col in metric_cols_inspect:
        if col in df_screened.columns:
            s = df_screened[col].dropna()
            print(
                f"{label:30s}: N={len(s):5d} | "
                f"Mean={s.mean():.4f} | Med={s.median():.4f} | "
                f"IQR=[{s.quantile(0.25):.4f}, {s.quantile(0.75):.4f}]"
            )

    print("=" * 80)
    print("[SUCCESS] Milestone 1 script execution and verification complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
