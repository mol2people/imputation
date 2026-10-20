"""
07_test_q_and_static_speed.py
Test extraction speed of static features and Q controls across participants.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "HEARTBEAT_LAB" / "src"))
from heartbeat_lab.data import STATS

def build_recording_nuisance_design(dates: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
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
) -> tuple[float, float, float, int, int]:
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


def process_user_features(row: dict) -> dict:
    cid = row["customer_id"]
    sel_dates = np.array(row["w1_selected_dates"])
    start_date = row["w1_start_date"]
    
    cache_path = f"experiments/apple_cohort_scaling/artifacts/cache_5min/customer_{cid}.npz"
    with np.load(cache_path) as npz:
        dates = npz["date"]
        hr = npz["hr"]
        stats = npz["stats"]
        
    date_to_idx = {d: i for i, d in enumerate(dates)}
    indices = [date_to_idx[d] for d in sel_dates]
    
    hr_sel = hr[indices] # (90, 288)
    stats_sel = stats[indices] # (90, 8)
    
    # 1. Static features B_L(u): 305 cols
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
    
    # HR status flags
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

def main() -> None:
    elig_path = "experiments/female_infradian_cycles_v2/results/window_eligibility.parquet"
    elig_df = pd.read_parquet(elig_path)
    eligible = elig_df[elig_df["eligible_w1"] == True].head(200).to_dict("records")
    
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(process_user_features, eligible))
    elapsed = time.time() - t0
    print(f"Processed {len(results)} users in {elapsed:.2f}s ({elapsed/len(results)*1000:.1f} ms/user)")
    est_total = (elapsed / len(results)) * 4631
    print(f"Estimated time for 4,631 users: {est_total:.1f}s")
    print(f"Static shape: {results[0]['static'].shape}, Q shape: {results[0]['q'].shape}, R shape: {results[0]['r'].shape}")

if __name__ == "__main__":
    main()
