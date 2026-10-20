"""
02_verify_q_and_recording_descriptors.py
Test implementation of Q controls and recording descriptors (daily occupancy and nocturnal mask fraction).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

def build_recording_nuisance_design(dates: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Builds nuisance design for recording outcomes:
    - Intercept
    - Elapsed days linear trend
    - Weekday indicators (Tue-Sun, Mon reference)
    Excludes clock coverage fractions.
    
    Deterministic sequential QR / orthogonal projection column screening.
    Returns:
        X_retained: (N, p_retained)
        retained_names: list of column names
        proj_resid: (N, N) residual projection matrix (I - H_0)
    """
    dates = np.asarray(dates, dtype=np.int64)
    n = len(dates)
    
    col_names = [
        "intercept",
        "elapsed_days",
        "weekday_tue",
        "weekday_wed",
        "weekday_thu",
        "weekday_fri",
        "weekday_sat",
        "weekday_sun",
    ]
    
    cols = [
        np.ones(n, dtype=np.float64),
        (dates - dates[0]).astype(np.float64),
    ]
    dow = (dates + 3) % 7 # Mon=0, ..., Sun=6
    for d in range(1, 7):
        cols.append((dow == d).astype(np.float64))
        
    X_raw = np.column_stack(cols)
    retained_cols = []
    retained_names = []
    Q = np.zeros((n, 0), dtype=np.float64)
    
    for idx, name in enumerate(col_names):
        v = X_raw[:, idx].copy()
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        if Q.shape[1] > 0:
            v_proj = Q @ (Q.T @ v)
            v_perp = v - v_proj
            perp_norm = np.linalg.norm(v_perp)
            if perp_norm / v_norm > 1e-4:
                q_new = v_perp / perp_norm
                Q = np.column_stack([Q, q_new])
                retained_cols.append(v)
                retained_names.append(name)
        else:
            q_new = v / v_norm
            Q = q_new[:, np.newaxis]
            retained_cols.append(v)
            retained_names.append(name)
            
    X_retained = np.column_stack(retained_cols)
    proj_resid = np.eye(n, dtype=np.float64) - (Q @ Q.T)
    return X_retained, retained_names, proj_resid


def fit_recording_infradian(
    y: np.ndarray,
    dates: np.ndarray,
    periods: np.ndarray,
) -> tuple[float, float, float, int, int]:
    """
    Fits harmonic cosinor regression across candidate periods on detrended recording outcome y.
    
    Returns:
        A: amplitude sqrt(beta_c^2 + beta_s^2)
        S: partial fit improvement max_T (RSS_0 - RSS_T) / RSS_0
        best_T: best-fit period in candidate grid
        fit_avail: 1 if finite valid fit, 0 if failed
        zero_var: 1 if zero variance / degenerate residual, 0 otherwise
    """
    n = len(y)
    total_ss = float(np.sum((y - np.mean(y)) ** 2))
    
    if np.std(y) < 1e-6:
        return np.nan, np.nan, np.nan, 0, 1
        
    _, _, proj_resid = build_recording_nuisance_design(dates)
    e_0 = proj_resid @ y
    rss_0 = float(np.sum(e_0 ** 2))
    
    if rss_0 <= 1e-12 * max(1.0, total_ss) or not np.isfinite(rss_0):
        return np.nan, np.nan, np.nan, 0, 1
        
    t_elapsed = (dates - dates[0]).astype(np.float64)
    best_S = -1.0
    best_T = np.nan
    best_A = np.nan
    
    for T in periods:
        omega = 2.0 * np.pi / T
        cos_t = np.cos(omega * t_elapsed)
        sin_t = np.sin(omega * t_elapsed)
        C_T = np.column_stack([cos_t, sin_t])
        
        C_tilde = proj_resid @ C_T
        CTC = C_tilde.T @ C_tilde
        det = CTC[0, 0] * CTC[1, 1] - CTC[0, 1] * CTC[1, 0]
        if det < 1e-12:
            CTC[0, 0] += 1e-8
            CTC[1, 1] += 1e-8
            det = CTC[0, 0] * CTC[1, 1] - CTC[0, 1] * CTC[1, 0]
            
        inv_CTC = np.array([
            [CTC[1, 1], -CTC[0, 1]],
            [-CTC[1, 0], CTC[0, 0]]
        ], dtype=np.float64) / det
        
        M_T = C_tilde @ inv_CTC
        beta = e_0 @ M_T
        b_c, b_s = beta[0], beta[1]
        
        delta_rss = b_c ** 2 * CTC[0, 0] + 2.0 * b_c * b_s * CTC[0, 1] + b_s ** 2 * CTC[1, 1]
        s_T = float(np.clip(delta_rss / rss_0, 0.0, 1.0))
        
        # Tie breaking: smallest candidate T on exact tie
        if s_T > best_S:
            best_S = s_T
            best_T = float(T)
            best_A = float(np.sqrt(b_c ** 2 + b_s ** 2))
            
    return best_A, best_S, best_T, 1, 0


def main() -> None:
    elig_path = "experiments/female_infradian_cycles_v2/results/window_eligibility.parquet"
    elig_df = pd.read_parquet(elig_path)
    eligible = elig_df[elig_df["eligible_w1"] == True].head(5)
    
    periods = np.linspace(25.0, 35.0, 21)
    
    for _, row in eligible.iterrows():
        cid = row["customer_id"]
        sel_dates = np.array(row["w1_selected_dates"])
        start_date = row["w1_start_date"]
        
        cache_path = f"experiments/apple_cohort_scaling/artifacts/cache_5min/customer_{cid}.npz"
        with np.load(cache_path) as npz:
            dates = npz["date"]
            hr = npz["hr"]
            
        date_to_idx = {d: i for i, d in enumerate(dates)}
        indices = [date_to_idx[d] for d in sel_dates]
        hr_sel = hr[indices]
        valid_mask = np.isfinite(hr_sel)
        
        # 1. Nonperiodic p10 controls
        p10 = np.empty(len(sel_dates), dtype=np.float64)
        for i in range(len(sel_dates)):
            p10[i] = np.percentile(hr_sel[i, valid_mask[i]], 10)
            
        p10_mean = float(np.mean(p10))
        p10_sd = float(np.std(p10, ddof=1))
        
        t = (sel_dates - sel_dates[0]).astype(np.float64)
        A_lin = np.column_stack([np.ones(len(t)), t])
        beta_lin, rss_lin, _, _ = np.linalg.lstsq(A_lin, p10, rcond=None)
        p10_slope = float(beta_lin[1])
        # linear residual SD: sqrt(RSS / (90 - 2))
        resids = p10 - (beta_lin[0] + beta_lin[1] * t)
        rss_val = np.sum(resids ** 2)
        p10_linear_resid_sd = float(np.sqrt(rss_val / (len(t) - 2)))
        
        # 2. Clock coverage means
        clock_00_06 = np.mean(valid_mask[:, 0:72], axis=1)
        clock_06_12 = np.mean(valid_mask[:, 72:144], axis=1)
        clock_12_18 = np.mean(valid_mask[:, 144:216], axis=1)
        clock_18_24 = np.mean(valid_mask[:, 216:288], axis=1)
        
        cov_00_06_mean = float(np.mean(clock_00_06))
        cov_06_12_mean = float(np.mean(clock_06_12))
        cov_12_18_mean = float(np.mean(clock_12_18))
        cov_18_24_mean = float(np.mean(clock_18_24))
        
        date_span = float(sel_dates[-1] - sel_dates[0])
        
        # Longest gap including boundaries
        gaps = [
            int(sel_dates[0] - start_date), # first - start
            int((start_date + 179) - sel_dates[-1]), # start + 179 - last
        ]
        gaps.extend((np.diff(sel_dates) - 1).tolist())
        longest_gap = float(max(gaps))
        
        # 3. Recording descriptors
        occ_daily = np.mean(valid_mask, axis=1)
        noct_mask = clock_00_06
        
        occ_A, occ_S, occ_T, occ_avail, occ_zvar = fit_recording_infradian(occ_daily, sel_dates, periods)
        noct_A, noct_S, noct_T, noct_avail, noct_zvar = fit_recording_infradian(noct_mask, sel_dates, periods)
        
        print(f"CID {cid}:")
        print(f"  p10: mean={p10_mean:.2f}, sd={p10_sd:.2f}, slope={p10_slope:.4f}, resid_sd={p10_linear_resid_sd:.2f}")
        print(f"  coverage means: [{cov_00_06_mean:.2f}, {cov_06_12_mean:.2f}, {cov_12_18_mean:.2f}, {cov_18_24_mean:.2f}], span={date_span}, longest_gap={longest_gap}")
        print(f"  occ: A={occ_A:.4f}, S={occ_S:.4f}, T={occ_T:.1f}, avail={occ_avail}, zvar={occ_zvar}")
        print(f"  noct: A={noct_A:.4f}, S={noct_S:.4f}, T={noct_T:.1f}, avail={noct_avail}, zvar={noct_zvar}")

if __name__ == "__main__":
    main()
