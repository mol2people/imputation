#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
00_verify_v2_methods.py

Fast Unit Verification Fixture for female_infradian_cycles_v2.
Verifies in <5s:
1. Calendar-time AR(1) estimation on unevenly spaced daily timestamps.
2. Nonperiodic baseline detrending (intercept, elapsed days, weekday indicators, clock coverage fractions).
3. Infradian search score S_i across 21 candidate periods T in [25, 35] days in steps of 0.5 days.
4. Dependence-aware null calibration E_i = S_i - mean(S_{null}) on synthetic test signals.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple, List, Dict, Any

import numpy as np
import scipy.optimize


# =============================================================================
# Core Mathematical Functions
# =============================================================================

def estimate_calendar_ar1(residuals: np.ndarray, dates: np.ndarray) -> Tuple[float, float]:
    """
    Fits stationary calendar-time AR(1) daily residual parameters:
    correlation phi^{|t - s|}, 0 <= phi < 1, and stationary variance sigma^2.
    
    Handles unevenly spaced calendar timestamps t_k via exact profile Gaussian likelihood.
    
    Parameters
    ----------
    residuals : np.ndarray of shape (N,)
        Residuals from nonperiodic baseline fit (assumed zero-mean).
    dates : np.ndarray of shape (N,)
        Calendar timestamps (integer days, strictly increasing).
        
    Returns
    -------
    phi : float
        Estimated 1-day autocorrelation parameter, constrained to [0, 0.999].
    sigma2 : float
        Estimated stationary variance of the process.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    dates = np.asarray(dates, dtype=np.int64)
    n = len(residuals)
    if n < 3:
        raise ValueError(f"At least 3 observations required for AR(1) estimation, got {n}")
    
    delta_t = np.diff(dates)
    if np.any(delta_t <= 0):
        raise ValueError("Calendar timestamps must be strictly increasing.")
    
    y = residuals
    y1_sq = y[0] ** 2
    y_prev = y[:-1]
    y_curr = y[1:]
    
    def neg_profile_log_lik(phi: float) -> float:
        if phi <= 1e-6:
            # White noise limit
            s2 = (y1_sq + np.sum((y_curr - phi ** delta_t * y_prev) ** 2)) / n
            return n * np.log(max(s2, 1e-12))
        
        phi_dt = phi ** delta_t
        one_minus_phi2_dt = 1.0 - phi_dt ** 2
        # Bound away from zero for numerical safety
        one_minus_phi2_dt = np.maximum(one_minus_phi2_dt, 1e-12)
        
        innovations_sq = (y_curr - phi_dt * y_prev) ** 2
        sum_scaled_sq = y1_sq + np.sum(innovations_sq / one_minus_phi2_dt)
        sigma2_hat = sum_scaled_sq / n
        if sigma2_hat <= 1e-12:
            return 1e12
        
        log_det = np.sum(np.log(one_minus_phi2_dt))
        return n * np.log(sigma2_hat) + log_det

    res = scipy.optimize.minimize_scalar(
        neg_profile_log_lik,
        bounds=(0.0, 0.995),
        method="bounded",
        options={"xatol": 1e-4}
    )
    
    phi_hat = float(np.clip(res.x, 0.0, 0.995))
    
    # Compute profile sigma2 at optimal phi
    if phi_hat <= 1e-6:
        sigma2_hat = float(np.mean(y ** 2))
    else:
        phi_dt = phi_hat ** delta_t
        one_minus_phi2_dt = np.maximum(1.0 - phi_dt ** 2, 1e-12)
        innovations_sq = (y_curr - phi_dt * y_prev) ** 2
        sigma2_hat = float((y1_sq + np.sum(innovations_sq / one_minus_phi2_dt)) / n)
        
    return phi_hat, max(sigma2_hat, 1e-12)


def simulate_calendar_ar1(
    phi: float,
    sigma2: float,
    dates: np.ndarray,
    n_reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulates n_reps sample paths of stationary calendar-time AR(1) noise
    on exact calendar timestamps.
    
    Parameters
    ----------
    phi : float
        1-day correlation parameter (0 <= phi < 1).
    sigma2 : float
        Stationary variance.
    dates : np.ndarray of shape (N,)
        Calendar timestamps.
    n_reps : int
        Number of simulation replicates.
    rng : np.random.Generator
        NumPy random number generator.
        
    Returns
    -------
    sims : np.ndarray of shape (n_reps, N)
        Simulated AR(1) realizations.
    """
    dates = np.asarray(dates, dtype=np.int64)
    n = len(dates)
    sigma = np.sqrt(max(sigma2, 1e-12))
    delta_t = np.diff(dates)
    
    z = rng.standard_normal((n_reps, n))
    out = np.empty((n_reps, n), dtype=np.float64)
    out[:, 0] = sigma * z[:, 0]
    
    for k in range(n - 1):
        dt = delta_t[k]
        decay = phi ** dt
        scale = sigma * np.sqrt(max(0.0, 1.0 - decay ** 2))
        out[:, k + 1] = decay * out[:, k] + scale * z[:, k + 1]
        
    return out


def build_nonperiodic_baseline(
    dates: np.ndarray,
    clock_fractions: np.ndarray,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Builds the nonperiodic baseline design matrix:
    - Intercept
    - Elapsed days linear trend
    - Weekday indicators (Tue-Sun, Mon reference)
    - Four 6-hour clock coverage fractions (00-06, 06-12, 12-18, 18-24)
    
    Deterministically drops redundant/collinear columns to guarantee full column rank.
    
    Parameters
    ----------
    dates : np.ndarray of shape (N,)
        Calendar dates (epoch days).
    clock_fractions : np.ndarray of shape (N, 4)
        Fractions of valid 5m bins in [00:00-06:00, 06:00-12:00, 12:00-18:00, 18:00-24:00].
        
    Returns
    -------
    X_retained : np.ndarray of shape (N, p_retained)
        Full-rank retained baseline design matrix.
    retained_names : list of str
        Names of retained columns.
    proj_resid : np.ndarray of shape (N, N)
        Residual projection matrix (I - H_0) where H_0 = X (X^T X)^{-1} X^T.
    """
    dates = np.asarray(dates, dtype=np.int64)
    n = len(dates)
    clock_fractions = np.asarray(clock_fractions, dtype=np.float64)
    if clock_fractions.shape != (n, 4):
        raise ValueError(f"clock_fractions must have shape ({n}, 4), got {clock_fractions.shape}")
        
    # Candidate columns
    col_names = [
        "intercept",
        "elapsed_days",
        "weekday_tue",
        "weekday_wed",
        "weekday_thu",
        "weekday_fri",
        "weekday_sat",
        "weekday_sun",
        "clock_00_06",
        "clock_06_12",
        "clock_12_18",
        "clock_18_24",
    ]
    
    cols = []
    # Intercept
    cols.append(np.ones(n, dtype=np.float64))
    # Elapsed days
    cols.append((dates - dates[0]).astype(np.float64))
    # Weekday indicators: (dates + 3) % 7 where 0=Mon, 1=Tue, ..., 6=Sun
    dow = (dates + 3) % 7
    for d in range(1, 7): # Tue=1 to Sun=6
        cols.append((dow == d).astype(np.float64))
    # Four clock coverage fractions
    for c in range(4):
        cols.append(clock_fractions[:, c].astype(np.float64))
        
    X_raw = np.column_stack(cols)
    
    # Deterministic sequential QR / orthogonal projection column screening
    retained_cols = []
    retained_names = []
    
    # Q will hold orthonormal basis for retained columns
    Q = np.zeros((n, 0), dtype=np.float64)
    
    for idx, name in enumerate(col_names):
        v = X_raw[:, idx].copy()
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            # Column is practically zero
            continue
            
        if Q.shape[1] > 0:
            # Project v onto Q
            v_proj = Q @ (Q.T @ v)
            v_perp = v - v_proj
            perp_norm = np.linalg.norm(v_perp)
            # Retain only if perpendicular component is significant relative to original norm
            if perp_norm / v_norm > 1e-4:
                q_new = v_perp / perp_norm
                Q = np.column_stack([Q, q_new])
                retained_cols.append(v)
                retained_names.append(name)
        else:
            # First column (intercept)
            q_new = v / v_norm
            Q = q_new[:, np.newaxis]
            retained_cols.append(v)
            retained_names.append(name)
            
    X_retained = np.column_stack(retained_cols)
    
    # Residual projection matrix (I - H_0) = I - Q Q^T
    proj_resid = np.eye(n, dtype=np.float64) - (Q @ Q.T)
    
    return X_retained, retained_names, proj_resid


def precompute_harmonic_operators(
    dates: np.ndarray,
    proj_resid: np.ndarray,
    periods: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precomputes Frisch-Waugh-Lovell harmonic projection operators for all candidate periods T.
    
    For each T, the two harmonic regressors [cos(2*pi*t/T), sin(2*pi*t/T)] are projected
    onto the orthogonal complement of the baseline:
        C_T_tilde = (I - H_0) C_T
        G_T = (C_T_tilde^T C_T_tilde)^{-1}  (2x2)
        M_T = C_T_tilde G_T                  (N x 2)
        
    Parameters
    ----------
    dates : np.ndarray of shape (N,)
        Calendar dates.
    proj_resid : np.ndarray of shape (N, N)
        (I - H_0) operator from baseline.
    periods : np.ndarray of shape (K,)
        Candidate period grid points.
        
    Returns
    -------
    M_stack : np.ndarray of shape (N, 2*K)
        Stacked M_T operators enabling batched matrix multiplication.
    CTC_stack : np.ndarray of shape (K, 2, 2)
        Precomputed C_T_tilde^T C_T_tilde matrices.
    C_tilde_stack : np.ndarray of shape (K, N, 2)
        Residualized harmonic regressors.
    """
    n = len(dates)
    k_periods = len(periods)
    t_elapsed = (dates - dates[0]).astype(np.float64)
    
    M_stack = np.empty((n, 2 * k_periods), dtype=np.float64)
    CTC_stack = np.empty((k_periods, 2, 2), dtype=np.float64)
    C_tilde_stack = np.empty((k_periods, n, 2), dtype=np.float64)
    
    for i, T in enumerate(periods):
        omega = 2.0 * np.pi / T
        cos_t = np.cos(omega * t_elapsed)
        sin_t = np.sin(omega * t_elapsed)
        C_T = np.column_stack([cos_t, sin_t])
        
        # Project onto baseline orthogonal complement
        C_tilde = proj_resid @ C_T
        CTC = C_tilde.T @ C_tilde
        
        # Invert 2x2 matrix
        det = CTC[0, 0] * CTC[1, 1] - CTC[0, 1] * CTC[1, 0]
        if det < 1e-12:
            # Near singular, add tiny ridge
            CTC[0, 0] += 1e-8
            CTC[1, 1] += 1e-8
            det = CTC[0, 0] * CTC[1, 1] - CTC[0, 1] * CTC[1, 0]
            
        inv_CTC = np.array([
            [CTC[1, 1], -CTC[0, 1]],
            [-CTC[1, 0], CTC[0, 0]]
        ], dtype=np.float64) / det
        
        M_T = C_tilde @ inv_CTC # Shape (N, 2)
        
        M_stack[:, 2 * i : 2 * i + 2] = M_T
        CTC_stack[i] = CTC
        C_tilde_stack[i] = C_tilde
        
    return M_stack, CTC_stack, C_tilde_stack


def compute_infradian_scores_batch(
    residual_batch: np.ndarray,
    proj_resid: np.ndarray,
    M_stack: np.ndarray,
    CTC_stack: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates infradian scores across candidate periods for a batch of residual vectors.
    
    Parameters
    ----------
    residual_batch : np.ndarray of shape (B, N)
        Batch of residual vectors or simulated replicates.
    proj_resid : np.ndarray of shape (N, N)
        (I - H_0) projection matrix.
    M_stack : np.ndarray of shape (N, 2*K)
        Stacked projection operators.
    CTC_stack : np.ndarray of shape (K, 2, 2)
        Precomputed 2x2 normal equation matrices.
        
    Returns
    -------
    max_scores : np.ndarray of shape (B,)
        Searched infradian score S = max_T (RSS_0 - RSS_T)/RSS_0 for each row.
    scores_by_T : np.ndarray of shape (B, K)
        Score for each period T.
    """
    # Ensure orthogonal projection onto baseline: e_proj = residual_batch @ (I - H_0)
    e_proj = residual_batch @ proj_resid # Shape (B, N)
    rss_0 = np.sum(e_proj ** 2, axis=1) # Shape (B,)
    rss_0_safe = np.maximum(rss_0, 1e-12)
    
    # Batched matrix product to get coefficients for all periods:
    # beta_all = e_proj @ M_stack has shape (B, 2*K)
    beta_all = e_proj @ M_stack
    
    b_count, n_cols = beta_all.shape
    k_periods = len(CTC_stack)
    scores_by_T = np.empty((b_count, k_periods), dtype=np.float64)
    
    for i in range(k_periods):
        b_c = beta_all[:, 2 * i]
        b_s = beta_all[:, 2 * i + 1]
        ctc = CTC_stack[i]
        # Delta RSS = b_c^2 * ctc[0,0] + 2 * b_c * b_s * ctc[0,1] + b_s^2 * ctc[1,1]
        delta_rss = b_c ** 2 * ctc[0, 0] + 2.0 * b_c * b_s * ctc[0, 1] + b_s ** 2 * ctc[1, 1]
        scores_by_T[:, i] = np.clip(delta_rss / rss_0_safe, 0.0, 1.0)
        
    max_scores = np.max(scores_by_T, axis=1)
    return max_scores, scores_by_T


def evaluate_participant_infradian(
    dates: np.ndarray,
    y: np.ndarray,
    clock_fractions: np.ndarray,
    periods: np.ndarray,
    b_replicates: int = 199,
    seed: int = 20261020,
) -> Dict[str, Any]:
    """
    Complete participant-level evaluation:
    1. Baseline detrending.
    2. Exact AR(1) estimation.
    3. Observed score S_i, best-fit period T_i, amplitude A_i.
    4. Fast batched AR(1) bootstrap null calibration (B replicates) -> E_i, p_i.
    """
    dates = np.asarray(dates, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    n = len(dates)
    
    # 1. Nonperiodic baseline
    X_base, base_names, proj_resid = build_nonperiodic_baseline(dates, clock_fractions)
    e_0 = proj_resid @ y
    rss_0 = float(np.sum(e_0 ** 2))
    
    # 2. Stationary AR(1) estimation on exact timestamps
    phi_hat, sigma2_hat = estimate_calendar_ar1(e_0, dates)
    
    # 3. Precompute harmonic operators
    M_stack, CTC_stack, C_tilde_stack = precompute_harmonic_operators(dates, proj_resid, periods)
    
    # 4. Observed score
    observed_batch = e_0[np.newaxis, :]
    s_obs_max, s_obs_by_t = compute_infradian_scores_batch(observed_batch, proj_resid, M_stack, CTC_stack)
    S_i = float(s_obs_max[0])
    best_t_idx = int(np.argmax(s_obs_by_t[0]))
    best_T = float(periods[best_t_idx])
    
    # Extract coefficients at best T
    b_c = float((e_0 @ M_stack[:, 2 * best_t_idx]))
    b_s = float((e_0 @ M_stack[:, 2 * best_t_idx + 1]))
    A_i = float(np.sqrt(b_c ** 2 + b_s ** 2))
    
    # 5. Dependence-aware null simulation
    rng = np.random.default_rng(seed)
    eps_boot = simulate_calendar_ar1(phi_hat, sigma2_hat, dates, b_replicates, rng)
    
    null_max_scores, _ = compute_infradian_scores_batch(eps_boot, proj_resid, M_stack, CTC_stack)
    mean_s_null = float(np.mean(null_max_scores))
    E_i = float(S_i - mean_s_null)
    p_i = float((1.0 + np.sum(null_max_scores >= S_i)) / (b_replicates + 1))
    
    return {
        "S_i": S_i,
        "best_T": best_T,
        "A_i": A_i,
        "beta_c": b_c,
        "beta_s": b_s,
        "phi_hat": phi_hat,
        "sigma2_hat": sigma2_hat,
        "E_i": E_i,
        "p_i": p_i,
        "mean_S_null": mean_s_null,
        "base_names": base_names,
        "rss_0": rss_0,
    }


# =============================================================================
# Unit Test Suite
# =============================================================================

def test_calendar_ar1_estimation():
    """Verifies calendar-time AR(1) estimation on unevenly spaced daily timestamps."""
    print("  Testing calendar-time AR(1) parameter recovery...")
    rng = np.random.default_rng(42)
    # 90 dates with uneven daily spacing
    gaps = rng.integers(1, 4, size=89)
    dates = np.concatenate([[18000], 18000 + np.cumsum(gaps)])
    
    true_phi = 0.55
    true_sigma2 = 2.25
    
    # Simulate AR(1) series
    sims = simulate_calendar_ar1(true_phi, true_sigma2, dates, n_reps=1, rng=rng)
    y_synth = sims[0]
    
    phi_est, sigma2_est = estimate_calendar_ar1(y_synth, dates)
    
    assert 0.0 <= phi_est < 1.0, f"phi_est {phi_est} out of bounds [0, 1)"
    assert 0.25 <= phi_est <= 0.85, f"phi_est {phi_est:.3f} failed recovery of true {true_phi}"
    assert 1.0 <= sigma2_est <= 4.0, f"sigma2_est {sigma2_est:.3f} failed recovery of true {true_sigma2}"
    print(f"    [OK] Recovered phi={phi_est:.3f} (true={true_phi}), sigma2={sigma2_est:.3f} (true={true_sigma2})")


def test_baseline_detrending():
    """Verifies nonperiodic baseline construction, redundancy handling, and full rank."""
    print("  Testing nonperiodic baseline detrending and full-rank projection...")
    rng = np.random.default_rng(123)
    dates = np.sort(rng.choice(np.arange(18000, 18180), size=90, replace=False))
    
    # 4 clock coverage fractions
    clock_fracs = rng.uniform(0.6, 1.0, size=(90, 4))
    
    X_base, names, proj_resid = build_nonperiodic_baseline(dates, clock_fracs)
    
    assert X_base.shape[0] == 90, f"Expected 90 rows, got {X_base.shape[0]}"
    assert "intercept" in names, "Missing intercept in baseline"
    assert "elapsed_days" in names, "Missing elapsed_days in baseline"
    assert "weekday_tue" in names, "Missing weekday indicators"
    
    # Verify full column rank
    rank = np.linalg.matrix_rank(X_base)
    assert rank == X_base.shape[1], f"Matrix rank {rank} < columns {X_base.shape[1]}"
    
    # Verify projection matrix properties: (I - H) is symmetric and idempotent
    assert np.allclose(proj_resid, proj_resid.T, atol=1e-10), "proj_resid not symmetric"
    assert np.allclose(proj_resid @ proj_resid, proj_resid, atol=1e-10), "proj_resid not idempotent"
    
    # Verify orthogonality to columns of X_base
    zeros_check = proj_resid @ X_base
    assert np.allclose(zeros_check, 0.0, atol=1e-9), "proj_resid not orthogonal to X_base"
    print(f"    [OK] Retained {len(names)} columns with full numerical rank {rank}")


def test_infradian_score_and_null_calibration():
    """Verifies infradian score peak detection and AR(1) null calibration in <5s."""
    print("  Testing infradian score peak detection and null calibration...")
    rng = np.random.default_rng(20261020)
    dates = np.sort(rng.choice(np.arange(18000, 18180), size=90, replace=False))
    clock_fracs = rng.uniform(0.6, 1.0, size=(90, 4))
    periods = np.arange(25.0, 35.5, 0.5)
    
    # Case 1: Pure AR(1) noise (Null signal)
    phi_null = 0.50
    sigma2_null = 1.50
    noise = simulate_calendar_ar1(phi_null, sigma2_null, dates, n_reps=1, rng=rng)[0]
    
    res_null = evaluate_participant_infradian(
        dates=dates,
        y=noise,
        clock_fractions=clock_fracs,
        periods=periods,
        b_replicates=199,
        seed=101,
    )
    
    # Under null, E_i should be close to 0 and p_i should not be highly significant
    assert -0.20 <= res_null["E_i"] <= 0.20, f"Null E_i={res_null['E_i']:.4f} too large"
    assert res_null["p_i"] > 0.02, f"Null p_i={res_null['p_i']:.4f} suspiciously small for null"
    print(f"    [OK] Null signal: S_i={res_null['S_i']:.4f}, mean(S_null)={res_null['mean_S_null']:.4f}, E_i={res_null['E_i']:.4f}, p_i={res_null['p_i']:.4f}")
    
    # Case 2: Injected 28.0-day sinusoid + AR(1) noise
    t_elapsed = (dates - dates[0]).astype(np.float64)
    true_T = 28.0
    true_A = 1.80
    signal = true_A * np.cos(2.0 * np.pi * t_elapsed / true_T) + noise
    
    res_sig = evaluate_participant_infradian(
        dates=dates,
        y=signal,
        clock_fractions=clock_fracs,
        periods=periods,
        b_replicates=199,
        seed=102,
    )
    
    assert abs(res_sig["best_T"] - true_T) <= 0.5, f"Recovered T={res_sig['best_T']} != true {true_T}"
    assert res_sig["E_i"] > 0.15, f"Signal excess score E_i={res_sig['E_i']:.4f} too small"
    assert res_sig["p_i"] <= 0.05, f"Signal p_i={res_sig['p_i']:.4f} failed significance"
    assert 1.0 <= res_sig["A_i"] <= 2.6, f"Recovered amplitude {res_sig['A_i']:.3f} failed recovery of true {true_A}"
    print(f"    [OK] Injected 28d signal: best_T={res_sig['best_T']:.1f}d, A_i={res_sig['A_i']:.3f} bpm, E_i={res_sig['E_i']:.4f}, p_i={res_sig['p_i']:.4f}")


def main() -> int:
    t_start = time.perf_counter()
    print("=" * 80)
    print("00_verify_v2_methods.py: Fast Unit Verification Fixture")
    print("=" * 80)
    
    test_calendar_ar1_estimation()
    test_baseline_detrending()
    test_infradian_score_and_null_calibration()
    
    elapsed = time.perf_counter() - t_start
    print("=" * 80)
    print(f"[PASSED] All unit tests passed in {elapsed:.3f} seconds (< 5.0s requirement).")
    print("=" * 80)
    
    if elapsed >= 5.0:
        print(f"[FAIL] Execution took {elapsed:.3f}s >= 5.0s limit!")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
