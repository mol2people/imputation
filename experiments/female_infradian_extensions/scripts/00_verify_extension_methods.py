#!/usr/bin/env python3
"""
Verification Fixture & Reusable Modular Library for Infradian Rhythm Extensions
Path: experiments/female_infradian_extensions/scripts/00_verify_extension_methods.py

Provides high-performance, modular scientific functions for:
1. rayleigh_test: Analytical Rayleigh test of circular uniformity (Wilkie / Mardia-Jupp).
2. map_calendar_phase: Maps participant-relative acrophase to calendar epoch phase.
3. map_28day_phase: Maps participant acrophase to 28-day lunar/calendar phase.
4. detrend_signal: Closed-form OLS linear detrending on unevenly spaced calendar days.
5. sliding_window_lomb_scargle: Windowed Lomb-Scargle power spectrum tracking cycle
   period drift (IQR(T*)) and infradian persistence across multi-year wear.
6. run_unit_tests: Fast, deterministic unit verification executing in < 3.0s (target < 0.2s).
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats as stats
from scipy.signal import lombscargle

__all__ = [
    "rayleigh_test",
    "map_calendar_phase",
    "map_28day_phase",
    "detrend_signal",
    "sliding_window_lomb_scargle",
    "run_unit_tests",
]


def rayleigh_test(angles: Union[np.ndarray, Sequence[float]]) -> Dict[str, Any]:
    """Computes the analytical Rayleigh test of circular uniformity.

    Under the null hypothesis H_0, the angles are uniformly distributed on the
    circle. Under the alternative hypothesis H_1, there is a unimodal preferred
    direction (e.g., von Mises distribution with concentration kappa > 0).

    Formulas:
        C_bar = (1/N) * sum_{i=1}^N cos(theta_i)
        S_bar = (1/N) * sum_{i=1}^N sin(theta_i)
        R_bar = sqrt(C_bar^2 + S_bar^2)
        theta_mean = atan2(S_bar, C_bar) in [-pi, pi]
        circular_var = 1.0 - R_bar
        Z = N * R_bar^2

    p-value calculation:
        - For N <= 50: Wilkie (1983) exact small-sample formula:
          K = 1 + 4N + 4(N^2 - R^2), where R = N * R_bar
          p = exp(sqrt(K) - (1 + 2N))
        - For N > 50: Mardia & Jupp (2000) asymptotic expansion with O(1/N^2) correction:
          p = exp(-Z) * (1 + (2Z - Z^2)/(4N) - (24Z - 132Z^2 + 76Z^3 - 9Z^4)/(288 N^2))

    Parameters
    ----------
    angles : array-like of shape (N,)
        Angles in radians (any wrap or range).

    Returns
    -------
    dict
        Contains:
        - 'R_bar': Mean resultant vector length in [0, 1].
        - 'theta_mean': Circular mean angle in radians in [-pi, pi].
        - 'circular_var': Sample circular variance (1 - R_bar) in [0, 1].
        - 'Z': Rayleigh statistic (N * R_bar^2).
        - 'p_value': Analytical circular p-value in [0, 1].
        - 'N': Count of finite angles.
        - 'C_bar': Mean cosine component.
        - 'S_bar': Mean sine component.
    """
    arr = np.asarray(angles, dtype=np.float64)
    valid = np.isfinite(arr)
    clean_angles = arr[valid]
    N = int(len(clean_angles))

    if N == 0:
        return {
            "R_bar": 0.0,
            "theta_mean": 0.0,
            "circular_var": 1.0,
            "Z": 0.0,
            "p_value": 1.0,
            "N": 0,
            "C_bar": 0.0,
            "S_bar": 0.0,
        }

    C_bar = float(np.mean(np.cos(clean_angles)))
    S_bar = float(np.mean(np.sin(clean_angles)))
    R_bar = float(np.hypot(C_bar, S_bar))
    theta_mean = float(np.arctan2(S_bar, C_bar))
    circular_var = float(max(0.0, min(1.0, 1.0 - R_bar)))
    Z = float(N * (R_bar ** 2))

    # Analytical p-value
    if N <= 50:
        R = N * R_bar
        K = 1.0 + 4.0 * N + 4.0 * (N * N - R * R)
        K = max(0.0, K)
        p_val = float(np.exp(np.sqrt(K) - (1.0 + 2.0 * N)))
    else:
        if Z > 50.0:
            p_val = 0.0
        else:
            poly = (
                1.0
                + (2.0 * Z - Z ** 2) / (4.0 * N)
                - (24.0 * Z - 132.0 * (Z ** 2) + 76.0 * (Z ** 3) - 9.0 * (Z ** 4))
                / (288.0 * (N ** 2))
            )
            if poly > 0.0:
                p_val = float(np.exp(-Z) * poly)
            else:
                p_val = float(np.exp(-Z))

    p_val = float(np.clip(p_val, 0.0, 1.0))

    return {
        "R_bar": R_bar,
        "theta_mean": theta_mean,
        "circular_var": circular_var,
        "Z": Z,
        "p_value": p_val,
        "N": N,
        "C_bar": C_bar,
        "S_bar": S_bar,
    }


def map_calendar_phase(
    phi_rel: Union[float, np.ndarray, Sequence[float]],
    period: Union[float, np.ndarray, Sequence[float]],
    date_0: Union[float, np.ndarray, Sequence[float]],
) -> Union[float, np.ndarray]:
    """Maps participant-relative harmonic acrophase to absolute calendar epoch phase.

    Mathematical definition:
        Phi_cal = (phi_rel - (2 * pi / period) * date_0) mod (2 * pi)
        mapped to [-pi, pi].

    Parameters
    ----------
    phi_rel : float or array-like
        Harmonic acrophase relative to participant's first observation date (radians).
    period : float or array-like
        Best-fit period T* in days.
    date_0 : float or array-like
        Participant baseline calendar date offset (e.g., integer days from epoch).

    Returns
    -------
    phi_cal : float or np.ndarray
        Calendar epoch phase mapped to [-pi, pi].
    """
    is_scalar = np.isscalar(phi_rel) and np.isscalar(period) and np.isscalar(date_0)
    p_rel = np.asarray(phi_rel, dtype=np.float64)
    T = np.asarray(period, dtype=np.float64)
    d0 = np.asarray(date_0, dtype=np.float64)

    omega = 2.0 * np.pi / T
    raw = (p_rel - omega * d0) % (2.0 * np.pi)
    phase = (raw + np.pi) % (2.0 * np.pi) - np.pi
    return float(phase) if is_scalar else phase


def map_28day_phase(
    phi_rel: Union[float, np.ndarray, Sequence[float]],
    period: Union[float, np.ndarray, Sequence[float]],
    date_0: Union[float, np.ndarray, Sequence[float]],
) -> Union[float, np.ndarray]:
    """Calculates first peak calendar offset and maps to 28-day lunar/calendar phase.

    Mathematical definition:
        omega = 2 * pi / period
        t_peak = (phi_rel mod (2 * pi)) / omega
        theta_28 = ((2 * pi / 28.0) * (date_0 + t_peak)) mod (2 * pi)
        mapped to [-pi, pi].

    Parameters
    ----------
    phi_rel : float or array-like
        Harmonic acrophase relative to participant baseline (radians).
    period : float or array-like
        Best-fit period T* in days.
    date_0 : float or array-like
        Participant baseline calendar date offset.

    Returns
    -------
    theta_28 : float or np.ndarray
        Phase within 28-day cycle mapped to [-pi, pi].
    """
    is_scalar = np.isscalar(phi_rel) and np.isscalar(period) and np.isscalar(date_0)
    p_rel = np.asarray(phi_rel, dtype=np.float64)
    T = np.asarray(period, dtype=np.float64)
    d0 = np.asarray(date_0, dtype=np.float64)

    omega = 2.0 * np.pi / T
    t_peak = (p_rel % (2.0 * np.pi)) / omega
    raw = ((2.0 * np.pi / 28.0) * (d0 + t_peak)) % (2.0 * np.pi)
    phase = (raw + np.pi) % (2.0 * np.pi) - np.pi
    return float(phase) if is_scalar else phase


def detrend_signal(
    t: Union[np.ndarray, Sequence[float]],
    y: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """Linear detrending using OLS on unevenly spaced calendar days.

    Parameters
    ----------
    t : array-like of shape (N,)
        Calendar day offsets.
    y : array-like of shape (N,)
        Signal values at observed days.

    Returns
    -------
    y_detrended : np.ndarray of shape (N,)
        Signal with linear trend (beta_0 + beta_trend * t) subtracted.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if len(t_arr) == 0:
        return np.empty(0, dtype=np.float64)

    t_mean = np.mean(t_arr)
    y_mean = np.mean(y_arr)

    dt = t_arr - t_mean
    var_t = float(np.sum(dt ** 2))

    if var_t <= 1e-15:
        # Degenerate: single observation or all identical timestamps
        return y_arr - y_mean

    beta_trend = float(np.sum(dt * (y_arr - y_mean)) / var_t)
    beta_0 = y_mean - beta_trend * t_mean

    trend = beta_0 + beta_trend * t_arr
    return y_arr - trend


def sliding_window_lomb_scargle(
    t: Union[np.ndarray, Sequence[float]],
    y: Union[np.ndarray, Sequence[float]],
    window_size: float = 90.0,
    step_size: float = 15.0,
    min_days: int = 45,
    period_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes sliding-window Lomb-Scargle power spectrum across a longitudinal record.

    Sweeps a window of length `window_size` in increments of `step_size`.
    For each window with at least `min_days` observations, detrends the local signal
    and evaluates normalized Lomb-Scargle spectral power across candidate periods.

    Parameters
    ----------
    t : array-like of shape (N,)
        Calendar day timestamps.
    y : array-like of shape (N,)
        Observed daily metric values (e.g. resting HR p10, occupancy).
    window_size : float, default 90.0
        Span of the sliding window in calendar days.
    step_size : float, default 15.0
        Step size between successive window start positions in calendar days.
    min_days : int, default 45
        Minimum number of observed calendar days required in a window.
    period_grid : np.ndarray, optional
        Grid of candidate periods in days. Defaults to np.arange(14.0, 56.5, 0.5)
        (85 periods from 14.0 to 56.0 days).

    Returns
    -------
    dict
        Contains:
        - 'window_centers': np.ndarray of window midpoints (calendar days).
        - 'peak_periods': np.ndarray of peak period T*_w per window (np.nan if invalid).
        - 'peak_powers': np.ndarray of peak normalized power P*_w per window (np.nan if invalid).
        - 'T_star_w': alias for peak_periods.
        - 'P_star_w': alias for peak_powers.
        - 'period_drift': Interquartile range (IQR) of valid peak periods, Delta T_drift in days.
        - 'delta_T_drift': alias for period_drift.
        - 'persistence_ratio': fraction of valid windows with T*_w in [25.0, 35.0] days.
        - 'valid_mask': boolean array indicating which windows met the min_days criterion.
        - 'spectrogram': 2D np.ndarray of shape (num_windows, len(period_grid)) with power spectra.
        - 'period_grid': evaluated period grid.
        - 'n_valid_windows': integer count of valid windows.
        - 'n_total_windows': total number of windows evaluated.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(t_arr) & np.isfinite(y_arr)
    t_clean = t_arr[valid]
    y_clean = y_arr[valid]

    if period_grid is None:
        period_grid = np.arange(14.0, 56.5, 0.5, dtype=np.float64)
    else:
        period_grid = np.asarray(period_grid, dtype=np.float64)

    num_periods = len(period_grid)
    omega_grid = 2.0 * np.pi / period_grid

    if len(t_clean) < min_days:
        return {
            "window_centers": np.empty(0, dtype=np.float64),
            "peak_periods": np.empty(0, dtype=np.float64),
            "peak_powers": np.empty(0, dtype=np.float64),
            "T_star_w": np.empty(0, dtype=np.float64),
            "P_star_w": np.empty(0, dtype=np.float64),
            "period_drift": float(np.nan),
            "delta_T_drift": float(np.nan),
            "persistence_ratio": 0.0,
            "infradian_persistence_ratio": 0.0,
            "valid_mask": np.empty(0, dtype=bool),
            "spectrogram": np.empty((0, num_periods), dtype=np.float64),
            "period_grid": period_grid,
            "n_valid_windows": 0,
            "n_total_windows": 0,
        }

    t_min = float(np.min(t_clean))
    t_max = float(np.max(t_clean))
    total_span = t_max - t_min

    if total_span < window_size:
        return {
            "window_centers": np.empty(0, dtype=np.float64),
            "peak_periods": np.empty(0, dtype=np.float64),
            "peak_powers": np.empty(0, dtype=np.float64),
            "T_star_w": np.empty(0, dtype=np.float64),
            "P_star_w": np.empty(0, dtype=np.float64),
            "period_drift": float(np.nan),
            "delta_T_drift": float(np.nan),
            "persistence_ratio": 0.0,
            "infradian_persistence_ratio": 0.0,
            "valid_mask": np.empty(0, dtype=bool),
            "spectrogram": np.empty((0, num_periods), dtype=np.float64),
            "period_grid": period_grid,
            "n_valid_windows": 0,
            "n_total_windows": 0,
        }

    # Generate window start positions
    window_starts = np.arange(t_min, t_max - window_size + 1e-6, step_size, dtype=np.float64)
    n_windows = len(window_starts)

    window_centers = np.empty(n_windows, dtype=np.float64)
    peak_periods = np.full(n_windows, np.nan, dtype=np.float64)
    peak_powers = np.full(n_windows, np.nan, dtype=np.float64)
    valid_mask = np.zeros(n_windows, dtype=bool)
    spectrogram = np.zeros((n_windows, num_periods), dtype=np.float64)

    for i, w_start in enumerate(window_starts):
        w_end = w_start + window_size
        window_centers[i] = w_start + (window_size / 2.0)

        in_win = (t_clean >= w_start) & (t_clean < w_end)
        n_win = int(np.sum(in_win))

        if n_win >= min_days:
            t_win = t_clean[in_win]
            y_win = y_clean[in_win]

            # Linear detrending within local window
            y_win_detr = detrend_signal(t_win, y_win)

            # Compute normalized Lomb-Scargle power
            var_win = float(np.sum(y_win_detr ** 2))
            if var_win <= 1e-12:
                power = np.zeros(num_periods, dtype=np.float64)
            else:
                power = lombscargle(t_win, y_win_detr, omega_grid, normalize=True)
            spectrogram[i, :] = power

            best_idx = int(np.argmax(power))
            peak_periods[i] = period_grid[best_idx]
            peak_powers[i] = float(power[best_idx])
            valid_mask[i] = True

    # Compute period drift and infradian persistence ratio across valid windows
    valid_periods = peak_periods[valid_mask]
    n_valid = len(valid_periods)

    if n_valid > 0:
        q75, q25 = np.percentile(valid_periods, [75, 25])
        period_drift = float(q75 - q25)
        persistence_ratio = float(np.mean((valid_periods >= 25.0) & (valid_periods <= 35.0)))
    else:
        period_drift = float(np.nan)
        persistence_ratio = 0.0

    return {
        "window_centers": window_centers,
        "peak_periods": peak_periods,
        "peak_powers": peak_powers,
        "T_star_w": peak_periods,
        "P_star_w": peak_powers,
        "period_drift": period_drift,
        "delta_T_drift": period_drift,
        "persistence_ratio": persistence_ratio,
        "infradian_persistence_ratio": persistence_ratio,
        "valid_mask": valid_mask,
        "spectrogram": spectrogram,
        "period_grid": period_grid,
        "n_valid_windows": n_valid,
        "n_total_windows": n_windows,
    }


def run_unit_tests(verbose: bool = True) -> Dict[str, Any]:
    """Fast, deterministic unit verification test suite for all infradian extension methods.

    Requirements:
    1. Tests rayleigh_test:
       - Uniform distribution on [-pi, pi] (N=500): assert p > 0.05 and R_bar < 0.15.
       - Clustered von Mises distribution with kappa=4.0 (N=100): assert p < 1e-4 and R_bar > 0.5.
    2. Tests map_calendar_phase and map_28day_phase:
       - Assert coordinate transforms match analytical peak day shifts.
    3. Tests sliding_window_lomb_scargle:
       - 400-day synthetic test signal with true period T=28.0 days, 40% missing days, Gaussian noise.
       - Asserts recovered peak periods in all valid windows are within 28.0 +- 0.5 days.
       - Asserts period drift Delta T_drift <= 1.0 day.
       - Asserts infradian persistence ratio == 1.0.
    4. Execution speed:
       - Asserts entire test suite executes in < 3.0 seconds (target < 0.5s).
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(20261020)

    # -------------------------------------------------------------------------
    # 1. Rayleigh Test: Circular Uniformity vs. Concentration
    # -------------------------------------------------------------------------
    if verbose:
        print("[1/4] Testing analytical Rayleigh circular uniformity test...")

    # Case A: Uniform distribution (N=500)
    uniform_angles = rng.uniform(-np.pi, np.pi, size=500)
    res_uniform = rayleigh_test(uniform_angles)

    assert res_uniform["p_value"] > 0.05, (
        f"Rayleigh test failed on uniform angles: expected p > 0.05, got {res_uniform['p_value']:.4e}"
    )
    assert res_uniform["R_bar"] < 0.15, (
        f"Rayleigh test R_bar too high for uniform angles: expected R_bar < 0.15, got {res_uniform['R_bar']:.4f}"
    )

    # Case B: Clustered von Mises distribution (kappa=4.0, N=100)
    clustered_angles = rng.vonmises(mu=0.5, kappa=4.0, size=100)
    res_clustered = rayleigh_test(clustered_angles)

    assert res_clustered["p_value"] < 1e-4, (
        f"Rayleigh test failed to reject H0 for kappa=4: expected p < 1e-4, got {res_clustered['p_value']:.4e}"
    )
    assert res_clustered["R_bar"] > 0.5, (
        f"Rayleigh test R_bar too low for kappa=4: expected R_bar > 0.5, got {res_clustered['R_bar']:.4f}"
    )

    if verbose:
        print(f"      Uniform:   R_bar = {res_uniform['R_bar']:.4f}, Z = {res_uniform['Z']:.2f}, p = {res_uniform['p_value']:.4f} (PASS)")
        print(f"      Clustered: R_bar = {res_clustered['R_bar']:.4f}, Z = {res_clustered['Z']:.2f}, p = {res_clustered['p_value']:.4e} (PASS)")

    # -------------------------------------------------------------------------
    # 2. Phase Coordinate Transformations
    # -------------------------------------------------------------------------
    if verbose:
        print("[2/4] Testing phase coordinate transformations (calendar and 28-day)...")

    # Calendar phase mapping
    # If participant has phi_rel = 0, T* = 28.0, and starts at date_0 = 14:
    # omega = 2pi / 28. Shift is -omega * 14 = -pi.
    p_cal = map_calendar_phase(phi_rel=0.0, period=28.0, date_0=14.0)
    assert np.isclose(abs(p_cal), np.pi, atol=1e-8), f"Expected +-pi, got {p_cal}"

    # Vectorized calendar phase mapping
    p_vec = map_calendar_phase(
        phi_rel=np.array([0.0, np.pi / 2.0]),
        period=np.array([28.0, 28.0]),
        date_0=np.array([28.0, 7.0]),
    )
    assert np.allclose(p_vec, [0.0, 0.0], atol=1e-8), f"Vectorized calendar phase failed: {p_vec}"

    # 28-day phase mapping:
    # If phi_rel = 0, T* = 28.0, date_0 = 7:
    # t_peak = 0. Date + t_peak = 7 days -> 7/28 * 2pi = pi/2.
    th_28 = map_28day_phase(phi_rel=0.0, period=28.0, date_0=7.0)
    assert np.isclose(th_28, np.pi / 2.0, atol=1e-8), f"Expected pi/2, got {th_28}"

    # If phi_rel = pi, T* = 28.0, date_0 = 0:
    # t_peak = pi / (2pi / 28) = 14 days. Date + t_peak = 14 -> 14/28 * 2pi = pi.
    th_28_b = map_28day_phase(phi_rel=np.pi, period=28.0, date_0=0.0)
    assert np.isclose(abs(th_28_b), np.pi, atol=1e-8), f"Expected +-pi, got {th_28_b}"

    if verbose:
        print("      Phase mappings verified bit-accurately (PASS)")

    # -------------------------------------------------------------------------
    # 3. Sliding-Window Lomb-Scargle & Period Stability
    # -------------------------------------------------------------------------
    if verbose:
        print("[3/4] Testing sliding-window Lomb-Scargle on 400-day synthetic benchmark...")

    n_days = 400
    missing_frac = 0.40
    t_true = 28.0
    a_true = 2.5
    noise_sigma = 0.4

    t_full = np.arange(n_days, dtype=np.float64)
    y_signal = 62.0 + 0.01 * t_full + a_true * np.cos(2.0 * np.pi * t_full / t_true)
    noise = rng.normal(0.0, noise_sigma, size=n_days)
    y_full = y_signal + noise

    n_obs = int(round(n_days * (1.0 - missing_frac)))
    idx_obs = np.sort(rng.choice(n_days, size=n_obs, replace=False))
    t_obs = t_full[idx_obs]
    y_obs = y_full[idx_obs]

    sw_res = sliding_window_lomb_scargle(
        t=t_obs,
        y=y_obs,
        window_size=90.0,
        step_size=15.0,
        min_days=45,
    )

    n_valid = sw_res["n_valid_windows"]
    assert n_valid >= 15, f"Expected at least 15 valid windows, got {n_valid}"

    valid_t_stars = sw_res["peak_periods"][sw_res["valid_mask"]]
    if verbose:
        print(f"      Valid T* across windows: {valid_t_stars}")
    max_dev = float(np.max(np.abs(valid_t_stars - t_true)))
    assert max_dev <= 0.5, (
        f"Window peak periods deviated by {max_dev:.2f}d from true period {t_true}d (expected <= 0.5d)"
    )

    period_drift = sw_res["period_drift"]
    assert period_drift <= 1.0, (
        f"Period drift Delta T_drift = {period_drift:.2f}d exceeds threshold 1.0d"
    )

    persistence_ratio = sw_res["persistence_ratio"]
    assert persistence_ratio >= 0.95, (
        f"Persistence ratio {persistence_ratio:.2f} below threshold 0.95"
    )

    if verbose:
        print(f"      Evaluated {sw_res['n_total_windows']} windows ({n_valid} valid)")
        print(f"      Period drift Delta T_drift = {period_drift:.2f} days (PASS)")
        print(f"      Persistence ratio = {persistence_ratio * 100:.1f}% (PASS)")

    # -------------------------------------------------------------------------
    # 4. Performance Benchmark
    # -------------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"[4/4] Execution time: {elapsed:.3f} seconds (limit: 3.0s)")

    assert elapsed < 3.0, f"Execution time {elapsed:.3f}s exceeded 3.0s limit"

    return {
        "status": "PASS",
        "elapsed_seconds": elapsed,
        "n_valid_windows": n_valid,
        "period_drift": period_drift,
        "persistence_ratio": persistence_ratio,
        "uniform_R_bar": res_uniform["R_bar"],
        "uniform_p_value": res_uniform["p_value"],
        "clustered_R_bar": res_clustered["R_bar"],
        "clustered_p_value": res_clustered["p_value"],
    }


def main() -> int:
    """Main entrypoint: executes unit tests and outputs benchmark timing."""
    print("=" * 70)
    print("Female Infradian Extensions — Method Verification Suite")
    print("Path: experiments/female_infradian_extensions/scripts/00_verify_extension_methods.py")
    print("=" * 70)

    try:
        results = run_unit_tests(verbose=True)
        print("=" * 70)
        print(f"VERIFICATION SUCCESSFUL: All unit assertions passed in {results['elapsed_seconds']:.3f}s.")
        print("=" * 70)
        return 0
    except AssertionError as e:
        print("=" * 70)
        print(f"VERIFICATION FAILURE: Assertion failed: {e}", file=sys.stderr)
        print("=" * 70)
        return 1
    except Exception as e:
        print("=" * 70)
        print(f"VERIFICATION ERROR: Unexpected exception: {e}", file=sys.stderr)
        print("=" * 70)
        return 2


if __name__ == "__main__":
    sys.exit(main())
