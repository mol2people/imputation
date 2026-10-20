#!/usr/bin/env python3
"""
Verification Fixture for Female Infradian Rhythm Detection Methods
Path: experiments/female_infradian_cycles/scripts/00_verify_infradian_methods.py

Provides modular implementations of:
1. detrend_signal(t, y): Linear detrending using OLS on unevenly spaced calendar days.
2. fit_lomb_scargle(t, y, periods, flank_periods=None): Lomb-Scargle power over period grid,
   extracting peak period T*, peak power P*, amplitude, and SNR over flanking background bands.
3. fit_cosinor_grid(t, y, periods): Closed-form OLS normal equations for
   y(t) = beta_0 + beta_trend*t + beta_c*cos(2*pi*t/T) + beta_s*sin(2*pi*t/T),
   sweeping periods T in [25, 35] days, extracting T*, amplitude A, acrophase phi,
   Delta R^2 over null linear trend model, and analytical F-test p-value.
4. run_synthetic_benchmark(): Fast (< 5.0s, target < 0.1s) deterministic unit verification
   on 120-day benchmark with 40% missing days, asserting accurate period and amplitude recovery.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    import scipy.stats as stats
    from scipy.signal import lombscargle
    HAS_NUMPY_SCIPY = True
except ImportError:
    HAS_NUMPY_SCIPY = False
    import math
    import random

__all__ = [
    "InfradianResult",
    "detrend_signal",
    "fit_lomb_scargle",
    "fit_cosinor_grid",
    "generate_synthetic_benchmark",
    "run_synthetic_benchmark",
]


class InfradianResult(dict):
    """Dictionary subclass providing attribute-style and dict-style access for infradian metrics."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"InfradianResult has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __repr__(self) -> str:
        scalar_items = {
            k: (round(v, 4) if isinstance(v, (float, np.floating)) else v)
            for k, v in self.items()
            if not isinstance(v, np.ndarray)
        }
        return f"InfradianResult({scalar_items})"


def detrend_signal(
    t: Union[np.ndarray, Sequence[float]],
    y: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """Linear detrending using OLS on unevenly spaced calendar days.

    Parameters
    ----------
    t : array-like of shape (N,)
        Calendar day offsets (e.g. days since baseline or Unix epoch).
    y : array-like of shape (N,)
        Signal values at observed days.

    Returns
    -------
    y_detrended : np.ndarray of shape (N,)
        Signal with linear trend (beta_0 + beta_trend * t) subtracted.
    """
    if not HAS_NUMPY_SCIPY:
        valid_pairs = [(ti, yi) for ti, yi in zip(t, y) if math.isfinite(ti) and math.isfinite(yi)]
        if not valid_pairs:
            return []
        t_arr = [p[0] for p in valid_pairs]
        y_arr = [p[1] for p in valid_pairs]
        N = len(t_arr)
        t_mean = sum(t_arr) / N
        y_mean = sum(y_arr) / N
        var_t = sum((ti - t_mean) ** 2 for ti in t_arr)
        if var_t <= 1e-15:
            return [yi - y_mean for yi in y]
        beta_trend = sum((ti - t_mean) * (yi - y_mean) for ti, yi in zip(t_arr, y_arr)) / var_t
        beta_0 = y_mean - beta_trend * t_mean
        return [yi - (beta_0 + beta_trend * ti) for ti, yi in zip(t, y)]

    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if len(t_arr) == 0:
        return np.empty(0, dtype=np.float64)

    t_mean = np.mean(t_arr)
    y_mean = np.mean(y_arr)

    dt = t_arr - t_mean
    var_t = np.sum(dt ** 2)

    if var_t <= 1e-15:
        # Degenerate: single observation or all identical timestamps
        return y_arr - y_mean

    beta_trend = np.sum(dt * (y_arr - y_mean)) / var_t
    beta_0 = y_mean - beta_trend * t_mean

    trend = beta_0 + beta_trend * t_arr
    return y_arr - trend


def fit_lomb_scargle(
    t: Union[np.ndarray, Sequence[float]],
    y: Union[np.ndarray, Sequence[float]],
    periods: Union[np.ndarray, Sequence[float]],
    flank_periods: Optional[Union[np.ndarray, Sequence[float]]] = None,
    flank_bands: Optional[Union[np.ndarray, Sequence[float]]] = None,
) -> InfradianResult:
    """Computes Lomb-Scargle power over period grid using scipy.signal.lombscargle.

    Parameters
    ----------
    t : array-like of shape (N,)
        Calendar days.
    y : array-like of shape (N,)
        Signal values (raw or detrended).
    periods : array-like of shape (K,)
        Candidate periods in days (e.g. np.arange(25.0, 35.5, 0.5)).
    flank_periods : array-like, optional
        Flanking background periods in days for red-noise floor and SNR estimation.
        If None and flank_bands is None, defaults to [14.0, 22.0] U [38.0, 56.0] days.
    flank_bands : array-like, optional
        Alias for flank_periods.

    Returns
    -------
    result : InfradianResult (dict-like)
        Contains:
        - peak_period (T*): period with maximum Lomb-Scargle power in candidate grid.
        - peak_power (P*): maximum normalized power in [0, 1].
        - snr: peak power divided by median power in flanking background bands.
        - amplitude: estimated sinusoidal amplitude at peak period.
        - power_grid: full normalized power array over candidate periods.
        - amp_grid: full amplitude array over candidate periods.
        - periods: candidate period grid.
        - noise_floor: median power in flanking background bands.
        - flank_periods: evaluated flanking periods.
    """
    if not HAS_NUMPY_SCIPY:
        valid_pairs = [(ti, yi) for ti, yi in zip(t, y) if math.isfinite(ti) and math.isfinite(yi)]
        periods_list = list(periods)
        if len(valid_pairs) < 4:
            return InfradianResult(
                peak_period=float(periods_list[0]) if periods_list else 28.0,
                peak_power=0.0,
                snr=0.0,
                amplitude=0.0,
                power_grid=[0.0] * len(periods_list),
                amp_grid=[0.0] * len(periods_list),
                periods=periods_list,
                noise_floor=0.0,
                flank_periods=[],
                T_star=float(periods_list[0]) if periods_list else 28.0,
                P_star=0.0,
                A_star=0.0,
                SNR=0.0,
            )
        t_clean = [p[0] for p in valid_pairs]
        y_clean = [p[1] for p in valid_pairs]
        N = len(t_clean)
        y_detr = detrend_signal(t_clean, y_clean)
        yy = sum(yi ** 2 for yi in y_detr)
        if yy <= 1e-15:
            return InfradianResult(
                peak_period=float(periods_list[0]) if periods_list else 28.0,
                peak_power=0.0,
                snr=0.0,
                amplitude=0.0,
                power_grid=[0.0] * len(periods_list),
                amp_grid=[0.0] * len(periods_list),
                periods=periods_list,
                noise_floor=0.0,
                flank_periods=[],
                T_star=float(periods_list[0]) if periods_list else 28.0,
                P_star=0.0,
                A_star=0.0,
                SNR=0.0,
            )

        def _calc_ls_power(T_val: float) -> Tuple[float, float]:
            omega = 2.0 * math.pi / T_val
            s2 = sum(math.sin(2.0 * omega * ti) for ti in t_clean)
            c2 = sum(math.cos(2.0 * omega * ti) for ti in t_clean)
            tau = math.atan2(s2, c2) / (2.0 * omega)
            c = [math.cos(omega * (ti - tau)) for ti in t_clean]
            s = [math.sin(omega * (ti - tau)) for ti in t_clean]
            cc = sum(ci ** 2 for ci in c)
            ss = sum(si ** 2 for si in s)
            yc = sum(yi * ci for yi, ci in zip(y_detr, c))
            ys = sum(yi * si for yi, si in zip(y_detr, s))
            p = 0.5 * ((yc ** 2 / cc if cc > 1e-15 else 0.0) + (ys ** 2 / ss if ss > 1e-15 else 0.0))
            p_norm = 2.0 * p / yy
            amp = math.sqrt(max(0.0, 4.0 * p / N))
            return p_norm, amp

        power_grid = []
        amp_grid = []
        for T_cand in periods_list:
            p_n, a_v = _calc_ls_power(T_cand)
            power_grid.append(p_n)
            amp_grid.append(a_v)

        best_idx = max(range(len(periods_list)), key=lambda i: power_grid[i])
        peak_period = float(periods_list[best_idx])
        peak_power = float(power_grid[best_idx])
        amplitude = float(amp_grid[best_idx])

        actual_flanks = flank_periods if flank_periods is not None else flank_bands
        if actual_flanks is None:
            p_min = min(periods_list)
            p_max = max(periods_list)
            low_band = [max(10.0, p_min - 11.0) + k for k in range(9)]
            high_band = [p_max + 3.0 + k for k in range(19)]
            actual_flanks = low_band + high_band
        else:
            actual_flanks = list(actual_flanks)

        flank_powers = sorted([_calc_ls_power(T_f)[0] for T_f in actual_flanks])
        n_fl = len(flank_powers)
        noise_floor = float(flank_powers[n_fl // 2]) if n_fl > 0 else 0.0
        snr = float(peak_power / max(noise_floor, 1e-6))

        return InfradianResult(
            peak_period=peak_period,
            peak_power=peak_power,
            snr=snr,
            amplitude=amplitude,
            power_grid=power_grid,
            amp_grid=amp_grid,
            periods=periods_list,
            noise_floor=noise_floor,
            flank_periods=actual_flanks,
            T_star=peak_period,
            P_star=peak_power,
            A_star=amplitude,
            SNR=snr,
        )

    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    periods_arr = np.asarray(periods, dtype=np.float64)

    # Filter non-finite observations defensively
    valid = np.isfinite(t_arr) & np.isfinite(y_arr)
    t_clean = t_arr[valid]
    y_clean = y_arr[valid]

    if len(t_clean) < 4:
        empty_res = InfradianResult(
            peak_period=float(periods_arr[0]) if len(periods_arr) > 0 else 28.0,
            peak_power=0.0,
            snr=0.0,
            amplitude=0.0,
            power_grid=np.zeros_like(periods_arr),
            amp_grid=np.zeros_like(periods_arr),
            periods=periods_arr,
            noise_floor=0.0,
            flank_periods=np.empty(0),
            T_star=float(periods_arr[0]) if len(periods_arr) > 0 else 28.0,
            P_star=0.0,
            A_star=0.0,
            SNR=0.0,
        )
        return empty_res

    # Calendar-day linear detrending prior to Lomb-Scargle
    y_detr = detrend_signal(t_clean, y_clean)

    # Convert candidate periods T to angular frequencies omega = 2*pi / T
    omega_grid = 2.0 * np.pi / periods_arr

    # Compute normalized power (fraction of variance in [0, 1]) and physical amplitude
    power_grid = lombscargle(t_clean, y_detr, omega_grid, normalize=True)
    amp_grid = np.abs(lombscargle(t_clean, y_detr, omega_grid, normalize="amplitude"))

    best_idx = int(np.argmax(power_grid))
    peak_period = float(periods_arr[best_idx])
    peak_power = float(power_grid[best_idx])
    amplitude = float(amp_grid[best_idx])

    # Flanking background bands for red-noise floor and SNR
    actual_flanks = flank_periods if flank_periods is not None else flank_bands
    if actual_flanks is None:
        p_min = float(np.min(periods_arr))
        p_max = float(np.max(periods_arr))
        # Default flanking bands: [14, 22] days and [38, 56] days when grid is [25, 35]
        low_band = np.arange(max(10.0, p_min - 11.0), p_min - 2.5, 1.0)
        high_band = np.arange(p_max + 3.0, p_max + 21.5, 1.0)
        actual_flanks = np.concatenate([low_band, high_band])
    else:
        actual_flanks = np.asarray(actual_flanks, dtype=np.float64)

    omega_flank = 2.0 * np.pi / actual_flanks
    flank_powers = lombscargle(t_clean, y_detr, omega_flank, normalize=True)
    noise_floor = float(np.median(flank_powers))
    snr = float(peak_power / max(noise_floor, 1e-6))

    result = InfradianResult(
        peak_period=peak_period,
        peak_power=peak_power,
        snr=snr,
        amplitude=amplitude,
        power_grid=power_grid,
        amp_grid=amp_grid,
        periods=periods_arr,
        noise_floor=noise_floor,
        flank_periods=actual_flanks,
        T_star=peak_period,
        P_star=peak_power,
        A_star=amplitude,
        SNR=snr,
    )
    return result


def fit_cosinor_grid(
    t: Union[np.ndarray, Sequence[float]],
    y: Union[np.ndarray, Sequence[float]],
    periods: Union[np.ndarray, Sequence[float]],
) -> InfradianResult:
    """Formulates closed-form OLS normal equations for harmonic cosinor regression.

    Model:
        y(t) = beta_0 + beta_trend * t + beta_c * cos(2*pi*t / T) + beta_s * sin(2*pi*t / T)

    Sweeps candidate periods T in closed form, extracting peak period T*,
    amplitude A = sqrt(beta_c^2 + beta_s^2), acrophase phi = atan2(beta_s, beta_c),
    incremental variance explained Delta R^2 over the linear trend null model,
    and the exact analytical extra-sum-of-squares F-test p-value.

    Parameters
    ----------
    t : array-like of shape (N,)
        Calendar days.
    y : array-like of shape (N,)
        Signal values.
    periods : array-like of shape (K,)
        Candidate period grid in days (e.g. np.arange(25.0, 35.5, 0.5)).

    Returns
    -------
    result : InfradianResult (dict-like)
        Contains:
        - peak_period (T*): period maximizing Delta R^2 in candidate grid.
        - amplitude (A*): physical amplitude in native signal units.
        - acrophase (phi*): phase angle in radians in (-pi, pi].
        - delta_r2: incremental variance explained over null linear trend model.
        - r2_full: total coefficient of determination R^2 of full harmonic model.
        - r2_null: R^2 of null linear trend model.
        - f_stat: extra-sum-of-squares F-statistic.
        - p_value: analytical p-value from F(2, N - 4) distribution.
        - beta: OLS parameter vector [beta_0, beta_trend, beta_c, beta_s] at T*.
        - periods: evaluated period grid.
        - delta_r2_grid: Delta R^2 across all evaluated periods.
        - amplitudes_grid: amplitudes across all evaluated periods.
    """
    if not HAS_NUMPY_SCIPY:
        valid_pairs = [(ti, yi) for ti, yi in zip(t, y) if math.isfinite(ti) and math.isfinite(yi)]
        periods_list = list(periods)
        N = len(valid_pairs)
        K = len(periods_list)
        if N < 5 or K == 0:
            return InfradianResult(
                peak_period=float(periods_list[0]) if K > 0 else 28.0,
                amplitude=0.0,
                acrophase=0.0,
                delta_r2=0.0,
                r2_full=0.0,
                r2_null=0.0,
                f_stat=0.0,
                p_value=1.0,
                beta=[0.0, 0.0, 0.0, 0.0],
                periods=periods_list,
                delta_r2_grid=[0.0] * K,
                amplitudes_grid=[0.0] * K,
                T_star=float(periods_list[0]) if K > 0 else 28.0,
                A_star=0.0,
                phi_star=0.0,
                dR2=0.0,
                f_statistic=0.0,
            )
        t_clean = [p[0] for p in valid_pairs]
        y_clean = [p[1] for p in valid_pairs]
        t_mean = sum(t_clean) / N
        y_mean = sum(y_clean) / N
        var_t = sum((ti - t_mean) ** 2 for ti in t_clean)
        beta_trend = sum((ti - t_mean) * (yi - y_mean) for ti, yi in zip(t_clean, y_clean)) / var_t if var_t > 1e-15 else 0.0
        beta_0 = y_mean - beta_trend * t_mean
        SS_tot = sum((yi - y_mean) ** 2 for yi in y_clean)
        SS_res_null = sum((yi - (beta_0 + beta_trend * ti)) ** 2 for ti, yi in zip(t_clean, y_clean))
        R2_null = 1.0 - (SS_res_null / SS_tot) if SS_tot > 1e-15 else 0.0

        def _solve_4x4(A, b):
            M = [A[i][:] + [b[i]] for i in range(4)]
            for i in range(4):
                max_r = max(range(i, 4), key=lambda r: abs(M[r][i]))
                M[i], M[max_r] = M[max_r], M[i]
                p = M[i][i]
                if abs(p) < 1e-12:
                    return None
                for j in range(i, 5):
                    M[i][j] /= p
                for r in range(4):
                    if r != i:
                        f = M[r][i]
                        for j in range(i, 5):
                            M[r][j] -= f * M[i][j]
            return [M[i][4] for i in range(4)]

        delta_r2_grid = []
        amplitudes_grid = []
        results_cand = []
        for T in periods_list:
            omega = 2.0 * math.pi / T
            cos_t = [math.cos(omega * ti) for ti in t_clean]
            sin_t = [math.sin(omega * ti) for ti in t_clean]
            cols = [[1.0] * N, t_clean, cos_t, sin_t]
            XtX = [[sum(cols[i][k] * cols[j][k] for k in range(N)) for j in range(4)] for i in range(4)]
            Xty = [sum(cols[i][k] * y_clean[k] for k in range(N)) for i in range(4)]
            beta_vec = _solve_4x4(XtX, Xty)
            if beta_vec is None:
                delta_r2_grid.append(0.0)
                amplitudes_grid.append(0.0)
                results_cand.append((T, 0.0, 0.0, 0.0, 0.0, SS_res_null, [0.0]*4, 0.0, 1.0))
                continue
            y_pred = [beta_vec[0] + beta_vec[1] * t_clean[k] + beta_vec[2] * cos_t[k] + beta_vec[3] * sin_t[k] for k in range(N)]
            ss_full = sum((y_clean[k] - y_pred[k]) ** 2 for k in range(N))
            ss_full = max(0.0, min(ss_full, SS_res_null))
            dr2 = (SS_res_null - ss_full) / SS_tot if SS_tot > 1e-15 else 0.0
            r2_full = 1.0 - (ss_full / SS_tot) if SS_tot > 1e-15 else 0.0
            amp = math.sqrt(beta_vec[2] ** 2 + beta_vec[3] ** 2)
            phi = math.atan2(beta_vec[3], beta_vec[2])
            df2 = N - 4
            if df2 > 0 and ss_full > 1e-15:
                f_stat = ((SS_res_null - ss_full) / 2.0) / (ss_full / df2)
                f_stat = max(0.0, f_stat)
                p_val = (1.0 + (2.0 / df2) * f_stat) ** (-df2 / 2.0)
            elif ss_full <= 1e-15 and (SS_res_null - ss_full) > 1e-15:
                f_stat = float("inf")
                p_val = 0.0
            else:
                f_stat = 0.0
                p_val = 1.0
            delta_r2_grid.append(dr2)
            amplitudes_grid.append(amp)
            results_cand.append((T, amp, phi, dr2, r2_full, ss_full, beta_vec, f_stat, p_val))

        best_idx = max(range(len(periods_list)), key=lambda i: delta_r2_grid[i])
        best_T, best_amp, best_phi, best_dr2, best_r2_full, best_ss, best_beta, best_f, best_p = results_cand[best_idx]
        return InfradianResult(
            peak_period=best_T,
            amplitude=best_amp,
            acrophase=best_phi,
            delta_r2=best_dr2,
            r2_full=best_r2_full,
            r2_null=R2_null,
            f_stat=best_f,
            p_value=best_p,
            beta=best_beta,
            periods=periods_list,
            delta_r2_grid=delta_r2_grid,
            amplitudes_grid=amplitudes_grid,
            T_star=best_T,
            A_star=best_amp,
            phi_star=best_phi,
            dR2=best_dr2,
            f_statistic=best_f,
        )

    t_arr = np.asarray(t, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    periods_arr = np.asarray(periods, dtype=np.float64)

    # Defensive filtering for finite observations
    valid = np.isfinite(t_arr) & np.isfinite(y_arr)
    t_clean = t_arr[valid]
    y_clean = y_arr[valid]

    N = len(t_clean)
    K = len(periods_arr)

    if N < 5 or K == 0:
        empty_res = InfradianResult(
            peak_period=float(periods_arr[0]) if K > 0 else 28.0,
            amplitude=0.0,
            acrophase=0.0,
            delta_r2=0.0,
            r2_full=0.0,
            r2_null=0.0,
            f_stat=0.0,
            p_value=1.0,
            beta=np.zeros(4, dtype=np.float64),
            periods=periods_arr,
            delta_r2_grid=np.zeros(K, dtype=np.float64),
            amplitudes_grid=np.zeros(K, dtype=np.float64),
            T_star=float(periods_arr[0]) if K > 0 else 28.0,
            A_star=0.0,
            phi_star=0.0,
            dR2=0.0,
            f_statistic=0.0,
        )
        return empty_res

    # -------------------------------------------------------------
    # 1. Null Model (Linear Trend Only: y = beta_0 + beta_trend * t)
    # -------------------------------------------------------------
    X_null = np.column_stack([np.ones(N, dtype=np.float64), t_clean])
    beta_null, _, _, _ = np.linalg.lstsq(X_null, y_clean, rcond=None)
    y_hat_null = X_null @ beta_null

    y_bar = np.mean(y_clean)
    SS_tot = float(np.sum((y_clean - y_bar) ** 2))
    SS_res_null = float(np.sum((y_clean - y_hat_null) ** 2))
    R2_null = float(1.0 - (SS_res_null / SS_tot)) if SS_tot > 1e-15 else 0.0

    # -------------------------------------------------------------
    # 2. Vectorized 3D OLS Solve across all K candidate periods
    # -------------------------------------------------------------
    # arg: shape (K, N)
    arg = 2.0 * np.pi * t_clean[None, :] / periods_arr[:, None]
    cos_part = np.cos(arg)
    sin_part = np.sin(arg)
    ones_part = np.ones((K, N), dtype=np.float64)
    t_part = np.broadcast_to(t_clean[None, :], (K, N))

    # Design tensor X: shape (K, N, 4)
    X = np.stack([ones_part, t_part, cos_part, sin_part], axis=-1)

    # Gram matrices XtX: shape (K, 4, 4), and RHS Xty: shape (K, 4)
    Xt = X.transpose(0, 2, 1)  # shape (K, 4, N)
    XtX = np.matmul(Xt, X)     # shape (K, 4, 4)
    Xty = np.matmul(Xt, y_clean)  # shape (K, 4)

    # Batched solve of (XtX) @ beta = Xty
    beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]  # shape (K, 4)

    # Residual sum of squares: SS_res = y^T y - beta^T (X^T y)
    yTy = float(np.dot(y_clean, y_clean))
    SS_res_full = yTy - np.sum(beta * Xty, axis=1)  # shape (K,)
    SS_res_full = np.maximum(SS_res_full, 0.0)
    SS_res_full = np.minimum(SS_res_full, SS_res_null)  # full nests null

    if SS_tot > 1e-15:
        R2_full = 1.0 - (SS_res_full / SS_tot)
        delta_R2 = (SS_res_null - SS_res_full) / SS_tot
        delta_R2 = np.maximum(delta_R2, 0.0)
    else:
        R2_full = np.zeros(K, dtype=np.float64)
        delta_R2 = np.zeros(K, dtype=np.float64)

    beta_c = beta[:, 2]
    beta_s = beta[:, 3]
    amplitudes = np.sqrt(beta_c ** 2 + beta_s ** 2)
    acrophases = np.arctan2(beta_s, beta_c)

    best_idx = int(np.argmax(delta_R2))
    peak_period = float(periods_arr[best_idx])
    amplitude = float(amplitudes[best_idx])
    acrophase = float(acrophases[best_idx])
    delta_r2_val = float(delta_R2[best_idx])
    r2_full_val = float(R2_full[best_idx])
    ss_res_best = float(SS_res_full[best_idx])
    beta_best = beta[best_idx]

    # -------------------------------------------------------------
    # 3. Closed-Form Extra-Sum-of-Squares F-Test
    # -------------------------------------------------------------
    # F = ((SS_null - SS_full) / 2) / (SS_full / (N - 4))
    df1 = 2
    df2 = N - 4
    if df2 > 0 and ss_res_best > 1e-15:
        f_stat = float(((SS_res_null - ss_res_best) / df1) / (ss_res_best / df2))
        f_stat = max(f_stat, 0.0)
        p_value = float(stats.f.sf(f_stat, df1, df2))
    elif ss_res_best <= 1e-15 and (SS_res_null - ss_res_best) > 1e-15:
        f_stat = float("inf")
        p_value = 0.0
    else:
        f_stat = 0.0
        p_value = 1.0

    result = InfradianResult(
        peak_period=peak_period,
        amplitude=amplitude,
        acrophase=acrophase,
        delta_r2=delta_r2_val,
        r2_full=r2_full_val,
        r2_null=R2_null,
        f_stat=f_stat,
        p_value=p_value,
        beta=beta_best,
        periods=periods_arr,
        delta_r2_grid=delta_R2,
        amplitudes_grid=amplitudes,
        T_star=peak_period,
        A_star=amplitude,
        phi_star=acrophase,
        dR2=delta_r2_val,
        f_statistic=f_stat,
    )
    return result


def generate_synthetic_benchmark(
    n_days: int = 120,
    missing_frac: float = 0.40,
    t_true: float = 28.0,
    a_true: float = 2.5,
    mean_val: float = 62.0,
    trend_val: float = 0.01,
    noise_sigma: float = 0.8,
    phi_true: float = 0.0,
    seed: int = 20261019,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic longitudinal daily benchmark with missing days.

    Parameters
    ----------
    n_days : int
        Total calendar day span (default 120 days).
    missing_frac : float
        Fraction of calendar days randomly unobserved (default 0.40).
    t_true : float
        True period in days (default 28.0 days).
    a_true : float
        True amplitude in native units (default 2.5 bpm).
    mean_val : float
        Baseline signal mean (default 62.0 bpm).
    trend_val : float
        Linear drift slope per day (default 0.01 bpm/day).
    noise_sigma : float
        Gaussian noise standard deviation (default 0.8 bpm).
    phi_true : float
        True phase angle in radians (default 0.0).
    seed : int
        RNG seed for determinism.

    Returns
    -------
    t_obs : np.ndarray
        Calendar days observed (shape ~ n_days * (1 - missing_frac)).
    y_obs : np.ndarray
        Observed signal values at t_obs.
    t_full : np.ndarray
        Full grid of calendar days [0, ..., n_days - 1].
    y_full : np.ndarray
        Full signal before missingness drop.
    """
    if not HAS_NUMPY_SCIPY:
        rng = random.Random(seed)
        t_full = [float(i) for i in range(n_days)]
        y_signal = [mean_val + trend_val * ti + a_true * math.cos(2.0 * math.pi * ti / t_true - phi_true) for ti in t_full]
        y_full = [ys + rng.gauss(0.0, noise_sigma) for ys in y_signal]
        n_obs = int(round(n_days * (1.0 - missing_frac)))
        idx_obs = sorted(rng.sample(range(n_days), n_obs))
        t_obs = [t_full[i] for i in idx_obs]
        y_obs = [y_full[i] for i in idx_obs]
        return t_obs, y_obs, t_full, y_full

    rng = np.random.default_rng(seed)
    t_full = np.arange(n_days, dtype=np.float64)

    # True periodic signal with baseline, trend, and noise
    y_signal = mean_val + trend_val * t_full + a_true * np.cos(2.0 * np.pi * t_full / t_true - phi_true)
    noise = rng.normal(0.0, noise_sigma, size=n_days)
    y_full = y_signal + noise

    # Randomly drop 40% of calendar days
    n_obs = int(round(n_days * (1.0 - missing_frac)))
    idx_obs = np.sort(rng.choice(n_days, size=n_obs, replace=False))
    t_obs = t_full[idx_obs]
    y_obs = y_full[idx_obs]

    return t_obs, y_obs, t_full, y_full


def run_synthetic_benchmark(
    seed: int = 20261019,
    phi_true: float = 0.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Executes the unit verification benchmark and asserts recovery accuracy.

    Requirements from dispatch:
    - 120-day span with 40% random missing days (~72 observed days).
    - True period T_true = 28.0 days, true amplitude A_true = 2.5 bpm,
      mean = 62.0 bpm, linear trend = 0.01 bpm/day, Gaussian noise sigma = 0.8 bpm.
    - Assert Lomb-Scargle recovers peak period in 28.0 +- 0.5 days.
    - Assert Cosinor recovers peak period in 28.0 +- 0.5 days.
    - Assert Cosinor recovers amplitude within +- 20% of true amplitude.
    - Assert execution time is well under 5.0 seconds (target < 0.1s).
    """
    t_start = time.perf_counter()

    # Ground truth parameters
    n_days = 120
    missing_frac = 0.40
    t_true = 28.0
    a_true = 2.5
    mean_val = 62.0
    trend_val = 0.01
    noise_sigma = 0.8

    # Generate benchmark signal
    t_obs, y_obs, _, _ = generate_synthetic_benchmark(
        n_days=n_days,
        missing_frac=missing_frac,
        t_true=t_true,
        a_true=a_true,
        mean_val=mean_val,
        trend_val=trend_val,
        noise_sigma=noise_sigma,
        phi_true=phi_true,
        seed=seed,
    )

    n_obs = len(t_obs)
    # Target period grid: 25.0 to 35.0 days in 0.5d steps (21 points)
    if not HAS_NUMPY_SCIPY:
        periods = [25.0 + 0.5 * k for k in range(21)]
    else:
        periods = np.arange(25.0, 35.5, 0.5)

    # -------------------------------------------------------------
    # 1. Lomb-Scargle Verification
    # -------------------------------------------------------------
    t_ls_start = time.perf_counter()
    res_ls = fit_lomb_scargle(t_obs, y_obs, periods)
    t_ls_elapsed = time.perf_counter() - t_ls_start

    # Assertions for Lomb-Scargle
    err_period_ls = abs(res_ls.peak_period - t_true)
    assert err_period_ls <= 0.5, (
        f"Lomb-Scargle peak period recovery failed: got {res_ls.peak_period}d, "
        f"expected {t_true} +- 0.5d (error {err_period_ls:.2f}d)"
    )
    assert res_ls.snr > 3.0, (
        f"Lomb-Scargle SNR failed: got {res_ls.snr:.2f}, expected > 3.0"
    )

    # -------------------------------------------------------------
    # 2. Harmonic Cosinor Verification
    # -------------------------------------------------------------
    t_cos_start = time.perf_counter()
    res_cos = fit_cosinor_grid(t_obs, y_obs, periods)
    t_cos_elapsed = time.perf_counter() - t_cos_start

    # Assertions for Cosinor
    err_period_cos = abs(res_cos.peak_period - t_true)
    assert err_period_cos <= 0.5, (
        f"Cosinor peak period recovery failed: got {res_cos.peak_period}d, "
        f"expected {t_true} +- 0.5d (error {err_period_cos:.2f}d)"
    )

    # Amplitude assertion: within +- 20% of true amplitude
    amp_rel_err = abs(res_cos.amplitude - a_true) / a_true
    assert amp_rel_err <= 0.20, (
        f"Cosinor amplitude recovery failed: got {res_cos.amplitude:.3f} bpm, "
        f"expected {a_true} +- 20% ([{a_true*0.8:.2f}, {a_true*1.2:.2f}]), "
        f"relative error {amp_rel_err*100:.1f}%"
    )

    # Delta R^2 and analytical p-value assertions
    assert res_cos.delta_r2 > 0.30, (
        f"Cosinor Delta R^2 unexpectedly low: got {res_cos.delta_r2:.4f}, expected > 0.30"
    )
    assert res_cos.p_value < 1e-4, (
        f"Cosinor analytical p-value failed: got {res_cos.p_value:.4e}, expected < 1e-4"
    )

    # -------------------------------------------------------------
    # 3. Null Control Verification (A = 0.0)
    # -------------------------------------------------------------
    if not HAS_NUMPY_SCIPY:
        rng_null = random.Random(seed + 999)
        y_null = [mean_val + trend_val * ti + rng_null.gauss(0.0, noise_sigma) for ti in t_obs]
    else:
        rng_null = np.random.default_rng(seed + 999)
        y_null = mean_val + trend_val * t_obs + rng_null.normal(0.0, noise_sigma, size=n_obs)
    res_cos_null = fit_cosinor_grid(t_obs, y_null, periods)

    assert res_cos_null.delta_r2 < 0.15, (
        f"Cosinor null model produced excessive Delta R^2: {res_cos_null.delta_r2:.4f}, expected < 0.15"
    )

    # -------------------------------------------------------------
    # 4. Runtime Assertion (< 5.0 seconds, target < 0.1s)
    # -------------------------------------------------------------
    t_total = time.perf_counter() - t_start
    assert t_total < 5.0, f"Benchmark execution exceeded 5.0s: {t_total:.4f}s"

    metrics = {
        "n_days": n_days,
        "n_obs": n_obs,
        "t_true": t_true,
        "a_true": a_true,
        "phi_true": phi_true,
        "ls_peak_period": res_ls.peak_period,
        "ls_peak_power": res_ls.peak_power,
        "ls_snr": res_ls.snr,
        "ls_amplitude": res_ls.amplitude,
        "cos_peak_period": res_cos.peak_period,
        "cos_amplitude": res_cos.amplitude,
        "cos_acrophase": res_cos.acrophase,
        "cos_delta_r2": res_cos.delta_r2,
        "cos_f_stat": res_cos.f_stat,
        "cos_p_value": res_cos.p_value,
        "null_cos_max_delta_r2": res_cos_null.delta_r2,
        "time_ls_ms": t_ls_elapsed * 1000.0,
        "time_cos_ms": t_cos_elapsed * 1000.0,
        "time_total_ms": t_total * 1000.0,
    }

    if verbose:
        print("=" * 70)
        print("INFRADIAN RHYTHM VERIFICATION BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Signal span:           {n_days} calendar days ({missing_frac*100:.0f}% missing, {n_obs} observed)")
        print(f"Ground truth:          T = {t_true:.1f} days, A = {a_true:.2f} bpm, phi = {phi_true:.3f} rad, trend = {trend_val} bpm/day")
        print("-" * 70)
        print(f"Lomb-Scargle Recovery: T* = {res_ls.peak_period:.1f} days (err = {err_period_ls:.2f}d)")
        print(f"                       Peak Power = {res_ls.peak_power:.4f}, SNR = {res_ls.snr:.2f}")
        print(f"                       Est. Amplitude = {res_ls.amplitude:.3f} bpm")
        print(f"                       Compute Time = {t_ls_elapsed*1000.0:.3f} ms")
        print("-" * 70)
        print(f"Cosinor OLS Recovery:  T* = {res_cos.peak_period:.1f} days (err = {err_period_cos:.2f}d)")
        print(f"                       A* = {res_cos.amplitude:.3f} bpm (rel err = {amp_rel_err*100:.1f}%)")
        print(f"                       phi* = {res_cos.acrophase:.3f} rad")
        print(f"                       Delta R^2 = {res_cos.delta_r2:.4f}, F = {res_cos.f_stat:.2f}, p = {res_cos.p_value:.2e}")
        print(f"                       Compute Time = {t_cos_elapsed*1000.0:.3f} ms")
        print("-" * 70)
        print(f"Null Control (A=0):    Max Delta R^2 = {res_cos_null.delta_r2:.4f} (< 0.15 threshold)")
        print(f"Total Benchmark Time:  {t_total*1000.0:.2f} ms (< 5000 ms limit, target < 100 ms)")
        print("=" * 70)
        print("[PASS] All infradian verification assertions successfully passed.")

    return metrics


def run_multi_seed_battery(n_seeds: int = 10, verbose: bool = True) -> None:
    """Verifies robustness across multiple random seeds and phase shifts."""
    t_start = time.perf_counter()
    passed = 0
    seeds = [1000 + i * 37 for i in range(n_seeds)]

    for seed in seeds:
        pi_val = math.pi if not HAS_NUMPY_SCIPY else np.pi
        phi = (seed % 100) / 100.0 * 2.0 * pi_val - pi_val
        m = run_synthetic_benchmark(seed=seed, phi_true=phi, verbose=False)
        assert abs(m["ls_peak_period"] - 28.0) <= 0.5, f"Seed {seed} failed LS period"
        assert abs(m["cos_peak_period"] - 28.0) <= 0.5, f"Seed {seed} failed Cos period"
        assert abs(m["cos_amplitude"] - 2.5) / 2.5 <= 0.20, f"Seed {seed} failed Cos amplitude"
        passed += 1

    elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"[PASS] Multi-seed battery ({passed}/{n_seeds} seeds) passed in {elapsed*1000.0:.2f} ms.")


def main() -> int:
    """CLI entry point running primary benchmark, phase test, and multi-seed suite."""
    try:
        # Primary benchmark: exact parameters from prompt (phi=0.0)
        print("\n--- Gate 1: Primary Synthetic Benchmark (phi=0.0) ---")
        run_synthetic_benchmark(seed=20261019, phi_true=0.0, verbose=True)

        # Secondary benchmark: phase shifted (phi=0.45 rad)
        print("\n--- Gate 2: Phase-Shifted Benchmark (phi=0.45 rad) ---")
        run_synthetic_benchmark(seed=20261019, phi_true=0.45, verbose=True)

        # Multi-seed battery: 10 distinct seeds and random phases
        print("\n--- Gate 3: Multi-Seed Robustness Battery (10 seeds) ---")
        run_multi_seed_battery(n_seeds=10, verbose=True)

        print("\n[SUCCESS] All verification gates passed. Exit 0.\n")
        return 0
    except AssertionError as exc:
        print(f"[FAIL] Assertion failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] Unexpected exception: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
