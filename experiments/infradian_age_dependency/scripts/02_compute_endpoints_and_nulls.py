#!/usr/bin/env python3
r"""
02_compute_endpoints_and_nulls.py: Milestone 2 implementation for the Continuous Age RCS Infradian Study.

Pipeline:
1. Daily signal extraction:
   - For every selected date in `artifacts/windows/selected_dates.parquet`, extract daily resting HR proxy `HRp10`
     (10th percentile linear interpolation of observed 5m bin means) and the four 6-hour clock coverage fractions.
2. Nuisance baseline regression:
   - Design matrix: intercept, linear elapsed calendar time (days from window start), weekday indicators (Monday reference, 6 columns),
     and the four clock coverage fractions (12 raw columns).
   - Use deterministic sequential QR screening (drop column if norm < 1e-10 or perpendicular norm / norm <= 1e-4).
   - Form idempotent residual projector P_0 = I - Q Q^T.
3. Spectrum computation:
   - Broad frequency grid: sorted union of {k/720 : k=12,...,240} and exact boundaries 1/35 and 1/25 (231 frequencies).
   - Narrow period grid: T in [25, 35] days, step 0.5d (21 periods).
   - Compute partial R^2: P_i(f) = (RSS_0 - RSS_f) / RSS_0.
4. Two-component covariance null calibration:
   - Model: Cov(e(t), e(u)) = v_1 phi_1^{|t-u|} + v_2 phi_2^{|t-u|}, 0 < phi_1 < 0.95, phi_2 in {0.98, 0.995}.
   - Fit on residual subspace basis Q via Nelder-Mead with 6 starts: variance shares (0.3, 0.5, 0.7) x phi_2 in {0.98, 0.995},
     initial phi_1 from single-AR(1) profile MLE clipped to [1e-4, 0.95 - 1e-4].
   - Tolerances: xatol=1e-6, fatol=1e-6, maxiter=2000.
5. Null simulation:
   - B=199 independent null series per participant on exact selected calendar dates via continuous-time Markov recursions (O(N) path generation).
   - PRNG stream: PCG64 initialized with uint64 from first 8 bytes of SHA-256 of:
     `20261020|infradian_age|{source}|{user_id}|{window_start_day}|null_two_component|{selected_date_hash}`
6. Four physiological endpoints:
   - Primary Q_i: band-integrated mean signed excess over [1/35, 1/25] d^-1 via normalized trapezoidal quadrature:
     Q_i = \int_{1/35}^{1/25} D_i(f) df / (1/25 - 1/35) where D_i(f) = P_i(f) - \bar{P}_{i,null}(f).
   - Discrete peak excess: E_{i2} = \max_{T \in [25, 35]} P_i(1/T) - \bar{P}_{null}(1/T).
   - Physical cosinor amplitude: \hat{A}_i = \sqrt{\beta_{cos}^2 + \beta_{sin}^2} in bpm at selected peak period.
   - Residual standard deviation of resting heart rate: \hat{\sigma}_{resid} = \sqrt{RSS_0 / (N - p)} in bpm.
   - Monte Carlo SE of Q_i: \sigma_{MC, i} = std(Q_{i,null}) / \sqrt{B}.
7. Execution & Artifacts:
   - Multiprocessing via ProcessPoolExecutor with fault-tolerant checkpointing.
   - Saves:
     * `artifacts/endpoints/endpoints.parquet`
     * `artifacts/endpoints/null_diagnostics.parquet`
     * `artifacts/spectra/spectra_summary.parquet`
     * `artifacts/spectra/spectra_broad.npz`
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
from scipy.special import expit
from numpy.random import Generator, PCG64

# Canonical repository and experiment paths
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]

ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
WINDOWS_DIR = ARTIFACTS_DIR / "windows"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"
ENDPOINTS_DIR = ARTIFACTS_DIR / "endpoints"
SPECTRA_DIR = ARTIFACTS_DIR / "spectra"
CHECKPOINTS_DIR = ENDPOINTS_DIR / "checkpoints"

UPSTREAM_APPLE_CACHE = REPO_ROOT / "experiments/apple_cohort_scaling/artifacts/cache_5min"
UPSTREAM_GARMIN_CACHE = REPO_ROOT / "experiments/garmin_infradian_replication/artifacts/cache_5min"
SAMSUNG_CACHE = ARTIFACTS_DIR / "cache_5min"

# Grid constants
BROAD_FREQS = np.unique(np.sort(np.concatenate([
    np.arange(12, 241) / 720.0,
    [1.0 / 35.0, 1.0 / 25.0]
])))
NARROW_PERIODS = np.arange(25.0, 35.5, 0.5)
NARROW_FREQS = 1.0 / NARROW_PERIODS
DELTA_F = (1.0 / 25.0) - (1.0 / 35.0)
IN_BAND_MASK = (BROAD_FREQS >= 1.0 / 35.0 - 1e-12) & (BROAD_FREQS <= 1.0 / 25.0 + 1e-12)
F_BAND = BROAD_FREQS[IN_BAND_MASK]

B_NULL_REPLICATES = 199
PHI2_CANDIDATES = (0.98, 0.995)
FAST_SHARES = (0.3, 0.5, 0.7)


# =============================================================================
# 1. Mathematical Algorithms & Operators
# =============================================================================

def extract_features_for_selected_dates(
    hr_arr: np.ndarray,
    date_to_row: Dict[int, int],
    selected_dates: np.ndarray,
    percentile: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Extracts HRp10 (10th percentile linear interpolation of observed 5m bin means)
    and 4 clock coverage fractions (00:00-06:00, 06:00-12:00, 12:00-18:00, 18:00-24:00)
    for each date in selected_dates.
    """
    n = len(selected_dates)
    hr_p10 = np.empty(n, dtype=np.float64)
    clock_fracs = np.empty((n, 4), dtype=np.float64)
    valid_counts = np.empty(n, dtype=np.int64)

    for i, d in enumerate(selected_dates):
        row_idx = date_to_row[int(d)]
        row_bins = hr_arr[row_idx]
        mask = np.isfinite(row_bins)
        v_count = int(np.sum(mask))
        valid_counts[i] = v_count

        if v_count == 0:
            hr_p10[i] = np.nan
            clock_fracs[i, :] = 0.0
        else:
            # Linear quantile interpolation (NumPy default)
            hr_p10[i] = float(np.percentile(row_bins[mask], percentile))
            clock_fracs[i, 0] = float(np.sum(mask[0:72]) / 72.0)
            clock_fracs[i, 1] = float(np.sum(mask[72:144]) / 72.0)
            clock_fracs[i, 2] = float(np.sum(mask[144:216]) / 72.0)
            clock_fracs[i, 3] = float(np.sum(mask[216:288]) / 72.0)

    mean_occ = float(np.mean(valid_counts / 288.0))
    mean_night_cov = float(np.mean(clock_fracs[:, 0]))
    date_span = int(selected_dates[-1] - selected_dates[0])

    return hr_p10, clock_fracs, mean_occ, mean_night_cov, date_span


def build_and_screen_nuisance_baseline(
    dates: np.ndarray,
    clock_fracs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Builds the 12-column nuisance baseline matrix and screens columns sequentially via QR:
    - Col 0: Intercept
    - Col 1: Linear elapsed calendar time (dates - dates[0])
    - Cols 2-7: Weekday indicators (Tue-Sun, Mon reference)
    - Cols 8-11: Four clock coverage fractions
    
    Screens column v:
      drop if norm(v) < 1e-10 or norm(v_perp) / norm(v) <= 1e-4.
    
    Returns:
      P_0: Idempotent residual projector (I - Q Q^T), shape (N, N)
      Q_basis: Orthonormal basis of retained columns, shape (N, p_retained)
      p_retained: Number of retained columns
    """
    n = len(dates)
    cols = [
        np.ones(n, dtype=np.float64),
        (dates - dates[0]).astype(np.float64),
    ]
    dow = (dates + 3) % 7  # 0=Mon, ..., 6=Sun
    for d in range(1, 7):
        cols.append((dow == d).astype(np.float64))
    for c in range(4):
        cols.append(clock_fracs[:, c].astype(np.float64))

    X_raw = np.column_stack(cols)

    Q_basis = np.zeros((n, 0), dtype=np.float64)
    for j in range(X_raw.shape[1]):
        v = X_raw[:, j]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        if Q_basis.shape[1] > 0:
            v_proj = Q_basis @ (Q_basis.T @ v)
            v_perp = v - v_proj
            perp_norm = np.linalg.norm(v_perp)
            if perp_norm / v_norm > 1e-4:
                q_new = v_perp / perp_norm
                Q_basis = np.column_stack([Q_basis, q_new])
        else:
            q_new = v / v_norm
            Q_basis = q_new[:, np.newaxis]

    p_retained = Q_basis.shape[1]
    P_0 = np.eye(n, dtype=np.float64) - (Q_basis @ Q_basis.T)
    return P_0, Q_basis, p_retained


def profile_ar1_mle(e_0: np.ndarray, dates: np.ndarray) -> float:
    """
    Estimates stationary calendar-time AR(1) daily autocorrelation phi
    via exact profile Gaussian likelihood, clipped to [1e-4, 0.95 - 1e-4].
    """
    n = len(e_0)
    delta_t = np.diff(dates)
    y1_sq = e_0[0] ** 2
    y_prev = e_0[:-1]
    y_curr = e_0[1:]

    def neg_prof_lik(phi: float) -> float:
        phi_dt = phi ** delta_t
        om_p2 = np.maximum(1.0 - phi_dt ** 2, 1e-12)
        inno_sq = (y_curr - phi_dt * y_prev) ** 2
        s2 = (y1_sq + np.sum(inno_sq / om_p2)) / n
        return n * np.log(max(s2, 1e-12)) + np.sum(np.log(om_p2))

    res = scipy.optimize.minimize_scalar(neg_prof_lik, bounds=(1e-4, 0.95 - 1e-4), method="bounded")
    return float(np.clip(res.x, 1e-4, 0.95 - 1e-4))


def fit_two_component_reml(
    z: np.ndarray,
    Q_sub: np.ndarray,
    dates: np.ndarray,
    rss_0: float,
    r_rank: int,
    phi_init: float,
) -> Tuple[str, Dict[str, Any]]:
    """
    Fits the two-component calendar covariance model:
      Cov(e(t), e(u)) = v1 * phi1^|t-u| + v2 * phi2^|t-u|, 0 < phi1 < 0.95, phi2 in {0.98, 0.995}
    on the residual subspace basis Q_sub via Nelder-Mead with 6 starts:
      variance shares alpha in {0.3, 0.5, 0.7} x phi2 in {0.98, 0.995}.
    """
    t_start = time.perf_counter()
    dt_abs = np.abs(dates[:, None] - dates[None, :]).astype(np.int64)
    max_dt = int(np.max(dt_abs))
    powers_idx = np.arange(max_dt + 1)
    s_init = rss_0 / r_rank
    th1_init = np.log(phi_init / (0.95 - phi_init))

    best_result: Optional[Dict[str, Any]] = None
    num_converged = 0
    start_idx = 0

    for phi2 in PHI2_CANDIDATES:
        p2_powers = phi2 ** powers_idx
        K2 = p2_powers[dt_abs]
        S2 = Q_sub.T @ (K2 @ Q_sub)

        for alpha in FAST_SHARES:
            start_idx += 1
            th0 = np.array([
                np.log(alpha * s_init),
                th1_init,
                np.log((1.0 - alpha) * s_init),
            ], dtype=np.float64)

            def nll(th: np.ndarray) -> float:
                s1 = np.exp(th[0])
                p1 = 0.95 * expit(th[1])
                s2 = np.exp(th[2])
                p1_powers = p1 ** powers_idx
                C1 = p1_powers[dt_abs]
                S = s1 * (Q_sub.T @ (C1 @ Q_sub)) + s2 * S2
                try:
                    L = scipy.linalg.cholesky(S, lower=True)
                except scipy.linalg.LinAlgError:
                    return 1e10
                a = scipy.linalg.solve_triangular(L, z, lower=True)
                return float(np.sum(np.log(np.diag(L))) + 0.5 * (a @ a))

            opt = scipy.optimize.minimize(
                nll,
                th0,
                method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
            )

            if opt.success and np.isfinite(opt.fun):
                num_converged += 1
                if best_result is None or opt.fun < best_result["nll"]:
                    v1 = float(np.exp(opt.x[0]))
                    phi1 = float(0.95 * expit(opt.x[1]))
                    v2 = float(np.exp(opt.x[2]))
                    best_result = {
                        "nll": float(opt.fun),
                        "v1": v1,
                        "phi1": phi1,
                        "v2": v2,
                        "phi2": float(phi2),
                        "nit": int(opt.nit),
                        "best_start_idx": start_idx,
                    }

    t_elapsed = time.perf_counter() - t_start

    if best_result is None:
        return "failed", {
            "s_init": float(s_init),
            "phi_init": float(phi_init),
            "v1": np.nan,
            "phi1": np.nan,
            "v2": np.nan,
            "phi2": np.nan,
            "reml_nll": np.nan,
            "num_starts_converged": 0,
            "best_start_idx": -1,
            "nit": 0,
            "reml_elapsed_sec": float(t_elapsed),
        }

    best_result["s_init"] = float(s_init)
    best_result["phi_init"] = float(phi_init)
    best_result["reml_nll"] = best_result.pop("nll")
    best_result["num_starts_converged"] = num_converged
    best_result["reml_elapsed_sec"] = float(t_elapsed)
    return "converged", best_result


def precompute_fwl_harmonic_operators(
    dates: np.ndarray,
    P_0: np.ndarray,
    freqs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precomputes Frisch-Waugh-Lovell harmonic projection operators M and CTC for given frequencies.
    M_stack shape: (N, 2*K)
    CTC_stack shape: (K, 2, 2)
    """
    n = len(dates)
    K = len(freqs)
    t_elapsed = (dates - dates[0]).astype(np.float64)

    M_stack = np.empty((n, 2 * K), dtype=np.float64)
    CTC_stack = np.empty((K, 2, 2), dtype=np.float64)

    for i, f in enumerate(freqs):
        omega = 2.0 * np.pi * f
        C = np.column_stack([np.cos(omega * t_elapsed), np.sin(omega * t_elapsed)])
        C_tilde = P_0 @ C
        CTC = C_tilde.T @ C_tilde
        det = CTC[0, 0] * CTC[1, 1] - CTC[0, 1] * CTC[1, 0]
        if det < 1e-12:
            CTC[0, 0] += 1e-8
            CTC[1, 1] += 1e-8
            det = CTC[0, 0] * CTC[1, 1] - CTC[0, 1] * CTC[1, 0]
        inv_CTC = np.array([
            [CTC[1, 1], -CTC[0, 1]],
            [-CTC[1, 0], CTC[0, 0]],
        ], dtype=np.float64) / det
        M_stack[:, 2 * i : 2 * i + 2] = C_tilde @ inv_CTC
        CTC_stack[i] = CTC

    return M_stack, CTC_stack


def simulate_two_component_ar1(
    dates: np.ndarray,
    v1: float,
    phi1: float,
    v2: float,
    phi2: float,
    b_replicates: int,
    rng: Generator,
) -> np.ndarray:
    """
    Generates B independent null series via O(N) continuous-time Markov recursions.
    """
    n = len(dates)
    delta_t = np.diff(dates)

    def _sim_component(v: float, phi: float) -> np.ndarray:
        sigma = np.sqrt(max(v, 1e-12))
        z = rng.standard_normal((b_replicates, n))
        out = np.empty((b_replicates, n), dtype=np.float64)
        out[:, 0] = sigma * z[:, 0]
        for k in range(n - 1):
            decay = phi ** delta_t[k]
            scale = sigma * np.sqrt(max(0.0, 1.0 - decay ** 2))
            out[:, k + 1] = decay * out[:, k] + scale * z[:, k + 1]
        return out

    x1 = _sim_component(v1, phi1)
    x2 = _sim_component(v2, phi2)
    return x1 + x2


# =============================================================================
# 2. Worker Task: Participant Processing
# =============================================================================

def process_participant_task(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Processes all eligible windows for a single participant (user_id, source).
    Loads the participant's 5m cache once, and computes endpoints, nulls, and spectra for every window.
    """
    user_id = task["user_id"]
    source = task["source"]
    cache_path_str = task["cache_path"]
    demographics = task["demographics"]
    windows = task["windows"]
    percentile = float(task.get("percentile", 10.0))
    b_null_replicates = int(task.get("null_replicates", B_NULL_REPLICATES))

    out_records = []
    cache_path = Path(cache_path_str)
    if not cache_path.exists():
        for win in windows:
            out_records.append({
                "status": "cache_missing",
                "user_id": user_id,
                "source": source,
                "protocol": win["protocol"],
                "window_index": win["window_index"],
            })
        return out_records

    try:
        with np.load(cache_path) as d:
            hr_arr = d["hr"]
            date_arr = d["date"]

        date_to_row = {int(d_val): idx for idx, d_val in enumerate(date_arr)}
    except Exception as e:
        for win in windows:
            out_records.append({
                "status": f"cache_error: {e}",
                "user_id": user_id,
                "source": source,
                "protocol": win["protocol"],
                "window_index": win["window_index"],
            })
        return out_records

    # Process each window
    for win in windows:
        protocol = win["protocol"]
        w_idx = win["window_index"]
        start_day = int(win["start_day"])
        selected_dates_hash = str(win["selected_dates_hash"])
        selected_dates = np.asarray(win["dates"], dtype=np.int64)

        record: Dict[str, Any] = {
            "status": "ok",
            "user_id": user_id,
            "source": source,
            "protocol": protocol,
            "window_index": w_idx,
            "start_day": start_day,
            "selected_dates_hash": selected_dates_hash,
            # Demographic attributes
            "salutation": demographics["salutation"],
            "birth_year": demographics["birth_year"],
            "age_dec1": demographics["age_dec1"],
            "age_jan1": demographics["age_jan1"],
            "age_dec31": demographics["age_dec31"],
            "age": demographics["age_dec1"],
            "sex": 1 if demographics["salutation"] == 10 else 0,
            "weight": demographics["weight"],
            "height": demographics["height"],
            "bmi": demographics["bmi"],
            "arm": str(task.get("arm", f"HRp{int(percentile)}")),
            "n_selected_dates": int(len(selected_dates)),
        }

        # 1. Daily signal extraction
        hr_p10, clock_fracs, mean_occ, mean_night_cov, date_span = extract_features_for_selected_dates(
            hr_arr, date_to_row, selected_dates, percentile
        )
        record["n_qualified_dates"] = int(np.sum(
            np.isfinite(hr_arr[[date_to_row[int(d)] for d in selected_dates]]).sum(axis=1) >= 120
        ))
        # Supplier values can replace the endpoint while preserving channel-3000 coverage covariates.
        if task.get("daily_values") is not None:
            daily_values = task["daily_values"]
            hr_p10 = np.asarray([daily_values.get(int(d), np.nan) for d in selected_dates], dtype=np.float64)
        if task.get("save_daily_values", False):
            record["daily_values"] = hr_p10.astype(float).tolist()
        record["mean_occ"] = mean_occ
        record["mean_night_cov"] = mean_night_cov
        record["date_span"] = date_span

        if np.isnan(hr_p10).any():
            record["status"] = "nan_hrp10"
            record["null_status"] = "failed"
            out_records.append(record)
            continue

        # 2. Nuisance baseline regression
        P_0, Q_basis, p_retained = build_and_screen_nuisance_baseline(selected_dates, clock_fracs)
        e_0 = P_0 @ hr_p10
        rss_0 = float(e_0 @ e_0)
        r_rank = 90 - p_retained
        record["rss_0"] = rss_0
        record["p_retained"] = p_retained

        if rss_0 < 1e-10 or not np.isfinite(rss_0) or r_rank <= 0:
            record["status"] = "rss0_degenerate"
            record["null_status"] = "failed"
            record["sigma_resid"] = np.nan
            out_records.append(record)
            continue

        sigma_resid = float(np.sqrt(rss_0 / r_rank))
        record["sigma_resid"] = sigma_resid

        # Residual subspace basis Q_sub
        eigvals, U = np.linalg.eigh(P_0)
        Q_sub = U[:, eigvals > 0.5]  # Shape (90, r_rank)
        z = Q_sub.T @ e_0

        # 3. Single AR(1) profile MLE
        phi_init = profile_ar1_mle(e_0, selected_dates)

        # 4. Two-component covariance null calibration (REML)
        # A caller may inject an alternative fitter (the multisource pipeline
        # uses this to opt into the analytical-gradient L-BFGS-B path).
        reml_fitter = task.get("reml_fitter", fit_two_component_reml)
        null_status, diag = reml_fitter(
            z, Q_sub, selected_dates, rss_0, r_rank, phi_init
        )
        record["null_status"] = null_status
        record["null_diagnostics"] = diag

        if null_status != "converged":
            record["status"] = "reml_failed"
            record["v1"] = np.nan
            record["phi1"] = np.nan
            record["v2"] = np.nan
            record["phi2"] = np.nan
            record["Q_i"] = np.nan
            record["E_i2"] = np.nan
            record["A_hat"] = np.nan
            record["Q_i_mc_se"] = np.nan
            record["peak_period"] = np.nan
            record["S_obs"] = np.nan
            record["mean_S_null"] = np.nan
            record["p_value_discrete"] = np.nan
            out_records.append(record)
            continue

        v1 = diag["v1"]
        phi1 = diag["phi1"]
        v2 = diag["v2"]
        phi2 = diag["phi2"]
        record["v1"] = v1
        record["phi1"] = phi1
        record["v2"] = v2
        record["phi2"] = phi2

        # 5. Operators for 231 broad and 21 narrow frequencies
        M_broad, CTC_broad = precompute_fwl_harmonic_operators(selected_dates, P_0, BROAD_FREQS)
        M_narrow, CTC_narrow = precompute_fwl_harmonic_operators(selected_dates, P_0, NARROW_FREQS)

        # 6. Null simulation
        seed_str = f"20261020|infradian_age|{source}|{user_id}|{start_day}|null_two_component|{selected_dates_hash}"
        seed_u64 = int.from_bytes(hashlib.sha256(seed_str.encode("utf-8")).digest()[:8], byteorder="little")
        rng = Generator(PCG64(seed_u64))
        sim_noise = simulate_two_component_ar1(selected_dates, v1, phi1, v2, phi2, b_null_replicates, rng)
        sim_e = sim_noise @ P_0
        sim_rss0 = np.maximum(np.sum(sim_e ** 2, axis=1), 1e-12)

        # Broad null spectra
        beta_broad_null = sim_e @ M_broad  # Shape (B, 2*K)
        P_broad_null = np.empty((b_null_replicates, len(BROAD_FREQS)), dtype=np.float64)
        for i in range(len(BROAD_FREQS)):
            bc = beta_broad_null[:, 2 * i]
            bs = beta_broad_null[:, 2 * i + 1]
            ctc = CTC_broad[i]
            drss = bc ** 2 * ctc[0, 0] + 2.0 * bc * bs * ctc[0, 1] + bs ** 2 * ctc[1, 1]
            P_broad_null[:, i] = np.clip(drss / sim_rss0, 0.0, 1.0)

        # Narrow null spectra
        beta_narrow_null = sim_e @ M_narrow
        P_narrow_null = np.empty((b_null_replicates, len(NARROW_FREQS)), dtype=np.float64)
        for i in range(len(NARROW_FREQS)):
            bc = beta_narrow_null[:, 2 * i]
            bs = beta_narrow_null[:, 2 * i + 1]
            ctc = CTC_narrow[i]
            drss = bc ** 2 * ctc[0, 0] + 2.0 * bc * bs * ctc[0, 1] + bs ** 2 * ctc[1, 1]
            P_narrow_null[:, i] = np.clip(drss / sim_rss0, 0.0, 1.0)

        # 7. Observed spectra
        beta_broad_obs = e_0 @ M_broad
        P_broad_obs = np.empty(len(BROAD_FREQS), dtype=np.float64)
        for i in range(len(BROAD_FREQS)):
            bc = beta_broad_obs[2 * i]
            bs = beta_broad_obs[2 * i + 1]
            ctc = CTC_broad[i]
            drss = bc ** 2 * ctc[0, 0] + 2.0 * bc * bs * ctc[0, 1] + bs ** 2 * ctc[1, 1]
            P_broad_obs[i] = np.clip(drss / rss_0, 0.0, 1.0)

        beta_narrow_obs = e_0 @ M_narrow
        P_narrow_obs = np.empty(len(NARROW_FREQS), dtype=np.float64)
        for i in range(len(NARROW_FREQS)):
            bc = beta_narrow_obs[2 * i]
            bs = beta_narrow_obs[2 * i + 1]
            ctc = CTC_narrow[i]
            drss = bc ** 2 * ctc[0, 0] + 2.0 * bc * bs * ctc[0, 1] + bs ** 2 * ctc[1, 1]
            P_narrow_obs[i] = np.clip(drss / rss_0, 0.0, 1.0)

        # 8. Four physiological endpoints
        P_null_mean_broad = np.mean(P_broad_null, axis=0)
        P_null_var_broad = np.var(P_broad_null, axis=0, ddof=1)
        D_obs = P_broad_obs - P_null_mean_broad

        # Primary Q_i via normalized trapezoidal quadrature over [1/35, 1/25]
        Q_i = float(np.trapezoid(D_obs[IN_BAND_MASK], F_BAND) / DELTA_F)
        record["Q_i"] = Q_i

        # Null Q distribution and Monte Carlo SE
        Q_null = np.array([
            np.trapezoid(P_broad_null[b, IN_BAND_MASK], F_BAND) / DELTA_F
            for b in range(b_null_replicates)
        ])
        Q_i_mc_se = float(np.std(Q_null, ddof=1) / np.sqrt(b_null_replicates))
        record["Q_i_mc_se"] = Q_i_mc_se
        diag["Q_i_mc_se"] = Q_i_mc_se

        # Discrete peak excess E_i2
        S_obs = float(np.max(P_narrow_obs))
        S_null = np.max(P_narrow_null, axis=1)
        mean_S_null = float(np.mean(S_null))
        E_i2 = float(S_obs - mean_S_null)
        p_val_discrete = float((1.0 + np.sum(S_null >= S_obs)) / (b_null_replicates + 1.0))
        record["S_obs"] = S_obs
        record["mean_S_null"] = mean_S_null
        record["E_i2"] = E_i2
        record["p_value_discrete"] = p_val_discrete

        # Cosinor amplitude A_hat at selected peak period
        peak_idx = int(np.argmax(P_narrow_obs))
        best_T = float(NARROW_PERIODS[peak_idx])
        record["peak_period"] = best_T
        bc_peak = beta_narrow_obs[2 * peak_idx]
        bs_peak = beta_narrow_obs[2 * peak_idx + 1]
        A_hat = float(np.sqrt(bc_peak ** 2 + bs_peak ** 2))
        record["A_hat"] = A_hat

        # Store spectra payload
        record["spectra"] = {
            "P_obs": P_broad_obs.astype(np.float32),
            "P_null_mean": P_null_mean_broad.astype(np.float32),
            "P_null_var": P_null_var_broad.astype(np.float32),
            "D_excess": D_obs.astype(np.float32),
        }

        out_records.append(record)

    return out_records


# =============================================================================
# 3. Main Runner Implementation
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute physiological endpoints, REML nulls, and spectra.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of participants for testing.")
    parser.add_argument("--source", type=int, default=None, help="Limit to specific source (3, 6, or 7).")
    parser.add_argument("--protocol", type=str, default=None, help="Limit to expanded90 or balanced90.")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing.")
    return parser.parse_args()


def main() -> None:
    t_start = time.perf_counter()
    args = parse_args()

    print("================================================================================")
    print("STARTING MILESTONE 2: Four-Endpoint Suite & Two-Component REML Null Calibration")
    print("================================================================================")

    ENDPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    SPECTRA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load inputs
    print("\n--- [Step 1/5] Loading Windows, Selected Dates, and Participants Manifest ---")
    p_participants = MANIFESTS_DIR / "participants.parquet"
    p_windows = WINDOWS_DIR / "windows.parquet"
    p_selected_dates = WINDOWS_DIR / "selected_dates.parquet"

    if not p_participants.exists() or not p_windows.exists() or not p_selected_dates.exists():
        raise FileNotFoundError("M1 artifacts missing! Run 01_build_manifests_and_windows.py first.")

    df_part = pd.read_parquet(p_participants)
    df_win = pd.read_parquet(p_windows)
    df_sel = pd.read_parquet(p_selected_dates)
    print(f"  Loaded {len(df_part)} participant metadata rows.")
    print(f"  Loaded {len(df_win)} window records.")
    print(f"  Loaded {len(df_sel)} selected date records.")

    # Apply filters if requested
    if args.source is not None:
        df_sel = df_sel[df_sel["source"] == args.source].copy()
        print(f"  Filtered to source={args.source}: {len(df_sel)} dates remaining.")
    if args.protocol is not None:
        df_sel = df_sel[df_sel["protocol"] == args.protocol].copy()
        print(f"  Filtered to protocol={args.protocol}: {len(df_sel)} dates remaining.")

    # 2. Build Demographic Lookup Dictionary
    part_demo_dict: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for _, row in df_part.iterrows():
        key = (int(row["user_id"]), int(row["source"]))
        part_demo_dict[key] = {
            "salutation": int(row["salutation"]) if pd.notna(row["salutation"]) else None,
            "birth_year": int(row["birth_year"]) if pd.notna(row["birth_year"]) else None,
            "age_dec1": int(row["age_dec1"]) if pd.notna(row["age_dec1"]) else None,
            "age_jan1": int(row["age_jan1"]) if pd.notna(row["age_jan1"]) else None,
            "age_dec31": int(row["age_dec31"]) if pd.notna(row["age_dec31"]) else None,
            "weight": float(row["weight"]) if pd.notna(row["weight"]) else np.nan,
            "height": float(row["height"]) if pd.notna(row["height"]) else np.nan,
            "bmi": float(row["bmi"]) if pd.notna(row["bmi"]) else np.nan,
        }

    # Build Window Metadata Lookup: (user_id, source, protocol, window_index) -> start_day
    win_start_dict: Dict[Tuple[int, int, str, int], int] = {}
    for _, row in df_win.iterrows():
        if bool(row["is_eligible"]):
            w_key = (int(row["user_id"]), int(row["source"]), str(row["protocol"]), int(row["window_index"]))
            win_start_dict[w_key] = int(row["start_day"])

    # 3. Assemble Participant Work Tasks
    print("\n--- [Step 2/5] Assembling Participant Tasks ---")
    grouped_dates = df_sel.groupby(["user_id", "source", "protocol", "window_index", "selected_dates_hash"])

    participant_windows_map: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for (uid, src, proto, w_idx, s_hash), group in grouped_dates:
        u_key = (int(uid), int(src))
        if u_key not in participant_windows_map:
            participant_windows_map[u_key] = []

        w_start = win_start_dict.get((int(uid), int(src), str(proto), int(w_idx)), int(group["date"].min()))
        sorted_dates = np.sort(group["date"].to_numpy(dtype=np.int64)).tolist()

        participant_windows_map[u_key].append({
            "protocol": str(proto),
            "window_index": int(w_idx),
            "start_day": w_start,
            "selected_dates_hash": str(s_hash),
            "dates": sorted_dates,
        })

    tasks: List[Dict[str, Any]] = []
    for (uid, src), wins in participant_windows_map.items():
        if src == 6:
            cache_p = UPSTREAM_APPLE_CACHE / f"customer_{uid}.npz"
        elif src == 3:
            cache_p = UPSTREAM_GARMIN_CACHE / f"customer_{uid}.npz"
        elif src == 7:
            cache_p = SAMSUNG_CACHE / f"customer_{uid}.npz"
        else:
            continue

        demo = part_demo_dict.get((uid, src), {
            "salutation": None, "birth_year": None, "age_dec1": None,
            "age_jan1": None, "age_dec31": None, "weight": np.nan,
            "height": np.nan, "bmi": np.nan,
        })

        tasks.append({
            "user_id": uid,
            "source": src,
            "cache_path": str(cache_p),
            "demographics": demo,
            "windows": wins,
        })

    if args.limit is not None and args.limit > 0:
        tasks = tasks[:args.limit]
        print(f"  Limited to first {len(tasks)} participant tasks.")

    print(f"  Total candidate tasks: {len(tasks)}")

    # Check for existing checkpoints
    all_records: List[Dict[str, Any]] = []
    completed_keys: Set[Tuple[int, int]] = set()

    if not args.no_checkpoint:
        ckpt_files = sorted(CHECKPOINTS_DIR.glob("checkpoint_*.parquet"))
        if len(ckpt_files) > 0:
            print(f"  Found {len(ckpt_files)} existing checkpoint files. Loading...")
            for ckpt_p in ckpt_files:
                df_c = pd.read_parquet(ckpt_p)
                for _, row in df_c.iterrows():
                    completed_keys.add((int(row["user_id"]), int(row["source"])))
            print(f"  Resuming from checkpoints: {len(completed_keys)} participants already completed.")
            tasks = [t for t in tasks if (t["user_id"], t["source"]) not in completed_keys]
            print(f"  Remaining participant tasks to process: {len(tasks)}")

    n_total_windows = sum(len(t["windows"]) for t in tasks)
    print(f"  Remaining window evaluations: {n_total_windows}")

    # 4. Multiprocessing Execution
    workers = args.workers or min(os.cpu_count() or 4, 12)
    print(f"\n--- [Step 3/5] Executing Pipeline with {workers} Worker Processes ---")
    t0 = time.perf_counter()

    chunk_size = max(1, min(50, len(tasks) // (workers * 4) + 1)) if len(tasks) > 0 else 1
    batch_records: List[Dict[str, Any]] = []
    completed_in_run = 0
    checkpoint_batch_idx = len(list(CHECKPOINTS_DIR.glob("checkpoint_*.parquet")))

    if len(tasks) > 0:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for res_list in executor.map(process_participant_task, tasks, chunksize=chunk_size):
                batch_records.extend(res_list)
                completed_in_run += 1

                # Checkpoint every 500 tasks
                if not args.no_checkpoint and completed_in_run % 500 == 0:
                    checkpoint_batch_idx += 1
                    ckpt_p = CHECKPOINTS_DIR / f"checkpoint_{checkpoint_batch_idx:05d}.parquet"
                    df_ckpt = pd.DataFrame([{
                        k: v for k, v in r.items() if k != "spectra" and k != "null_diagnostics"
                    } for r in batch_records])
                    df_ckpt.to_parquet(ckpt_p, index=False)

                    # Also save spectra batch
                    npz_ckpt_p = CHECKPOINTS_DIR / f"spectra_{checkpoint_batch_idx:05d}.npz"
                    spec_records = [r for r in batch_records if "spectra" in r]
                    if len(spec_records) > 0:
                        np.savez_compressed(
                            npz_ckpt_p,
                            user_id=np.array([r["user_id"] for r in spec_records], dtype=np.int64),
                            source=np.array([r["source"] for r in spec_records], dtype=np.int64),
                            protocol=np.array([r["protocol"] for r in spec_records], dtype=str),
                            window_index=np.array([r["window_index"] for r in spec_records], dtype=np.int64),
                            P_obs=np.stack([r["spectra"]["P_obs"] for r in spec_records]),
                            P_null_mean=np.stack([r["spectra"]["P_null_mean"] for r in spec_records]),
                            P_null_var=np.stack([r["spectra"]["P_null_var"] for r in spec_records]),
                        )
                    all_records.extend(batch_records)
                    batch_records = []

                if completed_in_run % 200 == 0 or completed_in_run == len(tasks):
                    elapsed = time.perf_counter() - t0
                    rate = completed_in_run / max(elapsed, 0.1)
                    eta = (len(tasks) - completed_in_run) / max(rate, 0.01)
                    pct = 100.0 * completed_in_run / len(tasks)
                    print(f"  [{completed_in_run:5d}/{len(tasks):5d}] ({pct:5.1f}%) | {rate:4.1f} tasks/s | Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s")

    # Final batch
    if len(batch_records) > 0:
        if not args.no_checkpoint:
            checkpoint_batch_idx += 1
            ckpt_p = CHECKPOINTS_DIR / f"checkpoint_{checkpoint_batch_idx:05d}.parquet"
            df_ckpt = pd.DataFrame([{
                k: v for k, v in r.items() if k != "spectra" and k != "null_diagnostics"
            } for r in batch_records])
            df_ckpt.to_parquet(ckpt_p, index=False)

            npz_ckpt_p = CHECKPOINTS_DIR / f"spectra_{checkpoint_batch_idx:05d}.npz"
            spec_records = [r for r in batch_records if "spectra" in r]
            if len(spec_records) > 0:
                np.savez_compressed(
                    npz_ckpt_p,
                    user_id=np.array([r["user_id"] for r in spec_records], dtype=np.int64),
                    source=np.array([r["source"] for r in spec_records], dtype=np.int64),
                    protocol=np.array([r["protocol"] for r in spec_records], dtype=str),
                    window_index=np.array([r["window_index"] for r in spec_records], dtype=np.int64),
                    P_obs=np.stack([r["spectra"]["P_obs"] for r in spec_records]),
                    P_null_mean=np.stack([r["spectra"]["P_null_mean"] for r in spec_records]),
                    P_null_var=np.stack([r["spectra"]["P_null_var"] for r in spec_records]),
                )
        all_records.extend(batch_records)

    print(f"  Execution phase complete in {time.perf_counter() - t0:.1f}s.")

    # 5. Build Final Artifact DataFrames from all records or checkpoints
    print("\n--- [Step 4/5] Compiling and Structuring Final Output Artifacts ---")
    
    # If checkpoints were used and we have checkpoint files on disk, aggregate them
    ckpt_files = sorted(CHECKPOINTS_DIR.glob("checkpoint_*.parquet"))
    npz_ckpt_files = sorted(CHECKPOINTS_DIR.glob("spectra_*.npz"))

    if len(ckpt_files) > 0:
        print(f"  Aggregating {len(ckpt_files)} checkpoint chunks into final tables...")
        dfs = [pd.read_parquet(p) for p in ckpt_files]
        df_endpoints = pd.concat(dfs, ignore_index=True)
        # Drop duplicates if any
        df_endpoints = df_endpoints.drop_duplicates(subset=["user_id", "source", "protocol", "window_index"]).reset_index(drop=True)

        # Build null diagnostics from endpoints table
        diag_cols = [
            "user_id", "source", "protocol", "window_index", "selected_dates_hash",
            "null_status", "rss_0", "v1", "phi1", "v2", "phi2", "Q_i_mc_se"
        ]
        df_null_diag = df_endpoints[[c for c in diag_cols if c in df_endpoints.columns]].copy()

        # Aggregate spectra npz
        npz_u = []
        npz_s = []
        npz_p = []
        npz_w = []
        npz_obs = []
        npz_null = []
        npz_var = []
        for p in npz_ckpt_files:
            with np.load(p, allow_pickle=True) as d:
                npz_u.append(d["user_id"])
                npz_s.append(d["source"])
                npz_p.append(np.array(d["protocol"], dtype=str))
                npz_w.append(d["window_index"])
                npz_obs.append(d["P_obs"])
                npz_null.append(d["P_null_mean"])
                npz_var.append(d["P_null_var"])

        if len(npz_u) > 0:
            arr_u = np.concatenate(npz_u)
            arr_s = np.concatenate(npz_s)
            arr_p = np.concatenate(npz_p)
            arr_w = np.concatenate(npz_w)
            arr_obs = np.concatenate(npz_obs, axis=0)
            arr_null = np.concatenate(npz_null, axis=0)
            arr_var = np.concatenate(npz_var, axis=0)

            # Build spectra_summary dataframe
            spec_rows = []
            for i in range(len(arr_u)):
                spec_rows.append({
                    "user_id": int(arr_u[i]),
                    "source": int(arr_s[i]),
                    "protocol": str(arr_p[i]),
                    "window_index": int(arr_w[i]),
                    "P_obs": arr_obs[i].tolist(),
                    "P_null_mean": arr_null[i].tolist(),
                    "P_null_var": arr_var[i].tolist(),
                    "D_excess": (arr_obs[i] - arr_null[i]).tolist(),
                })
            df_spectra_summary = pd.DataFrame(spec_rows)
            hash_map = df_endpoints[["user_id", "source", "protocol", "window_index", "selected_dates_hash"]].drop_duplicates()
            df_spectra_summary = df_spectra_summary.merge(hash_map, on=["user_id", "source", "protocol", "window_index"], how="left")
        else:
            df_spectra_summary = pd.DataFrame()
    else:
        # Build directly from all_records in memory
        endpoint_rows = []
        null_diag_rows = []
        spectra_rows = []
        npz_u, npz_s, npz_p, npz_w, npz_obs, npz_null, npz_var = [], [], [], [], [], [], []

        for rec in all_records:
            if rec.get("status") not in ("ok", "reml_failed"):
                continue
            u = rec["user_id"]
            s = rec["source"]
            p = rec["protocol"]
            w = rec["window_index"]
            h = rec["selected_dates_hash"]

            endpoint_rows.append({
                "user_id": u, "source": s, "protocol": p, "window_index": w,
                "start_day": rec["start_day"], "selected_dates_hash": h,
                "age": rec["age"], "sex": rec["sex"], "salutation": rec["salutation"],
                "birth_year": rec["birth_year"], "age_dec1": rec["age_dec1"],
                "age_jan1": rec["age_jan1"], "age_dec31": rec["age_dec31"],
                "bmi": rec["bmi"], "weight": rec["weight"], "height": rec["height"],
                "mean_occ": rec["mean_occ"], "mean_night_cov": rec["mean_night_cov"],
                "date_span": rec["date_span"], "Q_i": rec["Q_i"], "E_i2": rec["E_i2"],
                "A_hat": rec["A_hat"], "sigma_resid": rec["sigma_resid"],
                "Q_i_mc_se": rec["Q_i_mc_se"], "null_status": rec["null_status"],
                "v1": rec["v1"], "v2": rec["v2"], "phi1": rec["phi1"], "phi2": rec["phi2"],
                "peak_period": rec["peak_period"], "S_obs": rec["S_obs"],
                "mean_S_null": rec["mean_S_null"], "p_value_discrete": rec["p_value_discrete"],
                "rss_0": rec["rss_0"], "p_retained": rec["p_retained"],
            })

            diag = rec.get("null_diagnostics", {})
            null_diag_rows.append({
                "user_id": u, "source": s, "protocol": p, "window_index": w,
                "selected_dates_hash": h, "null_status": rec["null_status"],
                "s_init": diag.get("s_init", np.nan), "phi_init": diag.get("phi_init", np.nan),
                "v1": diag.get("v1", np.nan), "phi1": diag.get("phi1", np.nan),
                "v2": diag.get("v2", np.nan), "phi2": diag.get("phi2", np.nan),
                "reml_nll": diag.get("reml_nll", np.nan),
                "num_starts_converged": diag.get("num_starts_converged", 0),
                "best_start_idx": diag.get("best_start_idx", -1),
                "nit": diag.get("nit", 0),
                "reml_elapsed_sec": diag.get("reml_elapsed_sec", np.nan),
                "Q_i_mc_se": diag.get("Q_i_mc_se", np.nan),
            })

            spec = rec.get("spectra")
            if spec is not None:
                spectra_rows.append({
                    "user_id": u, "source": s, "protocol": p, "window_index": w,
                    "selected_dates_hash": h,
                    "P_obs": spec["P_obs"].tolist(),
                    "P_null_mean": spec["P_null_mean"].tolist(),
                    "P_null_var": spec["P_null_var"].tolist(),
                    "D_excess": spec["D_excess"].tolist(),
                })
                npz_u.append(u)
                npz_s.append(s)
                npz_p.append(p)
                npz_w.append(w)
                npz_obs.append(spec["P_obs"])
                npz_null.append(spec["P_null_mean"])
                npz_var.append(spec["P_null_var"])

        df_endpoints = pd.DataFrame(endpoint_rows)
        df_null_diag = pd.DataFrame(null_diag_rows)
        df_spectra_summary = pd.DataFrame(spectra_rows)
        arr_u = np.array(npz_u, dtype=np.int64) if len(npz_u) > 0 else np.array([], dtype=np.int64)
        arr_s = np.array(npz_s, dtype=np.int64) if len(npz_s) > 0 else np.array([], dtype=np.int64)
        arr_p = np.array(npz_p, dtype=object) if len(npz_p) > 0 else np.array([], dtype=object)
        arr_w = np.array(npz_w, dtype=np.int64) if len(npz_w) > 0 else np.array([], dtype=np.int64)
        arr_obs = np.stack(npz_obs) if len(npz_obs) > 0 else np.empty((0, len(BROAD_FREQS)), dtype=np.float32)
        arr_null = np.stack(npz_null) if len(npz_null) > 0 else np.empty((0, len(BROAD_FREQS)), dtype=np.float32)
        arr_var = np.stack(npz_var) if len(npz_var) > 0 else np.empty((0, len(BROAD_FREQS)), dtype=np.float32)

    # 6. Save Artifacts
    print("\n--- [Step 5/5] Writing Output Parquet and NPZ Artifacts ---")
    p_endpoints = ENDPOINTS_DIR / "endpoints.parquet"
    p_null_diag = ENDPOINTS_DIR / "null_diagnostics.parquet"
    p_spectra_summary = SPECTRA_DIR / "spectra_summary.parquet"
    p_spectra_broad_npz = SPECTRA_DIR / "spectra_broad.npz"

    df_endpoints.to_parquet(p_endpoints, index=False)
    print(f"  Saved endpoints.parquet ({len(df_endpoints)} rows): {p_endpoints}")

    df_null_diag.to_parquet(p_null_diag, index=False)
    print(f"  Saved null_diagnostics.parquet ({len(df_null_diag)} rows): {p_null_diag}")

    df_spectra_summary.to_parquet(p_spectra_summary, index=False)
    print(f"  Saved spectra_summary.parquet ({len(df_spectra_summary)} rows): {p_spectra_summary}")

    if len(arr_u) > 0:
        np.savez_compressed(
            p_spectra_broad_npz,
            user_id=arr_u,
            source=arr_s,
            protocol=np.array(arr_p, dtype=str),
            window_index=arr_w,
            broad_freqs=BROAD_FREQS,
            P_obs=arr_obs,
            P_null_mean=arr_null,
            P_null_var=arr_var,
        )
        print(f"  Saved spectra_broad.npz ({len(arr_u)} arrays): {p_spectra_broad_npz}")

    # Clean up checkpoint files if complete run
    if not args.limit and not args.source and not args.protocol:
        print("  Cleaning up temporary checkpoint chunks...")
        shutil.rmtree(CHECKPOINTS_DIR, ignore_errors=True)

    # Summary report
    print("\n================================================================================")
    print("MILESTONE 2 EXECUTION SUMMARY")
    print("================================================================================")
    print(f"Total endpoints generated: {len(df_endpoints)}")
    if len(df_endpoints) > 0:
        print("\nBreakdown by Source & Protocol (Window 1):")
        w1_df = df_endpoints[df_endpoints["window_index"] == 1]
        print(w1_df.groupby(["source", "protocol", "null_status"]).size().to_string())

        conv_w1 = w1_df[w1_df["null_status"] == "converged"]
        print(f"\nConverged Window 1 Physiological Metrics Summary (N = {len(conv_w1)}):")
        metrics_cols = [c for c in ["Q_i", "E_i2", "A_hat", "sigma_resid", "Q_i_mc_se", "v1", "phi1", "v2", "phi2"] if c in conv_w1.columns]
        print(conv_w1[metrics_cols].describe().round(4).to_string())

    print(f"\nTotal Pipeline Elapsed Time: {time.perf_counter() - t_start:.1f}s")
    print("================================================================================")


if __name__ == "__main__":
    main()
