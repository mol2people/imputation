#!/usr/bin/env python3
r"""
04_run_bootstrap_inference.py: Milestone 4 implementation for the Continuous Age RCS Infradian Study.

Executes:
1. Pure Non-parametric Stratified Bootstrap:
   - 2,000 stratified participant bootstrap resamples for primary configurations:
     Apple Watch (source 6) and Garmin Tier 1c (source 3), across expanded90 and balanced90,
     decomposing all 4 physiological endpoints (primary Q_i, and secondary E_i2, A_hat, sigma_resid).
   - Stratifies strictly by 5-year age band x sex, preserving exact demographic cell counts in every draw.
   - Deterministic PRNG seeding via PCG64 (seed = 20261020).
2. Continuous Curve Prediction & Uncertainty:
   - Evaluates predictions over continuous age grid a in [18, 80] with fine resolution (step 0.25 years, 249 grid points).
   - Computes pointwise 95% confidence intervals (empirical 2.5th and 97.5th percentiles of bootstrap replicates).
   - Computes simultaneous (uniform) 95% confidence bands over continuous age [18, 80] using maximum studentized absolute deviation:
     c_{0.95} = quantile_{0.95}( max_{a in [18, 80]} |theta^*(a) - theta(a)| / SE^*(a) ).
   - Computes empirical bootstrap p-values for all four Chunk Wald test statistics under HC3 sandwich covariance.
3. Three Physiological Transition Metrics (from delta curve Delta(a) = Q_female(a) - Q_male(a)):
   - Metric 1: Inflection point of steepest decline: argmin_{a in [35, 65]} dDelta / da (and slope at inflection).
   - Metric 2: Nominal zero-crossing: earliest age a in [40, 75] where Delta(a) <= 0.
   - Metric 3: Conservative statistical cessation boundary: earliest age >= 40 where upper 95% simultaneous bootstrap bound <= 0.
   - Extracts point estimate and 95% bootstrap confidence intervals for all three metrics.
4. Second-Window Repeatability:
   - For participants with two non-overlapping windows (Window 1 and Window 2, N=2,324 in Apple Watch balanced90):
     * Window 1 vs Window 2 correlation (Pearson r and Spearman rho) of Q_i, discrete peak period, and amplitude.
     * Frozen first-window harmonic prediction in Window 2 (retaining absolute calendar phase).
     * Repeatability stratified by age brackets (<40, 40-55, >=55) and sex.
5. Persists artifacts:
   - `artifacts/bootstrap/bootstrap_curves.parquet`
   - `artifacts/bootstrap/confidence_bands.parquet`
   - `artifacts/bootstrap/transition_metrics.csv`
   - `artifacts/bootstrap/bootstrap_chunk_pvalues.csv`
   - `artifacts/bootstrap/second_window_repeatability.parquet`
   - `artifacts/bootstrap/repeatability_summary.csv`
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator, PCG64
from scipy import stats

# Canonical repository and experiment paths
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]

# Add scripts directory to path
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import DEFAULT_KNOTS, rcs_stone_koo

# Paths to artifacts
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
ENDPOINTS_PARQUET = ARTIFACTS_DIR / "endpoints" / "endpoints.parquet"
WINDOWS_PARQUET = ARTIFACTS_DIR / "windows" / "windows.parquet"
DATES_PARQUET = ARTIFACTS_DIR / "windows" / "selected_dates.parquet"
MODELS_DIR = ARTIFACTS_DIR / "models"
SHARED_SUPPORT_PARQUET = MODELS_DIR / "shared_support_mask.parquet"
CHUNK_WALD_CSV = MODELS_DIR / "chunk_wald_tests.csv"
BOOTSTRAP_DIR = ARTIFACTS_DIR / "bootstrap"

UPSTREAM_APPLE_CACHE = REPO_ROOT / "experiments/apple_cohort_scaling/artifacts/cache_5min"
UPSTREAM_GARMIN_CACHE = REPO_ROOT / "experiments/garmin_infradian_replication/artifacts/cache_5min"

SOURCE_NAMES = {
    3: "Garmin Tier 1c",
    6: "Apple Watch",
    7: "Samsung",
}

ENDPOINTS = ["Q_i", "E_i2", "A_hat", "sigma_resid"]
PRIMARY_PROTOCOLS = ["expanded90", "balanced90"]
PRIMARY_SOURCES = [6, 3]  # Apple Watch and Garmin Tier 1c


# =============================================================================
# 1. Stone-Koo Basis & Analytical Derivatives
# =============================================================================
def d_rcs_stone_koo(
    ages: np.ndarray | Sequence[float] | float,
    knots: Sequence[float] = DEFAULT_KNOTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes analytical first derivatives d(rcs1)/da and d(rcs2)/da.
    """
    t1, t2, t3, t4 = knots
    norm = float((t4 - t1) ** 2)  # 1600.0
    x = np.asarray(ages, dtype=np.float64)
    scalar_input = x.ndim == 0
    if scalar_input:
        x = x[np.newaxis]

    def pos_sq(u: np.ndarray) -> np.ndarray:
        return 3.0 * (np.maximum(u, 0.0) ** 2)

    c3_1 = (t4 - t1) / (t4 - t3)  # 40/13
    c4_1 = (t3 - t1) / (t4 - t3)  # 27/13
    d_rcs1 = (pos_sq(x - t1) - c3_1 * pos_sq(x - t3) + c4_1 * pos_sq(x - t4)) / norm

    c3_2 = (t4 - t2) / (t4 - t3)  # 23/13
    c4_2 = (t3 - t2) / (t4 - t3)  # 10/13
    d_rcs2 = (pos_sq(x - t2) - c3_2 * pos_sq(x - t3) + c4_2 * pos_sq(x - t4)) / norm

    if scalar_input:
        return float(d_rcs1[0]), float(d_rcs2[0])
    return d_rcs1, d_rcs2


def build_design_matrix(
    ages: np.ndarray,
    sex: np.ndarray,
    knots: Sequence[float] = DEFAULT_KNOTS,
) -> np.ndarray:
    """
    Builds the 8-column primary unadjusted RCS design matrix:
      [1, age, rcs1, rcs2, sex, sex*age, sex*rcs1, sex*rcs2]
    """
    rcs1, rcs2 = rcs_stone_koo(ages, knots)
    col_1 = np.ones_like(ages, dtype=np.float64)
    col_age = ages.astype(np.float64)
    col_rcs1 = rcs1.astype(np.float64)
    col_rcs2 = rcs2.astype(np.float64)
    col_sex = sex.astype(np.float64)
    col_sex_age = col_sex * col_age
    col_sex_rcs1 = col_sex * col_rcs1
    col_sex_rcs2 = col_sex * col_rcs2

    return np.column_stack([
        col_1, col_age, col_rcs1, col_rcs2,
        col_sex, col_sex_age, col_sex_rcs1, col_sex_rcs2
    ])


def build_chunk_contrast_matrices() -> List[Tuple[int, str, str, np.ndarray]]:
    """
    Builds the 4 Chunk Wald contrast matrices (each size q x 8):
      Test 1 (Total female age effect, 3 d.f.): beta1+delta1=0, beta2+delta2=0, beta3+delta3=0
      Test 2 (Nonlinear female curvature, 2 d.f.): beta2+delta2=0, beta3+delta3=0
      Test 3 (Total Sex x Age interaction, 3 d.f.): delta1=0, delta2=0, delta3=0
      Test 4 (Nonlinear Sex x Age interaction, 2 d.f.): delta2=0, delta3=0
    """
    C1 = np.zeros((3, 8), dtype=np.float64)
    C1[0, 1] = 1.0; C1[0, 5] = 1.0
    C1[1, 2] = 1.0; C1[1, 6] = 1.0
    C1[2, 3] = 1.0; C1[2, 7] = 1.0

    C2 = np.zeros((2, 8), dtype=np.float64)
    C2[0, 2] = 1.0; C2[0, 6] = 1.0
    C2[1, 3] = 1.0; C2[1, 7] = 1.0

    C3 = np.zeros((3, 8), dtype=np.float64)
    C3[0, 5] = 1.0
    C3[1, 6] = 1.0
    C3[2, 7] = 1.0

    C4 = np.zeros((2, 8), dtype=np.float64)
    C4[0, 6] = 1.0
    C4[1, 7] = 1.0

    return [
        (1, "total_female_age", "beta_1 + delta_1 = 0, beta_2 + delta_2 = 0, beta_3 + delta_3 = 0", C1),
        (2, "nonlinear_female_curvature", "beta_2 + delta_2 = 0, beta_3 + delta_3 = 0", C2),
        (3, "total_sex_x_age_interaction", "delta_1 = 0, delta_2 = 0, delta_3 = 0", C3),
        (4, "nonlinear_sex_x_age_interaction", "delta_2 = 0, delta_3 = 0", C4),
    ]


# =============================================================================
# 2. Transition Metric Extraction Utilities
# =============================================================================
def extract_transition_metrics(
    delta_params: np.ndarray,
    grid_ages: np.ndarray,
    delta_pred: np.ndarray,
    upper_simult_band: Optional[np.ndarray] = None,
    fine_step: float = 0.05,
) -> Dict[str, Any]:
    """
    Computes the 3 physiological transition metrics from the delta curve parameters:
      delta(a) = delta_0 + delta_1*a + delta_2*rcs1(a) + delta_3*rcs2(a)
    1. Inflection point of steepest decline: argmin_{a in [35, 65]} dDelta / da
    2. Nominal zero-crossing: earliest age a in [40, 75] where Delta(a) <= 0
    3. Conservative statistical cessation boundary: earliest age a >= 40 where upper simultaneous bound <= 0
    """
    # 1. Steepest decline inflection point
    fine_grid = np.arange(35.0, 65.0 + fine_step, fine_step)
    d_rcs1, d_rcs2 = d_rcs_stone_koo(fine_grid)
    d_delta = delta_params[1] + delta_params[2] * d_rcs1 + delta_params[3] * d_rcs2
    idx_min = np.argmin(d_delta)
    age_inflection = float(fine_grid[idx_min])
    min_slope = float(d_delta[idx_min])
    is_inflection_boundary = bool(age_inflection <= 35.05 or age_inflection >= 64.95)

    # 2. Nominal zero-crossing in [40, 75]
    mask_zc = (grid_ages >= 40.0) & (grid_ages <= 75.0)
    zc_ages = grid_ages[mask_zc]
    zc_pred = delta_pred[mask_zc]

    hits = np.where(zc_pred <= 0.0)[0]
    if len(hits) > 0:
        first_hit = hits[0]
        if first_hit == 0:
            age_zero_crossing = float(zc_ages[0])
        else:
            # Linear interpolation of root
            a0, a1 = zc_ages[first_hit - 1], zc_ages[first_hit]
            y0, y1 = zc_pred[first_hit - 1], zc_pred[first_hit]
            if abs(y1 - y0) > 1e-12:
                age_zero_crossing = float(a0 - y0 * (a1 - a0) / (y1 - y0))
            else:
                age_zero_crossing = float(a0)
        zero_crossing_status = "observed"
    else:
        age_zero_crossing = np.nan
        zero_crossing_status = "not_reached_in_window"

    # 3. Conservative statistical cessation boundary (earliest age >= 40 where upper simultaneous bound <= 0)
    if upper_simult_band is not None:
        mask_cess = (grid_ages >= 40.0) & (upper_simult_band <= 0.0)
        if np.any(mask_cess):
            cess_ages = grid_ages[mask_cess]
            age_cessation = float(cess_ages[0])
            cessation_status = "observed"
        else:
            age_cessation = np.nan
            cessation_status = "not_reached_on_grid"
        min_upper_val = float(np.min(upper_simult_band[grid_ages >= 40.0]))
    else:
        age_cessation = np.nan
        cessation_status = "not_evaluated"
        min_upper_val = np.nan

    return {
        "inflection_age": age_inflection,
        "inflection_slope": min_slope,
        "is_inflection_boundary": is_inflection_boundary,
        "zero_crossing_age": age_zero_crossing,
        "zero_crossing_status": zero_crossing_status,
        "cessation_age": age_cessation,
        "cessation_status": cessation_status,
        "min_upper_simult_age40plus": min_upper_val,
    }


# =============================================================================
# 3. Stratified Bootstrap Resampling & Simultaneous Confidence Bands Engine
# =============================================================================
def run_stratified_bootstrap_for_config(
    df_config: pd.DataFrame,
    source: int,
    protocol: str,
    grid_ages: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20261020,
) -> Dict[str, Any]:
    """
    Executes 2,000 stratified participant bootstrap replicates for a single (source, protocol) dataset.
    Stratifies strictly by 5-year age band x sex.
    Decomposes all 4 physiological endpoints (Q_i, E_i2, A_hat, sigma_resid) simultaneously.
    """
    t_start = time.perf_counter()
    n_obs = len(df_config)
    ages = df_config["age"].to_numpy(dtype=np.float64)
    sex = df_config["sex"].to_numpy(dtype=np.float64)

    # 1. Precompute design matrix X (N x 8)
    X = build_design_matrix(ages, sex)
    Y = df_config[ENDPOINTS].to_numpy(dtype=np.float64)  # (N x 4)

    # 2. Grid design matrices
    g_rcs1, g_rcs2 = rcs_stone_koo(grid_ages)
    n_grid = len(grid_ages)
    X_grid_male = np.column_stack([np.ones(n_grid), grid_ages, g_rcs1, g_rcs2])
    X_grid_female = np.column_stack([np.ones(n_grid), grid_ages, g_rcs1, g_rcs2])
    X_grid_delta = np.column_stack([np.ones(n_grid), grid_ages, g_rcs1, g_rcs2])

    # 3. Sample OLS fits and HC3 covariances
    inv_XtX = np.linalg.inv(X.T @ X)
    theta_sample = inv_XtX @ (X.T @ Y)  # (8 x 4)

    # Sample predictions
    # Female: theta[0:4] + theta[4:8]
    # Male: theta[0:4]
    # Delta: theta[4:8]
    sample_female_pred = X_grid_female @ (theta_sample[0:4, :] + theta_sample[4:8, :])  # (249 x 4)
    sample_male_pred = X_grid_male @ theta_sample[0:4, :]  # (249 x 4)
    sample_delta_pred = X_grid_delta @ theta_sample[4:8, :]  # (249 x 4)

    # Hat matrix diagonal for sample HC3
    h_sample = np.sum(X * (X @ inv_XtX), axis=1)
    e_sample = Y - X @ theta_sample  # (N x 4)
    r_sample = e_sample / (1.0 - h_sample[:, None])

    # Sample HC3 covariances and Chunk Wald test statistics for all 4 endpoints
    contrasts = build_chunk_contrast_matrices()
    sample_wald_stats = np.empty((4, len(contrasts)), dtype=np.float64)  # (4 endpoints x 4 tests)

    for ep_idx in range(4):
        r_ep = r_sample[:, ep_idx]
        meat_ep = (X * r_ep[:, None]).T @ (X * r_ep[:, None])
        V_hc3_ep = inv_XtX @ meat_ep @ inv_XtX
        th_ep = theta_sample[:, ep_idx]
        for t_idx, (t_num, t_name, t_null, C) in enumerate(contrasts):
            C_th = C @ th_ep
            middle_inv = np.linalg.pinv(C @ V_hc3_ep @ C.T)
            w_stat = float(C_th.T @ middle_inv @ C_th)
            sample_wald_stats[ep_idx, t_idx] = max(0.0, w_stat)

    # 4. Strata indices: 5-year age band x sex
    bins = np.arange(0, 130, 5)
    band_idx = np.digitize(ages, bins)
    strata = band_idx * 2 + sex.astype(np.int64)
    unique_strata = np.unique(strata)
    stratum_indices = [np.where(strata == s)[0] for s in unique_strata]
    stratum_sizes = [len(idx) for idx in stratum_indices]

    # 5. Bootstrap resampling loop
    rng = Generator(PCG64(seed))

    # Allocation arrays for replicates
    # Shape: (n_boot, n_grid, 4 endpoints)
    boot_female_curves = np.empty((n_boot, n_grid, 4), dtype=np.float32)
    boot_male_curves = np.empty((n_boot, n_grid, 4), dtype=np.float32)
    boot_delta_curves = np.empty((n_boot, n_grid, 4), dtype=np.float32)

    # Bootstrap Wald statistics for empirical p-values: (n_boot, 4 endpoints, 4 tests)
    boot_wald_stats = np.empty((n_boot, 4, len(contrasts)), dtype=np.float64)

    # Bootstrap delta parameters for transition metric CIs: (n_boot, 4 endpoints, 4 params)
    boot_delta_thetas = np.empty((n_boot, 4, 4), dtype=np.float64)

    chosen_idx = np.empty(n_obs, dtype=np.int64)

    for b in range(n_boot):
        # Draw stratified sample preserving exact stratum counts
        pos = 0
        for idx_arr, sz in zip(stratum_indices, stratum_sizes):
            chosen_idx[pos : pos + sz] = rng.choice(idx_arr, size=sz, replace=True)
            pos += sz

        X_b = X[chosen_idx]
        Y_b = Y[chosen_idx]

        # Fast OLS solve for 4 endpoints simultaneously
        XtX_b = X_b.T @ X_b
        inv_XtX_b = np.linalg.inv(XtX_b)
        theta_b = inv_XtX_b @ (X_b.T @ Y_b)  # (8 x 4)

        # Predict continuous curves
        fem_b = X_grid_female @ (theta_b[0:4, :] + theta_b[4:8, :])
        male_b = X_grid_male @ theta_b[0:4, :]
        delta_b = X_grid_delta @ theta_b[4:8, :]

        boot_female_curves[b] = fem_b
        boot_male_curves[b] = male_b
        boot_delta_curves[b] = delta_b
        boot_delta_thetas[b] = theta_b[4:8, :].T  # (4 endpoints x 4 params)

        # HC3 Wald tests for empirical p-values
        h_b = np.sum(X_b * (X_b @ inv_XtX_b), axis=1)
        e_b = Y_b - X_b @ theta_b
        r_b = e_b / (1.0 - h_b[:, None])

        for ep_idx in range(4):
            r_ep_b = r_b[:, ep_idx]
            meat_ep_b = (X_b * r_ep_b[:, None]).T @ (X_b * r_ep_b[:, None])
            V_hc3_ep_b = inv_XtX_b @ meat_ep_b @ inv_XtX_b
            # Centered restriction: C @ (theta_b - theta_sample)
            d_th = theta_b[:, ep_idx] - theta_sample[:, ep_idx]
            for t_idx, (t_num, t_name, t_null, C) in enumerate(contrasts):
                C_d = C @ d_th
                m_inv = np.linalg.pinv(C @ V_hc3_ep_b @ C.T)
                boot_w = float(C_d.T @ m_inv @ C_d)
                boot_wald_stats[b, ep_idx, t_idx] = max(0.0, boot_w)

    t_elapsed = time.perf_counter() - t_start

    # 6. Compute Pointwise CIs, Studentized Sup-Norm, and Simultaneous Bands
    # Results per endpoint and curve type
    confidence_band_records = []
    pvalue_records = []
    transition_metric_records = []

    for ep_idx, ep_name in enumerate(ENDPOINTS):
        # Wald test empirical bootstrap p-values
        for t_idx, (t_num, t_name, t_null, C) in enumerate(contrasts):
            w_obs = sample_wald_stats[ep_idx, t_idx]
            boot_w_dist = boot_wald_stats[:, ep_idx, t_idx]
            p_boot = float((1.0 + np.sum(boot_w_dist >= w_obs)) / (n_boot + 1.0))
            q_df = C.shape[0]
            f_stat = w_obs / q_df
            df_resid = n_obs - 8
            p_chi2 = float(stats.chi2.sf(w_obs, df=q_df))
            p_f = float(stats.f.sf(f_stat, dfn=q_df, dfd=df_resid))

            pvalue_records.append({
                "source": source,
                "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
                "protocol": protocol,
                "endpoint": ep_name,
                "test_num": t_num,
                "test_name": t_name,
                "null_hypothesis": t_null,
                "df_test": q_df,
                "df_resid": df_resid,
                "wald_stat_hc3": float(w_obs),
                "f_stat_hc3": float(f_stat),
                "asymptotic_p_chi2": p_chi2,
                "asymptotic_p_f": p_f,
                "bootstrap_p_value": p_boot,
                "n_boot": n_boot,
            })

        # Process each curve type: delta, female, male
        curves_dict = {
            "delta": (sample_delta_pred[:, ep_idx], boot_delta_curves[:, :, ep_idx]),
            "female": (sample_female_pred[:, ep_idx], boot_female_curves[:, :, ep_idx]),
            "male": (sample_male_pred[:, ep_idx], boot_male_curves[:, :, ep_idx]),
        }

        simult_upper_delta = None

        for c_type, (sample_pred, boot_arr) in curves_dict.items():
            # Pointwise statistics
            boot_mean = np.mean(boot_arr, axis=0)
            boot_se = np.std(boot_arr, axis=0, ddof=1)
            boot_se = np.maximum(boot_se, 1e-12)

            ci_lower = np.percentile(boot_arr, 2.5, axis=0)
            ci_upper = np.percentile(boot_arr, 97.5, axis=0)

            # Studentized sup-norm over continuous grid [18, 80]
            # D^(b) = max_{a in [18, 80]} |theta^*(a) - theta(a)| / SE^*(a)
            abs_devs = np.abs(boot_arr - sample_pred[None, :]) / boot_se[None, :]
            sup_devs = np.max(abs_devs, axis=1)  # (n_boot,)
            c_095 = float(np.percentile(sup_devs, 95.0))

            band_lower = sample_pred - c_095 * boot_se
            band_upper = sample_pred + c_095 * boot_se

            if c_type == "delta":
                simult_upper_delta = band_upper

            for a_idx, a_val in enumerate(grid_ages):
                confidence_band_records.append({
                    "source": source,
                    "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
                    "protocol": protocol,
                    "endpoint": ep_name,
                    "curve_type": c_type,
                    "age": float(a_val),
                    "point_estimate": float(sample_pred[a_idx]),
                    "boot_mean": float(boot_mean[a_idx]),
                    "boot_se": float(boot_se[a_idx]),
                    "ci_lower_pointwise": float(ci_lower[a_idx]),
                    "ci_upper_pointwise": float(ci_upper[a_idx]),
                    "simult_c_095": c_095,
                    "band_lower_simultaneous": float(band_lower[a_idx]),
                    "band_upper_simultaneous": float(band_upper[a_idx]),
                })

        # 7. Transition Metrics for Delta Curve
        sample_delta_th = theta_sample[4:8, ep_idx]
        sample_metrics = extract_transition_metrics(
            sample_delta_th,
            grid_ages,
            sample_delta_pred[:, ep_idx],
            upper_simult_band=simult_upper_delta,
        )

        # Bootstrap replicates for transition metrics
        boot_inf_ages = np.empty(n_boot, dtype=np.float64)
        boot_inf_slopes = np.empty(n_boot, dtype=np.float64)
        boot_zc_ages = np.empty(n_boot, dtype=np.float64)

        for b in range(n_boot):
            th_b_delta = boot_delta_thetas[b, ep_idx, :]
            pred_b_delta = boot_delta_curves[b, :, ep_idx]
            b_m = extract_transition_metrics(th_b_delta, grid_ages, pred_b_delta, upper_simult_band=None)
            boot_inf_ages[b] = b_m["inflection_age"]
            boot_inf_slopes[b] = b_m["inflection_slope"]
            boot_zc_ages[b] = b_m["zero_crossing_age"]

        # Metric 1: Inflection point of steepest decline
        inf_ci_l = float(np.percentile(boot_inf_ages, 2.5))
        inf_ci_u = float(np.percentile(boot_inf_ages, 97.5))
        slope_ci_l = float(np.percentile(boot_inf_slopes, 2.5))
        slope_ci_u = float(np.percentile(boot_inf_slopes, 97.5))

        transition_metric_records.append({
            "source": source,
            "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
            "protocol": protocol,
            "endpoint": ep_name,
            "metric_id": 1,
            "metric_name": "steepest_decline_inflection_age",
            "units": "years",
            "point_estimate": sample_metrics["inflection_age"],
            "boot_mean": float(np.mean(boot_inf_ages)),
            "boot_se": float(np.std(boot_inf_ages, ddof=1)),
            "ci_lower_95": inf_ci_l,
            "ci_upper_95": inf_ci_u,
            "status": "boundary_hit" if sample_metrics["is_inflection_boundary"] else "interior_minimum",
            "notes": f"Slope at inflection: {sample_metrics['inflection_slope']:.6f} (95% CI: [{slope_ci_l:.6f}, {slope_ci_u:.6f}])",
        })

        transition_metric_records.append({
            "source": source,
            "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
            "protocol": protocol,
            "endpoint": ep_name,
            "metric_id": 1,
            "metric_name": "steepest_decline_slope",
            "units": "units/year",
            "point_estimate": sample_metrics["inflection_slope"],
            "boot_mean": float(np.mean(boot_inf_slopes)),
            "boot_se": float(np.std(boot_inf_slopes, ddof=1)),
            "ci_lower_95": slope_ci_l,
            "ci_upper_95": slope_ci_u,
            "status": "valid",
            "notes": "Minimum first derivative on [35, 65]",
        })

        # Metric 2: Nominal zero-crossing
        valid_zc = boot_zc_ages[np.isfinite(boot_zc_ages)]
        if len(valid_zc) > 0.5 * n_boot:
            zc_ci_l = float(np.percentile(valid_zc, 2.5))
            zc_ci_u = float(np.percentile(valid_zc, 97.5))
            zc_mean = float(np.mean(valid_zc))
            zc_se = float(np.std(valid_zc, ddof=1))
        else:
            zc_ci_l = np.nan
            zc_ci_u = np.nan
            zc_mean = np.nan
            zc_se = np.nan

        transition_metric_records.append({
            "source": source,
            "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
            "protocol": protocol,
            "endpoint": ep_name,
            "metric_id": 2,
            "metric_name": "nominal_zero_crossing_age",
            "units": "years",
            "point_estimate": sample_metrics["zero_crossing_age"],
            "boot_mean": zc_mean,
            "boot_se": zc_se,
            "ci_lower_95": zc_ci_l,
            "ci_upper_95": zc_ci_u,
            "status": sample_metrics["zero_crossing_status"],
            "notes": f"Valid bootstrap zero-crossings in {len(valid_zc)}/{n_boot} replicates",
        })

        # Metric 3: Conservative statistical cessation boundary
        transition_metric_records.append({
            "source": source,
            "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
            "protocol": protocol,
            "endpoint": ep_name,
            "metric_id": 3,
            "metric_name": "conservative_cessation_age",
            "units": "years",
            "point_estimate": sample_metrics["cessation_age"],
            "boot_mean": np.nan,
            "boot_se": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "status": sample_metrics["cessation_status"],
            "notes": f"Minimum upper simultaneous band value on age>=40: {sample_metrics['min_upper_simult_age40plus']:.6f}",
        })

    print(f"  [{SOURCE_NAMES.get(source)} {protocol}] {n_boot} bootstrap resamples completed in {t_elapsed:.2f} s")

    return {
        "confidence_bands": confidence_band_records,
        "pvalues": pvalue_records,
        "transition_metrics": transition_metric_records,
        "boot_female_curves": boot_female_curves,
        "boot_male_curves": boot_male_curves,
        "boot_delta_curves": boot_delta_curves,
    }


# =============================================================================
# 4. Second-Window Repeatability Engine
# =============================================================================
def compute_second_window_repeatability(
    df_endpoints: pd.DataFrame,
    df_dates: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes second-window repeatability across participants with two non-overlapping windows:
      - Endpoint correlations (Q_i, discrete peak period, cosinor amplitude A_hat)
      - Frozen first-window harmonic prediction in Window 2 (retaining absolute calendar phase)
      - Repeatability stratified by age brackets (<40, 40-55, >=55) and sex
    """
    print("\nComputing Second-Window Repeatability...")
    t0 = time.perf_counter()

    # Pre-index dates by (user_id, source, protocol, window_index)
    dates_grouped = df_dates.groupby(["user_id", "source", "protocol", "window_index"])["date"].apply(np.sort).to_dict()

    participant_rows = []

    # Iterate over primary sources (Apple Watch source 6, Garmin Tier 1c source 3)
    for src in [6, 3]:
        for proto in ["balanced90", "expanded90"]:
            sub_end = df_endpoints[
                (df_endpoints["source"] == src)
                & (df_endpoints["protocol"] == proto)
                & (df_endpoints["null_status"] == "converged")
            ].copy()

            w1_end = sub_end[sub_end["window_index"] == 1].set_index("user_id")
            w2_end = sub_end[sub_end["window_index"] == 2].set_index("user_id")
            common_ids = sorted(list(set(w1_end.index).intersection(set(w2_end.index))))

            print(f"  Source {src} ({proto}): {len(common_ids)} two-window eligible participants")

            cache_dir = UPSTREAM_APPLE_CACHE if src == 6 else UPSTREAM_GARMIN_CACHE

            for u in common_ids:
                row_w1 = w1_end.loc[u]
                row_w2 = w2_end.loc[u]

                # Demographic and basic info
                sex = int(row_w1["sex"])
                salutation = int(row_w1["salutation"])
                age_w1 = float(row_w1["age"])
                age_w2 = float(row_w2["age"])
                brk = "<40" if age_w1 < 40.0 else ("40-55" if age_w1 < 55.0 else ">=55")

                q1 = float(row_w1["Q_i"])
                q2 = float(row_w2["Q_i"])
                t1 = float(row_w1["peak_period"])
                t2 = float(row_w2["peak_period"])
                a1 = float(row_w1["A_hat"])
                a2 = float(row_w2["A_hat"])
                sig1 = float(row_w1["sigma_resid"])
                sig2 = float(row_w2["sigma_resid"])

                start_day_w1 = int(row_w1["start_day"])
                start_day_w2 = int(row_w2["start_day"])
                cal_gap = start_day_w2 - start_day_w1

                # Load cache for frozen harmonic prediction
                cache_file = cache_dir / f"customer_{u}.npz"
                r_frozen = np.nan
                r2_frozen = np.nan
                bc1 = np.nan
                bs1 = np.nan

                if cache_file.exists():
                    try:
                        with np.load(cache_file) as d:
                            hr_cache = d["hr"]
                            dates_cache = d["date"]

                        date_to_row = {int(dv): i for i, dv in enumerate(dates_cache)}

                        # Window 1 dates & HR
                        d1 = dates_grouped.get((u, src, proto, 1))
                        d2 = dates_grouped.get((u, src, proto, 2))

                        if d1 is not None and d2 is not None:
                            idx1 = [date_to_row[int(dv)] for dv in d1]
                            hr1 = hr_cache[idx1]
                            mask1 = np.isfinite(hr1)
                            hr_p10_1 = np.array([np.percentile(row[m], 10.0) for row, m in zip(hr1, mask1)])

                            # Nuisance baseline for Window 1
                            n1 = len(d1)
                            cols1 = [np.ones(n1, dtype=np.float64), (d1 - d1[0]).astype(np.float64)]
                            dow1 = (d1 + 3) % 7
                            for d_idx in range(1, 7):
                                cols1.append((dow1 == d_idx).astype(np.float64))
                            clock_f1 = np.column_stack([
                                np.mean(mask1[:, 0:72], axis=1),
                                np.mean(mask1[:, 72:144], axis=1),
                                np.mean(mask1[:, 144:216], axis=1),
                                np.mean(mask1[:, 216:288], axis=1),
                            ])
                            for c in range(4):
                                cols1.append(clock_f1[:, c].astype(np.float64))

                            Q1_qr, _ = np.linalg.qr(np.column_stack(cols1))
                            P0_1 = np.eye(n1) - Q1_qr @ Q1_qr.T
                            e0_1 = P0_1 @ hr_p10_1

                            # Fit harmonic in Window 1 at selected peak period T1
                            f1 = 1.0 / t1
                            omega1 = 2.0 * np.pi * f1
                            t_el1 = (d1 - d1[0]).astype(np.float64)
                            C1 = np.column_stack([np.cos(omega1 * t_el1), np.sin(omega1 * t_el1)])
                            C1_tilde = P0_1 @ C1
                            beta_harm = np.linalg.solve(C1_tilde.T @ C1_tilde, C1_tilde.T @ e0_1)
                            bc1, bs1 = float(beta_harm[0]), float(beta_harm[1])

                            # Window 2 dates & HR
                            idx2 = [date_to_row[int(dv)] for dv in d2]
                            hr2 = hr_cache[idx2]
                            mask2 = np.isfinite(hr2)
                            hr_p10_2 = np.array([np.percentile(row[m], 10.0) for row, m in zip(hr2, mask2)])

                            # Nuisance baseline for Window 2
                            n2 = len(d2)
                            cols2 = [np.ones(n2, dtype=np.float64), (d2 - d2[0]).astype(np.float64)]
                            dow2 = (d2 + 3) % 7
                            for d_idx in range(1, 7):
                                cols2.append((dow2 == d_idx).astype(np.float64))
                            clock_f2 = np.column_stack([
                                np.mean(mask2[:, 0:72], axis=1),
                                np.mean(mask2[:, 72:144], axis=1),
                                np.mean(mask2[:, 144:216], axis=1),
                                np.mean(mask2[:, 216:288], axis=1),
                            ])
                            for c in range(4):
                                cols2.append(clock_f2[:, c].astype(np.float64))

                            Q2_qr, _ = np.linalg.qr(np.column_stack(cols2))
                            P0_2 = np.eye(n2) - Q2_qr @ Q2_qr.T
                            e0_2 = P0_2 @ hr_p10_2

                            # Frozen harmonic prediction: t - d1[0] retains absolute calendar phase
                            t_el2 = (d2 - d1[0]).astype(np.float64)
                            h_frozen = bc1 * np.cos(omega1 * t_el2) + bs1 * np.sin(omega1 * t_el2)
                            h_frozen_proj = P0_2 @ h_frozen

                            # Correlation and R^2
                            if np.std(h_frozen_proj) > 1e-12 and np.std(e0_2) > 1e-12:
                                r_corr, _ = stats.pearsonr(e0_2, h_frozen_proj)
                                r_frozen = float(r_corr)
                                ss_tot = float(np.sum(e0_2 ** 2))
                                ss_res = float(np.sum((e0_2 - h_frozen_proj) ** 2))
                                r2_frozen = float(1.0 - ss_res / ss_tot)
                    except Exception as err:
                        pass

                participant_rows.append({
                    "user_id": int(u),
                    "source": int(src),
                    "source_name": SOURCE_NAMES.get(src, f"Source {src}"),
                    "protocol": str(proto),
                    "salutation": salutation,
                    "sex": sex,
                    "age_w1": age_w1,
                    "age_w2": age_w2,
                    "age_bracket": brk,
                    "w1_start_day": start_day_w1,
                    "w2_start_day": start_day_w2,
                    "calendar_gap_days": cal_gap,
                    "w1_Q_i": q1,
                    "w2_Q_i": q2,
                    "delta_Q_i": q2 - q1,
                    "w1_peak_period": t1,
                    "w2_peak_period": t2,
                    "delta_peak_period": t2 - t1,
                    "w1_A_hat": a1,
                    "w2_A_hat": a2,
                    "delta_A_hat": a2 - a1,
                    "w1_sigma_resid": sig1,
                    "w2_sigma_resid": sig2,
                    "w1_beta_cos": bc1,
                    "w1_beta_sin": bs1,
                    "forecast_corr_r": r_frozen,
                    "forecast_r2": r2_frozen,
                })

    df_part_rep = pd.DataFrame(participant_rows)
    t_elapsed = time.perf_counter() - t0
    print(f"  Second-window repeatability completed for {len(df_part_rep)} participant records in {t_elapsed:.2f} s")

    # Aggregate repeatability summary table
    summary_rows = []
    for (src, proto), grp in df_part_rep.groupby(["source", "protocol"]):
        # Overall
        for sex_filter, sex_name in [(None, "All"), (1, "Female"), (0, "Male")]:
            sub_grp = grp if sex_filter is None else grp[grp["sex"] == sex_filter]
            if len(sub_grp) < 10:
                continue

            r_q, p_q = stats.pearsonr(sub_grp["w1_Q_i"], sub_grp["w2_Q_i"])
            rho_q, _ = stats.spearmanr(sub_grp["w1_Q_i"], sub_grp["w2_Q_i"])

            r_t, p_t = stats.pearsonr(sub_grp["w1_peak_period"], sub_grp["w2_peak_period"])
            rho_t, _ = stats.spearmanr(sub_grp["w1_peak_period"], sub_grp["w2_peak_period"])

            r_a, p_a = stats.pearsonr(sub_grp["w1_A_hat"], sub_grp["w2_A_hat"])
            rho_a, _ = stats.spearmanr(sub_grp["w1_A_hat"], sub_grp["w2_A_hat"])

            fc_valid = sub_grp["forecast_corr_r"].dropna()
            r2_valid = sub_grp["forecast_r2"].dropna()

            summary_rows.append({
                "source": src,
                "source_name": SOURCE_NAMES.get(src, f"Source {src}"),
                "protocol": proto,
                "sex": sex_name,
                "age_bracket": "All",
                "n_participants": len(sub_grp),
                "pearson_r_Q_i": float(r_q),
                "p_val_Q_i": float(p_q),
                "spearman_rho_Q_i": float(rho_q),
                "pearson_r_peak_period": float(r_t),
                "spearman_rho_peak_period": float(rho_t),
                "pearson_r_A_hat": float(r_a),
                "spearman_rho_A_hat": float(rho_a),
                "mean_forecast_r": float(fc_valid.mean()) if len(fc_valid) > 0 else np.nan,
                "median_forecast_r": float(fc_valid.median()) if len(fc_valid) > 0 else np.nan,
                "frac_positive_forecast_r": float((fc_valid > 0).mean()) if len(fc_valid) > 0 else np.nan,
                "mean_forecast_r2": float(r2_valid.mean()) if len(r2_valid) > 0 else np.nan,
            })

            # By age bracket
            for brk in ["<40", "40-55", ">=55"]:
                sub_brk = sub_grp[sub_grp["age_bracket"] == brk]
                if len(sub_brk) < 10:
                    continue

                r_qb, p_qb = stats.pearsonr(sub_brk["w1_Q_i"], sub_brk["w2_Q_i"])
                rho_qb, _ = stats.spearmanr(sub_brk["w1_Q_i"], sub_brk["w2_Q_i"])

                r_tb, _ = stats.pearsonr(sub_brk["w1_peak_period"], sub_brk["w2_peak_period"])
                rho_tb, _ = stats.spearmanr(sub_brk["w1_peak_period"], sub_brk["w2_peak_period"])

                r_ab, _ = stats.pearsonr(sub_brk["w1_A_hat"], sub_brk["w2_A_hat"])
                rho_ab, _ = stats.spearmanr(sub_brk["w1_A_hat"], sub_brk["w2_A_hat"])

                fc_v_brk = sub_brk["forecast_corr_r"].dropna()
                r2_v_brk = sub_brk["forecast_r2"].dropna()

                summary_rows.append({
                    "source": src,
                    "source_name": SOURCE_NAMES.get(src, f"Source {src}"),
                    "protocol": proto,
                    "sex": sex_name,
                    "age_bracket": brk,
                    "n_participants": len(sub_brk),
                    "pearson_r_Q_i": float(r_qb),
                    "p_val_Q_i": float(p_qb),
                    "spearman_rho_Q_i": float(rho_qb),
                    "pearson_r_peak_period": float(r_tb),
                    "spearman_rho_peak_period": float(rho_tb),
                    "pearson_r_A_hat": float(r_ab),
                    "spearman_rho_A_hat": float(rho_ab),
                    "mean_forecast_r": float(fc_v_brk.mean()) if len(fc_v_brk) > 0 else np.nan,
                    "median_forecast_r": float(fc_v_brk.median()) if len(fc_v_brk) > 0 else np.nan,
                    "frac_positive_forecast_r": float((fc_v_brk > 0).mean()) if len(fc_v_brk) > 0 else np.nan,
                    "mean_forecast_r2": float(r2_v_brk.mean()) if len(r2_v_brk) > 0 else np.nan,
                })

    df_summary = pd.DataFrame(summary_rows)
    return df_part_rep, df_summary


# =============================================================================
# 5. Main Production Execution Orchestrator
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run Milestone 4 Bootstrap Inference and Repeatability.")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap resamples (default 2000).")
    parser.add_argument("--seed", type=int, default=20261020, help="PRNG deterministic seed (default 20261020).")
    parser.add_argument("--skip-repeatability", action="store_true", help="Skip second-window repeatability calculation.")
    args = parser.parse_args()

    print("================================================================================")
    print("Milestone 4: Continuous Age RCS Bootstrap Inference & Second-Window Repeatability")
    print(f"  Bootstrap Resamples: {args.n_boot}")
    print(f"  PRNG PCG64 Seed: {args.seed}")
    print("================================================================================")

    # Ensure output directory exists
    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load endpoints and shared support mask
    print("\n[Step 1/5] Loading Input Artifacts...")
    df_end = pd.read_parquet(ENDPOINTS_PARQUET)
    df_mask = pd.read_parquet(SHARED_SUPPORT_PARQUET)
    df_dates = pd.read_parquet(DATES_PARQUET)

    # Primary analysis population: Window 1 with converged nulls
    df_w1 = df_end[(df_end["window_index"] == 1) & (df_end["null_status"] == "converged")].copy()
    mask_w1 = df_mask[df_mask["window_index"] == 1][["user_id", "source", "protocol", "in_shared_support"]].drop_duplicates()
    df_w1 = df_w1.merge(mask_w1, on=["user_id", "source", "protocol"], how="inner")

    # Continuous age grid: [18, 80], step 0.25 (249 points)
    grid_ages = np.arange(18.0, 80.0 + 0.25, 0.25)
    print(f"  Continuous age grid: {len(grid_ages)} points from {grid_ages[0]:.2f} to {grid_ages[-1]:.2f} years")

    # 2. Run Stratified Bootstrap across Primary Configurations
    print("\n[Step 2/5] Running 2,000 Stratified Participant Bootstrap Resamples...")
    all_band_records = []
    all_pvalue_records = []
    all_transition_records = []
    curve_records = []

    config_idx = 0
    total_configs = len(PRIMARY_SOURCES) * len(PRIMARY_PROTOCOLS)

    for src in PRIMARY_SOURCES:
        for proto in PRIMARY_PROTOCOLS:
            config_idx += 1
            src_name = SOURCE_NAMES.get(src, f"Source {src}")
            print(f"\nConfiguration [{config_idx}/{total_configs}]: {src_name} ({proto})")

            # Filter shared support participants
            df_config = df_w1[
                (df_w1["source"] == src)
                & (df_w1["protocol"] == proto)
                & (df_w1["in_shared_support"])
            ].copy().reset_index(drop=True)

            n_total = len(df_config)
            n_fem = int((df_config["sex"] == 1).sum())
            n_male = int((df_config["sex"] == 0).sum())
            print(f"  Shared support sample size: N={n_total:,} (Females={n_fem:,}, Males={n_male:,})")

            # Execute stratified bootstrap
            boot_res = run_stratified_bootstrap_for_config(
                df_config=df_config,
                source=src,
                protocol=proto,
                grid_ages=grid_ages,
                n_boot=args.n_boot,
                seed=args.seed + config_idx * 100,
            )

            all_band_records.extend(boot_res["confidence_bands"])
            all_pvalue_records.extend(boot_res["pvalues"])
            all_transition_records.extend(boot_res["transition_metrics"])

            # Save individual curves per replicate for bootstrap_curves.parquet
            # 2000 replicates x 4 endpoints
            for rep in range(args.n_boot):
                for ep_idx, ep_name in enumerate(ENDPOINTS):
                    curve_records.append({
                        "replicate": rep,
                        "source": src,
                        "source_name": src_name,
                        "protocol": proto,
                        "endpoint": ep_name,
                        "female_curve": boot_res["boot_female_curves"][rep, :, ep_idx].astype(np.float32),
                        "male_curve": boot_res["boot_male_curves"][rep, :, ep_idx].astype(np.float32),
                        "delta_curve": boot_res["boot_delta_curves"][rep, :, ep_idx].astype(np.float32),
                    })

    # 3. Second-Window Repeatability
    if not args.skip_repeatability:
        print("\n[Step 3/5] Evaluating Second-Window Repeatability...")
        df_part_rep, df_rep_summary = compute_second_window_repeatability(df_end, df_dates)
    else:
        print("\n[Step 3/5] Skipping Second-Window Repeatability (--skip-repeatability requested).")
        df_part_rep = pd.DataFrame()
        df_rep_summary = pd.DataFrame()

    # 4. Consolidate and Persist Deliverables
    print("\n[Step 4/5] Persisting Bootstrap Artifacts...")

    # A. Confidence Bands Parquet
    df_bands = pd.DataFrame(all_band_records)
    bands_path = BOOTSTRAP_DIR / "confidence_bands.parquet"
    df_bands.to_parquet(bands_path, compression="snappy", index=False)
    print(f"  Saved confidence bands: {bands_path} ({len(df_bands):,} rows, {bands_path.stat().st_size / 1e6:.2f} MB)")

    # B. Bootstrap Curves Parquet
    df_curves = pd.DataFrame(curve_records)
    curves_path = BOOTSTRAP_DIR / "bootstrap_curves.parquet"
    df_curves.to_parquet(curves_path, compression="snappy", index=False)
    print(f"  Saved bootstrap curves: {curves_path} ({len(df_curves):,} rows, {curves_path.stat().st_size / 1e6:.2f} MB)")

    # C. Transition Metrics CSV
    df_trans = pd.DataFrame(all_transition_records)
    trans_path = BOOTSTRAP_DIR / "transition_metrics.csv"
    df_trans.to_csv(trans_path, index=False)
    print(f"  Saved transition metrics: {trans_path} ({len(df_trans):,} rows)")

    # D. Bootstrap Chunk P-Values CSV
    df_pvals = pd.DataFrame(all_pvalue_records)
    pvals_path = BOOTSTRAP_DIR / "bootstrap_chunk_pvalues.csv"
    df_pvals.to_csv(pvals_path, index=False)
    print(f"  Saved bootstrap Chunk Wald p-values: {pvals_path} ({len(df_pvals):,} tests)")

    # E. Second-Window Repeatability Parquet & Summary CSV
    if not df_part_rep.empty:
        rep_path = BOOTSTRAP_DIR / "second_window_repeatability.parquet"
        df_part_rep.to_parquet(rep_path, compression="snappy", index=False)
        print(f"  Saved second-window repeatability: {rep_path} ({len(df_part_rep):,} rows, {rep_path.stat().st_size / 1e6:.2f} MB)")

        rep_sum_path = BOOTSTRAP_DIR / "repeatability_summary.csv"
        df_rep_summary.to_csv(rep_sum_path, index=False)
        print(f"  Saved repeatability summary: {rep_sum_path} ({len(df_rep_summary):,} rows)")

    # 5. Print Executive Summary
    print("\n[Step 5/5] Executive Statistical Summary")
    print("--------------------------------------------------------------------------------")
    print("A. Transition Metrics for Primary Q_i Delta Curve (Female - Male):")
    primary_trans = df_trans[(df_trans["endpoint"] == "Q_i") & (df_trans["protocol"] == "expanded90")]
    for _, r in primary_trans.iterrows():
        est_str = f"{r['point_estimate']:.2f}" if np.isfinite(r['point_estimate']) else "None"
        ci_str = f"[{r['ci_lower_95']:.2f}, {r['ci_upper_95']:.2f}]" if np.isfinite(r['ci_lower_95']) else "[N/A]"
        print(f"  {r['source_name']:16s} | {r['metric_name']:32s} | Est: {est_str:>6s} {r['units']:<10s} | 95% CI: {ci_str:<16s} | Status: {r['status']}")

    print("\nB. Bootstrap Chunk Wald Test Empirical P-Values (expanded90, Q_i):")
    primary_pvals = df_pvals[(df_pvals["endpoint"] == "Q_i") & (df_pvals["protocol"] == "expanded90")]
    for _, r in primary_pvals.iterrows():
        print(f"  {r['source_name']:16s} | Test {r['test_num']}: {r['test_name']:32s} | W_hc3: {r['wald_stat_hc3']:8.2f} | p_boot: {r['bootstrap_p_value']:.4f} (asymp p_chi2: {r['asymptotic_p_chi2']:.2e})")

    if not df_rep_summary.empty:
        print("\nC. Second-Window Repeatability of Q_i (Apple Watch, balanced90 N=2,324):")
        apple_rep = df_rep_summary[(df_rep_summary["source"] == 6) & (df_rep_summary["protocol"] == "balanced90")]
        for _, r in apple_rep.iterrows():
            print(f"  Sex: {r['sex']:6s} | Age: {r['age_bracket']:6s} (N={r['n_participants']:4d}) | Pearson r(Q_i): {r['pearson_r_Q_i']:+.4f} (p={r['p_val_Q_i']:.2e}) | Forecast r: {r['mean_forecast_r']:+.4f}")

    print("================================================================================")
    print("Milestone 4 execution completed successfully.")
    print("================================================================================")


if __name__ == "__main__":
    main()
