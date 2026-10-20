#!/usr/bin/env python3
r"""
03_fit_rcs_models.py: Milestone 3 implementation for the Continuous Age RCS Infradian Study.

Fits Frank Harrell's Restricted Cubic Splines (RCS) models across adult lifespan age,
evaluates four Chunk Wald hypothesis tests on shared age support, runs wear-compliance,
BMI, and birthday sensitivities, and decomposes all four physiological endpoints in parallel.

Mathematical Formulation:
  Stone-Koo RCS basis with landmark knots t = [25.0, 42.0, 52.0, 65.0] years:
    rcs_1(a) = [ (a - 25)_+^3 - (40/13)(a - 52)_+^3 + (27/13)(a - 65)_+^3 ] / 1600
    rcs_2(a) = [ (a - 42)_+^3 - (23/13)(a - 52)_+^3 + (10/13)(a - 65)_+^3 ] / 1600
  where (u)_+ = max(u, 0) and normalization factor is (t_4 - t_1)^2 = 1600.

Bivariate OLS Model:
  Y_i = \beta_0 + \beta_1 age_i + \beta_2 rcs_{1,i} + \beta_3 rcs_{2,i}
        + F_i (\delta_0 + \delta_1 age_i + \delta_2 rcs_{1,i} + \delta_3 rcs_{2,i}) + \varepsilon_i
  where F_i = 1 for females (salutation 10) and F_i = 0 for males (salutation 20).

Four Chunk Wald Hypothesis Tests:
  Test 1 (Total female age effect, 3 d.f.): joint test \beta_1+\delta_1 = 0, \beta_2+\delta_2 = 0, \beta_3+\delta_3 = 0.
  Test 2 (Nonlinear female curvature, 2 d.f.): joint test \beta_2+\delta_2 = 0, \beta_3+\delta_3 = 0.
  Test 3 (Total Sex x Age interaction, 3 d.f.): joint test \delta_1 = 0, \delta_2 = 0, \delta_3 = 0.
  Test 4 (Nonlinear Sex x Age interaction, 2 d.f.): joint test \delta_2 = 0, \delta_3 = 0.

Shared Age Support Rule:
  Evaluated strictly on 5-year age bands having >= 30 females and >= 30 males
  (empirically exactly [25, 75) years across 10 contiguous bands for Apple and Garmin).
  Samsung is evaluated and reported as insufficient support (< 30 participants per band).

Sensitivities:
  - Wear-compliance adjusted model: adds centered mean occupied bin fraction, nocturnal coverage fraction,
    and actual calendar day span.
  - BMI sensitivity model: fitted on participants with valid recorded BMI.
  - Birthday sensitivities: evaluated using January-1 and December-31 age conventions.

Four Endpoints in Parallel:
  1. Primary Q_i: band-integrated signed excess over [1/35, 1/25] d^-1.
  2. Discrete peak excess E_{i2} on historical grid.
  3. Physical cosinor amplitude \hat{A}_i (bpm).
  4. Residual standard deviation of resting heart rate \hat{\sigma}_{resid} (bpm).

Artifacts Persisted:
  - `artifacts/models/shared_support_mask.parquet`
  - `artifacts/models/model_fits.parquet`
  - `artifacts/models/chunk_wald_tests.csv`
  - `artifacts/models/model_summary.csv`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

# Canonical repository and experiment paths
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]

# Add scripts directory to path if needed
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import DEFAULT_KNOTS, rcs_stone_koo

ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
ENDPOINTS_PARQUET = ARTIFACTS_DIR / "endpoints" / "endpoints.parquet"
MODELS_DIR = ARTIFACTS_DIR / "models"

SOURCE_NAMES = {
    3: "Garmin Tier 1c",
    6: "Apple Watch",
    7: "Samsung",
}


# =============================================================================
# 1. Stone-Koo Restricted Cubic Spline Basis Functions & Derivatives
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


# =============================================================================
# 2. Design Matrix Construction
# =============================================================================
def build_rcs_design_matrix(
    ages: np.ndarray,
    sex: np.ndarray,
    covariates: Optional[Dict[str, np.ndarray]] = None,
    knots: Sequence[float] = DEFAULT_KNOTS,
) -> Tuple[np.ndarray, List[str]]:
    """
    Builds the design matrix:
      Columns: [const, age, rcs1, rcs2, sex, sex_age, sex_rcs1, sex_rcs2, (covariates...)]
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

    cols = [
        col_1, col_age, col_rcs1, col_rcs2,
        col_sex, col_sex_age, col_sex_rcs1, col_sex_rcs2
    ]
    col_names = [
        "const", "age", "rcs1", "rcs2",
        "sex", "sex_age", "sex_rcs1", "sex_rcs2"
    ]

    if covariates is not None:
        for cov_name, cov_arr in covariates.items():
            cols.append(cov_arr.astype(np.float64))
            col_names.append(cov_name)

    X = np.column_stack(cols)
    return X, col_names


# =============================================================================
# 3. Chunk Wald Hypothesis Testing
# =============================================================================
def compute_chunk_wald(
    theta: np.ndarray,
    cov: np.ndarray,
    C: np.ndarray,
    df_resid: int,
) -> Dict[str, float]:
    """
    Computes quadratic Wald statistic W and finite-sample F-statistic for H0: C @ theta = 0.
      W = (C theta)^T [C cov C^T]^-1 (C theta) ~ chi2(q)
      F = W / q ~ F(q, df_resid)
    """
    q = C.shape[0]
    C_theta = C @ theta
    middle = C @ cov @ C.T
    middle_inv = np.linalg.pinv(middle)
    wald_stat = float(C_theta.T @ middle_inv @ C_theta)
    # Ensure non-negative
    wald_stat = max(0.0, wald_stat)
    f_stat = wald_stat / q
    p_chi2 = float(stats.chi2.sf(wald_stat, df=q))
    p_f = float(stats.f.sf(f_stat, dfn=q, dfd=df_resid))
    return {
        "df_test": int(q),
        "wald_stat": wald_stat,
        "f_stat": f_stat,
        "p_chi2": p_chi2,
        "p_f": p_f,
    }


def build_chunk_contrast_matrices(p_total: int) -> Dict[int, Tuple[str, str, np.ndarray]]:
    """
    Constructs the 4 Chunk Wald hypothesis restriction matrices for a p_total column design:
      Columns 0..7 are [const, age, rcs1, rcs2, sex, sex_age, sex_rcs1, sex_rcs2].
      Additional columns (covariates) receive 0.0 in restrictions.
    """
    # Test 1 (Total female age effect, 3 d.f.): beta1+delta1=0, beta2+delta2=0, beta3+delta3=0
    C1 = np.zeros((3, p_total), dtype=np.float64)
    C1[0, 1] = 1.0; C1[0, 5] = 1.0  # age + sex_age
    C1[1, 2] = 1.0; C1[1, 6] = 1.0  # rcs1 + sex_rcs1
    C1[2, 3] = 1.0; C1[2, 7] = 1.0  # rcs2 + sex_rcs2

    # Test 2 (Nonlinear female curvature, 2 d.f.): beta2+delta2=0, beta3+delta3=0
    C2 = np.zeros((2, p_total), dtype=np.float64)
    C2[0, 2] = 1.0; C2[0, 6] = 1.0  # rcs1 + sex_rcs1
    C2[1, 3] = 1.0; C2[1, 7] = 1.0  # rcs2 + sex_rcs2

    # Test 3 (Total Sex x Age interaction, 3 d.f.): delta1=0, delta2=0, delta3=0
    C3 = np.zeros((3, p_total), dtype=np.float64)
    C3[0, 5] = 1.0  # sex_age
    C3[1, 6] = 1.0  # sex_rcs1
    C3[2, 7] = 1.0  # sex_rcs2

    # Test 4 (Nonlinear Sex x Age interaction, 2 d.f.): delta2=0, delta3=0
    C4 = np.zeros((2, p_total), dtype=np.float64)
    C4[0, 6] = 1.0  # sex_rcs1
    C4[1, 7] = 1.0  # sex_rcs2

    return {
        1: ("total_female_age", "beta_1 + delta_1 = 0, beta_2 + delta_2 = 0, beta_3 + delta_3 = 0", C1),
        2: ("nonlinear_female_curvature", "beta_2 + delta_2 = 0, beta_3 + delta_3 = 0", C2),
        3: ("total_sex_x_age_interaction", "delta_1 = 0, delta_2 = 0, delta_3 = 0", C3),
        4: ("nonlinear_sex_x_age_interaction", "delta_2 = 0, delta_3 = 0", C4),
    }


# =============================================================================
# 4. Age Support Masking Computation
# =============================================================================
def compute_shared_support_table(df_endpoints: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes 5-year age band support across all participants.
    Identifies shared support bands satisfying:
      N_female >= 30 AND N_male >= 30.
    Returns:
      df_support_mask: participant-level DataFrame with boolean support indicators.
      df_support_summary: band-level summary counts.
    """
    bins = np.arange(0, 130, 5)
    labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins) - 1)]

    df = df_endpoints[df_endpoints["null_status"] == "converged"].copy()
    df["age_band"] = pd.cut(df["age"], bins=bins, right=False, labels=labels)

    mask_rows = []
    summary_rows = []

    for (source, protocol), group in df.groupby(["source", "protocol"]):
        # Band counts
        ct = pd.crosstab(group["age_band"], group["sex"], dropna=False)
        m_counts = ct[0].to_dict() if 0 in ct.columns else {}
        f_counts = ct[1].to_dict() if 1 in ct.columns else {}

        shared_bands = set()
        female_bands = set()

        for b in labels:
            n_m = m_counts.get(b, 0)
            n_f = f_counts.get(b, 0)
            is_shared = (n_m >= 30) and (n_f >= 30)
            is_female = n_f >= 30
            if is_shared:
                shared_bands.add(b)
            if is_female:
                female_bands.add(b)

            summary_rows.append({
                "source": source,
                "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
                "protocol": protocol,
                "age_band": b,
                "male_count": n_m,
                "female_count": n_f,
                "total_count": n_m + n_f,
                "is_shared_support": is_shared,
                "is_female_support": is_female,
            })

        for idx, row in group.iterrows():
            b = str(row["age_band"])
            in_sh = b in shared_bands
            in_fe = b in female_bands
            mask_rows.append({
                "user_id": int(row["user_id"]),
                "source": int(source),
                "protocol": str(protocol),
                "window_index": int(row["window_index"]),
                "salutation": int(row["salutation"]),
                "sex": int(row["sex"]),
                "age": int(row["age"]),
                "age_band": b,
                "in_shared_support": bool(in_sh),
                "in_female_support": bool(in_fe),
                "male_band_count": int(m_counts.get(b, 0)),
                "female_band_count": int(f_counts.get(b, 0)),
            })

    df_mask = pd.DataFrame(mask_rows)
    df_summary = pd.DataFrame(summary_rows)
    return df_mask, df_summary


# =============================================================================
# 5. Core OLS Fit & Test Evaluator
# =============================================================================
def fit_rcs_model(
    df_sub: pd.DataFrame,
    endpoint: str,
    source: int,
    protocol: str,
    model_type: str,
    sample_scope: str,
    age_col: str = "age",
    covariates: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Fits OLS model, extracts parameter estimates, HC3 covariance,
    and evaluates the 4 Chunk Wald hypothesis tests.
    """
    model_id = f"src{source}_{protocol}_{sample_scope}_{endpoint}_{model_type}"
    n_obs = len(df_sub)
    n_female = int((df_sub["sex"] == 1).sum())
    n_male = int((df_sub["sex"] == 0).sum())

    # Check minimum support for estimation
    if n_female < 10 or n_male < 10 or n_obs < 30:
        fit_record = {
            "model_id": model_id,
            "source": source,
            "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
            "protocol": protocol,
            "endpoint": endpoint,
            "model_type": model_type,
            "sample_scope": sample_scope,
            "status": "insufficient_support",
            "n_obs": n_obs,
            "n_female": n_female,
            "n_male": n_male,
            "df_model": 0,
            "df_resid": 0,
            "r_squared": np.nan,
            "adj_r_squared": np.nan,
            "sigma_hat": np.nan,
            "f_stat": np.nan,
            "f_pvalue": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "beta_0": np.nan, "beta_1": np.nan, "beta_2": np.nan, "beta_3": np.nan,
            "delta_0": np.nan, "delta_1": np.nan, "delta_2": np.nan, "delta_3": np.nan,
            "se_beta_0_hc3": np.nan, "se_beta_1_hc3": np.nan, "se_beta_2_hc3": np.nan, "se_beta_3_hc3": np.nan,
            "se_delta_0_hc3": np.nan, "se_delta_1_hc3": np.nan, "se_delta_2_hc3": np.nan, "se_delta_3_hc3": np.nan,
            "female_b0": np.nan, "female_b1": np.nan, "female_b2": np.nan, "female_b3": np.nan,
            "se_female_b0_hc3": np.nan, "se_female_b1_hc3": np.nan, "se_female_b2_hc3": np.nan, "se_female_b3_hc3": np.nan,
            "param_names": "",
            "params_json": "",
            "cov_hc3_json": "",
            "cov_ols_json": "",
            "notes": "Insufficient sample size or support in strata",
        }
        test_records = []
        for t_num, (t_name, null_str, _) in build_chunk_contrast_matrices(8).items():
            test_records.append({
                "model_id": model_id,
                "source": source,
                "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
                "protocol": protocol,
                "endpoint": endpoint,
                "model_type": model_type,
                "sample_scope": sample_scope,
                "test_num": t_num,
                "test_name": t_name,
                "null_hypothesis": null_str,
                "df_test": 3 if t_num in (1, 3) else 2,
                "df_resid": 0,
                "wald_stat_hc3": np.nan,
                "f_stat_hc3": np.nan,
                "p_value_hc3": np.nan,
                "wald_stat_ols": np.nan,
                "f_stat_ols": np.nan,
                "p_value_ols": np.nan,
                "status": "not_estimable",
            })
        return fit_record, test_records

    ages = df_sub[age_col].values.astype(np.float64)
    sex = df_sub["sex"].values.astype(np.float64)
    y = df_sub[endpoint].values.astype(np.float64)

    X, col_names = build_rcs_design_matrix(ages, sex, covariates=covariates)
    p_total = X.shape[1]
    df_resid = n_obs - p_total

    # Fit OLS model with both classical OLS and HC3 robust sandwich covariance
    ols_model = sm.OLS(y, X)
    res_ols = ols_model.fit()
    res_hc3 = ols_model.fit(cov_type="HC3")

    theta = res_ols.params
    cov_ols = res_ols.cov_params()
    cov_hc3 = res_hc3.cov_params()

    se_ols = res_ols.bse
    se_hc3 = res_hc3.bse

    # Core parameters
    b0, b1, b2, b3 = theta[0], theta[1], theta[2], theta[3]
    d0, d1, d2, d3 = theta[4], theta[5], theta[6], theta[7]

    se_b0_hc3, se_b1_hc3, se_b2_hc3, se_b3_hc3 = se_hc3[0], se_hc3[1], se_hc3[2], se_hc3[3]
    se_d0_hc3, se_d1_hc3, se_d2_hc3, se_d3_hc3 = se_hc3[4], se_hc3[5], se_hc3[6], se_hc3[7]

    # Female parameters: beta_j + delta_j
    fb0 = b0 + d0
    fb1 = b1 + d1
    fb2 = b2 + d2
    fb3 = b3 + d3

    # SE for female parameters: sqrt(Var(b) + Var(d) + 2*Cov(b,d))
    se_fb0_hc3 = float(np.sqrt(cov_hc3[0, 0] + cov_hc3[4, 4] + 2.0 * cov_hc3[0, 4]))
    se_fb1_hc3 = float(np.sqrt(cov_hc3[1, 1] + cov_hc3[5, 5] + 2.0 * cov_hc3[1, 5]))
    se_fb2_hc3 = float(np.sqrt(cov_hc3[2, 2] + cov_hc3[6, 6] + 2.0 * cov_hc3[2, 6]))
    se_fb3_hc3 = float(np.sqrt(cov_hc3[3, 3] + cov_hc3[7, 7] + 2.0 * cov_hc3[3, 7]))

    fit_record = {
        "model_id": model_id,
        "source": source,
        "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
        "protocol": protocol,
        "endpoint": endpoint,
        "model_type": model_type,
        "sample_scope": sample_scope,
        "status": "converged",
        "n_obs": n_obs,
        "n_female": n_female,
        "n_male": n_male,
        "df_model": int(res_ols.df_model),
        "df_resid": int(df_resid),
        "r_squared": float(res_ols.rsquared),
        "adj_r_squared": float(res_ols.rsquared_adj),
        "sigma_hat": float(np.sqrt(res_ols.mse_resid)),
        "f_stat": float(res_ols.fvalue),
        "f_pvalue": float(res_ols.f_pvalue),
        "aic": float(res_ols.aic),
        "bic": float(res_ols.bic),
        "beta_0": float(b0), "beta_1": float(b1), "beta_2": float(b2), "beta_3": float(b3),
        "delta_0": float(d0), "delta_1": float(d1), "delta_2": float(d2), "delta_3": float(d3),
        "se_beta_0_hc3": float(se_b0_hc3), "se_beta_1_hc3": float(se_b1_hc3),
        "se_beta_2_hc3": float(se_b2_hc3), "se_beta_3_hc3": float(se_b3_hc3),
        "se_delta_0_hc3": float(se_d0_hc3), "se_delta_1_hc3": float(se_d1_hc3),
        "se_delta_2_hc3": float(se_d2_hc3), "se_delta_3_hc3": float(se_d3_hc3),
        "female_b0": float(fb0), "female_b1": float(fb1),
        "female_b2": float(fb2), "female_b3": float(fb3),
        "se_female_b0_hc3": float(se_fb0_hc3), "se_female_b1_hc3": float(se_fb1_hc3),
        "se_female_b2_hc3": float(se_fb2_hc3), "se_female_b3_hc3": float(se_fb3_hc3),
        "param_names": ",".join(col_names),
        "params_json": json.dumps(theta.tolist()),
        "cov_hc3_json": json.dumps(cov_hc3.tolist()),
        "cov_ols_json": json.dumps(cov_ols.tolist()),
        "notes": "Successfully fitted with OLS and HC3 covariance",
    }

    # Evaluate the 4 Chunk Wald tests
    test_matrices = build_chunk_contrast_matrices(p_total)
    test_records = []
    for t_num, (t_name, null_str, C) in test_matrices.items():
        res_test_hc3 = compute_chunk_wald(theta, cov_hc3, C, df_resid)
        res_test_ols = compute_chunk_wald(theta, cov_ols, C, df_resid)

        test_records.append({
            "model_id": model_id,
            "source": source,
            "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
            "protocol": protocol,
            "endpoint": endpoint,
            "model_type": model_type,
            "sample_scope": sample_scope,
            "test_num": t_num,
            "test_name": t_name,
            "null_hypothesis": null_str,
            "df_test": res_test_hc3["df_test"],
            "df_resid": int(df_resid),
            "wald_stat_hc3": res_test_hc3["wald_stat"],
            "f_stat_hc3": res_test_hc3["f_stat"],
            "p_value_hc3": res_test_hc3["p_chi2"],
            "wald_stat_ols": res_test_ols["wald_stat"],
            "f_stat_ols": res_test_ols["f_stat"],
            "p_value_ols": res_test_ols["p_chi2"],
            "status": "evaluated",
        })

    return fit_record, test_records


# =============================================================================
# 6. Full Pipeline Driver
# =============================================================================
def run_model_fitting_pipeline(
    endpoints_path: Path = ENDPOINTS_PARQUET,
    output_dir: Path = MODELS_DIR,
    target_sources: Sequence[int] = (3, 6, 7),
    target_protocols: Sequence[str] = ("expanded90", "balanced90"),
    window_index: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the complete Milestone 3 fitting pipeline across sources, protocols,
    endpoints, support scopes, and sensitivities.
    """
    print("=" * 80)
    print("Continuous Age RCS Modeling Pipeline (Milestone 3)")
    print("=" * 80)
    print(f"Endpoints file: {endpoints_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target sources: {target_sources}")
    print(f"Target protocols: {target_protocols}")
    print(f"Window index: {window_index}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load endpoints data
    df_ep = pd.read_parquet(endpoints_path)
    print(f"\nLoaded {len(df_ep)} total endpoint rows from {endpoints_path.name}")

    # Filter to primary window and converged nulls
    df_primary = df_ep[
        (df_ep["window_index"] == window_index) & (df_ep["null_status"] == "converged")
    ].copy()
    print(f"Retained {len(df_primary)} converged rows for window_index={window_index}")

    # 2. Compute shared support mask table
    print("\n[Stage 1/4] Computing 5-Year Age Support & Shared Support Masks...")
    df_mask, df_support_summary = compute_shared_support_table(df_primary)
    mask_out_path = output_dir / "shared_support_mask.parquet"
    df_mask.to_parquet(mask_out_path, index=False)
    print(f"  Saved shared support mask to {mask_out_path} ({len(df_mask)} rows)")

    # Print support overview
    for (src, prot), grp in df_mask.groupby(["source", "protocol"]):
        n_tot = len(grp)
        n_sh = grp["in_shared_support"].sum()
        pct_sh = (n_sh / n_tot) * 100.0 if n_tot > 0 else 0.0
        sh_bands = grp[grp["in_shared_support"]]["age_band"].unique().tolist()
        print(f"  Source {src} ({SOURCE_NAMES.get(src)}) | {prot}: {n_sh}/{n_tot} ({pct_sh:.1f}%) in shared support")
        if sh_bands:
            print(f"    Shared bands: {min(sh_bands)} to {max(sh_bands)} ({len(sh_bands)} bands)")
        else:
            print("    Shared bands: None (< 30 participants per cell)")

    # 3. Model Fitting Loop
    print("\n[Stage 2/4] Fitting RCS OLS Models & Evaluating Chunk Wald Tests...")
    all_fit_records: List[Dict[str, Any]] = []
    all_test_records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    endpoints_list = ["Q_i", "E_i2", "A_hat", "sigma_resid"]

    for source in target_sources:
        for protocol in target_protocols:
            # Subset for this source and protocol
            sub_cohort = df_primary[
                (df_primary["source"] == source) & (df_primary["protocol"] == protocol)
            ].copy()

            # Join shared support indicator
            sub_mask = df_mask[
                (df_mask["source"] == source) & (df_mask["protocol"] == protocol)
            ][["user_id", "in_shared_support", "in_female_support"]]
            sub_cohort = sub_cohort.merge(sub_mask, on="user_id", how="left")

            n_total = len(sub_cohort)
            n_shared = int(sub_cohort["in_shared_support"].sum()) if n_total > 0 else 0

            # Check if source satisfies minimum cohort size and shared support
            is_source_viable = (n_total >= 50) and (n_shared >= 50)

            # Pooled within-source eligible means for wear centering
            mean_occ_bar = float(sub_cohort["mean_occ"].mean()) if n_total > 0 else 0.0
            mean_night_bar = float(sub_cohort["mean_night_cov"].mean()) if n_total > 0 else 0.0
            date_span_bar = float(sub_cohort["date_span"].mean()) if n_total > 0 else 0.0

            for endpoint in endpoints_list:
                if not is_source_viable:
                    # Sparse source (e.g. Samsung source 7)
                    fit_rec, test_recs = fit_rcs_model(
                        sub_cohort, endpoint, source, protocol,
                        model_type="primary_unadjusted", sample_scope="shared_support"
                    )
                    all_fit_records.append(fit_rec)
                    all_test_records.extend(test_recs)

                    summary_rows.append({
                        "model_id": fit_rec["model_id"],
                        "source": source,
                        "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
                        "protocol": protocol,
                        "endpoint": endpoint,
                        "model_type": "primary_unadjusted",
                        "sample_scope": "shared_support",
                        "status": "insufficient_support",
                        "n_obs": n_total,
                        "n_female": int((sub_cohort["sex"] == 1).sum()),
                        "n_male": int((sub_cohort["sex"] == 0).sum()),
                        "r_squared": np.nan,
                        "sigma_hat": np.nan,
                        "test1_p_hc3": np.nan,
                        "test2_p_hc3": np.nan,
                        "test3_p_hc3": np.nan,
                        "test4_p_hc3": np.nan,
                        "beta_1": np.nan,
                        "delta_1": np.nan,
                        "female_b1": np.nan,
                        "notes": f"Cohort total N={n_total} or shared support N={n_shared} below threshold (<50)",
                    })
                    continue

                # Viable source: fit the 8 models
                # -------------------------------------------------------------
                # Model 1: Primary unadjusted on shared_support
                # -------------------------------------------------------------
                df_shared = sub_cohort[sub_cohort["in_shared_support"]].copy()
                fit1, tests1 = fit_rcs_model(
                    df_shared, endpoint, source, protocol,
                    model_type="primary_unadjusted", sample_scope="shared_support"
                )
                all_fit_records.append(fit1)
                all_test_records.extend(tests1)

                p1_hc3 = {t["test_num"]: t["p_value_hc3"] for t in tests1}
                summary_rows.append({
                    "model_id": fit1["model_id"],
                    "source": source,
                    "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
                    "protocol": protocol,
                    "endpoint": endpoint,
                    "model_type": "primary_unadjusted",
                    "sample_scope": "shared_support",
                    "status": "converged",
                    "n_obs": fit1["n_obs"],
                    "n_female": fit1["n_female"],
                    "n_male": fit1["n_male"],
                    "r_squared": fit1["r_squared"],
                    "sigma_hat": fit1["sigma_hat"],
                    "test1_p_hc3": p1_hc3.get(1, np.nan),
                    "test2_p_hc3": p1_hc3.get(2, np.nan),
                    "test3_p_hc3": p1_hc3.get(3, np.nan),
                    "test4_p_hc3": p1_hc3.get(4, np.nan),
                    "beta_1": fit1["beta_1"],
                    "delta_1": fit1["delta_1"],
                    "female_b1": fit1["female_b1"],
                    "notes": "Primary inferential model on shared age support [25, 75)",
                })

                # -------------------------------------------------------------
                # Model 2: Primary unadjusted on all_participants
                # -------------------------------------------------------------
                fit2, tests2 = fit_rcs_model(
                    sub_cohort, endpoint, source, protocol,
                    model_type="primary_unadjusted", sample_scope="all_participants"
                )
                all_fit_records.append(fit2)
                all_test_records.extend(tests2)

                p2_hc3 = {t["test_num"]: t["p_value_hc3"] for t in tests2}
                summary_rows.append({
                    "model_id": fit2["model_id"],
                    "source": source,
                    "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
                    "protocol": protocol,
                    "endpoint": endpoint,
                    "model_type": "primary_unadjusted",
                    "sample_scope": "all_participants",
                    "status": "converged",
                    "n_obs": fit2["n_obs"],
                    "n_female": fit2["n_female"],
                    "n_male": fit2["n_male"],
                    "r_squared": fit2["r_squared"],
                    "sigma_hat": fit2["sigma_hat"],
                    "test1_p_hc3": p2_hc3.get(1, np.nan),
                    "test2_p_hc3": p2_hc3.get(2, np.nan),
                    "test3_p_hc3": p2_hc3.get(3, np.nan),
                    "test4_p_hc3": p2_hc3.get(4, np.nan),
                    "beta_1": fit2["beta_1"],
                    "delta_1": fit2["delta_1"],
                    "female_b1": fit2["female_b1"],
                    "notes": "Primary female age curve across all valid ages",
                })

                # -------------------------------------------------------------
                # Model 3: Wear-compliance adjusted on shared_support
                # -------------------------------------------------------------
                covs_wear_shared = {
                    "z_occ": df_shared["mean_occ"].values - mean_occ_bar,
                    "z_night": df_shared["mean_night_cov"].values - mean_night_bar,
                    "z_span": df_shared["date_span"].values - date_span_bar,
                }
                fit3, tests3 = fit_rcs_model(
                    df_shared, endpoint, source, protocol,
                    model_type="wear_adjusted", sample_scope="shared_support",
                    covariates=covs_wear_shared
                )
                all_fit_records.append(fit3)
                all_test_records.extend(tests3)

                p3_hc3 = {t["test_num"]: t["p_value_hc3"] for t in tests3}
                summary_rows.append({
                    "model_id": fit3["model_id"],
                    "source": source,
                    "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
                    "protocol": protocol,
                    "endpoint": endpoint,
                    "model_type": "wear_adjusted",
                    "sample_scope": "shared_support",
                    "status": "converged",
                    "n_obs": fit3["n_obs"],
                    "n_female": fit3["n_female"],
                    "n_male": fit3["n_male"],
                    "r_squared": fit3["r_squared"],
                    "sigma_hat": fit3["sigma_hat"],
                    "test1_p_hc3": p3_hc3.get(1, np.nan),
                    "test2_p_hc3": p3_hc3.get(2, np.nan),
                    "test3_p_hc3": p3_hc3.get(3, np.nan),
                    "test4_p_hc3": p3_hc3.get(4, np.nan),
                    "beta_1": fit3["beta_1"],
                    "delta_1": fit3["delta_1"],
                    "female_b1": fit3["female_b1"],
                    "notes": "Adjusted for centered occupancy, nocturnal coverage, and day span",
                })

                # -------------------------------------------------------------
                # Model 4: Wear-compliance adjusted on all_participants
                # -------------------------------------------------------------
                covs_wear_all = {
                    "z_occ": sub_cohort["mean_occ"].values - mean_occ_bar,
                    "z_night": sub_cohort["mean_night_cov"].values - mean_night_bar,
                    "z_span": sub_cohort["date_span"].values - date_span_bar,
                }
                fit4, tests4 = fit_rcs_model(
                    sub_cohort, endpoint, source, protocol,
                    model_type="wear_adjusted", sample_scope="all_participants",
                    covariates=covs_wear_all
                )
                all_fit_records.append(fit4)
                all_test_records.extend(tests4)

                p4_hc3 = {t["test_num"]: t["p_value_hc3"] for t in tests4}
                summary_rows.append({
                    "model_id": fit4["model_id"],
                    "source": source,
                    "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
                    "protocol": protocol,
                    "endpoint": endpoint,
                    "model_type": "wear_adjusted",
                    "sample_scope": "all_participants",
                    "status": "converged",
                    "n_obs": fit4["n_obs"],
                    "n_female": fit4["n_female"],
                    "n_male": fit4["n_male"],
                    "r_squared": fit4["r_squared"],
                    "sigma_hat": fit4["sigma_hat"],
                    "test1_p_hc3": p4_hc3.get(1, np.nan),
                    "test2_p_hc3": p4_hc3.get(2, np.nan),
                    "test3_p_hc3": p4_hc3.get(3, np.nan),
                    "test4_p_hc3": p4_hc3.get(4, np.nan),
                    "beta_1": fit4["beta_1"],
                    "delta_1": fit4["delta_1"],
                    "female_b1": fit4["female_b1"],
                    "notes": "Wear-compliance adjusted across all valid participants",
                })

                # -------------------------------------------------------------
                # Model 5 & 6: BMI Sensitivity on shared_support
                # -------------------------------------------------------------
                df_bmi = df_shared[~df_shared["bmi"].isna()].copy()
                if len(df_bmi) >= 50:
                    mean_bmi_bar = float(df_bmi["bmi"].mean())
                    covs_bmi = {
                        "z_bmi": df_bmi["bmi"].values - mean_bmi_bar,
                    }
                    fit5, tests5 = fit_rcs_model(
                        df_bmi, endpoint, source, protocol,
                        model_type="bmi_adjusted", sample_scope="shared_support",
                        covariates=covs_bmi
                    )
                    all_fit_records.append(fit5)
                    all_test_records.extend(tests5)

                    p5_hc3 = {t["test_num"]: t["p_value_hc3"] for t in tests5}
                    summary_rows.append({
                        "model_id": fit5["model_id"],
                        "source": source,
                        "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
                        "protocol": protocol,
                        "endpoint": endpoint,
                        "model_type": "bmi_adjusted",
                        "sample_scope": "shared_support",
                        "status": "converged",
                        "n_obs": fit5["n_obs"],
                        "n_female": fit5["n_female"],
                        "n_male": fit5["n_male"],
                        "r_squared": fit5["r_squared"],
                        "sigma_hat": fit5["sigma_hat"],
                        "test1_p_hc3": p5_hc3.get(1, np.nan),
                        "test2_p_hc3": p5_hc3.get(2, np.nan),
                        "test3_p_hc3": p5_hc3.get(3, np.nan),
                        "test4_p_hc3": p5_hc3.get(4, np.nan),
                        "beta_1": fit5["beta_1"],
                        "delta_1": fit5["delta_1"],
                        "female_b1": fit5["female_b1"],
                        "notes": "Adjusted for centered BMI on complete BMI cases",
                    })

                    # Model 6: unadjusted on identical BMI complete subset
                    fit6, tests6 = fit_rcs_model(
                        df_bmi, endpoint, source, protocol,
                        model_type="bmi_unadjusted_subset", sample_scope="shared_support"
                    )
                    all_fit_records.append(fit6)
                    all_test_records.extend(tests6)

                    p6_hc3 = {t["test_num"]: t["p_value_hc3"] for t in tests6}
                    summary_rows.append({
                        "model_id": fit6["model_id"],
                        "source": source,
                        "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
                        "protocol": protocol,
                        "endpoint": endpoint,
                        "model_type": "bmi_unadjusted_subset",
                        "sample_scope": "shared_support",
                        "status": "converged",
                        "n_obs": fit6["n_obs"],
                        "n_female": fit6["n_female"],
                        "n_male": fit6["n_male"],
                        "r_squared": fit6["r_squared"],
                        "sigma_hat": fit6["sigma_hat"],
                        "test1_p_hc3": p6_hc3.get(1, np.nan),
                        "test2_p_hc3": p6_hc3.get(2, np.nan),
                        "test3_p_hc3": p6_hc3.get(3, np.nan),
                        "test4_p_hc3": p6_hc3.get(4, np.nan),
                        "beta_1": fit6["beta_1"],
                        "delta_1": fit6["delta_1"],
                        "female_b1": fit6["female_b1"],
                        "notes": "Unadjusted model on identical BMI complete cases for direct contrast",
                    })

                # -------------------------------------------------------------
                # Model 7 & 8: Birthday Sensitivities (January-1 and December-31)
                # -------------------------------------------------------------
                for bday_col, model_name in [("age_jan1", "sens_jan1"), ("age_dec31", "sens_dec31")]:
                    # Filter to shared support under this birthday convention [25, 75)
                    in_sh_bday = (sub_cohort[bday_col] >= 25) & (sub_cohort[bday_col] < 75)
                    df_bday = sub_cohort[in_sh_bday].copy()

                    fit_b, tests_b = fit_rcs_model(
                        df_bday, endpoint, source, protocol,
                        model_type=model_name, sample_scope="shared_support",
                        age_col=bday_col
                    )
                    all_fit_records.append(fit_b)
                    all_test_records.extend(tests_b)

                    pb_hc3 = {t["test_num"]: t["p_value_hc3"] for t in tests_b}
                    summary_rows.append({
                        "model_id": fit_b["model_id"],
                        "source": source,
                        "source_name": SOURCE_NAMES.get(source, f"Source {source}"),
                        "protocol": protocol,
                        "endpoint": endpoint,
                        "model_type": model_name,
                        "sample_scope": "shared_support",
                        "status": "converged",
                        "n_obs": fit_b["n_obs"],
                        "n_female": fit_b["n_female"],
                        "n_male": fit_b["n_male"],
                        "r_squared": fit_b["r_squared"],
                        "sigma_hat": fit_b["sigma_hat"],
                        "test1_p_hc3": pb_hc3.get(1, np.nan),
                        "test2_p_hc3": pb_hc3.get(2, np.nan),
                        "test3_p_hc3": pb_hc3.get(3, np.nan),
                        "test4_p_hc3": pb_hc3.get(4, np.nan),
                        "beta_1": fit_b["beta_1"],
                        "delta_1": fit_b["delta_1"],
                        "female_b1": fit_b["female_b1"],
                        "notes": f"Birthday sensitivity using {bday_col} convention",
                    })

    # 4. Convert to DataFrames and Persist Artifacts
    print("\n[Stage 3/4] Persisting Output Artifacts...")
    df_fits = pd.DataFrame(all_fit_records)
    df_tests = pd.DataFrame(all_test_records)
    df_summary = pd.DataFrame(summary_rows)

    fits_path = output_dir / "model_fits.parquet"
    df_fits.to_parquet(fits_path, index=False)
    print(f"  Saved {len(df_fits)} model fits to {fits_path}")

    tests_path = output_dir / "chunk_wald_tests.csv"
    df_tests.to_csv(tests_path, index=False)
    print(f"  Saved {len(df_tests)} Chunk Wald tests to {tests_path}")

    summary_path = output_dir / "model_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"  Saved {len(df_summary)} model summaries to {summary_path}")

    # 5. Output Verification & Sanity Checks
    print("\n[Stage 4/4] Output Verification & Key Statistical Highlights:")
    primary_fits = df_summary[
        (df_summary["model_type"] == "primary_unadjusted") &
        (df_summary["sample_scope"] == "shared_support") &
        (df_summary["endpoint"] == "Q_i")
    ]
    for _, row in primary_fits.iterrows():
        print(f"\n--- {row['source_name']} | {row['protocol']} | Primary Q_i on Shared Support ---")
        print(f"  Status: {row['status']}, N_obs: {row['n_obs']} (Females: {row['n_female']}, Males: {row['n_male']})")
        if row["status"] == "converged":
            print(f"  R^2: {row['r_squared']:.4f}, sigma_hat: {row['sigma_hat']:.4f}")
            print(f"  Test 1 (Total female age effect):         p(HC3) = {row['test1_p_hc3']:.4e}")
            print(f"  Test 2 (Nonlinear female curvature):       p(HC3) = {row['test2_p_hc3']:.4e}")
            print(f"  Test 3 (Total Sex x Age interaction):      p(HC3) = {row['test3_p_hc3']:.4e}")
            print(f"  Test 4 (Nonlinear Sex x Age interaction):  p(HC3) = {row['test4_p_hc3']:.4e}")
            print(f"  Key Slopes: Male beta_1={row['beta_1']:.6f}, Delta delta_1={row['delta_1']:.6f}, Female beta_1={row['female_b1']:.6f}")

    print("\n" + "=" * 80)
    print("Milestone 3 Model Fitting Completed Successfully.")
    print("=" * 80)
    return df_mask, df_fits, df_tests, df_summary


# =============================================================================
# 7. Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fit continuous age RCS OLS models and Chunk Wald hypothesis tests (Milestone 3)"
    )
    parser.add_argument(
        "--endpoints-file",
        type=Path,
        default=ENDPOINTS_PARQUET,
        help="Path to endpoints.parquet input artifact",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help="Path to artifacts/models directory for outputs",
    )
    parser.add_argument(
        "--sources",
        type=int,
        nargs="+",
        default=[3, 6, 7],
        help="List of sources to analyze (default: 3 6 7)",
    )
    parser.add_argument(
        "--protocols",
        type=str,
        nargs="+",
        default=["expanded90", "balanced90"],
        help="List of protocols to analyze (default: expanded90 balanced90)",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=1,
        help="Window index to analyze (default: 1 for primary window)",
    )
    args = parser.parse_args()

    run_model_fitting_pipeline(
        endpoints_path=args.endpoints_file,
        output_dir=args.output_dir,
        target_sources=args.sources,
        target_protocols=args.protocols,
        window_index=args.window_index,
    )


if __name__ == "__main__":
    main()
