#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
02_summarize_infradian_contrasts.py

Age-Stratified Statistical Inference for female_infradian_cycles_v2:
1. Loads window_eligibility.parquet (N=4,631 first-window eligible participants).
2. Primary model: Participant-level OLS regression with categorical age
   ('Young Adults', 'Middle Ager', 'Elderly'), salutation (y in {0, 1}),
   and their interaction on excess infradian score E_i.
   Adjusts for observation geometry: selected-day occupancy rate, night-clock coverage,
   and date span, centered within age strata.
3. Performs the shared family of four HC3 robust Wald tests with Holm adjustment at alpha=0.05:
   - 2-df age x sex interaction omnibus test.
   - Within-age contrast: Young Adults (y0 vs y1).
   - Within-age contrast: Middle Ager (y0 vs y1).
   - Within-age contrast: Elderly (y0 vs y1).
4. Computes:
   - 1,000 stratified participant bootstrap resamples for 95% pointwise confidence intervals.
   - Fixed age weights w_a = N_a / 4631 and standardized contrast Delta_std = sum_a w_a Delta_a.
   - Unadjusted contrasts and sensitivity on physical amplitude A_i and score S_i.
   - Second-window temporal persistence metrics (N=2,324).
   - Wear compliance contrasts (occupancy rate and night coverage) for double dissociation.
5. Saves results to:
   experiments/female_infradian_cycles_v2/results/calibrated_group_contrasts.csv
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))


def compute_ols_contrasts(
    df: pd.DataFrame,
    endpoint: str = "E_i",
    adjust_covariates: bool = True,
) -> Dict[str, Any]:
    """
    Fits participant-level OLS regression with 6 cell indicators (age x salutation)
    and optional within-age-centered observation covariates.
    
    Returns point estimates, HC3 covariance matrix, Wald statistics, and p-values.
    """
    # Exclude any rows with NaN in endpoint
    valid = df[df[endpoint].notna()].copy()
    n_total = len(valid)
    
    age_categories = ["Young Adults", "Middle Ager", "Elderly"]
    
    # 6 cell indicators: (Young_0, Young_1, Middle_0, Middle_1, Elderly_0, Elderly_1)
    cell_names = []
    cell_cols = []
    for a in age_categories:
        for y_val in [0, 1]:
            c_name = f"{a}_y{y_val}"
            cell_names.append(c_name)
            is_cell = ((valid["age_group"] == a) & (valid["y"] == y_val)).astype(np.float64).values
            cell_cols.append(is_cell)
            
    X_mat = np.column_stack(cell_cols)
    col_names = list(cell_names)
    
    if adjust_covariates:
        # Centering within pooled age strata
        cov_names = ["w1_mean_occ_5m", "w1_mean_clock_night", "w1_actual_span"]
        for c in cov_names:
            c_vals = valid[c].astype(np.float64).values.copy()
            # Center within each age stratum
            for a in age_categories:
                mask_a = (valid["age_group"] == a).values
                if np.any(mask_a):
                    c_vals[mask_a] -= np.mean(c_vals[mask_a])
            X_mat = np.column_stack([X_mat, c_vals])
            col_names.append(f"{c}_centered")
            
    y_vec = valid[endpoint].values
    
    # Fit OLS with HC3 robust covariance
    model = sm.OLS(y_vec, X_mat)
    results = model.fit(cov_type="HC3")
    
    beta = results.params
    cov_hc3 = results.cov_params()
    
    # Within-age contrasts: Delta_a = mu(a, y0) - mu(a, y1)
    # Cell indices:
    # 0: Young_0, 1: Young_1
    # 2: Middle_0, 3: Middle_1
    # 4: Elderly_0, 5: Elderly_1
    p_cols = len(col_names)
    
    c_young = np.zeros(p_cols)
    c_young[0] = 1.0
    c_young[1] = -1.0
    
    c_middle = np.zeros(p_cols)
    c_middle[2] = 1.0
    c_middle[3] = -1.0
    
    c_elderly = np.zeros(p_cols)
    c_elderly[4] = 1.0
    c_elderly[5] = -1.0
    
    # Age weights w_a = N_a / 4631
    n_young = np.sum(valid["age_group"] == "Young Adults")
    n_middle = np.sum(valid["age_group"] == "Middle Ager")
    n_elderly = np.sum(valid["age_group"] == "Elderly")
    
    w_young = n_young / n_total
    w_middle = n_middle / n_total
    w_elderly = n_elderly / n_total
    
    c_std = w_young * c_young + w_middle * c_middle + w_elderly * c_elderly
    
    contrasts = {
        "Young Adults": (c_young, float(c_young @ beta), float(np.sqrt(c_young @ cov_hc3 @ c_young))),
        "Middle Ager": (c_middle, float(c_middle @ beta), float(np.sqrt(c_middle @ cov_hc3 @ c_middle))),
        "Elderly": (c_elderly, float(c_elderly @ beta), float(np.sqrt(c_elderly @ cov_hc3 @ c_elderly))),
        "Standardized": (c_std, float(c_std @ beta), float(np.sqrt(c_std @ cov_hc3 @ c_std))),
    }
    
    # 2-df age x sex interaction omnibus test:
    # H0: Delta_Young == Delta_Middle and Delta_Elderly == Delta_Middle
    # R has 2 rows
    R_omnibus = np.zeros((2, p_cols))
    R_omnibus[0] = c_young - c_middle
    R_omnibus[1] = c_elderly - c_middle
    
    wald_test_res = results.wald_test(R_omnibus, use_f=False, scalar=True)
    omnibus_stat = float(np.squeeze(wald_test_res.statistic))
    omnibus_p = float(np.squeeze(wald_test_res.pvalue))
    
    return {
        "results": results,
        "beta": beta,
        "cov_hc3": cov_hc3,
        "contrasts": contrasts,
        "omnibus_stat": omnibus_stat,
        "omnibus_p": omnibus_p,
        "weights": {"Young Adults": w_young, "Middle Ager": w_middle, "Elderly": w_elderly},
        "valid_df": valid,
    }


def run_stratified_bootstrap(
    df: pd.DataFrame,
    endpoint: str = "E_i",
    n_boot: int = 1000,
    seed: int = 20261020,
) -> Dict[str, Tuple[float, float]]:
    """
    1,000 participant bootstrap resamples, stratified by age x salutation,
    for 95% pointwise contrast intervals.
    """
    valid = df[df[endpoint].notna()].copy()
    rng = np.random.default_rng(seed)
    
    age_categories = ["Young Adults", "Middle Ager", "Elderly"]
    
    # Stratified indices
    strata_indices = {}
    for a in age_categories:
        for y_val in [0, 1]:
            idx = valid[(valid["age_group"] == a) & (valid["y"] == y_val)].index.to_numpy()
            strata_indices[(a, y_val)] = idx
            
    boot_estimates = {
        "Young Adults": np.empty(n_boot),
        "Middle Ager": np.empty(n_boot),
        "Elderly": np.empty(n_boot),
        "Standardized": np.empty(n_boot),
    }
    
    for b in range(n_boot):
        # Sample with replacement within each stratum
        resampled_idx = []
        for idx in strata_indices.values():
            sampled = rng.choice(idx, size=len(idx), replace=True)
            resampled_idx.append(sampled)
        boot_df = valid.loc[np.concatenate(resampled_idx)]
        
        # Fit model on bootstrap sample
        res_b = compute_ols_contrasts(boot_df, endpoint=endpoint, adjust_covariates=True)
        for k in ["Young Adults", "Middle Ager", "Elderly", "Standardized"]:
            boot_estimates[k][b] = res_b["contrasts"][k][1]
            
    ci_95 = {}
    for k in boot_estimates:
        low = float(np.percentile(boot_estimates[k], 2.5))
        high = float(np.percentile(boot_estimates[k], 97.5))
        ci_95[k] = (low, high)
        
    return ci_95


def main() -> int:
    t_start = time.perf_counter()
    print("=" * 80)
    print("02_summarize_infradian_contrasts.py")
    print("Age-Stratified HC3 Wald Inference & Temporal Persistence Analysis")
    print("=" * 80)
    
    parquet_path = os.path.join(
        PROJECT_ROOT,
        "experiments/female_infradian_cycles_v2/results/window_eligibility.parquet"
    )
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Eligibility file not found: {parquet_path}")
        
    df_all = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df_all)} total Apple Watch participants from {parquet_path}")
    
    df_w1 = df_all[df_all["eligible_w1"]].copy()
    n_w1 = len(df_w1)
    print(f"Eligible First-Window cohort: N = {n_w1} participants.")
    
    # -------------------------------------------------------------------------
    # 1. Primary Model: HC3 OLS on Excess Infradian Score E_i
    # -------------------------------------------------------------------------
    print("\n--- Fitting Primary Observation-Adjusted Model on E_i ---")
    ols_res_e = compute_ols_contrasts(df_w1, endpoint="E_i", adjust_covariates=True)
    
    # Four-test family:
    # 1. Omnibus 2-df age x sex interaction
    # 2. Young Adults (y0 vs y1)
    # 3. Middle Ager (y0 vs y1)
    # 4. Elderly (y0 vs y1)
    
    c_young = ols_res_e["contrasts"]["Young Adults"]
    c_middle = ols_res_e["contrasts"]["Middle Ager"]
    c_elderly = ols_res_e["contrasts"]["Elderly"]
    c_std = ols_res_e["contrasts"]["Standardized"]
    
    # 1-df Wald statistics: W = (Delta / SE)^2
    w_young = (c_young[1] / c_young[2]) ** 2
    p_young = float(1.0 - stats.chi2.cdf(w_young, df=1))
    
    w_middle = (c_middle[1] / c_middle[2]) ** 2
    p_middle = float(1.0 - stats.chi2.cdf(w_middle, df=1))
    
    w_elderly = (c_elderly[1] / c_elderly[2]) ** 2
    p_elderly = float(1.0 - stats.chi2.cdf(w_elderly, df=1))
    
    omnibus_w = ols_res_e["omnibus_stat"]
    omnibus_p = ols_res_e["omnibus_p"]
    
    # Holm adjustment across the 4 tests
    family_pvals = [omnibus_p, p_young, p_middle, p_elderly]
    reject, pvals_holm, _, _ = multipletests(family_pvals, alpha=0.05, method="holm")
    
    print("\n" + "=" * 80)
    print("SHARED FAMILY OF FOUR HC3 ROBUST WALD TESTS (alpha=0.05, Holm adjusted)")
    print("=" * 80)
    print(f"1. Age x Sex Interaction Omnibus (df=2):")
    print(f"   Wald chi2 = {omnibus_w:.4f}, p_unadj = {omnibus_p:.4e}, p_holm = {pvals_holm[0]:.4e}, Sig = {reject[0]}")
    print(f"2. Young Adults Contrast (y0 vs y1, df=1):")
    print(f"   Delta = {c_young[1]:+.4f} (SE={c_young[2]:.4f}), Wald chi2 = {w_young:.4f}, p_unadj = {p_young:.4e}, p_holm = {pvals_holm[1]:.4e}, Sig = {reject[1]}")
    print(f"3. Middle Ager Contrast (y0 vs y1, df=1):")
    print(f"   Delta = {c_middle[1]:+.4f} (SE={c_middle[2]:.4f}), Wald chi2 = {w_middle:.4f}, p_unadj = {p_middle:.4e}, p_holm = {pvals_holm[2]:.4e}, Sig = {reject[2]}")
    print(f"4. Elderly Contrast (y0 vs y1, df=1):")
    print(f"   Delta = {c_elderly[1]:+.4f} (SE={c_elderly[2]:.4f}), Wald chi2 = {w_elderly:.4f}, p_unadj = {p_elderly:.4e}, p_holm = {pvals_holm[3]:.4e}, Sig = {reject[3]}")
    print("-" * 80)
    print(f"Standardized Contrast Delta_std:")
    w_std = (c_std[1] / c_std[2]) ** 2
    p_std = float(1.0 - stats.chi2.cdf(w_std, df=1))
    print(f"   Delta_std = {c_std[1]:+.4f} (SE={c_std[2]:.4f}), Wald chi2 = {w_std:.4f}, p_unadj = {p_std:.4e}")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 2. Stratified Participant Bootstrap for 95% Pointwise CIs
    # -------------------------------------------------------------------------
    print("\nRunning 1,000 stratified participant bootstrap resamples for 95% pointwise CIs...")
    t_boot_0 = time.perf_counter()
    ci_bootstrap = run_stratified_bootstrap(df_w1, endpoint="E_i", n_boot=1000, seed=20261020)
    t_boot_el = time.perf_counter() - t_boot_0
    print(f"Bootstrap completed in {t_boot_el:.1f} s.")
    for k, (ci_lo, ci_hi) in ci_bootstrap.items():
        print(f"  {k:15s}: 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        
    # -------------------------------------------------------------------------
    # 3. Unadjusted Contrasts and Other Endpoints
    # -------------------------------------------------------------------------
    ols_unadj_e = compute_ols_contrasts(df_w1, endpoint="E_i", adjust_covariates=False)
    ols_res_a = compute_ols_contrasts(df_w1, endpoint="A_i", adjust_covariates=True)
    ols_res_s = compute_ols_contrasts(df_w1, endpoint="S_i", adjust_covariates=True)
    
    # Wear / Compliance Contrasts for Double Dissociation
    ols_res_occ = compute_ols_contrasts(df_w1, endpoint="w1_mean_occ_5m", adjust_covariates=False)
    ols_res_night = compute_ols_contrasts(df_w1, endpoint="w1_mean_clock_night", adjust_covariates=False)
    
    # -------------------------------------------------------------------------
    # 4. Second-Window Temporal Persistence Analysis (N=2,324)
    # -------------------------------------------------------------------------
    df_w2 = df_all[df_all["eligible_w2"]].copy()
    n_w2 = len(df_w2)
    print(f"\n--- Second-Window Temporal Persistence (N = {n_w2}) ---")
    
    print("\nSecond-Window Persistence Metrics by Age Group and Salutation:")
    persist_summary = df_w2.groupby(["age_group", "y"])[
        ["w2_delta_mse", "w2_r2_persist", "w2_corr_persist", "delta_T_w1_w2"]
    ].agg(["mean", "median", "std"])
    print(persist_summary)
    
    # Overall and Age-Stratified delta MSE and R2
    print("\nSecond-Window Improvement in MSE (MSE_base - MSE_full):")
    for a in ["Young Adults", "Middle Ager", "Elderly"]:
        sub_0 = df_w2[(df_w2["age_group"] == a) & (df_w2["y"] == 0)]
        sub_1 = df_w2[(df_w2["age_group"] == a) & (df_w2["y"] == 1)]
        d_mse_0 = sub_0["w2_delta_mse"].mean()
        d_mse_1 = sub_1["w2_delta_mse"].mean()
        r2_0 = sub_0["w2_r2_persist"].mean()
        r2_1 = sub_1["w2_r2_persist"].mean()
        print(f"  {a:15s}: y0 delta_MSE={d_mse_0:+.4f} (R2={r2_0:+.4f}) | y1 delta_MSE={d_mse_1:+.4f} (R2={r2_1:+.4f})")
        
    # -------------------------------------------------------------------------
    # 5. Compile and Save Final Contrasts Table
    # -------------------------------------------------------------------------
    records: List[Dict[str, Any]] = []
    
    # Row 1: Interaction Omnibus
    records.append({
        "family": "primary_four_test_family",
        "endpoint": "E_i",
        "contrast_id": "omnibus_age_x_sex",
        "contrast_name": "Age x Sex Interaction Omnibus",
        "contrast_type": "adjusted_hc3_ols",
        "df": 2,
        "estimate": np.nan,
        "se_hc3": np.nan,
        "ci95_lower": np.nan,
        "ci95_upper": np.nan,
        "stat_val": omnibus_w,
        "stat_name": "Wald_chi2",
        "p_unadjusted": omnibus_p,
        "p_holm": pvals_holm[0],
        "sig_holm_05": bool(reject[0]),
        "n_y0": int(df_w1[df_w1["y"] == 0].shape[0]),
        "n_y1": int(df_w1[df_w1["y"] == 1].shape[0]),
        "mean_y0": float(df_w1[df_w1["y"] == 0]["E_i"].mean()),
        "mean_y1": float(df_w1[df_w1["y"] == 1]["E_i"].mean()),
        "description": "2-df interaction omnibus testing H0: Delta_Young == Delta_Middle == Delta_Elderly",
    })
    
    # Rows 2-4: Primary Within-Age Contrasts on E_i
    contrast_keys = [
        ("Young Adults", "within_age_young", 1, p_young, pvals_holm[1], reject[1]),
        ("Middle Ager", "within_age_middle", 2, p_middle, pvals_holm[2], reject[2]),
        ("Elderly", "within_age_elderly", 3, p_elderly, pvals_holm[3], reject[3]),
    ]
    
    for age_cat, c_id, test_num, p_unadj, p_h, rej in contrast_keys:
        est = ols_res_e["contrasts"][age_cat][1]
        se = ols_res_e["contrasts"][age_cat][2]
        stat_w = (est / se) ** 2
        sub = df_w1[df_w1["age_group"] == age_cat]
        sub_0 = sub[sub["y"] == 0]
        sub_1 = sub[sub["y"] == 1]
        ci_lo, ci_hi = ci_bootstrap[age_cat]
        
        records.append({
            "family": "primary_four_test_family",
            "endpoint": "E_i",
            "contrast_id": c_id,
            "contrast_name": f"{age_cat} (y0 vs y1)",
            "contrast_type": "adjusted_hc3_ols",
            "df": 1,
            "estimate": est,
            "se_hc3": se,
            "ci95_lower": ci_lo,
            "ci95_upper": ci_hi,
            "stat_val": stat_w,
            "stat_name": "Wald_chi2",
            "p_unadjusted": p_unadj,
            "p_holm": p_h,
            "sig_holm_05": bool(rej),
            "n_y0": len(sub_0),
            "n_y1": len(sub_1),
            "mean_y0": float(sub_0["E_i"].mean()),
            "mean_y1": float(sub_1["E_i"].mean()),
            "description": f"Observation-adjusted 1-df within-age female-male contrast on E_i for {age_cat}",
        })
        
    # Standardized Contrast on E_i
    records.append({
        "family": "secondary_summary",
        "endpoint": "E_i",
        "contrast_id": "standardized_delta_std",
        "contrast_name": "Standardized Delta_std",
        "contrast_type": "adjusted_hc3_ols",
        "df": 1,
        "estimate": c_std[1],
        "se_hc3": c_std[2],
        "ci95_lower": ci_bootstrap["Standardized"][0],
        "ci95_upper": ci_bootstrap["Standardized"][1],
        "stat_val": w_std,
        "stat_name": "Wald_chi2",
        "p_unadjusted": p_std,
        "p_holm": np.nan,
        "sig_holm_05": bool(p_std < 0.05),
        "n_y0": int(df_w1[df_w1["y"] == 0].shape[0]),
        "n_y1": int(df_w1[df_w1["y"] == 1].shape[0]),
        "mean_y0": float(df_w1[df_w1["y"] == 0]["E_i"].mean()),
        "mean_y1": float(df_w1[df_w1["y"] == 1]["E_i"].mean()),
        "description": "Fixed-weight standardized contrast Delta_std = sum_a w_a Delta_a",
    })
    
    # Unadjusted Contrasts on E_i
    for age_cat in ["Young Adults", "Middle Ager", "Elderly"]:
        est = ols_unadj_e["contrasts"][age_cat][1]
        se = ols_unadj_e["contrasts"][age_cat][2]
        stat_w = (est / se) ** 2
        p_u = float(1.0 - stats.chi2.cdf(stat_w, df=1))
        sub = df_w1[df_w1["age_group"] == age_cat]
        records.append({
            "family": "sensitivity_unadjusted",
            "endpoint": "E_i",
            "contrast_id": f"unadj_{age_cat.lower().replace(' ', '_')}",
            "contrast_name": f"Unadjusted {age_cat} (y0 vs y1)",
            "contrast_type": "unadjusted",
            "df": 1,
            "estimate": est,
            "se_hc3": se,
            "ci95_lower": est - 1.96 * se,
            "ci95_upper": est + 1.96 * se,
            "stat_val": stat_w,
            "stat_name": "Wald_chi2",
            "p_unadjusted": p_u,
            "p_holm": np.nan,
            "sig_holm_05": bool(p_u < 0.05),
            "n_y0": len(sub[sub["y"] == 0]),
            "n_y1": len(sub[sub["y"] == 1]),
            "mean_y0": float(sub[sub["y"] == 0]["E_i"].mean()),
            "mean_y1": float(sub[sub["y"] == 1]["E_i"].mean()),
            "description": f"Unadjusted difference in sample means on E_i for {age_cat}",
        })
        
    # Amplitude A_i Contrasts
    for age_cat in ["Young Adults", "Middle Ager", "Elderly", "Standardized"]:
        est = ols_res_a["contrasts"][age_cat][1]
        se = ols_res_a["contrasts"][age_cat][2]
        stat_w = (est / se) ** 2
        p_u = float(1.0 - stats.chi2.cdf(stat_w, df=1))
        sub = df_w1[df_w1["age_group"] == age_cat] if age_cat != "Standardized" else df_w1
        records.append({
            "family": "sensitivity_amplitude",
            "endpoint": "A_i",
            "contrast_id": f"amplitude_{age_cat.lower().replace(' ', '_')}",
            "contrast_name": f"Amplitude {age_cat} (y0 vs y1)",
            "contrast_type": "adjusted_hc3_ols",
            "df": 1,
            "estimate": est,
            "se_hc3": se,
            "ci95_lower": est - 1.96 * se,
            "ci95_upper": est + 1.96 * se,
            "stat_val": stat_w,
            "stat_name": "Wald_chi2",
            "p_unadjusted": p_u,
            "p_holm": np.nan,
            "sig_holm_05": bool(p_u < 0.05),
            "n_y0": len(sub[sub["y"] == 0]),
            "n_y1": len(sub[sub["y"] == 1]),
            "mean_y0": float(sub[sub["y"] == 0]["A_i"].mean()),
            "mean_y1": float(sub[sub["y"] == 1]["A_i"].mean()),
            "description": f"Adjusted cosinor amplitude (bpm) contrast for {age_cat}",
        })
        
    # Wear Compliance Contrasts (Double Dissociation)
    for c_var, label in [("w1_mean_occ_5m", "Daily 5m Occupancy Rate"), ("w1_mean_clock_night", "Nocturnal Sleep Wear Coverage")]:
        for age_cat in ["Young Adults", "Middle Ager", "Elderly"]:
            sub = df_w1[df_w1["age_group"] == age_cat]
            sub_0 = sub[sub["y"] == 0][c_var].values
            sub_1 = sub[sub["y"] == 1][c_var].values
            m0, m1 = float(np.mean(sub_0)), float(np.mean(sub_1))
            diff = m0 - m1
            ttest = stats.ttest_ind(sub_0, sub_1, equal_var=False)
            records.append({
                "family": "wear_compliance_double_dissociation",
                "endpoint": c_var,
                "contrast_id": f"{c_var}_{age_cat.lower().replace(' ', '_')}",
                "contrast_name": f"{label} {age_cat} (y0 vs y1)",
                "contrast_type": "welch_ttest",
                "df": 1,
                "estimate": diff,
                "se_hc3": np.nan,
                "ci95_lower": np.nan,
                "ci95_upper": np.nan,
                "stat_val": float(ttest.statistic),
                "stat_name": "Welch_t",
                "p_unadjusted": float(ttest.pvalue),
                "p_holm": np.nan,
                "sig_holm_05": bool(ttest.pvalue < 0.05),
                "n_y0": len(sub_0),
                "n_y1": len(sub_1),
                "mean_y0": m0,
                "mean_y1": m1,
                "description": f"Wear compliance contrast for {label} in {age_cat}",
            })

    out_csv = os.path.join(
        PROJECT_ROOT,
        "experiments/female_infradian_cycles_v2/results/calibrated_group_contrasts.csv"
    )
    res_df = pd.DataFrame(records)
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(res_df)} contrast records to {out_csv}")
    
    elapsed = time.perf_counter() - t_start
    print(f"Summarize infradian contrasts complete in {elapsed:.1f} s.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
