#!/usr/bin/env python3
"""
BMI & Metabolic Stratification of Infradian Heart Rate Rhythms in Reproductive-Age Females
Target Script: experiments/female_infradian_extensions/scripts/03_bmi_metabolic_stratification.py

Outputs:
- experiments/female_infradian_extensions/results/bmi_infradian_contrasts.csv
- experiments/female_infradian_extensions/figures/fig3_bmi_amplitude_and_period_violins.png

Methodology & Statistical Rigor:
1. Cohort Screening & Stratification:
   - Target cohort: Screened reproductive-age females (sex == 'Female' & age_group == 'Young Adults' & screened_pass == True, N = 502).
   - Stratified into four standard WHO BMI categories:
     * Underweight: BMI < 18.5 (N = 19)
     * Normal:      18.5 <= BMI < 25.0 (N = 296, Reference)
     * Overweight:  25.0 <= BMI < 30.0 (N = 123)
     * Obese:       BMI >= 30.0 (N = 64)
2. Evaluated Infradian Metrics:
   - Resting HR (hr_p10): Cosinor amplitude, LS power, LS SNR, cycle period, delta R^2, rhythmicity rates.
   - Nocturnal HR (night_hr): Cosinor amplitude, LS power, LS SNR, cycle period, delta R^2, rhythmicity rates.
   - Wear compliance controls: occ_daily_mean (mean daily fraction of valid bins), n_adequate_days.
3. Statistical Analyses:
   - Summary statistics (Mean, SD, Median, IQR, Q25, Q75, rhythmicity proportions at p < 0.05 and p < 0.01).
   - Non-parametric omnibus test: Kruskal-Wallis H-test across the 4 BMI tiers.
   - Pairwise contrasts:
     * Normal vs. Overweight
     * Normal vs. Obese
     * Normal vs. Underweight
     * Obese vs. Non-Obese (Normal + Overweight + Underweight)
     Reporting difference in means, Welch's t-test with Welch-Satterthwaite df, 95% CI, Cohen's d, and Mann-Whitney U test.
   - Multivariable ANCOVA OLS regression:
     Amplitude = beta_0 + beta_overweight * Overweight + beta_obese * Obese + beta_underweight * Underweight + beta_occ * Occupancy + beta_days * Days
     Reporting both conventional OLS and HC3 robust standard errors, t-statistics, p-values, model R^2, adjusted R^2, and F-statistic.
4. Publication Figure 3:
   - 3-panel high-resolution publication figure (300 dpi) displaying violin plots, overlaid boxplots, individual jitter points, sample sizes, and statistical brackets.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

# Paths
REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments/female_infradian_extensions"
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR = EXP_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = REPO_ROOT / "experiments/female_infradian_cycles/results/infradian_metrics_by_participant.csv"
CACHE_5MIN_DIR = REPO_ROOT / "experiments/apple_cohort_scaling/artifacts/cache_5min"

OUTPUT_CSV = RESULTS_DIR / "bmi_infradian_contrasts.csv"
OUTPUT_FIG = FIG_DIR / "fig3_bmi_amplitude_and_period_violins.png"

BMI_ORDER = ["underweight", "normal", "overweight", "obese"]
BMI_LABELS = {
    "underweight": "Underweight (<18.5)",
    "normal": "Normal (18.5–24.9)",
    "overweight": "Overweight (25–29.9)",
    "obese": "Obese (≥30)",
}
BMI_COLORS = {
    "underweight": "#3498DB",  # Blue
    "normal": "#2ECC71",       # Emerald green
    "overweight": "#F39C12",   # Amber / Orange
    "obese": "#E74C3C",        # Red
}


# =============================================================================
# 1. Cohort Loading & Extraction
# =============================================================================
def load_and_screen_cohort() -> pd.DataFrame:
    """Loads metrics CSV, filters to young reproductive females, extracts occupancy."""
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Missing input metrics CSV: {METRICS_CSV}")

    print(f"Loading participant metrics from: {METRICS_CSV}")
    df_raw = pd.read_csv(METRICS_CSV)

    # Screening filter: Young Adult Females who passed screening
    cohort = df_raw[
        (df_raw["sex"] == "Female") &
        (df_raw["age_group"] == "Young Adults") &
        (df_raw["screened_pass"] == True)
    ].copy()

    n_cohort = len(cohort)
    print(f"Screened Reproductive Females cohort size: N = {n_cohort}")
    if n_cohort != 502:
        print(f"WARNING: Expected N = 502, but found N = {n_cohort}")

    # Verify BMI group counts
    bmi_counts = cohort["bmi_group"].value_counts().to_dict()
    print("Cohort BMI group breakdown:")
    for bg in BMI_ORDER:
        cnt = bmi_counts.get(bg, 0)
        print(f"  {bg:12s}: N = {cnt}")

    # Compute mean daily occupancy from 5-minute cache
    print("Extracting mean daily wear occupancy from 5-minute cache...")
    t0 = time.perf_counter()
    occ_means = {}
    for cid in cohort["customer_id"]:
        npz_path = CACHE_5MIN_DIR / f"customer_{cid}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing 5-minute cache for customer: {cid}")
        with np.load(npz_path) as npz:
            hr = npz["hr"]
            valid_bins = np.isfinite(hr).sum(axis=1)
            is_adeq = valid_bins >= 120
            if np.sum(is_adeq) > 0:
                occ_daily = valid_bins[is_adeq] / 288.0
                occ_means[cid] = float(np.mean(occ_daily))
            else:
                occ_means[cid] = float("nan")

    cohort["occ_daily_mean"] = cohort["customer_id"].map(occ_means)
    dt = time.perf_counter() - t0
    print(f"Computed wear occupancy for {len(cohort)} participants in {dt:.2f}s")
    print(f"Overall mean daily wear occupancy: {cohort['occ_daily_mean'].mean():.3f} +/- {cohort['occ_daily_mean'].std():.3f}")

    return cohort


# =============================================================================
# 2. Summary Statistics & Rhythmicity Rates
# =============================================================================
def compute_group_summaries(df: pd.DataFrame, metric_cols: List[str]) -> List[Dict[str, Any]]:
    """Computes Mean, SD, Median, Q25, Q75, IQR, and rhythmicity rates per BMI group and overall."""
    rows: List[Dict[str, Any]] = []

    groups = [("overall", df)] + [(bg, df[df["bmi_group"] == bg]) for bg in BMI_ORDER]

    for m in metric_cols:
        for grp_name, grp_df in groups:
            series = grp_df[m].dropna()
            n = len(series)
            if n == 0:
                continue

            mean_val = float(series.mean())
            sd_val = float(series.std(ddof=1)) if n > 1 else 0.0
            med_val = float(series.median())
            q25_val = float(series.quantile(0.25))
            q75_val = float(series.quantile(0.75))
            iqr_val = float(q75_val - q25_val)

            # Rhythmicity rate if metric corresponds to amplitude or has matching p-value column
            rhythm_p05 = float("nan")
            rhythm_p01 = float("nan")
            if "hr_p10" in m:
                pval_col = "cos_pvalue_hr_p10"
            elif "night_hr" in m:
                pval_col = "cos_pvalue_night_hr"
            else:
                pval_col = None

            if pval_col and pval_col in grp_df.columns:
                pvals = grp_df[pval_col].dropna()
                if len(pvals) > 0:
                    rhythm_p05 = float((pvals < 0.05).mean())
                    rhythm_p01 = float((pvals < 0.01).mean())

            rows.append({
                "analysis_type": "group_summary",
                "metric": m,
                "bmi_group": grp_name,
                "n": n,
                "mean": mean_val,
                "sd": sd_val,
                "median": med_val,
                "q25": q25_val,
                "q75": q75_val,
                "iqr": iqr_val,
                "rhythm_rate_p05": rhythm_p05,
                "rhythm_rate_p01": rhythm_p01,
            })

    return rows


# =============================================================================
# 3. Non-Parametric Omnibus Kruskal-Wallis Test
# =============================================================================
def compute_kruskal_wallis(df: pd.DataFrame, metric_cols: List[str]) -> List[Dict[str, Any]]:
    """Runs Kruskal-Wallis H-test across the 4 BMI tiers for each metric."""
    rows: List[Dict[str, Any]] = []

    for m in metric_cols:
        groups_data = []
        for bg in BMI_ORDER:
            vals = df[df["bmi_group"] == bg][m].dropna().values
            groups_data.append(vals)

        # Check that at least two groups have data
        non_empty = [g for g in groups_data if len(g) > 0]
        if len(non_empty) >= 2:
            h_stat, p_val = stats.kruskal(*groups_data)
            df_kruskal = len(groups_data) - 1
            n_total = sum(len(g) for g in groups_data)
        else:
            h_stat, p_val, df_kruskal, n_total = float("nan"), float("nan"), 0, 0

        rows.append({
            "analysis_type": "omnibus_test",
            "metric": m,
            "test_type": "Kruskal-Wallis",
            "h_statistic": float(h_stat),
            "p_value": float(p_val),
            "df": df_kruskal,
            "n_total": n_total,
        })

    return rows


# =============================================================================
# 4. Pairwise Statistical Contrasts
# =============================================================================
def compute_pairwise_contrasts(df: pd.DataFrame, metric_cols: List[str]) -> List[Dict[str, Any]]:
    """Runs Welch's t-test and Mann-Whitney U test between specified BMI pairs."""
    rows: List[Dict[str, Any]] = []

    comparisons = [
        ("Normal vs Overweight", df[df["bmi_group"] == "normal"], df[df["bmi_group"] == "overweight"]),
        ("Normal vs Obese", df[df["bmi_group"] == "normal"], df[df["bmi_group"] == "obese"]),
        ("Normal vs Underweight", df[df["bmi_group"] == "normal"], df[df["bmi_group"] == "underweight"]),
        ("Obese vs Non-Obese", df[df["bmi_group"] == "obese"], df[df["bmi_group"] != "obese"]),
    ]

    for m in metric_cols:
        for comp_name, df1, df2 in comparisons:
            s1 = df1[m].dropna().values
            s2 = df2[m].dropna().values
            n1 = len(s1)
            n2 = len(s2)

            if n1 < 2 or n2 < 2:
                continue

            m1, m2 = float(np.mean(s1)), float(np.mean(s2))
            v1, v2 = float(np.var(s1, ddof=1)), float(np.var(s2, ddof=1))
            diff = m1 - m2

            # Welch's t-test
            t_res = stats.ttest_ind(s1, s2, equal_var=False)
            t_stat = float(t_res.statistic)
            t_pval = float(t_res.pvalue)

            # Welch-Satterthwaite degrees of freedom
            se_diff = math.sqrt((v1 / n1) + (v2 / n2))
            if se_diff > 1e-15:
                num_df = ((v1 / n1) + (v2 / n2)) ** 2
                den_df = ((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1))
                df_welch = num_df / den_df
                t_crit = stats.t.ppf(0.975, df=df_welch)
                ci_low = diff - t_crit * se_diff
                ci_high = diff + t_crit * se_diff
            else:
                df_welch = float("inf")
                ci_low, ci_high = diff, diff

            # Cohen's d using pooled SD
            s_pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 0.0
            cohens_d = diff / s_pooled if s_pooled > 1e-12 else 0.0

            # Mann-Whitney U test
            u_res = stats.mannwhitneyu(s1, s2, alternative="two-sided")
            u_stat = float(u_res.statistic)
            u_pval = float(u_res.pvalue)

            rows.append({
                "analysis_type": "pairwise_contrast",
                "metric": m,
                "comparison": comp_name,
                "n1": n1,
                "n2": n2,
                "mean1": m1,
                "mean2": m2,
                "diff_mean": diff,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "cohens_d": cohens_d,
                "welch_t_stat": t_stat,
                "welch_df": df_welch,
                "welch_pvalue": t_pval,
                "mwu_u_stat": u_stat,
                "mwu_pvalue": u_pval,
            })

    return rows


# =============================================================================
# 5. Multivariable ANCOVA OLS Regression
# =============================================================================
def compute_ancova_regressions(df: pd.DataFrame, dependent_vars: List[str]) -> List[Dict[str, Any]]:
    """Fits multivariable OLS ANCOVA models controlling for occupancy and days."""
    rows: List[Dict[str, Any]] = []

    for dep_var in dependent_vars:
        sub = df.dropna(subset=[dep_var, "occ_daily_mean", "n_adequate_days"]).copy()
        n_obs = len(sub)
        if n_obs < 30:
            continue

        # Indicator coding with Normal BMI as reference
        sub["overweight"] = (sub["bmi_group"] == "overweight").astype(float)
        sub["obese"] = (sub["bmi_group"] == "obese").astype(float)
        sub["underweight"] = (sub["bmi_group"] == "underweight").astype(float)

        X = sub[["overweight", "obese", "underweight", "occ_daily_mean", "n_adequate_days"]].copy()
        X = sm.add_constant(X)
        y = sub[dep_var]

        # Fit standard OLS
        ols_model = sm.OLS(y, X).fit()
        # Fit with HC3 heteroskedasticity-robust covariance
        hc3_model = sm.OLS(y, X).fit(cov_type="HC3")

        r2 = float(ols_model.rsquared)
        adj_r2 = float(ols_model.rsquared_adj)
        f_stat = float(ols_model.fvalue)
        f_pval = float(ols_model.f_pvalue)

        conf_int = ols_model.conf_int()

        for term in X.columns:
            coef = float(ols_model.params[term])
            se_ols = float(ols_model.bse[term])
            t_ols = float(ols_model.tvalues[term])
            p_ols = float(ols_model.pvalues[term])

            se_hc3 = float(hc3_model.bse[term])
            t_hc3 = float(hc3_model.tvalues[term])
            p_hc3 = float(hc3_model.pvalues[term])

            ci_low = float(conf_int.loc[term, 0])
            ci_high = float(conf_int.loc[term, 1])

            rows.append({
                "analysis_type": "ancova_model",
                "dependent_var": dep_var,
                "term": term,
                "coef": coef,
                "std_err": se_ols,
                "t_stat": t_ols,
                "p_value": p_ols,
                "std_err_hc3": se_hc3,
                "t_stat_hc3": t_hc3,
                "p_value_hc3": p_hc3,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "model_r2": r2,
                "model_adj_r2": adj_r2,
                "model_f_stat": f_stat,
                "model_f_pvalue": f_pval,
                "n_obs": n_obs,
            })

    return rows


# =============================================================================
# 6. Publication Figure 3 Generation
# =============================================================================
def draw_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    color: str = "#333333",
    lw: float = 1.0,
    fontsize: float = 8.5,
):
    """Draws a scientific bracket with significance or p-value label."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, lw=lw)
    ax.text(
        (x1 + x2) * 0.5,
        y + h * 1.15,
        text,
        ha="center",
        va="bottom",
        color=color,
        fontsize=fontsize,
        weight="medium",
    )


def generate_figure3(
    df: pd.DataFrame,
    pairwise_contrasts: List[Dict[str, Any]],
    omnibus_tests: List[Dict[str, Any]],
    out_path: Path,
):
    """Generates the 3-panel publication figure comparing BMI categories."""
    print("Generating Figure 3: fig3_bmi_amplitude_and_period_violins.png...")
    
    # Configure matplotlib style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.1,
        "grid.color": "#EAEAEA",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6), dpi=300)

    # Dictionary lookup for omnibus p-values
    kw_pvals = {r["metric"]: (r["h_statistic"], r["p_value"]) for r in omnibus_tests if r["analysis_type"] == "omnibus_test"}

    # Dictionary lookup for pairwise p-values: (metric, comp_name) -> pval
    pair_pvals = {
        (r["metric"], r["comparison"]): r["welch_pvalue"]
        for r in pairwise_contrasts if r["analysis_type"] == "pairwise_contrast"
    }

    panel_configs = [
        {
            "ax": axes[0],
            "metric": "cos_amplitude_hr_p10",
            "panel_letter": "(a)",
            "panel_title": "Resting HR Infradian Amplitude ($A_{10}$)",
            "ylabel": "Infradian Amplitude (bpm)",
            "ylim": (0.0, 5.6),
            "bracket_base": 3.7,
            "bracket_step": 0.55,
            "ref_line": None,
        },
        {
            "ax": axes[1],
            "metric": "cos_amplitude_night_hr",
            "panel_letter": "(b)",
            "panel_title": "Nocturnal HR Infradian Amplitude ($A_{\\mathrm{night}}$)",
            "ylabel": "Nocturnal Amplitude (bpm)",
            "ylim": (0.0, 16.5),
            "bracket_base": 11.2,
            "bracket_step": 1.6,
            "ref_line": None,
        },
        {
            "ax": axes[2],
            "metric": "cos_period_hr_p10",
            "panel_letter": "(c)",
            "panel_title": "Best-Fit Infradian Cycle Period ($T^*$)",
            "ylabel": "Cycle Period (days)",
            "ylim": (18.0, 43.0),
            "bracket_base": 35.8,
            "bracket_step": 2.2,
            "ref_line": 28.0,
        },
    ]

    for pcfg in panel_configs:
        ax = pcfg["ax"]
        m = pcfg["metric"]
        
        # Prepare data per BMI group
        data_per_group = []
        n_per_group = []
        for bg in BMI_ORDER:
            s = df[df["bmi_group"] == bg][m].dropna().values
            data_per_group.append(s)
            n_per_group.append(len(s))

        # Position coordinates on x-axis: 0, 1, 2, 3
        positions = np.arange(len(BMI_ORDER))

        # 1. Violin plots (clip extrema within data bounds)
        parts = ax.violinplot(
            data_per_group,
            positions=positions,
            widths=0.66,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for idx, (pc, bg) in enumerate(zip(parts["bodies"], BMI_ORDER)):
            pc.set_facecolor(BMI_COLORS[bg])
            pc.set_edgecolor(BMI_COLORS[bg])
            pc.set_alpha(0.38)
            pc.set_linewidth(1.2)

        # 2. Overlaid boxplots
        bp = ax.boxplot(
            data_per_group,
            positions=positions,
            widths=0.18,
            patch_artist=True,
            showcaps=True,
            showfliers=False,
            medianprops=dict(color="#111111", linewidth=2.2),
            whiskerprops=dict(color="#444444", linewidth=1.2),
            capprops=dict(color="#444444", linewidth=1.2),
            boxprops=dict(facecolor="#FFFFFF", edgecolor="#333333", linewidth=1.2, alpha=0.92),
        )

        # 3. Individual jitter points
        rng = np.random.default_rng(20261020)
        for idx, (vals, bg) in enumerate(zip(data_per_group, BMI_ORDER)):
            jitter = rng.uniform(-0.09, 0.09, size=len(vals))
            ax.scatter(
                positions[idx] + jitter,
                vals,
                s=11,
                color=BMI_COLORS[bg],
                alpha=0.48,
                edgecolors="none",
                zorder=3,
            )

        # 4. Reference line if requested
        if pcfg["ref_line"] is not None:
            ax.axhline(
                pcfg["ref_line"],
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
                label="Canonical 28.0 d",
                zorder=1,
            )
            ax.legend(loc="lower right", frameon=True, fontsize=8.5, framealpha=0.9)

        # 5. X-axis tick labels: compact multi-line format
        bmi_subtexts = {
            "underweight": "BMI < 18.5",
            "normal": "18.5–24.9",
            "overweight": "25.0–29.9",
            "obese": "BMI ≥ 30",
        }
        bmi_display_names = {
            "underweight": "Underweight",
            "normal": "Normal",
            "overweight": "Overweight",
            "obese": "Obese",
        }
        x_labels = [
            f"{bmi_display_names[bg]}\n{bmi_subtexts[bg]}\n($N={n_per_group[i]}$)"
            for i, bg in enumerate(BMI_ORDER)
        ]
        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels, fontsize=9.2, weight="medium")
        ax.set_ylabel(pcfg["ylabel"], fontsize=11.0, weight="bold")

        # 6. Title and Omnibus subtitle
        h_val, p_val = kw_pvals.get(m, (float("nan"), float("nan")))
        sig_str = "n.s." if p_val >= 0.05 else f"p = {p_val:.3f}"
        full_title = f"{pcfg['panel_letter']} {pcfg['panel_title']}\nKruskal-Wallis: $H = {h_val:.2f},\\ p = {p_val:.3f}$ ({sig_str})"
        ax.set_title(full_title, fontsize=11.0, weight="bold", pad=10)
        ax.set_ylim(pcfg["ylim"])
        ax.grid(axis="y", alpha=0.55)

        # 7. Pairwise statistical brackets
        h_br = pcfg["bracket_step"] * 0.22
        
        # Bracket 1: Normal (pos 1) vs Overweight (pos 2)
        p_norm_ow = pair_pvals.get((m, "Normal vs Overweight"), float("nan"))
        txt_ow = f"p = {p_norm_ow:.2f}" if math.isfinite(p_norm_ow) else "n.s."
        y1 = pcfg["bracket_base"]
        draw_bracket(ax, 1.0, 2.0, y1, h_br, txt_ow, fontsize=8.5)

        # Bracket 2: Normal (pos 1) vs Obese (pos 3)
        p_norm_ob = pair_pvals.get((m, "Normal vs Obese"), float("nan"))
        txt_ob = f"p = {p_norm_ob:.2f}" if math.isfinite(p_norm_ob) else "n.s."
        y2 = y1 + pcfg["bracket_step"]
        draw_bracket(ax, 1.0, 3.0, y2, h_br, txt_ob, fontsize=8.5)

        # Bracket 3: Underweight (pos 0) vs Normal (pos 1)
        p_norm_uw = pair_pvals.get((m, "Normal vs Underweight"), float("nan"))
        txt_uw = f"p = {p_norm_uw:.2f}" if math.isfinite(p_norm_uw) else "n.s."
        y3 = y2 + pcfg["bracket_step"]
        draw_bracket(ax, 0.0, 1.0, y3, h_br, txt_uw, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Publication figure successfully saved to: {out_path}")


# =============================================================================
# 7. Main Execution Pipeline
# =============================================================================
def main() -> int:
    t_start = time.perf_counter()
    print("=" * 80)
    print("Module 3: BMI & Metabolic Stratification of Female Infradian Rhythms")
    print("=" * 80)

    # 1. Load and screen cohort
    df = load_and_screen_cohort()

    # Define metrics of interest
    eval_metrics = [
        "cos_amplitude_hr_p10",
        "cos_amplitude_night_hr",
        "ls_power_hr_p10",
        "ls_power_night_hr",
        "cos_period_hr_p10",
        "cos_period_night_hr",
        "ls_period_hr_p10",
        "ls_period_night_hr",
        "ls_snr_hr_p10",
        "ls_snr_night_hr",
        "cos_delta_r2_hr_p10",
        "cos_delta_r2_night_hr",
        "occ_daily_mean",
        "n_adequate_days",
    ]

    # 2. Compute group summary statistics
    print("\nComputing group summaries...")
    summary_rows = compute_group_summaries(df, eval_metrics)

    # 3. Compute omnibus Kruskal-Wallis tests
    print("Computing non-parametric Kruskal-Wallis omnibus tests...")
    omnibus_rows = compute_kruskal_wallis(df, eval_metrics)

    # 4. Compute pairwise contrasts
    print("Computing pairwise Welch's t-tests and Mann-Whitney U contrasts...")
    pairwise_rows = compute_pairwise_contrasts(df, eval_metrics)

    # 5. Compute ANCOVA multivariable regressions
    print("Fitting multivariable ANCOVA OLS regressions...")
    ancova_dep_vars = [
        "cos_amplitude_hr_p10",
        "cos_amplitude_night_hr",
        "ls_power_hr_p10",
        "ls_power_night_hr",
        "cos_period_hr_p10",
        "cos_period_night_hr",
    ]
    ancova_rows = compute_ancova_regressions(df, ancova_dep_vars)

    # Combine all records into single unified CSV
    all_rows: List[Dict[str, Any]] = []
    all_rows.extend(summary_rows)
    all_rows.extend(omnibus_rows)
    all_rows.extend(pairwise_rows)
    all_rows.extend(ancova_rows)

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved comprehensive contrasts and regression results to: {OUTPUT_CSV}")
    print(f"Total rows written: {len(df_out)}")

    # 6. Generate Figure 3
    generate_figure3(df, pairwise_rows, omnibus_rows, OUTPUT_FIG)

    # Print summary highlights
    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY OF KEY EMPIRICAL FINDINGS:")
    print("=" * 80)
    print("Cohort breakdown (Screened Young Females):")
    for bg in BMI_ORDER:
        sub = df[df["bmi_group"] == bg]
        amp10 = sub["cos_amplitude_hr_p10"].dropna()
        amp_night = sub["cos_amplitude_night_hr"].dropna()
        period10 = sub["cos_period_hr_p10"].dropna()
        r05 = (sub["cos_pvalue_hr_p10"] < 0.05).mean()
        print(f"  {bg:12s} (N={len(sub):3d}): "
              f"A_10={amp10.mean():.2f} +/- {amp10.std():.2f} bpm, "
              f"A_night={amp_night.mean():.2f} +/- {amp_night.std():.2f} bpm, "
              f"T*={period10.mean():.2f} +/- {period10.std():.2f} d, "
              f"Rhythmicity(p<0.05)={r05:.1%}")

    kw_amp = next(r for r in omnibus_rows if r["metric"] == "cos_amplitude_hr_p10")
    kw_per = next(r for r in omnibus_rows if r["metric"] == "cos_period_hr_p10")
    print(f"\nOmnibus Kruskal-Wallis across 4 BMI tiers:")
    print(f"  Resting HR Amplitude (cos_amplitude_hr_p10): H = {kw_amp['h_statistic']:.4f}, p = {kw_amp['p_value']:.4f}")
    print(f"  Peak Period (cos_period_hr_p10):             H = {kw_per['h_statistic']:.4f}, p = {kw_per['p_value']:.4f}")

    print("\nANCOVA Regression for Resting HR Amplitude (cos_amplitude_hr_p10):")
    ancova_a10 = [r for r in ancova_rows if r["dependent_var"] == "cos_amplitude_hr_p10"]
    for term_row in ancova_a10:
        print(f"  {term_row['term']:16s}: beta = {term_row['coef']:+7.4f}, "
              f"SE = {term_row['std_err']:6.4f}, t = {term_row['t_stat']:+6.3f}, "
              f"p = {term_row['p_value']:.4f} (HC3 p = {term_row['p_value_hc3']:.4f})")

    t_total = time.perf_counter() - t_start
    print(f"\nCompleted Module 3 pipeline in {t_total:.2f}s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
