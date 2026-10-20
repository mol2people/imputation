#!/usr/bin/env python3
"""
Summarize Female Infradian Contrasts Across Age and Sex Groups
Target Path: experiments/female_infradian_cycles/scripts/02_summarize_infradian_contrasts.py
Input: experiments/female_infradian_cycles/results/infradian_metrics_by_participant.csv
Output: experiments/female_infradian_cycles/results/group_contrasts_age_and_sex.csv

Methodology & Statistical Rigor:
1. Screening Gate:
   - Filters to screened participants (screened_pass == True).
2. Stratification:
   - Target Cohort: Reproductive-age females ((y == 0) & (age_group == "Young Adults")).
   - Negative Control 1: Age-matched males ((y == 1) & (age_group == "Young Adults")).
   - Negative Control 2: Post-menopausal females ((y == 0) & (age_group == "Elderly")).
   - Secondary cohorts: Middle Ager females, Middle Ager males, overall females, overall males.
3. 7 Physiological and Behavioral Signals:
   - hr_p10, night_hr, day_hr, dipping_ratio, occ_daily, nocturnal_compliance, gap_episodes.
4. Evaluated Infradian Metrics (25–35 Day Target Band):
   - cos_amplitude: Cosinor amplitude (bpm or fraction).
   - ls_power: Lomb-Scargle normalized spectral peak power in [0, 1].
   - ls_snr: Lomb-Scargle peak power divided by median flanking background power.
   - cos_delta_r2: Incremental variance explained over linear trend null model.
5. Cohort Metrics & Rhythmicity Rates:
   - Mean, SD, Median, Interquartile Range (IQR = Q75 - Q25).
   - Significant rhythmicity proportion at F-test p < 0.05 and p < 0.01.
6. Pairwise Statistical Contrasts:
   - Target vs Control 1 (Reproductive Females vs Age-matched Males).
   - Target vs Control 2 (Reproductive Females vs Post-menopausal Females).
   - Control 1 vs Control 2 (Age-matched Males vs Post-menopausal Females).
   - Middle Ager Females vs Middle Ager Males.
   - Overall Females vs Overall Males.
   For each contrast:
   - Mean difference: Delta = mean_1 - mean_2.
   - 95% Confidence Interval via Welch's standard error.
   - Cohen's d effect size using pooled standard deviation.
   - Welch's two-sample t-test (unequal variances).
   - Mann-Whitney U rank-sum test with tie adjustment.
7. Multivariable OLS ANCOVA:
   - Model: metric ~ sex + age_group + bmi_group.
   - Solved via OLS normal equations with dummy coding (Young Adults and normal BMI reference).
   - Extracts adjusted sex coefficient (beta_sex), standard error, t-stat, p-value, and model R^2.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))


# -----------------------------------------------------------------------------
# Statistical Helper Functions
# -----------------------------------------------------------------------------
def normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function Phi(z)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def calc_stats(values: Sequence[float]) -> Dict[str, float]:
    """Computes mean, sd, median, iqr, q25, q75 for finite values."""
    clean = sorted([v for v in values if math.isfinite(v)])
    n = len(clean)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "iqr": float("nan"),
        }
    mean_v = sum(clean) / n
    var_v = sum((v - mean_v) ** 2 for v in clean) / (n - 1) if n > 1 else 0.0
    sd_v = math.sqrt(var_v)

    def _quantile(q: float) -> float:
        idx = q * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return clean[lo] + (idx - lo) * (clean[hi] - clean[lo])

    q25 = _quantile(0.25)
    med = _quantile(0.50)
    q75 = _quantile(0.75)
    iqr = q75 - q25

    return {
        "n": n,
        "mean": mean_v,
        "sd": sd_v,
        "median": med,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
    }


def calc_rhythm_rate(pvals: Sequence[float], alpha: float) -> Tuple[float, float, int]:
    """Calculates proportion with p < alpha, standard error, and count."""
    clean = [p for p in pvals if math.isfinite(p)]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), 0
    k = sum(1 for p in clean if p < alpha)
    prop = k / n
    se = math.sqrt(prop * (1.0 - prop) / n) if n > 0 else 0.0
    return prop, se, k


def welch_ttest(
    vals1: Sequence[float], vals2: Sequence[float]
) -> Dict[str, float]:
    """Performs Welch's t-test and Cohen's d between two independent samples."""
    c1 = [v for v in vals1 if math.isfinite(v)]
    c2 = [v for v in vals2 if math.isfinite(v)]
    n1, n2 = len(c1), len(c2)
    if n1 < 2 or n2 < 2:
        return {
            "diff_mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "cohens_d": float("nan"),
            "welch_t_stat": float("nan"),
            "welch_pvalue": float("nan"),
        }

    m1 = sum(c1) / n1
    m2 = sum(c2) / n2
    diff = m1 - m2

    v1 = sum((x - m1) ** 2 for x in c1) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in c2) / (n2 - 1)

    se_welch = math.sqrt((v1 / n1) + (v2 / n2))
    if se_welch <= 1e-15:
        return {
            "diff_mean": diff,
            "ci95_low": diff,
            "ci95_high": diff,
            "cohens_d": 0.0,
            "welch_t_stat": 0.0,
            "welch_pvalue": 1.0,
        }

    t_stat = diff / se_welch

    # Welch-Satterthwaite degrees of freedom
    num_df = ((v1 / n1) + (v2 / n2)) ** 2
    den_df = ((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1))
    df = num_df / den_df if den_df > 1e-15 else float("inf")

    # 95% CI margin of error (critical value ~ 1.960 for large df, or exact t-quantile approximation)
    t_crit = 1.95996 + (2.372 / df) + (2.82 / (df ** 2)) if df < 1000 else 1.95996
    ci_low = diff - t_crit * se_welch
    ci_high = diff + t_crit * se_welch

    # Two-sided p-value via normal approximation (df >= 200 guarantees error < 1e-4)
    # Refined t-distribution approximation via Hill (1970)
    a = df - 0.5
    b = 48.0 * a * a
    z = math.sqrt(a * math.log(1.0 + (t_stat ** 2) / df))
    z_corr = z + (z ** 3 + 3.0 * z) / b
    p_val = 2.0 * (1.0 - normal_cdf(abs(z_corr)))
    p_val = max(0.0, min(1.0, p_val))

    # Cohen's d using pooled standard deviation
    s_pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    cohens_d = diff / s_pooled if s_pooled > 1e-12 else 0.0

    return {
        "diff_mean": diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "cohens_d": cohens_d,
        "welch_t_stat": t_stat,
        "welch_pvalue": p_val,
    }


def mann_whitney_u(
    vals1: Sequence[float], vals2: Sequence[float]
) -> Dict[str, float]:
    """Calculates Mann-Whitney U test with exact tie handling."""
    c1 = [v for v in vals1 if math.isfinite(v)]
    c2 = [v for v in vals2 if math.isfinite(v)]
    n1, n2 = len(c1), len(c2)
    if n1 == 0 or n2 == 0:
        return {"mwu_u_stat": float("nan"), "mwu_pvalue": float("nan")}

    # Combine and sort with group tracking
    combined = [(v, 1) for v in c1] + [(v, 2) for v in c2]
    combined.sort(key=lambda x: x[0])

    N = n1 + n2
    ranks = [0.0] * N
    i = 0
    tie_adjustment = 0.0

    while i < N:
        j = i
        while j < N and combined[j][0] == combined[i][0]:
            j += 1
        tie_count = j - i
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        if tie_count > 1:
            tie_adjustment += tie_count ** 3 - tie_count
        i = j

    # Sum of ranks for group 1
    r1 = sum(ranks[k] for k in range(N) if combined[k][1] == 1)
    u1 = r1 - (n1 * (n1 + 1)) / 2.0

    # Mean and variance of U under null hypothesis
    mu_u = (n1 * n2) / 2.0
    var_u = (n1 * n2 / 12.0) * ((N + 1) - (tie_adjustment / (N * (N - 1)))) if N > 1 else 0.0

    if var_u <= 1e-15:
        return {"mwu_u_stat": u1, "mwu_pvalue": 1.0}

    # Continuity correction
    z_u = (u1 - mu_u - 0.5 * (1 if u1 > mu_u else (-1 if u1 < mu_u else 0))) / math.sqrt(var_u)
    p_val = 2.0 * (1.0 - normal_cdf(abs(z_u)))
    p_val = max(0.0, min(1.0, p_val))

    return {"mwu_u_stat": u1, "mwu_pvalue": p_val}


def solve_linear_system(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Gaussian elimination with partial pivoting for linear system A x = b."""
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    for i in range(n):
        max_r = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[max_r] = M[max_r], M[i]
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            return None
        for j in range(i, n + 1):
            M[i][j] /= pivot
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for j in range(i, n + 1):
                    M[r][j] -= factor * M[i][j]

    return [M[i][n] for i in range(n)]


def invert_matrix(A: List[List[float]]) -> Optional[List[List[float]]]:
    """Inverts an n x n matrix using Gauss-Jordan elimination."""
    n = len(A)
    M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for i in range(n):
        max_r = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[max_r] = M[max_r], M[i]
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            return None
        for j in range(2 * n):
            M[i][j] /= pivot
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for j in range(2 * n):
                    M[r][j] -= factor * M[i][j]

    return [[M[i][n + j] for j in range(n)] for i in range(n)]


def fit_ancova_ols(
    records: List[Dict[str, Any]], metric_key: str
) -> Dict[str, float]:
    """Fits multivariable OLS regression: metric ~ sex + age_group + bmi_group.

    Encodes:
    - Intercept: 1.0
    - sex: Male = 1.0, Female = 0.0
    - age_group: reference "Young Adults", dummies for "Middle Ager" and "Elderly"
    - bmi_group: reference "normal", dummies for "overweight", "obese", "underweight", "other"
    """
    clean_data: List[Tuple[float, int, str, str]] = []
    for r in records:
        val_str = r.get(metric_key, "")
        try:
            val = float(val_str)
            if math.isfinite(val):
                clean_data.append((val, int(r["y"]), str(r["age_group"]), str(r["bmi_group"])))
        except (ValueError, TypeError):
            continue

    N = len(clean_data)
    if N < 50:
        return {
            "ancova_beta_sex": float("nan"),
            "ancova_se_sex": float("nan"),
            "ancova_t_sex": float("nan"),
            "ancova_p_sex": float("nan"),
            "ancova_r2": float("nan"),
        }

    # Discover categories dynamically
    age_dummies = ["Middle Ager", "Elderly"]
    bmi_dummies = ["overweight", "obese", "underweight"]

    # Design matrix columns: [Intercept, Sex_Male, Age_Middle, Age_Elderly, BMI_overweight, BMI_obese, BMI_underweight]
    p = 1 + 1 + len(age_dummies) + len(bmi_dummies)
    X: List[List[float]] = []
    y: List[float] = []

    for val, sex_val, age_grp, bmi_grp in clean_data:
        row = [1.0, float(sex_val)]
        for ag in age_dummies:
            row.append(1.0 if age_grp == ag else 0.0)
        for bg in bmi_dummies:
            row.append(1.0 if bmi_grp == bg else 0.0)
        X.append(row)
        y.append(val)

    # Normal equations: (X^T X) beta = X^T y
    XtX = [[sum(X[k][i] * X[k][j] for k in range(N)) for j in range(p)] for i in range(p)]
    Xty = [sum(X[k][i] * y[k] for k in range(N)) for i in range(p)]

    beta = solve_linear_system(XtX, Xty)
    if beta is None:
        return {
            "ancova_beta_sex": float("nan"),
            "ancova_se_sex": float("nan"),
            "ancova_t_sex": float("nan"),
            "ancova_p_sex": float("nan"),
            "ancova_r2": float("nan"),
        }

    # Invert XtX for covariance matrix
    inv_XtX = invert_matrix(XtX)
    if inv_XtX is None:
        return {
            "ancova_beta_sex": beta[1],
            "ancova_se_sex": float("nan"),
            "ancova_t_sex": float("nan"),
            "ancova_p_sex": float("nan"),
            "ancova_r2": float("nan"),
        }

    # Residual sum of squares and variance
    y_mean = sum(y) / N
    ss_tot = sum((y[k] - y_mean) ** 2 for k in range(N))
    y_hat = [sum(X[k][j] * beta[j] for j in range(p)) for k in range(N)]
    ss_res = sum((y[k] - y_hat[k]) ** 2 for k in range(N))

    df_res = N - p
    sigma2 = ss_res / df_res if df_res > 0 else 0.0
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-15 else 0.0

    beta_sex = beta[1]
    se_sex = math.sqrt(max(0.0, sigma2 * inv_XtX[1][1])) if sigma2 > 0 else 0.0
    t_sex = beta_sex / se_sex if se_sex > 1e-15 else 0.0
    p_sex = 2.0 * (1.0 - normal_cdf(abs(t_sex)))

    return {
        "ancova_beta_sex": beta_sex,
        "ancova_se_sex": se_sex,
        "ancova_t_sex": t_sex,
        "ancova_p_sex": p_sex,
        "ancova_r2": r2,
    }


# -----------------------------------------------------------------------------
# Main Analysis Pipeline
# -----------------------------------------------------------------------------
def main() -> int:
    t_start = time.perf_counter()
    print("=" * 80)
    print("M2: Infradian Rhythm Group Contrasts and Demographic Stratification")
    print("=" * 80)

    input_csv = os.path.join(
        PROJECT_ROOT,
        "experiments/female_infradian_cycles/results/infradian_metrics_by_participant.csv",
    )
    output_csv = os.path.join(
        PROJECT_ROOT,
        "experiments/female_infradian_cycles/results/group_contrasts_age_and_sex.csv",
    )

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    print(f"Loading participant metrics from: {input_csv}")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_records = list(reader)

    n_total = len(all_records)
    print(f"Total cohort loaded: {n_total:,d} participants.")

    # 1. Filter screened participants
    screened_records = [
        r for r in all_records
        if str(r.get("screened_pass", "")).lower() in ("true", "1")
    ]
    n_screened = len(screened_records)
    print(f"Screened cohort:     {n_screened:,d} participants ({n_screened/n_total*100:.2f}% retention).")

    # 2. Stratification definitions
    cohort_definitions = {
        "Target_YoungFemales": lambda r: r["y"] == "0" and r["age_group"] == "Young Adults",
        "Control1_YoungMales": lambda r: r["y"] == "1" and r["age_group"] == "Young Adults",
        "Control2_ElderlyFemales": lambda r: r["y"] == "0" and r["age_group"] == "Elderly",
        "MiddleAger_Females": lambda r: r["y"] == "0" and r["age_group"] == "Middle Ager",
        "MiddleAger_Males": lambda r: r["y"] == "1" and r["age_group"] == "Middle Ager",
        "Elderly_Males": lambda r: r["y"] == "1" and r["age_group"] == "Elderly",
        "Overall_Females": lambda r: r["y"] == "0",
        "Overall_Males": lambda r: r["y"] == "1",
    }

    cohort_records: Dict[str, List[Dict[str, Any]]] = {
        name: [r for r in screened_records if fn(r)]
        for name, fn in cohort_definitions.items()
    }

    print("\n--- Screened Cohort Subgroup Sizes ---")
    for name, subset in cohort_records.items():
        print(f"  {name:25s}: N = {len(subset):5d}")

    signal_names = [
        "hr_p10",
        "night_hr",
        "day_hr",
        "dipping_ratio",
        "occ_daily",
        "nocturnal_compliance",
        "gap_episodes",
    ]

    metric_types = [
        "cos_amplitude",
        "ls_power",
        "ls_snr",
        "cos_delta_r2",
    ]

    contrast_pairs = [
        (
            "Target_vs_Control1",
            "Target_YoungFemales",
            "Control1_YoungMales",
            "Reproductive Females (18-39)",
            "Age-Matched Males (18-39)",
        ),
        (
            "Target_vs_Control2",
            "Target_YoungFemales",
            "Control2_ElderlyFemales",
            "Reproductive Females (18-39)",
            "Post-Menopausal Females (60+)",
        ),
        (
            "Control1_vs_Control2",
            "Control1_YoungMales",
            "Control2_ElderlyFemales",
            "Age-Matched Males (18-39)",
            "Post-Menopausal Females (60+)",
        ),
        (
            "MiddleAger_F_vs_M",
            "MiddleAger_Females",
            "MiddleAger_Males",
            "Middle-Aged Females (40-59)",
            "Middle-Aged Males (40-59)",
        ),
        (
            "Overall_F_vs_M",
            "Overall_Females",
            "Overall_Males",
            "All Screened Females",
            "All Screened Males",
        ),
    ]

    output_rows: List[Dict[str, Any]] = []

    print("\nCalculating subgroup statistics, pairwise contrasts, and ANCOVA regressions...")
    for sig in signal_names:
        pval_key = f"cos_pvalue_{sig}"

        # Fit ANCOVA across all screened participants for each metric
        ancova_fits = {
            m: fit_ancova_ols(screened_records, f"{m}_{sig}")
            for m in metric_types
        }

        for m in metric_types:
            val_col = f"{m}_{sig}"

            for c_id, g1_key, g2_key, g1_label, g2_label in contrast_pairs:
                rec1 = cohort_records[g1_key]
                rec2 = cohort_records[g2_key]

                # Extract metric values
                vals1 = [float(r[val_col]) for r in rec1 if r.get(val_col, "") not in ("", "nan", None)]
                vals2 = [float(r[val_col]) for r in rec2 if r.get(val_col, "") not in ("", "nan", None)]

                # Extract p-values for rhythmicity proportions
                pvals1 = [float(r[pval_key]) for r in rec1 if r.get(pval_key, "") not in ("", "nan", None)]
                pvals2 = [float(r[pval_key]) for r in rec2 if r.get(pval_key, "") not in ("", "nan", None)]

                s1 = calc_stats(vals1)
                s2 = calc_stats(vals2)

                prop1_05, _, _ = calc_rhythm_rate(pvals1, 0.05)
                prop1_01, _, _ = calc_rhythm_rate(pvals1, 0.01)
                prop2_05, _, _ = calc_rhythm_rate(pvals2, 0.05)
                prop2_01, _, _ = calc_rhythm_rate(pvals2, 0.01)

                wt = welch_ttest(vals1, vals2)
                mwu = mann_whitney_u(vals1, vals2)
                anc = ancova_fits[m]

                row = {
                    "signal": sig,
                    "metric": m,
                    "contrast_family": c_id,
                    "group1_name": g1_label,
                    "group1_n": s1["n"],
                    "group1_mean": round(s1["mean"], 5),
                    "group1_sd": round(s1["sd"], 5),
                    "group1_median": round(s1["median"], 5),
                    "group1_iqr": round(s1["iqr"], 5),
                    "group1_sig_p05": round(prop1_05, 4),
                    "group1_sig_p01": round(prop1_01, 4),
                    "group2_name": g2_label,
                    "group2_n": s2["n"],
                    "group2_mean": round(s2["mean"], 5),
                    "group2_sd": round(s2["sd"], 5),
                    "group2_median": round(s2["median"], 5),
                    "group2_iqr": round(s2["iqr"], 5),
                    "group2_sig_p05": round(prop2_05, 4),
                    "group2_sig_p01": round(prop2_01, 4),
                    "diff_mean": round(wt["diff_mean"], 5),
                    "ci95_low": round(wt["ci95_low"], 5),
                    "ci95_high": round(wt["ci95_high"], 5),
                    "cohens_d": round(wt["cohens_d"], 4),
                    "welch_t_stat": round(wt["welch_t_stat"], 3),
                    "welch_pvalue": f"{wt['welch_pvalue']:.4e}",
                    "mwu_u_stat": round(mwu["mwu_u_stat"], 1),
                    "mwu_pvalue": f"{mwu['mwu_pvalue']:.4e}",
                    "ancova_beta_sex": round(anc["ancova_beta_sex"], 5),
                    "ancova_se_sex": round(anc["ancova_se_sex"], 5),
                    "ancova_t_sex": round(anc["ancova_t_sex"], 3),
                    "ancova_p_sex": f"{anc['ancova_p_sex']:.4e}",
                    "ancova_r2": round(anc["ancova_r2"], 4),
                }
                output_rows.append(row)

    # Save to CSV
    fieldnames = list(output_rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print("CONTRAST SYNTHESIS COMPLETE")
    print("=" * 80)
    print(f"Output CSV:          {output_csv}")
    print(f"Total contrast rows: {len(output_rows)}")
    print(f"Execution time:      {elapsed:.2f} s")

    # Display Focal Contrast Readout for Resting HR (hr_p10) and Nocturnal Compliance
    print("\n" + "=" * 80)
    print("FOCAL EMPIRICAL READOUT: TARGET (Young Females) vs CONTROLS")
    print("=" * 80)

    focal_rows = [
        r for r in output_rows
        if r["signal"] in ("hr_p10", "nocturnal_compliance", "occ_daily")
        and r["metric"] == "cos_amplitude"
        and r["contrast_family"] in ("Target_vs_Control1", "Target_vs_Control2")
    ]

    for r in focal_rows:
        sig = r["signal"]
        fam = r["contrast_family"]
        g1 = r["group1_name"]
        g2 = r["group2_name"]
        m1, m2 = r["group1_mean"], r["group2_mean"]
        diff = r["diff_mean"]
        ci = f"[{r['ci95_low']}, {r['ci95_high']}]"
        d = r["cohens_d"]
        p_t = r["welch_pvalue"]
        p_u = r["mwu_pvalue"]
        p05_1, p05_2 = r["group1_sig_p05"], r["group2_sig_p05"]

        print(f"\nSignal: {sig:20s} | Contrast: {fam}")
        print(f"  {g1}: Mean={m1:.4f}, Rhythmicity (p<0.05)={p05_1*100:.1f}%")
        print(f"  {g2}: Mean={m2:.4f}, Rhythmicity (p<0.05)={p05_2*100:.1f}%")
        print(f"  Delta={diff:+.4f} (95% CI {ci}) | Cohen's d={d:+.3f}")
        print(f"  Welch's t-test p={p_t} | Mann-Whitney U p={p_u}")
        print(f"  Age/BMI-Adjusted Sex Effect: beta={r['ancova_beta_sex']:+.4f} (p={r['ancova_p_sex']})")

    print("\n" + "=" * 80)
    print("[SUCCESS] Milestone 2 contrast analysis complete. Output ready for REPORT.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
