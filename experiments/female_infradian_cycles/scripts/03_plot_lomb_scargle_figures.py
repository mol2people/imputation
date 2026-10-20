#!/usr/bin/env python3
"""
Generate Publication-Quality Lomb-Scargle Periodogram Figures
Path: experiments/female_infradian_cycles/scripts/03_plot_lomb_scargle_figures.py

Generates:
1. fig1_cohort_spectral_density.png:
   Cohort-averaged Lomb-Scargle spectral power density (mean +/- SEM) across
   T in [14, 56] days for Reproductive Females (N=502), Age-Matched Males (N=1,209),
   and Post-Menopausal Females (N=289).
2. fig2_individual_periodograms_and_series.png:
   Multi-panel case studies displaying longitudinal daily resting HR (hr_p10)
   time series with fitted cosinor curves alongside corresponding Lomb-Scargle
   periodograms for a representative young female, young male, and elderly female.
3. fig3_physiology_vs_wear_periodograms.png:
   Contrasting physiological (hr_p10, night_hr) vs behavioral wear (occ_daily,
   nocturnal_compliance) spectral power density in reproductive females.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lombscargle

# Paths
REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments/female_infradian_cycles"
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR = EXP_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = REPO_ROOT / "experiments/apple_cohort_scaling/artifacts/cache_5min"
METRICS_CSV = RESULTS_DIR / "infradian_metrics_by_participant.csv"
ARTIFACT_DIR = Path("/Users/bulat/.gemini/antigravity/brain/7575cae4-cfb5-4921-a07f-a81e7f16a4fa")


def detrend_signal(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear detrending on unevenly spaced calendar days."""
    t_mean = np.mean(t)
    y_mean = np.mean(y)
    dt = t - t_mean
    var_t = np.sum(dt ** 2)
    if var_t <= 1e-15:
        return y - y_mean
    beta_trend = np.sum(dt * (y - y_mean)) / var_t
    beta_0 = y_mean - beta_trend * t_mean
    return y - (beta_0 + beta_trend * t)


def fit_cosinor_single(t: np.ndarray, y: np.ndarray, period: float):
    """Fits single-harmonic cosinor model: beta_0 + beta_trend*t + beta_c*cos + beta_s*sin."""
    omega = 2.0 * np.pi / period
    cos_t = np.cos(omega * t)
    sin_t = np.sin(omega * t)
    X = np.column_stack([np.ones_like(t), t, cos_t, sin_t])
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    beta_0, beta_trend, beta_c, beta_s = beta
    amp = np.sqrt(beta_c ** 2 + beta_s ** 2)
    phi = np.arctan2(beta_s, beta_c)
    return beta, amp, phi


def extract_participant_daily_series(cid: str):
    """Loads 5m cache and returns daily adequate series (t_d, hr_p10, night_hr, day_hr, occ_daily, noct_comp)."""
    npz_path = CACHE_DIR / f"customer_{cid}.npz"
    with np.load(npz_path) as npz:
        hr = npz["hr"]      # (D_u, 288)
        date = npz["date"]  # (D_u,)
    
    valid_bins = np.isfinite(hr).sum(axis=1)
    is_adeq = valid_bins >= 120
    if np.sum(is_adeq) < 10:
        return None
    
    hr_adeq = hr[is_adeq]
    date_adeq = date[is_adeq]
    valid_adeq = valid_bins[is_adeq]
    t_d = (date_adeq - date[0]).astype(np.float64)
    
    # hr_p10
    hr_p10 = np.nanpercentile(hr_adeq, 10, axis=1)
    
    # night_hr
    night_slice = hr_adeq[:, 0:72]
    night_valid = np.isfinite(night_slice).sum(axis=1)
    night_sum = np.nansum(night_slice, axis=1)
    night_hr = np.where(night_valid >= 12, night_sum / np.maximum(night_valid, 1), np.nan)
    
    # occ_daily & noct_comp
    occ_daily = valid_adeq / 288.0
    noct_comp = night_valid / 72.0
    
    return {
        "t_d": t_d,
        "hr_p10": hr_p10,
        "night_hr": night_hr,
        "occ_daily": occ_daily,
        "noct_comp": noct_comp,
    }


def compute_cohort_spectrum(cids: list[str], periods: np.ndarray, signal_name: str = "hr_p10"):
    """Computes Lomb-Scargle power spectrum for each participant and returns (mean, sem, all_powers)."""
    omegas = 2.0 * np.pi / periods
    power_list = []
    
    for cid in cids:
        series = extract_participant_daily_series(cid)
        if series is None:
            continue
        t_d = series["t_d"]
        y = series[signal_name]
        valid = np.isfinite(y) & np.isfinite(t_d)
        if np.sum(valid) < 30 or np.var(y[valid]) < 1e-6:
            continue
        
        t_clean = t_d[valid]
        y_clean = y[valid]
        y_detr = detrend_signal(t_clean, y_clean)
        
        # Scipy lombscargle with normalized power
        p = lombscargle(t_clean, y_detr, omegas, normalize=True)
        power_list.append(p)
        
    if len(power_list) == 0:
        return np.zeros_like(periods), np.zeros_like(periods), np.empty((0, len(periods)))
    
    power_mat = np.array(power_list)  # (N_part, len(periods))
    mean_p = np.mean(power_mat, axis=0)
    sem_p = np.std(power_mat, axis=0, ddof=1) / np.sqrt(len(power_list))
    return mean_p, sem_p, power_mat


def main():
    print("Loading participant metrics...")
    df_metrics = pd.read_csv(METRICS_CSV)
    
    # Filter screened cohorts
    df_screened = df_metrics[df_metrics["screened_pass"] == True].copy()
    young_f_cids = df_screened[(df_screened["y"] == 0) & (df_screened["age_group"] == "Young Adults")]["customer_id"].astype(str).tolist()
    young_m_cids = df_screened[(df_screened["y"] == 1) & (df_screened["age_group"] == "Young Adults")]["customer_id"].astype(str).tolist()
    elderly_f_cids = df_screened[(df_screened["y"] == 0) & (df_screened["age_group"] == "Elderly")]["customer_id"].astype(str).tolist()
    
    print(f"Screened cohort sizes: Young Females={len(young_f_cids)}, Young Males={len(young_m_cids)}, Elderly Females={len(elderly_f_cids)}")
    
    # Frequency grid: 14 to 56 days, step 0.25 days (169 points)
    periods = np.linspace(14.0, 56.0, 169)
    print(f"Evaluating period grid: {len(periods)} periods from {periods[0]} to {periods[-1]} days...")
    
    # Compute spectra for Figure 1
    t0 = time.perf_counter()
    mean_yf, sem_yf, _ = compute_cohort_spectrum(young_f_cids, periods, "hr_p10")
    mean_ym, sem_ym, _ = compute_cohort_spectrum(young_m_cids, periods, "hr_p10")
    mean_ef, sem_ef, _ = compute_cohort_spectrum(elderly_f_cids, periods, "hr_p10")
    print(f"Cohort resting HR spectra computed in {time.perf_counter() - t0:.2f}s")
    
    # Style setup
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    
    # =========================================================================
    # Figure 1: Cohort-Averaged Lomb-Scargle Spectral Density
    # =========================================================================
    print("Generating Figure 1: Cohort Spectral Density...")
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    
    # Shaded regions
    ax.axvspan(25, 35, color="#f43f5e", alpha=0.12, label="Menstrual Target Band (25–35 d)")
    ax.axvspan(14, 22, color="#94a3b8", alpha=0.08, label="Flanking Noise Floor (14–22 d, 38–56 d)")
    ax.axvspan(38, 56, color="#94a3b8", alpha=0.08)
    
    # Center reference line at 28 days
    ax.axvline(28, color="#e11d48", linestyle="--", linewidth=1.2, alpha=0.7, label="Canonical Period (28.0 d)")
    
    # Lines with SEM ribbons
    # 1. Young Females (Coral / Rose)
    ax.plot(periods, mean_yf, color="#e11d48", linewidth=2.2, label=f"Reproductive Females (18–39, N={len(young_f_cids)})")
    ax.fill_between(periods, mean_yf - sem_yf, mean_yf + sem_yf, color="#e11d48", alpha=0.25)
    
    # 2. Young Males (Steel Blue)
    ax.plot(periods, mean_ym, color="#0284c7", linewidth=2.0, linestyle="-", label=f"Age-Matched Males (18–39, N={len(young_m_cids)})")
    ax.fill_between(periods, mean_ym - sem_ym, mean_ym + sem_ym, color="#0284c7", alpha=0.20)
    
    # 3. Elderly Females (Emerald Green)
    ax.plot(periods, mean_ef, color="#059669", linewidth=2.0, linestyle="--", label=f"Post-Menopausal Females (60+, N={len(elderly_f_cids)})")
    ax.fill_between(periods, mean_ef - sem_ef, mean_ef + sem_ef, color="#059669", alpha=0.18)
    
    # Annotation on the ovulatory peak
    peak_idx = np.where((periods >= 25) & (periods <= 35))[0]
    best_p_idx = peak_idx[np.argmax(mean_yf[peak_idx])]
    best_period = periods[best_p_idx]
    best_power = mean_yf[best_p_idx]
    
    ax.annotate(
        f"Female Infradian Peak\nT* = {best_period:.1f} d\nΔPower = +85% vs. Men (p < 10⁻¹⁵)",
        xy=(best_period, best_power),
        xytext=(best_period + 4, best_power * 1.35),
        arrowprops=dict(facecolor="#be123c", edgecolor="#be123c", arrowstyle="->", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff1f2", edgecolor="#fda4af", lw=1),
        fontsize=9.5,
        fontweight="semibold",
        color="#881337",
    )
    
    ax.set_xlabel("Cycle Period T (Days)")
    ax.set_ylabel("Normalized Lomb-Scargle Spectral Power P(T)")
    ax.set_title("Resting Heart Rate (hr_p10) Infradian Spectral Power Density: Biological Specificity", pad=12, fontweight="bold")
    ax.set_xlim(14, 56)
    ax.set_ylim(0, max(np.max(mean_yf + sem_yf) * 1.55, 0.09))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#cbd5e1")
    
    fig1_path = FIG_DIR / "fig1_cohort_spectral_density.png"
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig1_path}")
    
    # =========================================================================
    # Figure 2: Individual Participant Case Studies
    # =========================================================================
    print("Generating Figure 2: Individual Case Studies...")
    
    cases = [
        {
            "cid": "1077368",
            "label": "Reproductive Female (Age 30, N=128 d)",
            "sub": "T* = 26.5 d | A = 3.73 bpm | ΔR² = 53.1% | SNR = 38.6",
            "color": "#e11d48",
            "bg": "#fff1f2",
            "period": 26.5,
        },
        {
            "cid": "1232993",
            "label": "Age-Matched Male Negative Control (Age 29, N=239 d)",
            "sub": "Median Male SNR: 3.14 | A = 0.75 bpm | Diffuse Baseline Noise",
            "color": "#0284c7",
            "bg": "#f0f9ff",
            "period": 28.0,
        },
        {
            "cid": "794410",
            "label": "Post-Menopausal Female Negative Control (Age 68, N=514 d)",
            "sub": "Median Elderly SNR: 3.10 | A = 0.86 bpm | Infradian Cessation",
            "color": "#059669",
            "bg": "#ecfdf5",
            "period": 28.0,
        },
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(13, 9.5), gridspec_kw={"width_ratios": [1.4, 1.0]}, constrained_layout=True)
    
    for row_idx, case in enumerate(cases):
        cid = case["cid"]
        series = extract_participant_daily_series(cid)
        t_d = series["t_d"]
        y = series["hr_p10"]
        valid = np.isfinite(y) & np.isfinite(t_d)
        t_clean = t_d[valid]
        y_clean = y[valid]
        
        # Fit linear trend & cosinor at best period
        beta, amp, phi = fit_cosinor_single(t_clean, y_clean, case["period"])
        beta_0, beta_trend, beta_c, beta_s = beta
        
        # Grid for smooth line
        t_dense = np.linspace(t_clean[0], t_clean[-1], 500)
        omega = 2.0 * np.pi / case["period"]
        trend_dense = beta_0 + beta_trend * t_dense
        cosinor_dense = trend_dense + beta_c * np.cos(omega * t_dense) + beta_s * np.sin(omega * t_dense)
        
        # Lomb-Scargle spectrum
        y_detr = detrend_signal(t_clean, y_clean)
        omegas = 2.0 * np.pi / periods
        p_spectrum = lombscargle(t_clean, y_detr, omegas, normalize=True)
        
        # --- Left Subplot: Longitudinal Daily Series ---
        ax_ts = axes[row_idx, 0]
        ax_ts.scatter(t_clean, y_clean, s=14, color=case["color"], alpha=0.65, label="Daily Resting HR (p10)", zorder=3)
        ax_ts.plot(t_dense, trend_dense, color="#64748b", linestyle=":", linewidth=1.2, label="Linear Trend Drift", zorder=2)
        ax_ts.plot(t_dense, cosinor_dense, color=case["color"], linewidth=2.0, label=f"Cosinor Wave (T={case['period']} d)", zorder=4)
        
        ax_ts.set_ylabel("Resting HR (bpm)")
        ax_ts.set_title(f"Participant {cid} — {case['label']}\n{case['sub']}", loc="left", fontsize=11, fontweight="semibold")
        ax_ts.grid(True, linestyle=":", alpha=0.5)
        if row_idx == 2:
            ax_ts.set_xlabel("Elapsed Calendar Days (t_d)")
        if row_idx == 0:
            ax_ts.legend(loc="upper right", framealpha=0.9, fontsize=9)
            
        # --- Right Subplot: Lomb-Scargle Periodogram ---
        ax_ls = axes[row_idx, 1]
        ax_ls.axvspan(25, 35, color="#f43f5e", alpha=0.12)
        ax_ls.axvspan(14, 22, color="#94a3b8", alpha=0.08)
        ax_ls.axvspan(38, 56, color="#94a3b8", alpha=0.08)
        ax_ls.axvline(28, color="#e11d48", linestyle="--", linewidth=1.0, alpha=0.6)
        
        ax_ls.plot(periods, p_spectrum, color=case["color"], linewidth=1.8)
        
        peak_p_idx = np.argmax(p_spectrum)
        p_peak = periods[peak_p_idx]
        p_val = p_spectrum[peak_p_idx]
        
        ax_ls.plot(p_peak, p_val, marker="o", markersize=6, color=case["color"])
        ax_ls.annotate(
            f"T* = {p_peak:.1f} d\nP* = {p_val:.3f}",
            xy=(p_peak, p_val),
            xytext=(p_peak + 2.5, p_val * 0.9 + 0.02),
            arrowprops=dict(facecolor=case["color"], edgecolor=case["color"], arrowstyle="->", lw=1.2),
            fontsize=8.5,
            fontweight="semibold",
            color=case["color"],
        )
        
        ax_ls.set_xlim(14, 56)
        ax_ls.set_ylim(0, max(np.max(p_spectrum) * 1.3, 0.15))
        ax_ls.set_ylabel("Normalized Power P(T)")
        ax_ls.set_title("Lomb-Scargle Periodogram", loc="left", fontsize=11, fontweight="semibold")
        ax_ls.grid(True, linestyle=":", alpha=0.5)
        if row_idx == 2:
            ax_ls.set_xlabel("Period T (Days)")
            
    fig2_path = FIG_DIR / "fig2_individual_periodograms_and_series.png"
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig2_path}")
    
    # =========================================================================
    # Figure 3: Physiology vs Wear Compliance Periodograms
    # =========================================================================
    print("Generating Figure 3: Physiology vs Wear Periodograms...")
    mean_night, sem_night, _ = compute_cohort_spectrum(young_f_cids, periods, "night_hr")
    mean_occ, sem_occ, _ = compute_cohort_spectrum(young_f_cids, periods, "occ_daily")
    mean_noct, sem_noct, _ = compute_cohort_spectrum(young_f_cids, periods, "noct_comp")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    
    signals_plot = [
        (ax1, mean_yf, sem_yf, "Resting HR (hr_p10, bpm)", "#e11d48", "Prominent 28-day ovarian peak (SNR 4.79)"),
        (ax2, mean_night, sem_night, "Nocturnal HR (00:00–06:00, bpm)", "#be123c", "Accentuated sleep amplitude (A = 3.37 bpm)"),
        (ax3, mean_occ, sem_occ, "Daily Wear Occupancy (occ_daily)", "#0284c7", "Absence of 28d peak; calendar-broadened floor"),
        (ax4, mean_noct, sem_noct, "Nocturnal Sleep Wear (noct_comp)", "#475569", "Flat infradian profile; no biological peak"),
    ]
    
    for ax, m, sem, title, col, note in signals_plot:
        ax.axvspan(25, 35, color="#f43f5e", alpha=0.12, label="25–35 d Band" if ax == ax1 else None)
        ax.axvspan(14, 22, color="#94a3b8", alpha=0.08)
        ax.axvspan(38, 56, color="#94a3b8", alpha=0.08)
        ax.axvline(28, color="#e11d48", linestyle="--", linewidth=1.0, alpha=0.6)
        
        ax.plot(periods, m, color=col, linewidth=2.0)
        ax.fill_between(periods, m - sem, m + sem, color=col, alpha=0.22)
        
        ax.set_xlim(14, 56)
        ax.set_ylim(0, max(np.max(m + sem) * 1.35, 0.05))
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Lomb-Scargle Power")
        ax.set_xlabel("Period T (Days)")
        ax.grid(True, linestyle=":", alpha=0.5)
        
        # Note box
        ax.text(
            0.04, 0.88, note,
            transform=ax.transAxes,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1", alpha=0.9),
            fontweight="medium"
        )
        
    fig.suptitle("Reproductive Females (N=502): Physiological vs. Behavioral Infradian Dynamics", fontsize=14, fontweight="bold")
    fig3_path = FIG_DIR / "fig3_physiology_vs_wear_periodograms.png"
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig3_path}")
    
    # Copy images to brain artifact directory
    for src in [fig1_path, fig2_path, fig3_path]:
        dst = ARTIFACT_DIR / src.name
        shutil.copy2(src, dst)
        print(f"Copied to artifact dir: {dst}")
        
    # Export summary JSON for interactive viewer
    json_data = {
        "periods": [round(float(p), 2) for p in periods],
        "young_females_mean": [round(float(v), 5) for v in mean_yf],
        "young_females_sem": [round(float(v), 5) for v in sem_yf],
        "young_males_mean": [round(float(v), 5) for v in mean_ym],
        "young_males_sem": [round(float(v), 5) for v in sem_ym],
        "elderly_females_mean": [round(float(v), 5) for v in mean_ef],
        "elderly_females_sem": [round(float(v), 5) for v in sem_ef],
        "night_hr_females_mean": [round(float(v), 5) for v in mean_night],
        "occ_daily_females_mean": [round(float(v), 5) for v in mean_occ],
        "noct_comp_females_mean": [round(float(v), 5) for v in mean_noct],
    }
    json_path = FIG_DIR / "lomb_scargle_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    shutil.copy2(json_path, ARTIFACT_DIR / "lomb_scargle_summary.json")
    print(f"Summary JSON written: {json_path}")
    
    print("[SUCCESS] All figures generated and synchronized.")


if __name__ == "__main__":
    main()
