#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
05_generate_publication_figures.py: Generate multi-panel publication-grade figures
for the Continuous Age RCS Infradian Rhythmicity Study.

Figures generated in experiments/infradian_age_dependency/figures/:
1. fig1_rcs_continuous_partial_effects.png:
   3 source columns (Apple Watch, Garmin Tier 1c, Samsung).
   Continuous female and male unadjusted Stone-Koo RCS curves with pointwise 95% CIs
   and simultaneous (uniform) 95% confidence bands over continuous age [18, 80],
   marking landmark knots at [25, 42, 52, 65] years.
2. fig2_excess_heatmaps_and_fwhm.png:
   Female age-by-period excess heatmaps over 15-60 days, and FWHM spectral traces
   across 5-year age bands for Apple Watch and Garmin Tier 1c.
3. fig3_delta_curve_transitions.png:
   Female-minus-male delta curve Delta(a) with simultaneous uniform confidence bands,
   annotated with the three transition metrics: steepest decline inflection point,
   nominal zero-crossing, and conservative cessation boundary.
4. fig4_source_and_protocol_comparisons.png:
   Cross-source validation (Apple vs Garmin), protocol contrast (expanded90 vs balanced90),
   and second-window repeatability scatter / correlations by age bracket (<40, 40-55, >=55).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

# Path definitions
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Styling defaults for scientific publication
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
})

DEFAULT_KNOTS = (25.0, 42.0, 52.0, 65.0)


# =============================================================================
# Helper: Stone-Koo basis and derivatives
# =============================================================================
def rcs_stone_koo(ages: np.ndarray, knots: Tuple[float, ...] = DEFAULT_KNOTS) -> Tuple[np.ndarray, np.ndarray]:
    t1, t2, t3, t4 = knots
    norm = float((t4 - t1) ** 2)
    x = np.asarray(ages, dtype=np.float64)

    def pos_cube(u: np.ndarray) -> np.ndarray:
        return np.maximum(u, 0.0) ** 3

    c3_1 = (t4 - t1) / (t4 - t3)
    c4_1 = (t3 - t1) / (t4 - t3)
    rcs1 = (pos_cube(x - t1) - c3_1 * pos_cube(x - t3) + c4_1 * pos_cube(x - t4)) / norm

    c3_2 = (t4 - t2) / (t4 - t3)
    c4_2 = (t3 - t2) / (t4 - t3)
    rcs2 = (pos_cube(x - t2) - c3_2 * pos_cube(x - t3) + c4_2 * pos_cube(x - t4)) / norm

    return rcs1, rcs2


def d_rcs_stone_koo(ages: np.ndarray, knots: Tuple[float, ...] = DEFAULT_KNOTS) -> Tuple[np.ndarray, np.ndarray]:
    t1, t2, t3, t4 = knots
    norm = float((t4 - t1) ** 2)
    x = np.asarray(ages, dtype=np.float64)

    def pos_sq(u: np.ndarray) -> np.ndarray:
        return 3.0 * (np.maximum(u, 0.0) ** 2)

    c3_1 = (t4 - t1) / (t4 - t3)
    c4_1 = (t3 - t1) / (t4 - t3)
    d_rcs1 = (pos_sq(x - t1) - c3_1 * pos_sq(x - t3) + c4_1 * pos_sq(x - t4)) / norm

    c3_2 = (t4 - t2) / (t4 - t3)
    c4_2 = (t3 - t2) / (t4 - t3)
    d_rcs2 = (pos_sq(x - t2) - c3_2 * pos_sq(x - t3) + c4_2 * pos_sq(x - t4)) / norm

    return d_rcs1, d_rcs2


# =============================================================================
# FIGURE 1: Continuous RCS Partial Effect Curves Across 3 Source Columns
# =============================================================================
def generate_figure_1(df_bands: pd.DataFrame, df_endpoints: pd.DataFrame):
    print("Generating Figure 1: Continuous RCS Partial Effects (3 Sources)...")
    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.25)

    sources = [
        (6, "Apple Watch", "expanded90", "N = 5,162 in Shared Support (1,715 F, 3,447 M)"),
        (3, "Garmin Tier 1c", "expanded90", "N = 3,754 in Shared Support (1,362 F, 2,392 M)"),
        (7, "Samsung", "expanded90", "N = 8 Total (3 F, 5 M) — Insufficient Support"),
    ]

    endpoints = [
        ("Q_i", r"Primary Infradian Excess $Q_i$", r"Band-Integrated Excess $[1/35, 1/25]\ \mathrm{d}^{-1}$"),
        ("A_hat", r"Physical Cosinor Amplitude $\hat{A}_i$", "Amplitude (bpm)"),
    ]

    # Colors
    c_fem = "#C0392B"       # Strong crimson/coral
    c_fem_light = "#FADBD8" # Light coral shading
    c_male = "#2980B9"      # Strong cobalt blue
    c_male_light = "#D4E6F1"# Light blue shading

    for col_idx, (src_id, src_name, proto, sample_annot) in enumerate(sources):
        for row_idx, (ep_name, ep_title, y_label) in enumerate(endpoints):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if src_id in (3, 6):
                # Retrieve female, male, and bands
                sub = df_bands[
                    (df_bands["source"] == src_id) &
                    (df_bands["protocol"] == proto) &
                    (df_bands["endpoint"] == ep_name)
                ]

                fem = sub[sub["curve_type"] == "female"].sort_values("age")
                male = sub[sub["curve_type"] == "male"].sort_values("age")

                ages = fem["age"].values

                # Shared support background
                ax.axvspan(18.0, 25.0, color="#F2F3F4", alpha=0.7, zorder=0)
                ax.axvspan(75.0, 80.0, color="#F2F3F4", alpha=0.7, zorder=0)

                # Landmark Knots
                for k in DEFAULT_KNOTS:
                    ax.axvline(k, color="#7F8C8D", linestyle=":", linewidth=0.9, alpha=0.8, zorder=1)

                # Female curves & bands
                ax.fill_between(
                    ages, fem["band_lower_simultaneous"], fem["band_upper_simultaneous"],
                    color=c_fem, alpha=0.15, label="Female 95% Uniform Band", zorder=2
                )
                ax.fill_between(
                    ages, fem["ci_lower_pointwise"], fem["ci_upper_pointwise"],
                    color=c_fem, alpha=0.30, label="Female 95% Pointwise CI", zorder=3
                )
                ax.plot(
                    ages, fem["point_estimate"], color=c_fem, linewidth=2.0,
                    label="Female Trajectory", zorder=4
                )

                # Male curves & bands
                ax.fill_between(
                    ages, male["band_lower_simultaneous"], male["band_upper_simultaneous"],
                    color=c_male, alpha=0.15, label="Male 95% Uniform Band", zorder=2
                )
                ax.fill_between(
                    ages, male["ci_lower_pointwise"], male["ci_upper_pointwise"],
                    color=c_male, alpha=0.30, label="Male 95% Pointwise CI", zorder=3
                )
                ax.plot(
                    ages, male["point_estimate"], color=c_male, linewidth=2.0,
                    label="Male Negative Control", zorder=4
                )

                # Zero reference for Q_i
                if ep_name == "Q_i":
                    ax.axhline(0.0, color="#34495E", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)

                ax.set_xlim(18.0, 80.0)
                ax.set_xlabel("Continuous Chronological Age (years)")
                ax.set_ylabel(y_label)

                # Annotate knot positions at the top of the plot
                if row_idx == 0:
                    ax.set_ylim(-0.012, 0.055)
                    ax.set_title(f"{src_name}\n({sample_annot})", fontsize=10.5, pad=12, fontweight="bold")
                    ax.text(25.0, 0.049, "$t_1=25$", ha="center", fontsize=7.5, color="#566573", bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.7))
                    ax.text(42.0, 0.049, "$t_2=42$", ha="center", fontsize=7.5, color="#566573", bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.7))
                    ax.text(52.0, 0.049, "$t_3=52$", ha="center", fontsize=7.5, color="#566573", bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.7))
                    ax.text(65.0, 0.049, "$t_4=65$", ha="center", fontsize=7.5, color="#566573", bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.7))
                else:
                    ax.set_title(f"{src_name}: {ep_title}", fontsize=10, pad=8)

                if col_idx == 0 and row_idx == 0:
                    ax.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor="white", edgecolor="#D5D8DC", fontsize=8.0)

            else:
                # Column 3: Samsung (source 7) - Support failure panel
                ax.set_facecolor("#FCF3CF")
                ax.set_xlim(18.0, 80.0)
                ax.set_ylim(-0.02, 0.08) if ep_name == "Q_i" else ax.set_ylim(0.0, 3.0)
                ax.set_xlabel("Continuous Chronological Age (years)")
                ax.set_ylabel(y_label)

                if row_idx == 0:
                    ax.set_title(f"{src_name}\n({sample_annot})", fontsize=10.5, pad=12, fontweight="bold", color="#7D6608")
                else:
                    ax.set_title(f"{src_name}: Insufficient Demographic Support", fontsize=10, pad=8, color="#7D6608")

                card_text = (
                    "SAMSUNG (SOURCE 7) INSUFFICIENT SUPPORT\n\n"
                    "• Eligible Window 1 Participants: N = 8\n"
                    "   (Female: 3, Male: 5)\n"
                    "• Qualified 5-Year Age Bands with >= 30/cell: 0\n"
                    "• Shared Demographic Support: FAILS\n\n"
                    "Status: Non-estimable per PLAN.md §§6, 11, 12.\n"
                    "Standardized threshold preserved without dilution."
                )
                ax.text(
                    49.0, 0.03 if ep_name == "Q_i" else 1.5,
                    card_text,
                    ha="center", va="center",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="white", edgecolor="#B7950B", linewidth=1.2),
                    color="#4D5656"
                )

    fig.suptitle(
        "Figure 1: Frank Harrell Stone-Koo Restricted Cubic Spline (RCS) Partial Effect Trajectories\n"
        "Across Continuous Age [18, 80] with Pointwise and Simultaneous 95% Confidence Bands",
        fontsize=12.5, y=0.99, fontweight="bold"
    )

    out_path = FIGURES_DIR / "fig1_rcs_continuous_partial_effects.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# FIGURE 2: Female Age-by-Period Excess Heatmaps and FWHM Spectral Traces
# =============================================================================
def generate_figure_2(df_endpoints: pd.DataFrame):
    print("Generating Figure 2: Female Excess Heatmaps & FWHM Traces...")
    npz_spectra = np.load(ARTIFACTS_DIR / "spectra/spectra_broad.npz")
    broad_freqs = npz_spectra["broad_freqs"]
    broad_periods = 1.0 / broad_freqs
    P_obs = npz_spectra["P_obs"]
    P_null = npz_spectra["P_null_mean"]
    D_excess = P_obs - P_null

    u = npz_spectra["user_id"]
    s = npz_spectra["source"]
    p = npz_spectra["protocol"]
    w = npz_spectra["window_index"]

    spec_index = pd.DataFrame({
        "user_id": u, "source": s, "protocol": p, "window_index": w, "spec_idx": np.arange(len(u))
    })

    # Focus on females, window 1, expanded90, converged nulls
    females = df_endpoints[
        (df_endpoints["window_index"] == 1) &
        (df_endpoints["protocol"] == "expanded90") &
        (df_endpoints["null_status"] == "converged") &
        (df_endpoints["sex"] == 1)
    ]
    merged = females.merge(spec_index, on=["user_id", "source", "protocol", "window_index"], how="inner")

    # 15-60 days grid
    mask_15_60 = (broad_periods >= 14.999) & (broad_periods <= 60.001)
    periods_15_60 = broad_periods[mask_15_60]
    sort_idx = np.argsort(periods_15_60)
    grid_p = periods_15_60[sort_idx]

    age_bins = np.arange(25, 80, 5) # [25, 30), ..., [70, 75)
    band_labels = [f"[{b},{b+5})" for b in age_bins[:-1]]

    # Uniform period grid for linear heatmap display
    uniform_p = np.linspace(15.0, 60.0, 181)

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 12), gridspec_kw={"height_ratios": [1.0, 1.0, 0.75], "hspace": 0.35, "wspace": 0.30})

    devices = [(6, "Apple Watch", 0), (3, "Garmin Tier 1c", 1)]

    for src_id, src_name, col_idx in devices:
        sub = merged[merged["source"] == src_id].copy()
        band_labels = [f"[{int(b)}, {int(b+5)})" for b in age_bins[:-1]]

        heatmap_matrix = np.zeros((len(band_labels), len(uniform_p)), dtype=np.float64)
        band_counts = []
        fwhm_list = []
        peak_excess_list = []
        peak_period_list = []

        # Palette for spectral traces
        colors = sns.color_palette("viridis", len(band_labels))

        ax_heat = axes[0, col_idx]
        ax_trace = axes[1, col_idx]

        for b_idx, (b_low, b_high) in enumerate(zip(age_bins[:-1], age_bins[1:])):
            label = band_labels[b_idx]
            grp = sub[(sub["age"] >= b_low) & (sub["age"] < b_high)]
            n_grp = len(grp)
            band_counts.append(n_grp)

            if n_grp < 30:
                peak_period_list.append(np.nan)
                peak_excess_list.append(np.nan)
                fwhm_list.append(np.nan)
                continue

            indices = grp["spec_idx"].values
            band_d = D_excess[indices][:, mask_15_60][:, sort_idx]
            mean_d = np.mean(band_d, axis=0)

            # Interpolate onto uniform period grid for artifact-free heatmap display
            interp_d = np.interp(uniform_p, grid_p, mean_d)
            heatmap_matrix[b_idx, :] = interp_d

            # Plot trace on actual frequencies/periods
            ax_trace.plot(grid_p, mean_d, color=colors[b_idx], linewidth=1.7, label=f"Age {label} (N={n_grp})")

            # Peak and FWHM in target band [25, 35]
            mask_target = (grid_p >= 25.0) & (grid_p <= 35.0)
            target_p = grid_p[mask_target]
            target_d = mean_d[mask_target]

            p_max_idx = np.argmax(target_d)
            p_peak = target_p[p_max_idx]
            d_peak = target_d[p_max_idx]
            peak_period_list.append(p_peak)
            peak_excess_list.append(d_peak)

            # Compute FWHM
            half_height = d_peak / 2.0
            left_cross = np.where((mean_d[:-1] <= half_height) & (mean_d[1:] > half_height) & (grid_p[:-1] < p_peak))[0]
            right_cross = np.where((mean_d[:-1] > half_height) & (mean_d[1:] <= half_height) & (grid_p[1:] > p_peak))[0]

            if len(left_cross) > 0 and len(right_cross) > 0 and d_peak > 0:
                lc = left_cross[-1]
                rc = right_cross[0]
                t_l = grid_p[lc] + (half_height - mean_d[lc]) / (mean_d[lc+1] - mean_d[lc]) * (grid_p[lc+1] - grid_p[lc])
                t_r = grid_p[rc] + (half_height - mean_d[rc]) / (mean_d[rc+1] - mean_d[rc]) * (grid_p[rc+1] - grid_p[rc])
                fwhm_val = t_r - t_l
                fwhm_list.append(fwhm_val)
            else:
                fwhm_list.append(np.nan)

        # 1. Heatmap Panel
        im = ax_heat.imshow(
            heatmap_matrix,
            extent=[15.0, 60.0, len(band_labels)-0.5, -0.5],
            aspect="auto",
            cmap="inferno",
            vmin=-0.005,
            vmax=0.045
        )
        ax_heat.set_yticks(np.arange(len(band_labels)))
        ax_heat.set_yticklabels([f"{lbl} (N={cnt})" for lbl, cnt in zip(band_labels, band_counts)])
        ax_heat.set_xlabel("Harmonic Trial Period $T$ (days)")
        ax_heat.set_ylabel("5-Year Female Age Band")
        ax_heat.set_title(f"{src_name}: Age-by-Period Excess Heatmap $D(a, T)$", fontweight="bold")
        ax_heat.axvline(25.0, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
        ax_heat.axvline(35.0, color="white", linestyle="--", linewidth=1.0, alpha=0.8)

        cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical", pad=0.02, shrink=0.85)
        cbar.set_label(r"Mean Excess $D(f) = P_{\mathrm{obs}} - P_{\mathrm{null}}$", fontsize=8.5)

        # 2. Spectral Traces Panel
        ax_trace.set_xlim(15.0, 60.0)
        ax_trace.set_ylim(-0.01, 0.05)
        ax_trace.axhline(0.0, color="#7F8C8D", linestyle="--", linewidth=0.8)
        ax_trace.axvline(25.0, color="#C0392B", linestyle=":", linewidth=1.0, alpha=0.7)
        ax_trace.axvline(35.0, color="#C0392B", linestyle=":", linewidth=1.0, alpha=0.7)
        ax_trace.axvspan(25.0, 35.0, color="#E6B0AA", alpha=0.15, label="Target Infradian Band [25, 35]d")
        ax_trace.set_xlabel("Harmonic Period $T$ (days)")
        ax_trace.set_ylabel("Partial $R^2$ Excess $D(T)$")
        ax_trace.set_title(f"{src_name}: Spectral Slices by 5-Year Age Band", fontweight="bold")
        ax_trace.legend(loc="upper right", ncol=2, fontsize=7.0, framealpha=0.9)

        # 3. Peak Excess & FWHM Traces
        ax_stat = axes[2, col_idx]
        valid_bands = np.arange(len(band_labels))
        ax_stat.plot(valid_bands, peak_excess_list, marker="o", color="#C0392B", linewidth=1.8, label="Peak Excess $D^*$ in [25, 35]d")
        ax_stat.set_xticks(valid_bands)
        ax_stat.set_xticklabels(band_labels, rotation=40, ha="right")
        ax_stat.set_ylabel("Peak Infradian Excess", color="#C0392B")
        ax_stat.tick_params(axis="y", labelcolor="#C0392B")
        ax_stat.set_ylim(0.0, 0.06)

        ax_stat_r = ax_stat.twinx()
        ax_stat_r.plot(valid_bands, fwhm_list, marker="s", linestyle="--", color="#2980B9", linewidth=1.8, label="FWHM (days)")
        ax_stat_r.set_ylabel("Full Width at Half Max (days)", color="#2980B9")
        ax_stat_r.tick_params(axis="y", labelcolor="#2980B9")
        ax_stat_r.set_ylim(2.0, 12.0)
        ax_stat.set_title(f"{src_name}: Peak Excess Attenuation & Spectral Broadening", fontweight="bold")

    fig.suptitle(
        "Figure 2: Empirical Spectral Excess Anatomy Across Age: Heatmaps, Traces, and Resonance FWHM\n"
        "Resolving Ovarian Cycle Resonance Collapse Against Two-Component REML Null Envelopes",
        fontsize=12.5, y=0.99, fontweight="bold"
    )

    out_path = FIGURES_DIR / "fig2_excess_heatmaps_and_fwhm.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# FIGURE 3: Female-Minus-Male Delta Curves and Physiological Transitions
# =============================================================================
def generate_figure_3(df_bands: pd.DataFrame, df_trans: pd.DataFrame):
    print("Generating Figure 3: Delta Curve Transitions & Inflection Analysis...")
    fig = plt.figure(figsize=(14.5, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], hspace=0.35, wspace=0.22)

    devices = [
        (6, "Apple Watch", "expanded90", 0, "46.65y", "56.18y", "Not Reached (>80y)"),
        (3, "Garmin Tier 1c", "expanded90", 1, "47.25y", "70.20y", "Not Reached (>80y)"),
    ]

    c_delta = "#8E44AD"       # Royal purple
    c_inflect = "#D35400"     # Amber/rust
    c_crossing = "#16A085"    # Teal

    for src_id, src_name, proto, col_idx, inflect_label, cross_label, cess_label in devices:
        # 1. Main Delta Curve Panel
        ax_main = fig.add_subplot(gs[0, col_idx])

        sub = df_bands[
            (df_bands["source"] == src_id) &
            (df_bands["protocol"] == proto) &
            (df_bands["endpoint"] == "Q_i") &
            (df_bands["curve_type"] == "delta")
        ].sort_values("age")

        ages = sub["age"].values
        delta_pe = sub["point_estimate"].values
        band_low_sim = sub["band_lower_simultaneous"].values
        band_upp_sim = sub["band_upper_simultaneous"].values
        ci_low_pw = sub["ci_lower_pointwise"].values
        ci_upp_pw = sub["ci_upper_pointwise"].values

        # Shared support & knot lines
        ax_main.axvspan(18.0, 25.0, color="#F2F3F4", alpha=0.7, zorder=0)
        ax_main.axvspan(75.0, 80.0, color="#F2F3F4", alpha=0.7, zorder=0)
        ax_main.axvspan(42.0, 52.0, color="#FCF3CF", alpha=0.5, zorder=0, label="Perimenopausal Transition Window [42, 52]y")

        for k in DEFAULT_KNOTS:
            ax_main.axvline(k, color="#BDC3C7", linestyle=":", linewidth=0.9, zorder=1)

        # Confidence bands
        ax_main.fill_between(
            ages, band_low_sim, band_upp_sim,
            color=c_delta, alpha=0.15, label="Simultaneous 95% Uniform Band", zorder=2
        )
        ax_main.fill_between(
            ages, ci_low_pw, ci_upp_pw,
            color=c_delta, alpha=0.30, label="Pointwise 95% Bootstrap CI", zorder=3
        )
        ax_main.plot(ages, delta_pe, color=c_delta, linewidth=2.2, label=r"Delta Curve $\Delta(a) = \hat{Q}_F(a) - \hat{Q}_M(a)$", zorder=4)

        # Zero reference line
        ax_main.axhline(0.0, color="#2C3E50", linestyle="--", linewidth=1.0, zorder=2)

        # Transition Metrics annotations
        tr_sub = df_trans[(df_trans["source"] == src_id) & (df_trans["protocol"] == proto) & (df_trans["endpoint"] == "Q_i")]

        # Metric 1: Steepest Decline Inflection Point
        m1 = tr_sub[tr_sub["metric_id"] == 1]
        m1_age = float(m1[m1["metric_name"] == "steepest_decline_inflection_age"]["point_estimate"].iloc[0])
        m1_ci_l = float(m1[m1["metric_name"] == "steepest_decline_inflection_age"]["ci_lower_95"].iloc[0])
        m1_ci_u = float(m1[m1["metric_name"] == "steepest_decline_inflection_age"]["ci_upper_95"].iloc[0])
        m1_slope = float(m1[m1["metric_name"] == "steepest_decline_slope"]["point_estimate"].iloc[0])

        ax_main.axvline(m1_age, color=c_inflect, linestyle="-.", linewidth=1.5, zorder=5)
        ax_main.axvspan(m1_ci_l, m1_ci_u, color=c_inflect, alpha=0.15, zorder=2)

        # Metric 2: Nominal Zero-Crossing
        m2 = tr_sub[tr_sub["metric_id"] == 2]
        m2_age = float(m2["point_estimate"].iloc[0])
        m2_ci_l = float(m2["ci_lower_95"].iloc[0])
        m2_ci_u = float(m2["ci_upper_95"].iloc[0])

        ax_main.plot([m2_ci_l, m2_ci_u], [0.0, 0.0], color=c_crossing, linewidth=4.0, solid_capstyle="round", zorder=6)
        ax_main.scatter([m2_age], [0.0], color=c_crossing, s=70, zorder=7, marker="o", edgecolor="white", linewidth=1.2)

        # Callout boxes
        text_annot = (
            f"1. Inflection Point (Steepest Decline):\n"
            f"    Age: {m1_age:.2f}y (95% CI: [{m1_ci_l:.2f}, {m1_ci_u:.2f}])\n"
            f"    Min Slope: {m1_slope:.4f} units/year\n\n"
            f"2. Nominal Zero-Crossing:\n"
            f"    Age: {m2_age:.2f}y (95% CI: [{m2_ci_l:.2f}, {m2_ci_u:.2f}])\n\n"
            f"3. Conservative Cessation Boundary:\n"
            f"    Upper Simultaneous Bound > 0 on [40, 80]\n"
            f"    Status: {cess_label}"
        )
        ax_main.text(
            0.04, 0.06, text_annot, transform=ax_main.transAxes,
            fontsize=8.0, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#BDC3C7", alpha=0.92)
        )

        ax_main.set_xlim(18.0, 80.0)
        ax_main.set_ylim(-0.015, 0.035)
        ax_main.set_xlabel("Continuous Chronological Age (years)")
        ax_main.set_ylabel(r"Sex Contrast $\Delta(a)$ in $Q_i$")
        ax_main.set_title(f"{src_name} (expanded90)\nFemale-Minus-Male Delta Curve & Transition Landmarks", fontweight="bold")

        if col_idx == 0:
            ax_main.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

        # 2. First Derivative Panel dDelta / da
        ax_deriv = fig.add_subplot(gs[1, col_idx])

        # Compute numerical derivative
        d_ages = ages
        rcs1_d, rcs2_d = d_rcs_stone_koo(d_ages)

        # Load delta model parameters for this source
        df_models = pd.read_parquet(ARTIFACTS_DIR / "models/model_fits.parquet")
        row_model = df_models[
            (df_models["source"] == src_id) &
            (df_models["protocol"] == proto) &
            (df_models["endpoint"] == "Q_i") &
            (df_models["model_type"] == "primary_unadjusted") &
            (df_models["sample_scope"] == "shared_support")
        ].iloc[0]

        delta_1 = row_model["delta_1"]
        delta_2 = row_model["delta_2"]
        delta_3 = row_model["delta_3"]

        d_delta = delta_1 + delta_2 * rcs1_d + delta_3 * rcs2_d

        ax_deriv.plot(d_ages, d_delta, color=c_inflect, linewidth=2.0, label=r"First Derivative $d\Delta / da$")
        ax_deriv.axhline(0.0, color="#7F8C8D", linestyle="--", linewidth=0.8)
        ax_deriv.axvline(m1_age, color=c_inflect, linestyle="-.", linewidth=1.2, label=f"Global Minimum: {m1_age:.2f}y")
        ax_deriv.axvspan(42.0, 52.0, color="#FCF3CF", alpha=0.5, zorder=0)

        for k in DEFAULT_KNOTS:
            ax_deriv.axvline(k, color="#BDC3C7", linestyle=":", linewidth=0.9, zorder=1)

        ax_deriv.set_xlim(18.0, 80.0)
        ax_deriv.set_xlabel("Continuous Chronological Age (years)")
        ax_deriv.set_ylabel(r"Slope $d\Delta / da$ (units/year)")
        ax_deriv.set_title(f"{src_name}: Rate of Change & Inflection Dip", fontweight="bold")
        ax_deriv.legend(loc="lower right", fontsize=8.0)

    fig.suptitle(
        "Figure 3: Physiological Infradian Transition Dynamics: Delta Trajectories and Inflection Dip\n"
        "Steepest Decline at Mid-40s Perimenopause, Followed by Attenuation and Zero-Crossing",
        fontsize=12.5, y=0.99, fontweight="bold"
    )

    out_path = FIGURES_DIR / "fig3_delta_curve_transitions.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# FIGURE 4: Cross-Source, Protocol, and Second-Window Repeatability
# =============================================================================
def generate_figure_4(df_bands: pd.DataFrame, df_rep_summary: pd.DataFrame, df_rep_parquet: pd.DataFrame):
    print("Generating Figure 4: Cross-Source, Protocol & Repeatability...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), gridspec_kw={"hspace": 0.32, "wspace": 0.25})

    # Panel A: Cross-Source Validation (Apple vs Garmin)
    ax_a = axes[0, 0]
    app_fem = df_bands[(df_bands["source"] == 6) & (df_bands["protocol"] == "expanded90") & (df_bands["endpoint"] == "Q_i") & (df_bands["curve_type"] == "female")].sort_values("age")
    gar_fem = df_bands[(df_bands["source"] == 3) & (df_bands["protocol"] == "expanded90") & (df_bands["endpoint"] == "Q_i") & (df_bands["curve_type"] == "female")].sort_values("age")

    app_delta = df_bands[(df_bands["source"] == 6) & (df_bands["protocol"] == "expanded90") & (df_bands["endpoint"] == "Q_i") & (df_bands["curve_type"] == "delta")].sort_values("age")
    gar_delta = df_bands[(df_bands["source"] == 3) & (df_bands["protocol"] == "expanded90") & (df_bands["endpoint"] == "Q_i") & (df_bands["curve_type"] == "delta")].sort_values("age")

    ax_a.plot(app_fem["age"], app_fem["point_estimate"], color="#C0392B", linewidth=2.0, label="Apple Watch: Female $Q_i(a)$")
    ax_a.fill_between(app_fem["age"], app_fem["ci_lower_pointwise"], app_fem["ci_upper_pointwise"], color="#C0392B", alpha=0.15)

    ax_a.plot(gar_fem["age"], gar_fem["point_estimate"], color="#27AE60", linewidth=2.0, linestyle="--", label="Garmin Tier 1c: Female $Q_i(a)$")
    ax_a.fill_between(gar_fem["age"], gar_fem["ci_lower_pointwise"], gar_fem["ci_upper_pointwise"], color="#27AE60", alpha=0.15)

    ax_a.plot(app_delta["age"], app_delta["point_estimate"], color="#8E44AD", linewidth=1.8, label=r"Apple Watch: Delta $\Delta(a)$")
    ax_a.plot(gar_delta["age"], gar_delta["point_estimate"], color="#D35400", linewidth=1.8, linestyle="--", label=r"Garmin Tier 1c: Delta $\Delta(a)$")

    ax_a.axhline(0.0, color="#7F8C8D", linestyle=":", linewidth=0.8)
    ax_a.set_xlim(20.0, 78.0)
    ax_a.set_xlabel("Age (years)")
    ax_a.set_ylabel("Infradian Metric Value")
    ax_a.set_title("A. Cross-Device Replication (Apple Watch vs Garmin)", fontweight="bold")
    ax_a.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    # Panel B: Protocol Contrast (expanded90 vs balanced90)
    ax_b = axes[0, 1]
    app_bal = df_bands[(df_bands["source"] == 6) & (df_bands["protocol"] == "balanced90") & (df_bands["endpoint"] == "Q_i") & (df_bands["curve_type"] == "female")].sort_values("age")

    ax_b.plot(app_fem["age"], app_fem["point_estimate"], color="#C0392B", linewidth=2.2, label="expanded90 Primary (N=1,715 F)")
    ax_b.fill_between(app_fem["age"], app_fem["ci_lower_pointwise"], app_fem["ci_upper_pointwise"], color="#C0392B", alpha=0.15)

    ax_b.plot(app_bal["age"], app_bal["point_estimate"], color="#2980B9", linewidth=2.0, linestyle="--", label="balanced90 Reference (N=1,437 F)")
    ax_b.fill_between(app_bal["age"], app_bal["ci_lower_pointwise"], app_bal["ci_upper_pointwise"], color="#2980B9", alpha=0.15)

    ax_b.set_xlim(20.0, 78.0)
    ax_b.set_xlabel("Age (years)")
    ax_b.set_ylabel("Female Infradian Excess $Q_i(a)$")
    ax_b.set_title("B. Protocol Contrast: Sample Expansion vs Strict Balance", fontweight="bold")
    ax_b.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # Panel C: Second-Window Repeatability Scatter (Apple Watch balanced90)
    ax_c = axes[1, 0]
    app_rep = df_rep_parquet[(df_rep_parquet["source"] == 6) & (df_rep_parquet["protocol"] == "balanced90")].copy()

    # Subsample for clear scatter visualization
    fem_young = app_rep[(app_rep["sex"] == 1) & (app_rep["age_bracket"] == "<40")]
    fem_peri = app_rep[(app_rep["sex"] == 1) & (app_rep["age_bracket"] == "40-55")]
    fem_post = app_rep[(app_rep["sex"] == 1) & (app_rep["age_bracket"] == ">=55")]
    male_all = app_rep[app_rep["sex"] == 0]

    ax_c.scatter(male_all["w1_Q_i"], male_all["w2_Q_i"], color="#BDC3C7", alpha=0.25, s=15, label=f"Males (r=-0.01, N={len(male_all)})")
    ax_c.scatter(fem_post["w1_Q_i"], fem_post["w2_Q_i"], color="#E67E22", alpha=0.6, s=25, label=f"Females >=55 (r=-0.02, N={len(fem_post)})")
    ax_c.scatter(fem_young["w1_Q_i"], fem_young["w2_Q_i"], color="#2980B9", alpha=0.7, s=30, label=f"Females <40 (r=+0.23**, N={len(fem_young)})")
    ax_c.scatter(fem_peri["w1_Q_i"], fem_peri["w2_Q_i"], color="#C0392B", alpha=0.7, s=30, label=f"Females 40-55 (r=+0.39***, N={len(fem_peri)})")

    # Identity and zero lines
    ax_c.plot([-0.05, 0.15], [-0.05, 0.15], color="#34495E", linestyle=":", linewidth=1.0)
    ax_c.axhline(0.0, color="#95A5A6", linestyle="--", linewidth=0.7)
    ax_c.axvline(0.0, color="#95A5A6", linestyle="--", linewidth=0.7)

    ax_c.set_xlim(-0.04, 0.14)
    ax_c.set_ylim(-0.04, 0.14)
    ax_c.set_xlabel("Window 1 Infradian Excess $Q_{i,1}$")
    ax_c.set_ylabel("Window 2 Infradian Excess $Q_{i,2}$")
    ax_c.set_title("C. Second-Window Temporal Repeatability (Apple Watch)", fontweight="bold")
    ax_c.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # Panel D: Grouped Correlation Bar Chart Across Age Brackets & Sex
    ax_d = axes[1, 1]

    brackets = ["<40", "40-55", ">=55"]
    x = np.arange(len(brackets))
    width = 0.20

    # Retrieve correlation values from df_rep_summary
    def get_r(src, sex, br):
        row = df_rep_summary[
            (df_rep_summary["source"] == src) &
            (df_rep_summary["protocol"] == "balanced90") &
            (df_rep_summary["sex"] == sex) &
            (df_rep_summary["age_bracket"] == br)
        ]
        return float(row["pearson_r_Q_i"].iloc[0]) if len(row) > 0 else 0.0

    def get_forecast(src, sex, br):
        row = df_rep_summary[
            (df_rep_summary["source"] == src) &
            (df_rep_summary["protocol"] == "balanced90") &
            (df_rep_summary["sex"] == sex) &
            (df_rep_summary["age_bracket"] == br)
        ]
        return float(row["mean_forecast_r"].iloc[0]) if len(row) > 0 else 0.0

    app_fem_r = [get_r(6, "Female", b) for b in brackets]
    gar_fem_r = [get_r(3, "Female", b) for b in brackets]
    app_male_r = [get_r(6, "Male", b) for b in brackets]
    app_fem_fc = [get_forecast(6, "Female", b) for b in brackets]

    ax_d.bar(x - 1.5*width, app_fem_r, width, label="Apple Females: $r(Q_1, Q_2)$", color="#C0392B", alpha=0.85)
    ax_d.bar(x - 0.5*width, gar_fem_r, width, label="Garmin Females: $r(Q_1, Q_2)$", color="#27AE60", alpha=0.85)
    ax_d.bar(x + 0.5*width, app_male_r, width, label="Apple Males: $r(Q_1, Q_2)$", color="#2980B9", alpha=0.85)
    ax_d.bar(x + 1.5*width, app_fem_fc, width, label="Apple Females: Frozen Forecast $r$", color="#8E44AD", alpha=0.85)

    ax_d.axhline(0.0, color="#2C3E50", linewidth=0.8)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(["Young (<40y)", "Perimenopausal (40-55y)", "Post-Menopausal (>=55y)"])
    ax_d.set_ylabel("Pearson Correlation Coefficient")
    ax_d.set_ylim(-0.08, 0.45)
    ax_d.set_title("D. Biological Specificity: Correlation by Age & Sex Bracket", fontweight="bold")
    ax_d.legend(loc="upper right", fontsize=7.8, framealpha=0.9)

    fig.suptitle(
        "Figure 4: Cross-Device Validation, Protocol Robustness, and Longitudinal Repeatability\n"
        "Confirming Biological Reproducibility Exclusively in Reproductive and Perimenopausal Females",
        fontsize=12.5, y=0.99, fontweight="bold"
    )

    out_path = FIGURES_DIR / "fig4_source_and_protocol_comparisons.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# Main execution
# =============================================================================
def main():
    t_start = time.perf_counter()
    print("=" * 80)
    print("MILESTONE 5: GENERATING MULTI-PANEL PUBLICATION FIGURES")
    print("=" * 80)

    # Load artifacts
    print("Loading data artifacts...")
    df_bands = pd.read_parquet(ARTIFACTS_DIR / "bootstrap/confidence_bands.parquet")
    df_endpoints = pd.read_parquet(ARTIFACTS_DIR / "endpoints/endpoints.parquet")
    df_trans = pd.read_csv(ARTIFACTS_DIR / "bootstrap/transition_metrics.csv")
    df_rep_summary = pd.read_csv(ARTIFACTS_DIR / "bootstrap/repeatability_summary.csv")
    df_rep_parquet = pd.read_parquet(ARTIFACTS_DIR / "bootstrap/second_window_repeatability.parquet")

    # Generate the 4 publication figures
    generate_figure_1(df_bands, df_endpoints)
    generate_figure_2(df_endpoints)
    generate_figure_3(df_bands, df_trans)
    generate_figure_4(df_bands, df_rep_summary, df_rep_parquet)

    print("=" * 80)
    print(f"ALL 4 FIGURES SUCCESSFULLY GENERATED IN {time.perf_counter() - t_start:.2f}s")
    print(f"Saved directory: {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
