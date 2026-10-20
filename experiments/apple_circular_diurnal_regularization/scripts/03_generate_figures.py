"""Milestone 4: Publication Figures Generation Script.

Generates:
  1. figures/fig1_diurnal_template_smoothing.png:
     - 24-hour diurnal median template comparing raw H_clock vs circular regularized
       templates (sigma in {1, 2, 4} bins) on a representative low-occupancy participant.
     - High-magnification panel zooming in on the midnight boundary transition (22:30 - 01:30)
       demonstrating seamless periodic wrap-around without edge steps or reflection distortion.
  2. figures/fig2_imputation_error_curves.png:
     - Multi-panel benchmark evaluation (2x2 grid):
       (a) Reconstruction MAE across gap lengths (short, medium, long).
       (b) Reconstruction RMSE across gap lengths.
       (c) Boundary step jump Delta_boundary across gap lengths.
       (d) Reconstruction MAE stratified across participant occupancy tiers (low, medium, high).

Author: Computational Medicine / ML Specialist
Date: 2026-10-20
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
REPO_ROOT = PROJECT_DIR.parent.parent
CACHE_5M_DIR = REPO_ROOT / "experiments" / "apple_cohort_scaling" / "artifacts" / "cache_5min"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from circular_smoothing import circular_gaussian_smooth

# Configure matplotlib publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.title_fontsize": 10,
    "figure.titlesize": 13,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Distinct, accessible color palette
PALETTE = {
    "raw": "#4A5568",            # Slate gray
    "s1": "#2B6CB0",             # Strong blue (Primary recommendation)
    "s2": "#DD6B20",             # Vibrant orange
    "s4": "#805AD5",             # Purple
    "linear": "#38A169",         # Emerald green
}


def generate_figure1_diurnal_template_smoothing() -> Path:
    """Generate Figure 1: 24-hour diurnal profile and midnight wrap-around zoom."""
    print("Generating Figure 1: Diurnal Template Smoothing & Midnight Wrap-Around...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Low-occupancy participant: CID 1206718 (occupancy = 31.1%, 99 days recorded)
    # Exhibits physiological diurnal variation with high-frequency median jitter and midnight discontinuity
    cid = "1206718"
    npz_path = CACHE_5M_DIR / f"customer_{cid}.npz"
    with np.load(npz_path) as npz:
        hr = npz["hr"]  # (D, 288)

    with np.errstate(invalid="ignore"):
        t_clock = np.nanmedian(hr, axis=0).astype(np.float64)

    # Regularized templates via circular Gaussian convolution
    t_s1 = circular_gaussian_smooth(t_clock, sigma=1.0)
    t_s2 = circular_gaussian_smooth(t_clock, sigma=2.0)
    t_s4 = circular_gaussian_smooth(t_clock, sigma=4.0)

    # 288 bins = 24 hours (5 minutes per bin)
    bins = np.arange(288)
    hours = bins * (24.0 / 288.0)  # hours from 0.0 to 24.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.0, 8.2), gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.36}
    )

    # -------------------------------------------------------------
    # Panel A: Full 24-Hour Diurnal Template
    # -------------------------------------------------------------
    # Mark circadian phases
    ax1.axvspan(1.0, 5.5, color="#E2E8F0", alpha=0.5, label="Nocturnal Resting Quiescence (01:00–05:30)")
    ax1.axvspan(11.5, 17.5, color="#FEFCBF", alpha=0.35, label="Diurnal Active Plateau (11:30–17:30)")

    # Plot raw median template as points and faint step line
    ax1.plot(
        hours, t_clock, color=PALETTE["raw"], alpha=0.35, linewidth=1.1,
        linestyle="-", label=r"Raw Clock Median $H_{\mathrm{clock}}$"
    )
    ax1.scatter(
        hours[::2], t_clock[::2], color=PALETTE["raw"], alpha=0.55, s=11,
        zorder=3, edgecolors="none"
    )

    # Plot circular Gaussian regularized profiles
    ax1.plot(
        hours, t_s1, color=PALETTE["s1"], linewidth=2.4,
        label=r"Circular Gaussian $\sigma = 1$ bin (5 min, $\sigma^*$, Primary)"
    )
    ax1.plot(
        hours, t_s2, color=PALETTE["s2"], linewidth=1.8, linestyle="--",
        label=r"Circular Gaussian $\sigma = 2$ bins (10 min)"
    )
    ax1.plot(
        hours, t_s4, color=PALETTE["s4"], linewidth=1.8, linestyle="-.",
        label=r"Circular Gaussian $\sigma = 4$ bins (20 min)"
    )

    ax1.set_xlim(0, 24)
    ax1.set_ylim(64, 102)
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    ax1.set_xlabel("Clock Time (24-Hour Diurnal Cycle)")
    ax1.set_ylabel("Heart Rate (bpm)")
    ax1.set_title(
        r"$\mathbf{(a)}$ 24-Hour Diurnal Profile Regularization (Low-Occupancy Participant: CID 1206718, "
        r"$\mathrm{Occupancy} = 31.1\%$, $D=99$ days)",
        loc="left", pad=8
    )

    # Place legend on upper left / center where there is plenty of vertical headroom
    ax1.legend(
        loc="upper left", bbox_to_anchor=(0.02, 0.98), framealpha=0.94,
        edgecolor="#CBD5E0", ncol=2, fontsize=9.0
    )

    # Highlight midnight transition region
    ax1.axvspan(22.5, 24.0, color="#FED7D7", alpha=0.25)
    ax1.axvspan(0.0, 1.5, color="#FED7D7", alpha=0.25)
    ax1.text(
        23.25, 96.5, "Midnight\nTransition", color="#C53030", fontsize=8.5,
        ha="center", va="center", weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF5F5", edgecolor="#E53E3E", alpha=0.92)
    )

    # -------------------------------------------------------------
    # Panel B: High-Magnification Midnight Boundary Wrap-Around Transition
    # Bins 270-287 (22:30-24:00) and Bins 0-18 (00:00-01:30)
    # -------------------------------------------------------------
    pre_bins = np.arange(270, 288)
    post_bins = np.arange(0, 19)

    pre_rel_min = (pre_bins - 288) * 5.0
    post_rel_min = post_bins * 5.0
    rel_min = np.concatenate([pre_rel_min, post_rel_min])

    raw_zoom = np.concatenate([t_clock[pre_bins], t_clock[post_bins]])
    s1_zoom = np.concatenate([t_s1[pre_bins], t_s1[post_bins]])
    s2_zoom = np.concatenate([t_s2[pre_bins], t_s2[post_bins]])
    s4_zoom = np.concatenate([t_s4[pre_bins], t_s4[post_bins]])

    # Raw points and connection
    ax2.plot(
        rel_min, raw_zoom, color=PALETTE["raw"], alpha=0.45, linewidth=1.2,
        linestyle=":", marker="o", markersize=4.5, label=r"Raw Bins $H_{\mathrm{clock}}$"
    )

    # Regularized curves
    ax2.plot(
        rel_min, s1_zoom, color=PALETTE["s1"], linewidth=2.4,
        label=r"Circular Gaussian $\sigma = 1$ bin (5 min, $\sigma^*$)"
    )
    ax2.plot(
        rel_min, s2_zoom, color=PALETTE["s2"], linewidth=2.0, linestyle="--",
        label=r"Circular Gaussian $\sigma = 2$ bins (10 min)"
    )
    ax2.plot(
        rel_min, s4_zoom, color=PALETTE["s4"], linewidth=2.0, linestyle="-.",
        label=r"Circular Gaussian $\sigma = 4$ bins (20 min)"
    )

    # Midnight vertical boundary line
    ax2.axvline(0, color="#E53E3E", linestyle="-", linewidth=1.8, alpha=0.9)
    ax2.text(
        0, 86.8, "Midnight Wrap-Around Boundary\nBin 287 (23:55) $\\leftrightarrow$ Bin 0 (00:00)",
        color="#C53030", fontsize=9.0, ha="center", va="top", weight="semibold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFF5F5", edgecolor="#E53E3E", alpha=0.94)
    )

    # Callout annotations
    ax2.annotate(
        r"Raw midnight jump: $\Delta = -7.0$ bpm (80.0 $\to$ 73.0), then $+11.5$ bpm",
        xy=(-2.5, 76.5), xytext=(-82, 71.5),
        arrowprops=dict(arrowstyle="->", color="#C53030", lw=1.2),
        fontsize=8.8, color="#C53030", weight="medium",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF5F5", edgecolor="#FEB2B2", alpha=0.92)
    )

    ax2.annotate(
        r"Circular Gaussian wraps $C^\infty$ smoothly across midnight" "\n"
        r"with zero edge steps or reflection distortion",
        xy=(2.5, 78.0), xytext=(12, 82.5),
        arrowprops=dict(arrowstyle="->", color=PALETTE["s1"], lw=1.3),
        fontsize=8.8, color=PALETTE["s1"], weight="medium",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#EBF8FF", edgecolor="#BEE3F8", alpha=0.94)
    )

    ax2.set_xlim(-90, 90)
    ax2.set_ylim(69, 88.5)
    tick_locs = np.arange(-90, 95, 15)
    tick_labels = [
        "22:30\n(-90m)", "22:45\n(-75m)", "23:00\n(-60m)", "23:15\n(-45m)",
        "23:30\n(-30m)", "23:45\n(-15m)", "00:00\n(0m)", "00:15\n(+15m)",
        "00:30\n(+30m)", "00:45\n(+45m)", "01:00\n(+60m)", "01:15\n(+75m)", "01:30\n(+90m)"
    ]
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels(tick_labels, fontsize=8.5)
    ax2.set_xlabel("Relative Time Across Midnight Boundary (Clock Time & Elapsed Minutes)")
    ax2.set_ylabel("Heart Rate (bpm)")
    ax2.set_title(
        r"$\mathbf{(b)}$ High-Magnification Midnight Wrap-Around Boundary (22:30 to 01:30, Bins 270–287 $\to$ 0–18)",
        loc="left", pad=8
    )

    # Place legend in upper left of Panel B where values are ~75 bpm
    ax2.legend(loc="upper left", bbox_to_anchor=(0.015, 0.98), framealpha=0.94, edgecolor="#CBD5E0", fontsize=8.8)

    out_path = FIGURES_DIR / "fig1_diurnal_template_smoothing.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1 to: {out_path} (300 DPI, {out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def generate_figure2_imputation_error_curves() -> Path:
    """Generate Figure 2: Multi-panel imputation benchmark fidelity curves."""
    print("Generating Figure 2: Imputation Error Curves & Benchmark Fidelity...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(RESULTS_DIR / "imputation_benchmark_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.6))
    ax_mae, ax_rmse = axes[0, 0], axes[0, 1]
    ax_jump, ax_occ = axes[1, 0], axes[1, 1]

    gap_order = ["short", "medium", "long"]
    gap_labels = ["Short\n(1–3 bins, 5–15m)", "Medium\n(6–12 bins, 30–60m)", "Long\n(24–48 bins, 2–4h)"]
    x_pos = np.arange(len(gap_order))
    width = 0.16

    methods = [
        ("H_linear", r"Linear Interp $H_{\mathrm{linear}}$", PALETTE["linear"]),
        ("H_clock", r"Raw Clock Median $H_{\mathrm{clock}}$", PALETTE["raw"]),
        ("H_circ_gauss_s1", r"Circular Gauss $\sigma=1$ bin (5 min, $\sigma^*$)", PALETTE["s1"]),
        ("H_circ_gauss_s2", r"Circular Gauss $\sigma=2$ bins (10 min)", PALETTE["s2"]),
        ("H_circ_gauss_s4", r"Circular Gauss $\sigma=4$ bins (20 min)", PALETTE["s4"]),
    ]

    by_gap = metrics_df[metrics_df["slice"] == "by_gap_tier"]

    # -------------------------------------------------------------
    # (a) Reconstruction MAE across Gap Lengths
    # -------------------------------------------------------------
    for idx, (m_key, m_label, m_color) in enumerate(methods):
        sub = by_gap[by_gap["method"] == m_key].set_index("gap_tier").loc[gap_order]
        offsets = x_pos + (idx - 2) * width
        ax_mae.bar(
            offsets, sub["mae_mean"], width=width, yerr=sub["mae_sem"],
            label=m_label, color=m_color, alpha=0.88, edgecolor="black", linewidth=0.6,
            capsize=3, error_kw={"elinewidth": 1.0, "capthick": 1.0}
        )

    ax_mae.set_xticks(x_pos)
    ax_mae.set_xticklabels(gap_labels)
    ax_mae.set_ylabel("Reconstruction MAE (bpm)")
    ax_mae.set_ylim(0, 12.0)
    ax_mae.set_title(r"$\mathbf{(a)}$ Reconstruction MAE across Gap Lengths", loc="left", pad=8)

    # -------------------------------------------------------------
    # (b) Reconstruction RMSE across Gap Lengths
    # -------------------------------------------------------------
    for idx, (m_key, m_label, m_color) in enumerate(methods):
        sub = by_gap[by_gap["method"] == m_key].set_index("gap_tier").loc[gap_order]
        offsets = x_pos + (idx - 2) * width
        ax_rmse.bar(
            offsets, sub["rmse_mean"], width=width, yerr=sub["rmse_sem"],
            label=m_label, color=m_color, alpha=0.88, edgecolor="black", linewidth=0.6,
            capsize=3, error_kw={"elinewidth": 1.0, "capthick": 1.0}
        )

    ax_rmse.set_xticks(x_pos)
    ax_rmse.set_xticklabels(gap_labels)
    ax_rmse.set_ylabel("Reconstruction RMSE (bpm)")
    ax_rmse.set_ylim(0, 13.0)
    ax_rmse.set_title(r"$\mathbf{(b)}$ Reconstruction RMSE across Gap Lengths", loc="left", pad=8)

    # -------------------------------------------------------------
    # (c) Boundary Step Jump Delta_boundary across Gap Lengths
    # -------------------------------------------------------------
    for idx, (m_key, m_label, m_color) in enumerate(methods):
        sub = by_gap[by_gap["method"] == m_key].set_index("gap_tier").loc[gap_order]
        offsets = x_pos + (idx - 2) * width
        ax_jump.bar(
            offsets, sub["boundary_jump_mean"], width=width, yerr=sub["boundary_jump_sem"],
            label=m_label, color=m_color, alpha=0.88, edgecolor="black", linewidth=0.6,
            capsize=3, error_kw={"elinewidth": 1.0, "capthick": 1.0}
        )

    ax_jump.set_xticks(x_pos)
    ax_jump.set_xticklabels(gap_labels)
    ax_jump.set_ylabel(r"Boundary Step Jump $\Delta_{\mathrm{boundary}}$ (bpm)")
    ax_jump.set_ylim(0, 12.0)
    ax_jump.set_title(
        r"$\mathbf{(c)}$ Boundary Discontinuity $\Delta_{\mathrm{boundary}} = |x_{\mathrm{fill}}(t_0) - x_{\mathrm{obs}}(t_0-1)|$",
        loc="left", pad=8
    )

    # Annotate dramatic difference in boundary jump
    ax_jump.annotate(
        r"$H_{\mathrm{linear}}$ anchors to edge: $\Delta = 0.39$ bpm",
        xy=(2.0 - 2 * width, 0.45), xytext=(1.05, 3.2),
        arrowprops=dict(arrowstyle="->", color=PALETTE["linear"], lw=1.2),
        fontsize=8.5, color=PALETTE["linear"], weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0FFF4", edgecolor="#9AE6B4", alpha=0.92)
    )

    # -------------------------------------------------------------
    # (d) Error Stratified by Occupancy Tier
    # -------------------------------------------------------------
    by_occ = metrics_df[metrics_df["slice"] == "by_occupancy_tier"]
    occ_order = ["low", "medium", "high"]
    occ_labels = ["Low Occupancy\n(<50% valid bins)", "Medium Occupancy\n(50–80% valid bins)", "High Occupancy\n(>80% valid bins)"]
    x_occ = np.arange(len(occ_order))

    for idx, (m_key, m_label, m_color) in enumerate(methods):
        sub = by_occ[by_occ["method"] == m_key].set_index("occupancy_tier").loc[occ_order]
        offsets = x_occ + (idx - 2) * width
        ax_occ.bar(
            offsets, sub["mae_mean"], width=width, yerr=sub["mae_sem"],
            label=m_label, color=m_color, alpha=0.88, edgecolor="black", linewidth=0.6,
            capsize=3, error_kw={"elinewidth": 1.0, "capthick": 1.0}
        )

    ax_occ.set_xticks(x_occ)
    ax_occ.set_xticklabels(occ_labels)
    ax_occ.set_ylabel("Reconstruction MAE (bpm)")
    ax_occ.set_ylim(0, 13.0)
    ax_occ.set_title(r"$\mathbf{(d)}$ Imputation Fidelity Stratified by User Occupancy Tier", loc="left", pad=8)

    # Unified single legend placed at the top center of the entire figure
    handles, labels = ax_mae.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015),
        ncol=5, framealpha=0.95, edgecolor="#CBD5E0", fontsize=9.2
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = FIGURES_DIR / "fig2_imputation_error_curves.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 2 to: {out_path} (300 DPI, {out_path.stat().st_size / 1024:.1f} KB)")
    return out_path



def main() -> None:
    """Execute figure generation suite."""
    print("=" * 70)
    print("STARTING MILESTONE 4: PUBLICATION FIGURES GENERATION")
    print("=" * 70)
    f1 = generate_figure1_diurnal_template_smoothing()
    f2 = generate_figure2_imputation_error_curves()
    print("=" * 70)
    print("FIGURE GENERATION COMPLETED SUCCESSFULLY!")
    print(f"  Fig 1: {f1}")
    print(f"  Fig 2: {f2}")
    print("=" * 70)


if __name__ == "__main__":
    main()
