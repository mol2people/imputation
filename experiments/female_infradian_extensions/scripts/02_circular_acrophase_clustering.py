#!/usr/bin/env python3
"""
Circular Statistics & Acrophase Clustering Suite
Path: experiments/female_infradian_extensions/scripts/02_circular_acrophase_clustering.py

Evaluates biological desynchrony versus calendar/lunar entrainment using circular
coordinate transformations and the analytical Rayleigh test of circular uniformity.

Cohorts evaluated:
1. Target: Reproductive-age females (sex == 'Female', age_group == 'Young Adults', screened_pass == True, N=502).
2. Negative Control 1: Age-matched males (sex == 'Male', age_group == 'Young Adults', screened_pass == True, N=1209).

Signals evaluated:
- Resting HR proxy: hr_p10
- Nocturnal HR: night_hr
- Wear compliance: occ_daily

Phase transformations:
- Calendar Epoch Phase: Phi_cal = (phi_rel - (2*pi / T*) * date_0) mod (2*pi) mapped to [-pi, pi]
- 28-Day Month / Lunar Phase: t_peak = (phi_rel mod (2*pi)) / (2*pi / T*)
                              theta_28 = ((2*pi / 28.0) * (date_0 + t_peak)) mod (2*pi) mapped to [-pi, pi]

Deliverables:
- CSV: experiments/female_infradian_extensions/results/acrophase_uniformity_tests.csv
- Figure: experiments/female_infradian_extensions/figures/fig2_acrophase_circular_polar_plots.png
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Matplotlib publication styling
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
mpl.rcParams["axes.edgecolor"] = "#2c3e50"
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["grid.color"] = "#e0e0e0"
mpl.rcParams["grid.linestyle"] = ":"
mpl.rcParams["grid.alpha"] = 0.7

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXT_DIR = PROJECT_ROOT / "experiments" / "female_infradian_extensions"
SCRIPTS_DIR = EXT_DIR / "scripts"
RESULTS_DIR = EXT_DIR / "results"
FIGURES_DIR = EXT_DIR / "figures"
CYCLES_RESULTS = PROJECT_ROOT / "experiments" / "female_infradian_cycles" / "results"
CACHE_DIR = PROJECT_ROOT / "experiments" / "apple_cohort_scaling" / "artifacts" / "cache_5min"

# Dynamic import of 00_verify_extension_methods.py
verify_script_path = SCRIPTS_DIR / "00_verify_extension_methods.py"
spec = importlib.util.spec_from_file_location("verify_extension_methods", verify_script_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load module from {verify_script_path}")
verify_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_mod)

rayleigh_test = verify_mod.rayleigh_test
map_calendar_phase = verify_mod.map_calendar_phase
map_28day_phase = verify_mod.map_28day_phase


def load_infradian_metrics() -> pd.DataFrame:
    """Loads and filters infradian metrics for screened target and control cohorts."""
    metrics_path = CYCLES_RESULTS / "infradian_metrics_by_participant.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Infradian metrics file not found: {metrics_path}")

    print(f"Loading participant infradian metrics from: {metrics_path}")
    df = pd.read_csv(metrics_path)
    print(f"Total cohort size: {len(df):,} participants")

    # Filter to screened Young Adults
    young_pass = (df["screened_pass"] == True) & (df["age_group"] == "Young Adults")
    females_mask = young_pass & (df["sex"] == "Female")
    males_mask = young_pass & (df["sex"] == "Male")

    n_females = int(females_mask.sum())
    n_males = int(males_mask.sum())
    print(f"Screened Young Reproductive Females: N = {n_females} (expected 502)")
    print(f"Screened Young Control Males:        N = {n_males} (expected 1,209)")

    assert n_females == 502, f"Expected 502 screened females, found {n_females}"
    assert n_males == 1209, f"Expected 1,209 screened males, found {n_males}"

    cohort_df = df[females_mask | males_mask].copy()
    cohort_df["cohort_name"] = np.where(
        cohort_df["sex"] == "Female", "Reproductive Females", "Young Males"
    )
    return cohort_df


def extract_baseline_dates(customer_ids: List[int]) -> Dict[int, int]:
    """Extracts date_0 (first observed Unix epoch day) from cache_5min npz files."""
    print(f"Extracting baseline dates for {len(customer_ids):,} participants from cache_5min...")
    t0 = time.perf_counter()
    date_0_map: Dict[int, int] = {}

    missing_count = 0
    for cid in customer_ids:
        cache_path = CACHE_DIR / f"customer_{cid}.npz"
        if not cache_path.exists():
            missing_count += 1
            continue

        try:
            # Fast zip inspection without loading entire heart rate arrays
            with zipfile.ZipFile(cache_path, "r") as z:
                with z.open("date.npy") as f_date:
                    # Read npy header and first int64
                    # npy format: 6-byte magic (\x93NUMPY), 1-byte major, 1-byte minor, 2-byte header len
                    magic = f_date.read(10)
                    header_len = int.from_bytes(magic[8:10], byteorder="little")
                    header_bytes = f_date.read(header_len)
                    # Next 8 bytes are the first int64 date value
                    first_date_bytes = f_date.read(8)
                    date_0 = int(np.frombuffer(first_date_bytes, dtype=np.int64)[0])
                    date_0_map[cid] = date_0
        except Exception as e:
            # Fallback to standard numpy load if custom parser encounters variation
            try:
                npz = np.load(cache_path)
                date_0 = int(npz["date"][0])
                date_0_map[cid] = date_0
            except Exception as e2:
                print(f"Error reading cache for customer {cid}: {e2}")
                missing_count += 1

    elapsed = time.perf_counter() - t0
    print(f"Loaded {len(date_0_map):,} baseline dates in {elapsed:.2f}s (missing: {missing_count})")
    assert missing_count == 0, f"Missing cache files for {missing_count} participants!"
    return date_0_map


def compute_cohort_circular_statistics(df: pd.DataFrame, date_0_map: Dict[int, int]) -> pd.DataFrame:
    """Computes circular phase mappings and analytical Rayleigh tests across cohorts and signals."""
    cohorts = [
        ("Reproductive Females", df[df["cohort_name"] == "Reproductive Females"]),
        ("Young Males", df[df["cohort_name"] == "Young Males"]),
    ]

    signals = [
        ("hr_p10", "Resting HR (p10)"),
        ("night_hr", "Nocturnal HR"),
        ("occ_daily", "Wear Compliance"),
    ]

    phase_types = [
        ("calendar_epoch", "Calendar Epoch Phase (Phi_cal)"),
        ("month_28day", "28-Day Month / Lunar Phase (theta_28)"),
    ]

    results_rows = []

    print("\n" + "=" * 80)
    print("CIRCULAR STATISTICS & ANALYTICAL RAYLEIGH UNIFORMITY TESTS")
    print("=" * 80)

    for cohort_label, sub_df in cohorts:
        cids = sub_df["customer_id"].values
        d0_arr = np.array([date_0_map[cid] for cid in cids], dtype=np.float64)

        for sig_id, sig_name in signals:
            period_col = f"cos_period_{sig_id}"
            acro_col = f"cos_acrophase_{sig_id}"

            periods = sub_df[period_col].values.astype(np.float64)
            acrophases = sub_df[acro_col].values.astype(np.float64)

            # Valid mask for finite entries
            valid_mask = np.isfinite(periods) & np.isfinite(acrophases) & np.isfinite(d0_arr)
            valid_n = int(np.sum(valid_mask))

            p_valid = periods[valid_mask]
            a_valid = acrophases[valid_mask]
            d0_valid = d0_arr[valid_mask]

            # Compute phase coordinate transforms
            phi_cal = map_calendar_phase(a_valid, p_valid, d0_valid)
            theta_28 = map_28day_phase(a_valid, p_valid, d0_valid)

            for p_type_id, p_type_name in phase_types:
                angles = phi_cal if p_type_id == "calendar_epoch" else theta_28
                ray_res = rayleigh_test(angles)

                R_bar = ray_res["R_bar"]
                theta_mean = ray_res["theta_mean"]
                circ_var = ray_res["circular_var"]
                Z = ray_res["Z"]
                p_val = ray_res["p_value"]
                n_pts = ray_res["N"]

                mean_deg = float(np.rad2deg(theta_mean))
                rej_05 = bool(p_val < 0.05)
                rej_01 = bool(p_val < 0.01)

                # Formulate biological interpretation
                if sig_id in ["hr_p10", "night_hr"]:
                    if cohort_label == "Reproductive Females":
                        if not rej_05:
                            interp = (
                                "Circular uniformity confirmed (p > 0.05): consistent with "
                                "autonomous endogenous biological rhythmicity desynchronized across population"
                            )
                        else:
                            interp = "Significant phase clustering detected in female heart rate"
                    else:
                        if not rej_05:
                            interp = "Circular uniformity in male control heart rate (no population alignment)"
                        else:
                            interp = "Significant phase clustering in male heart rate"
                else:  # occ_daily
                    if rej_01:
                        interp = (
                            "Significant phase clustering (p < 0.01): confirms external calendar "
                            "and behavioral entrainment of device wear compliance"
                        )
                    elif rej_05:
                        interp = "Moderate phase clustering in wear compliance (p < 0.05)"
                    else:
                        interp = "Uniform wear compliance phase distribution"

                results_rows.append({
                    "cohort": cohort_label,
                    "signal": sig_id,
                    "phase_type": p_type_id,
                    "n_participants": n_pts,
                    "mean_resultant_R_bar": round(R_bar, 5),
                    "circular_mean_deg": round(mean_deg, 2),
                    "circular_var": round(circ_var, 5),
                    "rayleigh_Z": round(Z, 4),
                    "p_value": p_val,
                    "uniformity_rejected_p05": rej_05,
                    "uniformity_rejected_p01": rej_01,
                    "biological_interpretation": interp,
                })

                print(
                    f"[{cohort_label[:12]:<12}] {sig_id:<9} | {p_type_id:<14} | N={n_pts:<4} | "
                    f"R_bar={R_bar:.4f} | circ_var={circ_var:.4f} | Z={Z:6.2f} | p={p_val:.4e} | "
                    f"Rej(0.05): {str(rej_05):<5} | Rej(0.01): {str(rej_01):<5}"
                )

    print("=" * 80 + "\n")
    return pd.DataFrame(results_rows)


def plot_fig2_acrophase_polar_plots(
    df: pd.DataFrame,
    date_0_map: Dict[int, int],
    test_results_df: pd.DataFrame,
) -> Path:
    """Generates publication-quality Figure 2 polar circular plots.

    Layout:
    3-panel high-resolution (300 dpi) polar rose plots comparing:
    - Panel A: Female Resting HR acrophase distribution (demonstrating circular dispersion/uniformity).
    - Panel B: Male Resting HR acrophase distribution (negative control).
    - Panel C: Wear compliance acrophase distribution.

    Includes:
    - Binned rose histogram with density scale.
    - Mean resultant vector arrow R_bar with direction theta_mean.
    - Annotated Rayleigh Z and analytical circular p-value.
    - Critical significance radius R_crit at alpha = 0.05.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "fig2_acrophase_circular_polar_plots.png"
    print(f"Generating publication Figure 2 at: {out_path}")

    # Extract phase distributions
    females_df = df[df["cohort_name"] == "Reproductive Females"]
    males_df = df[df["cohort_name"] == "Young Males"]

    f_cids = females_df["customer_id"].values
    m_cids = males_df["customer_id"].values

    f_d0 = np.array([date_0_map[cid] for cid in f_cids], dtype=np.float64)
    m_d0 = np.array([date_0_map[cid] for cid in m_cids], dtype=np.float64)

    # 1. Female Resting HR (calendar phase & 28d phase)
    f_hr_t = females_df["cos_period_hr_p10"].values.astype(np.float64)
    f_hr_phi = females_df["cos_acrophase_hr_p10"].values.astype(np.float64)
    phi_cal_female_hr = map_calendar_phase(f_hr_phi, f_hr_t, f_d0)
    th_28_female_hr = map_28day_phase(f_hr_phi, f_hr_t, f_d0)

    # 2. Male Resting HR (calendar phase & 28d phase)
    m_hr_t = males_df["cos_period_hr_p10"].values.astype(np.float64)
    m_hr_phi = males_df["cos_acrophase_hr_p10"].values.astype(np.float64)
    phi_cal_male_hr = map_calendar_phase(m_hr_phi, m_hr_t, m_d0)
    th_28_male_hr = map_28day_phase(m_hr_phi, m_hr_t, m_d0)

    # 3. Female Wear Compliance (calendar phase & 28d phase)
    f_occ_t = females_df["cos_period_occ_daily"].values.astype(np.float64)
    f_occ_phi = females_df["cos_acrophase_occ_daily"].values.astype(np.float64)
    phi_cal_wear = map_calendar_phase(f_occ_phi, f_occ_t, f_d0)
    th_28_wear = map_28day_phase(f_occ_phi, f_occ_t, f_d0)

    # Setup 1x3 polar subplot figure with generous margins to avoid any label collision
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.8), subplot_kw={"projection": "polar"}, dpi=300)
    fig.subplots_adjust(top=0.68, bottom=0.24, left=0.06, right=0.94, wspace=0.34)

    # Define specifications for each panel
    panels_spec = [
        {
            "ax": axes[0],
            "angles_cal": phi_cal_female_hr,
            "angles_28": th_28_female_hr,
            "panel_letter": "A",
            "title": "Panel A: Reproductive Females\nResting HR (p10) Acrophases",
            "subtitle": "Autonomous Endogenous Rhythmicity",
            "color": "#1b7837",  # Deep green
            "cohort": "Reproductive Females",
            "signal": "hr_p10",
        },
        {
            "ax": axes[1],
            "angles_cal": phi_cal_male_hr,
            "angles_28": th_28_male_hr,
            "panel_letter": "B",
            "title": "Panel B: Young Males (Control)\nResting HR (p10) Acrophases",
            "subtitle": "Negative Control Physiology",
            "color": "#2166ac",  # Slate blue
            "cohort": "Young Males",
            "signal": "hr_p10",
        },
        {
            "ax": axes[2],
            "angles_cal": phi_cal_wear,
            "angles_28": th_28_wear,
            "panel_letter": "C",
            "title": "Panel C: Wear Compliance (occ_daily)\nDaily Recording Acrophases",
            "subtitle": "Exogenous Schedule Entrainment Test",
            "color": "#b2182b",  # Crimson / amber red
            "cohort": "Reproductive Females",
            "signal": "occ_daily",
        },
    ]

    n_bins = 24  # 15-degree angular bins
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = 2.0 * np.pi / n_bins

    for p in panels_spec:
        ax = p["ax"]
        angles_cal = p["angles_cal"]
        angles_28 = p["angles_28"]

        ray_cal = rayleigh_test(angles_cal)
        ray_28 = rayleigh_test(angles_28)

        R_bar = ray_cal["R_bar"]
        theta_mean = ray_cal["theta_mean"]
        Z = ray_cal["Z"]
        p_val = ray_cal["p_value"]
        N = ray_cal["N"]

        # Calculate histogram counts and density for Calendar Epoch Phase
        counts, _ = np.histogram(angles_cal, bins=bin_edges)
        density = counts / (N * bin_width)

        # Plot circular rose histogram
        bars = ax.bar(
            bin_centers,
            density,
            width=bin_width * 0.90,
            bottom=0.0,
            color=p["color"],
            alpha=0.50,
            edgecolor=p["color"],
            linewidth=1.2,
            zorder=2,
        )

        # Uniform null reference circle: density = 1 / (2*pi) ≈ 0.1592
        null_density = 1.0 / (2.0 * np.pi)
        theta_full = np.linspace(0, 2 * np.pi, 300)
        ax.plot(
            theta_full,
            np.full_like(theta_full, null_density),
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            zorder=3,
        )

        # Maximum radial scale: set to comfortable level above peak density
        max_density = max(0.32, float(np.max(density)) * 1.18)
        ax.set_ylim(0.0, max_density)

        # Draw Mean Resultant Vector Arrow R_bar
        # Scale visually by factor relative to density scale so it is prominent
        arrow_scale = max_density * 0.90
        arrow_r = R_bar * arrow_scale
        ax.annotate(
            "",
            xy=(theta_mean, max(0.015, arrow_r)),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#000000",
                lw=2.8,
                mutation_scale=16,
            ),
            zorder=5,
        )
        # Add marker at resultant vector tip for enhanced visibility
        ax.plot(
            [theta_mean],
            [max(0.015, arrow_r)],
            marker="o",
            color="#000000",
            markersize=5,
            zorder=6,
        )

        # Critical R_bar significance threshold at alpha = 0.05: R_crit = sqrt(-ln(0.05) / N)
        R_crit = np.sqrt(-np.log(0.05) / N)
        r_crit_scaled = R_crit * arrow_scale
        ax.plot(
            theta_full,
            np.full_like(theta_full, r_crit_scaled),
            color="#e7298a",
            linestyle=":",
            linewidth=1.4,
            zorder=3,
        )

        # Configure polar angular ticks and labels
        ax.set_theta_zero_location("E")  # 0 rad at 3 o'clock (East)
        ax.set_theta_direction(1)        # Counter-clockwise
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(
            ["$0$", "$\\frac{\\pi}{4}$", "$\\frac{\\pi}{2}$", "$\\frac{3\\pi}{4}$",
             "$\\pm\\pi$", "$-\\frac{3\\pi}{4}$", "$-\\frac{\\pi}{2}$", "$-\\frac{\\pi}{4}$"],
            fontsize=9.5,
            color="#2c3e50",
            fontweight="medium",
        )

        ax.set_rlabel_position(75)
        ax.tick_params(axis="y", labelsize=8, colors="#555555")

        # Titles and Subtitles
        ax.set_title(
            f"{p['title']}\n({p['subtitle']})",
            fontsize=11.5,
            fontweight="bold",
            pad=16,
            color="#111111",
        )

        # Statistical Annotation Box positioned cleanly centered beneath each subplot
        p_cal_str = f"p = {p_val:.3e}" if p_val < 0.001 else f"p = {p_val:.4f}"
        p_28_str = f"p = {ray_28['p_value']:.3e}" if ray_28['p_value'] < 0.001 else f"p = {ray_28['p_value']:.4f}"

        if p_val > 0.05:
            verdict = "Uniformity: NOT REJECTED (p > 0.05)\n(Consistent with Population Desynchrony)"
            verdict_color = "#1b7837"
        elif p_val < 0.01:
            verdict = "Uniformity: REJECTED (p < 0.01)\n(Significant Phase Clustering)"
            verdict_color = "#b2182b"
        else:
            verdict = "Uniformity: Marginal (0.01 <= p < 0.05)"
            verdict_color = "#d95f02"

        stat_text = (
            f"$\\mathbf{{Calendar\\ Epoch\\ Phase\\ (\\Phi_{{cal}})}}$\n"
            f"  Sample $N = {N}$\n"
            f"  Mean Resultant $\\bar{{R}} = {R_bar:.4f}$ (crit: ${R_crit:.3f}$)\n"
            f"  Rayleigh $Z = {Z:.2f}$, {p_cal_str}\n"
            f"  Circular Mean $\\bar{{\\theta}} = {np.rad2deg(theta_mean):+.1f}^\\circ$\n"
            f"───────────────\n"
            f"$\\mathbf{{28\\text{{-}}Day\\ Lunar\\ Phase\\ (\\theta_{{28}})}}$\n"
            f"  $\\bar{{R}} = {ray_28['R_bar']:.4f}$, $Z = {ray_28['Z']:.2f}$, {p_28_str}\n"
            f"───────────────\n"
            f"{verdict}"
        )

        ax.text(
            0.50,
            -0.16,
            stat_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            horizontalalignment="center",
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor="#fbfbfb",
                edgecolor=verdict_color,
                linewidth=1.4,
                alpha=0.96,
            ),
        )

    # Shared Legend Elements across all panels
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#555555", edgecolor="#333333", alpha=0.5, label="Acrophase Density (Rose Bins)"),
        Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.5, label="Uniform Null Density ($1/2\\pi \\approx 0.159$)"),
        Line2D([0], [0], color="#e7298a", linestyle=":", linewidth=1.5, label="Rayleigh $\\alpha = 0.05$ Threshold Radius ($\\bar{R}_{crit}$)"),
        Line2D([0], [0], color="#000000", marker="o", markersize=6, linewidth=2.0, label="Mean Resultant Vector $\\bar{R}$ (Direction $\\bar{\\theta}$)"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.86),
        ncol=4,
        fontsize=9.0,
        frameon=True,
        facecolor="#f8f9fa",
        edgecolor="#cccccc",
    )

    # Global Supertitle
    plt.suptitle(
        "Circular Statistics & Rayleigh Uniformity Analysis of Infradian Acrophases\n"
        "Testing Autonomous Biological Desynchrony versus Exogenous Calendar Entrainment",
        fontsize=14.5,
        fontweight="bold",
        y=0.95,
        color="#111111",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Publication figure saved successfully: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def main() -> int:
    """Main execution pipeline."""
    print("=" * 80)
    print("Milestone 3 — Module 2: Circular Statistics & Acrophase Clustering")
    print(f"Script: {Path(__file__).resolve()}")
    print("=" * 80)

    # Ensure output directories exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load screening metrics
    df = load_infradian_metrics()

    # 2. Extract baseline dates from cache
    customer_ids = df["customer_id"].tolist()
    date_0_map = extract_baseline_dates(customer_ids)

    # 3. Compute circular statistics and Rayleigh tests
    test_results_df = compute_cohort_circular_statistics(df, date_0_map)

    # 4. Save results CSV
    csv_out = RESULTS_DIR / "acrophase_uniformity_tests.csv"
    test_results_df.to_csv(csv_out, index=False)
    print(f"Test results saved to: {csv_out} ({len(test_results_df)} rows)")

    # 5. Generate publication Figure 2
    fig_out = plot_fig2_acrophase_polar_plots(df, date_0_map, test_results_df)

    print("\n" + "=" * 80)
    print("MODULE 2 EXECUTION COMPLETE")
    print(f"  CSV:    {csv_out}")
    print(f"  Figure: {fig_out}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
