#!/usr/bin/env python3
"""
Longitudinal Sliding-Window Spectrograms & Infradian Period Stability Analysis
Path: experiments/female_infradian_extensions/scripts/01_sliding_window_stability.py

Authoritative Implementation for Module 1 of Female Infradian Rhythm Extensions.

Key Objectives:
1. Screen longitudinal cohort from `experiments/female_infradian_cycles/results/infradian_metrics_by_participant.csv`:
   - Filters: screened_pass == True and timespan_days >= 365.
   - Young Females (N = 324, reproductive age target cohort).
   - Young Males (N = 819 / 769, age-matched behavioral negative control).
   - Elderly Females (N = 180 / 244, post-menopausal biological negative control).
2. For each participant:
   - Extract daily resting HR proxy (hr_p10) and daily wear occupancy (occ_daily).
   - Slide window duration W = 90 days with step S = 15 days.
   - Require >= 45 adequate wear days per window.
   - Compute normalized Lomb-Scargle spectral power across T in [14, 56] days (dt = 0.5 days).
   - Extract peak period trajectory {T*_1, ..., T*_M}.
   - Quantify period drift Delta T_drift = IQR(T*) and persistence ratio (fraction in [25, 35] days).
   - Repeat for occ_daily as behavioral compliance control.
3. Outputs:
   - `experiments/female_infradian_extensions/results/sliding_window_stability.csv`
   - `experiments/female_infradian_extensions/results/sliding_window_stability_group_summary.csv`
   - `experiments/female_infradian_extensions/figures/fig1_spectrogram_longitudinal_cases.png` (300 dpi)
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Module Setup & Imports from 00_verify_extension_methods.py
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
EXTENSIONS_DIR = SCRIPT_DIR.parent

# Dynamically import verified modular library
verify_script = SCRIPT_DIR / "00_verify_extension_methods.py"
if not verify_script.exists():
    raise FileNotFoundError(f"Missing verification library at {verify_script}")

spec = importlib.util.spec_from_file_location("ext_methods", verify_script)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load spec from {verify_script}")
ext_methods = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext_methods)

sliding_window_lomb_scargle = ext_methods.sliding_window_lomb_scargle
detrend_signal = ext_methods.detrend_signal

CACHE_DIR = PROJECT_ROOT / "experiments/apple_cohort_scaling/artifacts/cache_5min"
PARTICIPANT_METRICS_PATH = (
    PROJECT_ROOT / "experiments/female_infradian_cycles/results/infradian_metrics_by_participant.csv"
)
RESULTS_DIR = EXTENSIONS_DIR / "results"
FIGURES_DIR = EXTENSIONS_DIR / "figures"


# -----------------------------------------------------------------------------
# Cohort Screening
# -----------------------------------------------------------------------------
def load_and_screen_cohort(metrics_path: Path) -> pd.DataFrame:
    """Loads participant-level metrics and screens the longitudinal cohort."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Infradian metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    required_cols = [
        "customer_id",
        "sex",
        "age_group",
        "bmi_group",
        "timespan_days",
        "screened_pass",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in {metrics_path}")

    # Primary screening: screened_pass == True and timespan_days >= 365
    screened = df[(df["screened_pass"] == True) & (df["timespan_days"] >= 365)].copy()

    # Stratified target and control cohorts
    yf = screened[(screened["sex"] == "Female") & (screened["age_group"] == "Young Adults")]
    ym = screened[(screened["sex"] == "Male") & (screened["age_group"] == "Young Adults")]
    ef = screened[(screened["sex"] == "Female") & (screened["age_group"] == "Elderly")]

    cohort_mask = (
        ((screened["sex"] == "Female") & (screened["age_group"] == "Young Adults"))
        | ((screened["sex"] == "Male") & (screened["age_group"] == "Young Adults"))
        | ((screened["sex"] == "Female") & (screened["age_group"] == "Elderly"))
    )
    final_cohort = screened[cohort_mask].copy()

    print(f"Screened Longitudinal Cohort Summary (timespan >= 365 days):")
    print(f"  - Young Females (Reproductive Target):    N = {len(yf):>4d}")
    print(f"  - Young Males (Age-Matched Control):      N = {len(ym):>4d}")
    print(f"  - Elderly Females (Post-Menopausal Ctrl): N = {len(ef):>4d}")
    print(f"  - Total Evaluated Participants:           N = {len(final_cohort):>4d}")

    return final_cohort


# -----------------------------------------------------------------------------
# Single Participant Processing
# -----------------------------------------------------------------------------
def process_single_participant(
    customer_id: int,
    cache_dir: Path,
    window_size: float = 90.0,
    step_size: float = 15.0,
    min_days: int = 45,
    period_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Loads 5-min cache for a participant, extracts signals, and computes sliding-window metrics."""
    npz_path = cache_dir / f"customer_{customer_id}.npz"
    if not npz_path.exists():
        return {"success": False, "error": f"Missing cache file: {npz_path}"}

    try:
        data = np.load(npz_path)
        hr = data["hr"]
        dates = data["date"]
    except Exception as exc:
        return {"success": False, "error": f"Error loading {npz_path}: {exc}"}

    # Adequate wear days: >= 120 valid 5-min bins (>= 10 hours)
    valid_bins_per_day = np.isfinite(hr).sum(axis=1)
    adeq = valid_bins_per_day >= 120
    n_adequate = int(np.sum(adeq))

    if n_adequate < min_days:
        return {
            "success": True,
            "n_total_windows": 0,
            "n_valid_windows": 0,
            "hr_p10_mean_period": np.nan,
            "hr_p10_median_period": np.nan,
            "hr_p10_period_iqr_drift": np.nan,
            "hr_p10_persistence_ratio": 0.0,
            "hr_p10_mean_power": np.nan,
            "occ_mean_period": np.nan,
            "occ_period_iqr_drift": np.nan,
            "occ_persistence_ratio": 0.0,
            "res_hr": None,
            "res_occ": None,
        }

    hr_adeq = hr[adeq]
    dates_adeq = dates[adeq]

    # Daily signals
    hr_p10 = np.nanpercentile(hr_adeq, 10, axis=1)
    occ_daily = valid_bins_per_day[adeq] / 288.0
    t_d = (dates_adeq - dates[0]).astype(np.float64)

    # Sliding-window Lomb-Scargle for resting HR proxy
    res_hr = sliding_window_lomb_scargle(
        t=t_d,
        y=hr_p10,
        window_size=window_size,
        step_size=step_size,
        min_days=min_days,
        period_grid=period_grid,
    )

    # Sliding-window Lomb-Scargle for occupancy control
    res_occ = sliding_window_lomb_scargle(
        t=t_d,
        y=occ_daily,
        window_size=window_size,
        step_size=step_size,
        min_days=min_days,
        period_grid=period_grid,
    )

    n_valid_hr = res_hr["n_valid_windows"]
    if n_valid_hr > 0:
        valid_periods_hr = res_hr["peak_periods"][res_hr["valid_mask"]]
        valid_powers_hr = res_hr["peak_powers"][res_hr["valid_mask"]]
        hr_p10_mean_period = float(np.mean(valid_periods_hr))
        hr_p10_median_period = float(np.median(valid_periods_hr))
        hr_p10_mean_power = float(np.mean(valid_powers_hr))
    else:
        hr_p10_mean_period = np.nan
        hr_p10_median_period = np.nan
        hr_p10_mean_power = np.nan

    n_valid_occ = res_occ["n_valid_windows"]
    if n_valid_occ > 0:
        valid_periods_occ = res_occ["peak_periods"][res_occ["valid_mask"]]
        occ_mean_period = float(np.mean(valid_periods_occ))
    else:
        occ_mean_period = np.nan

    return {
        "success": True,
        "n_total_windows": res_hr["n_total_windows"],
        "n_valid_windows": n_valid_hr,
        "hr_p10_mean_period": hr_p10_mean_period,
        "hr_p10_median_period": hr_p10_median_period,
        "hr_p10_period_iqr_drift": res_hr["period_drift"],
        "hr_p10_persistence_ratio": res_hr["persistence_ratio"],
        "hr_p10_mean_power": hr_p10_mean_power,
        "occ_mean_period": occ_mean_period,
        "occ_period_iqr_drift": res_occ["period_drift"],
        "occ_persistence_ratio": res_occ["persistence_ratio"],
        "res_hr": res_hr,
        "res_occ": res_occ,
        "t_d": t_d,
        "dates_adeq": dates_adeq,
    }


# -----------------------------------------------------------------------------
# Main Analysis Pipeline
# -----------------------------------------------------------------------------
def run_stability_analysis() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Runs the full cohort sliding-window stability analysis and generates outputs."""
    t_start = time.time()
    cohort_df = load_and_screen_cohort(PARTICIPANT_METRICS_PATH)
    period_grid = np.arange(14.0, 56.5, 0.5, dtype=np.float64)

    results_records: List[Dict[str, Any]] = []
    case_store: Dict[int, Dict[str, Any]] = {}

    target_case_cids = {1059367, 1052359, 956894, 84783, 1056012, 1011450, 1025441}

    print(f"\nProcessing {len(cohort_df)} participants with sliding-window analysis...")
    for idx, (_, row) in enumerate(cohort_df.iterrows()):
        cid = int(row["customer_id"])
        res = process_single_participant(
            customer_id=cid,
            cache_dir=CACHE_DIR,
            window_size=90.0,
            step_size=15.0,
            min_days=45,
            period_grid=period_grid,
        )

        if not res["success"]:
            print(f"Warning: Failed processing for cid={cid}: {res.get('error')}")
            continue

        record = {
            "customer_id": cid,
            "sex": row["sex"],
            "age_group": row["age_group"],
            "bmi_group": row["bmi_group"],
            "timespan_days": int(row["timespan_days"]),
            "n_total_windows": res["n_total_windows"],
            "n_valid_windows": res["n_valid_windows"],
            "hr_p10_mean_period": res["hr_p10_mean_period"],
            "hr_p10_median_period": res["hr_p10_median_period"],
            "hr_p10_period_iqr_drift": res["hr_p10_period_iqr_drift"],
            "hr_p10_persistence_ratio": res["hr_p10_persistence_ratio"],
            "hr_p10_mean_power": res["hr_p10_mean_power"],
            "occ_mean_period": res["occ_mean_period"],
            "occ_period_iqr_drift": res["occ_period_iqr_drift"],
            "occ_persistence_ratio": res["occ_persistence_ratio"],
        }
        results_records.append(record)

        if cid in target_case_cids:
            case_store[cid] = {
                "metadata": record,
                "res_hr": res["res_hr"],
                "res_occ": res["res_occ"],
            }

        if (idx + 1) % 250 == 0 or (idx + 1) == len(cohort_df):
            print(f"  Processed {idx + 1:>4d}/{len(cohort_df)} participants ({(idx + 1)/len(cohort_df)*100:.1f}%)")

    stability_df = pd.DataFrame(results_records)

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_out_path = RESULTS_DIR / "sliding_window_stability.csv"
    stability_df.to_csv(csv_out_path, index=False)
    print(f"\nSaved participant stability metrics to: {csv_out_path}")

    # Compute group-level summaries
    group_summaries = []
    cohort_definitions = [
        ("Young Females (Target)", "Female", "Young Adults"),
        ("Young Males (Control 1)", "Male", "Young Adults"),
        ("Elderly Females (Control 2)", "Female", "Elderly"),
    ]

    for label, sex_val, age_val in cohort_definitions:
        sub = stability_df[(stability_df["sex"] == sex_val) & (stability_df["age_group"] == age_val)]
        # Participants with at least one valid sliding window
        valid_sub = sub[sub["n_valid_windows"] > 0]
        # Robust participants with >= 5 valid windows
        multi_sub = sub[sub["n_valid_windows"] >= 5]

        group_summaries.append({
            "cohort": label,
            "sex": sex_val,
            "age_group": age_val,
            "n_screened": len(sub),
            "n_with_valid_windows": len(valid_sub),
            "n_with_ge_5_windows": len(multi_sub),
            "pct_with_valid_windows": len(valid_sub) / len(sub) * 100.0 if len(sub) > 0 else 0.0,
            # HR p10 drift and persistence
            "hr_drift_mean": float(valid_sub["hr_p10_period_iqr_drift"].mean()),
            "hr_drift_median": float(valid_sub["hr_p10_period_iqr_drift"].median()),
            "hr_persistence_mean": float(valid_sub["hr_p10_persistence_ratio"].mean()),
            "hr_persistence_median": float(valid_sub["hr_p10_persistence_ratio"].median()),
            "hr_power_mean": float(valid_sub["hr_p10_mean_power"].mean()),
            "hr_median_period_mean": float(valid_sub["hr_p10_median_period"].mean()),
            # Multi-window (>=5) drift and persistence
            "hr_drift_median_ge5": float(multi_sub["hr_p10_period_iqr_drift"].median()) if len(multi_sub) > 0 else np.nan,
            "hr_persistence_mean_ge5": float(multi_sub["hr_p10_persistence_ratio"].mean()) if len(multi_sub) > 0 else np.nan,
            # Occupancy control
            "occ_drift_mean": float(valid_sub["occ_period_iqr_drift"].mean()),
            "occ_drift_median": float(valid_sub["occ_period_iqr_drift"].median()),
            "occ_persistence_mean": float(valid_sub["occ_persistence_ratio"].mean()),
            "occ_persistence_median": float(valid_sub["occ_persistence_ratio"].median()),
        })

    summary_df = pd.DataFrame(group_summaries)
    summary_out_path = RESULTS_DIR / "sliding_window_stability_group_summary.csv"
    summary_df.to_csv(summary_out_path, index=False)
    print(f"Saved group summary metrics to: {summary_out_path}")

    print("\n" + "=" * 80)
    print("GROUP-LEVEL SLIDING WINDOW STABILITY CONTRASTS")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"Cohort: {row['cohort']}")
        print(f"  N Screened: {row['n_screened']}, Valid Windows: {row['n_with_valid_windows']} ({row['pct_with_valid_windows']:.1f}%)")
        print(f"  HR p10 Period Drift (IQR): Mean = {row['hr_drift_mean']:.2f} d, Median = {row['hr_drift_median']:.2f} d")
        print(f"  HR p10 Persistence Ratio:  Mean = {row['hr_persistence_mean']*100:.1f}%, Median = {row['hr_persistence_median']*100:.1f}%")
        print(f"  Occupancy Persistence:     Mean = {row['occ_persistence_mean']*100:.1f}%, Median = {row['occ_persistence_median']*100:.1f}%")
        print("-" * 80)

    elapsed = time.time() - t_start
    print(f"Total analysis execution time: {elapsed:.2f}s")

    return stability_df, summary_df, case_store


# -----------------------------------------------------------------------------
# Publication Figure 1 Generation
# -----------------------------------------------------------------------------
def plot_figure_1(
    case_store: Dict[int, Dict[str, Any]],
    output_path: Path,
    stable_cid: int = 1059367,
    drifting_cid: int = 1052359,
    control_cid: int = 1056012,
) -> None:
    """Generates Figure 1: 3-panel publication spectrogram showing stable vs. drifting vs. control cases."""
    print(f"\nGenerating publication Figure 1: {output_path}...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Verify all selected cases are available
    for name, cid in [("Stable Case", stable_cid), ("Drifting Case", drifting_cid), ("Control Case", control_cid)]:
        if cid not in case_store or case_store[cid]["res_hr"] is None:
            raise KeyError(f"Case data for {name} (cid={cid}) not found in case_store")

    cases_to_plot = [
        {
            "panel": "A",
            "title_tag": "Stable Infradian Cycle Case",
            "subtitle": "Young Female (Autonomous ~28-Day Menstrual Pacemaker Across Multi-Year Record)",
            "cid": stable_cid,
            "case_data": case_store[stable_cid],
        },
        {
            "panel": "B",
            "title_tag": "Drifting / Irregular Infradian Cycle Case",
            "subtitle": "Young Female (Fluctuating Infradian Periodicity Across Multi-Year Record)",
            "cid": drifting_cid,
            "case_data": case_store[drifting_cid],
        },
        {
            "panel": "C",
            "title_tag": "Negative Control Case",
            "subtitle": "Young Male Control (Absence of Endogenous Menstrual Pacemaker)",
            "cid": control_cid,
            "case_data": case_store[control_cid],
        },
    ]

    # Global plotting aesthetics
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#222222"
    plt.rcParams["axes.linewidth"] = 0.9

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12.0, 11.0),
        dpi=300,
        constrained_layout=True,
    )

    period_grid = np.arange(14.0, 56.5, 0.5, dtype=np.float64)

    for i, item in enumerate(cases_to_plot):
        ax = axes[i]
        meta = item["case_data"]["metadata"]
        res = item["case_data"]["res_hr"]

        centers = res["window_centers"]
        spectrogram = res["spectrogram"]  # shape (n_windows, 85)
        valid_mask = res["valid_mask"]
        peak_periods = res["peak_periods"]

        # Ensure spectrogram dimensions align with pcolormesh
        # Centers x: n_windows, periods y: 85
        # Construct bin edges for pcolormesh
        step = 15.0
        p_step = 0.5
        x_edges = np.append(centers - step / 2.0, centers[-1] + step / 2.0)
        y_edges = np.append(period_grid - p_step / 2.0, period_grid[-1] + p_step / 2.0)

        # Plot spectrogram power
        # For non-valid windows, power is zero or nan
        display_spec = spectrogram.copy()
        display_spec[~valid_mask, :] = np.nan

        # Colormap: 'viridis' with clean contrast
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            display_spec.T,
            cmap="viridis",
            shading="flat",
            vmin=0.0,
            vmax=max(0.40, float(np.nanmax(display_spec))),
        )

        # Infradian target band (25–35 days)
        ax.axhline(25.0, color="#FFD700", linestyle="--", linewidth=1.4, alpha=0.9, zorder=4)
        ax.axhline(35.0, color="#FFD700", linestyle="--", linewidth=1.4, alpha=0.9, zorder=4)
        ax.axhspan(25.0, 35.0, color="#FFD700", alpha=0.07, zorder=1)

        # Overlay peak period trajectory T*_w
        valid_centers = centers[valid_mask]
        valid_peaks = peak_periods[valid_mask]
        ax.plot(
            valid_centers,
            valid_peaks,
            color="#FFFFFF",
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            markerfacecolor="#00FFFF",
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.8,
            label=r"Peak Period Trajectory $T^*_w$",
            zorder=5,
        )

        # Colorbar for each panel
        cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.015, aspect=18)
        cbar.set_label("Lomb-Scargle Power", fontsize=8.5, labelpad=5)
        cbar.ax.tick_params(labelsize=8)

        # Annotations text box
        drift_val = meta["hr_p10_period_iqr_drift"]
        pers_val = meta["hr_p10_persistence_ratio"] * 100.0
        n_val = meta["n_valid_windows"]
        t_span = meta["timespan_days"]
        sex_lbl = meta["sex"]
        age_lbl = meta["age_group"]

        annot_text = (
            f"Customer ID: {item['cid']} | {sex_lbl}, {age_lbl}\n"
            f"Timespan: {t_span} days | Valid Windows: {n_val}\n"
            f"Period Drift: $\\Delta T_{{\\mathrm{{drift}}}} = {drift_val:.2f}$ days\n"
            f"Persistence Ratio (25–35d): {pers_val:.1f}%\n"
            f"Mean Spectral Power: {meta['hr_p10_mean_power']:.3f}"
        )

        ax.text(
            0.015,
            0.95,
            annot_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#111111", alpha=0.75, edgecolor="#DDDDDD", linewidth=0.8),
            color="#FFFFFF",
            zorder=6,
        )

        # Target band label on right edge
        ax.text(
            x_edges[-1] - 5,
            30.0,
            "Target Infradian Band\n(25–35 days)",
            color="#FFD700",
            fontsize=7.5,
            fontweight="bold",
            horizontalalignment="right",
            verticalalignment="center",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#222222", alpha=0.6, edgecolor="none"),
            zorder=6,
        )

        # Title & Panel Labels
        ax.set_title(
            f"Panel {item['panel']}: {item['title_tag']} — {item['subtitle']}",
            fontsize=10.5,
            fontweight="bold",
            pad=7,
            loc="left",
        )
        ax.set_ylabel("Cycle Period $T$ (days)", fontsize=9.5)
        ax.set_ylim(13.0, 57.0)
        ax.set_yticks([14, 20, 25, 28, 35, 42, 50, 56])
        ax.tick_params(labelsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.35, color="#888888")

        if i == len(cases_to_plot) - 1:
            ax.set_xlabel("Longitudinal Calendar History (Days from First Observation)", fontsize=9.5)
        else:
            ax.set_xlabel("")

        if i == 0:
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85, facecolor="#FFFFFF")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Publication Figure 1 successfully saved to {output_path} (300 DPI).")


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    """Executes the complete sliding-window stability pipeline and produces all deliverables."""
    print("=" * 80)
    print("Female Infradian Rhythm Extensions — Module 1: Sliding Window Stability")
    print("Path: experiments/female_infradian_extensions/scripts/01_sliding_window_stability.py")
    print("=" * 80)

    # 1. Run analysis
    stability_df, summary_df, case_store = run_stability_analysis()

    # 2. Generate Figure 1
    fig1_path = FIGURES_DIR / "fig1_spectrogram_longitudinal_cases.png"
    plot_figure_1(
        case_store=case_store,
        output_path=fig1_path,
        stable_cid=1059367,
        drifting_cid=1052359,
        control_cid=1056012,
    )

    print("\n" + "=" * 80)
    print("MODULE 1 EXECUTION COMPLETE")
    print(f"  Deliverable 1: {RESULTS_DIR / 'sliding_window_stability.csv'}")
    print(f"  Deliverable 2: {RESULTS_DIR / 'sliding_window_stability_group_summary.csv'}")
    print(f"  Deliverable 3: {fig1_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
