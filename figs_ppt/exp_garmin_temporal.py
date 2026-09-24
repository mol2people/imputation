#!/usr/bin/env python3
"""Garmin salutation — MiniRocket vs wear-only vs shuffled MiniRocket."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DOT, GREY = "#1a1a1a", "#888888"
REF = "#555555"
AX_LABEL_SIZE = 15
Y_LABEL_SIZE = 17


def _mean_t_ci(values):
    """Mean + 95% t-CI for small n (dot lands at the bar's visual midpoint)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    mu = float(v.mean())
    sd = float(v.std(ddof=1))
    # 95% two-sided t, df = n - 1
    t_tab = {2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
             6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622}
    t = t_tab.get(n - 1, 2.2622)
    hw = t * sd / np.sqrt(n)
    return mu, mu - hw, mu + hw


def _median_pct_ci(values):
    """Median + [2.5, 97.5] percentile (analog of bootstrap median + percentile CI)."""
    v = np.asarray(values, dtype=float)
    lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
    return float(med), float(lo), float(hi)


def _garmin_refs():
    """Default and best Garmin AUROCs from the previous figure (sq_def, max of sq_tune/fs)."""
    sq = json.loads((R / "experiments" / "artifacts_sq" / "source_3" / "metrics_sq.json").read_text())
    default = sq["variants"]["all"]["test"]["auroc"]
    tuned = json.loads((R / "experiments" / "artifacts_sq" / "source_3" / "tuned_500" / "metrics_sq.json").read_text())["variants"]["all"]["test"]["auroc"]
    fs_df = pd.read_csv(R / "experiments" / "artifacts_sq" / "fsplit" / "fsplit_metrics.csv")
    fs_df = fs_df[(fs_df["arm"] == "A0") & (fs_df["variant"] == "all") & (fs_df["source"] == 3)]
    fs = float(fs_df["auroc_test"].mean())
    return default, max(tuned, fs)


def main():
    wear_df = pd.read_csv(R / "experiments" / "recording_structure_garmin" / "results" / "metrics.csv")
    wear_vals = wear_df.loc[wear_df["arm"] == "C_static", "auroc_test"].to_numpy()
    wear_mu, wear_lo, wear_hi = _mean_t_ci(wear_vals)             # n=3: mean + t-CI

    tmp_df = pd.read_csv(R / "experiments" / "temporal_garmin" / "results" / "temporal_metrics.csv")
    shuf_vals = tmp_df.loc[tmp_df["arm"] == "Shuffled_MR", "auroc_test"].to_numpy()
    mr_vals   = tmp_df.loc[tmp_df["arm"] == "MultiRocket", "auroc_test"].to_numpy()
    shuf_med, shuf_lo, shuf_hi = _median_pct_ci(shuf_vals)        # n=10: median + pct CI
    mr_med,   mr_lo,   mr_hi   = _median_pct_ci(mr_vals)

    default_garmin, best = _garmin_refs()

    points = [
        ("NAs pattern", wear_mu, wear_lo, wear_hi),
        ("Daily HR (true)", mr_med,   mr_lo,   mr_hi),
        ("Daily HR (shuffled)", shuf_med, shuf_lo, shuf_hi),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    fig.subplots_adjust(left=0.11, right=0.995, top=0.88, bottom=0.13)

    # horizontal references: default and best Garmin from the previous figure
    ax.axhline(default_garmin, color=REF, lw=1.4, ls=(0, (2, 2)), zorder=2)
    ax.axhline(best, color=REF, lw=1.4, ls=(0, (2, 2)), zorder=2)

    for i, (label, mu, lo, hi) in enumerate(points):
        ax.plot([i, i], [lo, hi],
                color=GREY, lw=2.5, solid_capstyle="butt",
                zorder=3, clip_on=False)
        ax.plot([i - 0.025, i + 0.025], [lo, lo],
                color=GREY, lw=2.5, solid_capstyle="butt", zorder=3, clip_on=False)
        ax.plot([i - 0.025, i + 0.025], [hi, hi],
                color=GREY, lw=2.5, solid_capstyle="butt", zorder=3, clip_on=False)
        ax.plot([i], [mu], "o", mfc=DOT, mec=DOT, ms=6, ls="none",
                zorder=5, clip_on=False)

    # chance line
    ax.axhline(0.5, color="#999", lw=1.5, ls=(0, (4, 3)), zorder=2)

    ax.set_xticks(range(len(points)))
    ax.set_xticklabels([p[0] for p in points], fontsize=AX_LABEL_SIZE)
    ax.set(xlim=(-0.45, 2.45), ylim=(0.43, 0.91), ylabel="AUROC")
    ax.yaxis.label.set_size(Y_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=AX_LABEL_SIZE)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.10))
    ax.grid(axis="y", color="#e4e4e4", lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    fig.suptitle("Circadian features double RF improvement",
                 fontsize=18, fontweight="bold", y=0.965)

    fig.savefig(HERE / "exp_garmin_temporal.png", dpi=400, facecolor="white")
    print(f"wrote {HERE / 'exp_garmin_temporal.png'}")
    print("\npoints (mean + 95% t-CI for n=3, median + [2.5,97.5] pct for n=10):")
    for label, mu, lo, hi in points:
        print(f"  {label:14s} {mu:.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"\nreference lines: default Garmin = {default_garmin:.4f}, best Garmin = {best:.4f}")


if __name__ == "__main__":
    main()
