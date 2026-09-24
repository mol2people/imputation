#!/usr/bin/env python3
"""Plot AUROC against the number of Garmin days used."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "temporal_garmin" / "results"
OUT_DIRS = (Path(__file__).resolve().parent, RESULTS)
MEDIAN_DAYS = 476
ARMS = {
    "MultiRocket_primary": ("#1a6fae", "Circadian features"),
    "Profile24": ("#2a8f3d", "Aggregated hourly HR"),
}


def main():
    data = pd.read_csv(RESULTS / "days_scaling_all_metrics.csv")
    fig, ax = plt.subplots(figsize=(11.7, 5.8))
    fig.subplots_adjust(left=0.11, right=0.76, top=0.84, bottom=0.13)

    for arm, (color, label) in ARMS.items():
        rows = data[data.arm == arm]
        means = rows.groupby("k").auroc_test.mean()
        sem = rows.groupby("k").auroc_test.sem()
        ci = t.ppf(0.975, 9) * sem
        ks = [k for k in (4, 7, 14, 21, 30, 40, 60, 80) if k in means.index]
        ax.errorbar(ks, means.loc[ks], yerr=ci.loc[ks], fmt="o-", color=color,
                    ms=5.5, lw=2, capsize=3)

        endpoint = means.loc[-1]
        endpoint_ci = ci.loc[-1]
        ax.plot([80, MEDIAN_DAYS], [means.loc[80], endpoint], ":", color=color, lw=1.5)
        ax.errorbar(MEDIAN_DAYS, endpoint, yerr=endpoint_ci, fmt="o", mfc="white",
                    mec=color, color=color, ms=7, lw=1.5, capsize=3)

    ax.axhline(0.7445, color="#777777", lw=1.4, ls="-",
               label="best ML")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Days of data per customer", fontsize=15)
    ax.set_ylabel("AUROC", fontsize=17)
    ax.set_xticks([4, 7, 14, 21, 30, 40, 60, 80, MEDIAN_DAYS])
    ax.set_xticklabels(["4", "7", "14", "21", "30", "40", "60", "80", "all available"])
    ax.set_xlim(right=800)
    ax.set_ylim(0.6, 0.95)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.tick_params(axis="both", labelsize=15)
    ax.tick_params(axis="x", length=6, width=1.4)
    ax.grid(axis="y", color="#e4e4e4", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        Line2D([], [], color=color, marker="o", lw=2, ms=5.5, label=label)
        for color, label in ARMS.values()
    ] + [Line2D([], [], color="#777777", lw=1.4, ls="-", label="best ML")]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=11, frameon=False)

    fig.suptitle("LLMs never considered features beyond summary stats",
                 fontsize=18, fontweight="bold", y=0.97, x=0.435)
    fig.text(0.435, 0.895,
             "n = 3_848 Garmin users",
             ha="center", va="center", fontsize=10.5, color="#555555")

    for directory in OUT_DIRS:
        fig.savefig(directory / "days_scaling_curves.png", dpi=400, facecolor="white")


if __name__ == "__main__":
    main()
