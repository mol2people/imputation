#!/usr/bin/env python3
"""Plot paired repeat-window change and repeatability (balanced90)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "RHR": "#d95f02",
    "HRp10": "#1b9e77",
    "HR_during_sleep": "#7570b3",
}
PRETTY = {
    "RHR": "RHR",
    "HRp10": "HRp10",
    "HR_during_sleep": "Sleep HR",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--protocol", default="balanced90")
    parser.add_argument("--min-n", type=int, default=10)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out = args.out or (run_dir / "figures/fig5_longitudinal_repeatability.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(run_dir / "longitudinal/headline.csv")
    data = data.loc[
        data["protocol"].eq(args.protocol)
        & data["subgroup"].eq("all")
        & data["n"].ge(args.min_n)
    ].copy()
    finite = [
        "mean_Q_annualized_rate", "Q_rate_ci_low", "Q_rate_ci_high",
        "pearson_Q_i", "pearson_Q_i_ci_low", "pearson_Q_i_ci_high",
    ]
    data = data.loc[np.isfinite(data[finite]).all(axis=1)].copy()
    if data.empty:
        raise SystemExit("no longitudinal rows meet the plotting filter")
    data["label"] = data["source_name"] + " " + data["endpoint"].map(
        lambda x: PRETTY.get(str(x), str(x))
    )
    data = data.sort_values(["source", "endpoint"]).reset_index(drop=True)
    y = np.arange(len(data))
    labels = [
        f"{row.label}  (n={int(row.n):,})" for row in data.itertuples(index=False)
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 6.2), sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    ax = axes[0]
    for i, row in data.iterrows():
        ax.errorbar(
            row["mean_Q_annualized_rate"], i,
            xerr=np.array([[row["mean_Q_annualized_rate"] - row["Q_rate_ci_low"]],
                           [row["Q_rate_ci_high"] - row["mean_Q_annualized_rate"]]]),
            fmt="o", color=COLORS.get(row["endpoint"], "#444444"),
            ecolor=COLORS.get(row["endpoint"], "#444444"),
            elinewidth=1.3, capsize=3, markersize=5, zorder=3,
        )
    ax.axvline(0, color="#777777", lw=0.9, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean annualized $\\Delta Q_i$ (per year)", fontsize=10)
    ax.set_title("Paired change", fontsize=11, loc="left")
    ax.grid(axis="x", color="#e5e5e5", lw=0.7, zorder=0)

    ax = axes[1]
    for i, row in data.iterrows():
        ax.errorbar(
            row["pearson_Q_i"], i,
            xerr=np.array([[row["pearson_Q_i"] - row["pearson_Q_i_ci_low"]],
                           [row["pearson_Q_i_ci_high"] - row["pearson_Q_i"]]]),
            fmt="o", color=COLORS.get(row["endpoint"], "#444444"),
            ecolor=COLORS.get(row["endpoint"], "#444444"),
            elinewidth=1.3, capsize=3, markersize=5, zorder=3,
        )
    ax.axvline(0, color="#777777", lw=0.9, ls="--", zorder=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Pearson $r(Q_{i,1}, Q_{i,2})$", fontsize=10)
    ax.set_title("Window-to-window association", fontsize=11, loc="left")
    ax.grid(axis="x", color="#e5e5e5", lw=0.7, zorder=0)

    fig.suptitle(
        "Repeat-window $Q_i$ change and repeatability",
        fontsize=15, x=0.01, ha="left", y=0.98,
    )
    fig.text(
        0.01, 0.925,
        f"{args.protocol}; all participants with age in [20,75) and two converged windows. "
        "Points are means; bars are pointwise 95% participant-bootstrap intervals.",
        fontsize=9, ha="left", color="#444444",
    )
    fig.text(
        0.01, 0.015,
        "Short-interval within-participant associations, not causal aging effects. "
        f"Arms with n < {args.min_n} omitted; see longitudinal/summary.csv.",
        fontsize=8.2, ha="left", color="#444444",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.90))
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
