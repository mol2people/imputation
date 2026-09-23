#!/usr/bin/env python3
"""Marginal improvement from feature engineering — AUROC plot."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

R = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
SQ = R / "experiments" / "artifacts_sq"
SOURCES = ["s3", "s6", "s7"]
T = 2.0452                     # t_{0.975, 29}
DOT, TUNE, FS, GREY = "#1a1a1a", "#c8102e", "#1565c0", "#888888"
AX_LABEL_SIZE = 15
Y_LABEL_SIZE = 17
WORD_SIZE = 12


def frozen(sub):
    """Frozen-fold SQ run: test AUROC + 2.5/97.5 percentile of 100 bootstrap draws."""
    out = {}
    for sid in SOURCES:
        base = SQ / f"source_{sid[1]}" / sub
        m = json.loads((base / "metrics_sq.json").read_text())["variants"]["all"]
        x = pd.read_csv(base / "bootstrap_test_draws.csv")
        x = x.loc[x["variant"].eq("all"), "auroc"].to_numpy()
        lo, hi = np.percentile(x, [2.5, 97.5])
        out[sid] = (m["test"]["auroc"], float(lo), float(hi))
    return out


def fs_a0():
    """FS-A0 baseline: 30-repeat mean +/- t*sd*sqrt(1+1/30) prediction interval."""
    df = pd.read_csv(SQ / "fsplit" / "fsplit_metrics.csv")
    df = df[df["arm"].eq("A0") & df["variant"].eq("all")]
    out = {}
    for s, g in df.groupby("source"):
        x = g["auroc_test"].to_numpy()
        mu, sd = float(x.mean()), float(x.std(ddof=1))
        hw = T * sd * np.sqrt(1.0 + 1.0 / len(x))
        out[f"s{s}"] = (mu, mu - hw, mu + hw)
    return out


def plot_point(ax, x, y, lo, hi, color=DOT):
    """Dot + thick grey bar with caps."""
    if hi > lo:
        kw = dict(color=GREY, lw=2.5, solid_capstyle="butt", zorder=3, clip_on=False)
        ax.plot([x, x], [lo, hi], **kw)
        for v in (lo, hi):
            ax.plot([x - 0.025, x + 0.025], [v, v], **kw)
    ax.plot([x], [y], "o", mfc=color, mec=color, ms=5.5, ls="none", zorder=5, clip_on=False)


def main():
    m = json.loads((R / "experiments" / "artifacts" / "metrics.json").read_text())
    v1 = (m["test"]["auroc"], m["bootstrap_test"]["auroc_min"], m["bootstrap_test"]["auroc_max"])
    sq_def, sq_tune, fs = frozen(""), frozen("tuned_500"), fs_a0()

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    fig.subplots_adjust(left=0.11, right=0.995, top=0.88, bottom=0.13)
    ax.axvspan(-0.5, 0.5, color="#f2f2f2", zorder=0)
    ax.axvline(0.5, color="#c8c8c8", lw=1.1, zorder=1)

    plot_point(ax, 0, *v1)
    colors = {"def": DOT, "tune": TUNE, "fs": FS}
    for i, sid in enumerate(SOURCES):
        for j, (src, tag) in enumerate([(sq_def, "def"), (sq_tune, "tune"), (fs, "fs")]):
            plot_point(ax, 1 + i + (-0.20, 0, 0.20)[j], *src[sid], color=colors[tag])

    # single-group legend: full-word DEF → TUNE → FS below the bars, above chance;
    # each segment is colored to match its dot variant
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    def _w(s):
        t = ax.text(0, 0, s, fontsize=WORD_SIZE)
        bb = t.get_window_extent(rend)
        c = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        t.remove()
        return float(c[1, 0] - c[0, 0])

    w_def, w_arr, w_tune, w_fs = (_w(s) for s in
                                   ("Default RF", " → ", "Tune", "Select features"))
    total_w = w_def + 2 * w_arr + w_tune + w_fs
    x_left = 1.0 - total_w / 2
    x_def   = x_left + w_def / 2
    x_arr1  = x_def  + (w_def + w_arr) / 2
    x_tune  = x_arr1 + (w_arr + w_tune) / 2
    x_arr2  = x_tune + (w_tune + w_arr) / 2
    x_fs    = x_arr2 + (w_arr + w_fs) / 2

    pad = 0.02
    box_h = 0.024
    bg = FancyBboxPatch((x_left - pad, 0.545 - box_h / 2), total_w + 2 * pad, box_h,
                        boxstyle="round,pad=0.005,rounding_size=0.010",
                        facecolor="white", edgecolor="#222", lw=1.0, alpha=0.9,
                        zorder=2, clip_on=False)
    ax.add_patch(bg)
    for x, label, color in (
        (x_def,  "Default RF",       DOT),
        (x_arr1, "→",                "#333"),
        (x_tune, "Tune",             TUNE),
        (x_arr2, "→",                "#333"),
        (x_fs,   "Select features",  FS),
    ):
        ax.text(x, 0.545, label, ha="center", va="center",
                fontsize=WORD_SIZE, color=color, zorder=3)

    # chance line starts at the Pooled (x=0) reference and extends to the right;
    # the y-tick label gutter on the left is kept clear
    ax.axhline(0.5, xmin=0.0, color="#999", lw=1.5, ls=(0, (4, 3)), zorder=2)
    ax.text(0.0, 0.485, "chance",
            ha="center", va="top", fontsize=WORD_SIZE, color="#999")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["All", "Garmin", "Apple", "Samsung"], fontsize=AX_LABEL_SIZE)
    ax.set(xlim=(-0.40, 3.42), ylim=(0.43, 0.825), ylabel="AUROC")
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

    fig.suptitle("ML approach is marginally beneficial",
                 fontsize=19, fontweight="bold", y=0.965)

    fig.savefig(HERE / "exp_1to5_best.png", dpi=400, facecolor="white")


if __name__ == "__main__":
    main()
