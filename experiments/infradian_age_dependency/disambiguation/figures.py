#!/usr/bin/env python3
"""Periodogram figures from the disambiguation run.

Four figures, all from stored artefacts (no recomputation of endpoints):

  fig1_excess_spectrum  mean D(f) = P_obs(f) - P_null(f) by stratum, expanded90.
                        Band-integrated D/df is Q_i.
  fig2_coherence        cross-participant Rbar(T), both protocols, with the
                        Rayleigh 95 % line and the lunar period.
  fig3_placebo_ratio    F/M fixed-period amplitude ratio at 18, 27, 28, 29, 34,
                        48, 55, 60 d (check A, band-localisation).
  fig4_coverage         Rbar(T) for HRp10 vs clock coverage, pooled and F20-35
                        (check B).

Usage: python figures.py --run DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent / "artifacts"
ENDPOINTS = ARTIFACTS / "endpoints" / "endpoints.parquet"
SPECTRA = ARTIFACTS / "spectra" / "spectra_broad.npz"

STRATA = ["F20-35", "F36-50", "F51-75", "M20-35", "M36-50", "M51-75"]
COLOR = {"F": "#b03a2e", "M": "#2c5f8a"}
BAND = (25.0, 35.0)
LUNAR = 29.530589
ANOM = 27.554550
HARM12 = 365.25 / 12.0

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def label(sex: int, age: float) -> str:
    band = "20-35" if age <= 35 else ("36-50" if age <= 50 else "51-75")
    return ("F" if sex == 1 else "M") + band


def _panel_grid(sharey=False):
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.2), sharex=True,
                             sharey=sharey, squeeze=True)
    return fig, axes


def _period_axis(ax, xlim=(62, 2.9)):
    ax.set_xscale("log")
    ax.set_xticks([3, 4, 5, 7, 10, 14, 20, 25, 30, 35, 45, 60])
    ax.set_xticklabels(["3", "4", "5", "7", "10", "14", "20", "25", "30", "35", "45", "60"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(*xlim)


def _band_axis(ax, lo=24.8, hi=35.2):
    ax.set_xticks([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])
    ax.set_xlim(lo, hi)


def _band_and_lunar(ax, annotate=False):
    ax.axvspan(*BAND, color="0.90", zorder=0, lw=0)
    ax.axvline(ANOM, color="0.25", lw=0.9, ls=":", zorder=1)
    ax.axvline(LUNAR, color="0.25", lw=0.9, ls="--", zorder=1)


def _first_window(keys: pd.DataFrame, by) -> pd.DataFrame:
    return (keys.sort_values("window_index")
                .groupby(by, as_index=False).head(1))


def fig_excess(run: Path, out: Path) -> None:
    z = np.load(SPECTRA)
    ep = pd.read_parquet(ENDPOINTS, columns=[
        "status", "user_id", "source", "protocol", "window_index", "age", "sex"])
    ep = ep.loc[ep.status.eq("ok")]
    D = z["P_obs"].astype(np.float64) - z["P_null_mean"].astype(np.float64)
    keys = pd.DataFrame({c: z[c] for c in
                         ["user_id", "source", "protocol", "window_index"]})
    keys["row"] = np.arange(len(keys))
    m = keys.merge(ep, on=["user_id", "source", "protocol", "window_index"],
                   how="inner", validate="one_to_one")
    m = _first_window(m, ["user_id", "source", "protocol"])
    m = m.loc[m.age.between(20, 75) & m.protocol.eq("expanded90")].copy()
    m["stratum"] = [label(int(s), float(a)) for s, a in zip(m.sex, m.age)]

    T = 1.0 / z["broad_freqs"]
    o = np.argsort(T)
    T = T[o]
    inband = (T >= BAND[0]) & (T <= BAND[1])

    fig, axes = _panel_grid(sharey="row")
    for j, (st, ax) in enumerate(zip(STRATA, axes.ravel())):
        sel = m.loc[m.stratum.eq(st)]
        n = len(sel)
        X = D[sel.row.to_numpy()][:, o]
        mu = X.mean(0)
        se = X.std(0, ddof=1) / np.sqrt(n)
        ax.fill_between(T, mu - 1.96 * se, mu + 1.96 * se,
                        color=COLOR[st[0]], alpha=0.20, lw=0)
        ax.plot(T, mu, color=COLOR[st[0]], lw=1.0)
        ax.axhline(0.0, color="0.2", lw=0.6)
        _band_and_lunar(ax)
        _period_axis(ax)
        q = float(mu[inband].mean())
        ax.set_title(f"{st}   n={n}   in-band mean D = {q:.3f}")
        if j >= 3:
            ax.set_xlabel("Period (d)")
        if j % 3 == 0:
            ax.set_ylabel("D(f) = P$_{obs}$ - P$_{null}$")
    fig.suptitle("Mean excess periodogram (expanded90); shaded 25-35 d; "
                 "dotted 27.55 d, dashed 29.53 d", y=0.99)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig1_excess_spectrum.{ext}")
    plt.close(fig)


def fig_coherence(run: Path, out: Path) -> None:
    d = pd.read_csv(run / "coherence_spectrum.csv")
    d = d.loc[d.stratum.isin(STRATA)]
    if "expanded90" not in set(d.protocol):
        raise SystemExit("coherence_spectrum.csv lacks expanded90")
    fig, axes = _panel_grid(sharey=True)
    for j, (st, ax) in enumerate(zip(STRATA, axes.ravel())):
        for protocol, ls, lw in (("expanded90", "-", 1.1), ("balanced90", "--", 0.9)):
            g = d.loc[d.stratum.eq(st) & d.protocol.eq(protocol)]
            if g.empty:
                continue
            ax.plot(g.period, g.rbar, ls=ls, lw=lw, color=COLOR[st[0]])
        g = d.loc[d.stratum.eq(st) & d.protocol.eq("expanded90")]
        n = int(g.n.iloc[0])
        ax.axhline(float(np.sqrt(-np.log(0.05) / n)), color="0.35", lw=0.8, ls=":")
        ax.axvline(ANOM, color="0.25", lw=0.9, ls=":")
        ax.axvline(LUNAR, color="#7a1f1f", lw=1.0, ls="--")
        _band_axis(ax)
        ax.set_ylim(0, 0.17)
        ax.set_title(f"{st}   n={n}")
        if j >= 3:
            ax.set_xlabel("Period (d)")
        if j % 3 == 0:
            ax.set_ylabel(r"$\bar{R}(T)$")
    handles = [matplotlib.lines.Line2D([], [], color="0.3", lw=1.1, ls="-",
                                       label="expanded90"),
               matplotlib.lines.Line2D([], [], color="0.3", lw=0.9, ls="--",
                                       label="balanced90"),
               matplotlib.lines.Line2D([], [], color="0.35", lw=0.8, ls=":",
                                       label="95 % null (expanded90)"),
               matplotlib.lines.Line2D([], [], color="0.25", lw=0.9, ls=":",
                                       label="27.55 d (anomalistic)"),
               matplotlib.lines.Line2D([], [], color="#7a1f1f", lw=1.0, ls="--",
                                       label="29.53 d (synodic)")]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.01), fontsize=7.5)
    fig.suptitle("Cross-participant phase coherence of HRp10", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig2_coherence.{ext}")
    plt.close(fig)


def fig_placebo(run: Path, out: Path) -> None:
    d = pd.read_csv(run / "placebo_bands.csv")
    pairs = ["F20-35/M20-35", "F36-50/M36-50"]
    periods = [18.0, 27.0, 28.0, 29.0, 34.0, 48.0, 55.0, 60.0]
    xs = np.arange(len(periods), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), sharey=True)
    for k, (pair, ax) in enumerate(zip(pairs, axes)):
        ax.axvspan(0.5, 4.5, color="0.90", zorder=0, lw=0)
        ax.axhline(1.0, color="0.2", lw=0.8)
        for protocol, mk, dx in (("expanded90", "o", -0.10),
                                 ("balanced90", "s", 0.10)):
            g = d.loc[d.pair.eq(pair) & d.protocol.eq(protocol)].sort_values("period_d")
            if g.empty:
                continue
            if len(g) != len(periods):
                raise SystemExit(f"{pair}/{protocol}: {len(g)} points")
            ax.errorbar(xs + dx, g.ratio_A2.to_numpy(),
                        yerr=[(g.ratio_A2 - g.ratio_A2_lo).to_numpy(),
                              (g.ratio_A2_hi - g.ratio_A2).to_numpy()],
                        fmt=mk, ms=4, lw=1.0, capsize=2.5, color=COLOR[pair[0]],
                        alpha=0.85, label=protocol)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{p:g}" for p in periods])
        ax.set_xlim(-0.6, len(periods) - 0.4)
        ax.set_xlabel("Fixed design period (d)")
        ax.set_title(pair.replace("/", " / "), fontsize=9)
        if k == 0:
            ax.set_ylabel(r"$A^2_F / A^2_M$ (95 % CI)")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Female/male fixed-period amplitude ratio; shaded = in-band points",
                 y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig3_placebo_ratio.{ext}")
    plt.close(fig)


def _rbar_nan(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    th = np.deg2rad(angles)
    r = np.hypot(np.nanmean(np.cos(th), 0), np.nanmean(np.sin(th), 0))
    n = np.isfinite(angles).sum(0)
    return r, n


def fig_coverage(run: Path, out: Path) -> None:
    z = np.load(run / "phase_coherence_expanded90.npz", allow_pickle=True)
    ep = pd.read_parquet(ENDPOINTS, columns=[
        "status", "user_id", "source", "protocol", "window_index", "age", "sex"])
    ep = ep.loc[ep.status.eq("ok")]
    keys = pd.DataFrame({c: z[c] for c in
                         ["user_id", "source", "protocol", "window_index"]})
    keys["row"] = np.arange(len(keys))
    m = keys.merge(ep, on=["user_id", "source", "protocol", "window_index"],
                   how="inner", validate="one_to_one")
    m = _first_window(m, ["user_id", "source"])
    m = m.loc[m.age.between(20, 75)].reset_index(drop=True)
    m["stratum"] = [label(int(s), float(a)) for s, a in zip(m.sex, m.age)]

    periods = z["periods"]
    series = [("HRp10", z["angles"], "#1a1a1a", "-", 1.2),
              ("coverage 00-06", z["ang_cov_night"], "#2c7fb8", "--", 1.0),
              ("coverage all-day", z["ang_cov_total"], "#e08214", "-.", 1.0)]

    panels = [("pooled (20-75)", np.ones(len(m), bool)),
              ("F20-35", m.stratum.eq("F20-35").to_numpy())]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for k, ((name, mask), ax) in enumerate(zip(panels, axes)):
        idx = m.loc[mask, "row"].to_numpy()
        ax.axvline(LUNAR, color="#7a1f1f", lw=1.0, ls="--")
        for lab, arr, col, ls, lw in series:
            r, n = _rbar_nan(arr[idx])
            ax.plot(periods, r, color=col, ls=ls, lw=lw, label=f"{lab} (n={int(n[0])})")
        n0 = len(idx)
        ax.axhline(float(np.sqrt(-np.log(0.05) / n0)), color="0.4", lw=0.8, ls=":")
        ax.set_xlim(25, 35)
        ax.set_xticks([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])
        ax.set_xlabel("Period (d)")
        ax.set_title(name, fontsize=9)
        ax.set_ylim(0, 0.30)
        if k == 0:
            ax.set_ylabel(r"$\bar{R}(T)$")
            ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Coherence of HRp10 vs clock coverage; dotted = 95 % null", y=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig4_coverage.{ext}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    out = args.run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_excess(args.run, out)
    fig_coherence(args.run, out)
    fig_placebo(args.run, out)
    fig_coverage(args.run, out)
    print("wrote", *(str(p.name) for p in sorted(out.glob("*.png"))))


if __name__ == "__main__":
    main()
