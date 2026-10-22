#!/usr/bin/env python3
"""Check A: are the young-female amplitude/coherence excesses band-localised?

The in-band excess (25-35 d) is measured against a per-participant AR(1) null, so a
*global* difference in HR variability between young women and young men would not
create it.  What it would do is lift the fixed-period amplitudes at every period.
This script therefore compares the F/M amplitude ratio inside the band with the same
ratio at out-of-band periods (18, 48, 55, 60 d), all >= 1.3 spectral-kernel widths
from the band edges.

Statistic: RE(T) = [A2_F / A2_M](T) / [A2_F / A2_M](28 d), with a participant-level
paired bootstrap.  Under a global scaling RE(T) = 1; under a band-localised excess
RE(T) < 1 out of band.

Usage: python placebo_bands.py --run DIR [--protocols balanced90 expanded90]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PERIODS = [("sham18", 18.0), ("sham27", 27.0), ("sham28", 28.0), ("sham29", 29.0),
           ("sham34", 34.0), ("sham48", 48.0), ("sham55", 55.0), ("sham60", 60.0)]
PAIRS = [("F20-35", "M20-35"), ("F36-50", "M36-50")]
REF = "sham28"


def _label(sex: int, age: float) -> str:
    band = "20-35" if age <= 35 else ("36-50" if age <= 50 else "51-75")
    return ("F" if sex == 1 else "M") + band


def _ratio_boot(f: np.ndarray, m: np.ndarray, periods: list[str],
                n_boot: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Paired bootstrap of the per-period F/M ratio; returns (ratio, RE)."""
    ratio = np.empty((n_boot, len(periods)))
    for b in range(n_boot):
        fx = f.iloc[rng.integers(0, len(f), len(f))]
        mx = m.iloc[rng.integers(0, len(m), len(m))]
        for k, tag in enumerate(periods):
            ratio[b, k] = fx[f"A2_{tag}"].mean() / mx[f"A2_{tag}"].mean()
    ref = ratio[:, periods.index(REF)]
    return ratio, ratio / ref[:, None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--protocols", nargs="+", default=["balanced90", "expanded90"])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(20261020)

    frames = []
    for p in args.protocols:
        f = args.run / f"phase_records_{p}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("window_index").groupby(
        ["user_id", "source", "protocol"], as_index=False).head(1)
    df = df[df.age.between(20, 75)].reset_index(drop=True)
    df["stratum"] = [_label(int(s), float(a)) for s, a in zip(df.sex, df.age)]
    for tag, _ in PERIODS:
        df[f"A2_{tag}"] = df[f"{tag}_amp"].astype(float) ** 2

    rows = []
    for protocol in args.protocols:
        sub = df.loc[df.protocol.eq(protocol)]
        for sf, sm in PAIRS:
            f = sub.loc[sub.stratum.eq(sf)].reset_index(drop=True)
            m = sub.loc[sub.stratum.eq(sm)].reset_index(drop=True)
            if len(f) < 30 or len(m) < 30:
                continue
            tags = [t for t, _ in PERIODS]
            ratio, re = _ratio_boot(f, m, tags, args.n_boot, rng)
            for k, (tag, period) in enumerate(PERIODS):
                rows.append({
                    "protocol": protocol, "pair": f"{sf}/{sm}", "tag": tag,
                    "period_d": period, "n_f": len(f), "n_m": len(m),
                    "A2_f": float(f[f"A2_{tag}"].mean()),
                    "A2_m": float(m[f"A2_{tag}"].mean()),
                    "ratio_A2": float(f[f"A2_{tag}"].mean() / m[f"A2_{tag}"].mean()),
                    "ratio_A2_lo": float(np.quantile(ratio[:, k], 0.025)),
                    "ratio_A2_hi": float(np.quantile(ratio[:, k], 0.975)),
                    "R2_f": float(f[f"{tag}_R2"].mean()),
                    "R2_m": float(m[f"{tag}_R2"].mean()),
                    "ratio_R2": float(f[f"{tag}_R2"].mean() / m[f"{tag}_R2"].mean()),
                    "RE": float(np.mean(re[:, k])),
                    "RE_lo": float(np.quantile(re[:, k], 0.025)),
                    "RE_hi": float(np.quantile(re[:, k], 0.975)),
                })

    res = pd.DataFrame(rows)
    res.to_csv(args.run / "placebo_bands.csv", index=False)
    print(res[["protocol", "pair", "period_d", "ratio_A2", "ratio_A2_lo",
               "ratio_A2_hi", "ratio_R2", "RE", "RE_lo", "RE_hi"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
