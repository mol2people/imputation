#!/usr/bin/env python3
"""Cross-participant phase-coherence spectrum of the 25-35 d band.

For every window the cosinor coefficient at each trial period T has a phase
(the day-of-window at which HR peaks, in calendar-referenced time).  Averaging
those phases across participants gives the mean resultant length Rbar(T).

Interpretation:

- An endogenous rhythm with random phases across people contributes amplitude
  but no phase coherence: Rbar stays at the random-direction null.
- An environmental driver (lunar, seasonal, weekly) imposes a common phase
  reference and produces a peak in Rbar(T) at its own period.
- A systematic artefact of the window geometry or the calendar distribution
  produces broad-band coherence, i.e. many periods exceed the null band.

Under the null Rbar has Rayleigh scale 1/sqrt(2n); the 95 % point is
sqrt(-ln 0.05 / n) = 1.73/sqrt(n).  With 101 grid points, roughly 5 % should
exceed it by chance.  The exceedance fraction is therefore a calibration
statistic for the whole spectrum.

Usage: python coherence.py --run DIR [--protocols balanced90 expanded90]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STRATA = [("F", (20, 35)), ("F", (36, 50)), ("F", (51, 75)),
          ("M", (20, 35)), ("M", (36, 50)), ("M", (51, 75))]


def _label(sex: int, age: float) -> str:
    band = "20-35" if age <= 35 else ("36-50" if age <= 50 else "51-75")
    return ("F" if sex == 1 else "M") + band


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--protocols", nargs="+", default=["balanced90", "expanded90"])
    args = ap.parse_args()

    rows, spec_rows = [], []
    for protocol in args.protocols:
        npz = np.load(args.run / f"phase_coherence_{protocol}.npz", allow_pickle=False)
        rec = pd.read_parquet(args.run / f"phase_records_{protocol}.parquet")
        rec = (rec.sort_values("window_index")
                  .groupby(["user_id", "source"], as_index=False).head(1))
        meta = rec.set_index(["user_id", "source"])[["age", "sex"]]
        angles = npz["angles"]
        keys = pd.DataFrame({"user_id": npz["user_id"], "source": npz["source"],
                             "window_index": npz["window_index"]})
        periods = npz["periods"]
        keys["row"] = np.arange(len(keys))
        # one window per participant: keep only the rows that are the first window
        first = (rec.sort_values("window_index")
                   .groupby(["user_id", "source"], as_index=False).head(1)
                   [["user_id", "source", "window_index"]])
        keep_rows = keys.merge(first, on=["user_id", "source", "window_index"],
                               how="inner").row.to_numpy()
        angles, keys = angles[keep_rows], keys.loc[keep_rows].reset_index(drop=True)
        keys = keys.join(meta, on=["user_id", "source"])
        keep = keys.age.between(20, 75).to_numpy()
        angles, keys = angles[keep], keys.loc[keep].reset_index(drop=True)
        keys["stratum"] = [_label(int(s), float(a))
                           for s, a in zip(keys.sex, keys.age)]

        for stratum in sorted(keys.stratum.unique()):
            m = keys.stratum.eq(stratum).to_numpy()
            a = angles[m]
            n = int(m.sum())
            th = np.deg2rad(a)
            rbar = np.hypot(np.cos(th).mean(0), np.sin(th).mean(0))
            null95 = float(np.sqrt(-np.log(0.05) / n))
            exceed = float(np.mean(rbar > null95))
            k29 = int(np.argmin(np.abs(periods - 29.530589)))
            near = [i for i in range(len(periods)) if 28.0 <= periods[i] <= 31.0]
            rows.append({
                "protocol": protocol, "stratum": stratum, "n": n,
                "rbar_2953": float(rbar[k29]),
                "rbar_band_median": float(np.median(rbar[near])),
                "rbar_max": float(rbar.max()),
                "period_at_max": float(periods[int(rbar.argmax())]),
                "null95": null95, "exceed_frac": exceed,
            })
            spec_rows.append(pd.DataFrame({
                "protocol": protocol, "stratum": stratum, "n": n,
                "period": periods, "rbar": rbar, "null95": null95}))

    summary = pd.DataFrame(rows)
    summary.to_csv(args.run / "coherence_summary.csv", index=False)
    pd.concat(spec_rows, ignore_index=True).to_csv(
        args.run / "coherence_spectrum.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
