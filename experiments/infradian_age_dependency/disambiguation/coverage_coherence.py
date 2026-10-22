#!/usr/bin/env python3
"""Check B: coherence of the real clock-coverage series.

The 28-d coherent component in HR is a coverage artefact only if the coverage
series itself is coherent at 28 d.  The four daily 6-hour fractions enter the
baseline linearly; HRp10 is a within-day quantile, so a nonlinear or lagged
dependence on coverage is not removed by those four columns.

The RAW coverage series is the right upper bound on the induced coherence:
a memoryless nonlinear map creates harmonics (higher frequencies), not
subharmonics, and a lagged map shifts the phase without changing the coherence
magnitude.  Residualising coverage on the baseline is an identity (the fractions
are columns of it), so no residualised arm is computed.

Decision: coverage-mediated measurement is not supported if Rbar_cov(28) is far
below Rbar_HR(28), equivalently if the implied HR-per-coverage slope is
implausible.

Usage: python coverage_coherence.py --run DIR [--protocols balanced90 expanded90]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SERIES = ["ang_cov_night", "ang_cov_total"]


def _label(sex: int, age: float) -> str:
    band = "20-35" if age <= 35 else ("36-50" if age <= 50 else "51-75")
    return ("F" if sex == 1 else "M") + band


def _rbar(a: np.ndarray) -> np.ndarray:
    th = np.deg2rad(a)
    return np.hypot(np.cos(th).mean(0), np.sin(th).mean(0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--protocols", nargs="+", default=["balanced90", "expanded90"])
    args = ap.parse_args()

    rows = []
    for protocol in args.protocols:
        npz = np.load(args.run / f"phase_coherence_{protocol}.npz", allow_pickle=False)
        rec = pd.read_parquet(args.run / f"phase_records_{protocol}.parquet")
        rec = rec.sort_values("window_index").groupby(
            ["user_id", "source"], as_index=False).head(1)
        rec = rec[rec.age.between(20, 75)].reset_index(drop=True)
        meta = rec.set_index(["user_id", "source"])[["age", "sex"]]
        keys = pd.DataFrame({"user_id": npz["user_id"], "source": npz["source"],
                             "window_index": npz["window_index"]})
        periods = npz["periods"]
        keys["row"] = np.arange(len(keys))
        keep = keys.merge(rec[["user_id", "source", "window_index"]],
                          on=["user_id", "source", "window_index"],
                          how="inner").row.to_numpy()
        keys = keys.loc[keep].reset_index(drop=True)
        keys = keys.join(meta, on=["user_id", "source"])
        keys = keys[keys.age.notna()].reset_index(drop=True)
        keys["stratum"] = [_label(int(s), float(a)) for s, a in zip(keys.sex, keys.age)]
        ridx = keys.row.to_numpy()

        # HR reference: coherent amplitude = Rbar(28) * mean cosinor amplitude at 28 d
        hr = keys[["user_id", "source", "stratum"]].merge(
            rec[["user_id", "source", "sham28_amp"]],
            on=["user_id", "source"], how="left")

        k28 = int(np.argmin(np.abs(periods - 28.0)))
        k34 = int(np.argmin(np.abs(periods - 34.0)))
        hr_amp28 = hr["sham28_amp"].to_numpy(float)

        strata = ["pooled"] + sorted(keys.stratum.unique())
        for series in SERIES:
            a_all = npz[series][ridx]
            m_all = npz[series.replace("ang_", "amp_")][ridx]
            for stratum in strata:
                m = np.ones(len(keys), bool) if stratum == "pooled" \
                    else keys.stratum.eq(stratum).to_numpy()
                a, amp = a_all[m], m_all[m]
                valid = np.isfinite(a).all(axis=1) & np.isfinite(amp).all(axis=1)
                if valid.sum() < 30:
                    continue
                rbar = _rbar(a[valid])
                n = int(valid.sum())
                hr_coh = float(_rbar(npz["angles"][ridx][m])[k28]
                               * np.nanmean(hr_amp28[m]))
                cov_coh = float(rbar[k28] * np.nanmean(amp[valid, k28]))
                rows.append({
                    "protocol": protocol, "series": series, "stratum": stratum,
                    "n": n,
                    "rbar_28": float(rbar[k28]), "rbar_34": float(rbar[k34]),
                    "rbar_max": float(rbar.max()),
                    "period_at_max": float(periods[int(rbar.argmax())]),
                    "null95_28": float(np.sqrt(-np.log(0.05) / n)),
                    "cov_amp_28": float(np.nanmean(amp[valid, k28])),
                    "hr_coh_amp_28": hr_coh,
                    "cov_coh_amp_28": cov_coh,
                    "beta_implied": hr_coh / cov_coh if cov_coh > 0 else np.nan,
                })

    res = pd.DataFrame(rows)
    res.to_csv(args.run / "coverage_coherence.csv", index=False)
    cols = ["protocol", "series", "stratum", "n", "rbar_28", "rbar_34",
            "rbar_max", "period_at_max", "cov_amp_28", "hr_coh_amp_28",
            "cov_coh_amp_28", "beta_implied"]
    print(res[res.stratum.eq("pooled")][cols].round(4).to_string(index=False))
    print()
    print("expanded90, per stratum at 28.0 d")
    print(res[(res.protocol == "expanded90") & ~res.stratum.eq("pooled")]
          [["series", "stratum", "n", "rbar_28", "null95_28",
            "cov_coh_amp_28", "beta_implied"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
