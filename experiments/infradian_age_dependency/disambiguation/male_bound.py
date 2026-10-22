#!/usr/bin/env python3
"""Male-cohort bound on the lunar-attributable share of the 25-35 d HR excess.

Q_i is a band excess measured against a calibrated per-subject null, so a
sex-independent lunar component would appear in men at the same absolute
magnitude as in women.  Under the no-sex-by-lunar-interaction assumption,

    lunar-attributable share of the young-female excess  <=  Q_male / Q_young_female

This script computes that ratio from the saved HRp10 endpoints for both devices
and protocols, with a participant-level bootstrap CI.  No new model fits.

Usage: python male_bound.py [--out FILE]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ENDPOINTS = HERE.parent / "artifacts/endpoints/endpoints.parquet"
STRATA = {
    "F20-35": lambda d: (d.sex == 1) & d.age.between(20, 35),
    "F36-50": lambda d: (d.sex == 1) & d.age.between(36, 50),
    "F51-75": lambda d: (d.sex == 1) & d.age.between(51, 75),
    "M20-35": lambda d: (d.sex == 0) & d.age.between(20, 35),
    "M36-50": lambda d: (d.sex == 0) & d.age.between(36, 50),
    "M51-75": lambda d: (d.sex == 0) & d.age.between(51, 75),
}


def _boot_mean_ratio(a: np.ndarray, b: np.ndarray, n_boot: int = 4000,
                     seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = float(np.mean(a) / np.mean(b))
    draws = np.empty(n_boot)
    for k in range(n_boot):
        draws[k] = np.mean(rng.choice(a, len(a), replace=True)) / np.mean(
            rng.choice(b, len(b), replace=True))
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "male_bound.csv")
    ap.add_argument("--endpoints", type=Path, default=ENDPOINTS)
    args = ap.parse_args()

    df = pd.read_parquet(args.endpoints)
    df = df.loc[(df.status == "ok") & df.source.isin([3, 6])].copy()
    # one window per participant keeps the bootstrap independent
    df = df.sort_values("window_index").groupby(["user_id", "source", "protocol"],
                                                as_index=False).head(1)

    rows = []
    for (protocol, source), g in df.groupby(["protocol", "source"]):
        for name, fn in STRATA.items():
            s = g.loc[fn(g)]
            if len(s) < 30:
                continue
            rows.append({"protocol": protocol, "source": {6: "apple", 3: "garmin"}[source],
                         "stratum": name, "n": len(s),
                         "Q_mean": float(s.Q_i.mean()), "Q_median": float(s.Q_i.median()),
                         "Q_se": float(s.Q_i.std(ddof=1) / np.sqrt(len(s))),
                         "E_mean": float(s.E_i2.mean())})
    tab = pd.DataFrame(rows)

    ratios = []
    for (protocol, source), g in df.groupby(["protocol", "source"]):
        for neg, label in (("M20-35", "male 20-35"), ("M36-50", "male 36-50"),
                           ("M51-75", "male 51-75"), ("F51-75", "female 51-75")):
            a = g.loc[STRATA[neg](g), "Q_i"].to_numpy()
            b = g.loc[STRATA["F20-35"](g), "Q_i"].to_numpy()
            if len(a) < 30 or len(b) < 30:
                continue
            point, lo, hi = _boot_mean_ratio(a, b)
            ratios.append({"protocol": protocol, "source": {6: "apple", 3: "garmin"}[source],
                           "reference": label, "n_ref": len(a), "n_young_f": len(b),
                           "ratio": point, "ci_lo": lo, "ci_hi": hi,
                           "max_lunar_share": point})
    rat = pd.DataFrame(ratios)
    print(tab.to_string(index=False))
    print()
    print(rat.to_string(index=False))
    tab.to_csv(args.out.with_name(args.out.stem + "_strata.csv"), index=False)
    rat.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
