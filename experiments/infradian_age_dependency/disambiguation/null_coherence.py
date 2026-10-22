#!/usr/bin/env python3
"""Null calibration of the cross-participant coherence spectrum.

Replaces the HR series with two-component AR(1) noise of the same length and
sampling geometry as the real windows, then runs the identical baseline
projection and cosinor fit.  Any structure in Rbar(T) produced here is a
property of the design (window length, baseline projection, red-noise
spectrum), not of the data.

Two nulls are generated:
  red    two-component AR(1): phi=0.70 (sd 3 bpm), phi=0.99 (sd 4 bpm)
  white  i.i.d. Gaussian (sd 8 bpm)

Usage: python null_coherence.py [--n 1500] [--protocol balanced90]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT_DIR / "rhr_sensitivity"))
sys.path.insert(0, str(EXPERIMENT_DIR / "scripts"))

import regressors as R  # noqa: E402
import shared  # noqa: E402
from phase_lock import _harmonic_fit, COHERENCE_PERIODS  # noqa: E402


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    e = rng.normal(0.0, sd * np.sqrt(1.0 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--protocol", default="balanced90")
    ap.add_argument("--out", type=Path, default=HERE / "null_coherence.csv")
    args = ap.parse_args()

    tasks = shared.build_tasks("HRp10", args.protocol, args.n)
    print(f"{len(tasks)} participant tasks, protocol {args.protocol}")
    rng = np.random.default_rng(20261020)

    acc: dict[str, list[np.ndarray]] = {"red": [], "white": []}
    for task in tasks:
        for win in task["windows"]:
            dates = np.asarray(win["dates"], dtype=np.int64)
            n = len(dates)
            clock = np.tile(rng.dirichlet(np.ones(4)), (n, 1))
            for kind, y in (("red", _ar1(rng, n, 0.70, 3.0) + _ar1(rng, n, 0.99, 4.0)),
                            ("white", rng.normal(0.0, 8.0, n))):
                Q0, _ = R.screen_qr(R.design_baseline(dates, clock))
                e0 = y - Q0 @ (Q0.T @ y)
                _, bc, bs = _harmonic_fit(Q0, e0, dates, 1.0 / COHERENCE_PERIODS)
                acc[kind].append(np.degrees(np.arctan2(bs, bc)) % 360.0)

    out = []
    for kind, lst in acc.items():
        ang = np.stack(lst)
        th = np.deg2rad(ang)
        rbar = np.hypot(np.cos(th).mean(0), np.sin(th).mean(0))
        nn = ang.shape[0]
        out.append(pd.DataFrame({"kind": kind, "n": nn,
                                 "period": COHERENCE_PERIODS, "rbar": rbar,
                                 "null95": np.sqrt(-np.log(0.05) / nn)}))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(args.out, index=False)
    print(res.groupby("kind").agg(n=("n", "first"), rbar_mean=("rbar", "mean"),
                                  rbar_max=("rbar", "max")).to_string())


if __name__ == "__main__":
    main()
