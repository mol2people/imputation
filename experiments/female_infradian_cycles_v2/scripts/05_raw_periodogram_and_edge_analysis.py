#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
05_raw_periodogram_and_edge_analysis.py

(1) Mean variance-fraction periodogram, 4-60 d, per sex x age, observed vs AR(1)-surrogate
    envelope (same baseline projection, same 90 sampled dates as the v2 pipeline).
(2) Origin of the LS peak at 35 d: argmax over the 25-35 d grid (step 0.5) for observed data,
    AR(1) surrogates (fitted phi, sigma2), and white-noise surrogates; plus spectral-window bias.

Score at period T: (RSS_0 - RSS_T)/RSS_0 after Frisch-Waugh projection onto the
non-periodic baseline (v2 definition; equals partial R^2 of a single harmonic).
Output: experiments/female_infradian_cycles_v2/results/raw_periodogram/{spectra.npz,*.png}
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


V = _load(os.path.join(HERE, "00_verify_v2_methods.py"), "verify_v2_methods")
P01 = _load(os.path.join(HERE, "01_extract_and_calibrate_infradian.py"), "p01")

K_FREQ = np.arange(12, 241)                     # f = k/720 cycles/day
WIDE = 720.0 / K_FREQ                           # 60 d ... 3 d
NARROW = np.arange(25.0, 35.5, 0.5)             # v2 search grid
B = 30
CACHE = os.path.join(ROOT, "experiments/apple_cohort_scaling/artifacts/cache_5min")


def worker(args):
    cid, w1_dates, phi, sigma2 = args
    try:
        with np.load(os.path.join(CACHE, f"customer_{cid}.npz")) as z:
            hr, dates = z["hr"], z["date"]
        w1 = np.asarray(w1_dates, dtype=np.int64)
        p10, clock, _, _ = P01.extract_features_on_dates(hr, dates, w1)
        _, _, proj = V.build_nonperiodic_baseline(w1, clock)
        e0 = proj @ p10
        if float(e0 @ e0) < 1e-10:
            return cid, None
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"raw_pgram|{cid}".encode()).digest()[:8], "little"))
        ar = V.simulate_calendar_ar1(phi, sigma2, w1, B, rng)
        wn = rng.standard_normal((B, len(w1)))
        out = {}
        for tag, grid in (("w", WIDE), ("n", NARROW)):
            M, C, _ = V.precompute_harmonic_operators(w1, proj, grid)
            out[tag + "_obs"] = V.compute_infradian_scores_batch(e0[None], proj, M, C)[1][0]
            out[tag + "_ar"] = V.compute_infradian_scores_batch(ar, proj, M, C)[1]
            out[tag + "_wn"] = V.compute_infradian_scores_batch(wn, proj, M, C)[1]
        return cid, {k: v.astype(np.float32) for k, v in out.items()}
    except Exception as e:  # noqa: BLE001
        return cid, None


def main():
    el = pd.read_parquet(os.path.join(ROOT, "experiments/female_infradian_cycles_v2/results/window_eligibility.parquet"))
    el = el[el.eligible_w1 & el.phi_hat.notna()].reset_index(drop=True)
    print("participants:", len(el))
    tasks = [(int(r.customer_id), list(r.w1_selected_dates), float(r.phi_hat), float(r.sigma2_hat)) for r in el.itertuples()]
    res = {}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 10)) as ex:
        for i, (cid, o) in enumerate(ex.map(worker, tasks, chunksize=25)):
            if o is not None:
                res[cid] = o
            if (i + 1) % 500 == 0:
                print(i + 1, flush=True)
    el = el[el.customer_id.isin(res)].reset_index(drop=True)
    arrs = {k: np.stack([res[c][k] for c in el.customer_id]) for k in next(iter(res.values()))}
    outdir = os.path.join(ROOT, "experiments/female_infradian_cycles_v2/results/raw_periodogram")
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, "spectra.npz"), cid=el.customer_id.values, y=el.y.values,
                        age=el.age_group.values, phi=el.phi_hat.values, wide=WIDE, narrow=NARROW, **arrs)
    print("saved", {k: v.shape for k, v in arrs.items()})


if __name__ == "__main__":
    main()
