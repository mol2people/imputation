#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
05_long_period_check.py -- why does Garmin male power rise above the v2 AR(1) surrogate at long periods?

Garmin k=15 window 1; periods 120 d ... 3 d (f = k/720, k = 6..240). For each participant:
  * observed spectrum with the v2 baseline (intercept, linear trend, weekday, clock coverage)
  * observed spectrum with baseline + quadratic + cubic time trend ("poly")
  * surrogate spectra (B=10) from (a) v2 AR(1) and (b) AR(1)+slow component (REML, see v2/07_*.py), both
    projected with each baseline.
Writes results/garmin_k15/long_period_check.npz ; refuses to overwrite.
"""
import hashlib
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(EXP, "../.."))
V2S = os.path.join(ROOT, "experiments/female_infradian_cycles_v2/scripts")


def _load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


V = _load(os.path.join(V2S, "00_verify_v2_methods.py"), "verify_v2_methods")
P01 = _load(os.path.join(V2S, "01_extract_and_calibrate_infradian.py"), "p01")
P07 = _load(os.path.join(V2S, "07_two_component_null_recalibration.py"), "p07")
WIDE = 720.0 / np.arange(6, 241)
K, B = 15, 10
CACHE = os.path.join(EXP, "artifacts/cache_5min")


def spec(series, proj, w1):
    M, C, _ = V.precompute_harmonic_operators(w1, proj, WIDE)
    return V.compute_infradian_scores_batch(series, proj, M, C)[1].astype(np.float32)


def worker(cid):
    try:
        with np.load(os.path.join(CACHE, f"customer_{cid}.npz")) as z:
            hr, dates = z["hr"], z["date"]
        qd = dates[np.isfinite(hr).sum(axis=1) >= 120]
        s1, _ = P01.find_qualifying_windows(qd, k_per_block=K)
        if s1 is None:
            return cid, None
        w1 = P01.select_window_dates(qd, s1, int(cid), k_per_block=K)
        p10, clock, _, _ = P01.extract_features_on_dates(hr, dates, w1)
        X, _, proj = V.build_nonperiodic_baseline(w1, clock)
        e0 = proj @ p10
        rss0 = float(e0 @ e0)
        if rss0 < 1e-10:
            return cid, None
        t = (w1 - w1[0]).astype(float)
        t = 2 * t / t.max() - 1
        Xp = np.column_stack([X, t ** 2, t ** 3])
        projp = np.eye(len(w1)) - Xp @ np.linalg.pinv(Xp)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"lp|{cid}".encode()).digest()[:8], "little"))
        phi, sg = V.estimate_calendar_ar1(e0, w1)
        ar = V.simulate_calendar_ar1(phi, sg, w1, B, rng)
        w, U = np.linalg.eigh(proj)
        Q = U[:, w > 0.5]
        dt = np.abs(w1[:, None] - w1[None, :]).astype(float)
        a1, p1, a2, p2, _ = P07.reml_fit(Q.T @ e0, Q, dt, rss0 / Q.shape[1], phi)
        tc = V.simulate_calendar_ar1(p1, a1, w1, B, rng) + V.simulate_calendar_ar1(p2, a2, w1, B, rng)
        out = dict(obs=spec(e0[None], proj, w1)[0], obs_poly=spec(e0[None], projp, w1)[0],
                   ar=spec(ar, proj, w1), ar_poly=spec(ar, projp, w1),
                   tc=spec(tc, proj, w1), tc_poly=spec(tc, projp, w1))
        return cid, out
    except Exception:  # noqa: BLE001
        return cid, None


def main():
    out = os.path.join(EXP, "results/garmin_k15/long_period_check.npz")
    if os.path.exists(out):
        raise SystemExit("exists; refusing to overwrite")
    p = pd.read_parquet(os.path.join(EXP, "results/garmin_k15/participants.parquet"))
    p = p[p.E_i2.notna()]
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 10)) as ex:
        res = dict(ex.map(worker, p.customer_id.tolist(), chunksize=15))
    p = p[p.customer_id.map(lambda c: res.get(c) is not None)]
    arrs = {k: np.stack([res[c][k] for c in p.customer_id]) for k in next(iter(res.values()))}
    np.savez_compressed(out, cid=p.customer_id.values, y=p.y.values, age=p.age_group.values, wide=WIDE, **arrs)
    print("saved", len(p), {k: v.shape for k, v in arrs.items()})


if __name__ == "__main__":
    main()
