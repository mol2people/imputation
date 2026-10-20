#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
03_garmin_spectra.py -- mean variance-fraction periodogram (3-60 d) of Garmin k=15 window 1 with AR(1) surrogates
(v2 null, B=10 per participant). Writes results/garmin_k15/spectra.npz; refuses to overwrite.
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
WIDE = 720.0 / np.arange(12, 241)
K, B = 15, 10
CACHE = os.path.join(EXP, "artifacts/cache_5min")


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
        _, _, proj = V.build_nonperiodic_baseline(w1, clock)
        e0 = proj @ p10
        if float(e0 @ e0) < 1e-10:
            return cid, None
        phi, sg = V.estimate_calendar_ar1(e0, w1)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"gspec|{cid}".encode()).digest()[:8], "little"))
        M, C, _ = V.precompute_harmonic_operators(w1, proj, WIDE)
        obs = V.compute_infradian_scores_batch(e0[None], proj, M, C)[1][0]
        ar = V.compute_infradian_scores_batch(V.simulate_calendar_ar1(phi, sg, w1, B, rng), proj, M, C)[1]
        return cid, (obs.astype(np.float32), ar.astype(np.float32))
    except Exception:  # noqa: BLE001
        return cid, None


def main():
    out = os.path.join(EXP, "results/garmin_k15/spectra.npz")
    if os.path.exists(out):
        raise SystemExit("exists; refusing to overwrite")
    p = pd.read_parquet(os.path.join(EXP, "results/garmin_k15/participants.parquet"))
    p = p[p.E_i2.notna()]
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 10)) as ex:
        res = dict(ex.map(worker, p.customer_id.tolist(), chunksize=20))
    p = p[p.customer_id.map(lambda c: res.get(c) is not None)]
    np.savez_compressed(out, cid=p.customer_id.values, y=p.y.values, age=p.age_group.values, wide=WIDE,
                        obs=np.stack([res[c][0] for c in p.customer_id]), ar=np.stack([res[c][1] for c in p.customer_id]))
    print("saved", len(p))


if __name__ == "__main__":
    main()
