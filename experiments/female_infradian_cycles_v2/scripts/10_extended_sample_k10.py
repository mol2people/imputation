#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
10_extended_sample_k10.py

Extends the v2 calibrated test to a larger sample by relaxing window eligibility from
k=15 to K_PER_BLOCK=10 qualified dates per 30-day block (60 observations / 180 d instead of 90).
Population: all participants in apple_10_allocations.parquet (N=11,139); no v1 screening.
Same date-selection hashing, baseline, 25-35 d grid (step 0.5), B=199 as v2; both nulls:
  (a) v2 stationary calendar AR(1)
  (b) AR(1) + slow AR(1) component (REML fit, phi2 in {0.98, 0.995}) -- see 07_*.py
Outputs: results/extended_k10/{participants_k10.parquet, contrasts_k10.csv, rejection_rates_k10.csv}
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
K_PER_BLOCK = int(os.environ.get("K_PER_BLOCK", "10"))
B = 199
PERIODS = np.arange(25.0, 35.5, 0.5)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


V = _load(os.path.join(HERE, "00_verify_v2_methods.py"), "verify_v2_methods")
P01 = _load(os.path.join(HERE, "01_extract_and_calibrate_infradian.py"), "p01")
P02 = _load(os.path.join(HERE, "02_summarize_infradian_contrasts.py"), "p02")
P07 = _load(os.path.join(HERE, "07_two_component_null_recalibration.py"), "p07")
CACHE = os.path.join(ROOT, "experiments/apple_cohort_scaling/artifacts/cache_5min")


def worker(cid):
    rec = dict(customer_id=cid, eligible=False)
    try:
        with np.load(os.path.join(CACHE, f"customer_{cid}.npz")) as z:
            hr, dates = z["hr"], z["date"]
        qd = dates[np.isfinite(hr).sum(axis=1) >= 120]
        start, _ = P01.find_qualifying_windows(qd, k_per_block=K_PER_BLOCK)
        if start is None:
            return rec
        w1 = P01.select_window_dates(qd, start, cid, k_per_block=K_PER_BLOCK)
        p10, clock, occ, night = P01.extract_features_on_dates(hr, dates, w1)
        _, _, proj = V.build_nonperiodic_baseline(w1, clock)
        e0 = proj @ p10
        rss0 = float(e0 @ e0)
        rec.update(eligible=True, n_obs=len(w1), w1_actual_span=int(w1[-1] - w1[0]),
                   w1_mean_occ_5m=occ, w1_mean_clock_night=night)
        if rss0 < 1e-10:
            return rec
        M, C, _ = V.precompute_harmonic_operators(w1, proj, PERIODS)
        mx, by = V.compute_infradian_scores_batch(e0[None], proj, M, C)
        S = float(mx[0])
        seed = int.from_bytes(hashlib.sha256(f"k10|{cid}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        phi, s2 = V.estimate_calendar_ar1(e0, w1)
        nm, _ = V.compute_infradian_scores_batch(V.simulate_calendar_ar1(phi, s2, w1, B, rng), proj, M, C)
        w, U = np.linalg.eigh(proj)
        Q = U[:, w > 0.5]
        dt = np.abs(w1[:, None] - w1[None, :]).astype(float)
        a1, p1, a2, p2, _ = P07.reml_fit(Q.T @ e0, Q, dt, rss0 / Q.shape[1], phi)
        sim = V.simulate_calendar_ar1(p1, a1, w1, B, rng) + V.simulate_calendar_ar1(p2, a2, w1, B, rng)
        n2, _ = V.compute_infradian_scores_batch(sim, proj, M, C)
        bc = float(e0 @ M[:, 2 * int(by[0].argmax())])
        bs = float(e0 @ M[:, 2 * int(by[0].argmax()) + 1])
        rec.update(S_i=S, best_T=float(PERIODS[int(by[0].argmax())]), A_i=float(np.hypot(bc, bs)), phi_hat=phi,
                   E_i=S - float(nm.mean()), p_i=float((1 + (nm >= S).sum()) / (B + 1)),
                   E_i2=S - float(n2.mean()), p_i2=float((1 + (n2 >= S).sum()) / (B + 1)))
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)
    return rec


def main():
    alloc = pd.read_parquet(os.path.join(ROOT, "experiments/apple_cohort_scaling/artifacts/apple_10_allocations.parquet"))
    ids = alloc.customer_id.astype(int).tolist()
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 10)) as ex:
        recs = []
        for i, r in enumerate(ex.map(worker, ids, chunksize=25)):
            recs.append(r)
            if (i + 1) % 1000 == 0:
                print(i + 1, flush=True)
    meta = alloc[["customer_id", "y", "age_group", "bmi_group"]].assign(customer_id=lambda d: d.customer_id.astype(int))
    df = pd.DataFrame(recs).merge(meta, on="customer_id")
    out = os.path.join(ROOT, f"experiments/female_infradian_cycles_v2/results/extended_k{K_PER_BLOCK}")
    os.makedirs(out, exist_ok=True)
    df.to_parquet(os.path.join(out, f"participants_k{K_PER_BLOCK}.parquet"))
    ok = df[df.E_i2.notna()]
    print("eligible:", int(df.eligible.sum()), "analysed:", len(ok), "errors:", int(df.get("error", pd.Series(dtype=object)).notna().sum()))
    print(ok.groupby(["y", "age_group"]).size())
    rows = []
    for ep in ["E_i", "E_i2"]:
        r = P02.compute_ols_contrasts(ok, endpoint=ep, adjust_covariates=True)
        rows.append(dict(endpoint=ep, test="omnibus", est=np.nan, se=np.nan, stat=r["omnibus_stat"], p=r["omnibus_p"]))
        for k, (c, est, se) in r["contrasts"].items():
            rows.append(dict(endpoint=ep, test=k, est=est, se=se, stat=(est / se) ** 2, p=float(st.chi2.sf((est / se) ** 2, 1))))
    ct = pd.DataFrame(rows)
    ct.to_csv(os.path.join(out, f"contrasts_k{K_PER_BLOCK}.csv"), index=False)
    print(ct.to_string())
    g = ok.groupby(["y", "age_group"]).agg(n=("p_i", "size"), rej_ar1=("p_i", lambda x: (x < .05).mean()),
                                          rej_2c=("p_i2", lambda x: (x < .05).mean()), E_ar1=("E_i", "mean"), E_2c=("E_i2", "mean"),
                                          A_med=("A_i", "median"), T35=("best_T", lambda x: (x == 35).mean()))
    g.to_csv(os.path.join(out, f"rejection_rates_k{K_PER_BLOCK}.csv"))
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
