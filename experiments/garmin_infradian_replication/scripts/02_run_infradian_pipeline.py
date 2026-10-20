#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
02_run_infradian_pipeline.py DEVICE K

DEVICE in {apple, garmin}; K = qualified dates required per 30-d block (10 or 15).
Runs the v2 calibrated infradian test (window 1) with two nulls (v2 AR(1); AR(1)+slow, see
female_infradian_cycles_v2/scripts/07_*.py) and the second-window persistence analysis
(window 2 = earliest qualifying start >= start1+180):
  * delta_T = |best_T1 - best_T2|
  * frozen-period/phase out-of-sample prediction of the window-2 baseline-adjusted residual
    (harmonic at T1 with window-1 coefficients, absolute calendar time): r2_persist, corr_persist.
All outputs go to experiments/garmin_infradian_replication/results/{DEVICE}_k{K}/ ; refuses to overwrite.
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
EXP = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
V2S = os.path.join(ROOT, "experiments/female_infradian_cycles_v2/scripts")
B = 199
PERIODS = np.arange(25.0, 35.5, 0.5)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


V = _load(os.path.join(V2S, "00_verify_v2_methods.py"), "verify_v2_methods")
P01 = _load(os.path.join(V2S, "01_extract_and_calibrate_infradian.py"), "p01")
P02 = _load(os.path.join(V2S, "02_summarize_infradian_contrasts.py"), "p02")
P07 = _load(os.path.join(V2S, "07_two_component_null_recalibration.py"), "p07")


def window_features(hr, dates, w):
    p10, clock, occ, night = P01.extract_features_on_dates(hr, dates, w)
    _, _, proj = V.build_nonperiodic_baseline(w, clock)
    return p10, proj, occ, night


def worker(args):
    cid, path, k, tag = args
    rec = dict(customer_id=cid, eligible=False)
    try:
        with np.load(path) as z:
            hr, dates = z["hr"], z["date"]
        qd = dates[np.isfinite(hr).sum(axis=1) >= 120]
        s1, s2 = P01.find_qualifying_windows(qd, k_per_block=k)
        if s1 is None:
            return rec
        w1 = P01.select_window_dates(qd, s1, int(cid), k_per_block=k)
        p10, proj, occ, night = window_features(hr, dates, w1)
        e0 = proj @ p10
        rss0 = float(e0 @ e0)
        rec.update(eligible=True, n_obs=len(w1), w1_actual_span=int(w1[-1] - w1[0]), w1_mean_occ_5m=occ,
                   w1_mean_clock_night=night, eligible_w2=s2 is not None)
        if rss0 < 1e-10:
            return rec
        M, C, _ = V.precompute_harmonic_operators(w1, proj, PERIODS)
        mx, by = V.compute_infradian_scores_batch(e0[None], proj, M, C)
        S = float(mx[0])
        bi = int(by[0].argmax())
        T1 = float(PERIODS[bi])
        bc, bs = float(e0 @ M[:, 2 * bi]), float(e0 @ M[:, 2 * bi + 1])
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{tag}|{cid}".encode()).digest()[:8], "little"))
        phi, sg = V.estimate_calendar_ar1(e0, w1)
        nm, _ = V.compute_infradian_scores_batch(V.simulate_calendar_ar1(phi, sg, w1, B, rng), proj, M, C)
        w, U = np.linalg.eigh(proj)
        Q = U[:, w > 0.5]
        dt = np.abs(w1[:, None] - w1[None, :]).astype(float)
        a1, p1, a2, p2, _ = P07.reml_fit(Q.T @ e0, Q, dt, rss0 / Q.shape[1], phi)
        sim = V.simulate_calendar_ar1(p1, a1, w1, B, rng) + V.simulate_calendar_ar1(p2, a2, w1, B, rng)
        n2, _ = V.compute_infradian_scores_batch(sim, proj, M, C)
        rec.update(S_i=S, best_T=T1, A_i=float(np.hypot(bc, bs)), phi_hat=phi,
                   E_i=S - float(nm.mean()), p_i=float((1 + (nm >= S).sum()) / (B + 1)),
                   E_i2=S - float(n2.mean()), p_i2=float((1 + (n2 >= S).sum()) / (B + 1)))
        if s2 is not None:
            wb = P01.select_window_dates(qd, s2, int(cid), k_per_block=k)
            p10b, projb, _, _ = window_features(hr, dates, wb)
            eb = projb @ p10b
            Mb, Cb, _ = V.precompute_harmonic_operators(wb, projb, PERIODS)
            mxb, byb = V.compute_infradian_scores_batch(eb[None], projb, Mb, Cb)
            T2 = float(PERIODS[int(byb[0].argmax())])
            om = 2 * np.pi / T1
            h1 = np.column_stack([np.cos(om * (w1 - w1[0])), np.sin(om * (w1 - w1[0]))])
            coef = np.linalg.lstsq(proj @ h1, e0, rcond=None)[0]
            h2 = np.column_stack([np.cos(om * (wb - w1[0])), np.sin(om * (wb - w1[0]))])
            pred = projb @ (h2 @ coef)
            ss = float(eb @ eb)
            rec.update(S_i_w2=float(mxb[0]), best_T_w2=T2, delta_T=abs(T1 - T2),
                       r2_persist=1 - float(((eb - pred) ** 2).sum()) / ss if ss > 1e-10 else np.nan,
                       corr_persist=float(np.corrcoef(eb, pred)[0, 1]) if ss > 1e-10 and pred.std() > 0 else np.nan)
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)
    return rec


def main():
    device, k = sys.argv[1], int(sys.argv[2])
    out = os.path.join(EXP, "results", f"{device}_k{k}")
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is non-empty; refusing to overwrite.")
    os.makedirs(out, exist_ok=True)
    if device == "apple":
        al = pd.read_parquet(os.path.join(ROOT, "experiments/apple_cohort_scaling/artifacts/apple_10_allocations.parquet"))
        meta = al[["customer_id", "y", "age_group", "bmi_group"]].assign(customer_id=lambda d: d.customer_id.astype(int).astype(str))
        cache = os.path.join(ROOT, "experiments/apple_cohort_scaling/artifacts/cache_5min")
    else:
        meta = pd.read_csv(os.path.join(EXP, "artifacts/garmin_tier1c_cached_manifest.csv"), dtype={"customer_id": str})
        meta = meta[["customer_id", "y", "age_group", "bmi_group"]]
        cache = os.path.join(EXP, "artifacts/cache_5min")
    tasks = [(c, os.path.join(cache, f"customer_{c}.npz"), k, f"{device}{k}") for c in meta.customer_id]
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 10)) as ex:
        recs = []
        for i, r in enumerate(ex.map(worker, tasks, chunksize=20)):
            recs.append(r)
            if (i + 1) % 1000 == 0:
                print(i + 1, flush=True)
    df = pd.DataFrame(recs).merge(meta, on="customer_id")
    df.to_parquet(os.path.join(out, "participants.parquet"))
    ok = df[df.E_i2.notna()]
    print("cohort", len(df), "eligible", int(df.eligible.sum()), "analysed", len(ok),
          "w2 pairs", int(ok.delta_T.notna().sum()), "errors", int(df.get("error", pd.Series(dtype=object)).notna().sum()))
    print(ok.groupby(["y", "age_group"]).size().to_string())
    rows = []
    for ep in ["E_i", "E_i2"]:
        r = P02.compute_ols_contrasts(ok, endpoint=ep, adjust_covariates=True)
        rows.append(dict(endpoint=ep, test="omnibus", est=np.nan, se=np.nan, stat=r["omnibus_stat"], p=r["omnibus_p"]))
        for name, (c, est, se) in r["contrasts"].items():
            rows.append(dict(endpoint=ep, test=name, est=est, se=se, stat=(est / se) ** 2, p=float(st.chi2.sf((est / se) ** 2, 1))))
    ct = pd.DataFrame(rows)
    ct.to_csv(os.path.join(out, "contrasts.csv"), index=False)
    print(ct.round(4).to_string())
    g = ok.groupby(["y", "age_group"]).agg(n=("p_i", "size"), rej_ar1=("p_i", lambda x: (x < .05).mean()),
                                          rej_2c=("p_i2", lambda x: (x < .05).mean()), E_2c=("E_i2", "mean"),
                                          A_med=("A_i", "median"), T35=("best_T", lambda x: (x == 35).mean()))
    g.to_csv(os.path.join(out, "rejection_rates.csv"))
    print(g.round(3).to_string())
    # persistence summary
    w = ok[ok.delta_T.notna()]
    rng = np.random.default_rng(0)
    rows = []
    for (y, a), d in w.groupby(["y", "age_group"]):
        shuf = np.mean([(np.abs(d.best_T.values - rng.permutation(d.best_T_w2.values)) <= 1.5).mean() for _ in range(500)])
        cp = d.corr_persist.dropna()
        rows.append(dict(y=y, age_group=a, n_pairs=len(d), P_dT_le_1p5=(d.delta_T <= 1.5).mean(), P_dT_le_1p5_shuffled=shuf,
                         mean_r2_persist=d.r2_persist.mean(), median_corr=cp.median(), frac_corr_pos=(cp > 0).mean(),
                         sign_test_p=float(st.binomtest(int((cp > 0).sum()), len(cp), 0.5).pvalue) if len(cp) else np.nan))
    ps = pd.DataFrame(rows)
    ps.to_csv(os.path.join(out, "persistence_summary.csv"), index=False)
    print(ps.round(3).to_string())


if __name__ == "__main__":
    main()
