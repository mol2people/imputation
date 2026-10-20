#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
07_two_component_null_recalibration.py

Re-calibrates the v2 infradian test with a heavier-tailed null:
    e ~ N(0, C),  C(tau) = s1 * phi1^tau + s2 * phi2^tau,  phi2 in {0.98, 0.995} (slow component)
fitted per participant by REML on the baseline-projected residual (profile over phi2 grid).
Same 90 dates, baseline, search grid [25,35] step 0.5, score S, B = 199 as v2.
Outputs: results/recalibration/{null2_participants.parquet, contrasts_null2.csv, *.png}
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
P02 = _load(os.path.join(HERE, "02_summarize_infradian_contrasts.py"), "p02")

CACHE = os.path.join(ROOT, "experiments/apple_cohort_scaling/artifacts/cache_5min")
PERIODS = np.arange(25.0, 35.5, 0.5)
WIDE = 720.0 / np.arange(12, 241)
PHI2_GRID = (0.98, 0.995)
B = 199


def reml_fit(z, Q, dt_abs, s_init, phi_init):
    """Return (s1, phi1, s2, phi2, ll) maximizing REML likelihood of z = Q^T e."""
    n = len(z)
    best = None
    for phi2 in PHI2_GRID:
        K2 = phi2 ** dt_abs

        def nll(th):
            s1, s2 = np.exp(th[0]), np.exp(th[2])
            p1 = 0.95 / (1 + np.exp(-th[1]))
            C = s1 * p1 ** dt_abs + s2 * K2
            S = Q.T @ C @ Q
            try:
                L = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                return 1e10
            a = np.linalg.solve(L, z)
            return float(np.sum(np.log(np.diag(L))) + 0.5 * a @ a)

        th0 = np.array([np.log(s_init * .7), 0.0, np.log(s_init * .3)])
        r = minimize(nll, th0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 300})
        if best is None or r.fun < best[0]:
            best = (r.fun, np.exp(r.x[0]), 0.95 / (1 + np.exp(-r.x[1])), np.exp(r.x[2]), phi2)
    return best[1], best[2], best[3], best[4], -best[0]


def worker(args):
    cid, w1_dates, phi_old = args
    try:
        with np.load(os.path.join(CACHE, f"customer_{cid}.npz")) as zf:
            hr, dates = zf["hr"], zf["date"]
        w1 = np.asarray(w1_dates, dtype=np.int64)
        p10, clock, _, _ = P01.extract_features_on_dates(hr, dates, w1)
        _, _, proj = V.build_nonperiodic_baseline(w1, clock)
        e0 = proj @ p10
        rss0 = float(e0 @ e0)
        if rss0 < 1e-10:
            return cid, None
        w, U = np.linalg.eigh(proj)
        Q = U[:, w > 0.5]
        z = Q.T @ e0
        dt_abs = np.abs(w1[:, None] - w1[None, :]).astype(float)
        s1, p1, s2, p2, ll = reml_fit(z, Q, dt_abs, rss0 / Q.shape[1], phi_old)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"null2|{cid}".encode()).digest()[:8], "little"))
        sim = V.simulate_calendar_ar1(p1, s1, w1, B, rng) + V.simulate_calendar_ar1(p2, s2, w1, B, rng)
        M, C, _ = V.precompute_harmonic_operators(w1, proj, PERIODS)
        mx_o, by_o = V.compute_infradian_scores_batch(e0[None], proj, M, C)
        mx_n, by_n = V.compute_infradian_scores_batch(sim, proj, M, C)
        S = float(mx_o[0])
        Mw, Cw, _ = V.precompute_harmonic_operators(w1, proj, WIDE)
        spec_null = V.compute_infradian_scores_batch(sim[:30], proj, Mw, Cw)[1].astype(np.float32)
        return cid, dict(
            s1=s1, phi1=p1, s2=s2, phi2=p2, ll=ll, S_i2=S, mean_S_null2=float(mx_n.mean()),
            E_i2=S - float(mx_n.mean()), p_i2=float((1 + np.sum(mx_n >= S)) / (B + 1)),
            argmax_obs=int(by_o[0].argmax()), argmax_null=by_n.argmax(1).astype(np.int8),
            spec_null=spec_null,
        )
    except Exception as e:  # noqa: BLE001
        return cid, None


def main():
    df = pd.read_parquet(os.path.join(ROOT, "experiments/female_infradian_cycles_v2/results/window_eligibility.parquet"))
    df = df[df.eligible_w1].copy()
    ok = df[df.phi_hat.notna()]
    tasks = [(int(r.customer_id), list(r.w1_selected_dates), float(r.phi_hat)) for r in ok.itertuples()]
    res = {}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 10)) as ex:
        for i, (cid, o) in enumerate(ex.map(worker, tasks, chunksize=20)):
            if o is not None:
                res[cid] = o
            if (i + 1) % 500 == 0:
                print(i + 1, flush=True)
    out = os.path.join(ROOT, "experiments/female_infradian_cycles_v2/results/recalibration")
    os.makedirs(out, exist_ok=True)
    cols = ["s1", "phi1", "s2", "phi2", "ll", "S_i2", "mean_S_null2", "E_i2", "p_i2", "argmax_obs"]
    t = pd.DataFrame({"customer_id": list(res)}).merge(df, on="customer_id")
    for c in cols:
        t[c] = [res[k][c] for k in t.customer_id]
    t.drop(columns=["w1_selected_dates", "w2_selected_dates"]).to_parquet(os.path.join(out, "null2_participants.parquet"))
    np.savez_compressed(os.path.join(out, "null2_extras.npz"), cid=t.customer_id.values,
                        argmax_null=np.stack([res[k]["argmax_null"] for k in t.customer_id]),
                        spec_null=np.stack([res[k]["spec_null"] for k in t.customer_id]))
    print("n =", len(t))

    # contrasts: original E_i vs new E_i2 (same model, same covariates)
    rows = []
    for ep in ["E_i", "E_i2"]:
        r = P02.compute_ols_contrasts(t, endpoint=ep, adjust_covariates=True)
        rows.append(dict(endpoint=ep, test="omnibus_age_x_sex", est=np.nan, se=np.nan, stat=r["omnibus_stat"], p=r["omnibus_p"]))
        for k, (c, est, se) in r["contrasts"].items():
            rows.append(dict(endpoint=ep, test=k, est=est, se=se, stat=(est / se) ** 2, p=float(1 - __import__("scipy.stats").stats.chi2.cdf((est / se) ** 2, 1))))
    pd.DataFrame(rows).to_csv(os.path.join(out, "contrasts_null2.csv"), index=False)
    print(pd.DataFrame(rows).to_string())

    # rejection rates
    t["p_old"] = t["p_i"]
    g = t.groupby(["y", "age_group"]).agg(n=("p_old", "size"), rej_old=("p_old", lambda x: (x < .05).mean()),
                                         rej_new=("p_i2", lambda x: (x < .05).mean()),
                                         E_old=("E_i", "mean"), E_new=("E_i2", "mean"),
                                         phi1=("phi1", "median"), s2_frac=("s2", lambda x: np.nan))
    g["s2_frac"] = t.groupby(["y", "age_group"]).apply(lambda d: (d.s2 / (d.s1 + d.s2)).median())
    g.to_csv(os.path.join(out, "rejection_rates.csv"))
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
