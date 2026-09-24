#!/usr/bin/env python
"""Days-scaling check, RANDOM day selection: AUROC vs k randomly sampled adequate days.

k grid: 4, 7, 14, 21, 30, 40. Sampling WITHOUT replacement (user decision
2026-09-24): per participant, per allocation, k distinct day indices drawn
uniformly from the 40 adequate days (seeded per allocation via a new component
ID; independent draws across k).

k=40 is special-cased to the identity (a uniform draw of 40 distinct days from
40 IS the full window): fill, MR fit seeds, transform, pooling order, statics
and hygiene all replicate the frozen v2 protocol exactly, so the k=40 block is
a v2 replay used as a validity gate (max |ΔAUROC| vs `temporal_metrics.csv`
per alloc must be ≤ 0.002, allowing the recorded 1-2 ULP threaded-BLAS ridge
nondeterminism).

Design (deviations from frozen v2 protocol, recorded):
  1. Per-participant day draws differ across k (independent realizations, not
     nested prefixes) — each k point is an unbiased "k random days" draw.
  2. MR kernels fit per alloc on 4 seeded days from ALL 40 days of train users
     (v2-identical), transform applied to all 40 days of everyone once, then
     pooled over each k's sampled indices (pooling is ~free vs transform).
  3. No HYDRA, no Shuffled_MR. α frozen at v2 values (MR 1e3 + 3e3 sensitivity;
     Profile24 1e3). B40_k / P24_k recomputed on the sampled day rows.
  4. Fill: train-only clock-bin medians from all 40 days (v2-identical).

Comparison output: paired deltas vs the first-k check
(`results/days_scaling_metrics.csv`) — the position-vs-budget contrast.

Outputs: `results/days_scaling_random_metrics.csv`, `results/days_scaling_random.md`.
"""
from __future__ import annotations

import os
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_temporal as rt  # noqa: E402

K_GRID = (4, 7, 14, 21, 30, 40)
ALPHA_PRIMARY = 1e3        # v2 frozen (MultiRocket, Profile24)
ALPHA_SENS = 3e3           # extended-grid MR optimum at k=40 (alpha_addendum)
COMP_DAY_SAMPLE = 5        # new component ID (v2 used 0-4)
GATE_TOL = 0.002           # k=40 v2-replay gate per alloc per arm
RESULTS = rt.RESULTS


def build_static_idx(users, d40, mask_pu, idx):
    """B40 (305) + P24_extra (48) on the given per-participant day indices.

    idx: (n, k) int64 day indices within each user's 40-day block, or None for
    the identity (all 40 days in natural order → v2 `build_static` replay).
    P24_extra is float64 (v2 convention).
    """
    n = len(users)
    starts = np.arange(n) * rt.DAYS
    if idx is None:
        idx = np.tile(np.arange(rt.DAYS), (n, 1))
    rows = (starts[:, None] + idx).ravel()
    d = d40.iloc[rows]
    g = d.groupby("user", sort=True)
    stat_cols = [f"d_ch3000_{s}" for s in
                 ("mean", "median", "sd", "vmin", "vmax", "n", "cov_h", "hours")]
    means = g[stat_cols].mean().reindex(users).to_numpy(dtype=np.float32)
    sds = g[stat_cols].std(ddof=1).reindex(users).to_numpy(dtype=np.float32)

    u_pos = pd.Series(np.arange(n), index=users)
    g_pos = d.user.map(u_pos).to_numpy()
    is_we = d.d_is_weekend.to_numpy().astype(bool)
    ch_mean = d.d_ch3000_mean.to_numpy(dtype=np.float32)
    wd_sum = np.bincount(g_pos[~is_we], weights=ch_mean[~is_we], minlength=n)
    wd_cnt = np.bincount(g_pos[~is_we], minlength=n)
    we_sum = np.bincount(g_pos[is_we], weights=ch_mean[is_we], minlength=n)
    we_cnt = np.bincount(g_pos[is_we], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        wd_mean = np.where(wd_cnt > 0, wd_sum / wd_cnt, np.nan)
        we_mean = np.where(we_cnt > 0, we_sum / we_cnt, np.nan)
    weekend_diff = (wd_mean - we_mean).astype(np.float32)[:, None]

    mask_sel = mask_pu[np.arange(n)[:, None], idx]            # (n, k, 288)
    mask_means = mask_sel.mean(axis=1).astype(np.float32)
    B40 = np.concatenate([means, sds, weekend_diff, mask_means], axis=1)
    assert B40.shape == (n, 305)

    hour_cols = [f"hour_h{h}" for h in range(24)]
    p24_mean = g[hour_cols].mean().reindex(users).to_numpy(dtype=np.float64)
    p24_sd = g[hour_cols].std(ddof=1).reindex(users).to_numpy(dtype=np.float64)
    P24 = np.concatenate([p24_mean, p24_sd], axis=1)
    return B40, P24


def apply_mr_pooled_sampled(mr_transform, params_base, params_diff,
                              hr_filled, mask_pu, sample_idx):
    """Transform all 40 days chunk-wise; pool mean + ddof=1 SD over each k's
    sampled indices. k=40 (idx None) pools the chunk slab directly in natural
    order — bit-identical to v2 `apply_mr_pooled`'s reduction order.

    Returns pooled_mean, pooled_sd: (n, len(K_GRID), MR_W) ordered by K_GRID.
    """
    n = hr_filled.shape[0]
    Nk = len(K_GRID)
    pooled_mean = np.zeros((n, Nk, rt.MR_WIDTH), dtype=np.float32)
    pooled_sd = np.zeros((n, Nk, rt.MR_WIDTH), dtype=np.float32)
    for s in range(0, n, rt.CHUNK_USERS):
        sz = min(rt.CHUNK_USERS, n - s)
        X = hr_filled[s:s + sz].reshape(sz * rt.DAYS, rt.N_BINS).astype(np.float64)
        X1 = np.diff(X, axis=1)
        F = mr_transform(X, X1, params_base, params_diff,
                          n_features_per_kernel=4)
        F = np.nan_to_num(F).astype(np.float32).reshape(sz, rt.DAYS, rt.MR_WIDTH)
        obs = mask_pu[s:s + sz].any(axis=2)                   # (sz, 40)
        F = np.where(obs[:, :, None], F, np.float32(np.nan))
        row = np.arange(sz)[:, None]
        with np.errstate(invalid="ignore", all="ignore"):
            for ki, k in enumerate(K_GRID):
                idx = sample_idx[k]
                G = F if idx is None else F[row, idx[s:s + sz]]   # (sz, k, W)
                pooled_mean[s:s + sz, ki] = np.nan_to_num(
                    np.nanmean(G, axis=1, dtype=np.float64),
                    nan=0.0).astype(np.float32)
                pooled_sd[s:s + sz, ki] = np.nan_to_num(
                    np.nanstd(G, axis=1, ddof=1, dtype=np.float64),
                    nan=0.0).astype(np.float32)
        del F
    return pooled_mean, pooled_sd


def main():
    t0 = time.perf_counter()
    print(f"[rand] K_GRID={K_GRID}, sampling WITHOUT replacement per participant; "
          f"α(MR)={ALPHA_PRIMARY:g}/{ALPHA_SENS:g}, α(P24)={ALPHA_PRIMARY:g}; "
          f"k=40 = identity = v2 replay gate (tol {GATE_TOL})", flush=True)

    Hydra, SparseScaler, mr_fit, mr_transform = rt.vendor_imports()
    users, y, d40, hr, mask, cnt, pu, pd_, folds, r1b = rt.load_data()
    d40 = d40.sort_values(["user", "date"], kind="stable").reset_index(drop=True)
    assert len(d40) == rt.COHORT_SIZE * rt.DAYS
    assert np.array_equal(d40.user.to_numpy(),
                          np.repeat(np.sort(np.unique(pu)), rt.DAYS))

    n = len(users)
    hr_raw = hr.reshape(n, rt.DAYS, rt.N_BINS)
    mask_pu = mask.reshape(n, rt.DAYS, rt.N_BINS)

    v2 = pd.read_csv(RESULTS / "temporal_metrics.csv")
    rows, gate_rows = [], []

    for r in range(10):
        wall_a = time.perf_counter()
        fold = (folds[folds.alloc == r].set_index("user_id")["fold"].astype(str))
        fold.index = fold.index.astype("int64")
        fold = fold.reindex(users).to_numpy()
        tr = np.flatnonzero(fold == "train")
        va = np.flatnonzero(fold == "val")
        te = np.flatnonzero(fold == "test")
        y_tr, y_va, y_te = y[tr], y[va], y[te]
        assert set(np.unique(fold)) <= {"train", "val", "test"}

        # per-participant day draws (independent across k; k=40 → identity)
        day_rng = np.random.default_rng(rt.comp_seed(r, COMP_DAY_SAMPLE))
        sample_idx = {}
        for k in K_GRID:
            sample_idx[k] = (None if k == rt.DAYS else
                             np.argsort(day_rng.random((n, rt.DAYS)),
                                        axis=1)[:, :k].astype(np.int64))

        # fill + MR fit: v2-identical (all 40 days, same seed streams)
        hr_filled, _ = rt.fill_hr(hr_raw, mask_pu, tr)
        fit_rng = np.random.default_rng(rt.comp_seed(r, rt.COMP_DAY_SELECT))
        fit_days = np.stack(
            [fit_rng.choice(rt.DAYS, 4, replace=False) for _ in range(len(tr))])
        X_fit = hr_filled[tr[:, None], fit_days].reshape(-1, rt.N_BINS).astype(np.float64)
        X1_fit = np.diff(X_fit, axis=1)
        rt._nb_seed(rt.comp_seed(r, rt.COMP_MR_FIT))
        mr_pb = mr_fit(X_fit, num_features=rt.MR_NUM_FEATURES_PER_TRANS,
                        max_dilations_per_kernel=32)
        rt._nb_seed(rt.comp_seed(r, rt.COMP_MR_FIT) + 1)
        mr_pd = mr_fit(X1_fit, num_features=rt.MR_NUM_FEATURES_PER_TRANS,
                        max_dilations_per_kernel=32)

        mr_mean_all, mr_sd_all = apply_mr_pooled_sampled(
            mr_transform, mr_pb, mr_pd, hr_filled, mask_pu, sample_idx)

        for ki, k in enumerate(K_GRID):
            B40_k, P24_k = build_static_idx(users, d40, mask_pu, sample_idx[k])
            mr_mean = mr_mean_all[:, ki, :]
            mr_sd = mr_sd_all[:, ki, :]

            B40Z_tr, B40Z_va, B40Z_te, *_ = rt.hygiene_std(
                B40_k[tr], B40_k[va], B40_k[te])
            P24Z_tr, P24Z_va, P24Z_te, *_ = rt.hygiene_std(
                P24_k[tr], P24_k[va], P24_k[te])
            mr_block = np.concatenate([mr_mean, mr_sd], axis=1)
            MRZ_tr, MRZ_va, MRZ_te, *_ = rt.hygiene_std(
                mr_block[tr], mr_block[va], mr_block[te])
            # v2 MR arm = B40Z ⊕ MRZ (B40 hygiene block added; 305 + 18816 = 19121)
            MRARM_tr = np.concatenate([B40Z_tr, MRZ_tr], axis=1)
            MRARM_va = np.concatenate([B40Z_va, MRZ_va], axis=1)
            MRARM_te = np.concatenate([B40Z_te, MRZ_te], axis=1)
            mr_n_cols = int(MRARM_tr.shape[1])

            recs = []
            for tag, alpha in (("primary", ALPHA_PRIMARY), ("sens3e3", ALPHA_SENS)):
                m = Ridge(alpha=alpha, fit_intercept=True).fit(MRARM_tr, y_tr)
                recs.append({
                    "alloc": r, "k": k, "arm": f"MultiRocket_{tag}",
                    "alpha": float(alpha),
                    "auroc_val": float(roc_auc_score(y_va, m.predict(MRARM_va))),
                    "auroc_test": float(roc_auc_score(y_te, m.predict(MRARM_te))),
                    "n_cols": mr_n_cols,
                })
            Z_tr = np.concatenate([B40Z_tr, P24Z_tr], axis=1)
            Z_va = np.concatenate([B40Z_va, P24Z_va], axis=1)
            Z_te = np.concatenate([B40Z_te, P24Z_te], axis=1)
            m = Ridge(alpha=ALPHA_PRIMARY, fit_intercept=True).fit(Z_tr, y_tr)
            recs.append({
                "alloc": r, "k": k, "arm": "Profile24",
                "alpha": float(ALPHA_PRIMARY),
                "auroc_val": float(roc_auc_score(y_va, m.predict(Z_va))),
                "auroc_test": float(roc_auc_score(y_te, m.predict(Z_te))),
                "n_cols": int(Z_tr.shape[1]),
            })
            rf = RandomForestClassifier(n_estimators=rt.N_TREES,
                                         random_state=rt.comp_seed(r, rt.COMP_RF),
                                         n_jobs=1, **rt.G1).fit(B40Z_tr, y_tr)
            j1 = list(rf.classes_).index(1)
            recs.append({
                "alloc": r, "k": k, "arm": "Summary_RF",
                "alpha": None,
                "auroc_val": float(roc_auc_score(
                    y_va, rf.predict_proba(B40Z_va)[:, j1])),
                "auroc_test": float(roc_auc_score(
                    y_te, rf.predict_proba(B40Z_te)[:, j1])),
                "n_cols": int(B40Z_tr.shape[1]),
            })
            rows.extend(recs)

            # k=40 v2-replay gate
            if k == rt.DAYS:
                for my_arm, v2_arm in (("MultiRocket_primary", "MultiRocket"),
                                        ("Profile24", "Profile24"),
                                        ("Summary_RF", "Summary_RF")):
                    mine = next(q for q in recs if q["arm"] == my_arm)
                    ref = v2[(v2.alloc == r) & (v2.arm == v2_arm)].iloc[0]
                    gate_rows.append({
                        "alloc": r, "arm": my_arm,
                        "auroc_val_mine": mine["auroc_val"],
                        "auroc_val_v2": float(ref.auroc_val),
                        "auroc_test_mine": mine["auroc_test"],
                        "auroc_test_v2": float(ref.auroc_test),
                        "max_abs_dval": abs(mine["auroc_val"] - float(ref.auroc_val)),
                        "max_abs_dtest": abs(mine["auroc_test"] - float(ref.auroc_test)),
                    })

        print(f"[rand] alloc {r} done in {time.perf_counter()-wall_a:.1f}s | "
              f"peak RSS {rt.peak_rss_mb():.0f} MiB", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS / "days_scaling_random_metrics.csv"
    df.to_csv(out, index=False)
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(RESULTS / "days_scaling_random_gate.csv", index=False)
    gate_pass = bool((gate[["max_abs_dval", "max_abs_dtest"]].max().max() <= GATE_TOL))
    print(f"\n[rand] wrote {out}")
    print(f"[rand] k=40 v2-replay gate: "
          f"{'PASS' if gate_pass else 'FAIL'} "
          f"(max |Δval|={gate.max_abs_dval.max():.2e}, "
          f"max |Δtest|={gate.max_abs_dtest.max():.2e}, tol {GATE_TOL})",
          flush=True)

    gv = df.groupby(["k", "arm"]).auroc_val.agg(["mean", "std"])
    gt = df.groupby(["k", "arm"]).auroc_test.agg(["mean", "std"])
    print("\n--- val AUROC (mean ± SD, 10 allocs) ---")
    print(gv.to_string())
    print("\n--- test AUROC (mean ± SD, 10 allocs) ---")
    print(gt.to_string())

    # paired deltas vs first-k check
    first = pd.read_csv(RESULTS / "days_scaling_metrics.csv")
    md = ["# Days-scaling check — RANDOM k adequate days (without replacement)\n",
          "Per participant per allocation, k distinct day indices drawn uniformly "
          "from the 40 adequate days (seeded, new component ID 5; independent "
          "draws across k). MR kernels/fill v2-identical (all 40 days); "
          "transform applied to all 40 days once, pooled per k over sampled "
          "indices. α frozen (MR 1e3 + 3e3 sensitivity, P24 1e3). "
          "**k=40 = identity = v2 replay** (validity gate).\n"]
    md.append("\n## Validation AUROC (mean ± SD, 10 allocs)\n")
    md.append("| k | arm | val AUROC | test AUROC | n cols | α |\n|---|---|---|---|---|---|\n")
    for (k, arm), row in gv.iterrows():
        sub = df[(df.k == k) & (df.arm == arm)]
        a = sub.alpha.iloc[0]
        a_str = "—" if pd.isna(a) else f"{a:g}"
        rt_ = gt.loc[(k, arm)]
        md.append(f"| {k} | `{arm}` | {row['mean']:.4f} ± {row['std']:.4f} | "
                  f"{rt_['mean']:.4f} ± {rt_['std']:.4f} | "
                  f"{int(sub.n_cols.iloc[0]):,} | {a_str} |\n")

    md.append(f"\n## k=40 v2-replay gate — "
              f"{'**PASS**' if gate_pass else '**FAIL**'} "
              f"(max |Δval| = {gate.max_abs_dval.max():.2e}, "
              f"max |Δtest| = {gate.max_abs_dtest.max():.2e}, "
              f"tol {GATE_TOL}; recorded 1–2 ULP threaded-BLAS ridge "
              f"nondeterminism)\n")

    md.append("\n## Random-k − first-k (paired over 10 allocs; val)\n")
    md.append("| k | arm | mean Δ | 95% t-CI | share > 0 |\n|---|---|---|---|---|\n")
    from scipy import stats as st
    for arm in ("MultiRocket_primary", "MultiRocket_sens3e3",
                "Profile24", "Summary_RF"):
        for k in (4, 7, 14, 21, 30):
            a = df[(df.k == k) & (df.arm == arm)].sort_values("alloc").auroc_val
            b = (first[(first.k == k) & (first.arm == arm)]
                 .sort_values("alloc").auroc_val)
            d = a.to_numpy() - b.to_numpy()
            m_, sd_ = float(d.mean()), float(d.std(ddof=1))
            h = float(st.t.ppf(0.975, len(d) - 1)) * sd_ / np.sqrt(len(d))
            md.append(f"| {k} | `{arm}` | {m_:+.4f} | "
                      f"[{m_-h:+.4f}, {m_+h:+.4f}] | {(d > 0).mean():.2f} |\n")
    md.append("\nPositive Δ = random days beat the *first* k days at the same k "
              "(early-history specialness); Δ ≈ 0 = only the budget matters "
              "(signal stationary across the 40-day window).\n")
    md_path = RESULTS / "days_scaling_random.md"
    md_path.write_text("".join(md))
    print(f"[rand] wrote {md_path}")
    print(f"[rand] total wall {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
