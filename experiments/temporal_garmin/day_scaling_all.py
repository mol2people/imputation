#!/usr/bin/env python
"""Full-history days-scaling: sample k days from EACH USER'S ENTIRE adequate-day
pool (dayall; mean 425, min 80, p50 476, p90 539, max 979), not just the first 40.

Phase A — fixed-k curve, k ∈ {4, 7, 14, 21, 30, 40, 60, 80} (80 = min pool ⇒
  no per-k cohort selection), 10 allocations:
  - ONE nested random permutation per user per allocation (component seed 5);
    k-points are prefixes of that permutation (budget axis on a fixed random
    day sequence; kernels shared across k within an allocation).
  - MR kernels fit per allocation on 4 seeded days drawn from the train users'
    80-day sampled window (component seeds 0/1, v2 stream).
  - Fill: per-allocation train-only clock-bin medians from train users' 80-day
    sampled windows (mask-aware; v2 convention).
  - B40_k / P24_k from dayall rows of the sampled days; mask block from the
    sampled bins. Arms: MR α=1e3 / α=3e3 (B40Z ⊕ MRZ), Profile24, Summary_RF.
Phase B — all-days endpoint (the "up to mean" point): pooled features over
  each user's FULL pool. SINGLE transform pass with a dedicated kernel draw
  (component seed 6; fit on 4 seeded days from alloc-0 train users' full
  pools; alloc-0 fill), then evaluated under ALL 10 fold-splits (hygiene/
  ridge/RF refit per split) — fold-split variability only, no kernel
  resampling at the endpoint (recorded).

Requires cache/bins_all/ from scan_bins_all.py (memmap; keeps the 2.4 GB bin
arrays out of RSS).

Outputs: results/days_scaling_all_metrics.csv (k=-1 ⇒ all-days endpoint),
         results/days_scaling_all.md.
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

K_GRID = (4, 7, 14, 21, 30, 40, 60, 80)
K_MAX = max(K_GRID)
ALPHA_PRIMARY = 1e3
ALPHA_SENS = 3e3
COMP_PERM = 5            # nested per-user day permutation
COMP_ENDPOINT = 6        # endpoint kernel draw
BINS_ALL = HERE / "cache" / "bins_all"
DAYALL = (rt.REPO / "experiments" / "r1b_dateaware_garmin_2026-09-21"
          / "cache" / "dayall_table.parquet")
RESULTS = rt.RESULTS

STAT_COLS = [f"d_ch3000_{s}" for s in
             ("mean", "median", "sd", "vmin", "vmax", "n", "cov_h", "hours")]
HOUR_COLS = [f"hour_h{h}" for h in range(24)]


def load_all():
    """v2 cohort/folds/labels + dayall statics table + memmap bins."""
    users, y, d40, hr, mask, cnt, pu, pd_, folds, r1b = rt.load_data()
    da = pd.read_parquet(DAYALL)
    need = {"user", "date", "d_is_weekend", *STAT_COLS, *HOUR_COLS}
    missing = need - set(da.columns)
    assert not missing, f"dayall missing columns: {missing}"
    da = da.sort_values(["user", "date"], kind="stable").reset_index(drop=True)

    dpu = np.load(BINS_ALL / "days_per_user.npy")
    starts = np.load(BINS_ALL / "starts.npy")
    scan_users = np.load(BINS_ALL / "user_ids.npy")
    n = len(users)
    assert np.array_equal(scan_users, users), "scan cohort != v2 cohort"
    assert dpu.min() >= K_MAX, f"min pool {dpu.min()} < K_MAX {K_MAX}"
    assert np.array_equal(da.user.to_numpy(), np.repeat(users, dpu))
    hr_m = np.load(BINS_ALL / "hr_all.npy", mmap_mode="r")
    mask_m = np.load(BINS_ALL / "mask_all.npy", mmap_mode="r")
    assert hr_m.shape == mask_m.shape == (int(dpu.sum()), rt.N_BINS)
    return users, y, folds, da, dpu, starts, hr_m, mask_m


def fill_from_window(hr_K, mask_K, tr):
    """Train-only clock-bin medians from train users' sampled 80-day windows
    (mask-aware; v2 fill_hr convention, applied to the (n, K_MAX, 288) slab)."""
    sub = hr_K[tr]; msub = mask_K[tr]
    obs = sub[msub]
    overall = float(np.nanmedian(obs)) if obs.size else 80.0
    fill = np.full(rt.N_BINS, overall, dtype=np.float32)
    with np.errstate(invalid="ignore", all="ignore"):
        for b in range(rt.N_BINS):
            col = sub[:, :, b][msub[:, :, b]]
            if col.size:
                fill[b] = np.float32(np.nanmedian(col))
    return np.where(np.isfinite(fill), fill, overall).astype(np.float32)


def build_static_rows(users, da, mask_sel):
    """B40 (305) + P24_extra (48) from a per-user day-row subset.
    `da` must be pre-sorted (user, date); mask_sel (n, k, 288) bool."""
    n = len(users)
    g = da.groupby("user", sort=True)
    means = g[STAT_COLS].mean().reindex(users).to_numpy(dtype=np.float32)
    sds = g[STAT_COLS].std(ddof=1).reindex(users).to_numpy(dtype=np.float32)

    u_pos = pd.Series(np.arange(n), index=users)
    g_pos = da.user.map(u_pos).to_numpy()
    is_we = da.d_is_weekend.to_numpy().astype(bool)
    ch_mean = da.d_ch3000_mean.to_numpy(dtype=np.float32)
    wd_sum = np.bincount(g_pos[~is_we], weights=ch_mean[~is_we], minlength=n)
    wd_cnt = np.bincount(g_pos[~is_we], minlength=n)
    we_sum = np.bincount(g_pos[is_we], weights=ch_mean[is_we], minlength=n)
    we_cnt = np.bincount(g_pos[is_we], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        wd_mean = np.where(wd_cnt > 0, wd_sum / wd_cnt, np.nan)
        we_mean = np.where(we_cnt > 0, we_sum / we_cnt, np.nan)
    weekend_diff = (wd_mean - we_mean).astype(np.float32)[:, None]

    mask_means = mask_sel.mean(axis=1).astype(np.float32)
    B40 = np.concatenate([means, sds, weekend_diff, mask_means], axis=1)
    assert B40.shape == (n, 305)

    p24_mean = g[HOUR_COLS].mean().reindex(users).to_numpy(dtype=np.float64)
    p24_sd = g[HOUR_COLS].std(ddof=1).reindex(users).to_numpy(dtype=np.float64)
    P24 = np.concatenate([p24_mean, p24_sd], axis=1)
    return B40, P24


def pool_prefixes(F, obs, pooled_mean, pooled_sd, s):
    """Pool mean + ddof=1 SD over prefixes of the (sz, K_MAX, W) slab."""
    with np.errstate(invalid="ignore", all="ignore"):
        for ki, k in enumerate(K_GRID):
            G = F[:, :k, :]
            pooled_mean[s:s + F.shape[0], ki] = np.nan_to_num(
                np.nanmean(G, axis=1, dtype=np.float64), nan=0.0).astype(np.float32)
            pooled_sd[s:s + F.shape[0], ki] = np.nan_to_num(
                np.nanstd(G, axis=1, ddof=1, dtype=np.float64), nan=0.0).astype(np.float32)


def fit_models(k, r, B40_k, P24_k, mr_mean, mr_sd, tr, va, te,
               y_tr, y_va, y_te):
    """Hygiene + ridge/RF for the four arms. Returns metric rows."""
    B40Z_tr, B40Z_va, B40Z_te, *_ = rt.hygiene_std(
        B40_k[tr], B40_k[va], B40_k[te])
    P24Z_tr, P24Z_va, P24Z_te, *_ = rt.hygiene_std(
        P24_k[tr], P24_k[va], P24_k[te])
    mr_block = np.concatenate([mr_mean, mr_sd], axis=1)
    MRZ_tr, MRZ_va, MRZ_te, *_ = rt.hygiene_std(
        mr_block[tr], mr_block[va], mr_block[te])
    MRARM_tr = np.concatenate([B40Z_tr, MRZ_tr], axis=1)
    MRARM_va = np.concatenate([B40Z_va, MRZ_va], axis=1)
    MRARM_te = np.concatenate([B40Z_te, MRZ_te], axis=1)
    mr_n_cols = int(MRARM_tr.shape[1])

    rows = []
    for tag, alpha in (("primary", ALPHA_PRIMARY), ("sens3e3", ALPHA_SENS)):
        m = Ridge(alpha=alpha, fit_intercept=True).fit(MRARM_tr, y_tr)
        rows.append({"alloc": r, "k": k, "arm": f"MultiRocket_{tag}",
                     "alpha": float(alpha),
                     "auroc_val": float(roc_auc_score(y_va, m.predict(MRARM_va))),
                     "auroc_test": float(roc_auc_score(y_te, m.predict(MRARM_te))),
                     "n_cols": mr_n_cols})
    Z_tr = np.concatenate([B40Z_tr, P24Z_tr], axis=1)
    Z_va = np.concatenate([B40Z_va, P24Z_va], axis=1)
    Z_te = np.concatenate([B40Z_te, P24Z_te], axis=1)
    m = Ridge(alpha=ALPHA_PRIMARY, fit_intercept=True).fit(Z_tr, y_tr)
    rows.append({"alloc": r, "k": k, "arm": "Profile24",
                 "alpha": float(ALPHA_PRIMARY),
                 "auroc_val": float(roc_auc_score(y_va, m.predict(Z_va))),
                 "auroc_test": float(roc_auc_score(y_te, m.predict(Z_te))),
                 "n_cols": int(Z_tr.shape[1])})
    rf = RandomForestClassifier(n_estimators=rt.N_TREES,
                                 random_state=rt.comp_seed(r, rt.COMP_RF),
                                 n_jobs=1, **rt.G1).fit(B40Z_tr, y_tr)
    j1 = list(rf.classes_).index(1)
    rows.append({"alloc": r, "k": k, "arm": "Summary_RF", "alpha": None,
                 "auroc_val": float(roc_auc_score(
                     y_va, rf.predict_proba(B40Z_va)[:, j1])),
                 "auroc_test": float(roc_auc_score(
                     y_te, rf.predict_proba(B40Z_te)[:, j1])),
                 "n_cols": int(B40Z_tr.shape[1])})
    return rows


def main():
    t0 = time.perf_counter()
    print(f"[all] K_GRID={K_GRID} from full pools; endpoint = all days. "
          f"α(MR)={ALPHA_PRIMARY:g}/{ALPHA_SENS:g}, α(P24)={ALPHA_PRIMARY:g}",
          flush=True)

    Hydra, SparseScaler, mr_fit, mr_transform = rt.vendor_imports()
    users, y, folds, da, dpu, starts, hr_m, mask_m = load_all()
    n = len(users)
    print(f"[all] loaded: {n} users, {int(dpu.sum()):,} pooled day-rows "
          f"(mean {dpu.mean():.0f}/user)", flush=True)

    # fold splits (v2 folds, reused)
    fold_of = {}
    for r in range(10):
        f = (folds[folds.alloc == r].set_index("user_id")["fold"].astype(str))
        f.index = f.index.astype("int64")
        fold_of[r] = f.reindex(users).to_numpy()

    rows = []

    # ================= Phase A: fixed-k curve =================
    for r in range(10):
        wall_a = time.perf_counter()
        fold = fold_of[r]
        tr = np.flatnonzero(fold == "train")
        va = np.flatnonzero(fold == "val")
        te = np.flatnonzero(fold == "test")
        y_tr, y_va, y_te = y[tr], y[va], y[te]

        # nested per-user permutation of the full pool → first K_MAX days
        perm_rng = np.random.default_rng(rt.comp_seed(r, COMP_PERM))
        perm = np.stack([np.argsort(perm_rng.random(int(dpu[u])))
                         [:K_MAX].astype(np.int64) for u in range(n)])
        rows_idx = starts[:, None] + perm                    # (n, K_MAX) global

        # gather sampled windows into RAM (memmap reads)
        hr_K = np.empty((n, K_MAX, rt.N_BINS), dtype=np.float32)
        mask_K = np.empty((n, K_MAX, rt.N_BINS), dtype=bool)
        for u in range(n):
            hr_K[u] = hr_m[rows_idx[u]]
            mask_K[u] = mask_m[rows_idx[u]]

        # per-alloc train-only fill from the sampled windows
        fill_vec = fill_from_window(hr_K, mask_K, tr)
        hr_filled_K = np.where(mask_K, hr_K, fill_vec[None, None, :]
                               ).astype(np.float32)

        # MR fit: 4 seeded days within each train user's 80-day window
        fit_rng = np.random.default_rng(rt.comp_seed(r, rt.COMP_DAY_SELECT))
        fit_days = np.stack([fit_rng.choice(K_MAX, 4, replace=False)
                             for _ in range(len(tr))])
        X_fit = hr_filled_K[tr[:, None], fit_days].reshape(
            -1, rt.N_BINS).astype(np.float64)
        X1_fit = np.diff(X_fit, axis=1)
        rt._nb_seed(rt.comp_seed(r, rt.COMP_MR_FIT))
        mr_pb = mr_fit(X_fit, num_features=rt.MR_NUM_FEATURES_PER_TRANS,
                        max_dilations_per_kernel=32)
        rt._nb_seed(rt.comp_seed(r, rt.COMP_MR_FIT) + 1)
        mr_pd = mr_fit(X1_fit, num_features=rt.MR_NUM_FEATURES_PER_TRANS,
                        max_dilations_per_kernel=32)

        # transform + pool prefixes, chunked
        Nk = len(K_GRID)
        pooled_mean = np.zeros((n, Nk, rt.MR_WIDTH), dtype=np.float32)
        pooled_sd = np.zeros((n, Nk, rt.MR_WIDTH), dtype=np.float32)
        for s in range(0, n, rt.CHUNK_USERS):
            sz = min(rt.CHUNK_USERS, n - s)
            X = hr_filled_K[s:s + sz].reshape(
                sz * K_MAX, rt.N_BINS).astype(np.float64)
            X1 = np.diff(X, axis=1)
            F = mr_transform(X, X1, mr_pb, mr_pd, n_features_per_kernel=4)
            F = np.nan_to_num(F).astype(np.float32).reshape(
                sz, K_MAX, rt.MR_WIDTH)
            obs = mask_K[s:s + sz].any(axis=2)
            F = np.where(obs[:, :, None], F, np.float32(np.nan))
            pool_prefixes(F, obs, pooled_mean, pooled_sd, s)
            del F

        for ki, k in enumerate(K_GRID):
            da_k = da.iloc[(starts[:, None] + perm[:, :k]).ravel()]
            B40_k, P24_k = build_static_rows(users, da_k, mask_K[:, :k, :])
            rows.extend(fit_models(
                k, r, B40_k, P24_k, pooled_mean[:, ki, :], pooled_sd[:, ki, :],
                tr, va, te, y_tr, y_va, y_te))

        print(f"[all] alloc {r} done in {time.perf_counter()-wall_a:.1f}s | "
              f"peak RSS {rt.peak_rss_mb():.0f} MiB", flush=True)

    # ================= Phase B: all-days endpoint =================
    print("[all] endpoint: single transform pass over full pools...", flush=True)
    wall_b = time.perf_counter()

    # alloc-0 fill (reuse the alloc-0 permutation's windows)
    fold0 = fold_of[0]
    tr0 = np.flatnonzero(fold0 == "train")
    perm0_rng = np.random.default_rng(rt.comp_seed(0, COMP_PERM))
    perm0 = np.stack([np.argsort(perm0_rng.random(int(dpu[u])))
                      [:K_MAX].astype(np.int64) for u in range(n)])
    hr_K0 = np.stack([hr_m[starts[u] + perm0[u]] for u in range(n)])
    mask_K0 = np.stack([mask_m[starts[u] + perm0[u]] for u in range(n)])
    fill_vec0 = fill_from_window(hr_K0, mask_K0, tr0)
    del hr_K0, mask_K0

    # endpoint kernels: 4 seeded days from alloc-0 train users' FULL pools
    efit_rng = np.random.default_rng(rt.comp_seed(0, COMP_ENDPOINT))
    efit_days = np.stack([efit_rng.choice(int(dpu[u]), 4, replace=False)
                          for u in tr0])
    X_fit = np.concatenate([hr_m[starts[u] + efit_days[i]]
                            for i, u in enumerate(tr0)]).astype(np.float64)
    X_fit = np.where(np.concatenate([mask_m[starts[u] + efit_days[i]]
                                     for i, u in enumerate(tr0)]),
                     X_fit, fill_vec0[None, :].astype(np.float64))
    X1_fit = np.diff(X_fit, axis=1)
    rt._nb_seed(rt.comp_seed(0, COMP_ENDPOINT) + 1)
    e_pb = mr_fit(X_fit, num_features=rt.MR_NUM_FEATURES_PER_TRANS,
                   max_dilations_per_kernel=32)
    rt._nb_seed(rt.comp_seed(0, COMP_ENDPOINT) + 2)
    e_pd = mr_fit(X1_fit, num_features=rt.MR_NUM_FEATURES_PER_TRANS,
                   max_dilations_per_kernel=32)

    # per-user full-pool transform + pool (sequential; ~0.1 s/user)
    ep_mean = np.zeros((n, rt.MR_WIDTH), dtype=np.float32)
    ep_sd = np.zeros((n, rt.MR_WIDTH), dtype=np.float32)
    for u in range(n):
        rows_u = starts[u] + np.arange(int(dpu[u]))
        hr_u = np.asarray(hr_m[rows_u])
        mask_u = np.asarray(mask_m[rows_u])
        X = np.where(mask_u, hr_u, fill_vec0[None, :]).astype(np.float64)
        X1 = np.diff(X, axis=1)
        F = mr_transform(X, X1, e_pb, e_pd, n_features_per_kernel=4)
        F = np.nan_to_num(F).astype(np.float32)
        obs = mask_u.any(axis=1)
        F = np.where(obs[:, None], F, np.float32(np.nan))
        with np.errstate(invalid="ignore", all="ignore"):
            ep_mean[u] = np.nan_to_num(
                np.nanmean(F, axis=0, dtype=np.float64), nan=0.0).astype(np.float32)
            ep_sd[u] = np.nan_to_num(
                np.nanstd(F, axis=0, ddof=1, dtype=np.float64), nan=0.0).astype(np.float32)

    # full-pool statics
    mask_means_full = np.zeros((n, rt.N_BINS), dtype=np.float32)
    for u in range(n):
        mask_means_full[u] = np.asarray(
            mask_m[starts[u]:starts[u] + dpu[u]]).mean(axis=0)
    g = da.groupby("user", sort=True)
    means = g[STAT_COLS].mean().reindex(users).to_numpy(dtype=np.float32)
    sds = g[STAT_COLS].std(ddof=1).reindex(users).to_numpy(dtype=np.float32)
    u_pos = pd.Series(np.arange(n), index=users)
    g_pos = da.user.map(u_pos).to_numpy()
    is_we = da.d_is_weekend.to_numpy().astype(bool)
    ch_mean = da.d_ch3000_mean.to_numpy(dtype=np.float32)
    wd_sum = np.bincount(g_pos[~is_we], weights=ch_mean[~is_we], minlength=n)
    wd_cnt = np.bincount(g_pos[~is_we], minlength=n)
    we_sum = np.bincount(g_pos[is_we], weights=ch_mean[is_we], minlength=n)
    we_cnt = np.bincount(g_pos[is_we], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        wd_mean = np.where(wd_cnt > 0, wd_sum / wd_cnt, np.nan)
        we_mean = np.where(we_cnt > 0, we_sum / we_cnt, np.nan)
    B40_full = np.concatenate(
        [means, sds, (wd_mean - we_mean).astype(np.float32)[:, None],
         mask_means_full], axis=1)
    p24_mean = g[HOUR_COLS].mean().reindex(users).to_numpy(dtype=np.float64)
    p24_sd = g[HOUR_COLS].std(ddof=1).reindex(users).to_numpy(dtype=np.float64)
    P24_full = np.concatenate([p24_mean, p24_sd], axis=1)

    # evaluate the fixed features under all 10 fold-splits
    for r in range(10):
        fold = fold_of[r]
        tr = np.flatnonzero(fold == "train")
        va = np.flatnonzero(fold == "val")
        te = np.flatnonzero(fold == "test")
        rows.extend(fit_models(
            -1, r, B40_full, P24_full, ep_mean, ep_sd,
            tr, va, te, y[tr], y[va], y[te]))

    print(f"[all] endpoint done in {time.perf_counter()-wall_b:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS / "days_scaling_all_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\n[all] wrote {out} | total wall {time.perf_counter()-t0:.1f}s",
          flush=True)

    gv = df.groupby(["k", "arm"]).auroc_val.agg(["mean", "std"])
    gt = df.groupby(["k", "arm"]).auroc_test.agg(["mean", "std"])
    print("\n--- val AUROC (mean ± SD) ---")
    print(gv.to_string())
    print("\n--- test AUROC (mean ± SD) ---")
    print(gt.to_string())


if __name__ == "__main__":
    main()
