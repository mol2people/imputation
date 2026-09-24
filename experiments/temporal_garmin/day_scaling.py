#!/usr/bin/env python
"""Days-scaling check: AUROC vs k adequate days for MultiRocket, Profile24, Summary_RF.

k grid: 4, 7, 14, 21, 30 (k=40 anchor cited from v2 `temporal_metrics.csv`).

Design (deliberate deviations from the frozen v2 protocol, recorded):
  1. MR kernels are FIT ONCE PER ALLOC on 4 seeded days drawn from the first
     K_MAX=30 days of training users (same seed stream as v2:
     `comp_seed(r, COMP_DAY_SELECT/MR_FIT)`). Kernels are then held fixed across
     all k, so the k-axis isolates the pooling window (not kernel-sampling
     noise). Mild unsupervised "future-day" usage for k<30 (train users only).
  2. No HYDRA, no Shuffled_MR (cost-control for a quick check).
  3. α frozen at the v2 value (1e3) for Profile24 and MultiRocket across all k
     (matches the v2 anchor at k=40). MR is also reported at α=3e3 (extended-
     grid optimum at k=40, `results/alpha_addendum.{md,json}`) as a shrinkage
     sensitivity column.
  4. Fill: train-only clock-bin medians from the first K_MAX=30 days
     (vs per-k-window fill — secondary, small effect on observed-bin fills).
  5. B40_k and P24_k are recomputed on the first k adequate days per user.

Outputs: `results/days_scaling_metrics.csv`, `results/days_scaling.md`.
"""
from __future__ import annotations

import os
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
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

K_GRID = (4, 7, 14, 21, 30)
K_MAX = max(K_GRID)
ALPHA_PRIMARY = 1e3        # v2 frozen (MultiRocket, Profile24)
ALPHA_SENS = 3e3           # extended-grid MR optimum at k=40 (alpha_addendum)
RESULTS = rt.RESULTS


def build_static_k(users, d40, mask, k):
    """B40_k (305) + P24_extra_k (48) on the first k adequate days per user.

    d40 is sorted (user, date) stable with 40 rows per user in load_data();
    first k rows per user = first k adequate days.
    """
    n = len(users)
    starts = np.arange(n) * rt.DAYS
    rows = (starts[:, None] + np.arange(k)[None, :]).ravel()
    d = d40.iloc[rows]
    g = d.groupby("user", sort=True)
    stat_cols = [f"d_ch3000_{s}" for s in
                 ("mean", "median", "sd", "vmin", "vmax", "n", "cov_h", "hours")]
    means = g[stat_cols].mean().reindex(users).to_numpy(dtype=np.float32)
    sds = g[stat_cols].std(ddof=1).reindex(users).to_numpy(dtype=np.float32)

    # weekend − weekday mean HR (vectorized bincount)
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

    mask_k = mask.reshape(n, rt.DAYS, rt.N_BINS)[:, :k, :]
    mask_means = mask_k.mean(axis=1).astype(np.float32)
    B40_k = np.concatenate([means, sds, weekend_diff, mask_means], axis=1)
    assert B40_k.shape == (n, 305)

    hour_cols = [f"hour_h{h}" for h in range(24)]
    p24_mean = g[hour_cols].mean().reindex(users).to_numpy(dtype=np.float64)
    p24_sd = g[hour_cols].std(ddof=1).reindex(users).to_numpy(dtype=np.float64)
    P24_k = np.concatenate([p24_mean, p24_sd], axis=1).astype(np.float32)
    return B40_k, P24_k


def apply_mr_pooled_multi_k(mr_transform, params_base, params_diff,
                              hr_filled_K, mask_pu_K):
    """Pool MR features (mean + ddof=1 SD) over the first k days for each k in K_GRID.

    hr_filled_K (n, K_MAX, 288) float32; mask_pu_K (n, K_MAX, 288) bool.
    Returns pooled_mean (n, len(K_GRID), MR_W) and pooled_sd (same) ordered by K_GRID.
    """
    n, K, L = hr_filled_K.shape
    assert K == K_MAX
    Nk = len(K_GRID)
    pooled_mean = np.zeros((n, Nk, rt.MR_WIDTH), dtype=np.float32)
    pooled_sd = np.zeros((n, Nk, rt.MR_WIDTH), dtype=np.float32)
    for s in range(0, n, rt.CHUNK_USERS):
        sz = min(rt.CHUNK_USERS, n - s)
        X = hr_filled_K[s:s + sz].reshape(sz * K, L).astype(np.float64)
        X1 = np.diff(X, axis=1)
        F = mr_transform(X, X1, params_base, params_diff,
                          n_features_per_kernel=4)
        F = np.nan_to_num(F).astype(np.float32).reshape(sz, K, rt.MR_WIDTH)
        obs = mask_pu_K[s:s + sz].any(axis=2)               # (sz, K)
        F = np.where(obs[:, :, None], F, np.float32(np.nan))
        with np.errstate(invalid="ignore", all="ignore"):
            for ki, k in enumerate(K_GRID):
                Fk = F[:, :k, :]
                pooled_mean[s:s + sz, ki] = np.nan_to_num(
                    np.nanmean(Fk, axis=1, dtype=np.float64),
                    nan=0.0).astype(np.float32)
                pooled_sd[s:s + sz, ki] = np.nan_to_num(
                    np.nanstd(Fk, axis=1, ddof=1, dtype=np.float64),
                    nan=0.0).astype(np.float32)
        del F
    return pooled_mean, pooled_sd


def main():
    t0 = time.perf_counter()
    print(f"[scan] K_GRID={K_GRID}, K_MAX={K_MAX}, α(MR primary)={ALPHA_PRIMARY:g}, "
          f"α(MR sens)={ALPHA_SENS:g}, α(P24)={ALPHA_PRIMARY:g}", flush=True)

    Hydra, SparseScaler, mr_fit, mr_transform = rt.vendor_imports()
    users, y, d40, hr, mask, cnt, pu, pd_, folds, r1b = rt.load_data()
    # d40 user-major layout (verified at scan time; reassert for safety)
    d40_sorted = d40.sort_values(["user", "date"], kind="stable").reset_index(drop=True)
    assert len(d40_sorted) == rt.COHORT_SIZE * rt.DAYS
    assert np.array_equal(d40_sorted.user.to_numpy(),
                          np.repeat(np.sort(np.unique(pu)), rt.DAYS))
    d40 = d40_sorted

    n = len(users)
    hr_raw = hr.reshape(n, rt.DAYS, rt.N_BINS)
    mask_pu = mask.reshape(n, rt.DAYS, rt.N_BINS)
    hr_raw_K = hr_raw[:, :K_MAX].astype(np.float32)
    mask_pu_K = mask_pu[:, :K_MAX]

    # alloc-independent B40_k, P24_k
    print("[scan] building per-k static features...", flush=True)
    stat_k = {k: build_static_k(users, d40, mask, k) for k in K_GRID}

    rows = []
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

        # fill (first K_MAX days; train-only)
        hr_filled_K, _ = rt.fill_hr(hr_raw_K, mask_pu_K, tr)

        # MR fit: 4 seeded days from the first K_MAX days of train users
        fit_rng = np.random.default_rng(rt.comp_seed(r, rt.COMP_DAY_SELECT))
        fit_days = np.stack(
            [fit_rng.choice(K_MAX, 4, replace=False) for _ in range(len(tr))])
        X_fit = hr_filled_K[tr[:, None], fit_days].reshape(-1, rt.N_BINS).astype(np.float64)
        X1_fit = np.diff(X_fit, axis=1)
        rt._nb_seed(rt.comp_seed(r, rt.COMP_MR_FIT))
        mr_pb = mr_fit(X_fit, num_features=rt.MR_NUM_FEATURES_PER_TRANS,
                        max_dilations_per_kernel=32)
        rt._nb_seed(rt.comp_seed(r, rt.COMP_MR_FIT) + 1)
        mr_pd = mr_fit(X1_fit, num_features=rt.MR_NUM_FEATURES_PER_TRANS,
                        max_dilations_per_kernel=32)

        # MR transform (first K_MAX days of all users), pool per k
        mr_mean_all, mr_sd_all = apply_mr_pooled_multi_k(
            mr_transform, mr_pb, mr_pd, hr_filled_K, mask_pu_K)

        for ki, k in enumerate(K_GRID):
            B40_k = stat_k[k][0]
            P24_k = stat_k[k][1]
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

            # MultiRocket — primary α=1e3, sensitivity α=3e3
            for tag, alpha in (("primary", ALPHA_PRIMARY), ("sens3e3", ALPHA_SENS)):
                m = Ridge(alpha=alpha, fit_intercept=True).fit(MRARM_tr, y_tr)
                rows.append({
                    "alloc": r, "k": k, "arm": f"MultiRocket_{tag}",
                    "alpha": float(alpha),
                    "auroc_val": float(roc_auc_score(y_va, m.predict(MRARM_va))),
                    "auroc_test": float(roc_auc_score(y_te, m.predict(MRARM_te))),
                    "n_cols": mr_n_cols,
                })
            # Profile24
            Z_tr = np.concatenate([B40Z_tr, P24Z_tr], axis=1)
            Z_va = np.concatenate([B40Z_va, P24Z_va], axis=1)
            Z_te = np.concatenate([B40Z_te, P24Z_te], axis=1)
            m = Ridge(alpha=ALPHA_PRIMARY, fit_intercept=True).fit(Z_tr, y_tr)
            rows.append({
                "alloc": r, "k": k, "arm": "Profile24",
                "alpha": float(ALPHA_PRIMARY),
                "auroc_val": float(roc_auc_score(y_va, m.predict(Z_va))),
                "auroc_test": float(roc_auc_score(y_te, m.predict(Z_te))),
                "n_cols": int(Z_tr.shape[1]),
            })
            # Summary_RF on B40 std block (RF G1, same seed stream as v2)
            rf = RandomForestClassifier(n_estimators=rt.N_TREES,
                                         random_state=rt.comp_seed(r, rt.COMP_RF),
                                         n_jobs=1, **rt.G1).fit(B40Z_tr, y_tr)
            j1 = list(rf.classes_).index(1)
            rows.append({
                "alloc": r, "k": k, "arm": "Summary_RF",
                "alpha": None,
                "auroc_val": float(roc_auc_score(
                    y_va, rf.predict_proba(B40Z_va)[:, j1])),
                "auroc_test": float(roc_auc_score(
                    y_te, rf.predict_proba(B40Z_te)[:, j1])),
                "n_cols": int(B40Z_tr.shape[1]),
            })

        wall = time.perf_counter() - wall_a
        print(f"[scan] alloc {r} done in {wall:.1f}s | "
              f"peak RSS {rt.peak_rss_mb():.0f} MiB", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS / "days_scaling_metrics.csv"
    df.to_csv(out, index=False)

    # v2 k=40 anchor (cited, not recomputed)
    v2 = pd.read_csv(RESULTS / "temporal_metrics.csv")
    anchor_rows = []
    for arm in ("Summary_RF", "Profile24", "MultiRocket"):
        s = v2[v2.arm == arm].sort_values("alloc")
        anchor_rows.append({
            "alloc": "v2", "k": 40, "arm": arm, "alpha": float(s.alpha.iloc[0])
                if not pd.isna(s.alpha.iloc[0]) else None,
            "auroc_val_mean": float(s.auroc_val.mean()),
            "auroc_val_sd": float(s.auroc_val.std(ddof=1)),
            "auroc_test_mean": float(s.auroc_test.mean()),
            "auroc_test_sd": float(s.auroc_test.std(ddof=1)),
            "n_cols_mean": float(s.n_cols.mean()),
        })

    # ---- summary table ----
    print(f"\n[scan] wrote {out}", flush=True)
    print(f"[scan] total wall {time.perf_counter()-t0:.1f}s", flush=True)
    print("\n--- val AUROC (mean ± SD, 10 allocs) ---", flush=True)
    gv = df.groupby(["k", "arm"]).auroc_val.agg(["mean", "std"])
    print(gv.to_string(), flush=True)
    print("\n--- test AUROC (mean ± SD, 10 allocs) ---", flush=True)
    gt = df.groupby(["k", "arm"]).auroc_test.agg(["mean", "std"])
    print(gt.to_string(), flush=True)

    # ---- markdown report ----
    md = ["# Days-scaling check — AUROC vs k adequate days\n",
          "MultiRocket / Profile24 / Summary_RF, 10 allocations. k=40 anchor cited "
          "from the frozen v2 run (`results/temporal_metrics.csv`).\n",
          "**Design** (`day_scaling.py`): MR kernels fit once per allocation on 4 "
          "seeded days drawn from the first K_MAX=30 adequate days of training "
          "users (same seed stream as v2); kernels held fixed across all k. "
          "α frozen at the v2 value (1e3) for Profile24 and MultiRocket_primary; "
          "MultiRocket_sens3e3 = sensitivity at α=3e3 (extended-grid optimum at "
          "k=40). No HYDRA, no Shuffled_MR. Fill: train-only clock-bin medians "
          "from the first 30 days. B40_k / P24_k recomputed on the first k "
          "adequate days per user.\n",
          "**Caveat:** the k=4 → k=30 trajectory isolates the pooling-window "
          "effect (kernels held fixed). For a fully deployment-realistic run "
          "(kernels refit per k), kernel-sampling noise would add to the curve.\n"]
    md.append("\n## Validation AUROC (mean ± SD, 10 allocs)\n")
    md.append("| k | arm | val AUROC | test AUROC | n cols | α |\n|---|---|---|---|---|---|\n")
    for (k, arm), row in gv.iterrows():
        sub = df[(df.k == k) & (df.arm == arm)]
        a_val = sub.alpha.iloc[0]
        a_str = "—" if pd.isna(a_val) else f"{a_val:g}"
        n_cols = int(sub.n_cols.iloc[0])
        row_t = gt.loc[(k, arm)]
        md.append(f"| {k} | `{arm}` | "
                  f"{row['mean']:.4f} ± {row['std']:.4f} | "
                  f"{row_t['mean']:.4f} ± {row_t['std']:.4f} | "
                  f"{n_cols:,} | {a_str} |\n")
    md.append("\n## v2 k=40 anchor (10 allocs, frozen α=1e3)\n")
    md.append("| k | arm | val AUROC | test AUROC | α |\n|---|---|---|---|---|\n")
    for a in anchor_rows:
        a_str = "—" if a["alpha"] is None else f"{a['alpha']:g}"
        md.append(f"| 40 | `{a['arm']}` | "
                  f"{a['auroc_val_mean']:.4f} ± {a['auroc_val_sd']:.4f} | "
                  f"{a['auroc_test_mean']:.4f} ± {a['auroc_test_sd']:.4f} | "
                  f"{a_str} |\n")
    md.append("\n**Note:** the MultiRocket v2 anchor at α=1e3 corresponds to "
              "`MultiRocket_primary` in this check (same α, same protocol minus "
              "the kernel-sharing/k-window design noted above).\n")
    md_path = RESULTS / "days_scaling.md"
    md_path.write_text("".join(md))
    print(f"[scan] wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
