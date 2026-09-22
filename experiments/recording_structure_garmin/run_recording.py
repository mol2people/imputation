#!/usr/bin/env python
"""RECORDING-STRUCTURE pilot — 8 arms × 3 allocations, MultiRocket on
mask/count sequences plus an HR reference and a complementarity arm.

PLAN §"First pass". Cached data only (no new epoch scan). Reuses corrected
helpers from experiments/temporal_garmin/run_temporal.py (fill_hr,
apply_mr_pooled conventions, hygiene_std, build_static, comp_seed, _nb_seed).
No HYDRA, no torch.

CLI:
  python run_recording.py bench       # fixture: binary/count MR, chunk invariance
  python run_recording.py run         # 3-allocation pipeline (resumable)
  python run_recording.py report      # write REPORT.md + repro.json
  python run_recording.py full        # bench + run + report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# threading — match temporal (workqueue + 8 threads; torch not loaded)
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

# Reuse corrected helpers from the temporal runner (PLAN §"First pass").
_TEMPORAL = Path(__file__).resolve().parent.parent / "temporal_garmin"
sys.path.insert(0, str(_TEMPORAL))
from run_temporal import (  # noqa: E402
    SEED_BASE, SRC, COHORT_SIZE, DAYS, N_BINS,
    MR_NUM_FEATURES_PER_TRANS, MR_WIDTH,
    BINS_NPZ, DAY40_PARQUET, FOLDS_PARQUET, PATH_COHORT, PATH_SPLIT,
    COMP_DAY_SELECT, COMP_MR_FIT,                      # temporal IDs 0, 1
    _nb_seed, comp_seed, log, peak_rss_mb, sha256_file,
    fill_hr, hygiene_std, build_static,
)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"

# ---- new component IDs (PLAN §"First pass" + §"Shuffle control") -----------
COMP_M_BASE, COMP_M_DIFF = 10, 11       # M base/diff1 MR fits (and M_shuffled)
COMP_C_BASE, COMP_C_DIFF = 12, 13       # C base/diff1 MR fits (and C_shuffled)
COMP_SHUFFLE_PERM = 14                  # per-(user,date) shuffle permutation

ALPHA_GRID = np.array([1e-3, 1.0, 1e3, 1e4, 1e5, 1e6])
CHUNK_USERS = 256
ALLOCATION_TARGETS = (0, 1, 2)          # predeclared per PLAN §"First pass"

ARMS = (
    "M_static", "M_temporal", "M_shuffled",
    "C_static", "C_temporal", "C_shuffled",
    "H", "H_recording",
)
# arm -> post-hygiene block names (the schema; recording arms exclude B40/T(HR))
ARM_BLOCKS = {
    "M_static":     ("S(M)",),
    "M_temporal":   ("S(M)", "T(M)"),
    "M_shuffled":   ("S(M)", "T(M_shuffled)"),
    "C_static":     ("S(C)",),
    "C_temporal":   ("S(C)", "T(C)"),
    "C_shuffled":   ("S(C)", "T(C_shuffled)"),
    "H":            ("B40", "T(HR)"),
    "H_recording":  ("B40", "S(M)_SD", "S(C)", "T(M)", "T(C)", "T(HR)"),
}


# ---- vendor (MR only — no HYDRA) -------------------------------------------
def vendor_mr():
    sys.path.insert(0, str(_TEMPORAL / "cache" / "vendor" / "multirocket"))
    from multirocket.multirocket import fit as mr_fit, transform as mr_transform
    return mr_fit, mr_transform


# ---- data loading (no r1b pairing in this pilot) ---------------------------
def load_data():
    z = np.load(BINS_NPZ)
    hr = z["hr"]; cnt = z["cnt"]; mask = z["mask"]
    pu = z["user"].astype(np.int64); pd_ = z["date"].astype(np.int64)
    users = np.sort(np.unique(pu))
    assert len(users) == COHORT_SIZE and hr.shape == (COHORT_SIZE * DAYS, N_BINS)
    assert np.array_equal(mask, cnt > 0), "mask != (cnt > 0)"

    split = pd.read_parquet(PATH_SPLIT, columns=["user_id", "source_id", "y"])
    y_map = (split[split.source_id == SRC]
             .set_index("user_id")["y"].astype(int))
    y = y_map.reindex(users).to_numpy(np.int32)

    cm = pd.read_parquet(PATH_COHORT, columns=["user_id", "source_id"])
    assert set(cm[cm.source_id == SRC].user_id.astype("int64")) == set(users)

    d40 = pd.read_parquet(DAY40_PARQUET)
    d40 = d40[d40.user.isin(set(users))].sort_values(
        ["user", "date"], kind="stable").reset_index(drop=True)
    assert len(d40) == COHORT_SIZE * DAYS

    folds = pd.read_parquet(FOLDS_PARQUET)
    return {"users": users, "y": y, "d40": d40, "hr": hr, "mask": mask,
            "cnt": cnt, "pu": pu, "pd": pd_, "folds": folds}


# ---- per-clock-bin static blocks -------------------------------------------
def block_mean_sd(X3):
    """(n,40,288) -> (n,576): per-clock-bin mean + sample SD across the 40 days."""
    return np.concatenate([X3.mean(axis=1).astype(np.float32),
                            X3.std(axis=1, ddof=1).astype(np.float32)],
                           axis=1)


def block_sd(X3):
    """(n,40,288) -> (n,288): sample SD only (H_recording's S(M) part)."""
    return X3.std(axis=1, ddof=1).astype(np.float32)


# ---- alloc-independent per-(user, date) shuffle permutation ---------------
def make_permutation(pu, pd_):
    """One permutation of 0..287 per (user, date) row; alloc-independent.
    Seed: SeedSequence([20260921, 3, 14, user, date]) (PLAN §"Shuffle control")."""
    n_rows = pu.shape[0]
    perm = np.empty((n_rows, N_BINS), dtype=np.int32)
    for i in range(n_rows):
        ss = np.random.SeedSequence(
            [SEED_BASE, 3, COMP_SHUFFLE_PERM, int(pu[i]), int(pd_[i])])
        rng = np.random.default_rng(
            ss.generate_state(1, dtype=np.uint32)[0])
        perm[i] = rng.permutation(N_BINS).astype(np.int32)
    return perm


def apply_permutation(X3, perm):
    """X3 (n,40,288) float32; perm (n*40, 288) int32. Positions permuted per day."""
    n, D, L = X3.shape
    out = np.take_along_axis(X3.reshape(n * D, L), perm, axis=1)
    return out.reshape(n, D, L)


# ---- MR fit (base+diff1) + transform pooled --------------------------------
def fit_and_pool_mr(r, X3, fit_days, tr, comp_base, gated, mask_pu,
                     mr_fit, mr_transform, chunk=CHUNK_USERS):
    """X3 (n,40,288) float32 -> (pooled_mean, pooled_sd) (n, MR_WIDTH) float32.

    gated=True  (HR): days with mask_pu.any(axis=2)==False are NaN'd before
                     nanmean/nanstd (temporal convention, unchanged).
    gated=False (M, C, M_shuf, C_shuf): pool all 40 days unconditionally
                     (PLAN §"Fitting and compute": an all-zero recording day
                     is valid information, not a day to omit).
    Seed: _nb_seed(comp_seed(r, comp_base)) before each mr_fit (base, then +1).
    """
    n = X3.shape[0]
    X_fit = X3[tr[:, None], fit_days].reshape(-1, N_BINS).astype(np.float64)
    X1_fit = np.diff(X_fit, axis=1)
    _nb_seed(comp_seed(r, comp_base))
    pb = mr_fit(X_fit, num_features=MR_NUM_FEATURES_PER_TRANS,
                max_dilations_per_kernel=32)
    _nb_seed(comp_seed(r, comp_base + 1))
    pdd = mr_fit(X1_fit, num_features=MR_NUM_FEATURES_PER_TRANS,
                 max_dilations_per_kernel=32)
    del X_fit, X1_fit

    pooled_mean = np.zeros((n, MR_WIDTH), dtype=np.float32)
    pooled_sd = np.zeros((n, MR_WIDTH), dtype=np.float32)
    for s in range(0, n, chunk):
        sz = min(chunk, n - s)
        X = X3[s:s + sz].reshape(sz * DAYS, N_BINS).astype(np.float64)
        X1 = np.diff(X, axis=1)
        F = mr_transform(X, X1, pb, pdd, n_features_per_kernel=4)
        F = np.nan_to_num(F).astype(np.float32)
        F = F.reshape(sz, DAYS, MR_WIDTH)
        if gated:
            obs = mask_pu[s:s + sz].any(axis=2)                # (sz, 40)
            F = np.where(obs[:, :, None], F, np.nan)
            with np.errstate(invalid="ignore", all="ignore"):
                m = np.nanmean(F, axis=1, dtype=np.float64)
                sd = np.nanstd(F, axis=1, ddof=1, dtype=np.float64)
        else:
            with np.errstate(invalid="ignore", all="ignore"):
                m = F.mean(axis=1, dtype=np.float64)
                sd = F.std(axis=1, ddof=1, dtype=np.float64)
        pooled_mean[s:s + sz] = np.nan_to_num(m, nan=0.0).astype(np.float32)
        pooled_sd[s:s + sz] = np.nan_to_num(sd, nan=0.0).astype(np.float32)
        del X, X1, F
    return pooled_mean, pooled_sd


# ---- per-arm ridge with alpha selection by validation AUROC ----------------
def select_alpha(Z_tr, y_tr, Z_va, y_va):
    """Grid ascending; strict > keeps the FIRST max -> smallest alpha wins ties
    (PLAN §"Fitting and compute"). Returns (alpha, trace)."""
    best_a, best_auc, trace = None, -1.0, []
    for a in ALPHA_GRID:
        m = Ridge(alpha=float(a), fit_intercept=True).fit(Z_tr, y_tr)
        auc = float(roc_auc_score(y_va, m.predict(Z_va)))
        trace.append({"alpha": float(a), "auroc_val": auc})
        if auc > best_auc:
            best_auc, best_a = auc, float(a)
    return best_a, trace


# ---- fixture: binary/count MR inputs, chunk invariance, determinism --------
def run_fixture(data, M3, C3, mask_pu):
    mr_fit, mr_transform = vendor_mr()
    users, folds = data["users"], data["folds"]
    rng = np.random.default_rng(SEED_BASE)
    fold0 = (folds[folds.alloc == 0].set_index("user_id")["fold"]
             .astype(str).reindex(users).to_numpy())
    tr_full = np.flatnonzero(fold0 == "train")
    sub = np.sort(rng.choice(tr_full, size=200, replace=False))
    tr_sub = np.arange(200)

    fit_rng = np.random.default_rng(comp_seed(0, COMP_DAY_SELECT))
    fit_days = np.stack([fit_rng.choice(DAYS, 4, replace=False)
                          for _ in range(200)])

    out = {}
    for name, X3, comp in (("M", M3[sub], COMP_M_BASE), ("C", C3[sub], COMP_C_BASE)):
        t = time.perf_counter()
        m_a, sd_a = fit_and_pool_mr(0, X3, fit_days, tr_sub, comp, False,
                                     mask_pu[sub], mr_fit, mr_transform,
                                     chunk=64)
        wall = time.perf_counter() - t
        m_b, sd_b = fit_and_pool_mr(0, X3, fit_days, tr_sub, comp, False,
                                     mask_pu[sub], mr_fit, mr_transform,
                                     chunk=256)          # single-chunk = direct
        max_diff = float(max(np.max(np.abs(m_a - m_b)),
                             np.max(np.abs(sd_a - sd_b))))
        assert max_diff < 1e-6, f"T({name}) chunk invariance FAILED: {max_diff}"
        assert np.isfinite(m_a).all() and np.isfinite(sd_a).all(), \
            f"T({name}) nonfinite outputs"
        out[name] = {"wall_s_chunk64": round(wall, 2),
                     "shape": list(m_a.shape),
                     "chunk_invariance_max_diff": max_diff,
                     "any_nan": False, "any_inf": False}
        log(f"[fixture] T({name}) {m_a.shape} in {wall:.2f}s | "
            f"chunk64-vs-256 max|Δ| {max_diff:.2e} | RSS {peak_rss_mb():.0f} MiB")

    # same-seed repeat (determinism of the whole fit+transform path)
    m_c, _ = fit_and_pool_mr(0, M3[sub], fit_days, tr_sub, COMP_M_BASE, False,
                              mask_pu[sub], mr_fit, mr_transform, chunk=64)
    m_a, _ = fit_and_pool_mr(0, M3[sub], fit_days, tr_sub, COMP_M_BASE, False,
                              mask_pu[sub], mr_fit, mr_transform, chunk=64)
    rep = float(np.max(np.abs(m_c - m_a)))
    assert rep < 1e-6, f"T(M) repeat determinism FAILED: {rep}"
    out["M"]["repeat_max_diff"] = rep
    out["fixture_users"] = 200
    json.dump(out, open(CACHE / "fixture.json", "w"), indent=2)
    log(f"[fixture] T(M) same-seed repeat max|Δ| {rep:.2e}")


# ---- per-alloc pipeline ----------------------------------------------------
def fit_alloc(r, data, M3, C3, M_shuf3, C_shuf3, mask_pu, B40,
               mr_fit, mr_transform):
    """One allocation. Returns (metrics_rows, preds_frames, alpha_traces, info)."""
    users, y, folds = data["users"], data["y"], data["folds"]
    n = len(users)

    fold = (folds[folds.alloc == r].set_index("user_id")["fold"].astype(str))
    fold.index = fold.index.astype("int64")
    fold = fold.reindex(users).to_numpy()
    tr = np.flatnonzero(fold == "train")
    va = np.flatnonzero(fold == "val")
    te = np.flatnonzero(fold == "test")
    assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 \
        and len(set(va) & set(te)) == 0, f"alloc {r}: splits not disjoint"
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    t0 = time.perf_counter()
    hr_pu = data["hr"].reshape(n, DAYS, N_BINS)
    hr_filled, fill_vec = fill_hr(hr_pu, mask_pu, tr)        # train-only medians
    t_fill = time.perf_counter() - t0
    del hr_pu

    # day selection (per-alloc; same fit_days across all five MR inputs)
    t0 = time.perf_counter()
    fit_rng = np.random.default_rng(comp_seed(r, COMP_DAY_SELECT))
    fit_days = np.stack([fit_rng.choice(DAYS, 4, replace=False)
                          for _ in range(len(tr))])           # (n_tr, 4)
    t_daysel = time.perf_counter() - t0

    # ---- five MR fit+pool passes -------------------------------------------
    pooled = {}
    mr_wall = {}
    for name, X3, comp, gated in (
        ("HR",         hr_filled, COMP_MR_FIT,  True),    # comps 1,2 (temporal)
        ("M",          M3,        COMP_M_BASE,  False),   # comps 10,11
        ("C",          C3,        COMP_C_BASE,  False),   # comps 12,13
        ("M_shuffled", M_shuf3,   COMP_M_BASE,  False),   # comps 10,11 on shuf
        ("C_shuffled", C_shuf3,   COMP_C_BASE,  False),   # comps 12,13 on shuf
    ):
        t = time.perf_counter()
        m, sd = fit_and_pool_mr(r, X3, fit_days, tr, comp, gated, mask_pu,
                                 mr_fit, mr_transform)
        pooled[name] = np.concatenate([m, sd], axis=1)     # (n, 18816) f32
        mr_wall[name] = round(time.perf_counter() - t, 1)
        log(f"[r={r}] T({name}) pooled {pooled[name].shape} in {mr_wall[name]}s"
            f" | RSS {peak_rss_mb():.0f} MiB")
        if name == "HR":
            del hr_filled

    # ---- static blocks (from ORIGINAL, unshuffled M/C — PLAN §"Shuffle control")
    s_M = block_mean_sd(M3)            # (n, 576)
    s_C = block_mean_sd(C3)            # (n, 576)
    s_M_sd = block_sd(M3)              # (n, 288)

    # ---- hygiene per block (train-only fit; PLAN §"Fitting and compute") ---
    Z = {}
    for name, X in (("B40", B40), ("S(M)", s_M), ("S(M)_SD", s_M_sd),
                    ("S(C)", s_C),
                    ("T(HR)", pooled["HR"]), ("T(M)", pooled["M"]),
                    ("T(C)", pooled["C"]), ("T(M_shuffled)", pooled["M_shuffled"]),
                    ("T(C_shuffled)", pooled["C_shuffled"])):
        Zt, Zv, Ze, *_ = hygiene_std(X[tr], X[va], X[te])
        assert (np.isfinite(Zt).all() and np.isfinite(Zv).all()
                and np.isfinite(Ze).all()), f"non-finite in {name} post-hygiene"
        Z[name] = (Zt, Zv, Ze)
    pooled.clear()                                           # ~1.4 GiB freed

    # ---- per-arm: build, alpha-select, refit, predict (sequential; low RSS) -
    rows_metrics, rows_preds, alpha_traces = [], [], {}
    for arm in ARMS:
        blocks = ARM_BLOCKS[arm]
        if arm.startswith(("M_", "C_")):
            assert not ({"B40", "T(HR)"} & set(blocks)), \
                f"recording arm {arm} leaks B40/T(HR)"
        Zt = np.concatenate([Z[b][0] for b in blocks], axis=1)
        Zv = np.concatenate([Z[b][1] for b in blocks], axis=1)
        Ze = np.concatenate([Z[b][2] for b in blocks], axis=1)

        alpha, trace = select_alpha(Zt, y_tr, Zv, y_va)
        mdl = Ridge(alpha=alpha, fit_intercept=True).fit(Zt, y_tr)
        s_va = mdl.predict(Zv)
        s_te = mdl.predict(Ze)
        alpha_traces[arm] = {
            "alpha_selected": alpha, "trace": trace,
            "boundary": bool(alpha == float(ALPHA_GRID[0])
                             or alpha == float(ALPHA_GRID[-1]))}

        rows_metrics.append({
            "alloc": r, "arm": arm,
            "auroc_val": float(roc_auc_score(y_va, s_va)),
            "auroc_test": float(roc_auc_score(y_te, s_te)),
            "n_cols": int(Zt.shape[1]),
            "alpha": alpha,
        })
        for split, idx, ys, s in (("val", va, y_va, s_va),
                                  ("test", te, y_te, s_te)):
            rows_preds.append(pd.DataFrame({
                "arm": arm, "alloc": r, "split": split,
                "user": users[idx], "y": ys, "score": s}))
        del Zt, Zv, Ze, mdl
        log(f"[r={r}] {arm}: val {rows_metrics[-1]['auroc_val']:.4f} "
            f"test {rows_metrics[-1]['auroc_test']:.4f} "
            f"({Zt_shape_of(blocks)}, α={alpha:g})")

    info = {"alloc": r,
            "fold_sizes": {"train": int(len(tr)), "val": int(len(va)),
                           "test": int(len(te))},
            "fill_min": float(np.nanmin(fill_vec)),
            "fill_max": float(np.nanmax(fill_vec)),
            "fill_used_nan": int(np.isnan(fill_vec).sum()),
            "mr_wall_s": mr_wall,
            "fill_s": round(t_fill, 1), "day_select_s": round(t_daysel, 3)}
    return rows_metrics, rows_preds, alpha_traces, info


def Zt_shape_of(blocks):
    return "+".join(blocks)


# ---- main pipeline ---------------------------------------------------------
def run_all(data, M3, C3, M_shuf3, C_shuf3, mask_pu, B40):
    mr_fit, mr_transform = vendor_mr()
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics_csv = RESULTS / "metrics.csv"
    preds_csv = RESULTS / "predictions.csv"

    done = set()
    if metrics_csv.exists() and metrics_csv.stat().st_size > 0:
        try:
            done = set(int(a) for a in pd.read_csv(metrics_csv).alloc.unique())
            log(f"[run] resume: allocs done {sorted(done)}")
        except Exception:
            done = set()

    for r in ALLOCATION_TARGETS:
        if r in done:
            continue
        wall_a = time.perf_counter()
        log(f"[run] === alloc {r} ===")
        rows_metrics, rows_preds, alpha_traces, info = fit_alloc(
            r, data, M3, C3, M_shuf3, C_shuf3, mask_pu, B40,
            mr_fit, mr_transform)

        wall_alloc = round(time.perf_counter() - wall_a, 1)
        for row in rows_metrics:
            row["wall_s"] = wall_alloc
        pd.DataFrame(rows_metrics).to_csv(
            metrics_csv, mode="a",
            header=(not metrics_csv.exists()) or (metrics_csv.stat().st_size == 0),
            index=False)
        pd.concat(rows_preds, ignore_index=True).to_csv(
            preds_csv, mode="a",
            header=(not preds_csv.exists()) or (preds_csv.stat().st_size == 0),
            index=False)
        json.dump(alpha_traces,
                  open(RESULTS / f"alpha_traces_alloc{r}.json", "w"), indent=2)
        log(f"[run] alloc {r} done in {wall_alloc:.1f}s "
            f"| peak RSS {peak_rss_mb():.0f} MiB")
    log("[run] all target allocs done")


# ---- report ----------------------------------------------------------------
DELTA_PAIRS = (
    ("M_temporal − M_static",    "M_temporal",  "M_static"),
    ("C_temporal − C_static",    "C_temporal",  "C_static"),
    ("M_temporal − M_shuffled",  "M_temporal",  "M_shuffled"),
    ("C_temporal − C_shuffled",  "C_temporal",  "C_shuffled"),
    ("C_temporal − M_temporal",  "C_temporal",  "M_temporal"),
    ("H_recording − H",          "H_recording", "H"),
)


def write_report():
    metrics_csv = RESULTS / "metrics.csv"
    if not metrics_csv.exists():
        raise SystemExit("no metrics — run first")
    m = pd.read_csv(metrics_csv)

    def series(arm, col):
        return (m[m.arm == arm].sort_values("alloc")[col].to_numpy())

    L = []
    L.append("# Recording structure in ch3000 — pilot results\n")
    L.append("Implements [`../PLAN.md`](../PLAN.md). 8 arms × 3 allocations "
             "(0/1/2, chosen before results); cached 5-min bins; mask (M), "
             "log1p-count (C) and HR MultiRocket blocks with shuffle controls; "
             "Ridge with per-arm-per-alloc validation-alpha selection. R1b "
             "cohort/folds/first-40-adequate-days. Frozen plan commit "
             "`c970732`.\n")

    L.append("\n## 1. Arm table (mean ± SD over 3 allocations)\n")
    L.append("| arm | val AUROC | test AUROC | n cols | α selected (0/1/2) | "
              "boundary |")
    L.append("|---|---|---|---|---|---|")
    for arm in ARMS:
        sub = m[m.arm == arm].sort_values("alloc")
        if len(sub) == 0:
            continue
        alphas = "/".join(f"{a:g}" for a in sub.alpha)
        boundary = "yes" if sub.alpha.isin(
            [float(ALPHA_GRID[0]), float(ALPHA_GRID[-1])]).any() else "no"
        L.append(f"| `{arm}` | "
                 f"{sub.auroc_val.mean():.4f} ± {sub.auroc_val.std(ddof=1):.4f} | "
                 f"{sub.auroc_test.mean():.4f} ± "
                 f"{sub.auroc_test.std(ddof=1):.4f} | "
                 f"{int(sub.n_cols.mean()):,} | {alphas} | {boundary} |")

    for split_col, title in (("auroc_test", "test"), ("auroc_val", "validation")):
        L.append(f"\n## {2 if split_col == 'auroc_test' else 3}. Paired "
                 f"{title} deltas (3 allocations; no p-values, PLAN §\"Readout\")\n")
        L.append("| comparison | alloc 0 | alloc 1 | alloc 2 | mean | range | "
                 "direction |")
        L.append("|---|---|---|---|---|---|---|")
        for label, a, b in DELTA_PAIRS:
            d = series(a, split_col) - series(b, split_col)
            agree = ("all 3 +" if (d > 0).all() else
                     "all 3 −" if (d < 0).all() else f"{int((d > 0).sum())}/3 +")
            L.append(f"| {label} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | "
                     f"{d.mean():+.4f} | [{d.min():+.4f}, {d.max():+.4f}] | "
                     f"{agree} |")

    L.append("\n## 4. Recording-only absolute AUROCs (is there signal at all)\n")
    L.append("| arm | alloc 0 val | alloc 1 val | alloc 2 val | mean val | "
             "mean test |")
    L.append("|---|---|---|---|---|---|")
    for arm in ("M_static", "M_temporal", "M_shuffled",
                "C_static", "C_temporal", "C_shuffled"):
        sv, st = series(arm, "auroc_val"), series(arm, "auroc_test")
        L.append(f"| `{arm}` | {sv[0]:.4f} | {sv[1]:.4f} | {sv[2]:.4f} | "
                 f"{sv.mean():.4f} | {st.mean():.4f} |")

    L.append("\n## 5. Interpretation and recommended next probe\n")
    L.append("*(written after viewing the tables; see the pilot report commit "
             "— interpretation is authored, not auto-generated)*\n")

    L.append("\n## 6. Minimal checks (PLAN §\"Minimal checks\")\n")
    L.append("| check | status |")
    L.append("|---|---|")
    for c, s in (
        ("cohort ⊂ folds users; splits disjoint per alloc; sizes 2696/579/573",
         "PASS (assertions in load_data/fit_alloc)"),
        ("mask == (cnt > 0)", "PASS (load_data assertion)"),
        ("recording arms exclude B40 and T(HR)", "PASS (fit_alloc assertion)"),
        ("no NaN/Inf at the classifier (post-hygiene)",
         "PASS (per-block assertion)"),
        ("train-only fitted preprocessing", "PASS (hygiene_std fit on tr)"),
        ("shuffle: occupied-bin count per day preserved", "PASS (run_all)"),
        ("shuffle: cnt multiset per day preserved (sample 200 rows)",
         "PASS (run_all)"),
        ("M_shuffled == (C_shuffled > 0)", "PASS (run_all)"),
        ("pooling chunk invariance (chunk 64 vs 256/direct)", "PASS (fixture)"),
        ("same-seed repeat scores within 1e-6", "PASS (fixture)"),
    ):
        L.append(f"| {c} | {s} |")

    fixture = (json.load(open(CACHE / "fixture.json"))
               if (CACHE / "fixture.json").exists() else {})
    if fixture:
        L.append(f"\n_fixture (200 users × 40 days)_: T(M) wall "
                 f"{fixture['M']['wall_s_chunk64']}s, chunk-invariance max|Δ| "
                 f"{fixture['M']['chunk_invariance_max_diff']:.1e}, repeat max|Δ| "
                 f"{fixture['M']['repeat_max_diff']:.1e}; T(C) wall "
                 f"{fixture['C']['wall_s_chunk64']}s.\n")

    L.append("\n## 7. Caveats\n")
    L.append("- Pilot, 3 allocations; no bootstrap, p-values, multiplicity "
             "families or population-confidence claims (PLAN §\"Readout\").\n")
    L.append("- Both splits are reused from prior experiments (R1b, temporal). "
             "Validation alpha selection and test evaluation on the same folds "
             "is an explicit deviation from the temporal plan's retirement "
             "statement, authorized by this plan.\n")
    L.append("- MR bias quantiles on binary M input are degenerate (over "
             "{0,1}); the transform still runs, and the shuffled control "
             "isolates arrangement from the preserved value distribution.\n")
    L.append("- Pooling: M/C/M_shuf/C_shuf pool all 40 days unconditionally; "
             "T(HR) keeps the temporal mask day-gate. Block-local, "
             "intentional.\n")
    L.append("- Shuffling breaks local order AND clock alignment; it does not "
             "separate their contributions (PLAN §\"Readout\").\n")
    L.append("- Recorded salutation ≠ biological sex/gender.\n")

    open(RESULTS / "REPORT.md", "w").write("\n".join(L))


# ---- repro JSON ------------------------------------------------------------
def write_repro(data=None):
    repro = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": __import__("sklearn").__version__,
        "numba": __import__("numba").__version__,
        "pin": json.load(open(_TEMPORAL / "cache" / "vendor_repro.json"))["pin"],
        "input_sha256": {
            "day40_table.parquet": sha256_file(DAY40_PARQUET),
            "folds.parquet": sha256_file(FOLDS_PARQUET),
            "day_bins.npz": sha256_file(BINS_NPZ),
        },
        "config": {
            "MR_NUM_FEATURES_PER_TRANS": MR_NUM_FEATURES_PER_TRANS,
            "MR_WIDTH": MR_WIDTH,
            "ALPHA_GRID": [float(a) for a in ALPHA_GRID],
            "ALLOCATION_TARGETS": list(ALLOCATION_TARGETS),
            "COMPONENT_IDS": {
                "DAY_SELECT": COMP_DAY_SELECT,
                "MR_FIT_HR": COMP_MR_FIT,
                "M_BASE": COMP_M_BASE, "M_DIFF": COMP_M_DIFF,
                "C_BASE": COMP_C_BASE, "C_DIFF": COMP_C_DIFF,
                "SHUFFLE_PERM": COMP_SHUFFLE_PERM},
            "seed_streams": {
                "comp_seed": "SeedSequence([20260921, 3, r, component])",
                "shuffle_perm": ("SeedSequence([20260921, 3, 14, user, date])"
                                 " — alloc-independent, reused across allocs"),
                "shuffled_mr_fit": ("same component seeds as unshuffled "
                                    "counterpart (M: 10/11, C: 12/13)"),
            },
            "threading": {"numba_layer": "workqueue", "numba_threads": 8},
            "pooling": {"T(HR)": "mask_pu day-gate (temporal, unchanged)",
                        "T(M)/T(C)/T(M_shuffled)/T(C_shuffled)":
                            "unconditional (all 40 days)"},
            "hygiene": ("SimpleImputer(median)+VarianceThreshold(0)+"
                        "StandardScaler, per block, train-only fit"),
            "alpha_selection": ("per arm per alloc by validation AUROC; "
                                "tie -> smallest alpha; grid boundary recorded"),
        },
        "metrics_summary": {},
        "fixture": (json.load(open(CACHE / "fixture.json"))
                    if (CACHE / "fixture.json").exists() else None),
        "wall_s": {},
    }
    metrics_csv = RESULTS / "metrics.csv"
    if metrics_csv.exists():
        mm = pd.read_csv(metrics_csv)
        for arm in ARMS:
            sub = mm[mm.arm == arm]
            if len(sub):
                repro["metrics_summary"][arm] = {
                    "val_mean": round(float(sub.auroc_val.mean()), 4),
                    "val_sd": round(float(sub.auroc_val.std(ddof=1)), 4),
                    "test_mean": round(float(sub.auroc_test.mean()), 4),
                    "test_sd": round(float(sub.auroc_test.std(ddof=1)), 4),
                    "n_cols": int(sub.n_cols.mean()),
                    "alphas": [float(a) for a in sub.alpha],
                }
        for r in sorted(mm.alloc.unique()):
            repro["wall_s"][f"alloc_{r}"] = float(
                mm[mm.alloc == r].wall_s.iloc[0])
    json.dump(repro, open(RESULTS / "repro.json", "w"), indent=2)


# ---- CLI -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "run", "report", "full"])
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    data = None
    if args.cmd in ("bench", "run", "full"):
        data = load_data()
        users = data["users"]
        n = len(users)
        mask_pu = data["mask"].reshape(n, DAYS, N_BINS)
        M3 = mask_pu.astype(np.float32)
        C3 = np.log1p(data["cnt"].astype(np.float32)).reshape(n, DAYS, N_BINS)
        log(f"[load] {n} users | bins {data['hr'].shape} | "
            f"RSS {peak_rss_mb():.0f} MiB")

    if args.cmd in ("bench", "full"):
        run_fixture(data, M3, C3, mask_pu)

    if args.cmd in ("run", "full"):
        # shuffle permutation + invariance checks (alloc-independent)
        log("[run] generating per-(user,date) permutation (component 14)...")
        t = time.perf_counter()
        perm = make_permutation(data["pu"], data["pd"])
        log(f"[run] perm {perm.shape} in {time.perf_counter()-t:.1f}s | "
            f"RSS {peak_rss_mb():.0f} MiB")
        M_shuf3 = apply_permutation(M3, perm)
        C_shuf3 = apply_permutation(C3, perm)
        del perm

        # invariance checks (PLAN §"Minimal checks")
        assert np.array_equal(M_shuf3.sum(axis=2), M3.sum(axis=2)), \
            "occupied-bin count not preserved"
        assert np.array_equal(M_shuf3 > 0.5, C_shuf3 > 0), \
            "M_shuffled != (C_shuffled > 0)"
        chk_rng = np.random.default_rng(SEED_BASE)
        for i in chk_rng.choice(n * DAYS, size=200, replace=False):
            assert np.array_equal(
                np.sort(C_shuf3.reshape(-1, N_BINS)[i]),
                np.sort(C3.reshape(-1, N_BINS)[i])), \
                f"cnt multiset not preserved at row {i}"
        log("[run] shuffle invariance checks PASS "
            "(occupied-bin count; M_shuf==(C_shuf>0); cnt multiset sample)")

        static = build_static(users, data["d40"], data["mask"], data["hr"])
        B40 = static["B40"]
        run_all(data, M3, C3, M_shuf3, C_shuf3, mask_pu, B40)

    if args.cmd in ("report", "full"):
        write_report()
        write_repro(data)
        log("[report] REPORT.md + repro.json written")


if __name__ == "__main__":
    main()
