#!/usr/bin/env python
"""R1b — date-aware circadian features (Garmin). Implements PLAN.md.

Reuses the committed R1a runner for A1 hygiene, RF seed streams, the residualise
convention (transductive OLS), and the participant-level BASE matrix; the
allocator is the user's cohort_allocator package, R=10 allocations at the
R1a-AF spec verbatim; the per-day curve computation is a verbatim port of
``run_r1a.compute_r1a`` extended by ``date``. BASE predictions are reused
from R1a-AF's saved predictions (no BASE refits); pairing is verified by a
per-allocation test-user-set equality gate (PLAN §13.2).

CLI:
  python run_r1b.py alloc   # re-derive 10 allocations + gates only
  python run_r1b.py scan    # targeted rescan (delegates to scan_perday.main(8))
  python run_r1b.py build   # dayall/day40/p40 caches
  python run_r1b.py fits    # R=10 day-level + participant fits
  python run_r1b.py sens    # first-k + A40_alldays + random-40 sensitivities
  python run_r1b.py full    # alloc -> scan -> build -> fits -> sens -> report
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
R1A_DIR = REPO / "experiments" / "r1a_circadian_garmin_2026-09-21"
R1AAF_DIR = REPO / "experiments" / "r1a_allocfolds_garmin_2026-09-21"
sys.path.insert(0, str(R1A_DIR))
import run_r1a as r1a  # noqa: E402  (committed frozen R1a runner, 4868482)
sys.path.insert(0, str(REPO / "cohort_allocator" / "src"))
from cohort_allocator import allocate_cohort, make_config  # noqa: E402

try:
    import ortools
    ORT_VERSION = getattr(ortools, "__version__", "unknown")
except Exception:
    ORT_VERSION = "unavailable"

# ---- frozen protocol constants (mirror PLAN.md) ----------------------------
FS_SEED = 20260919
SRC = r1a.SRC                              # 3
R = 10
N_TREES = r1a.N_TREES
G1 = r1a.G1
N_JOBS_OUTER = 8                           # user grant 2026-09-21
ALLOC_SEED_BASE = 20260921
ALLOC_TIME_LIMIT = 300
DAY_SLOT = 7
RANDOM_TRIAL_BASE = 50
COHORT_SIZE = 3848
DESIGNATED_DAYS = 40

PATH_SAL = REPO / "data" / "13Aug_1222.csv"
R1AAF_BALANCE = R1AAF_DIR / "cache" / "balance_run{r}.csv"
R1AAF_PRED = R1AAF_DIR / "results" / "r1aaf_predictions.csv"
R1AAF_METRICS = R1AAF_DIR / "results" / "r1aaf_metrics.csv"
R1AAF_DEMO = R1AAF_DIR / "cache" / "demographics.parquet"
PATH_EPOCH_DAYS = REPO / "experiments" / "artifacts_v2" / "epoch_days.parquet"

RESULTS = HERE / "results"
CACHE = HERE / "cache"
SCAN_PARQUET = CACHE / "per_day_hour_ch3000.parquet"
DAYALL_PARQUET = CACHE / "dayall_table.parquet"
DAY40_PARQUET = CACHE / "day40_table.parquet"
P40_PARQUET = CACHE / "p40.parquet"
FOLDS_PARQUET = CACHE / "folds.parquet"
SENS_TRAIN_PARQUET = CACHE / "sens_train_table.parquet"

CURVE5 = ["curve_night_mean", "curve_morning_slope",
          "curve_day_night_contrast", "curve_range", "curve_entropy"]
STATS = ("mean", "median", "sd", "vmin", "vmax", "n", "cov_h", "hours")
STAT_COLS = [f"d_ch{c}_{s}" for c in (3000, 3001, 3002) for s in STATS]
GAP_COLS = ["d_gap_3000_3001_mean", "d_gap_3000_3002_mean",
            "d_gap_3000_3001_sd", "d_gap_3000_3002_sd"]
A40_COLS = STAT_COLS + GAP_COLS + ["d_is_weekend"]
C40_COLS = A40_COLS + r1a.R1A_FEATURES
DAY_ARMS = ("A40", "C40", "C40_resid")

KS = (1, 2, 4, 8, 16, 24, 32, 40)
RANDOM_TRIALS = 50


# ---- helpers ---------------------------------------------------------------
def log(msg):
    print(msg, flush=True)


def alloc_seed(r: int) -> int:
    return int(np.random.SeedSequence([ALLOC_SEED_BASE, SRC, r])
               .generate_state(1, dtype=np.uint32)[0])


def day_seed(r: int) -> int:
    return int(np.random.SeedSequence([FS_SEED, SRC, r, DAY_SLOT])
               .generate_state(1, dtype=np.uint32)[0])


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_ci(d, q=0.975):
    d = np.asarray(d, dtype=float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(q, n - 1)) * sd / np.sqrt(n)
    return m, sd, half, float((d > 0).mean())


# ============================================================================
# Stage: allocator re-derivation + cross-experiment determinism gate
# ============================================================================
def stage_alloc():
    t0 = time.time()
    log(f"[r1b] alloc start | R={R}")
    demo = pd.read_parquet(R1AAF_DEMO)
    assert len(demo) == COHORT_SIZE, f"demographics shape {len(demo)}"

    split = pd.read_parquet(r1a.PATH_SPLIT)
    y_map = (split[split.source_id == SRC].set_index("user_id")["y"]
             .astype(int))
    users = np.sort(y_map.index.to_numpy().astype("int64"))

    folds_rows, balance_eq = [], []
    last_parts = None
    for r in range(R):
        seed = alloc_seed(r)
        cfg = make_config(id_column="user_id",
                          stratify_columns=("salutation",),
                          partition_columns=("age_band", "bmi_group"),
                          fold_sizes={"train": 70, "val": 15, "test": 15},
                          seed=seed, time_limit_seconds=ALLOC_TIME_LIMIT)
        res = allocate_cohort(demo, cfg)
        parts = res.summary["partitions"]
        last_parts = parts
        if not all(p["all_phases_optimal"] for p in parts):
            raise SystemExit(f"ALLOCATOR GATE FAILED at alloc {r}: "
                             "all_phases_optimal=False")
        asg = res.assignments.copy()
        asg["user_id"] = asg["user_id"].astype("int64")
        folds_rows.append(asg.assign(alloc=r))

        mine = (asg.groupby(["fold", "age_band", "bmi_group", "salutation"])
                .size().rename("n").reset_index())
        mine.to_csv(CACHE / f"balance_run{r}.csv", index=False)
        json.dump({"alloc": r, "alloc_seed": seed, "summary": res.summary},
                  open(CACHE / f"alloc_summary_run{r}.json", "w"),
                  indent=1, default=str)
        ref_path = R1AAF_BALANCE.with_name(
            R1AAF_BALANCE.name.format(r=r))
        if ref_path.exists():
            mine2 = pd.read_csv(CACHE / f"balance_run{r}.csv")
            ref2 = pd.read_csv(ref_path)
            eq = (mine2.sort_values(list(mine2.columns)).reset_index(drop=True)
                  .equals(ref2.sort_values(list(ref2.columns))
                          .reset_index(drop=True)))
            balance_eq.append(int(eq))
        else:
            balance_eq.append(-1)

    folds = pd.concat(folds_rows, ignore_index=True)
    folds.to_parquet(FOLDS_PARQUET, index=False)

    # gate 2: cross-experiment test-user-set equality vs R1a-AF BASE preds
    saved = pd.read_csv(R1AAF_PRED)
    base = saved[saved.arm == "BASE"].copy()
    base["user_id"] = base["user_id"].astype("int64")
    saved_test = {r: set(base[base.alloc == r].user_id) for r in range(R)}
    cross_eq = {}
    for r in range(R):
        fold_r = (folds[folds.alloc == r].set_index("user_id")["fold"]
                  .astype(str))
        fold_r.index = fold_r.index.astype("int64")
        my_test = set(fold_r[fold_r == "test"].index)
        cross_eq[r] = {"mine": len(my_test), "ref": len(saved_test[r]),
                       "eq": my_test == saved_test[r],
                       "symdiff": len(my_test.symmetric_difference(
                           saved_test[r]))}
    if not all(v["eq"] for v in cross_eq.values()):
        bad = [r for r, v in cross_eq.items() if not v["eq"]]
        raise SystemExit(
            f"CROSS-EXPT GATE FAILED: test-user-set mismatch for allocs "
            f"{bad}; BASE⊕P40 vs BASE pairing invalidated")

    # gate 3: BASE AUROC recompute vs R1a-AF metrics
    mdf = pd.read_csv(R1AAF_METRICS)
    base_metrics = mdf[mdf.arm == "BASE"].set_index("alloc")
    recompute = {}
    for r in range(R):
        sub = base[base.alloc == r]
        recomputed = float(roc_auc_score(sub.y_true, sub.p_class1))
        recorded = float(base_metrics.loc[r, "auroc_test"])
        diff = abs(recomputed - recorded)
        recompute[r] = {"recomputed": recomputed, "recorded": recorded,
                        "max_abs_diff": diff}
    if max(v["max_abs_diff"] for v in recompute.values()) > 1e-9:
        raise SystemExit(f"BASE AUROC RECOMPUTE MISMATCH: {recompute}")

    verify = {
        "all_phases_optimal": True,
        "n_partitions": len(last_parts),
        "balance_csv_equality": balance_eq,
        "cross_expt_test_user_set": cross_eq,
        "base_auroc_recompute": recompute,
        "ortools_version": ORT_VERSION,
        "folds_sha256": sha256(FOLDS_PARQUET),
        "demographics_sha256": sha256(R1AAF_DEMO),
        "r1aaf_predictions_sha256": sha256(R1AAF_PRED),
    }
    json.dump(verify, open(CACHE / "alloc_verify.json", "w"), indent=2,
              default=str)
    log(f"[r1b] alloc OK | balance-eq "
        f"{sum(1 for b in balance_eq if b == 1)}/{len(balance_eq)} | "
        f"cross-expt eq {sum(v['eq'] for v in cross_eq.values())}/{R} | "
        f"wall {time.time()-t0:.1f}s")
    return folds


def load_folds() -> pd.DataFrame:
    return pd.read_parquet(FOLDS_PARQUET)


# ============================================================================
# Stage: scan (delegates to scan_perday.main(8))
# ============================================================================
def stage_scan():
    log("[r1b] scan (8 workers)")
    subprocess.run([sys.executable, str(HERE / "scan_perday.py"),
                    "--workers", str(N_JOBS_OUTER)], check=True)


# ============================================================================
# Per-day curve computation — verbatim port of run_r1a.compute_r1a (PLAN §7.2)
# ============================================================================
def compute_day_curves(dh: pd.DataFrame) -> pd.DataFrame:
    g = (dh.groupby(["user", "date", "hour"], sort=True)
         .agg(n=("n", "sum"), vsum=("vsum", "sum")).reset_index())
    g["hmean"] = g.vsum / g.n.replace(0, np.nan)
    pv = g.pivot(index=["user", "date"], columns="hour", values="hmean")
    pv = pv.reindex(columns=list(range(24)))
    wt = (g.pivot(index=["user", "date"], columns="hour", values="n")
          .reindex(columns=list(range(24))))
    n_total = wt.sum(axis=1, min_count=1)

    h = np.arange(24, dtype=float)
    X_design = np.column_stack([
        np.ones(24),
        np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * h / 24),
        np.cos(4 * np.pi * h / 24), np.sin(4 * np.pi * h / 24),
        np.cos(6 * np.pi * h / 24), np.sin(6 * np.pi * h / 24)])
    grid_h = np.arange(0, 24, 0.25)
    G_design = np.column_stack([
        np.ones_like(grid_h),
        np.cos(2 * np.pi * grid_h / 24), np.sin(2 * np.pi * grid_h / 24),
        np.cos(4 * np.pi * grid_h / 24), np.sin(4 * np.pi * grid_h / 24),
        np.cos(6 * np.pi * grid_h / 24), np.sin(6 * np.pi * grid_h / 24)])

    keys = pv.index
    Y = pv.to_numpy(dtype=float)
    W = wt.to_numpy(dtype=float)
    nrows = len(keys)
    out = {c: np.full(nrows, np.nan) for c in r1a.R1A_FEATURES + r1a.WEIGHTS}

    for i in range(nrows):
        yv, w = Y[i], W[i]
        for hcol in range(24):
            out[f"w_h{hcol}"][i] = float(w[hcol]) if np.isfinite(w[hcol]) \
                else np.nan
            out[f"hour_h{hcol}"][i] = float(yv[hcol]) \
                if np.isfinite(yv[hcol]) else np.nan
        valid = np.isfinite(yv) & np.isfinite(w) & (w > 0)
        if not valid.any():
            continue
        y_clean = np.where(valid, yv, 0.0)
        sw = np.sqrt(np.where(valid, w, 0.0))
        coef, *_ = np.linalg.lstsq(X_design * sw[:, None], y_clean * sw,
                                   rcond=None)
        fit_obs = X_design @ coef
        resid = y_clean - fit_obs
        resid_sd = float(np.sqrt(
            np.sum(w * np.where(valid, resid, 0.0) ** 2)
            / max(w[valid].sum(), 1e-9)))
        out["cosinor_M"][i] = float(coef[0])
        out["cosinor_A1"][i] = float(np.hypot(coef[1], coef[2]))
        out["cosinor_A2"][i] = float(np.hypot(coef[3], coef[4]))
        out["cosinor_A3"][i] = float(np.hypot(coef[5], coef[6]))
        out["cosinor_acro_h"][i] = float(grid_h[np.argmax(G_design @ coef)])
        out["cosinor_resid_sd"][i] = resid_sd
        night_m = (np.arange(24) < 6) & valid
        day_m = (np.arange(24) >= 8) & (np.arange(24) <= 19) & valid
        morn_m = (np.arange(24) >= 5) & (np.arange(24) <= 9) & valid
        night = float(np.average(yv[night_m], weights=w[night_m])) \
            if night_m.any() else np.nan
        day_v = float(np.average(yv[day_m], weights=w[day_m])) \
            if day_m.any() else np.nan
        out["curve_night_mean"][i] = night
        out["curve_day_night_contrast"][i] = (
            day_v - night if np.isfinite(day_v) and np.isfinite(night)
            else np.nan)
        if morn_m.sum() >= 2:
            idx = np.flatnonzero(morn_m)
            xs = idx.astype(float); ys = yv[idx]; ws = w[idx]
            wm = np.average(xs, weights=ws)
            wy = np.average(ys, weights=ws)
            wss = np.average((xs - wm) ** 2, weights=ws)
            wsy = np.average((xs - wm) * (ys - wy), weights=ws)
            out["curve_morning_slope"][i] = float(wsy / wss) if wss > 0 \
                else np.nan
        else:
            out["curve_morning_slope"][i] = np.nan
        yv_v = yv[valid]
        out["curve_range"][i] = float(yv_v.max() - yv_v.min())
        shifted = yv_v - yv_v.min()
        ssum = shifted.sum()
        out["curve_entropy"][i] = float(
            -(shifted / ssum * np.log(shifted / ssum + 1e-30)).sum()) \
            if ssum > 0 else 0.0

    df = pd.DataFrame(out, index=keys).reset_index()
    df["n_total"] = n_total.to_numpy()
    df["log_n"] = np.log(df["n_total"].replace(0, np.nan))
    return df


# ============================================================================
# Stage: build dayall/day40/p40 caches
# ============================================================================
def stage_build():
    t0 = time.time()
    log("[r1b] build | epoch_days stats pivot")
    ed = pd.read_parquet(
        PATH_EPOCH_DAYS,
        columns=["user", "date", "channel", "mean", "median", "sd",
                 "vmin", "vmax", "n", "cov_s", "hours"])
    ed["user"] = ed["user"].astype("int64")
    ed["date"] = ed["date"].astype("int64")
    ed["channel"] = ed["channel"].astype("int64")
    cm = pd.read_parquet(r1a.PATH_COHORT)
    s3 = set(cm[cm.source_id == SRC].user_id.astype("int64"))
    ed = ed[ed.user.isin(s3)]

    h3 = ed[ed.channel == 3000]
    adequate = (h3.hours >= 8) & ((h3.cov_s >= 8 * 3600) | (h3.n >= 60))
    h3_ad = (h3[adequate][["user", "date", "mean", "median", "sd", "vmin",
                            "vmax", "n", "cov_s", "hours"]]
             .assign(channel=3000))
    pairs = (h3_ad[["user", "date"]].drop_duplicates()
             .sort_values(["user", "date"]))
    ed_o = ed[ed.channel.isin([3001, 3002])].merge(
        pairs, on=["user", "date"], how="inner")
    ed_keep = pd.concat(
        [h3_ad,
         ed_o[["user", "date", "channel", "mean", "median", "sd", "vmin",
               "vmax", "n", "cov_s", "hours"]]],
        ignore_index=True)
    ed_keep["cov_h"] = (ed_keep["cov_s"] / 3600.0).astype("float32")
    assert not ed_keep.duplicated(["user", "channel", "date"]).any(), \
        "duplicate (user,channel,date) rows in ed_keep"

    piv = ed_keep.pivot(
        index=["user", "date"], columns="channel",
        values=["mean", "median", "sd", "vmin", "vmax", "n", "cov_h",
                "hours"])
    piv.columns = [f"d_ch{c}_{s}" for s, c in piv.columns]
    piv = piv.reset_index()
    for hi, lo in ((3000, 3001), (3000, 3002)):
        piv[f"d_gap_{hi}_{lo}_mean"] = (piv[f"d_ch{hi}_mean"]
                                        - piv[f"d_ch{lo}_mean"])
        piv[f"d_gap_{hi}_{lo}_sd"] = (piv[f"d_ch{hi}_sd"]
                                      - piv[f"d_ch{lo}_sd"])
    log(f"[r1b] build | piv {len(piv):,} day-rows | {time.time()-t0:.1f}s")

    split = pd.read_parquet(r1a.PATH_SPLIT)
    y_map = (split[split.source_id == SRC].set_index("user_id")["y"]
             .astype(int))
    piv["y"] = piv.user.map(y_map).astype("int32")

    dow = (piv["date"].to_numpy() + 3) % 7   # 1970-01-01 is Thursday (dow=3)
    piv["d_is_weekend"] = (dow >= 5).astype("int8")

    # CURVE35 + per-day weights via chunked compute over users
    log(f"[r1b] build | read rescan {SCAN_PARQUET.name}")
    dh = pd.read_parquet(SCAN_PARQUET,
                         columns=["user", "date", "hour", "n", "vsum"])
    log(f"[r1b] build | rescan rows {len(dh):,}")

    uids = np.sort(piv["user"].unique())
    chunks = np.array_split(uids, max(8, N_JOBS_OUTER * 5))
    chunk_jobs = [(c.tolist(), dh[dh.user.isin(c.tolist())].copy())
                  for c in chunks]
    curve_parts = Parallel(n_jobs=N_JOBS_OUTER, backend="loky", verbose=0)(
        delayed(_curve_chunk)(uj, dh_) for uj, dh_ in chunk_jobs)
    curves = pd.concat(curve_parts, ignore_index=True)
    del curve_parts
    log(f"[r1b] build | curves {len(curves):,} day-rows, "
        f"{curves.shape[1]} cols | {time.time()-t0:.1f}s")

    # coverage gate: every designated pair has CURVE35
    pairs["is_first40"] = pairs.groupby("user").cumcount() < DESIGNATED_DAYS
    des = pairs[pairs.is_first40][["user", "date"]]
    miss = des.merge(curves[["user", "date"]].assign(_ok=1),
                     on=["user", "date"], how="left")
    n_missing = int(miss["_ok"].isna().sum())
    if n_missing:
        raise SystemExit(f"BUILD COVERAGE GATE: {n_missing} designated days "
                         f"missing CURVE35")

    dayall = (piv.merge(curves, on=["user", "date"], how="left")
              .sort_values(["user", "date"], kind="stable")
              .reset_index(drop=True))
    dayall["is_first40"] = (dayall.groupby("user", sort=True).cumcount()
                            < DESIGNATED_DAYS).astype("bool")
    for c in STAT_COLS + GAP_COLS:
        dayall[c] = dayall[c].astype("float32")
    for c in r1a.R1A_FEATURES + r1a.WEIGHTS:
        dayall[c] = dayall[c].astype("float64")
    dayall["n_total"] = dayall["n_total"].astype("float64")
    dayall["log_n"] = dayall["log_n"].astype("float64")

    f40 = dayall[dayall.is_first40].reset_index(drop=True)
    counts = f40.groupby("user").size()
    assert (counts == DESIGNATED_DAYS).all(), \
        f"first-40 view row-counts: min {counts.min()} max {counts.max()}"
    assert len(f40) == COHORT_SIZE * DESIGNATED_DAYS, \
        f"first-40 rows {len(f40)} != {COHORT_SIZE * DESIGNATED_DAYS}"

    dayall.to_parquet(DAYALL_PARQUET, index=False)
    f40.to_parquet(DAY40_PARQUET, index=False)
    log(f"[r1b] build | dayall {len(dayall):,} rows | day40 "
        f"{len(f40):,} rows | {time.time()-t0:.1f}s")

    p40 = _build_p40(f40)
    p40.to_parquet(P40_PARQUET, index=True)
    log(f"[r1b] build | p40 {p40.shape[0]} users x {p40.shape[1]} cols | "
        f"{time.time()-t0:.1f}s")


def _curve_chunk(uid_list, dh_chunk):
    return compute_day_curves(dh_chunk)


def _build_p40(f40: pd.DataFrame) -> pd.DataFrame:
    g = f40.groupby("user", sort=True)
    mean35 = g[r1a.R1A_FEATURES].mean().add_suffix("_mean")
    sd35 = g[r1a.R1A_FEATURES].std(ddof=1).add_suffix("_sd")
    we_mask = f40.d_is_weekend.astype(bool)
    wd = (f40[~we_mask].groupby("user")[CURVE5].mean().add_suffix("_wd"))
    we = (f40[we_mask].groupby("user")[CURVE5].mean().add_suffix("_we"))
    diff = wd.values - we.reindex(wd.index).values
    diff = pd.DataFrame(diff, index=wd.index,
                        columns=[f"{c}_wd_diff" for c in CURVE5])
    p40 = pd.concat([mean35, sd35, wd, we, diff], axis=1)
    assert p40.shape[1] == 85, f"p40 cols {p40.shape[1]}"
    return p40


# ============================================================================
# Per-day residualisation — verbatim port of run_r1a.residualise (PLAN §7.3)
# ============================================================================
def residualise_day(day_sub: pd.DataFrame, is_train: np.ndarray) -> pd.DataFrame:
    n_total = day_sub["n_total"].to_numpy(dtype=float)
    Wm = day_sub[r1a.WEIGHTS].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        shares = Wm / n_total[:, None]
    shares = np.where(np.isfinite(shares), shares, 0.0)
    log_n = day_sub["log_n"].to_numpy(dtype=float)
    log_n = np.where(np.isfinite(log_n), log_n, np.nanmean(log_n))
    X = np.column_stack([np.ones(len(day_sub)), shares[:, :-1], log_n])
    out = {}
    for c in r1a.R1A_FEATURES:
        y = day_sub[c].to_numpy(dtype=float)
        ok = np.isfinite(y) & is_train
        if ok.sum() < X.shape[1] + 5:
            out[c] = y
            continue
        coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        fitted = X @ coef
        out[c] = y - np.where(np.isfinite(y), fitted, np.nan)
    return pd.DataFrame(out, index=day_sub.index)


# ============================================================================
# Stage: per-allocation fits (day-level + participant-level)
# ============================================================================
def stage_fits():
    t0 = time.time()
    folds = load_folds()
    log(f"[r1b] fits start | {R} allocs | workers={N_JOBS_OUTER}")

    rows = Parallel(n_jobs=N_JOBS_OUTER, backend="loky", verbose=0)(
        delayed(_fit_alloc)(r, folds) for r in range(R))
    rows.sort(key=lambda d: d["alloc"])
    log(f"[r1b] fits done | {time.time()-t0:.1f}s")
    _write_fits_artifacts(rows, t0)


def _fit_alloc(r: int, folds: pd.DataFrame) -> dict:
    wall0 = time.time()
    split = pd.read_parquet(r1a.PATH_SPLIT)
    y_map = (split[split.source_id == SRC].set_index("user_id")["y"]
             .astype(int))
    users = np.sort(y_map.index.to_numpy().astype("int64"))
    y_all = y_map.loc[users].to_numpy(int)
    fold_r = (folds[folds.alloc == r].set_index("user_id")["fold"]
              .astype(str))
    fold_r.index = fold_r.index.astype("int64")
    fold_arr = fold_r.loc[users].to_numpy()
    tr = np.flatnonzero(fold_arr == "train")
    va = np.flatnonzero(fold_arr == "val")
    te = np.flatnonzero(fold_arr == "test")

    day40 = pd.read_parquet(DAY40_PARQUET)
    day40 = day40.sort_values(["user", "date"], kind="stable").reset_index(
        drop=True)
    d_users = day40["user"].to_numpy()
    is_train_day = np.isin(d_users, users[tr])
    is_val_day = np.isin(d_users, users[va])
    is_test_day = np.isin(d_users, users[te])
    d_tr = day40.iloc[is_train_day].reset_index(drop=True)
    d_va = day40.iloc[is_val_day].reset_index(drop=True)
    d_te = day40.iloc[is_test_day].reset_index(drop=True)
    y_tr = d_tr["y"].to_numpy(int)
    y_va = d_va["y"].to_numpy(int)
    y_te = d_te["y"].to_numpy(int)

    # BASE (must match R1a-AF X exactly for pairing)
    fa = pd.read_parquet(r1a.PATH_FEATS)
    fa = (fa[fa.source_id == SRC].sort_values("user_id")
          .reset_index(drop=True))
    assert list(fa.user_id) == list(users)
    X_base = fa[[c for c in fa.columns
                 if c not in ("user_id", "source_id", "y")]].copy()

    # C40_resid: transductive OLS on full day40, then slice (PLAN §7.3)
    resid_all = residualise_day(day40, is_train_day)

    seed_d = day_seed(r)
    metrics = {"alloc": r, "arms": {}, "fold_sizes": {
        "train": int(len(tr)), "val": int(len(va)), "test": int(len(te))}}

    def eval_arm(X_tr, X_va, X_te, ytr, yva, yte):
        Z_tr, Z_va, Z_te, names = r1a.apply_arm_transform(X_tr, X_va, X_te)
        rf = RandomForestClassifier(n_estimators=N_TREES,
                                    random_state=seed_d, n_jobs=1, **G1)
        rf.fit(Z_tr, ytr)
        j1 = list(rf.classes_).index(1)
        p_va = rf.predict_proba(Z_va)[:, j1]
        p_te = rf.predict_proba(Z_te)[:, j1]
        return {
            "n_cols_a0": int(X_tr.shape[1]),
            "n_cols_hygiene": int(Z_tr.shape[1]),
            "auroc_val": float(roc_auc_score(yva, p_va)),
            "auroc_test": float(roc_auc_score(yte, p_te)),
            "_preds_te": (yte, p_te),
            "_gini_names": list(names),
            "_gini": rf.feature_importances_.astype("float32"),
        }

    # A40
    metrics["arms"]["A40"] = eval_arm(
        d_tr[A40_COLS].copy(), d_va[A40_COLS].copy(),
        d_te[A40_COLS].copy(), y_tr, y_va, y_te)
    metrics["arms"]["A40"].update(
        n_train_days=int(len(d_tr)), n_test_days=int(len(d_te)),
        n_users_test=int(len(te)),
        _uids_te=(d_te["user"].to_numpy(), d_te["date"].to_numpy()))

    # C40
    metrics["arms"]["C40"] = eval_arm(
        d_tr[C40_COLS].copy(), d_va[C40_COLS].copy(),
        d_te[C40_COLS].copy(), y_tr, y_va, y_te)
    metrics["arms"]["C40"].update(
        n_train_days=int(len(d_tr)), n_test_days=int(len(d_te)),
        n_users_test=int(len(te)),
        _uids_te=(d_te["user"].to_numpy(), d_te["date"].to_numpy()))

    # C40_resid
    df_tr = pd.concat([d_tr[A40_COLS].reset_index(drop=True),
                       resid_all.iloc[is_train_day].reset_index(drop=True)],
                      axis=1)
    df_va = pd.concat([d_va[A40_COLS].reset_index(drop=True),
                       resid_all.iloc[is_val_day].reset_index(drop=True)],
                      axis=1)
    df_te = pd.concat([d_te[A40_COLS].reset_index(drop=True),
                       resid_all.iloc[is_test_day].reset_index(drop=True)],
                      axis=1)
    metrics["arms"]["C40_resid"] = eval_arm(
        df_tr, df_va, df_te, y_tr, y_va, y_te)
    metrics["arms"]["C40_resid"].update(
        n_train_days=int(len(d_tr)), n_test_days=int(len(d_te)),
        n_users_test=int(len(te)),
        _uids_te=(d_te["user"].to_numpy(), d_te["date"].to_numpy()))

    # BASE_x_P40
    p40 = pd.read_parquet(P40_PARQUET)
    p40_aligned = p40.reindex(users).reset_index(drop=True)
    X_p40 = pd.concat([X_base.reset_index(drop=True),
                       p40_aligned.reset_index(drop=True)], axis=1)
    seed_p = r1a.rf_seed(r)
    Z_tr, Z_va, Z_te, names_p = r1a.apply_arm_transform(
        X_p40.iloc[tr], X_p40.iloc[va], X_p40.iloc[te])
    rf = RandomForestClassifier(n_estimators=N_TREES,
                                random_state=seed_p, n_jobs=1, **G1)
    rf.fit(Z_tr, y_all[tr])
    j1 = list(rf.classes_).index(1)
    p_te_p = rf.predict_proba(Z_te)[:, j1]
    p_va_p = rf.predict_proba(Z_va)[:, j1]
    metrics["arms"]["BASE_x_P40"] = {
        "n_cols_a0": int(X_p40.shape[1]),
        "n_cols_hygiene": int(Z_tr.shape[1]),
        "auroc_val": float(roc_auc_score(y_all[va], p_va_p)),
        "auroc_test": float(roc_auc_score(y_all[te], p_te_p)),
        "n_train_days": None, "n_test_days": None,
        "n_users_test": int(len(te)),
        "_preds_te": (y_all[te], p_te_p),
        "_gini_names": list(names_p),
        "_gini": rf.feature_importances_.astype("float32"),
    }

    # also stash test_user series per alloc for the participant preds
    metrics["_test_users"] = users[te]
    metrics["_alloc_seed"] = alloc_seed(r)
    metrics["_wall_s"] = round(time.time() - wall0, 1)
    return metrics


def _write_fits_artifacts(rows, t0):
    recs = []
    for d in rows:
        for arm, a in d["arms"].items():
            recs.append({
                "alloc": d["alloc"], "arm": arm,
                "auroc_val": a["auroc_val"], "auroc_test": a["auroc_test"],
                "n_cols_a0": a["n_cols_a0"], "n_cols_hygiene": a["n_cols_hygiene"],
                "n_train_days": a["n_train_days"], "n_test_days": a["n_test_days"],
                "n_users_test": a["n_users_test"],
                "n_train_fold": d["fold_sizes"]["train"],
                "n_val_fold": d["fold_sizes"]["val"],
                "n_test_fold": d["fold_sizes"]["test"],
                "wall_s": d["_wall_s"]})
    pd.DataFrame(recs).to_csv(RESULTS / "r1b_metrics.csv", index=False)
    log(f"[r1b] metrics rows {len(recs)}")

    shutil.copy(R1AAF_PRED, RESULTS / "r1b_predictions_base.csv")

    # day-level per-(alloc, user, date) predictions for all three day arms
    parts = []
    for d in rows:
        for arm in DAY_ARMS:
            a = d["arms"][arm]
            yte, p = a["_preds_te"]
            u_te, dt_te = a["_uids_te"]
            parts.append(pd.DataFrame({
                "arm": arm, "alloc": d["alloc"], "user": u_te,
                "date": dt_te, "y": yte, "p_class1": p}))
    pd.concat(parts, ignore_index=True).to_parquet(
        RESULTS / "r1b_predictions_day.parquet", index=False)
    log(f"[r1b] day preds rows {sum(len(p) for p in parts)}")

    # participant predictions BASE_x_P40
    parts = []
    for d in rows:
        yte, p = d["arms"]["BASE_x_P40"]["_preds_te"]
        parts.append(pd.DataFrame({
            "arm": "BASE_x_P40", "alloc": d["alloc"],
            "user": d["_test_users"], "y": yte, "p_class1": p}))
    pd.concat(parts, ignore_index=True).to_csv(
        RESULTS / "r1b_predictions_part.csv", index=False)

    fi = []
    for d in rows:
        for arm in ("C40", "C40_resid", "BASE_x_P40"):
            names, g = d["arms"][arm]["_gini_names"], d["arms"][arm]["_gini"]
            for nm, gi in zip(names, g):
                fi.append({"alloc": d["alloc"], "arm": arm,
                           "feature": nm, "gini": float(gi),
                           "is_curve": nm in r1a.R1A_FEATURES,
                           "is_p40": nm.endswith(("_mean", "_sd", "_wd",
                                                   "_we", "_wd_diff"))})
    pd.DataFrame(fi).to_csv(RESULTS / "r1b_top_features.csv", index=False)
    log(f"[r1b] fits artifacts written | wall {time.time()-t0:.1f}s")


# ============================================================================
# Stage: sensitivities (first-k, A40_alldays, random-40)
# ============================================================================
def stage_sens_train_table():
    folds = load_folds()
    fold_r0 = folds[folds.alloc == 0].set_index("user_id")["fold"].to_dict()
    tr_u = {u for u, f in fold_r0.items() if f == "train"}
    dayall = pd.read_parquet(DAYALL_PARQUET, columns=[
        "user", "date", "y", "d_is_weekend"] + STAT_COLS + GAP_COLS
        + r1a.R1A_FEATURES)
    sub = dayall[dayall.user.isin(tr_u)].reset_index(drop=True)
    sub.to_parquet(SENS_TRAIN_PARQUET, index=False)
    log(f"[r1b] sens_train_table {len(sub):,} rows, {sub.shape[1]} cols")


def stage_sens():
    t0 = time.time()
    folds = load_folds()
    log(f"[r1b] sens start | first-k + A40_alldays + random-40")

    # ---- first-k (from saved day predictions) -------------------------------
    day = pd.read_parquet(RESULTS / "r1b_predictions_day.parquet")
    day = day[day.arm.isin(("A40", "C40"))]
    eps = 1e-7
    rows = []
    for arm in ("A40", "C40"):
        for r in range(R):
            sub = (day[(day.arm == arm) & (day.alloc == r)]
                   .sort_values(["user", "date"], kind="stable"))
            lgp = np.log(np.clip(sub.p_class1.to_numpy(), eps, 1 - eps)
                         / np.clip(1 - sub.p_class1.to_numpy(),
                                    eps, 1 - eps))
            arr = sub["user"].to_numpy()
            yv = sub["y"].to_numpy()
            y_first = pd.Series(yv).groupby(arr, sort=True).first()
            uniq_u = y_first.index.to_numpy()
            y_map_u = y_first.to_dict()
            order = np.lexsort((sub["date"].to_numpy(), arr))
            su = arr[order]
            _, starts = np.unique(su, return_index=True)
            groups = np.split(np.arange(len(order)), starts[1:])
            for k in KS:
                scores = np.array([lgp[order][g[:min(k, len(g))]].mean()
                                    for g in groups])
                ys = np.array([y_map_u[int(su[g[0]])] for g in groups])
                if len(np.unique(ys)) < 2:
                    continue
                rows.append({"arm": arm, "alloc": r, "k": k,
                              "auroc": float(roc_auc_score(ys, scores)),
                              "n_users": int(len(groups))})
    pd.DataFrame(rows).to_csv(RESULTS / "r1b_firstk.csv", index=False)
    log(f"[r1b] first-k rows {len(rows)} | {time.time()-t0:.1f}s")

    # ---- A40_alldays (r=0) ---------------------------------------------------
    fold_r0 = (folds[folds.alloc == 0].set_index("user_id")["fold"]
               .astype(str).to_dict())
    te_u = {u for u, f in fold_r0.items() if f == "test"}
    va_u = {u for u, f in fold_r0.items() if f == "val"}
    tr_u = {u for u, f in fold_r0.items() if f == "train"}
    dayall = pd.read_parquet(DAYALL_PARQUET,
                              columns=["user", "date", "y", "d_is_weekend"]
                              + STAT_COLS + GAP_COLS)
    # day40 with full C40 columns (CURVE35 needed for the random-40 trials)
    day40 = pd.read_parquet(DAY40_PARQUET,
                             columns=["user", "date", "y"] + C40_COLS)
    train = dayall[dayall.user.isin(tr_u)].reset_index(drop=True)
    val = day40[day40.user.isin(va_u)].reset_index(drop=True)
    test = day40[day40.user.isin(te_u)].reset_index(drop=True)
    Z_tr, Z_va, Z_te, _ = r1a.apply_arm_transform(
        train[A40_COLS], val[A40_COLS], test[A40_COLS])
    rf = RandomForestClassifier(n_estimators=N_TREES,
                                random_state=day_seed(0), n_jobs=1, **G1)
    rf.fit(Z_tr, train["y"].to_numpy(int))
    j1 = list(rf.classes_).index(1)
    p_te = rf.predict_proba(Z_te)[:, j1]
    auc_alldays = float(roc_auc_score(test["y"].to_numpy(int), p_te))
    main_r0 = pd.read_csv(RESULTS / "r1b_metrics.csv")
    main_r0_A40 = float(main_r0[(main_r0.alloc == 0)
                                & (main_r0.arm == "A40")]["auroc_test"].iloc[0])
    alldays_df = pd.DataFrame([{
        "alloc": 0, "main_A40_auroc": main_r0_A40,
        "alldays_A40_auroc": auc_alldays,
        "delta": auc_alldays - main_r0_A40,
        "n_train_days": int(len(train)),
        "n_test_days": int(len(test))}])
    alldays_df.to_csv(RESULTS / "r1b_alldays.csv", index=False)
    log(f"[r1b] alldays vs main: {auc_alldays:.4f} vs {main_r0_A40:.4f} | "
        f"{time.time()-t0:.1f}s")

    # ---- random-40 (r=0, 50 trials, batched 8-way) -------------------------
    if not SENS_TRAIN_PARQUET.exists():
        stage_sens_train_table()
    test40 = day40[day40.user.isin(te_u)].reset_index(drop=True)
    val40 = day40[day40.user.isin(va_u)].reset_index(drop=True)

    trial_batches = np.array_split(np.arange(RANDOM_TRIALS), N_JOBS_OUTER)
    aucs_all = Parallel(n_jobs=N_JOBS_OUTER, backend="loky", verbose=0)(
        delayed(_random40_batch)(b.tolist(),
                                  SENS_TRAIN_PARQUET, val40, test40)
        for b in trial_batches)
    aucs = [a for sub in aucs_all for a in sub]
    rnd = pd.DataFrame({"trial": range(RANDOM_TRIALS), "auroc_test": aucs})
    main_C40_r0 = float(main_r0[(main_r0.alloc == 0)
                                 & (main_r0.arm == "C40")]["auroc_test"].iloc[0])
    rnd["delta_vs_main_C40_r0"] = rnd["auroc_test"] - main_C40_r0
    rnd.to_csv(RESULTS / "r1b_random40.csv", index=False)
    log(f"[r1b] random-40 mean {np.mean(aucs):.4f} ± "
        f"{np.std(aucs, ddof=1):.4f} | wall {time.time()-t0:.1f}s")


def _random40_batch(t_list, sens_path, val_df, test_df):
    sens = pd.read_parquet(sens_path)
    pos_by_user = sens.groupby("user", sort=True).indices
    users_sorted = np.sort(list(pos_by_user.keys()))
    out = []
    for t in t_list:
        seed = int(np.random.SeedSequence(
            [FS_SEED, SRC, 0, RANDOM_TRIAL_BASE + t])
            .generate_state(1, dtype=np.uint32)[0])
        rng = np.random.default_rng(seed)
        chosen = []
        for u in users_sorted:
            idx = pos_by_user[u]
            if len(idx) >= DESIGNATED_DAYS:
                chosen.append(rng.choice(idx, size=DESIGNATED_DAYS,
                                          replace=False))
        chosen = np.sort(np.concatenate(chosen)) if chosen \
            else np.empty(0, int)
        tr = sens.take(chosen).reset_index(drop=True)
        Z_tr, Z_va, Z_te, _ = r1a.apply_arm_transform(
            tr[C40_COLS], val_df[C40_COLS], test_df[C40_COLS])
        rf = RandomForestClassifier(n_estimators=N_TREES,
                                    random_state=day_seed(0),
                                    n_jobs=1, **G1)
        rf.fit(Z_tr, tr["y"].to_numpy(int))
        j1 = list(rf.classes_).index(1)
        p_te = rf.predict_proba(Z_te)[:, j1]
        out.append(float(roc_auc_score(test_df["y"].to_numpy(int), p_te)))
    return out


# ============================================================================
# Stage: report + repro
# ============================================================================
def stage_report():
    t0 = time.time()
    m = pd.read_csv(RESULTS / "r1b_metrics.csv")
    a40 = m[m.arm == "A40"].sort_values("alloc")
    c40 = m[m.arm == "C40"].sort_values("alloc")
    crd = m[m.arm == "C40_resid"].sort_values("alloc")
    bp40 = m[m.arm == "BASE_x_P40"].sort_values("alloc")

    bm = pd.read_csv(R1AAF_METRICS)
    base = (bm[bm.arm == "BASE"]
            .rename(columns={"n_test": "n_users_test",
                              "n_hygiene": "n_cols_hygiene"})
            .sort_values("alloc"))

    d1 = c40.auroc_test.to_numpy() - a40.auroc_test.to_numpy()
    d2 = crd.auroc_test.to_numpy() - a40.auroc_test.to_numpy()
    d3 = bp40.auroc_test.to_numpy() - base.auroc_test.to_numpy()
    Q3 = 1.0 - 0.05 / (2 * 3)

    L = []
    L.append("# R1b — date-aware circadian features (Garmin) — results\n")
    L.append(f"Run {datetime.now().isoformat(timespec='seconds')} | "
             "implements [`../PLAN.md`](../PLAN.md) | R = 10 allocator "
             "allocations (verbatim R1a-AF spec); 3 day-level arms + "
             "1 new participant arm (P40); BASE reused from R1a-AF; "
             f"{N_JOBS_OUTER} workers, n_jobs=1 per fit; first 40 "
             "adequate ch3000 days per user, chronological.\n")

    alloc_verify = json.load(open(CACHE / "alloc_verify.json"))
    L.append(f"**Allocator gate:** all_phases_optimal=True "
             f"({alloc_verify['n_partitions']} partitions x {R} allocs); "
             f"ortools {alloc_verify['ortools_version']}.\n")
    L.append(f"**Cross-experiment determinism:** test-user-set equality "
             f"vs R1a-AF BASE predictions "
             f"{sum(v['eq'] for v in alloc_verify['cross_expt_test_user_set'].values())}/{R}; "
             f"balance-CSV equality "
             f"{sum(1 for b in alloc_verify['balance_csv_equality'] if b == 1)}/{R} "
             f"({sum(1 for b in alloc_verify['balance_csv_equality'] if b == -1)} missing ref); "
             f"BASE AUROC recompute max |delta| "
             f"{max(v['max_abs_diff'] for v in alloc_verify['base_auroc_recompute'].values()):.1e}.\n")
    L.append(f"**Cohort:** s3 strict, n = {COHORT_SIZE} (PLAN §16 erratum to "
             "dayscale PLAN §2: predicted 3,847 from record-level channel-"
             "pass, but epoch_days ch3000 day-rows cover all 3,848 s3 users "
             "and every user has ≥ 80 adequate days; the dayscale day_table "
             "itself has 3,848 users — verified read-only).\n")

    L.append(f"## 1. Absolute AUROC (mean ± SD over {R} allocations)\n")
    L.append("| arm | level | test AUROC | val AUROC | n cols (hygiene) | "
             "test units |")
    L.append("|---|---|---|---|---|---|")
    for arm, sub, units in (
        ("A40", a40, "days"),
        ("C40", c40, "days"),
        ("C40_resid", crd, "days"),
        ("BASE_x_P40", bp40, "users"),
        ("BASE (R1a-AF reuse)", base, "users"),
    ):
        L.append(f"| `{arm}` | "
                 f"{'day' if units=='days' else 'participant'} | "
                 f"{sub.auroc_test.mean():.4f} ± "
                 f"{sub.auroc_test.std(ddof=1):.4f} | "
                 f"{sub.auroc_val.mean():.4f} | "
                 f"{int(sub.n_cols_hygiene.mean())} | "
                 f"{int((sub.n_test_days if units=='days' else sub.n_users_test).mean()):,} |")

    L.append("\n## 2. Primary family (3 paired deltas; Bonferroni 95%-simultaneous over the family)\n")
    L.append("| comparison | mean Δ | 95% t-CI | Bonferroni 95%-simult | "
             "SD | share > 0 |")
    L.append("|---|---|---|---|---|---|")
    for name, d in (("C40 − A40", d1), ("C40_resid − A40", d2),
                    ("BASE⊕P40 − BASE", d3)):
        m95, sd, h95, share = mean_ci(d, 0.975)
        _, _, hB, _ = mean_ci(d, Q3)
        L.append(f"| {name} | {m95:+.4f} | "
                 f"[{m95-h95:+.4f}, {m95+h95:+.4f}] | "
                 f"[{m95-hB:+.4f}, {m95+hB:+.4f}] | "
                 f"{sd:.4f} | {share:.2f} |")

    fk = pd.read_csv(RESULTS / "r1b_firstk.csv")
    if len(fk):
        L.append("\n## 3. First-k curve (chronological; pooled test users)\n")
        L.append("Mean participant AUROC by k (days pooled, chronological), "
                 f"mean over {R} allocations:\n")
        L.append("| k | A40 | C40 |")
        L.append("|---|---|---|")
        for k in KS:
            for_arm = {arm: fk[(fk.arm == arm) & (fk.k == k)].auroc.mean()
                       for arm in ("A40", "C40")}
            if not any(np.isfinite(v) for v in for_arm.values()):
                continue
            L.append(f"| {k} | {for_arm['A40']:.4f} | "
                     f"{for_arm['C40']:.4f} |")

    alld = pd.read_csv(RESULTS / "r1b_alldays.csv")
    if len(alld):
        L.append("\n## 4. A40_alldays sensitivity (r=0; train on ALL adequate "
                 "ch3000 days; evaluate on first-40 test days)\n")
        L.append("| main A40 (r=0) | A40_alldays (r=0) | Δ | n_train_days |")
        L.append("|---|---|---|---|")
        L.append(f"| {alld.main_A40_auroc.iloc[0]:.4f} | "
                 f"{alld.alldays_A40_auroc.iloc[0]:.4f} | "
                 f"{alld.delta.iloc[0]:+.4f} | "
                 f"{int(alld.n_train_days.iloc[0]):,} |")

    rnd = pd.read_csv(RESULTS / "r1b_random40.csv")
    if len(rnd):
        d = rnd.delta_vs_main_C40_r0.to_numpy()
        main_C40 = float(m[(m.alloc == 0) & (m.arm == "C40")]
                         ["auroc_test"].iloc[0])
        L.append("\n## 5. Random-40 sensitivity (r=0, "
                 f"{RANDOM_TRIALS} trials; sample 40 days per training user "
                 "from full adequate pool; evaluate on first-40 test days; "
                 "arm C40; RF seed fixed = day_seed(0))\n")
        m_, sd_, h_, share_ = mean_ci(d, 0.975)
        L.append("| metric | value |\n|---|---|")
        L.append(f"| random-40 mean test AUROC | "
                 f"{rnd.auroc_test.mean():.4f} ± "
                 f"{rnd.auroc_test.std(ddof=1):.4f} |")
        L.append(f"| main C40 r=0 test AUROC | {main_C40:.4f} |")
        L.append(f"| Δ (random − main) mean | {d.mean():+.4f} |")
        L.append(f"| Δ 95% t-CI | [{m_-h_:+.4f}, {m_+h_:+.4f}] |")
        L.append(f"| Δ share > 0 | {share_:.2f} |")

    fi = pd.read_csv(RESULTS / "r1b_top_features.csv")
    if len(fi):
        L.append("\n## 6. Top features (mean Gini over " + str(R) + ")\n")
        L.append("**Arm C40 (day-level):**\n")
        L.append("| feature | Gini (mean) | R1a curve? |")
        L.append("|---|---|---|")
        top_c = (fi[fi.arm == "C40"].groupby("feature")
                 .agg(gini=("gini", "mean"),
                      is_curve=("is_curve", "first"))
                 .sort_values("gini", ascending=False).head(15))
        for name, row in top_c.iterrows():
            L.append(f"| `{name}` | {row.gini:.4f} | "
                     f"{'**yes**' if row.is_curve else 'no'} |")
        L.append("\n**Arm BASE⊕P40 (participant-level):**\n")
        L.append("| feature | Gini (mean) | P40? |")
        L.append("|---|---|---|")
        top_p = (fi[fi.arm == "BASE_x_P40"].groupby("feature")
                 .agg(gini=("gini", "mean"),
                      is_p40=("is_p40", "first"))
                 .sort_values("gini", ascending=False).head(15))
        for name, row in top_p.iterrows():
            L.append(f"| `{name}` | {row.gini:.4f} | "
                     f"{'**yes**' if row.is_p40 else 'no'} |")

    L.append("\n## 7. Caveats\n")
    L.append("- R = 10 under-powers the 3-comparison primary family for true "
             "deltas ≲ 0.005; Bonferroni 95%-simult CIs are wide. Same "
             "caveat as R1a-AF.")
    L.append("- **3,847 → 3,848 cohort correction** vs dayscale PLAN §2. "
             "Dayscale PLAN predicted 3,847 from `FROZEN_CHANNEL_PASS[3][3000]` "
             "(record-level); the actual `epoch_days` ch3000 day-rows cover "
             "all 3,848 s3 users (verified read-only) and the built "
             "`day_table.parquet` has 3,848 users. R1b uses 3,848 throughout.")
    L.append("- First-40 is a prospective-window estimator, not a random "
             "sample; random-40 sensitivity (§5) probes this directly.")
    L.append("- Within-day clustered units (days within users) inflate day-"
             "level test AUROC apparent precision; no within-allocation "
             "cluster bootstrap (RAM-pressure precedent — dayscale). Across-"
             "allocation t-CIs only.")
    L.append("- Same-participant reuse across R1a-AF and R1b is intentional "
             "(paired BASE predictions); not external validation.")
    L.append("- ch3000-only convention for new curve features (vendor caveat "
             "ch3001/ch3002); arm A40 retains dayscale ch3001/3002 stat/gap "
             "cols.")
    L.append("- Recorded salutation ≠ biological sex/gender.")
    L.append("- No vendor-circularity audit (user decision 2026-09-21).")
    L.append("- R1a formula identity: per-day curve computation is a verbatim "
             "port of `run_r1a.compute_r1a` (commit `4868482`) extended by "
             "`date`; per-day residualisation a verbatim port of "
             "`run_r1a.residualise`. CURVE35/WEIGHTS/n_total/log_n stored as "
             "float64 (fidelity to the port); STAT/GAP stored as float32 "
             "(dayscale-matching).")
    L.append("- Dayscale k-curve saturation used within-user with-replacement "
             "sampling; the R1b first-k curve (§3) uses chronological first-k "
             "on the actual R1b model (apples-to-apples on this experiment).\n")
    L.append("## 8. Interpretation\n_(filled after results inspection)_\n")
    open(RESULTS / "REPORT.md", "w").write("\n".join(L))

    def _sha(p):
        return sha256(p) if p.exists() else None
    repro = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "script_sha256": sha256(Path(__file__)),
        "scan_script_sha256": sha256(HERE / "scan_perday.py"),
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "joblib": joblib.__version__, "ortools": ORT_VERSION,
        "config": {
            "R": R, "N_JOBS_OUTER": N_JOBS_OUTER,
            "N_TREES": N_TREES, "G1": G1,
            "DAY_SLOT": DAY_SLOT, "RANDOM_TRIAL_BASE": RANDOM_TRIAL_BASE,
            "ALLOC_SEED_BASE": ALLOC_SEED_BASE,
            "ALLOC_TIME_LIMIT": ALLOC_TIME_LIMIT,
            "DESIGNATED_DAYS": DESIGNATED_DAYS,
            "COHORT_SIZE": COHORT_SIZE,
            "allocator_spec": {
                "id_column": "user_id",
                "stratify_columns": ["salutation"],
                "partition_columns": ["age_band", "bmi_group"],
                "fold_sizes": {"train": 70, "val": 15, "test": 15}}},
        "seed_streams": {
            "allocator": "SeedSequence([20260921, 3, r])",
            "day_level_rf": "SeedSequence([20260919, 3, r, 7])",
            "participant_rf": "SeedSequence([20260919, 3, r, 5])  (R1a slot)",
            "random40_trial": "SeedSequence([20260919, 3, 0, 50+t])"},
        "allocator_gate": alloc_verify,
        "input_sha256": {
            "features_all.parquet": sha256(r1a.PATH_FEATS),
            "split_manifest_sq.parquet": sha256(r1a.PATH_SPLIT),
            "cohort_manifest_model_sources.parquet": sha256(r1a.PATH_COHORT),
            "13Aug_1222.csv": sha256(PATH_SAL),
            "epoch_days.parquet": sha256(PATH_EPOCH_DAYS),
            "r1aaf_demographics.parquet": sha256(R1AAF_DEMO),
            "r1aaf_predictions.csv": sha256(R1AAF_PRED),
            "r1aaf_metrics.csv": sha256(R1AAF_METRICS),
            "rescan_output.parquet": _sha(SCAN_PARQUET),
            "dayall_table.parquet": _sha(DAYALL_PARQUET),
            "day40_table.parquet": _sha(DAY40_PARQUET),
            "p40.parquet": _sha(P40_PARQUET),
            "folds.parquet": _sha(FOLDS_PARQUET)},
        "r1a_runner": {"path": str(r1a.__file__),
                       "sha256": sha256(Path(r1a.__file__))},
        "wall_seconds": round(time.time() - t0, 1),
    }
    json.dump(repro, open(RESULTS / "r1b_repro.json", "w"), indent=1,
              default=str)
    log(f"[r1b] report + repro written | wall {time.time()-t0:.1f}s")


# ============================================================================
# main
# ============================================================================
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"[r1b] start {datetime.now().isoformat(timespec='seconds')} | "
        f"mode={mode} | R={R} | workers={N_JOBS_OUTER}")

    if mode in ("alloc", "full"):
        stage_alloc()
    if mode in ("scan", "full"):
        if SCAN_PARQUET.exists() and (CACHE / "scan_log.json").exists():
            log("[r1b] scan cache hit")
        else:
            stage_scan()
    if mode in ("build", "full"):
        if (DAYALL_PARQUET.exists() and DAY40_PARQUET.exists()
                and P40_PARQUET.exists()):
            log("[r1b] build cache hit")
        else:
            stage_build()
    if mode in ("fits", "full"):
        stage_fits()
    if mode in ("sens", "full"):
        stage_sens()
    if mode == "full":
        stage_report()

    log(f"[r1b] done ({mode}) | total wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
