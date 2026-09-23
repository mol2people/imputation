#!/usr/bin/env python
"""R1a — participant-level circadian curve features from cached hour-of-day
data (Garmin). Implements PLAN.md. Read-only w.r.t. the rest of the repo.

CLI:
  python run_r1a.py             # full run (30 repeats x 5 arms; 2 workers)
  python run_r1a.py bench       # build R1a features + r=0 arm timings
"""
from __future__ import annotations

import hashlib
import json
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
sys.path.insert(0, str(REPO / "src" / "sidequest"))
from preprocess import SQPreprocessor  # noqa: E402

# ---- frozen protocol constants ---------------------------------------------
FS_SEED = 20260919
SRC = 3
R = 30
VI_ALL = 5                              # FS 'all' slot -> BASE replicates
N_TREES = 100
G1 = {"max_features": 0.4, "min_samples_leaf": 10, "max_depth": None,
      "class_weight": "balanced_subsample"}
N_JOBS_OUTER = 2                        # dayscale owns the other 5 cores
ARMS = ("BASE", "R1a_only", "BASE_x_R1a",
        "R1a_resid_only", "BASE_x_R1a_resid")
COVER_THRESHOLD = 0.99                  # min cohort coverage to accept cache

RESULTS = HERE / "results"
CACHE = HERE / "cache"
ART = REPO / "experiments" / "artifacts_sq"

PATH_SPLIT = ART / "split_manifest_sq.parquet"
PATH_FEATS = ART / "features_all.parquet"
PATH_COHORT = ART / "cohort_manifest_model_sources.parquet"
PATH_SAVED_METRICS = ART / "fsplit" / "fsplit_metrics.csv"
HOURS_CANDIDATES = (REPO / "experiments" / "artifacts_v2" / "epoch_hours.parquet",
                    REPO / "experiments" / "artifacts" / "epoch_hours.parquet")

HOURS = [f"hour_h{h}" for h in range(24)]
WEIGHTS = [f"w_h{h}" for h in range(24)]
COSINOR = ["cosinor_M", "cosinor_A1", "cosinor_A2", "cosinor_A3",
           "cosinor_acro_h", "cosinor_resid_sd"]
CURVE = ["curve_night_mean", "curve_morning_slope",
         "curve_day_night_contrast", "curve_range", "curve_entropy"]
R1A_FEATURES = COSINOR + CURVE + HOURS          # 35 target features


def log(msg):
    print(msg, flush=True)


def rf_seed(r: int) -> int:
    return int(np.random.SeedSequence([FS_SEED, SRC, r, VI_ALL])
               .generate_state(1, dtype=np.uint32)[0])


def stratified_split(y: np.ndarray, rng: np.random.Generator):
    tr, va, te = [], [], []
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n = len(idx)
        n_te = int(round(0.15 * n)); n_va = int(round(0.15 * n))
        te.append(idx[:n_te])
        va.append(idx[n_te:n_te + n_va])
        tr.append(idx[n_te + n_va:])
    return (np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)),
            np.sort(np.concatenate(te)))


def nzv_keep(Z: np.ndarray) -> np.ndarray:
    n = Z.shape[0]; keep = np.ones(Z.shape[1], dtype=bool)
    for j in range(Z.shape[1]):
        vals, counts = np.unique(Z[:, j], return_counts=True)
        if counts.max() / n > 0.95 and len(vals) / n < 0.10:
            keep[j] = False
    return keep


# ---- hour cache loading ----------------------------------------------------
def load_hours(cohort_users: set):
    """Return (DataFrame ch3000 rows for cohort, chosen_path, coverage).

    Tries candidates in order; accepts the first with >= 99% cohort
    coverage; ids coerced to int64 (parquet stores them as float64).
    """
    for p in HOURS_CANDIDATES:
        if not p.exists():
            continue
        eh = pd.read_parquet(p, columns=["user", "channel", "hour",
                                         "n", "vsum", "vsumsq"])
        # integral float64 ids -> int64 (exact for |id| < 2^53)
        eh["user"] = eh["user"].astype("int64")
        eh["channel"] = eh["channel"].astype("int64")
        eh["hour"] = eh["hour"].astype("int64")
        eh = eh[(eh.user.isin(cohort_users)) & (eh.channel == 3000)]
        cov = eh.user.nunique() / len(cohort_users)
        log(f"[r1a] candidate {p.name}: {len(eh):,} ch3000 cohort rows, "
            f"coverage {cov:.4f} ({eh.user.nunique()}/{len(cohort_users)})")
        if cov >= COVER_THRESHOLD:
            return eh, p, cov
        log("[r1a] coverage below threshold; trying next candidate")
    raise SystemExit("no hour cache with sufficient cohort coverage")


# ---- R1a feature construction ----------------------------------------------
def compute_r1a(eh: pd.DataFrame, users: np.ndarray) -> pd.DataFrame:
    """35 R1a features + coverage columns per user (index = user ids)."""
    g = (eh.groupby(["user", "hour"], sort=True)
         .agg(n=("n", "sum"), vsum=("vsum", "sum")).reset_index())
    g["hmean"] = g.vsum / g.n.replace(0, np.nan)

    pv = g.pivot(index="user", columns="hour", values="hmean")
    pv = pv.reindex(columns=list(range(24)))
    wt = g.pivot(index="user", columns="hour", values="n").reindex(
        columns=list(range(24)))
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

    rec = {c: {} for c in R1A_FEATURES + WEIGHTS}
    for u in users:
        if u in pv.index:
            yv = pv.loc[u].to_numpy(dtype=float)
            w = wt.loc[u].to_numpy(dtype=float)
        else:
            yv = np.full(24, np.nan)
            w = np.full(24, np.nan)
        for hcol in range(24):
            rec[f"w_h{hcol}"][u] = float(w[hcol]) if np.isfinite(w[hcol]) \
                else np.nan
            rec[f"hour_h{hcol}"][u] = float(yv[hcol]) \
                if np.isfinite(yv[hcol]) else np.nan

        valid = np.isfinite(yv) & np.isfinite(w) & (w > 0)
        if not valid.any():
            for c in COSINOR + CURVE:
                rec[c][u] = np.nan
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
        rec["cosinor_M"][u] = float(coef[0])
        rec["cosinor_A1"][u] = float(np.hypot(coef[1], coef[2]))
        rec["cosinor_A2"][u] = float(np.hypot(coef[3], coef[4]))
        rec["cosinor_A3"][u] = float(np.hypot(coef[5], coef[6]))
        rec["cosinor_acro_h"][u] = float(grid_h[np.argmax(G_design @ coef)])
        rec["cosinor_resid_sd"][u] = resid_sd

        night_m = (np.arange(24) < 6) & valid
        day_m = (np.arange(24) >= 8) & (np.arange(24) <= 19) & valid
        morn_m = (np.arange(24) >= 5) & (np.arange(24) <= 9) & valid
        night = float(np.average(yv[night_m], weights=w[night_m])) \
            if night_m.any() else np.nan
        day_v = float(np.average(yv[day_m], weights=w[day_m])) \
            if day_m.any() else np.nan
        rec["curve_night_mean"][u] = night
        rec["curve_day_night_contrast"][u] = (
            day_v - night if np.isfinite(day_v) and np.isfinite(night)
            else np.nan)
        if morn_m.sum() >= 2:
            idx = np.flatnonzero(morn_m)
            xs = idx.astype(float); ys = yv[idx]; ws = w[idx]
            wm = np.average(xs, weights=ws)
            wy = np.average(ys, weights=ws)
            wss = np.average((xs - wm) ** 2, weights=ws)
            wsy = np.average((xs - wm) * (ys - wy), weights=ws)
            rec["curve_morning_slope"][u] = float(wsy / wss) if wss > 0 \
                else np.nan
        else:
            rec["curve_morning_slope"][u] = np.nan
        yv_v = yv[valid]
        rec["curve_range"][u] = float(yv_v.max() - yv_v.min())
        shifted = yv_v - yv_v.min()
        ssum = shifted.sum()
        rec["curve_entropy"][u] = float(
            -(shifted / ssum * np.log(shifted / ssum + 1e-30)).sum()) \
            if ssum > 0 else 0.0

    df = pd.DataFrame(rec).reindex(users)
    df["n_total"] = n_total.reindex(users).to_numpy()
    df["log_n"] = np.log(df["n_total"].replace(0, np.nan))
    return df


def residualise(r1a_full: pd.DataFrame, train_pos: np.ndarray) -> pd.DataFrame:
    """Linearly partial the 35 R1a features out of hour-of-day coverage.

    Per feature: OLS on train users with regressors
    [1, share_h0..share_h22, log_n]; residuals returned for all users.
    """
    n_total = r1a_full["n_total"].to_numpy(dtype=float)
    W = r1a_full[WEIGHTS].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        shares = W / n_total[:, None]
    shares = np.where(np.isfinite(shares), shares, 0.0)
    log_n = r1a_full["log_n"].to_numpy(dtype=float)
    log_n = np.where(np.isfinite(log_n), log_n, np.nanmean(log_n))
    X = np.column_stack([np.ones(len(r1a_full)), shares[:, :-1], log_n])

    is_train = np.zeros(len(r1a_full), dtype=bool)
    is_train[train_pos] = True
    Xtr = X[is_train]

    out = {}
    for c in R1A_FEATURES:
        y = r1a_full[c].to_numpy(dtype=float)
        ok = np.isfinite(y) & is_train
        if ok.sum() < X.shape[1] + 5:
            out[c] = y
            continue
        coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        fitted = X @ coef
        out[c] = y - np.where(np.isfinite(y), fitted, np.nan)
    return pd.DataFrame(out, index=r1a_full.index)


# ---- preprocessing / fit ---------------------------------------------------
def apply_arm_transform(X_tr, X_va, X_te):
    prep = SQPreprocessor().fit(X_tr)
    Z_tr = prep.transform(X_tr); Z_va = prep.transform(X_va)
    Z_te = prep.transform(X_te)
    names = np.array(prep.feature_names_out_)
    ind = np.array([n.endswith("__missing") for n in names])
    base = ~ind
    sub = nzv_keep(Z_tr[:, base])
    a1 = base.copy(); a1[np.flatnonzero(base)[~sub]] = False
    return Z_tr[:, a1], Z_va[:, a1], Z_te[:, a1], names[a1]


# ---- per-repeat worker -----------------------------------------------------
def run_repeat(r: int, r1a_full_sorted: pd.DataFrame) -> dict:
    """r1a_full_sorted must be indexed by the sorted cohort user list."""
    split = pd.read_parquet(PATH_SPLIT)
    y_map = split[split.source_id == SRC].set_index("user_id")["y"].astype(int)
    users = np.sort(y_map.index.to_numpy())
    y_all = y_map.loc[users].to_numpy(dtype=int)
    rng = np.random.default_rng(np.random.SeedSequence([FS_SEED, SRC, r]))
    tr, va, te = stratified_split(y_all, rng)

    fa = pd.read_parquet(PATH_FEATS)
    fa = fa[fa.source_id == SRC].sort_values("user_id").reset_index(drop=True)
    assert list(fa.user_id) == list(users)
    y = fa.user_id.map(y_map).to_numpy(dtype=int)
    X_base = fa[[c for c in fa.columns
                 if c not in ("user_id", "source_id", "y")]].copy()

    r1a_full = r1a_full_sorted.reindex(users)
    r1a_feat = r1a_full[R1A_FEATURES].copy()
    train_pos = np.searchsorted(users, users[tr])
    r1a_resid = residualise(r1a_full, train_pos)

    seed = rf_seed(r)

    def eval_arm(X: pd.DataFrame, do_gini: bool = False) -> dict:
        Z_tr, Z_va, Z_te, names = apply_arm_transform(
            X.iloc[tr], X.iloc[va], X.iloc[te])
        rf = RandomForestClassifier(n_estimators=N_TREES,
                                    random_state=seed, n_jobs=1, **G1)
        rf.fit(Z_tr, y[tr])
        j1 = list(rf.classes_).index(1)
        p_va = rf.predict_proba(Z_va)[:, j1]
        p_te = rf.predict_proba(Z_te)[:, j1]
        res = {"n_a0": int(len(names)), "n_hygiene": int(Z_tr.shape[1]),
               "auroc_val_hygiene": float(roc_auc_score(y[va], p_va)),
               "auroc_test_hygiene": float(roc_auc_score(y[te], p_te))}
        if do_gini:
            res["_gini_names"] = list(names)
            res["_gini"] = rf.feature_importances_.astype("float32")
            res["_uids"] = (users[va], users[te])
            res["_y"] = (y[va], y[te])
            res["_preds"] = {"hygiene": (p_va, p_te)}
        return res

    out = {"repeat": r, "arms": {}}
    out["arms"]["BASE"] = eval_arm(X_base)
    out["arms"]["R1a_only"] = eval_arm(r1a_feat)
    out["arms"]["BASE_x_R1a"] = eval_arm(
        pd.concat([X_base.reset_index(drop=True),
                   r1a_feat.reset_index(drop=True)], axis=1), do_gini=True)
    out["arms"]["R1a_resid_only"] = eval_arm(r1a_resid)
    out["arms"]["BASE_x_R1a_resid"] = eval_arm(
        pd.concat([X_base.reset_index(drop=True),
                   r1a_resid.reset_index(drop=True)], axis=1))
    return out


# ---- validation gate -------------------------------------------------------
def validate_base(rows) -> str:
    saved = pd.read_csv(PATH_SAVED_METRICS)
    sv = saved[(saved.source == SRC) & (saved.variant == "all")
               & (saved.arm == "A1")].sort_values("repeat")
    if len(sv) != R:
        raise SystemExit(f"VALIDATION REFERENCE MISSING: {len(sv)} A1 rows")
    max_d = 0.0
    for r in range(R):
        ref = sv[sv.repeat == r].iloc[0]
        mine = rows[r]["arms"]["BASE"]
        max_d = max(max_d,
                    abs(mine["auroc_val_hygiene"] - ref.auroc_val),
                    abs(mine["auroc_test_hygiene"] - ref.auroc_test))
    msg = (f"max |delta AUROC| vs frozen FS A1 (s3 all) = {max_d:.2e} "
           f"across {R}/{R} repeats")
    if max_d > 1e-12:
        raise SystemExit(f"VALIDATION FAILED: {msg}")
    return msg


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_ci(d):
    d = np.asarray(d, dtype=float); n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
    return m, sd, half, float((d > 0).mean())


# ---- main ------------------------------------------------------------------
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"[r1a] start {datetime.now().isoformat(timespec='seconds')} "
        f"| mode={mode}")

    cm = pd.read_parquet(PATH_COHORT)
    cohort_users = set(cm[cm.source_id == SRC].user_id.astype("int64"))
    log(f"[r1a] s3 cohort: {len(cohort_users)} users")

    eh, hours_path, coverage = load_hours(cohort_users)
    cache_path = CACHE / f"r1a_features_{hours_path.stem}.parquet"
    if cache_path.exists():
        r1a_df = pd.read_parquet(cache_path)
        log(f"[r1a] r1a features: cache hit ({r1a_df.shape[0]} users)")
    else:
        split = pd.read_parquet(PATH_SPLIT)
        y_map = split[split.source_id == SRC].set_index("user_id")["y"]
        users_sorted = np.sort(y_map.index.to_numpy().astype("int64"))
        r1a_df = compute_r1a(eh, users_sorted)
        r1a_df.to_parquet(cache_path, index=True)
        log(f"[r1a] r1a features: built {r1a_df.shape[0]} x "
            f"{r1a_df.shape[1]} ({time.time()-t0:.0f}s)")

    # univariate AUROCs (descriptive, pooled cohort, label-blind)
    split = pd.read_parquet(PATH_SPLIT)
    y_map = split[split.source_id == SRC].set_index("user_id")["y"].astype(int)
    y_vec = r1a_df.index.map(y_map).to_numpy(dtype=int)
    uni = {}
    for c in R1A_FEATURES:
        v = r1a_df[c].to_numpy(dtype=float)
        m = np.isfinite(v)
        if m.sum() > 100 and len(np.unique(y_vec[m])) == 2:
            uni[c] = float(roc_auc_score(y_vec[m], v[m]))
        else:
            uni[c] = np.nan
    pd.Series(uni, name="auroc_cohort").to_csv(
        RESULTS / "r1a_univariate.csv")

    if mode == "bench":
        r0 = run_repeat(0, r1a_df)
        for arm in ARMS:
            m = r0["arms"][arm]
            log(f"[r1a] bench r=0 {arm:18s} test "
                f"{m['auroc_test_hygiene']:.4f} | cols {m['n_hygiene']}")
        # single-repeat gate preview (full 30-repeat gate runs in full mode)
        saved = pd.read_csv(PATH_SAVED_METRICS)
        sv = saved[(saved.source == SRC) & (saved.variant == "all")
                   & (saved.arm == "A1") & (saved.repeat == 0)]
        if len(sv) == 1:
            ref = sv.iloc[0]
            dv = abs(r0["arms"]["BASE"]["auroc_val_hygiene"] - ref.auroc_val)
            dt = abs(r0["arms"]["BASE"]["auroc_test_hygiene"] - ref.auroc_test)
            ok = max(dv, dt) <= 1e-12
            log(f"[r1a] bench r=0 BASE vs frozen FS A1: |d_val|={dv:.2e} "
                f"|d_test|={dt:.2e} -> {'BIT-EXACT OK' if ok else 'MISMATCH'}")
        else:
            log(f"[r1a] bench: A1 repeat-0 reference rows found = {len(sv)} "
                "(skip preview)")
        log(f"[r1a] bench total {time.time()-t0:.0f}s; est full ~ "
            f"{(time.time()-t0) * R / N_JOBS_OUTER / 60:.1f} min")
        return

    t1 = time.time()
    rows = Parallel(n_jobs=N_JOBS_OUTER, backend="loky", verbose=0)(
        delayed(run_repeat)(r, r1a_df) for r in range(R))
    rows.sort(key=lambda d: d["repeat"])
    log(f"[r1a] {R} repeats x {len(ARMS)} arms done ({time.time()-t1:.0f}s)")

    val_msg = validate_base(rows)
    log(f"[r1a] BASE gate: {val_msg}")

    # ---- metrics ---------------------------------------------------------------
    recs = []
    for r in range(R):
        for arm in ARMS:
            a = rows[r]["arms"][arm]
            recs.append({"arm": arm, "repeat": r,
                         "auroc_val": a["auroc_val_hygiene"],
                         "auroc_test": a["auroc_test_hygiene"],
                         "n_a0": a["n_a0"], "n_hygiene": a["n_hygiene"]})
    pd.DataFrame(recs).to_csv(RESULTS / "r1a_metrics.csv", index=False)

    # ---- predictions (BASE_x_R1a) ----------------------------------------------
    prec = []
    for r in range(R):
        a = rows[r]["arms"]["BASE_x_R1a"]
        p_va, p_te = a["_preds"]["hygiene"]
        u_va, u_te = a["_uids"]; y_va, y_te = a["_y"]
        for sname, u, yv, p in (("val", u_va, y_va, p_va),
                                ("test", u_te, y_te, p_te)):
            prec.append(pd.DataFrame({"arm": "BASE_x_R1a", "repeat": r,
                                      "split": sname, "user_id": u,
                                      "y_true": yv, "p_class1": p}))
    pd.concat(prec, ignore_index=True).to_csv(
        RESULTS / "r1a_predictions.csv", index=False)

    # ---- top features (Gini in BASE_x_R1a) --------------------------------------
    fi = []
    for r in range(R):
        a = rows[r]["arms"]["BASE_x_R1a"]
        for nm, g in zip(a["_gini_names"], a["_gini"]):
            fi.append({"repeat": r, "feature": nm, "gini": float(g),
                       "is_r1a": nm.startswith(("cosinor_", "curve_",
                                                "hour_h"))})
    fi_df = pd.DataFrame(fi)
    fi_df.to_csv(RESULTS / "r1a_top_features.csv", index=False)

    # ---- report ------------------------------------------------------------------
    def arr(arm, key="auroc_test_hygiene"):
        return np.array([rows[r]["arms"][arm][key] for r in range(R)])

    d1 = arr("BASE_x_R1a") - arr("BASE")
    d2 = arr("BASE_x_R1a_resid") - arr("BASE")
    d3 = arr("BASE_x_R1a") - arr("R1a_only")
    d4 = arr("BASE_x_R1a_resid") - arr("R1a_resid_only")

    L = []
    L.append("# R1a — participant-level circadian curve (Garmin) — results\n")
    L.append(f"Run {datetime.now().isoformat(timespec='seconds')} | "
             "implements [`../PLAN.md`](../PLAN.md) | 30 FS-phase repeats "
             f"(same splits/seeds), paired; {N_JOBS_OUTER} workers, "
             "n_jobs=1 per fit.\n")
    L.append(f"**BASE gate:** {val_msg}.\n")
    L.append(f"**Cache:** `{hours_path.name}` | cohort coverage "
             f"{coverage:.4f} ({r1a_df.shape[0]}/{len(cohort_users)}).\n")

    L.append("## 1. Absolute AUROC (mean ± SD over 30 repeats)\n")
    L.append("| arm | test AUROC | val AUROC | n cols (hygiene) |")
    L.append("|---|---|---|---|")
    for arm in ARMS:
        t, v = arr(arm), arr(arm, "auroc_val_hygiene")
        nh = int(np.mean([rows[r]["arms"][arm]["n_hygiene"]
                          for r in range(R)]))
        L.append(f"| `{arm}` | {t.mean():.4f} ± {t.std(ddof=1):.4f} | "
                 f"{v.mean():.4f} | {nh} |")

    L.append("\n## 2. Paired test-AUROC deltas (97.5% t-CI)\n")
    L.append("| comparison | mean Δ | 97.5% t-CI | SD | share > 0 |")
    L.append("|---|---|---|---|---|")
    for name, d in (("BASE⊕R1a − BASE", d1),
                    ("BASE⊕R1a_resid − BASE", d2),
                    ("BASE⊕R1a − R1a_only", d3),
                    ("BASE⊕R1a_resid − R1a_resid_only", d4)):
        m, sd, half, share = mean_ci(d)
        L.append(f"| {name} | {m:+.4f} | [{m-half:+.4f}, {m+half:+.4f}] | "
                 f"{sd:.4f} | {share:.2f} |")

    L.append("\n## 3. Top features (BASE⊕R1a Gini, mean over 30)\n")
    L.append("| feature | Gini (mean) | R1a? |")
    L.append("|---|---|---|")
    top = (fi_df.groupby("feature")
           .agg(gini=("gini", "mean"), is_r1a=("is_r1a", "first"))
           .sort_values("gini", ascending=False).head(15))
    for name, row in top.iterrows():
        L.append(f"| `{name}` | {row.gini:.4f} | "
                 f"{'**yes**' if row.is_r1a else 'no'} |")

    L.append("\n## 4. Univariate AUROCs (descriptive, pooled cohort)\n")
    L.append("| feature | AUROC |")
    L.append("|---|---|")
    for name, v in sorted(uni.items(), key=lambda kv: -(
            kv[1] if np.isfinite(kv[1]) else -1))[:15]:
        L.append(f"| `{name}` | {v:.4f} |")

    L.append("\n## 5. Caveats\n")
    L.append("- Same-partition reuse — not independent validation.")
    L.append("- Curve features correlate with wear pattern by construction; "
             "`BASE⊕R1a_resid` is the deconfounded comparison.")
    L.append("- ch3001/ch3002 excluded (vendor caveat).")
    L.append("- Weekday/weekend contrast not derivable from the hour-of-day "
             "cache (lives in R1b, deferred).\n")
    L.append("## 6. Interpretation\n_(filled after results inspection)_\n")
    open(RESULTS / "REPORT.md", "w").write("\n".join(L))

    repro = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "script_sha256": sha256(Path(__file__)),
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "config": {"FS_SEED": FS_SEED, "SRC": SRC, "R": R,
                   "VI_ALL": VI_ALL, "N_TREES": N_TREES, "G1": G1,
                   "N_JOBS_OUTER": N_JOBS_OUTER, "ARMS": list(ARMS),
                   "R1A_FEATURES": R1A_FEATURES},
        "validation": val_msg,
        "hours_cache": {"path": str(hours_path), "coverage": coverage},
        "input_sha256": {
            "features_all.parquet": sha256(PATH_FEATS),
            "split_manifest_sq.parquet": sha256(PATH_SPLIT),
            "cohort_manifest_model_sources.parquet": sha256(PATH_COHORT),
            "epoch_hours (chosen)": sha256(hours_path),
            "fsplit_metrics.csv": sha256(PATH_SAVED_METRICS)},
        "wall_seconds": round(time.time() - t0, 1)}
    json.dump(repro, open(RESULTS / "r1a_repro.json", "w"), indent=1)
    log(f"[r1a] results written to {RESULTS}")
    log(f"[r1a] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
