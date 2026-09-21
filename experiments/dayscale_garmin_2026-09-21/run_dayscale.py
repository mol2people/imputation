#!/usr/bin/env python
"""Day-level salutation classification (Garmin).

Implements PLAN.md. Read-only w.r.t. the rest of the repo; writes into
./cache/ (git-ignored) and ./results/. 5 workers, n_jobs=1 per fit.

CLI:
  python run_dayscale.py [build|bench|full]
    build  build cache/day_table.parquet (skipped if present)
    bench  build (if needed) + full r=0 repeat with per-arm timing
    full   build (if needed) + 30 repeats x 3 arms + SA_K + outputs (default)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src" / "sidequest"))
from preprocess import SQPreprocessor  # noqa: E402

# --- protocol constants (mirror PLAN.md) ------------------------------------
FS_SEED = 20260919
BASELINE_SEED = 20260921
SRC = 3
R = 30
K_CAP = 300
VI_DAY = 6                       # seed slot after "all" (=5)
N_TREES = 100
PI_REPEATS = 3
G1 = {"max_features": 0.4, "min_samples_leaf": 10, "max_depth": None,
      "class_weight": "balanced_subsample"}
N_JOBS_OUTER = 5
KS = (1, 2, 4, 8, 16, 32, 64, 128)
TRIALS = 200
BOOT = 500
PI_SUB = 30000
CH_HR = (3000, 3001, 3002)
STATS = ("mean", "median", "sd", "vmin", "vmax", "n", "cov_h", "hours")
ARMS = ("A", "B", "A_covs")
SA_ARMS = {"A_K100": 100, "A_uncapped": None}   # label -> cap (None = uncapped)

RESULTS = HERE / "results"
CACHE = HERE / "cache"
PATH_SPLIT = REPO / "artifacts_sq" / "split_manifest_sq.parquet"
PATH_FEATS_ALL = REPO / "artifacts_sq" / "features_all.parquet"
PATH_EPOCH_DAYS = REPO / "artifacts_v2" / "epoch_days.parquet"
PATH_SALUTATION = REPO / "13Aug_1222.csv"

STAT_COLS = [f"d_ch{c}_{s}" for c in CH_HR for s in STATS]
GAP_COLS = ["d_gap_3000_3001_mean", "d_gap_3000_3002_mean",
            "d_gap_3000_3001_sd", "d_gap_3000_3002_sd"]
A_COLS = STAT_COLS + GAP_COLS


def rf_seed(r):
    return int(np.random.SeedSequence([FS_SEED, SRC, r, VI_DAY])
               .generate_state(1, dtype=np.uint32)[0])


def stratified_split(y, rng):
    tr, va, te = [], [], []
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n = len(idx)
        n_te = int(round(0.15 * n))
        n_va = int(round(0.15 * n))
        te.append(idx[:n_te])
        va.append(idx[n_te:n_te + n_va])
        tr.append(idx[n_te + n_va:])
    return (np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)),
            np.sort(np.concatenate(te)))


def nzv_keep(Z, n_threshold=0.95, unique_threshold=0.10):
    n = Z.shape[0]
    keep = np.ones(Z.shape[1], dtype=bool)
    for j in range(Z.shape[1]):
        vals, counts = np.unique(Z[:, j], return_counts=True)
        if counts.max() / n > n_threshold and len(vals) / n < unique_threshold:
            keep[j] = False
    return keep


def all_user_baselines(day: pd.DataFrame) -> dict:
    sub = day[day.is_baseline]
    grp = sub.groupby("user")[STAT_COLS].mean()
    return {c: grp[c].to_dict() for c in STAT_COLS}


def make_X(day_sub: pd.DataFrame, arm: str, baselines: dict) -> pd.DataFrame:
    X = day_sub[A_COLS].astype("float32").copy()
    if arm == "B":
        for c in STAT_COLS:
            bm = day_sub["user"].map(baselines[c]).astype("float32").to_numpy()
            X[f"dev_{c}"] = (day_sub[c].to_numpy(dtype="float32") - bm)
    elif arm == "A_covs":
        X["age_at_day"] = day_sub["age_at_day"].astype("float32").to_numpy()
        X["demo__bmi_grp"] = day_sub["demo__bmi_grp"].astype("object")
    return X


def apply_arm_transform(X_tr, X_va, X_te):
    prep = SQPreprocessor().fit(X_tr)
    Z_tr = prep.transform(X_tr)
    Z_va = prep.transform(X_va)
    Z_te = prep.transform(X_te)
    names = np.array(prep.feature_names_out_)
    ind = np.array([n.endswith("__missing") for n in names])
    base = ~ind
    sub = nzv_keep(Z_tr[:, base])
    a1 = base.copy()
    a1[np.flatnonzero(base)[~sub]] = False
    return Z_tr[:, a1], Z_va[:, a1], Z_te[:, a1], names[a1]


def ece_10bin(p, y):
    order = np.argsort(p)
    p_s, y_s = p[order], y[order]
    n = len(p)
    bin_ = n // 10
    e = 0.0
    for i in range(10):
        lo, hi = i * bin_, (i + 1) * bin_ if i < 9 else n
        e += (hi - lo) / n * abs(y_s[lo:hi].mean() - p_s[lo:hi].mean())
    return float(e)


def run_repeat(r: int, cache_path: Path, cap=K_CAP, arms=ARMS,
               sa_label=None, light=False):
    """One repeat. cap=None -> uncapped. light=True -> metrics+gini only."""
    day = pd.read_parquet(cache_path)
    split = pd.read_parquet(PATH_SPLIT)
    y_map = split[split.source_id == SRC].set_index("user_id")["y"].astype(int)
    fs_users = np.sort(y_map.index.to_numpy())
    y_all = y_map.loc[fs_users].to_numpy(int)
    rng_split = np.random.default_rng(np.random.SeedSequence([FS_SEED, SRC, r]))
    tr, va, te = stratified_split(y_all, rng_split)
    users_tr, users_va, users_te = fs_users[tr], fs_users[va], fs_users[te]
    seed = rf_seed(r)
    cap_val = 10 ** 9 if cap is None else cap

    model = day[~day.is_baseline].reset_index(drop=True)
    pos_by_user = model.groupby("user", sort=True).indices   # user -> positions

    # cap train days (seeded per repeat)
    rng_cap = np.random.default_rng(
        np.random.SeedSequence([FS_SEED, SRC, r, 10]).generate_state(1, dtype=np.uint32))
    train_pos = []
    for u in users_tr:
        idx = pos_by_user.get(u, np.empty(0, dtype=int))
        if len(idx) > cap_val:
            train_pos.append(rng_cap.choice(idx, size=cap_val, replace=False))
        elif len(idx):
            train_pos.append(idx)
    train_pos = np.sort(np.concatenate(train_pos)) if train_pos else np.empty(0, int)
    val_pos = (np.concatenate([pos_by_user[u] for u in users_va
                               if u in pos_by_user])
               if any(u in pos_by_user for u in users_va) else np.empty(0, int))
    test_pos = (np.concatenate([pos_by_user[u] for u in users_te
                                if u in pos_by_user])
                if any(u in pos_by_user for u in users_te) else np.empty(0, int))

    train_sub = model.take(train_pos)
    val_sub = model.take(val_pos)
    test_sub = model.take(test_pos)

    y_tr = train_sub["y"].astype(int).to_numpy()
    y_va = val_sub["y"].astype(int).to_numpy()
    y_te = test_sub["y"].astype(int).to_numpy()
    test_users = test_sub["user"].to_numpy()
    test_dates = test_sub["date"].to_numpy()

    baselines = all_user_baselines(day)

    metrics = {"repeat": r, "arms": {}}
    gini_out = {}
    pi_out = None
    p_te_store, p_va_store = {}, {}

    for arm in arms:
        t0 = time.time()
        X_tr = make_X(train_sub, arm, baselines)
        X_va = make_X(val_sub, arm, baselines)
        X_te = make_X(test_sub, arm, baselines)
        Z_tr, Z_va, Z_te, names = apply_arm_transform(X_tr, X_va, X_te)
        rf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                    n_jobs=1, **G1)
        rf.fit(Z_tr, y_tr)
        j1 = list(rf.classes_).index(1)
        p_va = rf.predict_proba(Z_va)[:, j1]
        p_te = rf.predict_proba(Z_te)[:, j1]
        elapsed = time.time() - t0
        label = sa_label if (sa_label is not None and arm == "A") else arm
        metrics["arms"][label] = dict(
            auroc_val=float(roc_auc_score(y_va, p_va)),
            auroc_test=float(roc_auc_score(y_te, p_te)),
            ece_val=ece_10bin(p_va, y_va),
            n_train_days=int(len(train_sub)), n_val_days=int(len(val_sub)),
            n_test_days=int(len(test_sub)), n_features=int(Z_tr.shape[1]),
            n_users_train=len(users_tr), n_users_val=len(users_va),
            n_users_test=len(users_te), fit_seconds=round(elapsed, 2))
        gini_out[label] = (names, rf.feature_importances_.astype("float32"))
        if arm == "A" and not light:
            rng_pi = np.random.default_rng(
                np.random.SeedSequence([FS_SEED, SRC, r, 30]).generate_state(1, dtype=np.uint32))
            sub_i = rng_pi.choice(len(p_va), size=min(PI_SUB, len(p_va)),
                                  replace=False)
            pi = permutation_importance(rf, Z_va[sub_i], y_va[sub_i],
                                        scoring="roc_auc",
                                        n_repeats=PI_REPEATS, n_jobs=1,
                                        random_state=int(rng_pi.integers(0, 2 ** 31 - 1)))
            pi_out = (names, pi.importances_mean.astype("float32"),
                      pi.importances_std.astype("float32"))
        p_te_store[arm] = p_te
        p_va_store[arm] = p_va
        del rf, Z_tr, Z_va, Z_te, X_tr, X_va, X_te

    if light:
        return {"metrics": metrics, "gini": gini_out, "pi": None,
                "scaling": [], "bootstrap": [], "user_scores": [],
                "preds_r0": None}

    # ---- shared test-day layout (sorted by user, then date) -------------------
    order_idx = np.lexsort((test_dates, test_users))
    su = test_users[order_idx]
    _, starts = np.unique(su, return_index=True)
    groups = np.split(np.arange(len(order_idx)), starts[1:])  # positions in sorted order
    y_by = {int(u): int(y_map[u]) for u in np.unique(su)}
    eps = 1e-7

    def logits(p):
        return np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))

    # ---- scaling curve: SAME trial indices across arms (paired) ---------------
    rng_cv = np.random.default_rng(
        np.random.SeedSequence([FS_SEED, SRC, r, 20]).generate_state(1, dtype=np.uint32))
    scaling = []
    for k in KS:
        valid = [g for g in groups if len(g) >= k]
        if not valid:
            continue
        trial_idx = [rng_cv.integers(0, len(g), size=(TRIALS, k)) for g in valid]
        ys = np.array([y_by[int(su[g[0]])] for g in valid])
        for arm in arms:
            lg_s = logits(p_te_store[arm])[order_idx]
            scores = np.empty((len(valid), TRIALS))
            for i, g in enumerate(valid):
                scores[i] = lg_s[g][trial_idx[i]].mean(axis=1)
            aucs = np.array([roc_auc_score(ys, scores[:, t])
                             for t in range(TRIALS)])
            scaling.append((arm, r, int(k), float(aucs.mean()),
                            float(aucs.std()), int(len(valid))))
    for arm in arms:
        scores_all = np.array([logits(p_te_store[arm])[order_idx][g].mean()
                               for g in groups])
        ys_all = np.array([y_by[int(su[g[0]])] for g in groups])
        scaling.append((arm, r, "all",
                        float(roc_auc_score(ys_all, scores_all)), 0.0,
                        int(len(groups))))

    # ---- user-level scores (all days pooled) -----------------------------------
    user_scores = []
    for arm in arms:
        lg_s = logits(p_te_store[arm])[order_idx]
        for g in groups:
            user_scores.append({"arm": arm, "repeat": r,
                                "user": int(su[g[0]]),
                                "y": y_by[int(su[g[0]])],
                                "n_model_days": int(len(g)),
                                "mean_logit": float(lg_s[g].mean())})

    # ---- participant-cluster bootstrap (SAME user draws across arms) -----------
    U = len(groups)
    rows_concat = order_idx                       # test rows in user-sorted order
    user_of_row = np.repeat(np.arange(U), [len(g) for g in groups])
    rng_b = np.random.default_rng(
        np.random.SeedSequence([FS_SEED, SRC, r, 40]).generate_state(1, dtype=np.uint32))
    draws = rng_b.integers(0, U, size=(BOOT, U))
    boot = []
    for arm in arms:
        p_sorted = p_te_store[arm][rows_concat]
        y_sorted = y_te[rows_concat]
        aucs = np.empty(BOOT)
        for bi in range(BOOT):
            mult = np.bincount(draws[bi], minlength=U)
            sel = np.repeat(np.arange(len(rows_concat)), mult[user_of_row])
            aucs[bi] = roc_auc_score(y_sorted[sel], p_sorted[sel])
        boot.append((arm, r, float(aucs.mean()),
                     float(np.quantile(aucs, 0.025)),
                     float(np.quantile(aucs, 0.975)),
                     metrics["arms"][arm]["auroc_test"]))

    preds_r0 = None
    if r == 0:
        rows = [pd.DataFrame({"arm": arm, "user": test_users,
                              "date": test_dates, "y": y_te,
                              "p": p_te_store[arm]}) for arm in arms]
        preds_r0 = pd.concat(rows, ignore_index=True)

    return {"metrics": metrics, "gini": gini_out, "pi": pi_out,
            "scaling": scaling, "bootstrap": boot,
            "user_scores": user_scores, "preds_r0": preds_r0}


def build_day_table():
    cache_path = CACHE / "day_table.parquet"
    CACHE.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    cm = pd.read_parquet(REPO / "artifacts_sq" /
                         "cohort_manifest_model_sources.parquet")
    s3_users = set(cm[cm.source_id == SRC].user_id.tolist())
    split = pd.read_parquet(PATH_SPLIT)
    y_map = split[split.source_id == SRC].set_index("user_id")["y"].astype(int)

    sal = pd.read_csv(PATH_SALUTATION, usecols=["user_id", "birth_date"])
    sal["birth_year"] = pd.to_numeric(sal.birth_date, errors="coerce")
    bmi = pd.read_parquet(PATH_FEATS_ALL,
                          columns=["source_id", "user_id", "demo__bmi_grp"])
    bmi = bmi[bmi.source_id == SRC][["user_id", "demo__bmi_grp"]]

    ed = pd.read_parquet(PATH_EPOCH_DAYS)
    ed = ed[ed.user.isin(s3_users)].copy()

    h3 = ed[ed.channel == 3000]
    adequate = (h3.hours >= 8) & ((h3.cov_s >= 8 * 3600) | (h3.n >= 60))
    h3_ad = h3[adequate][["user", "date", "mean", "median", "sd", "vmin",
                          "vmax", "n", "cov_s", "hours"]].assign(channel=3000)
    pairs = h3_ad[["user", "date"]].drop_duplicates()
    ed_o = ed[ed.channel.isin([3001, 3002])].merge(
        pairs, on=["user", "date"], how="inner")
    ed_keep = pd.concat(
        [h3_ad, ed_o[["user", "date", "channel", "mean", "median", "sd",
                      "vmin", "vmax", "n", "cov_s", "hours"]]],
        ignore_index=True)
    ed_keep["cov_h"] = (ed_keep["cov_s"] / 3600.0).astype("float32")

    piv = ed_keep.pivot_table(
        index=["user", "date"], columns="channel",
        values=["mean", "median", "sd", "vmin", "vmax", "n", "cov_h", "hours"],
        aggfunc="first")
    piv.columns = [f"d_ch{c}_{s}" for s, c in piv.columns]
    piv = piv.reset_index()

    for hi, lo in ((3000, 3001), (3000, 3002)):
        piv[f"d_gap_{hi}_{lo}_mean"] = piv[f"d_ch{hi}_mean"] - piv[f"d_ch{lo}_mean"]
        piv[f"d_gap_{hi}_{lo}_sd"] = piv[f"d_ch{hi}_sd"] - piv[f"d_ch{lo}_sd"]

    piv = piv.merge(sal[["user_id", "birth_year"]], left_on="user",
                    right_on="user_id", how="left").drop(columns=["user_id"])
    piv = piv.merge(bmi, left_on="user", right_on="user_id", how="left") \
            .drop(columns=["user_id"])
    piv = piv.merge(y_map.rename("y"), left_on="user", right_index=True,
                    how="left")
    dt = pd.to_datetime("1970-01-01") + pd.to_timedelta(piv["date"].astype(int), "D")
    piv["age_at_day"] = (dt.dt.year - piv["birth_year"] - 1 +
                         (dt.dt.month >= 12)).astype("float32")
    piv = piv.drop(columns=["birth_year"])

    piv = piv.sort_values(["user", "date"]).reset_index(drop=True)
    rng = np.random.default_rng(np.random.SeedSequence([BASELINE_SEED]))
    is_base = np.zeros(len(piv), dtype=bool)
    for _, idx in piv.groupby("user", sort=True).indices.items():
        n = len(idx)
        n_base = min(60, n // 4)
        if n_base > 0:
            is_base[rng.choice(idx, size=n_base, replace=False)] = True
    piv["is_baseline"] = is_base

    fixed = ["user", "date", "y", "is_baseline", "age_at_day", "demo__bmi_grp"]
    out = piv[fixed + STAT_COLS + GAP_COLS].copy()
    for c in STAT_COLS + GAP_COLS:
        out[c] = out[c].astype("float32")
    out.to_parquet(cache_path, index=False)
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_report(RESULTS, met, scal, boots, fi, sa_m):
    L = []
    L.append("# Day-level salutation classification (Garmin) — results\n")
    L.append("Implements [`../PLAN.md`](../PLAN.md). 30 FS-phase participant "
             "splits (paired); 5 workers, n_jobs=1 per fit.\n")
    L.append("Reference lines: participant-level AUROC 0.7445 (frozen A3_k50), "
             "0.7470 (AGE_all, age experiment).\n")

    L.append("## 1. Day-level AUROC (mean ± SD over 30 repeats)\n")
    L.append("| arm | test AUROC | val AUROC | ECE (val) | n features | "
             "train days | val days | test days |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        sub = met[met.arm == arm]
        L.append(f"| `{arm}` | {sub.auroc_test.mean():.4f} ± "
                 f"{sub.auroc_test.std(ddof=1):.4f} | {sub.auroc_val.mean():.4f} | "
                 f"{sub.ece_val.mean():.4f} | {int(sub.n_features.mean())} | "
                 f"{int(sub.n_train_days.mean()):,} | "
                 f"{int(sub.n_val_days.mean()):,} | "
                 f"{int(sub.n_test_days.mean()):,} |")
    for arm in SA_ARMS:
        sub = met[met.arm == arm]
        if len(sub):
            L.append(f"| `{arm}` (r=0) | {sub.auroc_test.iloc[0]:.4f} | "
                     f"{sub.auroc_val.iloc[0]:.4f} | {sub.ece_val.iloc[0]:.4f} | "
                     f"{int(sub.n_features.iloc[0])} | "
                     f"{int(sub.n_train_days.iloc[0]):,} | "
                     f"{int(sub.n_val_days.iloc[0]):,} | "
                     f"{int(sub.n_test_days.iloc[0]):,} |")

    L.append("\n## 2. Participant-cluster bootstrap 95% CI (day AUROC)\n")
    L.append("| arm | exact AUROC (mean) | boot mean | mean q2.5 | mean q97.5 |")
    L.append("|---|---|---|---|---|")
    for arm in ARMS:
        sub = boots[boots.arm == arm]
        L.append(f"| `{arm}` | {sub.exact_auroc_test.mean():.4f} | "
                 f"{sub.boot_mean.mean():.4f} | {sub.boot_q025.mean():.4f} | "
                 f"{sub.boot_q975.mean():.4f} |")

    L.append("\n## 3. k-day scaling curve (participant AUROC, mean over "
             "30 repeats)\n")
    L.append("| k (days) | A | B | A+covs | n users (A) |")
    L.append("|---|---|---|---|---|")
    kk = sorted([k for k in scal[scal.arm == "A"].k.unique() if k != "all"])
    for k in kk:
        row = [f"{k}"]
        for arm in ARMS:
            sub = scal[(scal.arm == arm) & (scal.k == k)]
            row.append(f"{sub.auc_mean.mean():.4f} ± "
                       f"{sub.auc_mean.std(ddof=1):.4f}")
        n_u = int(scal[(scal.arm == 'A') & (scal.k == k)].n_users.mean())
        L.append("| " + " | ".join(row + [str(n_u)]) + " |")
    row = ["all"]
    for arm in ARMS:
        sub = scal[(scal.arm == arm) & (scal.k == "all")]
        row.append(f"{sub.auc_mean.mean():.4f} ± {sub.auc_mean.std(ddof=1):.4f}")
    L.append("| " + " | ".join(row + [str(int(scal[(scal.arm == 'A') & (scal.k == 'all')].n_users.mean()))]) + " |")

    L.append("\n## 4. Covariates decomposition (A+covs − A, paired by repeat)\n")
    from scipy import stats as st
    for k in kk + ["all"]:
        a = scal[(scal.arm == "A_covs") & (scal.k == k)].sort_values("repeat") \
            .auc_mean.to_numpy()
        b = scal[(scal.arm == "A") & (scal.k == k)].sort_values("repeat") \
            .auc_mean.to_numpy()
        d = a - b
        half = st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        L.append(f"- k={k}: Δ={d.mean():+.4f} "
                 f"[{d.mean()-half:+.4f}, {d.mean()+half:+.4f}]")

    L.append("\n## 5. Top day-level features (arm A; Gini and PI averaged "
             "over 30 repeats)\n")
    L.append("| feature | Gini (mean) | PI mean (mean) |")
    L.append("|---|---|---|")
    top = fi.groupby("feature").agg(gini=("gini", "mean"),
                                    pi=("pi_mean", "mean")) \
        .sort_values("gini", ascending=False).head(15)
    for name, row in top.iterrows():
        L.append(f"| `{name}` | {row.gini:.4f} | {row.pi:.5f} |")

    L.append("\n## 6. Interpretation\n")
    L.append("_(filled after results inspection)_\n")
    L.append("## 7. Caveats\n")
    L.append("- Clustered units (days within participants): the 30-repeat SD "
             "is split noise; §2 bootstrap is day-pooling uncertainty.")
    L.append("- No within-day time-of-day decomposition (not cached); "
             "deferred to a targeted rescan.")
    L.append("- Adequate-day selection may interact with sex via compliance "
             "(medians M 467 / F 482 adequate days — near-null, recorded).")
    L.append("- ch3001/ch3002 vendor-processing caveat unchanged; ch3001 is "
             "literally a daily vendor value.")
    L.append("- Label = recorded salutation, not biological sex/gender.")
    L.append("- Exploratory; arm-B baselines are per-user, label-blind, "
             "own-days-only (transductive).")
    open(RESULTS / "REPORT.md", "w").write("\n".join(L))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[dayscale] start {datetime.now().isoformat(timespec='seconds')} "
          f"| mode={mode}", flush=True)

    print("[dayscale] building day table ...", flush=True)
    day = build_day_table()
    print(f"[dayscale] day table: {len(day):,} rows x {day.shape[1]} cols | "
          f"{day.user.nunique()} users ({time.time()-t0:.0f}s)", flush=True)
    cache_path = CACHE / "day_table.parquet"

    if mode == "bench":
        t = time.time()
        res = run_repeat(0, cache_path)
        for arm in ARMS:
            m = res["metrics"]["arms"][arm]
            print(f"[dayscale] bench r=0 {arm:7s}: fit {m['fit_seconds']:7.1f}s "
                  f"| test AUROC {m['auroc_test']:.4f} | "
                  f"train days {m['n_train_days']:,}", flush=True)
        print(f"[dayscale] bench total {time.time()-t:.0f}s -> est. full ~ "
              f"{(time.time()-t) * (R + 2) / N_JOBS_OUTER / 60:.0f} min wall",
              flush=True)
        return

    t = time.time()
    tasks = [delayed(run_repeat)(r, cache_path) for r in range(R)]
    tasks += [delayed(run_repeat)(0, cache_path, cap=capv, arms=("A",),
                                   sa_label=lab, light=True)
              for lab, capv in SA_ARMS.items()]
    out = Parallel(n_jobs=N_JOBS_OUTER, backend="loky", verbose=0)(tasks)
    print(f"[dayscale] all repeats + SA done ({time.time()-t:.0f}s)", flush=True)

    rep_results = out[:R]
    sa_results = out[R:]

    met_rows = []
    for r in range(R):
        for arm, m in rep_results[r]["metrics"]["arms"].items():
            met_rows.append({"repeat": r, "arm": arm, **m})
    for sa in sa_results:
        for arm, m in sa["metrics"]["arms"].items():
            met_rows.append({"repeat": 0, "arm": arm, **m})
    met = pd.DataFrame(met_rows)
    met.to_csv(RESULTS / "dayscale_metrics.csv", index=False)

    scal_rows = [dict(zip(("arm", "repeat", "k", "auc_mean", "auc_sd",
                           "n_users"), row))
                 for r in range(R) for row in rep_results[r]["scaling"]]
    scal = pd.DataFrame(scal_rows)
    scal.to_csv(RESULTS / "scaling_curve.csv", index=False)

    boot_rows = [dict(zip(("arm", "repeat", "boot_mean",
                           "boot_q025", "boot_q975", "exact_auroc_test"), row))
                 for r in range(R) for row in rep_results[r]["bootstrap"]]
    boots = pd.DataFrame(boot_rows)
    boots.to_csv(RESULTS / "cluster_bootstrap.csv", index=False)

    fi_rows = []
    for r in range(R):
        names, gini = rep_results[r]["gini"]["A"]
        if rep_results[r]["pi"] is not None:
            _, pi_m, pi_s = rep_results[r]["pi"]
        else:
            pi_m = np.full(len(names), np.nan)
            pi_s = np.full(len(names), np.nan)
        for i, nm in enumerate(names):
            fi_rows.append({"repeat": r, "feature": nm, "gini": float(gini[i]),
                            "pi_mean": float(pi_m[i]), "pi_sd": float(pi_s[i])})
    fi = pd.DataFrame(fi_rows)
    fi.to_csv(RESULTS / "feature_importance.csv", index=False)

    us = pd.DataFrame([row for r in range(R)
                        for row in rep_results[r]["user_scores"]])
    us.to_parquet(RESULTS / "user_scores_all_repeats.parquet", index=False)

    r0 = rep_results[0]["preds_r0"]
    if r0 is not None:
        r0.to_parquet(RESULTS / "day_predictions_r0.parquet", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        colors = {"A": "#1f77b4", "B": "#ff7f0e", "A_covs": "#2ca02c"}
        for arm in ARMS:
            sub = scal[(scal.arm == arm) & (scal.k != "all")].copy()
            sub["k"] = sub["k"].astype(int)
            grp = sub.groupby("k").agg(m=("auc_mean", "mean"),
                                       s=("auc_mean", "std"))
            ax.plot(grp.index, grp["m"], "o-", color=colors[arm], label=arm)
            ax.fill_between(grp.index, grp["m"] - grp["s"] / np.sqrt(R),
                            grp["m"] + grp["s"] / np.sqrt(R),
                            color=colors[arm], alpha=0.18)
        for arm in ARMS:
            v = scal[(scal.arm == arm) & (scal.k == "all")].auc_mean.mean()
            ax.axhline(v, color=colors[arm], ls=":", alpha=0.5)
        ax.axhline(0.7445, color="black", ls="--", alpha=0.6,
                   label="participant A3_k50 0.7445")
        ax.axhline(0.7470, color="black", ls="-.", alpha=0.6,
                   label="participant AGE_all 0.7470")
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("k (days pooled)")
        ax.set_ylabel("participant AUROC")
        ax.set_title("Garmin day-level salutation — k-day scaling curve\n"
                     "(mean over 30 repeats, ±SE band; dotted = k=all)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "scaling_curve.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("[dayscale] matplotlib unavailable; figure skipped", flush=True)

    write_report(RESULTS, met, scal, boots, fi, None)

    repro = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "joblib": joblib.__version__,
        "config": {"FS_SEED": FS_SEED, "BASELINE_SEED": BASELINE_SEED,
                   "SRC": SRC, "R": R, "K_CAP": K_CAP, "VI_DAY": VI_DAY,
                   "N_TREES": N_TREES, "PI_REPEATS": PI_REPEATS, "G1": G1,
                   "N_JOBS_OUTER": N_JOBS_OUTER, "KS": list(KS),
                   "TRIALS": TRIALS, "BOOT": BOOT, "PI_SUB": PI_SUB,
                   "ARMS": list(ARMS), "SA_ARMS": SA_ARMS},
        "input_sha256": {
            "epoch_days.parquet": _sha256(PATH_EPOCH_DAYS),
            "split_manifest_sq.parquet": _sha256(PATH_SPLIT),
            "features_all.parquet": _sha256(PATH_FEATS_ALL),
            "13Aug_1222.csv": _sha256(PATH_SALUTATION)},
        "wall_seconds": round(time.time() - t0, 1),
    }
    json.dump(repro, open(RESULTS / "dayscale_repro.json", "w"), indent=1)
    print(f"[dayscale] results written to {RESULTS}", flush=True)
    print(f"[dayscale] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
