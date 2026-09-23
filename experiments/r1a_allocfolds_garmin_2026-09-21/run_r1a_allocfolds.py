#!/usr/bin/env python
"""R1a-AF — R1a circadian curve features under demographically-balanced
CP-SAT allocator folds (user's `cohort_allocator` package, used as-is).
Implements PLAN.md. Read-only w.r.t. the rest of the repo.

Imports the committed R1a runner for feature construction, residualisation,
A1 hygiene and fit config (single source of truth). Only the split changes:
label-stratified random -> allocator folds.

CLI:
  python run_r1a_allocfolds.py           # full run (R=10 allocations x 5 arms; 4 workers)
  python run_r1a_allocfolds.py bench     # allocation r=0 + arm timings, no report
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
R1A_DIR = REPO / "experiments" / "r1a_circadian_garmin_2026-09-21"
sys.path.insert(0, str(R1A_DIR))
import run_r1a as r1a  # noqa: E402  (committed frozen R1a runner, commit 4868482)

sys.path.insert(0, str(REPO / "cohort_allocator" / "src"))
from cohort_allocator import allocate_cohort, make_config  # noqa: E402

try:
    import ortools
    ORT_VERSION = getattr(ortools, "__version__", "unknown")
except Exception:
    ORT_VERSION = "unavailable"

# ---- frozen protocol constants ---------------------------------------------
SRC = r1a.SRC                              # 3
R = 10                                     # user ruling 2026-09-21
N_TREES = r1a.N_TREES
G1 = r1a.G1
ARMS = r1a.ARMS
N_JOBS_OUTER = 4                           # user grant 2026-09-21 (dayscale owns 4)
ALLOC_SEED_BASE = 20260921
ALLOC_TIME_LIMIT = 300                     # bench worst phase 3.8 s; safety margin
R1A_FROZEN_BASE = 0.7434                   # R1a frozen-split BASE mean (descriptive)
BAND_MID = {"20-29": 25, "30-39": 35, "40-49": 45,
            "50-59": 55, "60-69": 65, "70+": 75}

RESULTS = HERE / "results"
CACHE = HERE / "cache"
R1A_CACHE = r1a.CACHE / "r1a_features_epoch_hours.parquet"
PATH_SAL = REPO / "data" / "13Aug_1222.csv"


def log(msg):
    print(msg, flush=True)


def alloc_seed(r: int) -> int:
    return int(np.random.SeedSequence([ALLOC_SEED_BASE, SRC, r])
               .generate_state(1, dtype=np.uint32)[0])


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- demographics (deterministic; cached) -----------------------------------
def build_demographics() -> pd.DataFrame:
    """s3 cohort x (user_id, salutation, age_band, bmi_group).

    window_start is DAYS since epoch; age uses the Dec-1 cutoff convention
    from the age experiment: age = ws.year - birth_year - 1 + [ws.month >= 12].
    demo__bmi_grp is a single categorical column (not one-hot).
    """
    sal = pd.read_csv(PATH_SAL, usecols=["user_id", "salutation", "birth_date"])
    sal["birth_year"] = pd.to_numeric(sal.birth_date, errors="coerce")

    cm = pd.read_parquet(r1a.PATH_COHORT)
    cm3 = cm[cm.source_id == SRC][["user_id", "window_start"]].copy()
    m = cm3.merge(sal[["user_id", "salutation", "birth_year"]],
                  on="user_id", how="left")
    if m.salutation.isna().any() or m.birth_year.isna().any():
        raise SystemExit(
            f"demographics incomplete: salutation NaN {int(m.salutation.isna().sum())}, "
            f"birth_year NaN {int(m.birth_year.isna().sum())}")

    ws = pd.Timestamp("1970-01-01") + pd.to_timedelta(m.window_start.astype(int), "D")
    m["age"] = (ws.dt.year - m.birth_year - 1 + (ws.dt.month >= 12)).astype(float)
    if m.age.isna().any():
        raise SystemExit(f"age NaN for {int(m.age.isna().sum())} users")
    a = m.age
    m["age_band"] = np.select(
        [a < 30, a < 40, a < 50, a < 60, a < 70],
        ["20-29", "30-39", "40-49", "50-59", "60-69"], default="70+")

    fa = pd.read_parquet(r1a.PATH_FEATS,
                         columns=["user_id", "source_id", "demo__bmi_grp"])
    fa3 = fa[fa.source_id == SRC].copy()
    fa3["bmi_group"] = fa3.demo__bmi_grp.astype("string").fillna("<MISSING>")

    demo = (m[["user_id", "salutation", "age_band"]]
            .merge(fa3[["user_id", "bmi_group"]], on="user_id", how="left"))
    if demo.bmi_group.isna().any():
        raise SystemExit(f"bmi_group NaN for {int(demo.bmi_group.isna().sum())} users")
    demo["salutation"] = demo.salutation.astype("Int64").astype(str)
    if len(demo) != 3848 or demo.user_id.duplicated().any():
        raise SystemExit(f"unexpected cohort shape: {demo.shape}")
    return demo[["user_id", "salutation", "age_band", "bmi_group"]]


def load_demographics() -> pd.DataFrame:
    p = CACHE / "demographics.parquet"
    if p.exists():
        return pd.read_parquet(p)
    demo = build_demographics()
    demo.to_parquet(p, index=False)
    return demo


def load_r1a_features() -> pd.DataFrame:
    if R1A_CACHE.exists():
        return pd.read_parquet(R1A_CACHE)
    log("[r1aaf] R1a cache missing -> rebuilding via run_r1a.compute_r1a")
    cm = pd.read_parquet(r1a.PATH_COHORT)
    cohort_users = set(cm[cm.source_id == SRC].user_id.astype("int64"))
    eh, _, _ = r1a.load_hours(cohort_users)
    split = pd.read_parquet(r1a.PATH_SPLIT)
    y_map = split[split.source_id == SRC].set_index("user_id")["y"]
    users_sorted = np.sort(y_map.index.to_numpy().astype("int64"))
    r1a_df = r1a.compute_r1a(eh, users_sorted)
    r1a_df.to_parquet(R1A_CACHE, index=True)
    return r1a_df


# ---- per-allocation worker --------------------------------------------------
def run_alloc(r: int, demo: pd.DataFrame, r1a_df: pd.DataFrame) -> dict:
    split = pd.read_parquet(r1a.PATH_SPLIT)
    y_map = split[split.source_id == SRC].set_index("user_id")["y"].astype(int)
    users = np.sort(y_map.index.to_numpy().astype("int64"))
    y_all = y_map.loc[users].to_numpy(dtype=int)

    # -- allocator folds (frozen spec, PLAN §4) -------------------------------
    seed = alloc_seed(r)
    cfg = make_config(id_column="user_id",
                      stratify_columns=("salutation",),
                      partition_columns=("age_band", "bmi_group"),
                      fold_sizes={"train": 70, "val": 15, "test": 15},
                      seed=seed, time_limit_seconds=ALLOC_TIME_LIMIT)
    t0 = time.time()
    res = allocate_cohort(demo, cfg)
    wall_alloc = time.time() - t0
    parts = res.summary["partitions"]
    all_opt = all(p["all_phases_optimal"] for p in parts)
    asg = res.assignments

    if set(asg.user_id.astype("int64")) != set(users):
        raise SystemExit(f"alloc {r}: assignments do not cover the cohort exactly")

    # per-run audit files (balance table + full allocator summary)
    (asg.groupby(["fold", "age_band", "bmi_group", "salutation"]).size()
        .rename("n").reset_index()
        .to_csv(CACHE / f"balance_run{r}.csv", index=False))
    json.dump({"alloc": r, "alloc_seed": seed, "summary": res.summary},
              open(CACHE / f"alloc_summary_run{r}.json", "w"),
              indent=1, default=str)

    fold_map = asg.set_index("user_id")["fold"]
    fold_map.index = fold_map.index.astype("int64")
    folds = fold_map.loc[users].to_numpy()
    tr = np.flatnonzero(folds == "train")
    va = np.flatnonzero(folds == "val")
    te = np.flatnonzero(folds == "test")
    if len(tr) + len(va) + len(te) != len(users):
        raise SystemExit(f"alloc {r}: fold sizes do not sum to cohort")

    # -- features (identical to R1a, only folds differ) -----------------------
    fa = pd.read_parquet(r1a.PATH_FEATS)
    fa = fa[fa.source_id == SRC].sort_values("user_id").reset_index(drop=True)
    assert list(fa.user_id) == list(users)
    y = fa.user_id.map(y_map).to_numpy(dtype=int)
    X_base = fa[[c for c in fa.columns
                 if c not in ("user_id", "source_id", "y")]].copy()

    r1a_full = r1a_df.reindex(users)
    r1a_feat = r1a_full[r1a.R1A_FEATURES].copy()
    r1a_resid = r1a.residualise(r1a_full, tr)

    rf_s = r1a.rf_seed(r)  # SeedSequence([FS_SEED, SRC, r, 5]) — same stream, new folds

    def eval_arm(X: pd.DataFrame, collect_preds=False, collect_gini=False) -> dict:
        Z_tr, Z_va, Z_te, names = r1a.apply_arm_transform(
            X.iloc[tr], X.iloc[va], X.iloc[te])
        rf = RandomForestClassifier(n_estimators=N_TREES,
                                    random_state=rf_s, n_jobs=1, **G1)
        rf.fit(Z_tr, y[tr])
        j1 = list(rf.classes_).index(1)
        p_va = rf.predict_proba(Z_va)[:, j1]
        p_te = rf.predict_proba(Z_te)[:, j1]
        out = {"n_a0": int(len(names)), "n_hygiene": int(Z_tr.shape[1]),
               "auroc_val": float(roc_auc_score(y[va], p_va)),
               "auroc_test": float(roc_auc_score(y[te], p_te))}
        if collect_gini:
            out["_gini_names"] = list(names)
            out["_gini"] = rf.feature_importances_.astype("float32")
        if collect_preds:
            out["_preds_te"] = (users[te], y[te], p_te)
        return out

    arms = {
        "BASE": eval_arm(X_base, collect_preds=True),
        "R1a_only": eval_arm(r1a_feat),
        "BASE_x_R1a": eval_arm(
            pd.concat([X_base.reset_index(drop=True),
                       r1a_feat.reset_index(drop=True)], axis=1),
            collect_preds=True, collect_gini=True),
        "R1a_resid_only": eval_arm(r1a_resid),
        "BASE_x_R1a_resid": eval_arm(
            pd.concat([X_base.reset_index(drop=True),
                       r1a_resid.reset_index(drop=True)], axis=1),
            collect_preds=True),
    }

    comp = {}
    for f in ("train", "val", "test"):
        sub = asg[asg.fold == f]
        comp[f] = {"n": int(len(sub)),
                   "mean_age": float(sub.age_band.map(BAND_MID).mean()),
                   "sal10_share": float((sub.salutation == "10").mean()),
                   "bmi_normal_share": float((sub.bmi_group == "normal").mean())}

    return {"alloc": r, "alloc_seed": seed, "all_opt": bool(all_opt),
            "n_partitions": len(parts), "wall_alloc": round(wall_alloc, 1),
            "fold_sizes": {f: int((folds == f).sum())
                           for f in ("train", "val", "test")},
            "composition": comp, "arms": arms}


# ---- inference helpers ------------------------------------------------------
def mean_ci(d, q=0.975):
    d = np.asarray(d, dtype=float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(q, n - 1)) * sd / np.sqrt(n)
    return m, sd, half, float((d > 0).mean())


def arr(rows, arm, key="auroc_test"):
    return np.array([rows[i]["arms"][arm][key] for i in range(len(rows))])


# ---- main -------------------------------------------------------------------
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"[r1aaf] start {datetime.now().isoformat(timespec='seconds')} "
        f"| mode={mode} | R={R} | workers={N_JOBS_OUTER}")

    demo = load_demographics()
    log(f"[r1aaf] demographics: {demo.shape[0]} users, "
        f"{demo.groupby(['age_band', 'bmi_group']).ngroups} partitions")
    r1a_df = load_r1a_features()
    log(f"[r1aaf] r1a features: {r1a_df.shape[0]} users x {r1a_df.shape[1]} cols")

    if mode == "bench":
        out = run_alloc(0, demo, r1a_df)
        for arm in ARMS:
            a = out["arms"][arm]
            log(f"[r1aaf] bench r=0 {arm:18s} test {a['auroc_test']:.4f} "
                f"| val {a['auroc_val']:.4f} | cols {a['n_hygiene']}")
        log(f"[r1aaf] bench alloc wall {out['wall_alloc']}s | "
            f"all_opt={out['all_opt']} | partitions={out['n_partitions']} | "
            f"folds={out['fold_sizes']}")
        log(f"[r1aaf] bench total {time.time()-t0:.0f}s")
        return

    rows = Parallel(n_jobs=N_JOBS_OUTER, backend="loky", verbose=0)(
        delayed(run_alloc)(r, demo, r1a_df) for r in range(R))
    rows.sort(key=lambda d: d["alloc"])
    log(f"[r1aaf] {R} allocations x {len(ARMS)} arms done ({time.time()-t0:.0f}s)")

    # -- gate: every allocation all_phases_optimal ---------------------------
    bad = [d["alloc"] for d in rows if not d["all_opt"]]
    if bad:
        raise SystemExit(f"ALLOCATOR GATE FAILED: all_phases_optimal=False "
                         f"for allocations {bad}; inspect cache/alloc_summary_*.json")
    gate_msg = (f"all_phases_optimal=True for {R}/{R} allocations "
                f"({sum(d['n_partitions'] for d in rows)} partition solves)")
    log(f"[r1aaf] allocator gate: {gate_msg}")

    # -- metrics ---------------------------------------------------------------
    recs = []
    for d in rows:
        for arm in ARMS:
            a = d["arms"][arm]
            recs.append({"arm": arm, "alloc": d["alloc"],
                         "auroc_val": a["auroc_val"],
                         "auroc_test": a["auroc_test"],
                         "n_a0": a["n_a0"], "n_hygiene": a["n_hygiene"],
                         "alloc_seed": d["alloc_seed"],
                         "all_opt": d["all_opt"],
                         "n_train": d["fold_sizes"]["train"],
                         "n_val": d["fold_sizes"]["val"],
                         "n_test": d["fold_sizes"]["test"],
                         "wall_alloc_s": d["wall_alloc"]})
    mdf = pd.DataFrame(recs)
    mdf.to_csv(RESULTS / "r1aaf_metrics.csv", index=False)

    # -- predictions (test only; BASE, BASE_x_R1a, BASE_x_R1a_resid) ----------
    prec = []
    for d in rows:
        for arm in ("BASE", "BASE_x_R1a", "BASE_x_R1a_resid"):
            u, yv, p = d["arms"][arm]["_preds_te"]
            prec.append(pd.DataFrame({"arm": arm, "alloc": d["alloc"],
                                      "split": "test", "user_id": u,
                                      "y_true": yv, "p_class1": p}))
    pd.concat(prec, ignore_index=True).to_csv(
        RESULTS / "r1aaf_predictions.csv", index=False)

    # -- top features (Gini in BASE_x_R1a) ------------------------------------
    fi = []
    for d in rows:
        a = d["arms"]["BASE_x_R1a"]
        for nm, g in zip(a["_gini_names"], a["_gini"]):
            fi.append({"alloc": d["alloc"], "feature": nm,
                       "gini": float(g),
                       "is_r1a": nm.startswith(("cosinor_", "curve_", "hour_h"))})
    fi_df = pd.DataFrame(fi)
    fi_df.to_csv(RESULTS / "r1aaf_top_features.csv", index=False)

    # -- report ----------------------------------------------------------------
    base = arr(rows, "BASE")
    d1 = arr(rows, "BASE_x_R1a") - base
    d2 = arr(rows, "BASE_x_R1a_resid") - base
    d3 = arr(rows, "BASE_x_R1a") - arr(rows, "R1a_only")
    d4 = arr(rows, "BASE_x_R1a_resid") - arr(rows, "R1a_resid_only")
    d5 = arr(rows, "R1a_only") - base
    d6 = arr(rows, "R1a_resid_only") - base
    d7 = arr(rows, "R1a_only") - arr(rows, "R1a_resid_only")

    L = []
    L.append("# R1a-AF — circadian curve features under demographically-"
             "balanced CP-SAT folds — results\n")
    L.append(f"Run {datetime.now().isoformat(timespec='seconds')} | implements "
             "[`../PLAN.md`](../PLAN.md) | R = 10 allocator allocations "
             f"(partition = age_band x bmi_group, stratify = salutation; 70/15/15), "
             "paired within allocation; 5 R1a arms per allocation, RF seed stream "
             f"identical to R1a (`SeedSequence([20260919, 3, r, 5])`); "
             f"{N_JOBS_OUTER} workers, n_jobs=1 per fit.\n")
    L.append(f"**Allocator gate:** {gate_msg} (ortools {ORT_VERSION}).\n")
    L.append(f"**Univariate AUROCs:** unchanged by folds — see "
             "`../r1a_circadian_garmin_2026-09-21/results/r1a_univariate.csv`.\n")

    L.append(f"## 1. Absolute AUROC (mean ± SD over {R} allocations)\n")
    L.append("| arm | test AUROC | val AUROC | n cols (hygiene) |")
    L.append("|---|---|---|---|")
    for arm in ARMS:
        t, v = arr(rows, arm), arr(rows, arm, "auroc_val")
        nh = int(np.mean([d["arms"][arm]["n_hygiene"] for d in rows]))
        L.append(f"| `{arm}` | {t.mean():.4f} ± {t.std(ddof=1):.4f} | "
                 f"{v.mean():.4f} | {nh} |")
    L.append(f"\n**BASE sanity (descriptive, different test users):** allocator "
             f"folds {base.mean():.4f} ± {base.std(ddof=1):.4f} vs R1a "
             f"frozen-split {R1A_FROZEN_BASE:.4f} (Δ {base.mean()-R1A_FROZEN_BASE:+.4f}). "
             "Not a paired test — split systems assign different users to test.\n")

    L.append("## 2. Paired test-AUROC deltas within allocation\n")
    L.append("### 2a. Primary family (2 comparisons)\n")
    L.append("| comparison | mean Δ | 95% t-CI | 97.5% t-CI (Bonferroni) | SD | share > 0 |")
    L.append("|---|---|---|---|---|---|")
    for name, d in (("BASE⊕R1a − BASE", d1),
                    ("BASE⊕R1a_resid − BASE", d2)):
        m, sd, h95, share = mean_ci(d, 0.975)
        _, _, h975, _ = mean_ci(d, 0.9875)
        L.append(f"| {name} | {m:+.4f} | [{m-h95:+.4f}, {m+h95:+.4f}] | "
                 f"[{m-h975:+.4f}, {m+h975:+.4f}] | {sd:.4f} | {share:.2f} |")
    L.append("\n_R1a frozen-split reference (commit `4868482`): +0.0029 "
             "95% t-CI [+0.0004, +0.0054]; wear-partialled +0.0036 "
             "[+0.0014, +0.0058]. Note: R1a's REPORT labels these CIs "
             '"97.5%" — that is a labeling error (the formula is '
             "`t.ppf(0.975, 29)`, a 95% t-CI); erratum to follow._\n")

    L.append("### 2b. Secondary comparisons\n")
    L.append("| comparison | mean Δ | 95% t-CI | SD | share > 0 |")
    L.append("|---|---|---|---|---|")
    for name, d in (("BASE⊕R1a − R1a_only", d3),
                    ("BASE⊕R1a_resid − R1a_resid_only", d4),
                    ("R1a_only − BASE", d5),
                    ("R1a_resid_only − BASE", d6),
                    ("R1a_only − R1a_resid_only (activity vs composition)", d7)):
        m, sd, half, share = mean_ci(d, 0.975)
        L.append(f"| {name} | {m:+.4f} | [{m-half:+.4f}, {m+half:+.4f}] | "
                 f"{sd:.4f} | {share:.2f} |")

    L.append(f"\n## 3. Top features (BASE⊕R1a Gini, mean over {R})\n")
    L.append("| feature | Gini (mean) | R1a? |")
    L.append("|---|---|---|")
    top = (fi_df.groupby("feature")
           .agg(gini=("gini", "mean"), is_r1a=("is_r1a", "first"))
           .sort_values("gini", ascending=False).head(15))
    for name, row in top.iterrows():
        L.append(f"| `{name}` | {row.gini:.4f} | "
                 f"{'**yes**' if row.is_r1a else 'no'} |")

    c0 = rows[0]["composition"]
    L.append("\n## 4. Fold composition (allocation 0; balance files for all "
             "runs in `cache/balance_run*.csv`)\n")
    L.append("| fold | n | mean age (band mid) | salutation 10 share | bmi normal share |")
    L.append("|---|---:|---:|---:|---:|")
    for f in ("train", "val", "test"):
        c = c0[f]
        L.append(f"| {f} | {c['n']} | {c['mean_age']:.2f} | "
                 f"{c['sal10_share']:.3f} | {c['bmi_normal_share']:.3f} |")
    fs = pd.DataFrame([d["fold_sizes"] for d in rows])
    L.append(f"\nFold sizes over runs: train {fs.train.mean():.1f} "
             f"[{fs.train.min()}, {fs.train.max()}], val {fs.val.mean():.1f} "
             f"[{fs.val.min()}, {fs.val.max()}], test {fs.test.mean():.1f} "
             f"[{fs.test.min()}, {fs.test.max()}] (largest-remainder per "
             "partition; totals vary by ±1–2 users across allocations).\n")

    L.append("\n## 5. Caveats\n")
    L.append("- Same-participant reuse — not external validation; folds are "
             "resamples of one cohort.")
    L.append(f"- **R = 10 under-powers the paired-Δ estimate** (SE ≈ 2× R1a's). "
             "Direction/magnitude check, not confirmatory; Bonferroni 97.5% "
             "CIs are correspondingly wide.")
    L.append("- Allocator tie-break introduces per-allocation variation beyond "
             "stratified permutation; per-allocation BASE spread is expected "
             "to exceed R1a's frozen-split spread.")
    L.append("- Structural integer-granularity imbalance at the "
             "`(20-29, salutation 10)` cell (worst-cell proportion deviation "
             "≈ 0.39, identical across seeds); aggregate composition is "
             "matched to 3 dp (see §4 and balance files).")
    L.append("- ch3001/ch3002 excluded (vendor caveat); vendor-circularity "
             "audit declined (user decision 2026-09-21).")
    L.append("- Allocator reproducibility: single CP-SAT worker + fixed seeds "
             f"+ ortools {ORT_VERSION}; portability across machines/versions "
             "not guaranteed by the package README.\n")
    L.append("## 6. Interpretation\n_(filled after results inspection)_\n")
    open(RESULTS / "REPORT.md", "w").write("\n".join(L))

    repro = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "script_sha256": sha256(Path(__file__)),
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "joblib": joblib.__version__, "ortools": ORT_VERSION,
        "config": {"R": R, "N_JOBS_OUTER": N_JOBS_OUTER,
                   "ARMS": list(ARMS), "N_TREES": N_TREES, "G1": G1,
                   "ALLOC_SEED_BASE": ALLOC_SEED_BASE,
                   "ALLOC_TIME_LIMIT": ALLOC_TIME_LIMIT,
                   "allocator_spec": {
                       "id_column": "user_id",
                       "stratify_columns": ["salutation"],
                       "partition_columns": ["age_band", "bmi_group"],
                       "fold_sizes": {"train": 70, "val": 15, "test": 15}},
                   "rf_seed_stream": "SeedSequence([20260919, 3, r, 5]) "
                                     "(= run_r1a.rf_seed(r))",
                   "alloc_seeds": [d["alloc_seed"] for d in rows]},
        "gate": gate_msg,
        "r1a_runner": {"path": str(r1a.__file__),
                       "sha256": sha256(Path(r1a.__file__))},
        "input_sha256": {
            "features_all.parquet": sha256(r1a.PATH_FEATS),
            "split_manifest_sq.parquet": sha256(r1a.PATH_SPLIT),
            "cohort_manifest_model_sources.parquet": sha256(r1a.PATH_COHORT),
            "13Aug_1222.csv": sha256(PATH_SAL),
            "demographics.parquet": sha256(CACHE / "demographics.parquet"),
            "r1a_features_epoch_hours.parquet": sha256(R1A_CACHE)},
        "wall_seconds": round(time.time() - t0, 1)}
    json.dump(repro, open(RESULTS / "r1aaf_repro.json", "w"), indent=1)
    log(f"[r1aaf] results written to {RESULTS}")
    log(f"[r1aaf] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
