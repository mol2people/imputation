#!/usr/bin/env python
"""Experiment: DOB-derived age counter vs categorical age_group (Garmin only).

Implements PLAN.md in this folder. Read-only w.r.t. the rest of the repo:
imports ``src/sidequest`` modules and reads frozen artifacts; writes only into
``./results/``. Compute: 5 joblib workers, ``n_jobs=1`` per fit/PI (phase
convention; RF is deterministic given the seed).

Run:
  cd <repo>
  ~/micromamba/envs/datenspende/bin/python \
      experiments/age_vs_agegroup_garmin_2026-09-20/run_age_vs_agegroup.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src" / "sidequest"))
from preprocess import SQPreprocessor  # noqa: E402

# ---- frozen protocol constants (mirror src/sidequest/feat_sel.py) -----------
FS_SEED = 20260919
SRC = 3                                    # Garmin
R = 30                                     # repeats
K = 50                                     # top-k
VI_ALL = 5                                 # VARIANTS.index("all")
N_TREES = 100
PI_REPEATS = 3
G1 = {"max_features": 0.4, "min_samples_leaf": 10, "max_depth": None,
      "class_weight": "balanced_subsample"}
N_JOBS_OUTER = 5                           # user cap: 5 cores total
CUTOFF_DATE = pd.Timestamp("2022-12-18")   # last window_end in the data
ARMS = ("REF50", "AGE50", "AGE50_force", "AGE50_nodemo", "AGE_all")

RESULTS = HERE / "results"

ART = REPO / "artifacts_sq"
PATH_SPLIT = ART / "split_manifest_sq.parquet"
PATH_FEATS = ART / "features_all.parquet"
PATH_COHORT = ART / "cohort_manifest_model_sources.parquet"
PATH_SAVED_METRICS = ART / "fsplit" / "fsplit_metrics.csv"
PATH_SAVED_CELLS = ART / "fsplit" / "cells"
PATH_SALUTATION = REPO / "13Aug_1222.csv"


def rf_seed(r: int) -> int:
    return int(np.random.SeedSequence([FS_SEED, SRC, r, VI_ALL]).generate_state(
        1, dtype=np.uint32)[0])


def pi_seed(r: int) -> int:
    return int(np.random.SeedSequence([FS_SEED, SRC, r, VI_ALL, 99])
               .generate_state(1, dtype=np.uint32)[0])


def stratified_split(y: np.ndarray, rng: np.random.Generator):
    """FS-phase split: per class, permutation, test first, then val, then train."""
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


def nzv_keep(Z: np.ndarray) -> np.ndarray:
    """Caret nzv rule on train: dominant value > 95% AND unique/n < 10%."""
    n = Z.shape[0]
    keep = np.ones(Z.shape[1], dtype=bool)
    for j in range(Z.shape[1]):
        vals, counts = np.unique(Z[:, j], return_counts=True)
        if counts.max() / n > 0.95 and len(vals) / n < 0.10:
            keep[j] = False
    return keep


def build_age_table() -> pd.DataFrame:
    """age at window_start, birthday = Dec 1 of the birth year (user convention).

    birth_date in the salutation export is the birth YEAR only. Dec-1 rollover:
        age = ws.year - B - 1 + 1[ws.month == 12]
    Sensitivity anchor: fixed data cutoff 2022-12-18 -> age_fixed = 2022 - B.
    """
    sal = pd.read_csv(PATH_SALUTATION, usecols=["user_id", "birth_date"])
    sal["birth_year"] = pd.to_numeric(sal["birth_date"], errors="coerce")
    cm = pd.read_parquet(PATH_COHORT)
    cm = cm[cm.source_id == SRC][["user_id", "window_start"]]
    m = cm.merge(sal[["user_id", "birth_year"]], on="user_id", how="left")
    assert m.birth_year.notna().all(), "missing birth year in s3 cohort"
    ws = pd.Timestamp("1970-01-01") + pd.to_timedelta(
        m.window_start.astype(int), "D")
    age_ws = (ws.dt.year - m.birth_year - 1 + (ws.dt.month >= 12)).astype(float)
    age_fix = (CUTOFF_DATE.year - m.birth_year - 1 +
               (CUTOFF_DATE.month >= 12)).astype(float)
    out = pd.DataFrame({"user_id": m.user_id.to_numpy(),
                        "age": age_ws.to_numpy(),
                        "age_fixed": age_fix.to_numpy()})
    return out.set_index("user_id")


def run_repeat(r: int, age_tbl: pd.DataFrame) -> dict:
    split = pd.read_parquet(PATH_SPLIT)
    y_map = split[split.source_id == SRC].set_index("user_id")["y"].astype(int)
    users = np.sort(y_map.index.to_numpy())
    y_all = y_map.loc[users].to_numpy(dtype=int)

    fa = pd.read_parquet(PATH_FEATS)
    fr = fa[fa.source_id == SRC].sort_values("user_id").reset_index(drop=True)
    assert list(fr.user_id) == list(users)

    rng = np.random.default_rng(np.random.SeedSequence([FS_SEED, SRC, r]))
    tr, va, te = stratified_split(y_all, rng)
    assert len(tr) + len(va) + len(te) == len(users)

    feat_cols = [c for c in fr.columns
                 if c not in ("user_id", "source_id", "y")]
    y = fr.user_id.map(y_map).to_numpy(dtype=int)
    age = fr.user_id.map(age_tbl["age"]).to_numpy(dtype=float)
    assert not np.isnan(age).any()

    X_ref = fr[feat_cols]
    X_age = X_ref.drop(columns=["demo__age_group"])
    X_age.insert(0, "age", age)
    X_nodemo = X_ref.drop(columns=["demo__age_group", "demo__bmi_grp"])
    X_nodemo.insert(0, "age", age)

    seed = rf_seed(r)

    def fit_eval(Z_tr, Z_va, Z_te, cols, y_tr, y_va, y_te):
        rf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                    n_jobs=1, **G1)
        rf.fit(Z_tr[:, cols], y_tr)
        j1 = list(rf.classes_).index(1)
        p_va = rf.predict_proba(Z_va[:, cols])[:, j1]
        p_te = rf.predict_proba(Z_te[:, cols])[:, j1]
        return rf, p_va, p_te

    def run_matrix(X: pd.DataFrame, do_pi: bool) -> dict:
        """A1 hygiene fit; optional PI ranking + top-K refits.

        Returns hygiene metrics, PI diagnostics and, if do_pi, the top-K
        ('compete') and force ('age' + top-49 others) refits. AGE50_force
        shares the PI ranking with AGE50 by construction (same matrix, same
        seeds).
        """
        prep = SQPreprocessor().fit(X.iloc[tr])
        Z_tr, Z_va, Z_te = (prep.transform(X.iloc[i]) for i in (tr, va, te))
        names = np.array(prep.feature_names_out_)

        ind = np.array([n.endswith("__missing") for n in names])
        base = ~ind
        sub = nzv_keep(Z_tr[:, base])
        a1 = base.copy()
        a1[np.flatnonzero(base)[~sub]] = False

        rf1, p_va_h, p_te_h = fit_eval(Z_tr, Z_va, Z_te, a1,
                                       y[tr], y[va], y[te])
        res = {
            "n_a0": int(len(names)),
            "n_hygiene": int(a1.sum()),
            "auroc_val_hygiene": float(roc_auc_score(y[va], p_va_h)),
            "auroc_test_hygiene": float(roc_auc_score(y[te], p_te_h)),
        }
        preds = {"hygiene": (p_va_h, p_te_h)}
        if not do_pi:
            res["_preds"] = preds
            res["_uids"] = (users[va], users[te])
            res["_y"] = (y[va], y[te])
            return res

        pi = permutation_importance(rf1, Z_va[:, a1], y[va],
                                    scoring="roc_auc", n_repeats=PI_REPEATS,
                                    n_jobs=1, random_state=pi_seed(r))
        order = np.argsort(-pi.importances_mean, kind="stable")
        a1_pos = np.flatnonzero(a1)
        a1_names = names[a1]
        age_pos = np.flatnonzero(a1_names == "age")

        def mask_from_positions(positions):
            cols = a1.copy()
            cols[:] = False
            cols[a1_pos[positions]] = True
            return cols

        top = order[:K]                                   # 'compete' arm
        cols_top = mask_from_positions(top)
        _, p_va_k, p_te_k = fit_eval(Z_tr, Z_va, Z_te, cols_top,
                                     y[tr], y[va], y[te])
        res["auroc_val"] = float(roc_auc_score(y[va], p_va_k))
        res["auroc_test"] = float(roc_auc_score(y[te], p_te_k))
        res["topk"] = [str(n) for n in a1_names[top]]
        preds["topk"] = (p_va_k, p_te_k)

        if len(age_pos) > 0:                # matrices that carry `age`
            top_forced = order[order != age_pos[0]][:K - 1]
            cols_force = mask_from_positions(top_forced)
            cols_force[a1_pos[age_pos[0]]] = True
            _, p_va_f, p_te_f = fit_eval(Z_tr, Z_va, Z_te, cols_force,
                                         y[tr], y[va], y[te])
            res["auroc_val_force"] = float(roc_auc_score(y[va], p_va_f))
            res["auroc_test_force"] = float(roc_auc_score(y[te], p_te_f))
            res["topk_force"] = [str(n) for n in
                                 np.concatenate((["age"],
                                                 a1_names[top_forced]))]
            preds["force"] = (p_va_f, p_te_f)

            j_age = int(age_pos[0])
            res["age_pi"] = float(pi.importances_mean[j_age])
            res["age_rank_overall"] = int(np.flatnonzero(order == j_age)[0]) + 1
            res["age_in_topk"] = bool(res["age_rank_overall"] <= K)
        res["_preds"] = preds
        res["_uids"] = (users[va], users[te])
        res["_y"] = (y[va], y[te])
        return res

    out = {"repeat": r, "arms": {}}
    out["arms"]["REF50"] = run_matrix(X_ref, True)
    age_res = run_matrix(X_age, True)          # AGE50 + AGE50_force + AGE_all
    out["arms"]["AGE50"] = dict(age_res)      # incl. private keys for preds
    out["arms"]["AGE50_force"] = {
        "n_a0": age_res["n_a0"], "n_hygiene": age_res["n_hygiene"],
        "auroc_val": age_res["auroc_val_force"],
        "auroc_test": age_res["auroc_test_force"],
        "topk": age_res["topk_force"],
        "age_pi": age_res["age_pi"],
        "age_rank_overall": age_res["age_rank_overall"],
        "age_in_topk": True, "_preds": {"topk": age_res["_preds"]["force"]},
        "_uids": age_res["_uids"], "_y": age_res["_y"]}
    out["arms"]["AGE50_nodemo"] = run_matrix(X_nodemo, True)
    out["arms"]["AGE_all"] = {
        "n_a0": age_res["n_a0"], "n_hygiene": age_res["n_hygiene"],
        "auroc_val": age_res["auroc_val_hygiene"],
        "auroc_test": age_res["auroc_test_hygiene"],
        "_preds": {"hygiene": age_res["_preds"]["hygiene"]},
        "_uids": age_res["_uids"], "_y": age_res["_y"]}
    return out


def mean_ci(d: np.ndarray):
    d = np.asarray(d, dtype=float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
    return m, sd, half, float((d > 0).mean())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_ref50(rows) -> str:
    """REF50 replica must reproduce frozen A3_k50 (and its A1 hygiene baseline)."""
    saved = pd.read_csv(PATH_SAVED_METRICS)
    sv = saved[(saved.source == SRC) & (saved.variant == "all")]
    max_d = 0.0
    for arm_saved, field in (("A3_k50", None), ("A1", "hygiene")):
        s = sv[sv.arm == arm_saved].sort_values("repeat")
        for r in range(R):
            row = rows[r]["arms"]["REF50"]
            key_v = "auroc_val_hygiene" if field == "hygiene" else "auroc_val"
            key_t = ("auroc_test_hygiene" if field == "hygiene"
                     else "auroc_test")
            max_d = max(max_d,
                        abs(row[key_v] - s[s.repeat == r].auroc_val.iloc[0]),
                        abs(row[key_t] - s[s.repeat == r].auroc_test.iloc[0]))
    set_ok = True
    for r in range(R):
        cell = json.load(open(PATH_SAVED_CELLS / f"{SRC}_{r}.json"))
        ref = set(cell["variants"]["all"]["topk"]["k50"])
        mine = set(rows[r]["arms"]["REF50"]["topk"])
        set_ok &= (ref == mine)
    msg = (f"max |delta AUROC| vs frozen A3_k50/A1 = {max_d:.2e}; "
           f"top-50 sets identical on all {R}/{R} repeats = {set_ok}")
    if max_d > 1e-12 or not set_ok:
        raise SystemExit(f"VALIDATION FAILED: {msg}")
    return msg


def main():
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"[age-exp] start {datetime.now().isoformat(timespec='seconds')}",
          flush=True)

    # ---- age table + univariate context --------------------------------------
    age_tbl = build_age_table()
    rho = float(stats.spearmanr(age_tbl["age"], age_tbl["age_fixed"]).statistic)
    split = pd.read_parquet(PATH_SPLIT)
    s3 = split[split.source_id == SRC].set_index("user_id")["y"].astype(int)
    y_u = s3.loc[age_tbl.index].to_numpy(dtype=int)
    au_age = float(roc_auc_score(y_u, age_tbl["age"].to_numpy()))
    au_age_fix = float(roc_auc_score(y_u, age_tbl["age_fixed"].to_numpy()))
    fa = pd.read_parquet(PATH_FEATS)
    fr = fa[fa.source_id == SRC].sort_values("user_id").reset_index(drop=True)
    elderly = (fr["demo__age_group"] == "Elderly").to_numpy(dtype=int)
    au_eld = float(roc_auc_score(s3.loc[fr.user_id].to_numpy(), elderly))
    print(f"[age-exp] age: n={len(age_tbl)}, "
          f"{age_tbl.age.min():.0f}-{age_tbl.age.max():.0f} y "
          f"(median {age_tbl.age.median():.0f}); "
          f"rho(ws-anchor, cutoff-anchor)={rho:.4f}", flush=True)
    print(f"[age-exp] univariate AUROC: age={au_age:.4f}, "
          f"age_fixed={au_age_fix:.4f}, Elderly one-hot={au_eld:.4f}",
          flush=True)

    # ---- 30 repeats x 5 arms, 5 workers, n_jobs=1 per fit ---------------------
    out = Parallel(n_jobs=N_JOBS_OUTER, backend="loky", verbose=0)(
        delayed(run_repeat)(r, age_tbl) for r in range(R))
    rows = sorted(out, key=lambda d: d["repeat"])
    print(f"[age-exp] repeats done ({time.time()-t0:.0f}s)", flush=True)

    val_msg = validate_ref50(rows)
    print(f"[age-exp] REF50 validation: {val_msg}", flush=True)

    # ---- metrics csv -----------------------------------------------------------
    recs = []
    for r in range(R):
        for arm in ARMS:
            a = rows[r]["arms"][arm]
            recs.append({
                "arm": arm, "repeat": r,
                "auroc_val": a.get("auroc_val", np.nan),
                "auroc_test": a.get("auroc_test", np.nan),
                "auroc_val_hygiene": a.get("auroc_val_hygiene", np.nan),
                "auroc_test_hygiene": a.get("auroc_test_hygiene", np.nan),
                "n_a0": a["n_a0"], "n_hygiene": a["n_hygiene"],
                "age_pi": a.get("age_pi", np.nan),
                "age_rank_overall": a.get("age_rank_overall", np.nan),
                "age_in_topk": a.get("age_in_topk", None)})
    pd.DataFrame(recs).to_csv(RESULTS / "age_metrics.csv", index=False)

    # ---- predictions csv (final evaluated fit per arm) -------------------------
    prec = []
    for r in range(R):
        for arm in ARMS:
            a = rows[r]["arms"][arm]
            uids, yy = a["_uids"], a["_y"]
            which = ("hygiene" if "topk" not in a else "topk")
            p_va, p_te = a["_preds"][which]
            for split_name, u, yv, p in (("val", uids[0], yy[0], p_va),
                                         ("test", uids[1], yy[1], p_te)):
                prec.append(pd.DataFrame({"arm": arm, "repeat": r,
                                          "split": split_name,
                                          "user_id": u, "y_true": yv,
                                          "p_class1": p}))
    pd.concat(prec, ignore_index=True).to_csv(
        RESULTS / "age_predictions.csv", index=False)

    # ---- top-50 sets json --------------------------------------------------------
    sets = {arm: {str(r): rows[r]["arms"][arm]["topk"]
                  for r in range(R) if "topk" in rows[r]["arms"][arm]}
            for arm in ("REF50", "AGE50", "AGE50_force", "AGE50_nodemo")}
    json.dump(sets, open(RESULTS / "age_top50_sets.json", "w"), indent=1)

    # ---- repro json ---------------------------------------------------------------
    repro = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script_sha256": sha256(Path(__file__)),
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "config": {"FS_SEED": FS_SEED, "SRC": SRC, "R": R, "K": K,
                   "VI_ALL": VI_ALL, "N_TREES": N_TREES,
                   "PI_REPEATS": PI_REPEATS, "G1": G1,
                   "N_JOBS_OUTER": N_JOBS_OUTER,
                   "fractions": [0.70, 0.15, 0.15],
                   "age_convention": "birthday = Dec 1 of birth year; "
                                     "anchor = window_start",
                   "cutoff_anchor": str(CUTOFF_DATE.date())},
        "validation": val_msg,
        "input_sha256": {
            "13Aug_1222.csv": sha256(PATH_SALUTATION),
            "features_all.parquet": sha256(PATH_FEATS),
            "split_manifest_sq.parquet": sha256(PATH_SPLIT),
            "cohort_manifest_model_sources.parquet": sha256(PATH_COHORT)},
        "wall_seconds": round(time.time() - t0, 1),
    }
    json.dump(repro, open(RESULTS / "age_repro.json", "w"), indent=1)

    # ---- report -------------------------------------------------------------------
    def s(arm, key):
        return np.array([rows[r]["arms"][arm].get(key, np.nan)
                         for r in range(R)], dtype=float)

    L = []
    L.append("# Age counter vs age_group (Garmin, source 3) — results\n")
    L.append(f"Run {repro['timestamp']} | implements [`../PLAN.md`](../PLAN.md) "
             f"| 30 FS-phase repeats (same splits/seeds), paired; "
             f"5 workers, n_jobs=1 per fit.\n")
    L.append(f"**REF50 replica validation:** {val_msg}\n")

    L.append("## 1. Absolute AUROC (mean ± SD over 30 repeats)\n")
    L.append("| arm | test AUROC | val AUROC (top-50) | val AUROC (hygiene) "
             "| n cols (hygiene) |")
    L.append("|---|---|---|---|---|")
    for arm in ARMS:
        t, v = s(arm, "auroc_test"), s(arm, "auroc_val")
        vh = s(arm, "auroc_val_hygiene")
        nh = int(np.nanmean(s(arm, "n_hygiene")))
        L.append(f"| `{arm}` | {np.nanmean(t):.4f} ± {np.nanstd(t, ddof=1):.4f} "
                 f"| {np.nanmean(v):.4f} | {np.nanmean(vh):.4f} | {nh} |")

    L.append("\n## 2. Paired test-AUROC deltas (same repeats)\n")
    L.append("| comparison | mean Δ | 95% t-CI | SD | share > 0 |")
    L.append("|---|---|---|---|---|")
    ref = s("REF50", "auroc_test")
    comps = [("AGE50 − REF50", s("AGE50", "auroc_test") - ref),
             ("AGE50_force − REF50", s("AGE50_force", "auroc_test") - ref),
             ("AGE50_nodemo − REF50", s("AGE50_nodemo", "auroc_test") - ref),
             ("AGE_all − REF50", s("AGE_all", "auroc_test") - ref),
             ("AGE50_force − AGE50",
              s("AGE50_force", "auroc_test") - s("AGE50", "auroc_test"))]
    for name, d in comps:
        m, sd, half, share = mean_ci(d)
        L.append(f"| {name} | {m:+.4f} | [{m-half:+.4f}, {m+half:+.4f}] "
                 f"| {sd:.4f} | {share:.2f} |")

    in_top = sum(bool(rows[r]["arms"]["AGE50"]["age_in_topk"])
                 for r in range(R))
    ranks = s("AGE50", "age_rank_overall")
    pis = s("AGE50", "age_pi")
    L.append("\n## 3. Age feature inside the AGE50 ranking\n")
    L.append(f"- selected into the top-50 in **{in_top}/{R}** repeats; "
             f"overall PI rank median {int(np.nanmedian(ranks))} "
             f"(min {int(np.nanmin(ranks))}, max {int(np.nanmax(ranks))}).")
    L.append(f"- mean PI value of `age` = {np.nanmean(pis):.5f} "
             f"(AUROC decay per shuffled column, val split).")

    def counter(arm):
        c = Counter()
        for r in range(R):
            if "topk" in rows[r]["arms"][arm]:
                c.update(rows[r]["arms"][arm]["topk"])
        return c

    c_ref, c_age = counter("REF50"), counter("AGE50")
    L.append("\n## 4. AGE50 top-50 membership frequency (top 20)\n")
    L.append("| x/30 | column | REF50 x/30 |")
    L.append("|---|---|---|")
    for name, cnt in c_age.most_common(20):
        L.append(f"| {cnt} | `{name}` | {c_ref.get(name, 0)} |")

    jac_within = np.mean([
        len(set(a) & set(b)) / len(set(a) | set(b))
        for a, b in combinations(
            [rows[r]["arms"]["AGE50"]["topk"] for r in range(R)], 2)])
    jac_cross = np.mean([
        len(set(rows[r]["arms"]["AGE50"]["topk"]) &
            set(rows[r]["arms"]["REF50"]["topk"])) /
        len(set(rows[r]["arms"]["AGE50"]["topk"]) |
            set(rows[r]["arms"]["REF50"]["topk"])) for r in range(R)])
    L.append("\n## 5. Selection stability\n")
    L.append(f"- mean pairwise Jaccard across repeats within `AGE50`: "
             f"{jac_within:.3f} (FS-phase `all` A3_k50 was ≈0.19).")
    L.append(f"- mean same-repeat Jaccard `AGE50` vs `REF50`: "
             f"{jac_cross:.3f}.")
    gained = sorted([n for n, c in c_age.items() if c > c_ref.get(n, 0)],
                    key=lambda n: -c_age[n])[:8]
    lost = sorted([n for n, c in c_ref.items() if c > c_age.get(n, 0)],
                  key=lambda n: -c_ref[n])[:8]
    L.append(f"- more often selected under `AGE50`: "
             f"{', '.join('`'+n+'`' for n in gained)}")
    L.append(f"- less often selected under `AGE50`: "
             f"{', '.join('`'+n+'`' for n in lost)}")

    L.append("\n## 6. Univariate context and anchor sensitivity\n")
    L.append(f"- AUROC of `age` alone (window_start anchor): {au_age:.4f}; "
             f"fixed-cutoff anchor (2022−B): {au_age_fix:.4f}; "
             f"`Elderly` one-hot: {au_eld:.4f}.")
    L.append(f"- Spearman ρ between the two age definitions: {rho:.4f} "
             f"(anchor choice is immaterial at this correlation).")
    L.append(f"- age at window_start: "
             f"{age_tbl.age.min():.0f}–{age_tbl.age.max():.0f} y, "
             f"median {age_tbl.age.median():.0f} (birth years 1935–2000, "
             f"complete for all 3,848).\n")

    L.append("## 7. Caveats\n")
    L.append("- Exploratory; reuses the FS-phase participants and split "
             "seeds — same-partition reuse is never independent validation.")
    L.append("- Age is a linked profile covariate (like the demo block it "
             "replaces), not epoch-derived; its association with salutation "
             "is cohort structure, not physiology.")
    L.append("- Label is recorded salutation, not biological sex/gender; the "
             "vendor-processing caveat for ch3001/ch3002 is unchanged.")
    L.append("- 95% t-CIs quantify split/model randomization conditional on "
             "this cohort, not sampling or transportability.\n")

    L.append("## 8. Interpretation\n")
    L.append("_(to be filled after results inspection)_\n")

    open(RESULTS / "REPORT.md", "w").write("\n".join(L))
    print(f"[age-exp] results written to {RESULTS}", flush=True)
    print(f"[age-exp] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
