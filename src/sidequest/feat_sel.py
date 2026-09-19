"""Side-quest FS phase: feature selection under repeated stratified splits.

Phase spec: ``artifacts_sq/plans/sidequest_feat_sel_repeated_splits_2026-09-19.md``.

Per model-eligible source (3, 6, 7) and R=30 repeats:

  * stratified train/val/test split (70/15/15) drawn from
    ``SeedSequence([FS_SEED, source, r])``; identical across variants (paired);
  * ``SQPreprocessor`` fitted on train_r only, shared by every arm;
  * ``RandomForestClassifier`` G1 config at 100 trees; ``random_state`` derived
    from ``(FS_SEED, source, r, variant)`` and IDENTICAL across arms within a
    cell, so paired arm deltas are deterministic given the split;
  * arms:
      A0       baseline transformed matrix (per-column ``__missing`` indicators in)
      A1       drop all ``__missing`` indicator columns, then the caret
               near-zero-variance rule (dominant value > 95% of train rows AND
               unique-value fraction < 10%); train-fitted, target-blind
      A2       A1 + greedy pairwise |Pearson rho| > 0.95 pruning (keep first in
               canonical order); train-fitted, target-blind
      A3_k50   top-50 columns by permutation importance (roc_auc, 3 repeats) of
               the A1 fit evaluated on val_r; refit on train_r
      A3_k100  top-100, same protocol (recorded no-op when A1 <= k columns)
  * test_r used exactly once per (arm, repeat); val AUROC stored as diagnostic;
  * no bootstrap: across-repeat SD is the uncertainty measure.

Outputs under ``artifacts_sq/<out>/`` (default ``fsplit``):

  cells/{source}_{repeat}.json   per-cell rows + pruned/top-k lists (resume unit)
  fsplit_metrics.csv             assembled long table
  fs_repro.json                  config echo, versions, script hashes
  feat_sel_report.md             paired-delta comparison (written post-run)

Run:  python src/sidequest/feat_sel.py [--repeats 30] [--jobs N]
                                      [--out fsplit] [--sources 3,6,7]
                                      [--report-only]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import scipy
import sklearn
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load_blocks, variant_matrix  # noqa: E402
from preprocess import SQPreprocessor  # noqa: E402
from sq_config import ARTIFACTS_SQ, MODEL_SOURCE_IDS, VARIANTS  # noqa: E402

FS_SEED = 20260919
N_TREES = 100
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
PI_REPEATS = 3
A3_KS = (50, 100)
G1 = {"max_features": 0.4, "min_samples_leaf": 10, "max_depth": None,
      "class_weight": "balanced_subsample"}
ARMS = ("A0", "A1", "A2", "A3_k50", "A3_k100")
NZV_DOMINANT, NZV_UNIQUE, CORR_THRESH = 0.95, 0.10, 0.95

_BLOCKS: dict[str, pd.DataFrame] | None = None


def get_blocks() -> dict[str, pd.DataFrame]:
    """Lazy module-level cache; one load per worker process."""
    global _BLOCKS
    if _BLOCKS is None:
        _BLOCKS = load_blocks()
    return _BLOCKS


# ------------------------------------------------------------------ splitting --

def stratified_split(y: np.ndarray, rng: np.random.Generator
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class shuffle; test/val round to nearest, train absorbs the rest."""
    tr, va, te = [], [], []
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n = len(idx)
        n_te = int(round(TEST_FRAC * n))
        n_va = int(round(VAL_FRAC * n))
        te.append(idx[:n_te])
        va.append(idx[n_te:n_te + n_va])
        tr.append(idx[n_te + n_va:])
    return (np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)),
            np.sort(np.concatenate(te)))


def rf_seed_for(src: int, r: int, variant: str) -> int:
    """Shared by every arm of the (src, r, variant) cell by construction."""
    vi = VARIANTS.index(variant)
    state = np.random.SeedSequence([FS_SEED, src, r, vi]).generate_state(
        1, dtype=np.uint32)
    return int(state[0])


def pi_seed_for(src: int, r: int, variant: str) -> int:
    vi = VARIANTS.index(variant)
    state = np.random.SeedSequence([FS_SEED, src, r, vi, 99]).generate_state(
        1, dtype=np.uint32)
    return int(state[0])


# ------------------------------------------------------------- selection rules --

def nzv_keep(Z: np.ndarray) -> np.ndarray:
    """Caret-style near-zero-variance rule on the train matrix (target-blind)."""
    n = Z.shape[0]
    keep = np.ones(Z.shape[1], dtype=bool)
    for j in range(Z.shape[1]):
        vals, counts = np.unique(Z[:, j], return_counts=True)
        if counts.max() / n > NZV_DOMINANT and len(vals) / n < NZV_UNIQUE:
            keep[j] = False
    return keep


def corr_keep(Z: np.ndarray) -> np.ndarray:
    """Greedy pairwise |rho| > threshold pruning, canonical order kept."""
    C = np.nan_to_num(np.corrcoef(Z, rowvar=False), nan=0.0)
    kept: list[int] = []
    for j in range(Z.shape[1]):
        if all(abs(C[j, i]) <= CORR_THRESH for i in kept):
            kept.append(j)
    mask = np.zeros(Z.shape[1], dtype=bool)
    mask[kept] = True
    return mask


def fit_eval(seed: int, Z_tr: np.ndarray, y_tr: np.ndarray,
             Z_va: np.ndarray, y_va: np.ndarray,
             Z_te: np.ndarray, y_te: np.ndarray, cols: np.ndarray
             ) -> tuple[RandomForestClassifier, float, float, float]:
    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                n_jobs=1, **G1)
    t0 = time.time()
    rf.fit(Z_tr[:, cols], y_tr)
    seconds = time.time() - t0
    out = []
    for Z, y in ((Z_va, y_va), (Z_te, y_te)):
        p1 = rf.predict_proba(Z[:, cols])[:, list(rf.classes_).index(1)]
        out.append(float(roc_auc_score(y, p1)))
    return rf, out[0], out[1], seconds


# ------------------------------------------------------------------ cell work --

def run_cell(src: int, r: int, out_dir: Path) -> list[dict]:
    t_cell = time.time()
    split = pd.read_parquet(ARTIFACTS_SQ / "split_manifest_sq.parquet")
    y_map = split[split["source_id"] == src].set_index("user_id")["y"].astype(int)
    users = np.sort(y_map.index.to_numpy())
    y_all = y_map.loc[users].to_numpy(dtype=int)

    rng = np.random.default_rng(np.random.SeedSequence([FS_SEED, src, r]))
    tr, va, te = stratified_split(y_all, rng)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(users)
    for cls in (0, 1):
        for idx, frac in ((tr, TRAIN_FRAC), (va, VAL_FRAC), (te, TEST_FRAC)):
            got = int((y_all[idx] == cls).sum())
            n_c = int((y_all == cls).sum())
            exp = n_c - int(round(VAL_FRAC * n_c)) - int(round(TEST_FRAC * n_c)) \
                if frac == TRAIN_FRAC else int(round(frac * n_c))
            assert got == exp, f"stratification drift src={src} r={r}"

    blocks = get_blocks()
    rows: list[dict] = []
    variants_out: dict[str, dict] = {}

    for variant in VARIANTS:
        fr = variant_matrix(blocks, variant)
        fr = fr[fr["source_id"] == src].sort_values("user_id").reset_index(drop=True)
        assert list(fr["user_id"]) == list(users), f"{variant}: user set mismatch"
        feat_cols = [c for c in fr.columns if c not in ("user_id", "source_id", "y")]
        fr["y"] = fr["user_id"].map(y_map)
        assert fr["y"].notna().all(), f"{variant}: missing labels"
        X = fr[feat_cols]
        y = fr["y"].to_numpy(dtype=int)
        assert len(feat_cols) == len(set(feat_cols)) and "y" not in feat_cols

        prep = SQPreprocessor().fit(X.iloc[tr])
        Z_tr, Z_va, Z_te = (prep.transform(X.iloc[i]) for i in (tr, va, te))
        names = np.array(prep.feature_names_out_)
        seed = rf_seed_for(src, r, variant)

        # --- A0: baseline matrix ------------------------------------------------
        _, v0, t0, s0 = fit_eval(seed, Z_tr, y[tr], Z_va, y[va], Z_te, y[te],
                                 np.ones(len(names), dtype=bool))
        rows.append({"source": src, "variant": variant, "arm": "A0", "repeat": r,
                     "auroc_val": v0, "auroc_test": t0, "n_features": len(names),
                     "n_ind_dropped": 0, "n_nzv_dropped": 0, "n_corr_dropped": 0,
                     "noop": False, "seconds": s0})

        # --- A1: drop indicators + nzv -----------------------------------------
        ind_mask = np.array([n.endswith("__missing") for n in names])
        base = ~ind_mask
        sub = nzv_keep(Z_tr[:, base])
        a1 = base.copy()
        a1[np.flatnonzero(base)[~sub]] = False
        rf1, v1, t1, s1 = fit_eval(seed, Z_tr, y[tr], Z_va, y[va], Z_te, y[te], a1)
        n_ind = int(ind_mask.sum())
        n_nzv = int((~sub).sum())
        rows.append({"source": src, "variant": variant, "arm": "A1", "repeat": r,
                     "auroc_val": v1, "auroc_test": t1, "n_features": int(a1.sum()),
                     "n_ind_dropped": n_ind, "n_nzv_dropped": n_nzv,
                     "n_corr_dropped": 0, "noop": False, "seconds": s1})

        # --- A2: A1 + correlation dedup ----------------------------------------
        sub_keep = corr_keep(Z_tr[:, a1])
        a2 = a1.copy()
        a2[np.flatnonzero(a1)[~sub_keep]] = False
        _, v2, t2, s2 = fit_eval(seed, Z_tr, y[tr], Z_va, y[va], Z_te, y[te], a2)
        rows.append({"source": src, "variant": variant, "arm": "A2", "repeat": r,
                     "auroc_val": v2, "auroc_test": t2, "n_features": int(a2.sum()),
                     "n_ind_dropped": n_ind, "n_nzv_dropped": n_nzv,
                     "n_corr_dropped": int(a1.sum() - a2.sum()), "noop": False,
                     "seconds": s2})

        # --- A3: top-k permutation importance on val (from the A1 fit) ----------
        topk: dict[str, list[str]] = {}
        noop_k: dict[str, bool] = {}
        arm_n = {"A0": len(names), "A1": int(a1.sum()), "A2": int(a2.sum())}
        a1_names = names[a1]
        if int(a1.sum()) > min(A3_KS):
            pi = permutation_importance(rf1, Z_va[:, a1], y[va], scoring="roc_auc",
                                        n_repeats=PI_REPEATS, n_jobs=1,
                                        random_state=pi_seed_for(src, r, variant))
            order = np.argsort(-pi.importances_mean, kind="stable")
        else:
            order = None
        for k in A3_KS:
            arm = f"A3_k{k}"
            if int(a1.sum()) <= k:
                arm_n[arm] = int(a1.sum())
                rows.append({"source": src, "variant": variant, "arm": arm,
                             "repeat": r, "auroc_val": v1, "auroc_test": t1,
                             "n_features": int(a1.sum()), "n_ind_dropped": n_ind,
                             "n_nzv_dropped": n_nzv, "n_corr_dropped": 0,
                             "noop": True, "seconds": 0.0})
                topk[f"k{k}"] = list(a1_names)
                noop_k[f"k{k}"] = True
                continue
            top = order[:k]
            cols = a1.copy()
            chosen = np.flatnonzero(a1)[top]
            cols[:] = False
            cols[chosen] = True
            _, v3, t3, s3 = fit_eval(seed, Z_tr, y[tr], Z_va, y[va], Z_te, y[te], cols)
            arm_n[arm] = int(cols.sum())
            rows.append({"source": src, "variant": variant, "arm": arm, "repeat": r,
                         "auroc_val": v3, "auroc_test": t3, "n_features": int(cols.sum()),
                         "n_ind_dropped": n_ind, "n_nzv_dropped": n_nzv,
                         "n_corr_dropped": 0, "noop": False, "seconds": s3})
            topk[f"k{k}"] = [str(n) for n in a1_names[top]]
            noop_k[f"k{k}"] = False

        variants_out[variant] = {
            "indicators_dropped": [str(n) for n in names[ind_mask]],
            "nzv_dropped": [str(n) for n in names[base][~sub]],
            "corr_dropped": [str(n) for n in a1_names[~sub_keep]],
            "topk": topk,
            "noop": noop_k,
            "n_features": arm_n,
        }

    payload = {"source": src, "repeat": r, "rows": rows,
               "variants": variants_out, "seconds": round(time.time() - t_cell, 1)}
    (out_dir / "cells" / f"{src}_{r}.json").write_text(json.dumps(payload))
    return rows


def _self_contained(src: int, r: int, out_dir: str) -> str:
    rows = run_cell(src, r, Path(out_dir))
    return f"source {src} repeat {r:>2}: {len(rows)} rows ({rows[0]['seconds']:.1f}s)"


# -------------------------------------------------------------------- outputs --

def assemble(out_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for f in sorted((out_dir / "cells").glob("*.json")):
        rows.extend(json.loads(f.read_text())["rows"])
    if not rows:
        raise RuntimeError(f"no cell files under {out_dir / 'cells'}")
    df = pd.DataFrame(rows)
    assert not df.duplicated(["source", "variant", "arm", "repeat"]).any()
    df = df.sort_values(["source", "variant", "arm", "repeat"]).reset_index(drop=True)
    df.to_csv(out_dir / "fsplit_metrics.csv", index=False)
    return df


def _mean_ci(d: np.ndarray) -> tuple[float, float, float]:
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    ci = float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
    return mean, sd, ci


def build_report(out_dir: Path, repeats: int, sources: tuple[int, ...]) -> None:
    df = assemble(out_dir)
    cells = out_dir / "cells"
    lines = ["# Feature selection under repeated splits - comparison report", "",
             f"R = {repeats} repeated stratified train/val/test (70/15/15) resamples "
             "per source; uniform G1 config at 100 trees; per-cell RF seed shared "
             "across arms. Primary estimand: paired test-AUROC delta vs A0 over the "
             "same resamples (95% t-CI). No per-fit bootstrap; no allocation "
             "matching; frozen-split artifacts untouched.", "",
             "Arms: A0 baseline / A1 hygiene (drop `__missing` indicators + caret "
             "nzv) / A2 = A1 + |rho|>0.95 dedup / A3_k{50,100} top-k by permutation "
             "importance on val (no-op when A1 <= k).", ""]
    for src in sources:
        lines += [f"## Source {src}", "",
                  "### Absolute test AUROC (mean +/- SD over repeats)", "",
                  "| variant | " + " | ".join(ARMS) + " |",
                  "|---|" + "---:|" * len(ARMS)]
        for v in VARIANTS:
            cells_v = [df[(df.source == src) & (df.variant == v) & (df.arm == a)]
                       ["auroc_test"] for a in ARMS]
            assert all(len(c) == repeats for c in cells_v), (src, v)
            lines.append(f"| {v} | " + " | ".join(
                f"{c.mean():.4f} +/- {c.std(ddof=1):.4f}" for c in cells_v) + " |")
        lines += ["", "### Paired test-AUROC delta vs A0 (mean [95% t-CI])", "",
                  "| variant | A1-A0 | A2-A0 | A3_k50-A0 | A3_k100-A0 | A2-A1 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for v in VARIANTS:
            sub = df[(df.source == src) & (df.variant == v)]
            base = sub[sub.arm == "A0"].sort_values("repeat")["auroc_test"].to_numpy()
            cells_txt = []
            for arm, ref, ref_col in (("A1", "A0", None), ("A2", "A0", None),
                                      ("A3_k50", "A0", None), ("A3_k100", "A0", None),
                                      ("A2", "A1", None)):
                cur = sub[sub.arm == arm].sort_values("repeat")["auroc_test"].to_numpy()
                refv = sub[sub.arm == ref].sort_values("repeat")["auroc_test"].to_numpy()
                m, _, ci = _mean_ci(cur - refv)
                noop = sub[sub.arm == arm]["noop"].any()
                cells_txt.append(f"{m:+.4f} [{m - ci:+.4f}, {m + ci:+.4f}]"
                                 + (" (no-op)" if noop else ""))
            lines.append(f"| {v} | " + " | ".join(cells_txt) + " |")
        lines.append("")

    lines += ["## Selection footprint and stability", "",
              "Mean retained columns (A2) and mean pairwise Jaccard overlap of "
              "A3 top-k sets across repeats:", "",
              "| source | variant | A0 cols | A1 cols | A2 cols | top-50 Jaccard | "
              "top-100 Jaccard |", "|---|---:|---:|---:|---:|---:|---:|"]
    for src in sources:
        for v in VARIANTS:
            sub = df[(df.source == src) & (df.variant == v)]
            cols = {a: sub[sub.arm == a]["n_features"].mean() for a in ARMS}
            js = []
            for k in A3_KS:
                sets = []
                for f in sorted(cells.glob(f"{src}_*.json")):
                    d = json.loads(f.read_text())["variants"][v]
                    if d["noop"][f"k{k}"]:
                        continue
                    sets.append(set(d["topk"][f"k{k}"]))
                if len(sets) < 2:
                    js.append(None)
                else:
                    vals = [len(a & b) / len(a | b)
                            for a, b in itertools.combinations(sets, 2)]
                    js.append(float(np.mean(vals)))
            fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
            lines.append(f"| {src} | {v} | {cols['A0']:.0f} | {cols['A1']:.0f} | "
                         f"{cols['A2']:.0f} | {fmt(js[0])} | {fmt(js[1])} |")
    lines += ["", "Notes: demo carries no indicators and <= 50 columns, so A3 is a "
              "recorded no-op there (deltas equal A1 by construction); the same "
              "holds for any variant/cell where A1 <= k. Samsung val n is ~81, so "
              "its permutation-importance ranking (and top-k membership) is noisy - "
              "read its A3 deltas with the same caution as its absolute levels.", ""]
    (out_dir / "feat_sel_report.md").write_text("\n".join(lines) + "\n")


def write_repro(out_dir: Path, repeats: int, sources: tuple[int, ...], jobs: int) -> None:
    versions = {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                "pandas": pd.__version__, "numpy": np.__version__,
                "scipy": scipy.__version__, "joblib": joblib.__version__}
    hashes = {}
    for name in ("feat_sel.py", "preprocess.py", "model.py", "sq_config.py"):
        p = Path(__file__).parent / name
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    payload = {"fs_seed": FS_SEED, "n_trees": N_TREES, "repeats": repeats,
               "sources": list(sources), "jobs": jobs,
               "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
               "g1_config": G1, "pi_repeats": PI_REPEATS, "a3_ks": list(A3_KS),
               "nzv": {"dominant": NZV_DOMINANT, "unique": NZV_UNIQUE},
               "corr_threshold": CORR_THRESH, "arms": list(ARMS),
               "versions": versions, "script_sha256": hashes}
    (out_dir / "fs_repro.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="fsplit")
    ap.add_argument("--sources", default=",".join(str(s) for s in MODEL_SOURCE_IDS))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    sources = tuple(int(s) for s in args.sources.split(","))
    out_dir = ARTIFACTS_SQ / args.out

    if args.report_only:
        build_report(out_dir, args.repeats, sources)
        print(f"report written: {out_dir / 'feat_sel_report.md'}")
        return

    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    write_repro(out_dir, args.repeats, sources, args.jobs)

    pending = [(s, r) for s in sources for r in range(args.repeats)
               if not (out_dir / "cells" / f"{s}_{r}.json").exists()]
    total = len(sources) * args.repeats
    print(f"FS phase: {total} cells, {total - len(pending)} cached, "
          f"{len(pending)} pending; jobs={args.jobs}", flush=True)
    if pending:
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
            delayed(_self_contained)(s, r, str(out_dir)) for s, r in pending)
    build_report(out_dir, args.repeats, sources)
    print(f"metrics: {out_dir / 'fsplit_metrics.csv'}")
    print(f"report:  {out_dir / 'feat_sel_report.md'}")


if __name__ == "__main__":
    main()
