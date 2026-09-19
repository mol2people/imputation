"""Side-quest FE phase: feature engineering under repeated stratified splits.

Phase spec (pre-registered): ``artifacts_sq/plans/sidequest_feature_engineering_repeated_splits_2026-09-19.md``.

Reuses the FS phase's split/seed machinery verbatim (``feat_sel.py``): the same
30 stratified 70/15/15 repeats per source, same per-cell RF seeds, same hygiene
rule, same G1 config at 100 trees. Arm ``B0`` (hygiene on the original matrix)
must therefore reproduce FS arm ``A1`` exactly — asserted against
``artifacts_sq/fsplit/fsplit_metrics.csv`` to < 1e-9 for every cell.

Engineered families (frozen before any FE evaluation; formulas in the plan):
  E1 wear-pattern interactions  `weekday_mean x tod_b`           (rec/win x 3 ch x 4 tod = 24)
  E2 channel ratios             `chA_{stat} / chB_{stat}`, den>0  (rec/win x 3 pairs x 2 stats = 12)
  E3 drift deltas               `w30_mean - w7_mean`             (2 roll blocks x 3 ch x 3 series = 36)

Arms per (source, repeat, variant): B0 baseline / F1 (+all applicable families)
/ F2 (= F1 + |rho|>0.95 dedup with ORIGINALS-FIRST keep order, so engineered
terms can be evicted by correlated originals but never evict their parents) /
F1_E{1,2,3} single-family arms (applicable families only; no-ops recorded for
`demo`, which has no applicable family).

Primary family (winner-capable): source 3 `all` and source 6 `all`, `F1 - B0`
and `F2 - B0`, two-sided 97.5% t intervals (Bonferroni-simultaneous 95%).
Everything else is secondary/exploratory. The whole phase is adaptive relative
to the cohort's analysis history and pre-registered relative to the FS phase's
test summaries; it is never independent validation.

Outputs under ``artifacts_sq/<out>`` (default ``feng``): cells/, metrics CSV,
repro JSON, comparison report.

Run:  python src/sidequest/feat_eng.py [--repeats 30] [--jobs N]
                                      [--out feng] [--sources 3,6,7]
                                      [--report-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feat_sel import (  # noqa: E402
    CORR_THRESH, FS_SEED, G1, N_TREES, _mean_ci, fit_eval, get_blocks,
    nzv_keep, rf_seed_for, stratified_split,
)
from model import variant_matrix  # noqa: E402
from preprocess import SQPreprocessor  # noqa: E402
from sq_config import ARTIFACTS_SQ, MODEL_SOURCE_IDS, VARIANTS  # noqa: E402

CHANNELS = (3000, 3001, 3002)
TOD = ("night", "morning", "afternoon", "evening")
FAMILIES = ("E1", "E2", "E3")
ARMS = ("B0", "F1", "F2", "F1_E1", "F1_E2", "F1_E3")


# ------------------------------------------------------- frozen FE formulas --

def family_specs() -> dict[str, list[tuple[str, tuple[str, str], object]]]:
    """(name, parent columns, vectorised formula) per engineered column."""
    fams: dict[str, list] = {"E1": [], "E2": [], "E3": []}
    for p in ("rec", "win"):
        for c in CHANNELS:
            for b in TOD:
                na, nb = f"{p}__ch{c}_weekday_mean", f"{p}__ch{c}_tod_{b}"
                fams["E1"].append((f"fe1__{p}_ch{c}_weekday_x_tod_{b}", (na, nb),
                                   lambda df, a=na, z=nb: df[a] * df[z]))
        for a, b in ((3000, 3002), (3000, 3001), (3001, 3002)):
            for stat, suffix in (("daily_mean", "mean_of_daily_mean"),
                                 ("weekday", "weekday_mean")):
                pn, pdn = f"{p}__ch{a}_{suffix}", f"{p}__ch{b}_{suffix}"
                fams["E2"].append((f"fe2__{p}_ch{a}_div_ch{b}_{stat}", (pn, pdn),
                                   lambda df, n=pn, d=pdn: df[n] / df[d].where(df[d] > 0)))
    for blk in ("roll_rec", "roll_win"):
        for c in CHANNELS:
            for ser in ("daily_mean", "daily_hours", "daily_sd"):
                p30, p7 = f"{blk}__ch{c}_{ser}_w30_mean", f"{blk}__ch{c}_{ser}_w7_mean"
                fams["E3"].append((f"fe3__{blk}_ch{c}_{ser}_w30_minus_w7", (p30, p7),
                                   lambda df, a=p30, z=p7: df[a] - df[z]))
    return fams


FAM_SPECS = family_specs()


def applicable_specs(variant: str, feat_cols: list[str]
                     ) -> dict[str, list[tuple[str, tuple[str, str], object]]]:
    out: dict[str, list] = {}
    for fam in FAMILIES:
        specs = [s for s in FAM_SPECS[fam] if all(p in feat_cols for p in s[1])]
        if specs:
            out[fam] = specs
    return out


def add_engineered(X: pd.DataFrame,
                   specs: list[tuple[str, tuple[str, str], object]]
                   ) -> tuple[pd.DataFrame, list[str]]:
    cols, names = {}, []
    for name, parents, fn in specs:
        assert all(p in X.columns for p in parents), f"parent missing for {name}"
        cols[name] = fn(X).astype(float)
        names.append(name)
    return pd.concat([X, pd.DataFrame(cols, index=X.index)], axis=1), names


def hygiene_mask(names: list[str], Z_tr: np.ndarray) -> np.ndarray:
    """FS A1 rule: drop all __missing indicators, then nzv on the remainder."""
    ind = np.array([n.endswith("__missing") for n in names])
    base = ~ind
    sub = nzv_keep(Z_tr[:, base])
    keep = base.copy()
    keep[np.flatnonzero(base)[~sub]] = False
    return keep, int(ind.sum()), int((~sub).sum())


def dedup_originals_first(Z: np.ndarray, names: list[str]) -> np.ndarray:
    """Greedy |rho|>threshold pruning; original columns keep-first, engineered
    columns are candidates only (an engineered term can be evicted by a
    correlated original but never evicts its parent)."""
    eng = np.array([n.startswith("fe") for n in names])
    order = np.concatenate([np.flatnonzero(~eng), np.flatnonzero(eng)])
    C = np.nan_to_num(np.corrcoef(Z[:, order], rowvar=False), nan=0.0)
    kept: list[int] = []
    for j in range(len(order)):
        if all(abs(C[j, i]) <= CORR_THRESH for i in kept):
            kept.append(j)
    keep = np.zeros(len(names), dtype=bool)
    keep[order[kept]] = True
    return keep


# ------------------------------------------------------------------ cell work --

def run_cell(src: int, r: int, out_dir: Path, fs_a1: pd.DataFrame) -> list[dict]:
    t_cell = time.time()
    split = pd.read_parquet(ARTIFACTS_SQ / "split_manifest_sq.parquet")
    y_map = split[split["source_id"] == src].set_index("user_id")["y"].astype(int)
    users = np.sort(y_map.index.to_numpy())
    y_all = y_map.loc[users].to_numpy(dtype=int)

    rng = np.random.default_rng(np.random.SeedSequence([FS_SEED, src, r]))
    tr, va, te = stratified_split(y_all, rng)
    assert len(tr) + len(va) + len(te) == len(users)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))

    blocks = get_blocks()
    rows: list[dict] = []
    variants_out: dict[str, dict] = {}

    for variant in VARIANTS:
        fr = variant_matrix(blocks, variant)
        fr = fr[fr["source_id"] == src].sort_values("user_id").reset_index(drop=True)
        assert list(fr["user_id"]) == list(users)
        feat_cols = [c for c in fr.columns if c not in ("user_id", "source_id", "y")]
        fr["y"] = fr["user_id"].map(y_map)
        assert fr["y"].notna().all()
        X = fr[feat_cols]
        y = fr["y"].to_numpy(dtype=int)
        seed = rf_seed_for(src, r, variant)

        # ---- B0: hygiene on the original matrix; must equal FS A1 --------------
        prep0 = SQPreprocessor().fit(X.iloc[tr])
        Z0tr, Z0va, Z0te = (prep0.transform(X.iloc[i]) for i in (tr, va, te))
        keep0, ind0, nzv0 = hygiene_mask(list(prep0.feature_names_out_), Z0tr)
        _, v0, t0, s0 = fit_eval(seed, Z0tr, y[tr], Z0va, y[va], Z0te, y[te], keep0)
        fs_row = fs_a1[(fs_a1.source == src) & (fs_a1.variant == variant)
                       & (fs_a1.repeat == r)]
        assert len(fs_row) == 1
        b0_gap = max(abs(v0 - float(fs_row["auroc_val"].iloc[0])),
                     abs(t0 - float(fs_row["auroc_test"].iloc[0])))
        assert b0_gap < 1e-9, f"B0 != FS A1 for src={src} r={r} {variant}: {b0_gap}"
        rows.append({"source": src, "variant": variant, "arm": "B0", "repeat": r,
                     "auroc_val": v0, "auroc_test": t0,
                     "n_features": int(keep0.sum()), "n_eng_input": 0,
                     "n_eng_after_hygiene": 0, "n_eng_after_dedup": 0,
                     "noop": False, "seconds": s0})
        b0_val, b0_test = v0, t0

        fams = applicable_specs(variant, feat_cols)
        variants_out[variant] = {"b0_gap_vs_fs_a1": b0_gap, "families": {}}
        if not fams:  # demo: no applicable family -> F-arms are recorded no-ops
            for arm in ("F1", "F2", "F1_E1", "F1_E2", "F1_E3"):
                rows.append({"source": src, "variant": variant, "arm": arm,
                             "repeat": r, "auroc_val": b0_val, "auroc_test": b0_test,
                             "n_features": int(keep0.sum()), "n_eng_input": 0,
                             "n_eng_after_hygiene": 0, "n_eng_after_dedup": 0,
                             "noop": True, "seconds": 0.0})
            continue

        # ---- one preprocessor/transform serves every F-arm ---------------------
        all_specs = [s for fam in FAMILIES for s in fams.get(fam, [])]
        Xf, eng_names = add_engineered(X, all_specs)
        eng_set = set(eng_names)
        prep1 = SQPreprocessor().fit(Xf.iloc[tr])
        Z1tr, Z1va, Z1te = (prep1.transform(Xf.iloc[i]) for i in (tr, va, te))
        names1 = list(prep1.feature_names_out_)
        keep1, _, _ = hygiene_mask(names1, Z1tr)

        def eng_stats(mask: np.ndarray) -> tuple[int, dict[str, int], list[str]]:
            n_eng = int(sum(1 for n, k in zip(names1, mask) if k and n in eng_set))
            per_fam = {fam: 0 for fam in fams}
            fam_of = {name: fam for fam in FAMILIES for name, _, _ in fams.get(fam, [])}
            for n, k in zip(names1, mask):
                if k and n in eng_set:
                    per_fam[fam_of[n]] += 1
            return n_eng, per_fam, [n for n, k in zip(names1, mask) if k and n in eng_set]

        # ---- F1: + all applicable families -------------------------------------
        n1, per_fam1, surv1 = eng_stats(keep1)
        _, v1, t1, s1 = fit_eval(seed, Z1tr, y[tr], Z1va, y[va], Z1te, y[te], keep1)
        rows.append({"source": src, "variant": variant, "arm": "F1", "repeat": r,
                     "auroc_val": v1, "auroc_test": t1, "n_features": int(keep1.sum()),
                     "n_eng_input": len(all_specs), "n_eng_after_hygiene": n1,
                     "n_eng_after_dedup": n1, "noop": False, "seconds": s1})

        # ---- F2: F1 + dedup, originals-first -----------------------------------
        sub_names = [n for n, k in zip(names1, keep1) if k]
        keep2_sub = dedup_originals_first(Z1tr[:, keep1], sub_names)
        keep2 = np.zeros(len(names1), dtype=bool)
        keep2[np.flatnonzero(keep1)[keep2_sub]] = True
        n2, per_fam2, surv2 = eng_stats(keep2)
        _, v2, t2, s2 = fit_eval(seed, Z1tr, y[tr], Z1va, y[va], Z1te, y[te], keep2)
        rows.append({"source": src, "variant": variant, "arm": "F2", "repeat": r,
                     "auroc_val": v2, "auroc_test": t2, "n_features": int(keep2.sum()),
                     "n_eng_input": len(all_specs), "n_eng_after_hygiene": n1,
                     "n_eng_after_dedup": n2, "noop": False, "seconds": s2})

        # ---- single-family arms ------------------------------------------------
        for fam in FAMILIES:
            arm = f"F1_{fam}"
            if fam not in fams:
                continue
            specs = fams[fam]
            Xg, eng_g = add_engineered(X, specs)
            gset = set(eng_g)
            prepg = SQPreprocessor().fit(Xg.iloc[tr])
            Zgtr, Zgva, Zgte = (prepg.transform(Xg.iloc[i]) for i in (tr, va, te))
            keepg, _, _ = hygiene_mask(list(prepg.feature_names_out_), Zgtr)
            ng = int(sum(1 for n, k in zip(prepg.feature_names_out_, keepg)
                         if k and n in gset))
            _, vg, tg, sg = fit_eval(seed, Zgtr, y[tr], Zgva, y[va], Zgte, y[te], keepg)
            rows.append({"source": src, "variant": variant, "arm": arm, "repeat": r,
                         "auroc_val": vg, "auroc_test": tg,
                         "n_features": int(keepg.sum()), "n_eng_input": len(specs),
                         "n_eng_after_hygiene": ng, "n_eng_after_dedup": ng,
                         "noop": False, "seconds": sg})
            variants_out[variant]["families"][fam] = {
                "input": len(specs), "after_hygiene": ng, "survivors": eng_g}

        variants_out[variant]["families"]["ALL"] = {
            "input": len(all_specs), "after_hygiene": n1, "after_dedup": n2,
            "survivors_f1": surv1, "survivors_f2": surv2,
            "per_family_hygiene": per_fam1, "per_family_dedup": per_fam2}
        variants_out[variant]["dedup_dropped"] = [n for n, k in zip(sub_names, keep2_sub)
                                                  if not k]

    payload = {"source": src, "repeat": r, "rows": rows, "variants": variants_out,
               "seconds": round(time.time() - t_cell, 1)}
    (out_dir / "cells" / f"{src}_{r}.json").write_text(json.dumps(payload))
    return rows


def _self_contained(src: int, r: int, out_dir: str, fsplit_csv: str) -> str:
    fs_a1 = pd.read_csv(fsplit_csv)
    fs_a1 = fs_a1[fs_a1.arm == "A1"]
    rows = run_cell(src, r, Path(out_dir), fs_a1)
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
    df.to_csv(out_dir / "feng_metrics.csv", index=False)
    return df


def build_report(out_dir: Path, repeats: int, sources: tuple[int, ...]) -> None:
    df = assemble(out_dir)
    t975 = __import__("scipy").stats.t.ppf(0.975, repeats - 1)
    lines = ["# Feature engineering under repeated splits - comparison report", "",
             f"Same {repeats} stratified 70/15/15 repeats and RF seeds as the FS "
             "phase; B0 = hygiene on the original matrix, verified identical to FS "
             "arm A1 (< 1e-9) in every cell. Engineered families E1 (wear-pattern "
             "interactions), E2 (channel ratios), E3 (w30-w7 drift deltas) are "
             "added as raw columns before preprocessing; hygiene and G1 @ 100 "
             "trees unchanged. F2 dedup keeps originals first.", "",
             "**Adaptive/exploratory per the phase guard**: pre-registered "
             "relative to the FS phase's test summaries, adaptive relative to the "
             "cohort's analysis history; same-partition reuse is never "
             "independent validation.", ""]
    lines += ["## Primary family (winner-capable): source 3 / 6, `all`, 97.5% t",
              "", "| comparison | mean | SD | MC SE | median | IQR | share>0 | "
              "97.5% t CI |", "|---|---:|---:|---:|---:|---|---:|---|"]
    for src in sources:
        sub = df[(df.source == src) & (df.variant == "all")]
        base = sub[sub.arm == "B0"].sort_values("repeat")["auroc_test"].to_numpy()
        for arm in ("F1", "F2"):
            cur = sub[sub.arm == arm].sort_values("repeat")["auroc_test"].to_numpy()
            d = cur - base
            se = d.std(ddof=1) / np.sqrt(repeats)
            lines.append(
                f"| s{src} `all` {arm}−B0 | {d.mean():+.4f} | {d.std(ddof=1):.4f} | "
                f"{se:.4f} | {np.median(d):+.4f} | "
                f"[{np.percentile(d,25):+.4f},{np.percentile(d,75):+.4f}] | "
                f"{(d>0).mean():.2f} | "
                f"[{d.mean()-t975*se:+.4f}, {d.mean()+t975*se:+.4f}] |")
    lines.append("")
    for src in sources:
        lines += [f"## Source {src}", "",
                  "### Paired test-AUROC delta vs B0 (mean [95% t-CI])", "",
                  "| variant | F1-B0 | F2-B0 | F1_E1-B0 | F1_E2-B0 | F1_E3-B0 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for v in VARIANTS:
            sub = df[(df.source == src) & (df.variant == v)]
            base = sub[sub.arm == "B0"].sort_values("repeat")["auroc_test"].to_numpy()
            txt = []
            for arm in ("F1", "F2", "F1_E1", "F1_E2", "F1_E3"):
                cur = sub[sub.arm == arm]
                if not len(cur):
                    txt.append("n/a")
                    continue
                d = cur.sort_values("repeat")["auroc_test"].to_numpy() - base
                m, _, ci = _mean_ci(d)
                txt.append(f"{m:+.4f} [{m-ci:+.4f}, {m+ci:+.4f}]"
                           + (" (no-op)" if bool(cur["noop"].iloc[0]) else ""))
            lines.append(f"| {v} | " + " | ".join(txt) + " |")
        lines.append("")
    lines += ["## Engineered-feature survival (mean over repeats)", "",
              "| source | variant | input | after hygiene | after dedup (F2) |",
              "|---|---:|---:|---:|---:|"]
    cells = out_dir / "cells"
    for src in sources:
        for v in VARIANTS:
            ins, hyg, ded = [], [], []
            for f in sorted(cells.glob(f"{src}_*.json")):
                d = json.loads(f.read_text())["variants"][v].get("families", {}).get("ALL")
                if d:
                    ins.append(d["input"]); hyg.append(d["after_hygiene"])
                    ded.append(d["after_dedup"])
            if ins:
                lines.append(f"| {src} | {v} | {np.mean(ins):.1f} | "
                             f"{np.mean(hyg):.1f} | {np.mean(ded):.1f} |")
    max_gap = max(json.loads(f.read_text())["variants"][v]["b0_gap_vs_fs_a1"]
                  for f in sorted(cells.glob("*.json"))
                  for v in json.loads(f.read_text())["variants"])
    lines += ["", f"B0 vs FS A1 determinism check: max |AUROC gap| over all cells "
              f"= {max_gap:.2e} (contract: < 1e-9).", ""]
    (out_dir / "feat_eng_report.md").write_text("\n".join(lines) + "\n")


def write_repro(out_dir: Path, repeats: int, sources: tuple[int, ...], jobs: int) -> None:
    import scipy
    versions = {"python": sys.version.split()[0], "sklearn": __import__("sklearn").__version__,
                "pandas": pd.__version__, "numpy": np.__version__,
                "scipy": scipy.__version__, "joblib": __import__("joblib").__version__}
    hashes = {}
    for name in ("feat_eng.py", "feat_sel.py", "preprocess.py", "model.py", "sq_config.py"):
        p = Path(__file__).parent / name
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    formulas = {fam: [{"name": n, "parents": list(p)} for n, p, _ in specs]
                for fam, specs in FAM_SPECS.items()}
    payload = {"seed": FS_SEED, "n_trees": N_TREES, "repeats": repeats,
               "sources": list(sources), "jobs": jobs, "g1_config": G1,
               "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
               "families": formulas, "arms": list(ARMS),
               "dedup": {"threshold": CORR_THRESH, "keep_order": "originals_first"},
               "b0_contract": "identical to FS A1 (< 1e-9)",
               "versions": versions, "script_sha256": hashes}
    (out_dir / "feng_repro.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="feng")
    ap.add_argument("--sources", default=",".join(str(s) for s in MODEL_SOURCE_IDS))
    ap.add_argument("--fsplit-csv", default=str(ARTIFACTS_SQ / "fsplit" / "fsplit_metrics.csv"))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    sources = tuple(int(s) for s in args.sources.split(","))
    out_dir = ARTIFACTS_SQ / args.out

    if args.report_only:
        build_report(out_dir, args.repeats, sources)
        print(f"report written: {out_dir / 'feat_eng_report.md'}")
        return

    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    write_repro(out_dir, args.repeats, sources, args.jobs)

    pending = [(s, r) for s in sources for r in range(args.repeats)
               if not (out_dir / "cells" / f"{s}_{r}.json").exists()]
    total = len(sources) * args.repeats
    print(f"FE phase: {total} cells, {total - len(pending)} cached, "
          f"{len(pending)} pending; jobs={args.jobs}", flush=True)
    if pending:
        from joblib import Parallel, delayed
        Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
            delayed(_self_contained)(s, r, str(out_dir), args.fsplit_csv)
            for s, r in pending)
    build_report(out_dir, args.repeats, sources)
    print(f"metrics: {out_dir / 'feng_metrics.csv'}")
    print(f"report:  {out_dir / 'feat_eng_report.md'}")


if __name__ == "__main__":
    main()
