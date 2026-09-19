"""Side-quest section 4: per-source participant train/val/test allocation.

Adapted from ``src/allocate.py`` (which hard-codes v2 paths and v2 profile
fields and cannot be reused unchanged).  For each model-eligible source:

  * hard targets: split totals ``T = largest_remainder(FRACS, N)`` and class-1
    (sal20) counts apportioned by the same rule (ties -> lowest split index);
    sal10 = ``T - sal20``.  This reproduces the frozen Garmin row and the
    recorded Apple row exactly and is asserted against ``FROZEN_SPLITS``;
  * participants aggregated into allocation profiles homogeneous in
    (y, age_group, bmi_grp, selected_channel, pass_pattern);
  * stage 1: MILP minimizing the total integer violation outside the requested
    floor/ceiling proportional bounds of the age-group, BMI-group, and
    qualifying-channel pass-pattern marginals, with totals and outcome counts
    hard;
  * stage 2: fix the stage-1 minimum total violation and minimize the secondary
    absolute-imbalance objective over occupied ``y x age_group x bmi_grp``
    cells and selected-channel marginals (equal weights: the plan names both
    families without a priority);
  * seeded (SEED) random assignment within each profile.

Outputs:
  artifacts_sq/split_manifest_sq.csv / .parquet
  artifacts_sq/split_violations_sq.csv
  artifacts_sq/split_allocation_report_sq.md / .json

Run:  python src/sidequest/allocate.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sq_config import (  # noqa: E402
    ARTIFACTS_SQ, FROZEN_SPLITS, MODEL_SOURCE_IDS, SEED, TEST_FRAC, TRAIN_FRAC,
    VAL_FRAC,
)

SPLITS = ["train", "val", "test"]
FRACS = np.array([TRAIN_FRAC, VAL_FRAC, TEST_FRAC])
BOUND_FAMILIES = ["age_group", "bmi_grp", "pass_pattern"]
SECONDARY_FAMILIES = ["_yab", "selected_channel"]


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """v2 apportionment: largest remainder, ties -> lowest split index."""
    exact = np.asarray(weights, dtype=float) / np.sum(weights) * total
    base = np.floor(exact).astype(int)
    rem = int(total - base.sum())
    order = np.lexsort((np.arange(len(exact)), -(exact - base)))
    for i in order[:rem]:
        base[i] += 1
    return base


def split_targets(N: int, n1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(totals, sal10, sal20) per split; class-1 (sal20) apportioned directly."""
    T = largest_remainder(FRACS, N)
    C1 = largest_remainder(FRACS * (n1 / N), n1) if n1 else np.zeros(3, dtype=int)
    return T, T - C1, C1


def _level_masks(pdf: pd.DataFrame, col: str) -> list[tuple[str, np.ndarray]]:
    """(level label, boolean mask over profile rows) for one categorical."""
    out = []
    for k, sub in pdf.groupby(col, dropna=False, sort=True):
        mask = np.zeros(len(pdf), dtype=bool)
        mask[sub.index.to_numpy()] = True
        label = str(int(k)) if pd.notna(k) and isinstance(k, (int, float, np.integer,
                                                              np.floating)) else str(k)
        out.append((label, mask))
    return out


def build_solve(pdf: pd.DataFrame, n_p: np.ndarray, y_p: np.ndarray,
                T: np.ndarray, C1: np.ndarray):
    """Two-stage MILP over profile counts x[p, s].

    Variable layout: x (P*3, integer), stage-1 violation v (one per requested
    bound and split, integer), stage-2 absolute-deviation a (one per secondary
    term and split, continuous).
    """
    P, S = len(pdf), 3
    N = int(n_p.sum())

    bound_terms: list[tuple[str, str, np.ndarray]] = []
    for fam in BOUND_FAMILIES:
        for label, mask in _level_masks(pdf, fam):
            bound_terms.append((fam, label, mask))
    yab = (pdf["y"].astype(int).astype(str) + "|" + pdf["age_group"].astype(str) +
           "|" + pdf["bmi_grp"].astype(str))
    sec_terms: list[tuple[str, str, np.ndarray]] = [
        ("_yab", label, mask) for label, mask in _level_masks(pdf.assign(_yab=yab), "_yab")
    ] + [("selected_channel", label, mask)
         for label, mask in _level_masks(pdf, "selected_channel")]

    nx = P * S
    nv = len(bound_terms) * S
    na = len(sec_terms) * S

    def xidx(p, s):
        return p * S + s

    def vidx(b, s):
        return nx + b * S + s

    def aidx(g, s):
        return nx + nv + g * S + s

    def build(stage: int):
        rows, cols, vals, lb, ub = [], [], [], [], []

        def add(r, c, v, lo, hi):
            rows.extend(r); cols.extend(c); vals.extend(v)
            lb.append(lo); ub.append(hi)

        # hard: per-profile sums, per-split totals, per-split sal20 counts
        for p in range(P):
            add([len(lb)] * S, [xidx(p, s) for s in range(S)], [1.0] * S,
                float(n_p[p]), float(n_p[p]))
        for s in range(S):
            add([len(lb)] * P, [xidx(p, s) for p in range(P)], [1.0] * P,
                float(T[s]), float(T[s]))
            ones = [p for p in range(P) if y_p[p] == 1]
            add([len(lb)] * len(ones), [xidx(p, s) for p in ones], [1.0] * len(ones),
                float(C1[s]), float(C1[s]))

        n_var = nx + (nv if stage == 1 else nv + na)
        obj = np.zeros(n_var)
        integrality = np.zeros(n_var, dtype=int)
        integrality[:nx] = 1
        integrality[nx:nx + nv] = 1

        # floor/ceiling proportional bounds with violation variables
        for b, (_fam, _label, mask) in enumerate(bound_terms):
            ps = np.flatnonzero(mask)
            cnt = float(n_p[ps].sum())
            for s in range(S):
                q = T[s] * cnt / N
                lo_b, hi_b = float(np.floor(q)), float(np.ceil(q))
                add([len(lb)] * (len(ps) + 1),
                    [xidx(p, s) for p in ps] + [vidx(b, s)],
                    [1.0] * len(ps) + [-1.0], -np.inf, hi_b)
                add([len(lb)] * (len(ps) + 1),
                    [xidx(p, s) for p in ps] + [vidx(b, s)],
                    [1.0] * len(ps) + [1.0], lo_b, np.inf)

        if stage == 1:
            obj[nx:nx + nv] = 1.0
        else:
            # fix the stage-1 minimum total violation
            add([len(lb)] * nv, list(range(nx, nx + nv)), [1.0] * nv,
                float(v_star), float(v_star))
            # absolute deviations from proportional targets (secondary objective)
            for g, (_fam, _label, mask) in enumerate(sec_terms):
                ps = np.flatnonzero(mask)
                cnt = float(n_p[ps].sum())
                for s in range(S):
                    tau = T[s] * cnt / N
                    add([len(lb)] * (len(ps) + 1),
                        [xidx(p, s) for p in ps] + [aidx(g, s)],
                        [1.0] * len(ps) + [-1.0], -np.inf, tau)
                    add([len(lb)] * (len(ps) + 1),
                        [xidx(p, s) for p in ps] + [aidx(g, s)],
                        [1.0] * len(ps) + [1.0], tau, np.inf)
            for g in range(len(sec_terms)):
                for s in range(S):
                    obj[aidx(g, s)] = 1.0

        A = sparse.coo_matrix((vals, (rows, cols)), shape=(len(lb), n_var)).tocsr()
        var_lb = np.zeros(n_var)
        var_ub = np.concatenate([np.repeat(n_p, S).astype(float),
                                 np.full(max(nv, 0), np.inf),
                                 np.full(max(na if stage > 1 else 0, 0), np.inf)])[:n_var]
        res = milp(c=obj, constraints=LinearConstraint(A, np.array(lb), np.array(ub)),
                   integrality=integrality, bounds=Bounds(var_lb, var_ub))
        return res

    res1 = build(1)
    if res1.status != 0:
        raise RuntimeError(f"stage-1 MILP failed: status={res1.status} {res1.message}")
    v_star = float(round(res1.x[nx:nx + nv].sum()))
    res2 = build(2)
    if res2.status != 0:
        raise RuntimeError(f"stage-2 MILP failed: status={res2.status} {res2.message}")
    x = np.round(res2.x[:nx]).astype(int).reshape(P, S)

    # achieved counts and nonzero bound violations
    viol_rows = []
    for b, (fam, label, mask) in enumerate(bound_terms):
        ps = np.flatnonzero(mask)
        cnt = int(n_p[ps].sum())
        for s in range(S):
            q = T[s] * cnt / N
            lo_b, hi_b = int(np.floor(q)), int(np.ceil(q))
            achieved = int(x[ps, s].sum())
            v = int(max(0, achieved - hi_b, lo_b - achieved))
            if v > 0:
                viol_rows.append({"family": fam, "level": label, "split": s,
                                  "requested_low": lo_b, "requested_high": hi_b,
                                  "achieved": achieved, "violation": v})
    return x, v_star, viol_rows, len(bound_terms), len(sec_terms)


def main() -> None:
    model = pd.read_parquet(ARTIFACTS_SQ / "cohort_manifest_model_sources.parquet")
    rng = np.random.default_rng(SEED)
    parts, viol_parts, report = [], [], {}
    for src in MODEL_SOURCE_IDS:
        df = (model[model["source_id"] == src].reset_index(drop=True).copy())
        N, n1 = len(df), int((df["y"] == 1).sum())
        T, C0, C1 = split_targets(N, n1)
        frozen = FROZEN_SPLITS[src]
        for si, s in enumerate(SPLITS):
            got = (int(T[si]), int(C0[si]), int(C1[si]))
            if got != frozen[s]:
                sys.exit(f"STOP: target drift source {src} {s}: {got} != {frozen[s]}")

        key = (df["y"].astype(int).astype(str) + "|" + df["age_group"].astype(str) +
               "|" + df["bmi_grp"].astype(str) + "|" +
               df["selected_channel"].astype(str) + "|" + df["pass_pattern"].astype(str))
        codes, uniq = pd.factorize(key, sort=True)
        df["_pid"] = codes
        pdf = df.drop_duplicates("_pid").sort_values("_pid").reset_index(drop=True)
        P = len(uniq)
        n_p = df.groupby("_pid").size().reindex(range(P)).to_numpy()
        y_p = pdf["y"].astype(int).to_numpy()

        x, v_star, viol_rows, _n_bounds, _n_sec = build_solve(pdf, n_p, y_p, T, C1)
        for r in viol_rows:
            r["source_id"] = src
        viol_parts.extend(viol_rows)

        df["split"] = pd.NA
        for pid in range(P):
            idx = np.array(df.index[df["_pid"] == pid].to_numpy(), copy=True)
            rng.shuffle(idx)
            pos = 0
            for si, s in enumerate(SPLITS):
                k = int(x[pid, si])
                df.loc[idx[pos:pos + k], "split"] = s
                pos += k
            assert pos == len(idx), (src, pid)
        for si, s in enumerate(SPLITS):
            g = df[df["split"] == s]
            assert len(g) == T[si] and int((g["y"] == 1).sum()) == C1[si], (src, s)

        parts.append(df[["user_id", "source_id", "y", "salutation", "split"]])
        print(f"source {src}: N={N:,}, sal20={n1:,}, profiles={P:,}, "
              f"stage1 total violation={v_star:g}, nonzero violations={len(viol_rows)}")

        # ---- per-source section-6 artifacts ------------------------------------
        d = ARTIFACTS_SQ / f"source_{src}"
        d.mkdir(parents=True, exist_ok=True)
        df[["user_id", "y", "split"]].to_csv(d / "split_manifest.csv", index=False)
        with open(d / "split_allocation_report.json", "w") as fh:
            json.dump({"N": int(N), "n_sal20": n1, "n_profiles": int(P),
                       "stage1_min_total_violation": v_star,
                       "targets": {s: {"total": int(T[i]), "sal10": int(C0[i]),
                                       "sal20": int(C1[i])}
                                   for i, s in enumerate(SPLITS)},
                       "violations": viol_rows}, fh, indent=2)
        lines = [f"# Source {src} split balance", "",
                 f"N = {N:,} ({n1:,} sal20); {P:,} allocation profiles; "
                 f"stage-1 minimum total bound violation = {v_star:g}.", "",
                 "| split | total | target | sal10 | sal20 |", "|---|---:|---:|---:|---:|"]
        for si, s in enumerate(SPLITS):
            g = df[df["split"] == s]
            lines.append(f"| {s} | {len(g)} | {T[si]} | "
                         f"{int((g['y'] == 0).sum())} | {int((g['y'] == 1).sum())} |")
        lines += ["", "## Marginal proportions by split (pp spread)", "",
                  "| variable | level | train % | val % | test % | max-min pp |",
                  "|---|---|---:|---:|---:|---:|"]
        for fam in BOUND_FAMILIES + ["selected_channel"]:
            tab = pd.crosstab(df[fam].astype(str), df["split"], normalize="columns")
            tab = tab.reindex(columns=SPLITS).fillna(0.0)
            for k, row in tab.iterrows():
                p = row.to_numpy() * 100
                lines.append(f"| {fam} | {k} | {p[0]:.2f} | {p[1]:.2f} | {p[2]:.2f} | "
                             f"{p.max() - p.min():.2f} |")
        yab = (df["y"].astype(int).astype(str) + "|" + df["age_group"].astype(str) +
               "|" + df["bmi_grp"].astype(str))
        tab = pd.crosstab(yab, df["split"], normalize="columns")
        tab = tab.reindex(columns=SPLITS).fillna(0.0)
        worst = (tab.max(axis=1) - tab.min(axis=1)).sort_values(ascending=False)
        lines += ["", f"Occupied y x age x bmi cells: {len(tab)}; "
                  f"max within-cell spread {worst.iloc[0] * 100:.2f} pp "
                  f"(cell {worst.index[0]}).", ""]
        (d / "split_balance_report.md").write_text("\n".join(lines) + "\n")

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(ARTIFACTS_SQ / "split_manifest_sq.csv", index=False)
    out.to_parquet(ARTIFACTS_SQ / "split_manifest_sq.parquet", index=False)
    vdf = pd.DataFrame(viol_parts,
                       columns=["source_id", "family", "level", "split",
                                "requested_low", "requested_high", "achieved",
                                "violation"])
    vdf.to_csv(ARTIFACTS_SQ / "split_violations_sq.csv", index=False)

    report = {"seed": int(SEED), "sources": {}, "violations": viol_parts}
    lines = ["# Side-quest split allocation report", ""]
    for src in MODEL_SOURCE_IDS:
        sub = out[out["source_id"] == src]
        frozen = FROZEN_SPLITS[src]
        lines += [f"## Source {src}", "",
                  "| split | achieved total | target | sal10 | sal20 |", "|---|---:|---:|---:|---:|"]
        for s in SPLITS:
            g = sub[sub["split"] == s]
            lines.append(f"| {s} | {len(g)} | {frozen[s][0]} | "
                         f"{int((g['y'] == 0).sum())} | {int((g['y'] == 1).sum())} |")
        lines.append("")
    n_v = len(viol_parts)
    lines += ["## Bound violations", ""]
    if n_v:
        lines += ["| source | family | level | split | requested | achieved | violation |",
                  "|---|---|---|---|---|---:|---:|"]
        for r in viol_parts:
            lines.append(f"| {r['source_id']} | {r['family']} | {r['level']} | "
                         f"{r['split']} | [{r['requested_low']}, {r['requested_high']}] | "
                         f"{r['achieved']} | {r['violation']} |")
    else:
        lines.append("No floor/ceiling bound violations were necessary.")
    with open(ARTIFACTS_SQ / "split_allocation_report_sq.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(ARTIFACTS_SQ / "split_allocation_report_sq.json", "w") as fh:
        json.dump({"sources": {str(s): {"targets": FROZEN_SPLITS[s],
                                        "n_violations": int((vdf["source_id"] == s).sum())}
                               for s in MODEL_SOURCE_IDS},
                   "stage_totals": vdf.groupby("source_id")["violation"].sum().to_dict()},
                  fh, indent=2)
    print(f"split manifest: {len(out):,} participants; violations: {n_v}")


if __name__ == "__main__":
    main()
