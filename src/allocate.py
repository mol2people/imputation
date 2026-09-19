"""Seeded constrained 80/10/10 allocation.

Aggregates participants into identical allocation profiles, then solves an
integer program (scipy/HiGHS) that:
  * fixes split totals by largest-remainder rounding of 80/10/10 (ties -> val);
  * fixes per-split salutation counts from a joint largest-remainder allocation;
  * (first attempt) constrains every one-variable category margin within
    floor/ceiling of its proportional target;
  * minimises weighted absolute deviations of salutation x covariate margins,
    joint strata (>=10) and pooled rarity groups.

Random assignment within allocated profiles uses SEED.

Outputs:
  artifacts/split_manifest.csv
  artifacts/split_allocation_report.md / .json
Run:  python src/allocate.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    ARTIFACTS, SEED, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, BALANCE_TOLERANCE,
)

SPLITS = ["train", "val", "test"]
FRACS = np.array([TRAIN_FRAC, VAL_FRAC, TEST_FRAC])
MAIN_VARS = ["daily_primary_source", "epoch_primary_source", "age_group", "bmi_grp",
             "daily_multisource", "epoch_multisource"]


def largest_remainder(weights: np.ndarray, total: int, tie_break="high_index") -> np.ndarray:
    """Apportion `total` across weights using largest-remainder rounding.

    tie_break='high_index' sends ties to the later index (used so that
    val precedes test in the plan's tie rule when splits are ordered train,val,test).
    """
    exact = weights / weights.sum() * total
    base = np.floor(exact).astype(int)
    rem = total - base.sum()
    frac = exact - base
    order = np.lexsort((np.arange(len(exact)), -frac))  # largest frac, then high index
    for i in order[:rem]:
        base[i] += 1
    return base


def margin_terms(df: pd.DataFrame, cols, y_col="y"):
    """Return list of (label, profile-indicator boolean array) for each category."""
    terms = []
    for c in cols:
        for k, mask in df.groupby(c, dropna=False).groups.items():
            terms.append((f"{c}={k}", np.asarray(df.index.isin(mask))))
    return terms


def cross_terms(df: pd.DataFrame, cols, y_col="y"):
    terms = []
    for c in cols:
        for k, mask in df.groupby([y_col, c], dropna=False).groups.items():
            terms.append((f"y={k[0]}|{c}={k[1]}", np.asarray(df.index.isin(mask))))
    return terms


def solve_once(df: pd.DataFrame, n_p: np.ndarray, y_p: np.ndarray,
               T: np.ndarray, C1: np.ndarray, terms, hard_margins: bool):
    P, S = len(df), 3
    n_terms = len(terms)
    nx = P * S
    na = n_terms * S
    nv = nx + 2 * na

    def xidx(p, s):
        return p * S + s

    def aidx(t, s, sign):  # sign 0=pos,1=neg
        return nx + (t * S + s) + sign * na

    rows, cols, vals, lb, ub = [], [], [], [], []

    def add(row_list, col_list, val_list, lo, hi):
        rows.extend(row_list); cols.extend(col_list); vals.extend(val_list)
        lb.append(lo); ub.append(hi)

    # 1) per-profile sum_s x = n_p
    for p in range(P):
        add([len(lb)] * S, [xidx(p, s) for s in range(S)], [1.0] * S, n_p[p], n_p[p])
    # 2) per-split totals and class-1 counts
    for s in range(S):
        add([len(lb)] * P, [xidx(p, s) for p in range(P)], [1.0] * P, T[s], T[s])
        idx = [p for p in range(P) if y_p[p] == 1]
        add([len(lb)] * len(idx), [xidx(p, s) for p in idx], [1.0] * len(idx), C1[s], C1[s])
    # 3) hard one-variable margins within floor/ceil (optional)
    if hard_margins:
        for t, (lab, mask, w) in enumerate(terms):
            if w < 1e4:            # only the main one-variable categories
                continue
            Nt = float(n_p[np.asarray(mask)].sum())
            for s in range(S):
                target = T[s] * Nt / float(n_p.sum())
                lo, hi = np.floor(target), np.ceil(target)
                idx = np.nonzero(mask)[0]
                add([len(lb)] * len(idx), [xidx(p, s) for p in idx], [1.0] * len(idx), lo, hi)
    # 4) deviation definitions: count - dpos + dneg = target
    for t, (lab, mask, w) in enumerate(terms):
        Nt = float(n_p[np.asarray(mask)].sum())
        idx = np.nonzero(mask)[0]
        for s in range(S):
            target = T[s] * Nt / float(n_p.sum())
            r, c, v = [len(lb)] * (len(idx) + 2), [xidx(p, s) for p in idx], [1.0] * len(idx)
            c += [aidx(t, s, 0), aidx(t, s, 1)]
            v += [-1.0, 1.0]
            add(r, c, v, target, target)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lb), nv))
    cons = LinearConstraint(A, np.array(lb), np.array(ub))

    c = np.zeros(nv)
    for t in range(n_terms):
        w = terms[t][2]
        for s in range(S):
            c[aidx(t, s, 0)] = w
            c[aidx(t, s, 1)] = w

    integrality = np.zeros(nv)
    integrality[:nx] = 1
    bounds = Bounds(np.zeros(nv), np.concatenate([np.repeat(n_p, S), np.full(2 * na, np.inf)]))
    res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds,
               options={"time_limit": 600, "mip_rel_gap": 0.0, "disp": False})
    return res, xidx


def main():
    m = pd.read_parquet(ARTIFACTS / "cohort_manifest.parquet")
    el = m[m["eligible"]].copy().reset_index(drop=True)
    N = len(el)
    y = el["y"].astype(int).to_numpy()
    print(f"eligible N={N}  class1={int(y.sum())} class0={int((1-y).sum())}")

    daily_bits = [c for c in el.columns if c.startswith("daily_src_")]
    epoch_bits = [c for c in el.columns if c.startswith("epoch_src_")]
    prof_cols = (["y"] + MAIN_VARS + daily_bits + epoch_bits)
    el["_prof"] = el[prof_cols].astype(str).agg("|".join, axis=1)
    codes, uniq = pd.factorize(el["_prof"], sort=True)
    el["_pid"] = codes
    P = len(uniq)
    prof = el.groupby("_pid").agg(n=("y", "size"), y=("y", "first")).sort_index()
    n_p = prof["n"].to_numpy()
    y_p = prof["y"].astype(int).to_numpy()
    print(f"allocation profiles: {P}")

    # ---- targets ----------------------------------------------------------
    T = largest_remainder(FRACS, N)              # [train,val,test]
    C1 = largest_remainder(FRACS * (y.mean()), int(y.sum()))
    # reconcile class-1 total and per-split ceilings
    C1 = C1.astype(int)
    while C1.sum() < int(y.sum()):
        C1[np.argmax(FRACS * y.mean() - C1)] += 1
    while C1.sum() > int(y.sum()):
        C1[np.argmin(FRACS * y.mean() - C1)] -= 1
    print("target totals", dict(zip(SPLITS, T.tolist())),
          "class1", dict(zip(SPLITS, C1.tolist())))

    # joint strata key (salutation x daily source set x epoch source set x age x bmi)
    el["_joint"] = (el["y"].astype(str) + "|" + el["daily_source_set"].astype(str) + "|" +
                    el["epoch_source_set"].astype(str) + "|" + el["age_group"].astype(str) + "|" +
                    el["bmi_grp"].astype(str))
    js = el.groupby("_joint").size()
    el["_jsize"] = el["_joint"].map(js)
    el["_rarity"] = np.where(el["_jsize"] <= 2, "1-2",
                             np.where(el["_jsize"] <= 9, "3-9", ">=10"))

    # profile-level table (one row per profile; categoricals are homogeneous)
    pdf = (el.drop_duplicates("_pid").sort_values("_pid").reset_index(drop=True))
    n_p = prof["n"].reindex(pdf["_pid"].to_numpy()).to_numpy()
    y_p = pdf["y"].astype(int).to_numpy()
    P = len(pdf)

    def term_list(lst, weight):
        return [(lab, mask, weight) for lab, mask in lst]

    main = margin_terms(pdf, MAIN_VARS + daily_bits + epoch_bits)
    inter = cross_terms(pdf, MAIN_VARS + daily_bits + epoch_bits)
    joint_all = margin_terms(pdf, ["_joint"])
    joint_ge10 = [(lab, mask) for lab, mask in joint_all
                  if int(n_p[np.asarray(mask)].sum()) >= 10]
    rarity = margin_terms(pdf, ["_rarity"])
    terms = (term_list(main, 1e4) + term_list(inter, 1e2) +
             term_list(joint_ge10, 1.0) + term_list(rarity, 10.0))
    print(f"balance terms: main={len(main)} inter={len(inter)} joint>=10={len(joint_ge10)} rarity={len(rarity)}")

    # ---- solve: hard margins, then soft fallback --------------------------
    res, xidx = solve_once(pdf, n_p, y_p, T, C1, terms, hard_margins=True)
    mode = "hard_margins"
    if res.status != 0:
        print(f"hard-margin MILP status={res.status} ({res.message}); retrying soft.")
        res, xidx = solve_once(pdf, n_p, y_p, T, C1, terms, hard_margins=False)
        mode = "soft_margins"
    print(f"solver status={res.status} message={res.message} mode={mode} "
          f"fun={res.fun:.2f} gap={getattr(res,'mip_gap',None)}")

    x = np.round(res.x[:P * 3]).astype(int)
    alloc = pd.DataFrame({"pid": np.repeat(np.arange(P), 3),
                          "split": np.tile(np.arange(3), P),
                          "n": x})
    alloc = alloc.pivot(index="pid", columns="split", values="n")
    alloc.columns = SPLITS
    alloc = alloc.reindex(range(P), fill_value=0)

    # ---- seeded random assignment within each profile/split ----------------
    rng = np.random.default_rng(SEED)
    el["split"] = pd.NA
    for pid, g in el.groupby("_pid", sort=True):
        idx = np.array(g.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        pos = 0
        for s in SPLITS:
            k = int(alloc.loc[pid, s])
            el.loc[idx[pos:pos + k], "split"] = s
            pos += k
        assert pos == len(idx), (pid, pos, len(idx))

    assert el["split"].notna().all()
    split_manifest = el[["user_id", "split"]].copy()
    split_manifest.to_csv(ARTIFACTS / "split_manifest.csv", index=False)

    # ---- report -----------------------------------------------------------
    el["split"] = pd.Categorical(el["split"], categories=SPLITS, ordered=True)
    report = {"N": N, "targets": dict(zip(SPLITS, T.tolist())),
              "class1_targets": dict(zip(SPLITS, C1.tolist())),
              "solver_status": int(res.status), "solver_mode": mode,
              "solver_fun": float(res.fun), "n_profiles": int(P)}
    abs_pairs = []
    out_lines = ["# Split allocation report", "",
                 f"Eligible N = {N}; solver status = {res.status} ({mode}); "
                 f"objective = {res.fun:.2f}; {P} allocation profiles.", "",
                 "## Split totals and salutation", "",
                 "| split | total | target | sal10 | sal20 |", "|---|---:|---:|---:|---:|"]
    for s in SPLITS:
        g = el[el["split"] == s]
        out_lines.append(f"| {s} | {len(g)} | {T[SPLITS.index(s)]} | "
                         f"{int((g.y==0).sum())} | {int((g.y==1).sum())} |")
    out_lines += ["", "## One-variable category proportions (pp difference vs train)", ""]
    for c in MAIN_VARS + daily_bits + epoch_bits + ["_rarity"]:
        tab = pd.crosstab(el[c], el["split"], normalize="columns")
        tab = tab.reindex(columns=SPLITS).fillna(0.0)
        for k in tab.index:
            props = tab.loc[k].to_numpy() * 100
            mx, mn = props.max(), props.min()
            abs_pairs.append((f"{c}={k}", mx - mn))
        out_lines.append(f"\n### {c}\n")
        out_lines.append("| category | train % | val % | test % | max pairwise pp |")
        out_lines.append("|---|---:|---:|---:|---:|")
        for k in tab.index:
            props = tab.loc[k].to_numpy() * 100
            out_lines.append(f"| {k} | {props[0]:.2f} | {props[1]:.2f} | {props[2]:.2f} | "
                             f"{props.max()-props.min():.2f} |")
    worst = sorted(abs_pairs, key=lambda t: -t[1])[:15]
    out_lines += ["", "## Largest one-variable pairwise differences (pp)", "",
                  "| category | max pairwise pp |", "|---|---:|"]
    out_lines += [f"| {k} | {v:.3f} |" for k, v in worst]
    over = [t for t in abs_pairs if t[1] > BALANCE_TOLERANCE * 100]
    out_lines += ["", f"Categories exceeding {BALANCE_TOLERANCE*100:.1f} pp: {len(over)}"]
    for k, v in over:
        out_lines.append(f"- {k}: {v:.3f} pp")

    with open(ARTIFACTS / "split_allocation_report.md", "w") as fh:
        fh.write("\n".join(out_lines) + "\n")
    with open(ARTIFACTS / "split_allocation_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"max one-variable pairwise diff = {max(v for _, v in abs_pairs):.3f} pp; "
          f"over 1pp: {len(over)}")
    for k, v in worst[:8]:
        print(f"  {k}: {v:.3f} pp")


if __name__ == "__main__":
    main()
