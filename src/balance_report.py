"""Comprehensive split balance report (main margins, interactions, joint strata,
rare-profile representation, continuous diagnostics).

Outputs: artifacts/split_balance.csv, artifacts/split_balance_report.md
Run:  python src/balance_report.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ARTIFACTS, WHO_CSV, BALANCE_TOLERANCE  # noqa: E402

SPLITS = ["train", "val", "test"]
MAIN_VARS = ["daily_primary_source", "epoch_primary_source", "age_group", "bmi_grp",
             "daily_multisource", "epoch_multisource"]


def smd(a, b):
    ma, mb = np.nanmean(a), np.nanmean(b)
    sa, sb = np.nanstd(a, ddof=1), np.nanstd(b, ddof=1)
    denom = np.sqrt((sa ** 2 + sb ** 2) / 2)
    return float((ma - mb) / denom) if denom > 0 else np.nan


def main():
    sm = pd.read_csv(ARTIFACTS / "split_manifest.csv")
    man = pd.read_parquet(ARTIFACTS / "cohort_manifest.parquet")
    el = man[man["eligible"]].merge(sm, on="user_id", how="left")
    who = pd.read_csv(WHO_CSV, usecols=["user", "age"]).rename(
        columns={"user": "user_id", "age": "birth_year"})
    el = el.merge(who, on="user_id", how="left")
    el["epoch_span_days"] = el["last_date"] - el["first_date"] + 1

    daily_bits = [c for c in el.columns if c.startswith("daily_src_")]
    epoch_bits = [c for c in el.columns if c.startswith("epoch_src_")]
    el["_joint"] = (el["y"].astype(str) + "|" + el["daily_source_set"].astype(str) + "|" +
                    el["epoch_source_set"].astype(str) + "|" + el["age_group"].astype(str) + "|" +
                    el["bmi_grp"].astype(str))
    jsize = el.groupby("_joint").size()
    el["_rarity"] = pd.cut(jsize.reindex(el["_joint"]).to_numpy(), [0, 2, 9, np.inf],
                           labels=["1-2", "3-9", ">=10"])

    rows = []
    abs_pairs = []

    def add_var(name, series, conditional=None):
        if conditional is None:
            tab = pd.crosstab(series, el["split"]).reindex(columns=SPLITS).fillna(0)
        else:
            tab = pd.crosstab([conditional, series], el["split"]).reindex(columns=SPLITS).fillna(0)
        props = tab.div(tab.sum(axis=0), axis=1)
        for idx in tab.index:
            p = props.loc[idx].to_numpy() * 100
            pair = float(np.nanmax(p) - np.nanmin(p))
            abs_pairs.append((name, str(idx), pair))
            rows.append(dict(variable=name, category=str(idx),
                             **{f"{s}_count": int(tab.loc[idx, s]) for s in SPLITS},
                             **{f"{s}_pct": float(props.loc[idx, s] * 100) for s in SPLITS},
                             train_val_pp=float(p[0] - p[1]),
                             train_test_pp=float(p[0] - p[2]),
                             val_test_pp=float(p[1] - p[2]),
                             max_pairwise_pp=pair))

    for c in MAIN_VARS + daily_bits + epoch_bits + ["_rarity"]:
        add_var(c, el[c])
        add_var(f"y_x_{c}", el[c], conditional=el["y"])

    # joint strata (n>=10): within-split prevalence, then pairwise pp
    jtab = pd.crosstab(el["_joint"], el["split"]).reindex(columns=SPLITS).fillna(0)
    jtot = jtab.sum(axis=1)
    big = jtab[jtot >= 10]
    jprops = jtab.div(jtab.sum(axis=0), axis=1)
    for idx in big.index:
        p = jprops.loc[idx].to_numpy() * 100
        abs_pairs.append((f"joint>=10", str(idx), float(np.nanmax(p) - np.nanmin(p))))

    pd.DataFrame(rows).to_csv(ARTIFACTS / "split_balance.csv", index=False)

    # continuous diagnostics
    cont = ["birth_year", "epoch_span_days", "adequate_days_max", "valid_events"]
    crows = []
    for c in cont:
        for s in SPLITS:
            v = el.loc[el.split == s, c].astype(float)
            crows.append(dict(variable=c, split=s, n=int(v.notna().sum()),
                              mean=v.mean(), sd=v.std(ddof=1), median=v.median(),
                              q25=v.quantile(.25), q75=v.quantile(.75),
                              smd_vs_train=np.nan if s == "train" else
                              smd(el.loc[el.split == "train", c], v)))
    cont_df = pd.DataFrame(crows)
    cont_df.to_csv(ARTIFACTS / "split_balance_continuous.csv", index=False)

    # rare-profile representation: strata with zero members in val/test
    jc = jtab.astype(int)
    zero_val = int((jc.loc[big.index, "val"] == 0).sum())
    zero_test = int((jc.loc[big.index, "test"] == 0).sum())
    zero_val_all = int((jc["val"] == 0).sum())
    zero_test_all = int((jc["test"] == 0).sum())
    rare = el.groupby("_rarity", observed=False)["split"].value_counts().unstack().reindex(columns=SPLITS).fillna(0)

    worst = sorted(abs_pairs, key=lambda t: -t[2])
    over = [t for t in abs_pairs if t[2] > BALANCE_TOLERANCE * 100]

    lines = ["# Split balance report", "",
             f"Eligible participants: {len(el)}; train/val/test = "
             f"{[int((el.split==s).sum()) for s in SPLITS]}.", "",
             "Target: <= 1 percentage point (pp) pairwise difference in every main category "
             "proportion. Continuous SMD < 0.05 is a supplementary target.", "",
             "## Largest pairwise differences (pp)", "",
             "| variable | category | max pairwise pp |", "|---|---|---:|"]
    lines += [f"| {v} | {k} | {p:.3f} |" for v, k, p in worst[:20]]
    lines += ["", f"Main/interaction categories exceeding 1 pp: {len(over)}. "
              f"Joint strata (size>=10) with no validation member: {zero_val}; no test member: {zero_test}. "
              f"All joint strata (including sparse source-set profiles) with no validation member: "
              f"{zero_val_all}; no test member: {zero_test_all}.", ""]
    if over:
        lines += ["### Exceeding categories", ""]
        lines += [f"- {v} [{k}]: {p:.3f} pp" for v, k, p in over]
        lines.append("")
    lines += ["## Salutation-conditional margins (max pairwise pp over categories)", "",
              "| variable | max pairwise pp |", "|---|---:|"]
    for c in MAIN_VARS + ["_rarity"]:
        m = max(p for v, k, p in abs_pairs if v == f"y_x_{c}")
        lines.append(f"| {c} | {m:.3f} |")
    lines += ["", "## Rare-profile representation (joint strata size buckets)", "",
              "| bucket | train | val | test |", "|---|---:|---:|---:|"]
    for b in ["1-2", "3-9", ">=10"]:
        lines.append(f"| {b} | {int(rare.loc[b,'train'])} | {int(rare.loc[b,'val'])} | "
                     f"{int(rare.loc[b,'test'])} |")
    lines += ["", "## Continuous diagnostics", "",
              "| variable | split | n | mean | SD | median | IQR | SMD vs train |",
              "|---|---|---:|---:|---:|---:|---|---:|"]
    for _, r in cont_df.iterrows():
        lines.append(f"| {r.variable} | {r.split} | {int(r.n)} | {r['mean']:.2f} | {r.sd:.2f} | "
                     f"{r['median']:.2f} | [{r.q25:.2f}, {r.q75:.2f}] | "
                     f"{'' if np.isnan(r.smd_vs_train) else f'{r.smd_vs_train:.3f}'} |")
    lines += ["", "Non-significant balance tests would not establish equality; these are descriptive "
              "design diagnostics. Covariate-based allocation never used labels through epoch values, "
              "classifier results or test performance."]
    with open(ARTIFACTS / "split_balance_report.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"max pairwise pp overall = {worst[0][2]:.3f} ({worst[0][0]} [{worst[0][1]}])")
    print(f"categories over 1pp: {len(over)}  joint zero-val {zero_val} zero-test {zero_test}")
    print(cont_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
