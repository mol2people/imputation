"""Side-quest sections 1-2: longitudinal strict-coverage cohorts.

Builds cohorts from the already-materialized v2 aggregates (no full raw
re-scan) and writes:

  artifacts_sq/eligibility_by_channel.parquet        one row per user/channel
  artifacts_sq/cohort_manifest_all_sources.parquet   every any-core passer
  artifacts_sq/cohort_manifest_model_sources.parquet union of sources 3/6/7
  artifacts_sq/source_cohorts.csv                    base/per-channel/any-core
  artifacts_sq/coverage_semantics_audit.csv          cov_s > 24 h by source/channel

The frozen count tables in sq_config must be reproduced exactly; otherwise the
run stops and reports input or semantic drift (plan section 2).

Run:  python src/sidequest/cohorts.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sq_config import (  # noqa: E402
    ARTIFACTS_SQ, ARTIFACTS_V2, CHANNEL_PRIORITY, CORE_CHANNELS,
    FROZEN_ALL_SOURCE_PASS, FROZEN_ANY_CORE, FROZEN_ANY_CORE_CLASS,
    FROZEN_BASE_SINGLE_SOURCE, FROZEN_CHANNEL_PASS, FROZEN_MODEL_UNION,
    MIN_ADEQUATE_WEEKS, MIN_CLASS_FOR_MODEL, MIN_STRICT_DAYS, MODEL_SOURCE_IDS,
    STRICT_COV_S, WINDOW_INCLUSIVE_DAYS, WINDOW_WEEKS,
)


def iso_monday(d: np.ndarray) -> np.ndarray:
    """ISO-week Monday (days since epoch) for day indices ``d``.

    1970-01-01 is a Thursday, so dow = (d + 3) mod 7 with Monday = 0 and the
    Monday of the ISO week containing ``d`` is ``d - dow``.
    """
    d = np.asarray(d, dtype=np.int64)
    return d - ((d + 3) % 7)


def _date_str(d) -> str | None:
    if d is None or not np.isfinite(d):
        return None
    return pd.Timestamp(int(d), unit="D").strftime("%Y-%m-%d")


def _pass_channel(g: pd.DataFrame) -> dict:
    """Gate evaluation for one (user, channel) weekly strict-day series."""
    f = int(g["monday"].iloc[0])
    l = int(g["monday"].iloc[-1])
    n_weeks = (l - f) // 7 + 1
    d = np.zeros(n_weeks, dtype=np.int64)          # dense grid; absent weeks = 0
    d[(g["monday"].to_numpy(dtype=np.int64) - f) // 7] = g["d"].to_numpy(dtype=np.int64)
    a = (d >= 5).astype(np.int64)                  # adequate week = >= 5 strict days

    rec = {"n_day_rows": -1, "n_weeks": int(n_weeks), "first_monday": f, "last_monday": l,
           "n_candidates": 0, "max_adequate_weeks": 0,
           "max_strict_days_among_week_qualifying": 0,
           "pass": False, "reason": "",
           "best_window_start": pd.NA, "best_adequate_weeks": pd.NA,
           "best_strict_days": pd.NA}

    if n_weeks < WINDOW_WEEKS:                     # candidates stay inside observed span
        rec["reason"] = "span_lt_13_weeks"
        return rec

    cs_d = np.concatenate([[0], np.cumsum(d)])
    cs_a = np.concatenate([[0], np.cumsum(a)])
    starts = np.arange(0, n_weeks - WINDOW_WEEKS + 1)
    sd = cs_d[starts + WINDOW_WEEKS] - cs_d[starts]     # strict days incl. slack weeks
    sa = cs_a[starts + WINDOW_WEEKS] - cs_a[starts]
    rec["n_candidates"] = int(starts.size)
    rec["max_adequate_weeks"] = int(sa.max())
    if sa.max() < MIN_ADEQUATE_WEEKS:
        rec["reason"] = "max_adequate_weeks_lt_11"
        return rec
    rec["max_strict_days_among_week_qualifying"] = int(sd[sa >= MIN_ADEQUATE_WEEKS].max())
    ok = (sa >= MIN_ADEQUATE_WEEKS) & (sd >= MIN_STRICT_DAYS)
    if not ok.any():
        rec["reason"] = "max_strict_days_among_week_qualifying_lt_65"
        return rec
    best = int(np.argmax(np.where(ok, sd, -1)))         # strict days desc, then earliest
    rec["pass"], rec["reason"] = True, "pass"
    rec["best_window_start"] = int(starts[best])
    rec["best_adequate_weeks"] = int(sa[best])
    rec["best_strict_days"] = int(sd[best])
    return rec


def sole_source_table() -> pd.DataFrame:
    """v2-eligible participants with exactly one retained epoch source."""
    man = pd.read_parquet(ARTIFACTS_V2 / "cohort_manifest.parquet")
    el = man[man["eligible"]].copy()
    es = pd.read_parquet(ARTIFACTS_V2 / "epoch_sources.parquet")
    es = es[es["events"] > 0]
    nsrc = es.groupby("user")["source"].nunique()
    sole = nsrc[nsrc == 1].index
    srcmap = es[es["user"].isin(sole)].groupby("user")["source"].first()
    el = el[el["user_id"].isin(sole)].copy()
    el["source_id"] = el["user_id"].map(srcmap).astype("int64")
    return el[["user_id", "source_id", "y", "salutation", "age_group", "bmi_grp"]].copy()


def build() -> None:
    el = sole_source_table()
    base_counts = el.groupby("source_id").size().to_dict()
    for s, n in FROZEN_BASE_SINGLE_SOURCE.items():
        if base_counts.get(s, 0) != n:
            sys.exit(f"STOP: single-source base count drift for source {s}: "
                     f"{base_counts.get(s, 0)} != {n}")
    print(f"single-source v2-eligible: {len(el):,}", flush=True)

    days = pd.read_parquet(ARTIFACTS_V2 / "epoch_days.parquet",
                           columns=["user", "channel", "date", "n", "cov_s", "hours"])
    users = set(el["user_id"])
    days = days[days["user"].isin(users)].copy()
    days["monday"] = iso_monday(days["date"].to_numpy())
    days["strict"] = days["cov_s"].to_numpy() >= STRICT_COV_S
    n_day_rows = days.groupby(["user", "channel"]).size().rename("n_day_rows")

    wk = (days.groupby(["user", "channel", "monday"], sort=True)["strict"]
          .sum().rename("d").reset_index())
    print(f"weekly rows: {len(wk):,} over {wk.groupby(['user', 'channel']).ngroups:,} "
          f"user/channel series", flush=True)

    rows = []
    for (u, c), g in wk.groupby(["user", "channel"], sort=False):
        rec = _pass_channel(g)
        rec.update({"user": u, "channel": c})
        rows.append(rec)
    ch = pd.DataFrame(rows)

    # complete the (user, channel) grid so absent channels get no_channel_rows
    grid = el[["user_id"]].copy()
    grid["key"] = 1
    chans = pd.DataFrame({"channel": list(CORE_CHANNELS), "key": 1})
    grid = grid.merge(chans, on="key").drop(columns="key")
    ch = grid.merge(ch, left_on=["user_id", "channel"], right_on=["user", "channel"],
                    how="left").drop(columns=["user"])
    ch["reason"] = ch["reason"].fillna("no_channel_rows")
    ch["pass"] = ch["pass"].fillna(False).astype(bool)
    ch = ch.merge(n_day_rows.reset_index(), left_on=["user_id", "channel"],
                  right_on=["user", "channel"], how="left",
                  suffixes=("", "_obs")).drop(columns=["user"])
    ch["n_day_rows"] = ch["n_day_rows_obs"].fillna(ch["n_day_rows"]).fillna(0).astype("int64")
    ch = ch.drop(columns=["n_day_rows_obs"])
    ch["source_id"] = ch["user_id"].map(el.set_index("user_id")["source_id"]).astype("Int64")

    # ---- frozen count checks -------------------------------------------------
    chan_counts = ch[ch["pass"]].groupby(["source_id", "channel"]).size().to_dict()
    for s, exp in FROZEN_CHANNEL_PASS.items():
        for c in CORE_CHANNELS:
            got = int(chan_counts.get((s, c), 0))
            if got != exp[c]:
                sys.exit(f"STOP: channel pass count drift: source {s} ch{c} {got} != {exp[c]}")
    qual = ch[ch["pass"]].groupby("user_id")["channel"].apply(list)
    el["qualified_channels"] = el["user_id"].map(qual.to_dict()).apply(
        lambda v: v if isinstance(v, list) else [])
    el["any_core_pass"] = el["qualified_channels"].apply(len) > 0
    for s, exp in FROZEN_ANY_CORE.items():
        got = int(((el["source_id"] == s) & el["any_core_pass"]).sum())
        if got != exp:
            sys.exit(f"STOP: any-core count drift for source {s}: {got} != {exp}")
        sub = el.loc[(el["source_id"] == s) & el["any_core_pass"], "y"]
        got_cls = (int((sub == 0).sum()), int((sub == 1).sum()))
        if got_cls != FROZEN_ANY_CORE_CLASS[s]:
            sys.exit(f"STOP: class split drift for source {s}: {got_cls} != "
                     f"{FROZEN_ANY_CORE_CLASS[s]}")
    n_all = int(el["any_core_pass"].sum())
    if n_all != FROZEN_ALL_SOURCE_PASS:
        sys.exit(f"STOP: all-source pass union drift: {n_all} != {FROZEN_ALL_SOURCE_PASS}")

    # ---- deterministic selection across channels and windows ------------------
    passing = ch[ch["pass"]].copy()
    passing["chan_rank"] = passing["channel"].map({c: i for i, c in enumerate(CHANNEL_PRIORITY)})
    passing = passing.sort_values(
        ["user_id", "best_strict_days", "best_window_start", "chan_rank"],
        ascending=[True, False, True, True])
    sel = passing.groupby("user_id").first()[["channel", "best_window_start",
                                              "first_monday", "best_adequate_weeks",
                                              "best_strict_days"]]
    el["selected_channel"] = el["user_id"].map(sel["channel"])
    # best_window_start is a 0-based week index counted from the selected
    # channel's first observed Monday; persist the actual window Monday
    # (epoch day, same space as epoch_days.date) instead of the index.
    el["window_start"] = (el["user_id"].map(sel["first_monday"])
                          + 7 * el["user_id"].map(sel["best_window_start"])).astype("Int64")
    el["adequate_weeks"] = el["user_id"].map(sel["best_adequate_weeks"])
    el["strict_days"] = el["user_id"].map(sel["best_strict_days"])
    ch["selected"] = ch["user_id"].isin(sel.index) & (
        ch["channel"] == ch["user_id"].map(sel["channel"]))

    el["pass_pattern"] = el["qualified_channels"].apply(
        lambda v: "+".join(str(c) for c in sorted(v)))
    el["window_end"] = el["window_start"] + WINDOW_INCLUSIVE_DAYS - 1
    el["window_start_date"] = el["window_start"].map(
        lambda v: _date_str(v) if pd.notna(v) else None)
    el["window_end_date"] = el["window_end"].map(
        lambda v: _date_str(v) if pd.notna(v) else None)

    # window sanity: epoch-day Mondays, inside the selected channel's candidate span
    _p = el[el["any_core_pass"]]
    _ws = _p["window_start"].astype("int64").to_numpy()
    assert (iso_monday(_ws) == _ws).all(), "window_start not on a Monday"
    _sel_span = ch[ch["selected"]].set_index("user_id")
    _fm = _p["user_id"].map(_sel_span["first_monday"]).astype("int64").to_numpy()
    _lm = _p["user_id"].map(_sel_span["last_monday"]).astype("int64").to_numpy()
    assert ((_ws >= _fm) & (_ws <= _lm - 7 * (WINDOW_WEEKS - 1))).all(), \
        "window_start outside candidate range"

    # ---- modelling disposition (count-only rule, plan section 2) ---------------
    cls_ok = {}
    for s in sorted(el["source_id"].unique()):
        sub = el.loc[(el["source_id"] == s) & el["any_core_pass"], "y"]
        cls_ok[s] = (int((sub == 0).sum()) >= MIN_CLASS_FOR_MODEL and
                     int((sub == 1).sum()) >= MIN_CLASS_FOR_MODEL)
    el["model_eligible"] = [bool(p) and bool(cls_ok[s])
                            for s, p in zip(el["source_id"], el["any_core_pass"])]
    el["disposition"] = np.where(el["model_eligible"], "model",
                                 np.where(el["any_core_pass"], "audit only", "not eligible"))
    n_model = int(el["model_eligible"].sum())
    if n_model != FROZEN_MODEL_UNION:
        sys.exit(f"STOP: model-eligible union drift: {n_model} != {FROZEN_MODEL_UNION}")

    # ---- persistence ------------------------------------------------------------
    ARTIFACTS_SQ.mkdir(parents=True, exist_ok=True)
    elig = ch.drop(columns=[c for c in ("selected_channel",) if c in ch.columns])
    elig.to_parquet(ARTIFACTS_SQ / "eligibility_by_channel.parquet", index=False)

    manifest_cols = ["user_id", "source_id", "y", "salutation", "age_group", "bmi_grp",
                     "qualified_channels", "pass_pattern", "any_core_pass",
                     "model_eligible", "disposition", "selected_channel",
                     "window_start", "window_end", "window_start_date", "window_end_date",
                     "adequate_weeks", "strict_days"]
    for name, mask in (("cohort_manifest_all_sources.parquet", el["any_core_pass"]),
                       ("cohort_manifest_model_sources.parquet", el["model_eligible"])):
        out = el.loc[mask, manifest_cols].copy()
        out["qualified_channels"] = out["qualified_channels"].apply(
            lambda v: "+".join(str(c) for c in sorted(v)))
        out.to_parquet(ARTIFACTS_SQ / name, index=False)
        if name.startswith("cohort_manifest_model"):
            for src in MODEL_SOURCE_IDS:
                d = ARTIFACTS_SQ / f"source_{src}"
                d.mkdir(parents=True, exist_ok=True)
                out[out["source_id"] == src].to_parquet(d / "cohort_manifest.parquet",
                                                        index=False)

    sc = []
    for s in sorted(el["source_id"].unique()):
        sub = el[(el["source_id"] == s) & el["any_core_pass"]]
        sc.append({"source_id": s,
                   "base_single_source": int((el["source_id"] == s).sum()),
                   **{f"ch{c}_pass": int(chan_counts.get((s, c), 0)) for c in CORE_CHANNELS},
                   "any_core_pass": len(sub),
                   "sal10": int((sub["y"] == 0).sum()),
                   "sal20": int((sub["y"] == 1).sum()),
                   "model_eligible": bool(cls_ok.get(s, False)),
                   "disposition": "model" if cls_ok.get(s, False) else "audit only"})
    pd.DataFrame(sc).to_csv(ARTIFACTS_SQ / "source_cohorts.csv", index=False)

    # ---- coverage semantics audit (frozen cov_s artifact semantics) --------------
    cov = pd.read_parquet(ARTIFACTS_V2 / "epoch_days.parquet",
                          columns=["user", "channel", "date", "cov_s"])
    cov = cov[cov["user"].isin(users)]
    dom = pd.read_parquet(ARTIFACTS_V2 / "epoch_sources.parquet")
    dom = (dom[dom["events"] > 0]
           .sort_values(["user", "channel", "events"], ascending=[True, True, False])
           .groupby(["user", "channel"]).first().reset_index()[["user", "channel", "source"]])
    cov = cov.merge(dom.rename(columns={"source": "source_id"}),
                    on=["user", "channel"], how="left")
    over = cov[cov["cov_s"] > 86_400.0]
    agg = (over.groupby(["source_id", "channel"])
           .agg(n_days_over_24h=("cov_s", "size"), max_cov_s=("cov_s", "max")).reset_index())
    ex = (over.loc[over.groupby(["source_id", "channel"])["cov_s"].idxmax(),
                   ["source_id", "channel", "user", "date", "cov_s"]]
          .rename(columns={"user": "example_user", "date": "example_date",
                           "cov_s": "example_cov_s"}))
    tot = (cov.groupby(["source_id", "channel"]).size().rename("n_day_rows").reset_index())
    audit = (tot.merge(agg, on=["source_id", "channel"], how="left")
             .merge(ex, on=["source_id", "channel"], how="left")
             .fillna({"n_days_over_24h": 0, "max_cov_s": 0.0}))
    audit["share_over_24h"] = audit["n_days_over_24h"] / audit["n_day_rows"]
    audit["note"] = ("cov_s unions full UTC intervals assigned to the local date of the "
                     "interval start without midnight clipping (frozen artifact semantics); "
                     "source_id is the dominant epoch source of that user/channel")
    audit.to_csv(ARTIFACTS_SQ / "coverage_semantics_audit.csv", index=False)

    print(f"any-core passers: {n_all:,}; model-eligible union: {n_model:,}")
    print(pd.DataFrame(sc).to_string(index=False))
    print("all frozen count checks passed; artifacts written to", ARTIFACTS_SQ)


if __name__ == "__main__":
    build()
