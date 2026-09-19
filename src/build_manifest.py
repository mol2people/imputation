"""Build the cohort manifest: daily source profile, labels, eligibility flags,
coverage gate, and epoch source profile.

Outputs:
  artifacts/cohort_manifest.parquet   one row per candidate daily user
  artifacts/cohort_flow.csv           mutually-exclusive exclusion flow

Run:  python src/build_manifest.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    ARTIFACTS, DAILY_CSV, SALUTATION_CSV, WHO_CSV, DAILY_TYPES,
    EXCLUDED_SOURCES, KEEP_SALUTATIONS, ADEQUATE_HOURS, ADEQUATE_COV_H,
    ADEQUATE_N, MIN_ADEQUATE_DAYS,
)


def build_daily_profile() -> pd.DataFrame:
    daily = pd.read_csv(DAILY_CSV, usecols=["user_id", "date", "type", "source"])
    daily = daily[daily["type"].isin(DAILY_TYPES)]
    all_users = pd.Index(sorted(daily["user_id"].unique()), name="user_id")
    allowed = daily[~daily["source"].isin(EXCLUDED_SOURCES)].copy()

    # distinct observed dates per user/source (rows and duplicated dates collapse)
    ds = (allowed.drop_duplicates(["user_id", "source", "date"])
          .groupby(["user_id", "source"], sort=True).size().rename("ndates").reset_index())
    # primary daily source: max distinct dates, smallest numeric code breaks ties
    ds = ds.sort_values(["user_id", "ndates", "source"], ascending=[True, False, True])
    primary = ds.drop_duplicates("user_id").set_index("user_id")["source"].rename("daily_primary_source")

    sets = (allowed.groupby("user_id")["source"]
            .apply(lambda s: ",".join(str(int(v)) for v in sorted(s.unique())))
            .rename("daily_source_set"))
    nsrc = sets.apply(len).rename("daily_n_sources")
    multisource = (nsrc > 1).rename("daily_multisource")

    # one binary indicator per source actually observed
    obs_sources = sorted(allowed["source"].unique().tolist())
    members = (allowed.drop_duplicates(["user_id", "source"])
               .assign(one=1)
               .pivot(index="user_id", columns="source", values="one")
               .reindex(all_users)
               .fillna(0).astype("int8"))
    members.columns = [f"daily_src_{c}" for c in members.columns]

    prof = pd.DataFrame(index=all_users)
    prof = prof.join(sets).join(primary).join(nsrc).join(multisource)
    prof = prof.join(members)
    return prof.reset_index()


def coverage_gate(days: pd.DataFrame) -> pd.DataFrame:
    days = days.copy()
    days["adequate"] = ((days["hours"] >= ADEQUATE_HOURS) &
                        ((days["cov_s"] >= ADEQUATE_COV_H * 3600) | (days["n"] >= ADEQUATE_N)))
    per_chan = (days[days["adequate"]].groupby(["user", "channel"]).size()
                .rename("adequate_days").reset_index())
    agg = per_chan.groupby("user").agg(
        adequate_days_max=("adequate_days", "max"),
        n_channels_adequate=("channel", "nunique"))
    gate = pd.DataFrame(index=pd.Index([], name="user"))
    gate = agg
    gate["gate_pass"] = gate["adequate_days_max"] >= MIN_ADEQUATE_DAYS
    return gate.reset_index().rename(columns={"user": "user_id"}).set_index("user_id")


def epoch_source_profile(esrc: pd.DataFrame):
    if esrc is None or esrc.empty:
        return pd.DataFrame(columns=["user_id", "epoch_source_set", "epoch_primary_source",
                                     "epoch_n_sources", "epoch_multisource"])
    # approx distinct valid days per source: max over channels (channels co-occur)
    per_src = (esrc.groupby(["user", "source"])["days"].max().rename("days").reset_index())
    per_src = per_src.sort_values(["user", "days", "source"], ascending=[True, False, True])
    primary = per_src.drop_duplicates("user").set_index("user")["source"].rename("epoch_primary_source")
    sets = (esrc.groupby("user")["source"]
            .apply(lambda s: ",".join(str(int(v)) for v in sorted(s.unique())))
            .rename("epoch_source_set"))
    nsrc = sets.apply(lambda x: len(x.split(","))).rename("epoch_n_sources")
    out = pd.DataFrame({"epoch_primary_source": primary,
                        "epoch_source_set": sets,
                        "epoch_n_sources": nsrc})
    out["epoch_multisource"] = out["epoch_n_sources"] > 1
    # one binary membership indicator per observed epoch source code
    members = (esrc.drop_duplicates(["user", "source"]).assign(one=1)
               .pivot(index="user", columns="source", values="one").fillna(0).astype("int8"))
    members.columns = [f"epoch_src_{c}" for c in members.columns]
    out = out.join(members)
    return out.reset_index().rename(columns={"user": "user_id"})


def main():
    prof = build_daily_profile()
    all_users = prof["user_id"].tolist()
    print(f"daily users (type 65/66, before source exclusion source): {len(all_users)}")

    daily_all = pd.read_csv(DAILY_CSV, usecols=["user_id", "type", "source"])
    daily_all = daily_all[daily_all["type"].isin(DAILY_TYPES)]
    allowed_users = set(daily_all.loc[~daily_all["source"].isin(EXCLUDED_SOURCES), "user_id"])

    sal = pd.read_csv(SALUTATION_CSV, usecols=["user_id", "salutation"])
    who = pd.read_csv(WHO_CSV, usecols=["user", "age_group", "bmi_grp"]).rename(columns={"user": "user_id"})
    prof = prof.merge(sal, on="user_id", how="left").merge(who, on="user_id", how="left")

    # ---- epoch file status from disk + scan metadata
    import glob
    meta = pd.read_parquet(ARTIFACTS / "epoch_meta.parquet").rename(columns={"user": "user_id"})
    present = {int(os.path.splitext(os.path.basename(p))[0]): os.path.getsize(p)
               for p in glob.glob(str(ARTIFACTS.parent / "out" / "*.csv"))}
    prof["epoch_file_size"] = prof["user_id"].map(present).fillna(-1).astype("int64")
    prof = prof.merge(meta, on="user_id", how="left")

    prof["has_allowed_daily"] = prof["user_id"].isin(allowed_users)
    prof["epoch_file_exists"] = prof["epoch_file_size"] >= 0
    prof["epoch_zero_byte"] = prof["epoch_file_size"] == 0
    prof["epoch_header_only"] = ((prof["epoch_file_size"] > 0) &
                                 prof["raw_events"].fillna(0).eq(0) &
                                 prof["unreadable"].isna() if "unreadable" in prof else False)
    prof["epoch_unreadable"] = prof.get("unreadable", pd.Series(index=prof.index)).notna()
    prof["epoch_no_valid"] = prof["valid_events"].fillna(0).eq(0) & ~prof["epoch_header_only"]

    days = pd.read_parquet(ARTIFACTS / "epoch_days.parquet")
    gate = coverage_gate(days).reset_index()
    prof = prof.merge(gate, on="user_id", how="left")

    esrc = pd.read_parquet(ARTIFACTS / "epoch_sources.parquet")
    prof = prof.merge(epoch_source_profile(esrc), on="user_id", how="left")
    for c in ["epoch_primary_source", "epoch_source_set", "epoch_n_sources"]:
        if c in prof:
            prof[c] = prof[c].fillna(pd.NA)
    if "epoch_multisource" in prof:
        prof["epoch_multisource"] = prof["epoch_multisource"].fillna(False)
    for c in [c for c in prof.columns if c.startswith("epoch_src_")]:
        prof[c] = prof[c].fillna(0).astype("int8")

    prof["salutation_missing"] = prof["salutation"].isna()
    prof["salutation_not_binary"] = prof["salutation"].notna() & ~prof["salutation"].isin(KEEP_SALUTATIONS)
    prof["gate_pass"] = prof["gate_pass"].fillna(False)
    prof["adequate_days_max"] = prof["adequate_days_max"].fillna(0)

    prof["eligible"] = (
        prof["has_allowed_daily"] & prof["epoch_file_exists"] & ~prof["epoch_zero_byte"] &
        ~prof["epoch_header_only"] & ~prof["epoch_unreadable"] & ~prof["epoch_no_valid"] &
        prof["gate_pass"] & ~prof["salutation_missing"] & ~prof["salutation_not_binary"])

    prof["y"] = np.where(prof["salutation"] == 20, 1,
                         np.where(prof["salutation"] == 10, 0, np.nan))

    # ---- mutually exclusive flow with fixed precedence
    def category(r):
        if not r["has_allowed_daily"]:
            return "no_allowed_source_daily"
        if not r["epoch_file_exists"]:
            return "epoch_file_missing"
        if r["epoch_zero_byte"]:
            return "epoch_zero_byte"
        if r["epoch_header_only"]:
            return "epoch_header_only"
        if r["epoch_unreadable"]:
            return "epoch_unreadable"
        if r["epoch_no_valid"]:
            return "no_valid_allowed_measurements"
        if r["salutation_missing"]:
            return "salutation_missing"
        if r["salutation_not_binary"]:
            return "salutation_not_10_20"
        if not r["gate_pass"]:
            return "below_coverage_gate"
        return "eligible"

    prof["cohort_category"] = prof.apply(category, axis=1)
    flow = prof["cohort_category"].value_counts().rename("participants").reset_index()
    flow.columns = ["category", "participants"]
    order = ["no_allowed_source_daily", "epoch_file_missing", "epoch_zero_byte",
             "epoch_header_only", "epoch_unreadable", "no_valid_allowed_measurements",
             "salutation_missing", "salutation_not_10_20", "below_coverage_gate", "eligible"]
    flow["category"] = pd.Categorical(flow["category"], categories=order, ordered=True)
    flow = flow.sort_values("category")

    prof.to_parquet(ARTIFACTS / "cohort_manifest.parquet", index=False)
    flow.to_csv(ARTIFACTS / "cohort_flow.csv", index=False)

    print("\ncohort flow:")
    print(flow.to_string(index=False))
    el = prof[prof["eligible"]]
    print(f"\neligible: {len(el)}  (sal10={int((el.salutation==10).sum())}, "
          f"sal20={int((el.salutation==20).sum())})")
    print("eligible primary epoch source:\n", el["epoch_primary_source"].value_counts(dropna=False).to_string())
    print("eligible primary daily source:\n", el["daily_primary_source"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
