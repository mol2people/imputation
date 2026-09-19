"""Cohort coverage, exclusion inventory and epoch channel/provenance audit.

Outputs:
  artifacts/epoch_channel_audit.csv
  artifacts/coverage_by_channel.csv
  artifacts/exclusion_by_group.csv
  artifacts/cohort_coverage_report.md
Run:  python src/cohort_report.py
"""
from __future__ import annotations

import glob
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    ARTIFACTS, EPOCH_DIR, DAILY_CSV, SALUTATION_CSV, DAILY_TYPES, EXCLUDED_SOURCES,
    CORE_CHANNELS, ADEQUATE_HOURS, ADEQUATE_COV_H, ADEQUATE_N, MIN_ADEQUATE_DAYS,
    HR_MIN, HR_MAX, EPOCH_DIR as _ED,
)

PROV_SEED = 20260918
PROV_N = 240


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(out)


def adequate(days):
    return ((days["hours"] >= ADEQUATE_HOURS) &
            ((days["cov_s"] >= ADEQUATE_COV_H * 3600) | (days["n"] >= ADEQUATE_N)))


def channel_audit():
    """Sample files and document per (type, source) representation and units."""
    files = [f for f in glob.glob(str(EPOCH_DIR / "*.csv")) if os.path.getsize(f) > 1024]
    rng = random.Random(PROV_SEED)
    sample = rng.sample(files, min(PROV_N, len(files)))
    recs = []
    ts_min, ts_max = np.inf, -np.inf
    tz_vals = []
    for f in sample:
        df = pd.read_csv(f)
        if df.empty:
            continue
        ts_min = min(ts_min, df["startTimestamp"].min())
        ts_max = max(ts_max, df["startTimestamp"].max())
        df = df[df["source"].notna() & ~df["source"].isin(EXCLUDED_SOURCES)]
        df = df[df["type"].isin(CORE_CHANNELS)]
        df = df.dropna(subset=["startTimestamp"])
        df = df[df["startTimestamp"] > 0]
        if df.empty:
            continue
        tz_vals.extend(df["timezoneOffset"].dropna().unique().tolist()[:3])
        dur = (df["endTimestamp"] - df["startTimestamp"]) / 1000.0
        point = (dur.fillna(0) <= 0)
        df = df.assign(_point=point, _dur=dur, _date=(df["startTimestamp"] // 86_400_000))
        for (t, s), g in df.groupby(["type", "source"]):
            recs.append(dict(channel=int(t), source=int(s), events=int(len(g)),
                             zero_duration_frac=float(g["_point"].mean()),
                             median_duration_s=float(g.loc[~g["_point"], "_dur"].median())
                             if (~g["_point"]).any() else 0.0,
                             median_events_per_day=float(g.groupby("_date").size().median())))
    rec = pd.DataFrame(recs).groupby(["channel", "source"]).agg(
        files=("events", "size"), events=("events", "sum"),
        zero_duration_frac=("zero_duration_frac", "mean"),
        median_duration_s=("median_duration_s", "median"),
        median_events_per_day=("median_events_per_day", "median")).reset_index()
    rec["representation"] = np.where(rec["zero_duration_frac"] > 0.5, "point-event", "interval")
    rec.to_csv(ARTIFACTS / "epoch_channel_audit.csv", index=False)
    units = {
        "sampled_files": len(sample),
        "startTimestamp_min": int(ts_min), "startTimestamp_max": int(ts_max),
        "startTimestamp_span_days": float((ts_max - ts_min) / 86_400_000),
        "timestamp_encoding": "epoch milliseconds (16-digit, /1000 -> seconds; "
                              "plausible 2020-2024 range)",
        "timezoneOffset_min": float(np.min(tz_vals)), "timezoneOffset_max": float(np.max(tz_vals)),
        "timezoneOffset_units": "minutes to add to UTC to obtain local time",
    }
    return rec, units


def main():
    man = pd.read_parquet(ARTIFACTS / "cohort_manifest.parquet")
    el = man[man["eligible"]]
    days = pd.read_parquet(ARTIFACTS / "epoch_days.parquet")
    days["adequate"] = adequate(days)

    # ---- coverage by channel ---------------------------------------------
    rows = []
    for c in CORE_CHANNELS:
        d = days[days["channel"] == c]
        ad = d[d["adequate"]].groupby("user").size()
        rows.append(dict(channel=c, name=CORE_CHANNELS[c],
                         participants_any_day=int(d["user"].nunique()),
                         participants_ge14_adequate=int((ad >= MIN_ADEQUATE_DAYS).sum()),
                         median_adequate_days=float(ad.median()) if len(ad) else 0.0))
    cov = pd.DataFrame(rows)
    cov.to_csv(ARTIFACTS / "coverage_by_channel.csv", index=False)
    gate_users = set(el["user_id"])
    n_any = len(set(days["user"]) & set(man["user_id"]))
    n_gate = len(gate_users)
    print(cov.to_string(index=False))
    print(f"participants with any valid day: {n_any}; passing gate: {n_gate}")

    # ---- exclusion by group ----------------------------------------------
    sal = pd.read_csv(SALUTATION_CSV, usecols=["user_id", "salutation"])
    man2 = man.merge(sal, on="user_id", how="left", suffixes=("", "_sal"))
    groups = []
    for cat in ["eligible", "below_coverage_gate", "no_valid_allowed_measurements"]:
        sub = man2[man2["cohort_category"] == cat]
        for var in ["salutation", "daily_primary_source", "age_group", "bmi_grp"]:
            vc = sub[var].value_counts(dropna=False)
            for k, v in vc.items():
                groups.append(dict(category=cat, variable=var, value=str(k), participants=int(v)))
    excl = pd.DataFrame(groups)
    excl.to_csv(ARTIFACTS / "exclusion_by_group.csv", index=False)

    # ---- channel / source inventory for eligible --------------------------
    esrc = pd.read_parquet(ARTIFACTS / "epoch_sources.parquet")
    esrc = esrc[esrc["user"].isin(gate_users)]
    inv = esrc.groupby(["channel", "source"]).agg(
        participants=("user", "nunique"), events=("events", "sum"),
        days=("days", "sum")).reset_index()

    # ---- provenance audit --------------------------------------------------
    rec, units = channel_audit()

    lines = ["# Cohort coverage, exclusions and epoch channel provenance", "",
             "## Discipline", "",
             "- Source exclusions 38/46/48 applied at the record level before any count, coverage "
             "measurement or feature.",
             f"- Valid HR measurement: {HR_MIN:.0f} <= longValue <= {HR_MAX:.0f} bpm; startTimestamp > 0; "
             "endTimestamp < startTimestamp or missing treated as a point event.",
             f"- Adequate day (per channel): >= {ADEQUATE_HOURS} distinct local clock hours AND "
             f"(union coverage >= {ADEQUATE_COV_H} h OR >= {ADEQUATE_N} measurements). Disjunction covers "
             "interval and point-event device representations without conflating them.",
             f"- Primary gate: >= {MIN_ADEQUATE_DAYS} adequate days in at least one core channel; days are "
             "never pooled across channels.", "", "## Coverage by core channel", "",
             _md(cov), "",
             "## Epoch channel representations (sampled audit)", "",
             _md(rec.round(3)), "",
             "## Timestamp / timezone units", ""]
    lines += [f"- {k}: {v}" for k, v in units.items()]
    lines += ["", "## Channel / source inventory among gate-passing participants", "",
              _md(inv), "",
              "## Provenance caveats", "",
              "- Types 3000/3001/3002 map to HeartRate / HeartRateResting / HeartRateRestingHourly in "
              "`mapping/epoch_value_types.csv`. The export establishes codes, not physiological "
              "interpretation or vendor derivation.",
              "- Apple (source 6) emits point-event HR samples; Garmin (source 3) emits 1-minute interval "
              "buckets. Per-source semantics are not interchangeable; source availability is therefore kept "
              "as explicit predictors and as balance variables.",
              "- Whether any processed channel (3001/3002) uses a user-entered sex/salutation in its vendor "
              "calculation is unverified from these exports. This limits causal interpretation: the "
              "experiment measures overall predictability of recorded salutation, not an isolated "
              "physiological contribution.", ""]
    with open(ARTIFACTS / "cohort_coverage_report.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote cohort_coverage_report.md")


if __name__ == "__main__":
    main()
