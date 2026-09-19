"""Build the participant feature table from epoch day/hour aggregates.

One row per eligible participant.  Features are grouped into:
  * value summaries of daily summaries (day-weighted, equal weight per day)
  * between/within-day variability
  * weekday/weekend contrast
  * local time-of-day profile (only where the timezone is fully usable)
  * acquisition/coverage summaries
  * epoch channel/source availability (kept distinct from measured values)

Writes artifacts/epoch_features.parquet and artifacts/feature_dictionary.csv.
Run:  python src/features.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    ARTIFACTS, CORE_CHANNELS, ADEQUATE_HOURS, ADEQUATE_COV_H, ADEQUATE_N,
    EXTRA_CATEGORICAL, INCLUDE_SOURCE_INDICATORS,
)

CH = list(CORE_CHANNELS)
BUCKETS = {"night": (0, 5), "morning": (6, 11), "afternoon": (12, 17), "evening": (18, 23)}


def _per_channel(days: pd.DataFrame) -> pd.DataFrame:
    """Value / variability / acquisition summaries per (user, channel).

    Acquisition counts use every observed day.  Core longitudinal value summaries
    use only adequately observed days, as required by the frozen coverage rules.
    """
    days = days.copy()
    days["cov_h"] = days["cov_s"] / 3600.0

    acq = days.groupby(["user", "channel"], sort=True).agg(
        observed_days=("date", "size"),
        events=("n", "sum"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        mean_cov_h=("cov_h", "mean"),
        mean_hours=("hours", "mean"),
    ).reset_index()
    acq["span_days"] = acq["last_date"] - acq["first_date"] + 1
    acq["events_per_day"] = acq["events"] / acq["observed_days"]

    ad = days[days["adequate"]].copy()
    val = ad.groupby(["user", "channel"], sort=True).agg(
        adequate_days=("date", "size"),
        mean_of_daily_mean=("mean", "mean"),
        sd_of_daily_mean=("mean", "std"),
        mean_of_daily_sd=("sd", "mean"),
        median_of_daily_median=("median", "median"),
        min_of_daily_min=("vmin", "min"),
        max_of_daily_max=("vmax", "max"),
        p25_of_daily_median=("median", lambda x: x.quantile(0.25)),
        p75_of_daily_median=("median", lambda x: x.quantile(0.75)),
    ).reset_index()
    val["iqr_of_daily_median"] = val["p75_of_daily_median"] - val["p25_of_daily_median"]
    val["range_of_daily_extremes"] = val["max_of_daily_max"] - val["min_of_daily_min"]

    # weekday / weekend contrast of daily means, on adequate days only
    ad["dow"] = (ad["date"] + 3) % 7          # 1970-01-01 = Thursday -> Monday=0
    ad["weekend"] = ad["dow"] >= 5
    wk = (ad[~ad["weekend"]].groupby(["user", "channel"])["mean"].mean()
          .rename("weekday_mean"))
    we = (ad[ad["weekend"]].groupby(["user", "channel"])["mean"].mean()
          .rename("weekend_mean"))
    val = val.merge(wk, on=["user", "channel"], how="left").merge(we, on=["user", "channel"], how="left")
    val["weekend_minus_weekday"] = val["weekend_mean"] - val["weekday_mean"]

    base = acq.merge(val, on=["user", "channel"], how="outer")

    stats = {
        "observed_days": "acquisition", "adequate_days": "acquisition", "events": "acquisition",
        "first_date": "acquisition", "last_date": "acquisition", "span_days": "acquisition",
        "mean_cov_h": "acquisition", "mean_hours": "acquisition", "events_per_day": "acquisition",
        "mean_of_daily_mean": "value", "sd_of_daily_mean": "variability",
        "mean_of_daily_sd": "variability", "median_of_daily_median": "value",
        "p25_of_daily_median": "value", "p75_of_daily_median": "value",
        "iqr_of_daily_median": "value", "min_of_daily_min": "value",
        "max_of_daily_max": "value", "range_of_daily_extremes": "value",
        "weekday_mean": "value", "weekend_mean": "value",
        "weekend_minus_weekday": "derived",
    }
    # reshape channel -> column suffix
    wide = base.set_index(["user", "channel"])
    out = {}
    for c in CH:
        sub = wide.xs(c, level="channel") if c in wide.index.get_level_values("channel") else None
        if sub is None or sub.empty:
            continue
        for stat in stats:
            out[f"ch{c}_{stat}"] = sub[stat]
    feats = pd.DataFrame(out)
    return feats, stats


def _diurnal(hours: pd.DataFrame, tz_ok: pd.Series) -> pd.DataFrame:
    h = hours.copy()
    h["bucket"] = pd.cut(h["hour"], bins=[-1, 5, 11, 17, 23],
                         labels=list(BUCKETS))
    agg = h.groupby(["user", "channel", "bucket"], observed=True).agg(
        n=("n", "sum"), vsum=("vsum", "sum")).reset_index()
    agg["bmean"] = agg["vsum"] / agg["n"]
    piv = agg.pivot_table(index="user", columns=["channel", "bucket"], values="bmean")
    piv.columns = [f"ch{c}_tod_{b}" for c, b in piv.columns]
    piv = piv.reindex(index=tz_ok.index)
    piv.loc[~tz_ok] = np.nan              # local time not established
    return piv


def main():
    m = pd.read_parquet(ARTIFACTS / "cohort_manifest.parquet")
    el = m[m["eligible"]].copy()
    users = el["user_id"].to_numpy()
    print(f"eligible participants: {len(users)}")

    days = pd.read_parquet(ARTIFACTS / "epoch_days.parquet")
    days = days[days["user"].isin(set(users))].copy()
    days["adequate"] = ((days["hours"] >= ADEQUATE_HOURS) &
                        ((days["cov_s"] >= ADEQUATE_COV_H * 3600) | (days["n"] >= ADEQUATE_N)))
    print(f"day rows for eligible: {len(days):,}")

    feats, stats = _per_channel(days)

    # overall (across channels) acquisition
    allg = days.groupby("user")
    overall = allg.agg(overall_events=("n", "sum"), overall_days=("date", "nunique"),
                       overall_adequate_days=("adequate", "sum"),
                       overall_first_date=("date", "min"),
                       overall_last_date=("date", "max"))
    overall["overall_span_days"] = overall["overall_last_date"] - overall["overall_first_date"] + 1
    feats = feats.join(overall, how="outer")

    meta = pd.read_parquet(ARTIFACTS / "epoch_meta.parquet").rename(columns={"user": "user_id"})
    meta = meta.set_index("user_id").reindex(feats.index)
    tz_ok = (meta["tz_missing_events"].fillna(1) == 0) & (meta["valid_events"].fillna(0) > 0)
    hours = pd.read_parquet(ARTIFACTS / "epoch_hours.parquet")
    hours = hours[hours["user"].isin(set(users))]
    diurnal = _diurnal(hours, tz_ok.reindex(feats.index).fillna(False))
    feats = feats.join(diurnal, how="left")
    feats["tz_fully_usable"] = tz_ok.reindex(feats.index).fillna(False).astype(int)

    # resting vs active contrasts
    for rc in [3001, 3002]:
        a, b = f"ch3000_mean_of_daily_mean", f"ch{rc}_mean_of_daily_mean"
        if a in feats and b in feats:
            feats[f"ch3000_minus_ch{rc}_mean"] = feats[a] - feats[b]

    # ---- categorical availability (kept distinct from measured values) -----
    keep_cols = ["epoch_primary_source", "epoch_multisource"] + EXTRA_CATEGORICAL
    if INCLUDE_SOURCE_INDICATORS:
        keep_cols += [c for c in el.columns if c.startswith("epoch_src_")]
    avail = el.set_index("user_id")[keep_cols].copy()
    if "epoch_primary_source" in avail:
        avail["epoch_primary_source"] = avail["epoch_primary_source"].astype("Int64")
    feats = feats.join(avail, how="left")

    feats = feats.reindex(users).reset_index().rename(columns={"index": "user_id"})
    feats = feats.rename(columns={"user": "user_id"})
    if "user_id" not in feats.columns:
        feats.insert(0, "user_id", users)
    feats.to_parquet(ARTIFACTS / "epoch_features.parquet", index=False)
    print(f"feature table: {feats.shape[0]} rows x {feats.shape[1]-1} features")
    n_meas = sum(1 for c in feats.columns if c.startswith("ch") or c.startswith("overall"))
    print(f"  measured/acquisition columns: {n_meas}; tz fully usable: {int(feats.tz_fully_usable.sum())}")

    # ---- feature dictionary ------------------------------------------------
    rows = []
    for col in feats.columns:
        if col in ("user_id", "tz_fully_usable"):
            continue
        if col == "epoch_primary_source":
            rows.append(dict(feature=col, group="availability", channel="", units="",
                             description="primary epoch source (categorical)"))
        elif col == "epoch_multisource":
            rows.append(dict(feature=col, group="availability", channel="", units="",
                             description="more than one epoch source (categorical)"))
        elif col.startswith("epoch_src_"):
            rows.append(dict(feature=col, group="availability", channel="", units="",
                             description="epoch source membership indicator (v2)"))
        elif col in EXTRA_CATEGORICAL:
            rows.append(dict(feature=col, group="demographic", channel="", units="",
                             description="linked covariate used as predictor (v2)"))
        elif col.startswith("ch") and "_tod_" in col:
            ch = col[2:6] if col.startswith("ch300") else col[2:7]
            rows.append(dict(feature=col, group="diurnal", channel=ch,
                             units="bpm", description="local time-of-day mean over all valid "
                             "local-time events (missing when tz unusable)"))
        elif col.startswith("ch"):
            ch = col[2:6]
            stat = col.split("_", 1)[1]
            grp = stats.get(stat, "value") if stat in stats else (
                "derived" if "minus" in stat else "value")
            rows.append(dict(feature=col, group=grp, channel=ch, units="bpm" if grp in ("value", "variability") else "",
                             description=stat))
        elif col.startswith("overall"):
            rows.append(dict(feature=col, group="acquisition", channel="all", units="",
                             description=col.replace("overall_", "")))
    pd.DataFrame(rows).to_csv(ARTIFACTS / "feature_dictionary.csv", index=False)


if __name__ == "__main__":
    main()
