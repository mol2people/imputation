"""Side-quest section 3: feature blocks D / A / B / C for the model union.

  D  demo__age_group, demo__bmi_grp                       (2 categorical)
  A  rec__    78 numeric, taken from artifacts_v2/epoch_features.parquet
  B  win__    the same 78 families recomputed inside each selected 91-day
             window from epoch_days.parquet + window_diurnal.parquet
  C  roll_rec__ / roll_win__  4 series x 2 widths x 5 summaries x 3 channels
             = 120 columns per scope on a full calendar-day grid

Writes features_demographic.parquet, features_recording.parquet,
features_window.parquet, features_rolling.parquet, features_all.parquet and
feature_dictionary_sq.csv.  Frozen raw widths (2/78/78/120/120/398) are asserted.

Interpretation note (documented in the dictionary and report): the plan defines
when a rolling value is *defined* (day t adequate; >= 4/7 or 15/30 trailing
calendar days non-missing).  The rolling value itself is taken to be the
trailing-window mean of the non-missing daily series values, the canonical
rolling statistic; the five endpoint summaries then describe the sequence of
rolling means across endpoints.

Run:  python src/sidequest/features.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sq_config import (  # noqa: E402
    ADEQUATE_COV_H, ADEQUATE_HOURS, ADEQUATE_N, ARTIFACTS_SQ, ARTIFACTS_V2,
    BUCKETS, CORE_CHANNELS, ROLL_SUMMARIES, ROLL_SERIES, ROLL_WIDTHS,
    WIDTH_A, WIDTH_ALL, WIDTH_B, WIDTH_C_REC, WIDTH_C_WIN, WIDTH_D,
)

CH = list(CORE_CHANNELS)
CH_STATS = ["observed_days", "adequate_days", "events", "first_date", "last_date",
            "span_days", "mean_cov_h", "mean_hours", "events_per_day",
            "mean_of_daily_mean", "sd_of_daily_mean", "mean_of_daily_sd",
            "median_of_daily_median", "p25_of_daily_median", "p75_of_daily_median",
            "iqr_of_daily_median", "min_of_daily_min", "max_of_daily_max",
            "range_of_daily_extremes", "weekday_mean", "weekend_mean",
            "weekend_minus_weekday"]
OVERALL_PREDICTORS = ["overall_events", "overall_days", "overall_span_days"]
STAT_GROUPS = {
    "observed_days": "acquisition", "adequate_days": "acquisition", "events": "acquisition",
    "first_date": "acquisition", "last_date": "acquisition", "span_days": "acquisition",
    "mean_cov_h": "acquisition", "mean_hours": "acquisition", "events_per_day": "acquisition",
    "mean_of_daily_mean": "value", "sd_of_daily_mean": "variability",
    "mean_of_daily_sd": "variability", "median_of_daily_median": "value",
    "p25_of_daily_median": "value", "p75_of_daily_median": "value",
    "iqr_of_daily_median": "value", "min_of_daily_min": "value",
    "max_of_daily_max": "value", "range_of_daily_extremes": "value",
    "weekday_mean": "value", "weekend_mean": "value", "weekend_minus_weekday": "derived",
}
PREDICTOR_STATS = [s for s in CH_STATS if s not in ("first_date", "last_date")]


def a_columns() -> list[str]:
    cols = [f"ch{c}_{s}" for c in CH for s in PREDICTOR_STATS]
    cols += OVERALL_PREDICTORS + ["overall_adequate_channel_days"]
    cols += ["ch3000_minus_ch3001_mean", "ch3000_minus_ch3002_mean"]
    cols += [f"ch{c}_tod_{b}" for c in CH for b in BUCKETS]
    return cols


def load_blocks_input() -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pd.read_parquet(ARTIFACTS_SQ / "cohort_manifest_model_sources.parquet")
    return model, model.set_index("user_id")


def block_demo(model: pd.DataFrame) -> pd.DataFrame:
    d = model[["user_id", "source_id", "age_group", "bmi_grp"]].copy()
    return d.rename(columns={"age_group": "demo__age_group", "bmi_grp": "demo__bmi_grp"})


def block_recording() -> pd.DataFrame:
    f = pd.read_parquet(ARTIFACTS_V2 / "epoch_features.parquet")
    f = f.set_index("user_id")
    want = a_columns()
    base = {f"ch{c}_{s}": f"ch{c}_{s}" for c in CH for s in PREDICTOR_STATS}
    rename = dict(base)
    rename["overall_adequate_days"] = "overall_adequate_channel_days"
    missing = [c for c in want if c not in f.columns and
               not (c == "overall_adequate_channel_days" and "overall_adequate_days" in f.columns)]
    if missing:
        raise KeyError(f"epoch_features is missing expected predictor columns: {missing}")
    out = f[list(base) + ["overall_events", "overall_days", "overall_adequate_days",
                          "overall_span_days", "ch3000_minus_ch3001_mean",
                          "ch3000_minus_ch3002_mean"]
            + [f"ch{c}_tod_{b}" for c in CH for b in BUCKETS]].rename(columns=rename)
    out = out[a_columns()]
    return out.add_prefix("rec__")


def _adequate(days: pd.DataFrame) -> pd.Series:
    return ((days["hours"] >= ADEQUATE_HOURS) &
            ((days["cov_s"] >= ADEQUATE_COV_H * 3600) | (days["n"] >= ADEQUATE_N)))


def per_channel_stats(days: pd.DataFrame) -> pd.DataFrame:
    """v2 ``_per_channel`` replication on an arbitrary day-row subset.

    Index: (user, channel); columns: the 22 per-channel statistics.
    """
    days = days.copy()
    days["cov_h"] = days["cov_s"] / 3600.0
    days["adequate"] = _adequate(days)

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

    ad = days[days["adequate"]]
    val = ad.groupby(["user", "channel"], sort=True).agg(
        adequate_days=("date", "size"),
        mean_of_daily_mean=("mean", "mean"),
        sd_of_daily_mean=("mean", "std"),
        mean_of_daily_sd=("sd", "mean"),
        median_of_daily_median=("median", "median"),
        min_of_daily_min=("vmin", "min"),
        max_of_daily_max=("vmax", "max"),
    ).reset_index()
    q25 = ad.groupby(["user", "channel"])["median"].quantile(0.25).rename("p25_of_daily_median")
    q75 = ad.groupby(["user", "channel"])["median"].quantile(0.75).rename("p75_of_daily_median")
    val = val.merge(q25, on=["user", "channel"], how="left").merge(
        q75, on=["user", "channel"], how="left")
    val["iqr_of_daily_median"] = val["p75_of_daily_median"] - val["p25_of_daily_median"]
    val["range_of_daily_extremes"] = val["max_of_daily_max"] - val["min_of_daily_min"]

    ad = ad.copy()
    ad["dow"] = (ad["date"] + 3) % 7
    ad["weekend"] = ad["dow"] >= 5
    wk = ad[~ad["weekend"]].groupby(["user", "channel"])["mean"].mean().rename("weekday_mean")
    we = ad[ad["weekend"]].groupby(["user", "channel"])["mean"].mean().rename("weekend_mean")
    val = val.merge(wk, on=["user", "channel"], how="left").merge(
        we, on=["user", "channel"], how="left")
    val["weekend_minus_weekday"] = val["weekend_mean"] - val["weekday_mean"]

    return acq.merge(val, on=["user", "channel"], how="outer")


def overall_stats(days: pd.DataFrame) -> pd.DataFrame:
    days = days.copy()
    days["adequate"] = _adequate(days)
    overall = days.groupby("user").agg(
        overall_events=("n", "sum"),
        overall_days=("date", "nunique"),
        overall_adequate_days=("adequate", "sum"),
        overall_first_date=("date", "min"),
        overall_last_date=("date", "max"),
    )
    overall["overall_span_days"] = (overall["overall_last_date"] -
                                    overall["overall_first_date"] + 1)
    return overall.drop(columns=["overall_first_date", "overall_last_date"])


def widen_channels(stats: pd.DataFrame) -> pd.DataFrame:
    """(user, channel) stat table -> one row per user with ch{c}_{stat} columns."""
    wide = stats.set_index(["user", "channel"])
    out = {}
    for c in CH:
        sub = wide.xs(c, level="channel") if c in wide.index.get_level_values("channel") else None
        if sub is None or sub.empty:
            continue
        for stat in CH_STATS:
            out[f"ch{c}_{stat}"] = sub[stat]
    return pd.DataFrame(out)


def block_window(model: pd.DataFrame) -> pd.DataFrame:
    days = pd.read_parquet(ARTIFACTS_V2 / "epoch_days.parquet")
    days = days[days["user"].isin(set(model["user_id"]))].copy()
    win = model[["user_id", "window_start", "window_end"]]
    days = days.merge(win, left_on="user", right_on="user_id", how="left")
    days = days[(days["date"] >= days["window_start"]) &
                (days["date"] <= days["window_end"])].drop(columns=["user_id"])
    print(f"window day rows: {len(days):,}", flush=True)

    stats = per_channel_stats(days)
    wide = widen_channels(stats)
    overall = overall_stats(days).rename(
        columns={"overall_adequate_days": "overall_adequate_channel_days"})

    # window diurnal: event-weighted bucket means from the targeted mini-scan
    di = pd.read_parquet(ARTIFACTS_SQ / "window_diurnal.parquet")
    di = di[(di["level"] == "bucket") & di["user_id"].isin(set(model["user_id"]))]
    di = di[di["n"] > 0].copy()
    di["bmean"] = di["vsum"] / di["n"]
    piv = di.pivot_table(index="user_id", columns=["channel", "bucket"], values="bmean")
    piv.columns = [f"ch{c}_tod_{b}" for c, b in piv.columns]

    meta = pd.read_parquet(ARTIFACTS_V2 / "epoch_meta.parquet").rename(columns={"user": "user_id"})
    meta = meta.set_index("user_id")
    tz_ok = ((meta["tz_missing_events"].fillna(1) == 0) &
             (meta["valid_events"].fillna(0) > 0))
    piv = piv.reindex(sorted(set(model["user_id"])))
    piv.loc[~tz_ok.reindex(piv.index).fillna(False).to_numpy()] = np.nan

    feats = wide.join(overall, how="outer").join(piv, how="left")
    for rc in (3001, 3002):
        a, b = "ch3000_mean_of_daily_mean", f"ch{rc}_mean_of_daily_mean"
        if a in feats and b in feats:
            feats[f"ch3000_minus_ch{rc}_mean"] = feats[a] - feats[b]
    missing = [c for c in a_columns() if c not in feats.columns]
    if missing:
        raise KeyError(f"window block missing columns: {missing}")
    feats = feats[a_columns()]
    feats.index.name = "user_id"
    return feats.add_prefix("win__")


def _rolling_endpoints(v: np.ndarray, adequate: np.ndarray, w: int, min_count: int):
    """Rolling trailing-mean values and defined endpoints for one series/grid.

    Returns (t_idx, values): endpoint grid indices and trailing means, defined
    when day t is adequate and at least ``min_count`` of the trailing ``w``
    calendar days have non-missing values.  Windows truncated at the grid start
    count only the non-missing grid days they contain (same convention as the
    plan's pre-window lookback for C_win).
    """
    L = len(v)
    if L == 0:
        return np.empty(0, dtype=np.int64), np.empty(0)
    m = np.isfinite(v).astype(np.float64)
    vf = np.where(np.isfinite(v), v, 0.0)
    cs_m = np.concatenate([[0.0], np.cumsum(m)])
    cs_v = np.concatenate([[0.0], np.cumsum(vf)])
    t = np.arange(L)
    lo = np.maximum(t + 1 - w, 0)
    n = cs_m[t + 1] - cs_m[lo]
    sy = cs_v[t + 1] - cs_v[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        ybar = np.where(n > 0, sy / np.where(n > 0, n, 1.0), np.nan)
    defined = (n >= min_count) & adequate[t]
    return t[defined], ybar[defined]


def _endpoint_summaries(t_idx: np.ndarray, values: np.ndarray, first_day: int):
    if len(values) == 0:
        return {s: np.nan for s in ROLL_SUMMARIES}
    x = (t_idx + first_day).astype(np.float64)      # actual calendar-day offsets
    y = values.astype(np.float64)
    out = {"mean": float(y.mean()), "sd": float(y.std(ddof=1)) if len(y) > 1 else np.nan,
           "min": float(y.min()), "max": float(y.max())}
    if len(y) >= 2:
        xm, ym = x - x.mean(), y - y.mean()
        den = float((xm * xm).sum())
        out["slope"] = float((xm * ym).sum() / den) if den > 0 else np.nan
    else:
        out["slope"] = np.nan
    return out


def block_rolling(model: pd.DataFrame) -> pd.DataFrame:
    days = pd.read_parquet(ARTIFACTS_V2 / "epoch_days.parquet")
    days = days[days["user"].isin(set(model["user_id"]))].copy()
    win = model.set_index("user_id")[["window_start", "window_end"]]

    series_map = {"daily_mean": "mean", "daily_median": "median",
                  "daily_sd": "sd", "daily_hours": "hours"}
    records, user_ids = [], []
    groups = days.groupby(["user", "channel"], sort=False)
    n_groups = groups.ngroups
    print(f"rolling over {n_groups:,} user/channel series ...", flush=True)
    for gi, ((u, c), g) in enumerate(groups):
        g = g.sort_values("date")
        first = int(g["date"].iloc[0])
        last = int(g["date"].iloc[-1])
        L = last - first + 1
        adequate = np.zeros(L, dtype=bool)
        pos = g["date"].to_numpy(dtype=np.int64) - first
        adequate[pos] = _adequate(g).to_numpy()
        ws = we = None
        if u in win.index:
            ws, we = int(win.at[u, "window_start"]), int(win.at[u, "window_end"])
        row = {}
        for W, minc in zip(ROLL_WIDTHS, (4, 15)):
            for sname, col in series_map.items():
                v = np.full(L, np.nan)
                v[pos] = g[col].to_numpy(dtype=np.float64)
                t_idx, vals = _rolling_endpoints(v, adequate, W, minc)
                for s, val_ in _endpoint_summaries(t_idx, vals, first).items():
                    row[f"roll_rec__ch{c}_{sname}_w{W}_{s}"] = val_
                if ws is None:
                    wrec = {s: np.nan for s in ROLL_SUMMARIES}
                else:
                    dates = t_idx + first
                    inwin = (dates >= ws) & (dates <= we)
                    wrec = _endpoint_summaries(t_idx[inwin], vals[inwin], first)
                for s, val_ in wrec.items():
                    row[f"roll_win__ch{c}_{sname}_w{W}_{s}"] = val_
        records.append(row)
        user_ids.append(u)
        if (gi + 1) % 5000 == 0:
            print(f"  rolling {gi + 1:,}/{n_groups:,}", flush=True)

    wide = pd.DataFrame(records, index=pd.Index(user_ids, name="user_id"))
    # one row per user/channel so far; each roll column is populated in exactly
    # one of a user's three channel rows -> collapse with first-non-null.
    wide = wide.groupby(level=0).first()
    wide = wide.sort_index(axis=1)
    return wide


def main() -> None:
    ARTIFACTS_SQ.mkdir(parents=True, exist_ok=True)
    model, _ = load_blocks_input()
    users = sorted(model["user_id"].unique())
    print(f"model union: {len(users):,} participants", flush=True)

    demo = block_demo(model).set_index("user_id").loc[users].reset_index()
    rec = block_recording().loc[users].reset_index()
    win = block_window(model).loc[users].reset_index()
    roll = block_rolling(model).loc[users].reset_index()
    src_map = model.set_index("user_id")["source_id"]
    for frame in (demo, rec, win, roll):
        if "source_id" not in frame.columns:
            frame.insert(1, "source_id", frame["user_id"].map(src_map))

    # ---- frozen width assertions -------------------------------------------
    assert demo.shape[1] - 2 == WIDTH_D, demo.shape
    assert rec.shape[1] - 2 == WIDTH_A, rec.shape
    assert win.shape[1] - 2 == WIDTH_B, win.shape
    n_rec_cols = sum(1 for c in roll.columns if c.startswith("roll_rec__"))
    n_win_cols = sum(1 for c in roll.columns if c.startswith("roll_win__"))
    assert n_rec_cols == WIDTH_C_REC and n_win_cols == WIDTH_C_WIN, (n_rec_cols, n_win_cols)
    allf = (demo.merge(rec, on=["user_id", "source_id"])
            .merge(win, on=["user_id", "source_id"])
            .merge(roll, on=["user_id", "source_id"]))
    n_all = sum(1 for c in allf.columns if "__" in c)
    assert n_all == WIDTH_ALL, n_all

    demo.to_parquet(ARTIFACTS_SQ / "features_demographic.parquet", index=False)
    rec.to_parquet(ARTIFACTS_SQ / "features_recording.parquet", index=False)
    win.to_parquet(ARTIFACTS_SQ / "features_window.parquet", index=False)
    roll.to_parquet(ARTIFACTS_SQ / "features_rolling.parquet", index=False)
    allf.to_parquet(ARTIFACTS_SQ / "features_all.parquet", index=False)
    print(f"widths OK: D={demo.shape[1]-2} A={rec.shape[1]-2} B={win.shape[1]-2} "
          f"C_rec={n_rec_cols} C_win={n_win_cols} all={n_all}")

    write_dictionary(demo, rec, win, roll)


def write_dictionary(demo, rec, win, roll) -> None:
    rows = []
    for c in demo.columns:
        if c in ("user_id", "source_id"):
            continue
        rows.append(dict(feature=c, block="D", group="demographic", channel="", width="",
                         summary="", units="", description="linked covariate (categorical)"))
    stat_desc = {s: s.replace("_", " ") for s in STAT_GROUPS}
    for frame, block, prefix in ((rec, "A", "rec__"), (win, "B", "win__")):
        for c in frame.columns:
            if c in ("user_id", "source_id"):
                continue
            c = c.removeprefix(prefix)
            if "_tod_" in c:
                ch, bucket = c[2:6], c[7:]
                rows.append(dict(feature=c, block=block, group="diurnal", channel=ch,
                                 width="", summary=bucket, units="bpm",
                                 description=("event-weighted local time-of-day mean HR"
                                              + (" within the selected window" if block == "B"
                                                 else " over the recording (v2 table)"))))
            elif "minus" in c:
                rows.append(dict(feature=c, block=block, group="derived", channel="3000-vs",
                                 width="", summary="", units="bpm",
                                 description="resting/active channel contrast"))
            elif c.startswith("overall"):
                rows.append(dict(feature=c, block=block, group="acquisition", channel="all",
                                 width="", summary="", units="",
                                 description=c.replace("overall_", "") +
                                 (" within the selected window" if block == "B" else "")))
            else:
                ch, stat = c[2:6], c.split("_", 1)[1]
                rows.append(dict(feature=c, block=block, group=STAT_GROUPS.get(stat, "value"),
                                 channel=ch, width="", summary="", units="bpm" if
                                 STAT_GROUPS.get(stat) in ("value", "variability") else "",
                                 description=stat_desc.get(stat, stat) +
                                 (" within the selected window" if block == "B" else "")))
    for c in roll.columns:
        if c in ("user_id", "source_id"):
            continue
        scope = "C_rec" if c.startswith("roll_rec__") else "C_win"
        body = c.split("__", 1)[1]
        ch, rest = body.split("_", 1)
        ch = ch.removeprefix("ch")
        sname, w, summary = rest.rsplit("_", 2)
        w = w.lstrip("w")
        rows.append(dict(
            feature=c, block=scope, group="rolling", channel=ch, width=f"{w}d",
            summary=summary, units="bpm" if sname in ("daily_mean", "daily_median",
                                                      "daily_sd") else "hours",
            description=(f"endpoint summaries of the trailing-{w}-day mean of {sname} "
                         f"(>= {4 if w == '7' else 15}/{w} non-missing support, endpoint day "
                         f"adequate; scope {'recording span' if scope == 'C_rec' else 'selected window'})")))
    pd.DataFrame(rows).to_csv(ARTIFACTS_SQ / "feature_dictionary_sq.csv", index=False)
    print(f"feature dictionary: {len(rows)} entries")


if __name__ == "__main__":
    main()
