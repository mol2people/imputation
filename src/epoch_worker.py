"""Per-participant epoch reducer used by epoch_scan.py.

Reads one ``out/<uid>.csv`` at a time, applies source/type/validity filtering,
and returns compact per-day / per-source / per-hour aggregates.  No raw epoch
values leave this module; the full export is never concatenated in memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    EPOCH_DIR, EXCLUDED_SOURCES, CORE_CHANNELS, HR_MIN, HR_MAX,
    LOCAL_DAY_MS, TZ_MS,
)

DAY_COLUMNS = ["user", "channel", "date", "n", "cov_s", "hours", "mean", "median", "sd", "vmin", "vmax"]
SRC_COLUMNS = ["user", "channel", "source", "events", "days"]
HOUR_COLUMNS = ["user", "channel", "hour", "n", "vsum", "vsumsq"]


def _union_seconds(s: np.ndarray, e: np.ndarray) -> float:
    """Union duration of intervals [s,e] in seconds, vectorised (O(n log n))."""
    if len(s) == 0:
        return 0.0
    order = np.argsort(s, kind="stable")
    s = s[order]
    e = e[order]
    em = np.maximum.accumulate(e)
    new = np.empty(len(s), dtype=bool)
    new[0] = True
    if len(s) > 1:
        new[1:] = s[1:] > em[:-1]
    starts = np.nonzero(new)[0]
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1] = len(s) - 1
    seg_end = em[ends]
    return float(np.sum(np.maximum(0.0, seg_end - s[starts])))


def _empty(uid):
    return ([], [], [], {"user": uid, "first_date": pd.NA, "last_date": pd.NA,
                         "valid_events": 0, "n_days": 0, "tz_missing_events": 0,
                         "raw_events": 0})


def process_user(uid: int):
    path = EPOCH_DIR / f"{uid}.csv"
    try:
        df = pd.read_csv(
            path,
            usecols=["startTimestamp", "endTimestamp", "type", "longValue",
                     "source", "timezoneOffset"],
        )
    except Exception as exc:  # unreadable file
        day, src, hour, meta = _empty(uid)
        meta["unreadable"] = str(exc)[:200]
        return day, src, hour, meta

    raw_n = len(df)
    if raw_n == 0:
        return _empty(uid)

    df = df[df["source"].notna() & ~df["source"].isin(EXCLUDED_SOURCES)]
    df = df[df["type"].isin(list(CORE_CHANNELS))]
    df = df.dropna(subset=["longValue", "startTimestamp"])
    df = df[df["startTimestamp"] > 0]
    df = df[df["longValue"].between(HR_MIN, HR_MAX)]
    if len(df) == 0:
        day, src, hour, meta = _empty(uid)
        meta["raw_events"] = int(raw_n)
        return day, src, hour, meta

    start = df["startTimestamp"].to_numpy(dtype=np.int64)
    end = df["endTimestamp"].to_numpy(dtype=np.float64)
    end = np.where(np.isfinite(end) & (end >= start), end, start)
    tz = df["timezoneOffset"].to_numpy(dtype=np.float64)
    tz_missing = np.isnan(tz)
    tz = np.where(tz_missing, 0.0, tz)

    local_ms = start.astype(np.float64) + tz * TZ_MS
    local_date = np.floor(local_ms / LOCAL_DAY_MS).astype(np.int64)
    local_hour = (np.floor(local_ms / 3_600_000).astype(np.int64)) % 24

    df = df.assign(
        _s=start / 1000.0,
        _e=end / 1000.0,
        _date=local_date,
        _hour=local_hour,
        _ssq=(df["longValue"].to_numpy(dtype=np.float64) ** 2),
        _tzmiss=tz_missing.astype(np.int64),
    )

    # ---- per (channel, local day) -----------------------------------------
    g = df.groupby(["type", "_date"], sort=True)
    stats = g.agg(
        n=("longValue", "size"),
        mean=("longValue", "mean"),
        median=("longValue", "median"),
        sd=("longValue", "std"),
        vmin=("longValue", "min"),
        vmax=("longValue", "max"),
        hours=("_hour", "nunique"),
    ).reset_index()
    cov = g.apply(lambda x: _union_seconds(x["_s"].to_numpy(), x["_e"].to_numpy()),
                  include_groups=False).rename("cov_s").reset_index()
    stats = stats.merge(cov, on=["type", "_date"], how="left")
    stats = stats.rename(columns={"type": "channel", "_date": "date"})
    stats.insert(0, "user", uid)
    stats["sd"] = stats["sd"].fillna(0.0)
    day_rows = stats[DAY_COLUMNS]

    # ---- per (channel, source) ---------------------------------------------
    ps = df.groupby(["type", "source"], sort=True).agg(
        events=("longValue", "size"), days=("_date", "nunique")).reset_index()
    ps = ps.rename(columns={"type": "channel"})
    ps.insert(0, "user", uid)
    src_rows = ps[SRC_COLUMNS]

    # ---- per (channel, local hour) ------------------------------------------
    ph = df.groupby(["type", "_hour"], sort=True).agg(
        n=("longValue", "size"), vsum=("longValue", "sum"), vsumsq=("_ssq", "sum")).reset_index()
    ph = ph.rename(columns={"type": "channel", "_hour": "hour"})
    ph.insert(0, "user", uid)
    hour_rows = ph[HOUR_COLUMNS]

    meta = {
        "user": uid,
        "first_date": int(local_date.min()),
        "last_date": int(local_date.max()),
        "valid_events": int(len(df)),
        "n_days": int(np.unique(local_date).size),
        "tz_missing_events": int(tz_missing.sum()),
        "raw_events": int(raw_n),
    }
    return day_rows, src_rows, hour_rows, meta
