#!/usr/bin/env python
"""R1b §6 — targeted raw rescan: ch3000 hour-of-day rows on ALL adequate
ch3000 days (Garmin s3 strict cohort).

Streams each cohort member's ``out/<uid>.csv`` once with the frozen upstream
validity rules (identical to ``src/sidequest/diurnal.py``; constants from
``sq_config``), keeps only ch3000 events whose (user, local_date) is an
adequate ch3000 day for that user, and aggregates per (user, date, hour):

  cache/per_day_hour_ch3000.parquet   user, date, hour, n, vsum

vsumsq is NOT emitted: the 35 R1a curve formulas consume only n and vsum
(``run_r1a.compute_r1a``); dropping it halves the emission width. Emission
covers all adequate days (not only the designated first 40) because the
random-40 sensitivity (PLAN §12.2) samples training days from the full
adequate pool; parse cost is unchanged, only the emitted aggregate grows.

Usage:  python scan_perday.py [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src" / "sidequest"))
from sq_config import (  # noqa: E402
    CORE_CHANNELS, EPOCH_DIR, EXCLUDED_SOURCES, HR_MAX, HR_MIN,
    LOCAL_DAY_MS, TZ_MS,
)

SRC = 3
CH = 3000
OUT_PARQUET = HERE / "cache" / "per_day_hour_ch3000.parquet"
SCAN_LOG = HERE / "cache" / "scan_log.json"
COLS = ["user", "date", "hour", "n", "vsum"]


def build_day_sets(with_first40: bool = True):
    """Per-user adequate ch3000 day arrays (sorted by date), days since epoch.

    Adequacy rule verbatim from run_dayscale.build_day_table. Returns
    (jobs_list, designated_count, first40_pairs) where jobs = [(uid,
    dates_ndarray)]; first40_pairs is a set of (uid, date) for the designated
    40-day windows (PLAN §4), used only for the coverage gate.
    """
    # restrict to the s3 strict cohort (3,848 users): epoch_days contains
    # all sole-source users (sources 3/6/7/9/13), but the rescan filter
    # ``source == 3`` rejects non-s3 events — scanning them would fail the
    # designated-day coverage gate (first run of this script did exactly that)
    cm = pd.read_parquet(
        REPO / "experiments" / "artifacts_sq" / "cohort_manifest_model_sources.parquet",
        columns=["user_id", "source_id"])
    s3 = set(cm[cm.source_id == SRC].user_id.astype("int64"))
    ed = pd.read_parquet(REPO / "experiments" / "artifacts_v2" / "epoch_days.parquet",
                         columns=["user", "channel", "date", "n", "cov_s",
                                  "hours"])
    ed["user"] = ed["user"].astype("int64")
    ed = ed[ed.user.isin(s3)]
    h3 = ed[ed.channel == CH]
    adequate = (h3.hours >= 8) & ((h3.cov_s >= 8 * 3600) | (h3.n >= 60))
    ad = h3[adequate].sort_values(["user", "date"])

    jobs, first40_pairs, n_adequate = [], set(), 0
    for uid, sub in ad.groupby("user", sort=True):
        dates = sub.date.to_numpy(dtype="int64")
        n_adequate += len(dates)
        jobs.append((int(uid), dates))
        if with_first40:
            first40_pairs.update((int(uid), int(d)) for d in dates[:40])
    return jobs, n_adequate, first40_pairs


def _worker(job):
    uid, dates = job
    try:
        return process_user(uid, dates), None
    except Exception as exc:  # collect, do not crash the sweep
        return None, f"user {uid}: {type(exc).__name__}: {exc}"[:300]


def process_user(uid: int, dates: np.ndarray) -> pd.DataFrame:
    """Per (date, hour) aggregates of valid ch3000 events on adequate days.

    Filters replicate src/sidequest/diurnal.process_window_user exactly, with
    the R1b restrictions: type == 3000 (ch3000-only convention) and
    (user, local_date) in the user's adequate-day set (replaces the window
    filter).
    """
    path = EPOCH_DIR / f"{uid}.csv"
    df = pd.read_csv(
        path,
        usecols=["startTimestamp", "endTimestamp", "type", "longValue",
                 "source", "timezoneOffset"],
    )
    df = df[df["source"].notna() & ~df["source"].isin(EXCLUDED_SOURCES)]
    df = df[df["type"].isin(list(CORE_CHANNELS))]
    df = df.dropna(subset=["longValue", "startTimestamp"])
    df = df[df["startTimestamp"] > 0]
    df = df[df["longValue"].between(HR_MIN, HR_MAX)]
    df = df[df["source"] == SRC]
    df = df[df["type"] == CH]
    if len(df) == 0:
        return pd.DataFrame(columns=COLS)

    start = df["startTimestamp"].to_numpy(dtype=np.int64)
    tz = df["timezoneOffset"].to_numpy(dtype=np.float64)
    tz = np.where(np.isnan(tz), 0.0, tz)
    local_ms = start.astype(np.float64) + tz * TZ_MS
    local_date = np.floor(local_ms / LOCAL_DAY_MS).astype(np.int64)
    keep = np.isin(local_date, dates)
    if not keep.any():
        return pd.DataFrame(columns=COLS)
    local_hour = (np.floor(local_ms / 3_600_000).astype(np.int64)) % 24

    sub = pd.DataFrame({
        "date": local_date[keep],
        "hour": local_hour[keep],
        "v": df["longValue"].to_numpy(dtype=np.float64)[keep],
    })
    g = sub.groupby(["date", "hour"], sort=True)
    agg = g.agg(n=("v", "size"), vsum=("v", "sum")).reset_index()
    out = pd.DataFrame({
        "user": np.full(len(agg), uid, dtype="int32"),
        "date": agg.date.to_numpy(dtype="int32"),
        "hour": agg.hour.to_numpy(dtype="int8"),
        "n": agg.n.to_numpy(dtype="int32"),
        "vsum": agg.vsum.to_numpy(dtype="float64"),
    })
    return out


def main(workers: int) -> None:
    t0 = time.perf_counter()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    jobs, n_adequate, first40_pairs = build_day_sets()
    print(f"[r1b-scan] adequate ch3000 days: {len(jobs)} users, "
          f"{n_adequate:,} user-days | designated first-40 pairs: "
          f"{len(first40_pairs):,}", flush=True)

    # load-balance: biggest files first (sidequest precedent)
    man = pd.read_parquet(REPO / "experiments" / "artifacts_sq" / "raw_file_manifest.parquet",
                          columns=["user_id", "bytes"])
    man["user_id"] = man["user_id"].astype("int64")
    man = man[man.user_id.isin({u for u, _ in jobs})]
    man = man.sort_values("bytes", ascending=False)
    by_uid = dict(jobs)
    jobs_sorted = [(int(u), by_uid[int(u)]) for u in man.user_id]
    print(f"[r1b-scan] {len(jobs_sorted)} files, "
          f"{man.bytes.sum() / 2**30:.2f} GiB, {workers} workers", flush=True)

    chunks, errors, n_rows, done = [], [], 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker, j) for j in jobs_sorted]
        for fut in as_completed(futures):
            r, err = fut.result()
            done += 1
            if err:
                errors.append(err)
            elif r is not None and len(r):
                chunks.append(r)
                n_rows += len(r)
            if done % 250 == 0:
                el = time.perf_counter() - t0
                print(f"  {done:,}/{len(jobs_sorted):,} files, "
                      f"{n_rows:,} hour-rows, errors={len(errors)}, "
                      f"{el:.0f} s elapsed", flush=True)

    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values(["user", "date", "hour"],
                          kind="stable").reset_index(drop=True)
    out.to_parquet(OUT_PARQUET, index=False)
    del chunks

    # coverage gate data (PLAN §13.4): designated days missing from the scan
    got = set(zip(out.user.astype("int64"), out.date.astype("int64")))
    missing = first40_pairs - got
    hrs = out.groupby(["user", "date"]).hour.nunique()
    designated_hrs = hrs[hrs.index.isin(first40_pairs)]
    elapsed = time.perf_counter() - t0
    stats = {
        "files": len(jobs_sorted), "failures": len(errors),
        "hour_rows": int(len(out)),
        "adequate_user_days": n_adequate,
        "designated_user_days": len(first40_pairs),
        "missing_designated_user_days": len(missing),
        "designated_distinct_hours": {
            "min": int(designated_hrs.min()),
            "p50": float(designated_hrs.median()),
            "lt8": int((designated_hrs < 8).sum())},
        "elapsed_s": round(elapsed, 1), "workers": workers,
    }
    if errors:
        open(OUT_PARQUET.parent / "scan_errors.log", "w").write(
            "\n".join(errors) + "\n")
    json.dump(stats, open(SCAN_LOG, "w"), indent=2)
    print(f"[r1b-scan] wrote {OUT_PARQUET.name}: {len(out):,} rows | "
          f"missing designated days: {len(missing)} | "
          f"designated hours/day min {stats['designated_distinct_hours']['min']} "
          f"lt8 {stats['designated_distinct_hours']['lt8']} | "
          f"{elapsed:.0f} s", flush=True)
    if missing:
        raise SystemExit(
            f"SCAN COVERAGE GATE FAILED: {len(missing)} designated "
            "(user, date) pairs missing from rescan output")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.workers)
