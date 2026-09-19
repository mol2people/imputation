"""Side-quest section 3.B: targeted raw mini-scan for window diurnal features.

Streams each model-eligible participant's ``out/<uid>.csv`` once, applying the
frozen upstream validity rules (sole retained source, core channels, HR range,
timezone handling, local-date derivation) plus the selected-window filter, and
aggregates event counts / heart-rate sums per (channel, local hour):

  artifacts_sq/window_diurnal.parquet   level='hour' rows (hourly diagnostics)
                                        + level='bucket' rows (night/morning/
                                        afternoon/evening aggregates used by the
                                        win__ block)

The plan's four local buckets are unions of clock hours, so bucket aggregates are
derived exactly by summing the hourly aggregates.

Run:  python src/sidequest/diurnal.py [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sq_config import (  # noqa: E402
    ARTIFACTS_SQ, BUCKETS, CORE_CHANNELS, EPOCH_DIR, EXCLUDED_SOURCES,
    HR_MAX, HR_MIN, LOCAL_DAY_MS, TZ_MS,
)

HOUR_COLUMNS = ["user_id", "source_id", "channel", "hour", "n", "vsum", "vsumsq"]


def process_window_user(uid: int, sole_source: int, win_start: int, win_end: int):
    """Hourly aggregates of valid sole-source core events inside the window.

    Filters replicate epoch_worker.process_user exactly, with two documented
    additions required by the plan: events must come from the participant's sole
    retained source, and the event's local date must lie inside the selected
    inclusive window [win_start, win_end] (same local-date definition).
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
    df = df[df["source"] == sole_source]
    if len(df) == 0:
        return []

    start = df["startTimestamp"].to_numpy(dtype=np.int64)
    tz = df["timezoneOffset"].to_numpy(dtype=np.float64)
    tz = np.where(np.isnan(tz), 0.0, tz)
    local_ms = start.astype(np.float64) + tz * TZ_MS
    local_date = np.floor(local_ms / LOCAL_DAY_MS).astype(np.int64)
    keep = (local_date >= win_start) & (local_date <= win_end)
    if not keep.any():
        return []
    local_hour = (np.floor(local_ms / 3_600_000).astype(np.int64)) % 24

    sub = pd.DataFrame({
        "channel": df["type"].to_numpy()[keep],
        "hour": local_hour[keep],
        "v": df["longValue"].to_numpy(dtype=np.float64)[keep],
    })
    g = sub.groupby(["channel", "hour"], sort=True)
    agg = g.agg(n=("v", "size"), vsum=("v", "sum"),
                vsumsq=("v", lambda x: float(np.square(x).sum()))).reset_index()
    rows = [(int(uid), int(sole_source), int(r.channel), int(r.hour), int(r.n),
             float(r.vsum), float(r.vsumsq))
            for r in agg.itertuples(index=False)]
    return rows


def _worker(args):
    uid, src, ws, we = args
    try:
        return process_window_user(uid, src, ws, we), None
    except Exception as exc:  # collect, do not crash the sweep
        return [], f"user {uid}: {type(exc).__name__}: {exc}"[:300]


def scan(workers: int) -> None:
    ARTIFACTS_SQ.mkdir(parents=True, exist_ok=True)
    model = pd.read_parquet(ARTIFACTS_SQ / "cohort_manifest_model_sources.parquet")
    man = pd.read_parquet(ARTIFACTS_SQ / "raw_file_manifest.parquet")
    jobs = (model[["user_id", "source_id", "window_start", "window_end"]]
            .merge(man[["user_id", "bytes"]], on="user_id", how="left")
            .sort_values("bytes", ascending=False))
    jobs = [(int(u), int(s), int(ws), int(we))
            for u, s, ws, we in zip(jobs["user_id"], jobs["source_id"],
                                    jobs["window_start"], jobs["window_end"])]
    print(f"mini-scan over {len(jobs):,} files with {workers} workers", flush=True)

    rows, errors = [], []
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker, j) for j in jobs]
        for fut in as_completed(futures):
            r, err = fut.result()
            done += 1
            if err:
                errors.append(err)
            if r:
                rows.extend(r)
            if done % 500 == 0:
                el = time.perf_counter() - t0
                print(f"  {done:,}/{len(jobs):,} files, {len(rows):,} hour-rows, "
                      f"errors={len(errors)}, {el:.0f} s elapsed", flush=True)

    hour_df = pd.DataFrame(rows, columns=HOUR_COLUMNS)
    print(f"hour rows: {len(hour_df):,}; failures: {len(errors)}", flush=True)
    if errors:
        with open(ARTIFACTS_SQ / "window_diurnal_errors.log", "w") as fh:
            fh.write("\n".join(errors) + "\n")

    # bucket rows: exact unions of clock hours
    bdf = hour_df.copy()
    bdf["bucket"] = pd.cut(bdf["hour"], bins=[-1, 5, 11, 17, 23],
                           labels=list(BUCKETS))
    bdf = (bdf.groupby(["user_id", "source_id", "channel", "bucket"], observed=True)
           .agg(n=("n", "sum"), vsum=("vsum", "sum"), vsumsq=("vsumsq", "sum"))
           .reset_index())
    bdf["hour"] = -1
    cols = ["user_id", "source_id", "channel", "level", "bucket", "hour",
            "n", "vsum", "vsumsq"]
    out = pd.concat([
        bdf.assign(level="bucket")[cols],
        hour_df.assign(level="hour", bucket="")[cols],
    ], ignore_index=True)
    out.to_parquet(ARTIFACTS_SQ / "window_diurnal.parquet", index=False)

    elapsed = time.perf_counter() - t0
    with open(ARTIFACTS_SQ / "window_diurnal_scan_log.json", "w") as fh:
        json.dump({"files": len(jobs), "failures": len(errors),
                   "hour_rows": int(len(hour_df)), "bucket_rows": int(len(bdf)),
                   "elapsed_s": round(elapsed, 1), "workers": workers}, fh, indent=2)
    print(f"window_diurnal.parquet written ({len(out):,} rows) in {elapsed:.0f} s",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    scan(a.workers)
