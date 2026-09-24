#!/usr/bin/env python
"""TEMPORAL full-history scan: 288-bin 5-min grid for EVERY adequate
user/date pair in the R1b `dayall_table.parquet` (mean 425 days/user,
min 80, p50 476, p90 539, max 979 across 3,848 users).

Design deviations from `scan_bins.py` (recorded):
  - Dates per user = full adequate history (dayall), not the first 40.
  - Workers = 4 (v2 used 2; reading the same files, but binning ~10×
    more events — added parallelism for the binning cost).
  - Output = uncompressed .npy files (hr + mask + user + date +
    days_per_user + user_ids) under cache/bins_all/, for downstream
    memmap. `cnt` is NOT stored — not a model input (v2 used it only
    for the span audit, whose verdict is already recorded).
  - No span audit, no scan_log json (v2's audit covered the cohort;
    extending past day40 doesn't change the per-day span distribution
    for the same user, and the verdict was "start-bin kept").

Usage:  python scan_bins_all.py [--workers 4]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src" / "sidequest"))
from sq_config import (  # noqa: E402
    EPOCH_DIR, EXCLUDED_SOURCES, HR_MAX, HR_MIN,
    LOCAL_DAY_MS, TZ_MS,
)

SRC = 3
CH = 3000
BIN_MS = 300_000
N_BINS = 288
COHORT_SIZE = 3848

OUT_DIR = HERE / "cache" / "bins_all"
DAYALL_PARQUET = (REPO / "experiments" / "r1b_dateaware_garmin_2026-09-21"
                  / "cache" / "dayall_table.parquet")


def _worker(job):
    uid, dates = job
    try:
        return _process_user(uid, dates), None
    except Exception as exc:
        return None, f"user {uid}: {type(exc).__name__}: {exc}"[:300]


def _process_user(uid, dates):
    """Read one cohort member's raw epoch file, filter, bin-aggregate ALL
    `dates` (variable length per user). Returns hr/mask (n_dates, 288).
    """
    n_dates = len(dates)
    path = EPOCH_DIR / f"{uid}.csv"
    df = pd.read_csv(path,
                     usecols=["startTimestamp", "endTimestamp", "type",
                              "longValue", "source", "timezoneOffset"])
    df = df[df["source"].notna() & ~df["source"].isin(EXCLUDED_SOURCES)]
    df = df[df["type"].isin([CH])]
    df = df.dropna(subset=["longValue", "startTimestamp"])
    df = df.dropna(subset=["timezoneOffset"])
    df = df[df["startTimestamp"] > 0]
    df = df[df["longValue"].between(HR_MIN, HR_MAX)]
    df = df[df["source"] == SRC]
    df = df[df["type"] == CH]
    if len(df) == 0 or n_dates == 0:
        return {"uid": int(uid),
                "hr": np.zeros((n_dates, N_BINS), dtype=np.float32),
                "mask": np.zeros((n_dates, N_BINS), dtype=bool),
                "n_after_filter": 0}

    start = df["startTimestamp"].to_numpy(dtype=np.int64)
    tz = df["timezoneOffset"].to_numpy(dtype=np.float64)
    long_v = df["longValue"].to_numpy(dtype=np.float64)
    local_ms = start + (tz * TZ_MS).astype(np.int64)
    local_date = local_ms // LOCAL_DAY_MS
    keep = np.isin(local_date, dates)
    if not keep.any():
        return {"uid": int(uid),
                "hr": np.zeros((n_dates, N_BINS), dtype=np.float32),
                "mask": np.zeros((n_dates, N_BINS), dtype=bool),
                "n_after_filter": 0}

    local_ms_k = local_ms[keep]
    date_k = local_date[keep]
    v_k = long_v[keep]
    bin_k = (local_ms_k % LOCAL_DAY_MS) // BIN_MS          # 0..287
    row_idx = np.searchsorted(dates, date_k).astype(np.int32)
    flat = row_idx * N_BINS + bin_k
    hr_sum = np.bincount(flat, weights=v_k, minlength=n_dates * N_BINS)
    cnt = np.bincount(flat, minlength=n_dates * N_BINS).astype(np.int32)
    mask = (cnt > 0).astype(bool)

    hr = np.zeros(n_dates * N_BINS, dtype=np.float32)
    nz = cnt > 0
    hr[nz] = (hr_sum[nz] / cnt[nz]).astype(np.float32)
    return {"uid": int(uid),
            "hr": hr.reshape(n_dates, N_BINS),
            "mask": mask.reshape(n_dates, N_BINS),
            "n_after_filter": int(keep.sum())}


def main(workers):
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    da = pd.read_parquet(DAYALL_PARQUET, columns=["user", "date"])
    da["user"] = da["user"].astype("int64")
    da["date"] = da["date"].astype("int64")
    da = da.sort_values(["user", "date"], kind="stable").reset_index(drop=True)
    if da.user.nunique() != COHORT_SIZE:
        raise SystemExit(f"dayall users {da.user.nunique()} != {COHORT_SIZE}")

    users_sorted = np.sort(da["user"].unique().astype("int64"))
    per_user_dates = {int(u): g.sort_values("date").date.to_numpy(np.int64)
                      for u, g in da.groupby("user", sort=True)}
    days_per_user = np.array([len(per_user_dates[int(u)]) for u in users_sorted],
                             dtype=np.int64)
    total_pairs = int(days_per_user.sum())
    starts = np.concatenate([[0], np.cumsum(days_per_user)[:-1]]).astype(np.int64)

    # load-balance: biggest files first (R1b/scan_bins precedent)
    man = pd.read_parquet(REPO / "experiments" / "artifacts_sq" / "raw_file_manifest.parquet",
                          columns=["user_id", "bytes"])
    man["user_id"] = man["user_id"].astype("int64")
    man = man[man.user_id.isin(set(users_sorted))]
    man = man.sort_values("bytes", ascending=False)
    jobs = [(int(u), per_user_dates[int(u)]) for u in man.user_id]
    total_bytes = int(man["bytes"].sum())
    print(f"[scan-all] {len(jobs)} files, {total_bytes/2**30:.2f} GiB, "
          f"{workers} workers | {total_pairs:,} pairs "
          f"(mean {days_per_user.mean():.0f}, min {days_per_user.min()}, "
          f"max {days_per_user.max()})", flush=True)

    parts, errors, done = [], [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker, j) for j in jobs]
        for fut in as_completed(futures):
            r, err = fut.result()
            done += 1
            if err:
                errors.append(err)
            elif r is not None:
                parts.append(r)
            if done % 500 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                print(f"  {done:,}/{len(jobs):,} files | "
                      f"errors={len(errors)} | {el:.0f}s", flush=True)

    if errors:
        for e in errors[:20]:
            print(f"  ERR: {e}", flush=True)
    if len(parts) != len(jobs):
        raise SystemExit(f"SCAN-ALL GATE: {len(parts)}/{len(jobs)} users ok")

    uid_to_pos = {int(u): i for i, u in enumerate(users_sorted)}
    hr = np.zeros((total_pairs, N_BINS), dtype=np.float32)
    mask = np.zeros((total_pairs, N_BINS), dtype=bool)
    n_after = np.zeros(COHORT_SIZE, dtype=np.int64)

    for r in parts:
        u = r["uid"]
        u_pos = uid_to_pos[u]
        n_d = days_per_user[u_pos]
        s = starts[u_pos]
        hr[s:s + n_d] = r["hr"]
        mask[s:s + n_d] = r["mask"]
        n_after[u_pos] = r["n_after_filter"]

    user_arr = np.repeat(users_sorted, days_per_user).astype(np.int64)
    date_arr = da["date"].to_numpy(np.int64)
    assert user_arr.shape == date_arr.shape == (total_pairs,)

    np.save(OUT_DIR / "hr_all.npy", hr)
    np.save(OUT_DIR / "mask_all.npy", mask)
    np.save(OUT_DIR / "user_all.npy", user_arr)
    np.save(OUT_DIR / "date_all.npy", date_arr)
    np.save(OUT_DIR / "days_per_user.npy", days_per_user)
    np.save(OUT_DIR / "user_ids.npy", users_sorted)
    np.save(OUT_DIR / "starts.npy", starts)

    obs_bins_per_day = mask.sum(axis=1)
    print(f"\n[scan-all] wrote {OUT_DIR.relative_to(HERE)} | "
          f"shape=({total_pairs},{N_BINS}) | "
          f"obs bins/day: min={int(obs_bins_per_day.min())}, "
          f"p50={float(np.median(obs_bins_per_day)):.0f}, "
          f"p99={float(np.percentile(obs_bins_per_day,99)):.0f} | "
          f"zero-mask days={int((obs_bins_per_day == 0).sum())}",
          flush=True)
    print(f"[scan-all] done | wall {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    main(a.workers)
