#!/usr/bin/env python
"""TEMPORAL stage 1 — bin scan: 288 five-minute bins per day for every
frozen designated user/date pair, with the predeclared epoch-span audit.

PLAN §3:
  - 3,848 Garmin/source-3 cohort, 40 adequate ch3000 days each, pairs from
    R1b's `day40_table.parquet` (READ-ONLY).
  - filters: source==3, type==3000, HR in [25, 230], startTimestamp > 0,
    no excluded sources, dropna(longValue, startTimestamp, timezoneOffset).
  - **reject** unresolved tz (NaN) — explicit divergence from
    `src/sidequest/diurnal.py`'s substitute-UTC convention; record the
    count of dropped events.
  - 288 five-minute bins per day, event-mean HR, count, observed mask,
    assigned by event START (no interval expansion).
  - audit: endTimestamp − startTimestamp distribution per user; quantiles,
    fraction > 5 min, fraction > 10 min, max; timezoneOffset uniques
    (tz-change audit); missing designated days (coverage gate).

Usage:  python scan_bins.py [--workers 2]
Output: cache/day_bins.npz + cache/scan_log.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# thread caps BEFORE numba/touch (paranoia; scan is pure pandas/numpy)
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
BIN_MS = 300_000            # 5 min
N_BINS = 288
BINS_PER_DAY = 288
COHORT_SIZE = 3848
DAYS = 40
OUT_NPZ = HERE / "cache" / "day_bins.npz"
SCAN_LOG = HERE / "cache" / "scan_log.json"
DAY40_PARQUET = (REPO / "experiments" / "r1b_dateaware_garmin_2026-09-21"
                 / "cache" / "day40_table.parquet")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _worker(job):
    uid, dates = job
    try:
        return _process_user(uid, dates), None
    except Exception as exc:
        return None, f"user {uid}: {type(exc).__name__}: {exc}"[:300]


def _process_user(uid: int, dates: np.ndarray) -> dict:
    """Read one cohort member's raw epoch file, filter, bin-aggregate."""
    path = EPOCH_DIR / f"{uid}.csv"
    df = pd.read_csv(path,
                     usecols=["startTimestamp", "endTimestamp", "type",
                              "longValue", "source", "timezoneOffset"])
    n_raw = len(df)

    df = df[df["source"].notna() & ~df["source"].isin(EXCLUDED_SOURCES)]
    df = df[df["type"].isin([CH])]   # ch3000-only convention
    # NaN longValue/start: drop
    df = df.dropna(subset=["longValue", "startTimestamp"])
    # NaN timezoneOffset: REJECT (PLAN §3 explicit divergence from diurnal.py)
    n_nan_tz = int(df["timezoneOffset"].isna().sum())
    df = df.dropna(subset=["timezoneOffset"])
    df = df[df["startTimestamp"] > 0]
    df = df[df["longValue"].between(HR_MIN, HR_MAX)]
    df = df[df["source"] == SRC]
    df = df[df["type"] == CH]
    if len(df) == 0:
        return _empty_result(uid, dates, n_raw=n_raw,
                             n_nan_tz=n_nan_tz, n_end_nan=0)

    start = df["startTimestamp"].to_numpy(dtype=np.int64)
    end = df["endTimestamp"].to_numpy(dtype=np.float64)
    tz = df["timezoneOffset"].to_numpy(dtype=np.float64)
    long_v = df["longValue"].to_numpy(dtype=np.float64)

    # local_ms in integer ms (start + tz_minutes * 60_000)
    local_ms = start + (tz * TZ_MS).astype(np.int64)
    local_date = local_ms // LOCAL_DAY_MS
    keep = np.isin(local_date, dates)

    # span audit subset: kept events that ALSO have a valid endTimestamp
    valid_end = ~np.isnan(end)
    span_sel = keep & valid_end
    n_end_nan_kept = int((keep & ~valid_end).sum())

    if not keep.any():
        return _empty_result(uid, dates, n_raw=n_raw, n_nan_tz=n_nan_tz,
                             n_end_nan=n_end_nan_kept)

    # ---- binning on ALL kept events (assign by event start; PLAN §3) ----
    local_ms_kept = local_ms[keep]
    date_kept = local_date[keep]
    v_kept = long_v[keep]
    bin_kept = (local_ms_kept % LOCAL_DAY_MS) // BIN_MS   # 0..287
    # dates are sorted (day40 order) → searchsorted = row index 0..39
    row_idx = np.searchsorted(dates, date_kept).astype(np.int32)
    flat = row_idx * N_BINS + bin_kept          # (40*288,) = 11520
    hr_sum = np.bincount(flat, weights=v_kept, minlength=DAYS * N_BINS)
    cnt = np.bincount(flat, minlength=DAYS * N_BINS).astype(np.int32)
    mask = (cnt > 0).astype(bool)

    hr = np.zeros(DAYS * N_BINS, dtype=np.float32)
    nz = cnt > 0
    hr[nz] = (hr_sum[nz] / cnt[nz]).astype(np.float32)
    hr = hr.reshape(DAYS, N_BINS)
    cnt = cnt.reshape(DAYS, N_BINS)
    mask = mask.reshape(DAYS, N_BINS)

    # ---- span audit (kept events with valid end) ----
    if span_sel.any():
        span_ms = (end[span_sel].astype(np.int64)
                   - start[span_sel]).astype(np.int64)
        span_ms = np.clip(span_ms, 0, None)
        s_p50, s_p90, s_p99 = (int(np.percentile(span_ms, q))
                               for q in (50, 90, 99))
        s_max = int(span_ms.max())
        frac_gt_5min = float((span_ms > 300_000).mean())
        frac_gt_10min = float((span_ms > 600_000).mean())
    else:
        s_p50 = s_p90 = s_p99 = s_max = 0
        frac_gt_5min = frac_gt_10min = 0.0

    # tz-change audit (per user)
    tz_unique = int(df["timezoneOffset"].nunique())

    return {
        "uid": int(uid),
        "hr": hr, "cnt": cnt, "mask": mask,
        "span_p50_ms": s_p50, "span_p90_ms": s_p90, "span_p99_ms": s_p99,
        "span_max_ms": s_max, "frac_span_gt_5min": frac_gt_5min,
        "frac_span_gt_10min": frac_gt_10min, "tz_unique": tz_unique,
        "n_raw_rows": n_raw, "n_after_filter": int(keep.sum()),
        "n_nan_tz_dropped": n_nan_tz,
        "n_end_nan_dropped": n_end_nan_kept,
    }


def _empty_result(uid: int, dates: np.ndarray, *, n_raw: int,
                  n_nan_tz: int, n_end_nan: int) -> dict:
    return {
        "uid": int(uid),
        "hr": np.zeros((DAYS, N_BINS), dtype=np.float32),
        "cnt": np.zeros((DAYS, N_BINS), dtype=np.int32),
        "mask": np.zeros((DAYS, N_BINS), dtype=bool),
        "span_p50_ms": 0, "span_p90_ms": 0, "span_p99_ms": 0,
        "span_max_ms": 0, "frac_span_gt_5min": 0.0,
        "frac_span_gt_10min": 0.0, "tz_unique": 1,
        "n_raw_rows": n_raw, "n_after_filter": 0,
        "n_nan_tz_dropped": n_nan_tz,
        "n_end_nan_dropped": n_end_nan,
    }


def main(workers: int) -> None:
    t0 = time.perf_counter()
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)

    # frozen (user, date) pairs from R1b day40_table — read-only
    d40 = pd.read_parquet(DAY40_PARQUET, columns=["user", "date"])
    d40["user"] = d40["user"].astype("int64")
    d40["date"] = d40["date"].astype("int64")
    d40 = d40.sort_values(["user", "date"], kind="stable").reset_index(drop=True)
    n_pairs = len(d40)
    if n_pairs != COHORT_SIZE * DAYS:
        raise SystemExit(f"day40 pairs {n_pairs} != {COHORT_SIZE}*{DAYS}")
    users_sorted = np.sort(d40["user"].unique().astype("int64"))
    if len(users_sorted) != COHORT_SIZE:
        raise SystemExit(f"cohort users {len(users_sorted)} != {COHORT_SIZE}")
    per_user_dates = {int(u): g.sort_values("date").date.to_numpy(np.int64)
                      for u, g in d40.groupby("user", sort=True)}

    # load-balance: biggest files first (sidequest/R1b precedent)
    man = pd.read_parquet(REPO / "experiments" / "artifacts_sq" / "raw_file_manifest.parquet",
                          columns=["user_id", "bytes"])
    man["user_id"] = man["user_id"].astype("int64")
    man = man[man.user_id.isin(set(users_sorted))]
    man = man.sort_values("bytes", ascending=False)
    jobs = [(int(u), per_user_dates[int(u)]) for u in man.user_id]
    total_bytes = int(man["bytes"].sum())
    print(f"[scan] {len(jobs)} files, {total_bytes/2**30:.2f} GiB, "
          f"{workers} workers | {n_pairs:,} designated pairs", flush=True)

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
            if done % 250 == 0:
                el = time.perf_counter() - t0
                print(f"  {done:,}/{len(jobs):,} files | "
                      f"errors={len(errors)} | {el:.0f}s", flush=True)

    if errors:
        for e in errors[:20]:
            print(f"  ERR: {e}", flush=True)
    if len(parts) != len(jobs):
        raise SystemExit(f"SCAN GATE: {len(parts)}/{len(jobs)} users ok")

    # assemble global (n_pairs, 288) arrays in day40 sort order
    uid_to_pos = {int(u): i for i, u in enumerate(users_sorted)}
    hr = np.zeros((n_pairs, N_BINS), dtype=np.float32)
    cnt = np.zeros((n_pairs, N_BINS), dtype=np.int32)
    mask = np.zeros((n_pairs, N_BINS), dtype=bool)
    # span stats, indexed per-user
    span_p50 = np.zeros(COHORT_SIZE, dtype=np.int64)
    span_p90 = np.zeros(COHORT_SIZE, dtype=np.int64)
    span_p99 = np.zeros(COHORT_SIZE, dtype=np.int64)
    span_max = np.zeros(COHORT_SIZE, dtype=np.int64)
    frac_gt_5 = np.zeros(COHORT_SIZE, dtype=np.float32)
    frac_gt_10 = np.zeros(COHORT_SIZE, dtype=np.float32)
    tz_unique = np.zeros(COHORT_SIZE, dtype=np.int32)
    n_after_filter = np.zeros(COHORT_SIZE, dtype=np.int64)
    n_nan_tz = np.zeros(COHORT_SIZE, dtype=np.int64)
    n_end_nan = np.zeros(COHORT_SIZE, dtype=np.int64)

    for r in parts:
        u = r["uid"]
        u_pos = uid_to_pos[u]
        user_dates = per_user_dates[u]
        for di in range(DAYS):
            global_row = u_pos * DAYS + di
            hr[global_row] = r["hr"][di]
            cnt[global_row] = r["cnt"][di]
            mask[global_row] = r["mask"][di]
        span_p50[u_pos] = r["span_p50_ms"]
        span_p90[u_pos] = r["span_p90_ms"]
        span_p99[u_pos] = r["span_p99_ms"]
        span_max[u_pos] = r["span_max_ms"]
        frac_gt_5[u_pos] = r["frac_span_gt_5min"]
        frac_gt_10[u_pos] = r["frac_span_gt_10min"]
        tz_unique[u_pos] = r["tz_unique"]
        n_after_filter[u_pos] = r["n_after_filter"]
        n_nan_tz[u_pos] = r["n_nan_tz_dropped"]
        n_end_nan[u_pos] = r["n_end_nan_dropped"]

    # the global arrays are in (user-sorted × day-sorted) order, which
    # matches day40_table (sorted by user, date stable) — verify
    user_arr = d40["user"].to_numpy(np.int64)
    date_arr = d40["date"].to_numpy(np.int64)
    # write
    np.savez_compressed(
        OUT_NPZ,
        hr=hr, cnt=cnt, mask=mask,
        user=user_arr, date=date_arr,
    )
    print(f"[scan] wrote {OUT_NPZ.relative_to(HERE)} | "
          f"shape=({n_pairs},{N_BINS}) | "
          f"size {OUT_NPZ.stat().st_size/2**20:.1f} MiB", flush=True)

    # global audit
    user_tot_obs_bins = mask.sum(axis=1).astype(np.float32)  # per-day
    day_obs_bins = user_tot_obs_bins  # (n_pairs,)
    n_zero_mask_days = int((day_obs_bins == 0).sum())
    global_frac_gt_5 = float(np.average(frac_gt_5,
                                         weights=np.clip(n_after_filter, 1, None)))
    global_frac_gt_10 = float(np.average(frac_gt_10,
                                          weights=np.clip(n_after_filter, 1, None)))

    audit = {
        "files": len(jobs),
        "workers": workers,
        "failures": len(errors),
        "designated_pairs": n_pairs,
        "designated_distinct_bins_per_day": {
            "min": int(day_obs_bins.min()),
            "p50": float(np.median(day_obs_bins)),
            "p99": float(np.percentile(day_obs_bins, 99)),
            "lt8": int((day_obs_bins < 8).sum()),
            "zero_mask_days": n_zero_mask_days,
        },
        "span_audit_ms": {
            "p50_per_user_median": int(np.median(span_p50)),
            "p90_per_user_median": int(np.median(span_p90)),
            "p99_per_user_median": int(np.median(span_p99)),
            "max_per_user_max": int(span_max.max()),
            "weighted_frac_gt_5min": round(global_frac_gt_5, 4),
            "weighted_frac_gt_10min": round(global_frac_gt_10, 4),
            "users_gt_5min_50pct_of_events": int((frac_gt_5 > 0.5).sum()),
            "predeclared_rule_triggered": bool(global_frac_gt_5 > 0.5),
        },
        "tz_change_audit": {
            "users_with_gt1_tz_offsets": int((tz_unique > 1).sum()),
            "users_max_tz_unique": int(tz_unique.max()),
        },
        "nan_tz_rejected_total": int(n_nan_tz.sum()),
        "end_nan_dropped_total": int(n_end_nan.sum()),
        "raw_event_rows_total_per_user_median": None,  # not collected
        "input_sha256": {
            "day40_table.parquet": sha256_file(DAY40_PARQUET),
        },
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    json.dump(audit, open(SCAN_LOG, "w"), indent=2, default=str)
    print(f"[scan] wrote {SCAN_LOG.relative_to(HERE)}", flush=True)
    print(f"[scan] DECISION (predeclared rule): "
          f"weighted frac>5min = {global_frac_gt_5:.4f} | "
          f"threshold >0.50 → {'FRACTIONAL Profile288 sensitivity REQUIRED' if global_frac_gt_5 > 0.5 else 'no fractional sensitivity (start-bin assignment kept)'}",
          flush=True)
    if n_zero_mask_days:
        print(f"[scan] WARNING: {n_zero_mask_days} designated days have an "
              f"all-zero mask (no events after NaN-tz rejection); these will "
              f"be filled entirely by training clock-bin medians.", flush=True)
    print(f"[scan] done | wall {audit['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    main(a.workers)
