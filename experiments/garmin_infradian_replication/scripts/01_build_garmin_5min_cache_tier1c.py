#!/Users/bulat/micromamba/envs/datenspende/bin/python
"""
01_build_garmin_5min_cache_tier1c.py

Builds 5-min day-level caches for ALL Tier 1c single-source Garmin participants (N=4,410) into a NEW
directory. Parsing and day-adequacy rules are identical to
experiments/garmin_temporal_binning/scripts/02_build_garmin_binning_caches.py (source 3, type 3000,
HR 25-230, local time = start + 60000*tz, >=8 clock hours and (>=8 h interval coverage or >=60 readings),
>=14 adequate days). Existing caches are never read for writing; refuses to run if the output exists.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]  # experiments/garmin_infradian_replication
REPO = HERE.parents[1]
MANIFEST = REPO / "HEARTBEAT_LAB/artifacts/relaxed_garmin_manifest.csv"
OUT = HERE / "artifacts" / "cache_5min"


def union_hours(start, end):
    if len(start) == 0:
        return 0.0
    o = np.argsort(start, kind="stable")
    s, e = start[o], end[o]
    cum = np.maximum.accumulate(e)
    first = np.r_[0, np.flatnonzero(s[1:] > cum[:-1]) + 1]
    last = np.r_[first[1:] - 1, len(s) - 1]
    return float(np.maximum(0, cum[last] - s[first]).sum() / 3_600_000)


def process(task):
    cid, rel = task
    p = REPO / rel
    try:
        df = pd.read_csv(p, usecols=["startTimestamp", "endTimestamp", "timezoneOffset", "longValue", "source", "type"])
        ok = ((df.source == 3) & (df.type == 3000) & df.longValue.between(25, 230) & (df.startTimestamp > 0)
              & np.isfinite(df.startTimestamp) & np.isfinite(df.timezoneOffset) & np.isfinite(df.longValue))
        sub = df.loc[ok].copy()
        if sub.empty:
            return None
        start = sub.startTimestamp.to_numpy(np.int64)
        local = start + (sub.timezoneOffset.to_numpy(np.float64) * 60_000).astype(np.int64)
        sub["date"] = pd.to_datetime(local, unit="ms").normalize()
        msd = local % 86_400_000
        sub["bin"] = msd // 300_000
        sub["hour"] = sub.bin // 12
        end = sub.endTimestamp.to_numpy()
        sub["endTimestamp"] = np.where(np.isfinite(end) & (end >= start), end, start)
        dl, sl, hl = [], [], []
        for date, day in sub.groupby("date", sort=True):
            n = len(day)
            hrs = day.hour.nunique()
            cov = union_hours(day.startTimestamp.to_numpy(), day.endTimestamp.to_numpy())
            if hrs < 8 or (cov < 8 and n < 60):
                continue
            v = day.longValue.to_numpy()
            hl.append(day.groupby("bin").longValue.mean().reindex(range(288)).to_numpy(np.float32))
            dl.append(np.datetime64(date, "D"))
            sl.append([v.mean(), np.median(v), v.std(ddof=1) if len(v) > 1 else 0.0, v.min(), v.max(), n, cov, hrs])
        if len(dl) < 14:
            return None
        dates = np.array(dl, dtype="datetime64[D]")
        np.savez_compressed(OUT / f"customer_{cid}.npz", hr=np.stack(hl), date=dates.view("int64"),
                            stats=np.array(sl, dtype=np.float32))
        return cid
    except Exception as e:  # noqa: BLE001
        print("error", cid, e, file=sys.stderr)
        return None


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"{OUT} already populated; refusing to overwrite.")
    OUT.mkdir(parents=True)
    m = pd.read_csv(MANIFEST, dtype={"customer_id": str})
    m = m[m.tier_1c == True]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        done = [r for r in ex.map(process, list(zip(m.customer_id, m.csv_path)), chunksize=20) if r]
    print(f"{len(done)}/{len(m)} cached in {time.time() - t0:.0f}s")
    m[m.customer_id.isin(done)].drop(columns=["csv_path"]).to_csv(HERE / "artifacts" / "garmin_tier1c_cached_manifest.csv", index=False)


if __name__ == "__main__":
    main()
