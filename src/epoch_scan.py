"""Full parallel scan of the epoch export.

Streams every candidate ``out/<uid>.csv`` once, writing compact aggregates:
  artifacts/epoch_days.parquet    one row per (user, channel, local day)
  artifacts/epoch_sources.parquet one row per (user, channel, source)
  artifacts/epoch_hours.parquet   one row per (user, channel, local hour)
  artifacts/epoch_meta.parquet    one row per user (span, counts, flags)

Run:  python src/epoch_scan.py [--workers 8]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ARTIFACTS, DAILY_CSV, EPOCH_DIR, EXCLUDED_SOURCES, DAILY_TYPES  # noqa: E402
import epoch_worker as ew  # noqa: E402


def candidate_users() -> list[int]:
    daily = pd.read_csv(DAILY_CSV, usecols=["user_id", "type", "source"])
    daily = daily[daily["type"].isin(DAILY_TYPES)]
    daily = daily[~daily["source"].isin(EXCLUDED_SOURCES)]
    users = sorted(daily["user_id"].unique().tolist())
    present = [u for u in users if (EPOCH_DIR / f"{u}.csv").exists()
               and (EPOCH_DIR / f"{u}.csv").stat().st_size > 0]
    return present


class ShardWriter:
    """Incremental parquet writer that appends arrow tables."""

    def __init__(self, path, schema):
        self.path = path
        self.schema = schema
        self.writer = None
        self.rows = 0

    def write(self, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(table)
        self.rows += len(df)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="scan only first N users (debug)")
    args = ap.parse_args()

    users = candidate_users()
    if args.limit:
        users = users[: args.limit]
    print(f"candidate users with nonempty epoch file: {len(users)}", flush=True)

    day_w = ShardWriter(ARTIFACTS / "epoch_days.parquet",
                        pa.schema([("user", pa.int64()), ("channel", pa.int64()),
                                   ("date", pa.int64()), ("n", pa.int64()),
                                   ("cov_s", pa.float64()), ("hours", pa.int64()),
                                   ("mean", pa.float64()), ("median", pa.float64()),
                                   ("sd", pa.float64()), ("vmin", pa.float64()),
                                   ("vmax", pa.float64())]))
    src_w = ShardWriter(ARTIFACTS / "epoch_sources.parquet",
                        pa.schema([("user", pa.int64()), ("channel", pa.int64()),
                                   ("source", pa.int64()), ("events", pa.int64()),
                                   ("days", pa.int64())]))
    hour_w = ShardWriter(ARTIFACTS / "epoch_hours.parquet",
                         pa.schema([("user", pa.int64()), ("channel", pa.int64()),
                                    ("hour", pa.int64()), ("n", pa.int64()),
                                    ("vsum", pa.float64()), ("vsumsq", pa.float64())]))
    metas: list[dict] = []

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ew.process_user, u): u for u in users}
        for fut in as_completed(futs):
            uid = futs[fut]
            try:
                day, src, hour, meta = fut.result()
            except Exception as exc:  # noqa: BLE001
                meta = {"user": uid, "first_date": pd.NA, "last_date": pd.NA,
                        "valid_events": 0, "n_days": 0, "tz_missing_events": 0,
                        "raw_events": 0, "unreadable": str(exc)[:200]}
                day, src, hour = None, None, None
            day_w.write(day)
            src_w.write(src)
            hour_w.write(hour)
            metas.append(meta)
            done += 1
            if done % 1000 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(users)}  {rate:.0f} files/s  "
                      f"days={day_w.rows:,}  eta={(len(users)-done)/rate/60:.1f} min",
                      flush=True)

    day_w.close(); src_w.close(); hour_w.close()
    pd.DataFrame(metas).to_parquet(ARTIFACTS / "epoch_meta.parquet", index=False)
    print(f"done in {(time.time()-t0)/60:.1f} min; days={day_w.rows:,} "
          f"sources={src_w.rows:,} hours={hour_w.rows:,}", flush=True)


if __name__ == "__main__":
    main()
