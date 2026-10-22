#!/usr/bin/env python3
"""Normalize the supplier daily table (types 65/66) to the endpoint CLI schema.

Input : data/15Sep_2230_PG_type65-66.csv  (user_id, date, type, value, source, timezone_offset)
Output: supplier_daily_normalized.parquet   (user_id, source, endpoint, date, value)

Date convention: epoch-day integers (days since 1970-01-01), matching the
experiment's selected-dates artifact.

Timezone handling: the three dominant offsets {0, 60, 120} (UTC, CET, CEST)
cover 99.4% of non-null rows. For these offsets the timezone conversion
(date + offset applied to midnight) does not change the calendar date:
midnight + 0/1/2 hours stays on the same day. Rows with null
timezone_offset are treated as Berlin time (CET/CEST) per user
instruction and kept. Rows with other offsets (extreme or rare) are
excluded per user instruction.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]  # /Users/bulat/Documents/datenspende_IMPUTE_sex
INPUT_CSV = REPO_ROOT / "data" / "15Sep_2230_PG_type65-66.csv"
OUTPUT_PARQUET = HERE / "supplier_daily_normalized.parquet"

ENDPOINT_NAMES = {65: "RHR", 66: "HR_during_sleep"}
ALLOWED_OFFSETS = {0.0, 60.0, 120.0}


def main() -> None:
    print(f"Reading {INPUT_CSV} ...")
    chunks = []
    for chunk in pd.read_csv(INPUT_CSV, chunksize=500_000):
        keep = chunk["timezone_offset"].isin(ALLOWED_OFFSETS) | chunk["timezone_offset"].isna()
        filtered = chunk.loc[keep].copy()
        if not filtered.empty:
            chunks.append(filtered)
    df = pd.concat(chunks, ignore_index=True)
    print(f"Rows after timezone filter ({{0, 60, 120}} + null): {len(df)}")

    df["endpoint"] = df["type"].map(ENDPOINT_NAMES)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    # Convert to epoch-day integers (days since 1970-01-01)
    df["date"] = (df["date"] - pd.Timestamp("1970-01-01")).dt.days

    out = df[["user_id", "source", "endpoint", "date", "value"]].copy()
    out.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Wrote {OUTPUT_PARQUET} ({len(out)} rows)")


if __name__ == "__main__":
    main()
