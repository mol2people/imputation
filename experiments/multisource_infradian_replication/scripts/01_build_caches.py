#!/usr/bin/env python3
"""Stage 1b: build source-tagged channel-3000 caches.

The command is intentionally bounded by default: pass ``--scan`` to refresh the
raw type-3000 index, and ``--limit`` for a smoke run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisource.runtime import configure_single_threaded_blas

configure_single_threaded_blas()

import pandas as pd

from multisource import channel3000, common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", type=int, default=list(common.SCOPE_SOURCES))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scan", action="store_true", help="Refresh raw type-3000 index")
    parser.add_argument(
        "--scan-mode", choices=("all", "candidates"), default="all",
        help="all: one pass over out/*.csv; candidates: daily+epoch metadata only",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cache_dir = args.cache_dir or (run_dir / "cache_5min")
    daily_clean = pd.read_parquet(run_dir / "daily_clean.parquet")
    candidates = channel3000.candidate_users(daily_clean, args.sources)

    index_path = run_dir / "raw_index.parquet"
    if args.scan or not index_path.exists():
        index = channel3000.build_raw_index(
            candidates, raw_dir=common.RAW_OUT_DIR, sources=args.sources,
            workers=args.workers, limit=args.limit,
            exhaustive=(args.scan_mode == "all"),
        )
        index.to_parquet(index_path, index=False)
    else:
        index = pd.read_parquet(index_path)

    build_cols = ["user_id", "source"] + [
        c for c in ("dense_rows", "raw_path") if c in index.columns
    ]
    cache_index = channel3000.build_caches(
        index[build_cols],
        cache_dir=cache_dir,
        raw_dir=common.RAW_OUT_DIR,
        workers=args.workers,
        limit=args.limit,
    )
    cache_index.to_parquet(run_dir / "cache_index.parquet", index=False)
    summary = {
        "candidates": int(len(candidates)),
        "raw_index_rows": int(len(index)),
        "caches_built": int(len(cache_index)),
        "cache_dir": str(cache_dir),
        "by_source": cache_index.groupby("source").size().to_dict() if not cache_index.empty else {},
    }
    (run_dir / "stage1b_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
