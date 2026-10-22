#!/usr/bin/env python3
"""Stage 2: build daily and channel-3000 eligibility manifests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisource.runtime import configure_single_threaded_blas

configure_single_threaded_blas()

from multisource import common, manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", type=int, default=list(common.SCOPE_SOURCES))
    parser.add_argument("--protocols", nargs="+", default=list(common.PROTOCOLS))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cache_dir = args.cache_dir or (Path(args.run_dir) / "cache_5min")
    summary = manifests.run_stage2(
        Path(args.run_dir), sources=args.sources, protocols=args.protocols,
        cache_dir=cache_dir, workers=args.workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
