#!/usr/bin/env python3
"""Stage 5b: matched endpoint comparisons with paired safeguards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisource.runtime import configure_single_threaded_blas

configure_single_threaded_blas()

from multisource import common, matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--sources", nargs="+", type=int)
    parser.add_argument("--endpoints", nargs="+", default=["RHR", "HR_during_sleep"])
    parser.add_argument("--protocols", nargs="+", default=list(common.PROTOCOLS))
    parser.add_argument("--null-replicates", type=int, default=common.B_NULL_REPLICATES)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cache_dir = args.cache_dir or (Path(args.run_dir) / "cache_5min")
    summary = matched.run_stage5(
        Path(args.run_dir), Path(cache_dir), sources=args.sources,
        endpoints_to_match=args.endpoints, protocols=args.protocols,
        null_replicates=args.null_replicates, n_boot=args.n_boot,
        seed=args.seed, workers=args.workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
