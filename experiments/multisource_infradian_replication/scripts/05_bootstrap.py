#!/usr/bin/env python3
"""Stage 5: stratified participant bootstrap."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisource.runtime import configure_single_threaded_blas

configure_single_threaded_blas()

from multisource import bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoints-file", type=Path)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--endpoints", nargs="+")
    parser.add_argument("--include-repeat-windows", action="store_true")
    args = parser.parse_args()
    summary = bootstrap.run_stage4(
        Path(args.run_dir), endpoints_file=args.endpoints_file,
        n_boot=args.n_boot, seed=args.seed, workers=args.workers,
        endpoint_names=args.endpoints,
        window_indices=(1, 2) if args.include_repeat_windows else (1,),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
