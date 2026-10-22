#!/usr/bin/env python3
"""Stage 4: fit source- and endpoint-specific RCS age models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisource.runtime import configure_single_threaded_blas

configure_single_threaded_blas()

from multisource import models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--endpoints-file", type=Path)
    parser.add_argument("--endpoints", nargs="+")
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--include-repeat-windows", action="store_true")
    args = parser.parse_args()
    summary = models.run_stage3(
        Path(args.run_dir), endpoints_file=args.endpoints_file,
        endpoint_names=args.endpoints, workers=args.workers,
        window_indices=(1, 2) if args.include_repeat_windows else (1,),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
