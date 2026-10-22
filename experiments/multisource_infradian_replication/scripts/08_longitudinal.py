#!/usr/bin/env python3
"""Stage 8: paired repeat-window (longitudinal) summaries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisource.runtime import configure_single_threaded_blas

configure_single_threaded_blas()

from multisource import longitudinal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    summary = longitudinal.run_stage(
        args.run_dir, n_boot=args.n_boot, seed=args.seed
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
