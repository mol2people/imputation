#!/usr/bin/env python3
"""Stage 6: render the run report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisource.runtime import configure_single_threaded_blas

configure_single_threaded_blas()

from multisource import report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    out = report.run_stage6(Path(args.run_dir))
    print(out)


if __name__ == "__main__":
    main()
