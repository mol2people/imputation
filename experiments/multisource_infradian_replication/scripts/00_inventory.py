#!/usr/bin/env python3
"""Stage 1: refresh inventory and provenance for the multisource expansion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisource import provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = provenance.run_stage1(args.run_dir, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
