"""Run the full experiment end to end, in the order required by EXPERIMENT_PLAN.md.

Usage:  python src/run_all.py [--skip-scan]

The epoch scan streams ~300 GiB and takes ~10 min; use --skip-scan to reuse
existing artifacts/epoch_*.parquet.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STAGES = [
    "epoch_scan.py",
    "build_manifest.py",
    "allocate.py",
    "features.py",
    "balance_report.py",
    "cohort_report.py",
    "model.py",
    "verify.py",
]


def main():
    stages = STAGES
    if "--skip-scan" in sys.argv:
        stages = [s for s in stages if s != "epoch_scan.py"]
    for s in stages:
        print(f"\n===== {s} =====", flush=True)
        subprocess.run([PY, os.path.join(HERE, s)], check=True)


if __name__ == "__main__":
    main()
