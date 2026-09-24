#!/usr/bin/env python3
"""Run the presentation figure script in figs_ppt."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from figs_ppt.plot_days_scaling import main

if __name__ == "__main__":
    main()
