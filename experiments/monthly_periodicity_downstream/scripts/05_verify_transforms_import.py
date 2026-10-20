"""
05_verify_transforms_import.py
Verify transforms import from experiments/apple_fill_and_mask_by_recording/scripts/transforms.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.apple_fill_and_mask_by_recording.scripts.transforms import (
    fit_fill_templates,
    interpolate_short_gaps,
)
import numpy as np

def main() -> None:
    print("Testing transforms import...")
    hr = np.array([[70.0, np.nan, 72.0, np.nan, np.nan, 75.0] + [70.0]*282], dtype=np.float32)
    clock, glob = fit_fill_templates([hr])
    interp = interpolate_short_gaps(hr, clock_template=clock, resolution="5m", k_max=5)
    print("clock template shape:", clock.shape, "global median:", glob)
    print("interp shape:", interp.shape)
    assert not np.isnan(interp).any()
    print("Transforms test passed!")

if __name__ == "__main__":
    main()
