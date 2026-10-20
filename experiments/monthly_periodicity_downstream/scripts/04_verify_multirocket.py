"""
04_verify_multirocket.py
Verify that MultiRocket imports from heartbeat_lab.model, fits, and transforms correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "HEARTBEAT_LAB" / "src"))

import numpy as np
from heartbeat_lab.data import STATS
from heartbeat_lab.model import reference_functions, seed_numba

def main() -> None:
    print("Testing MultiRocket import...")
    mr_fit, mr_transform = reference_functions()
    print("Successfully retrieved mr_fit and mr_transform.")
    
    # Test on a small synthetic array of shape (10, 288)
    rng = np.random.default_rng(20261020)
    sample_x = rng.standard_normal((10, 288)).astype(np.float64)
    sample_diff = np.diff(sample_x, axis=1)
    
    seed_numba(12345)
    mr_base = mr_fit(sample_x, num_features=1250, max_dilations_per_kernel=32)
    seed_numba(12346)
    mr_diff = mr_fit(sample_diff, num_features=1250, max_dilations_per_kernel=32)
    
    bx = sample_x[:2]
    bd = sample_diff[:2]
    feats = mr_transform(bx, bd, mr_base, mr_diff, n_features_per_kernel=4)
    print(f"Transformed features shape for 2 days: {feats.shape}")
    print(f"Features per day: {feats.shape[1]} (expected 9408)")
    assert feats.shape[1] == 9408, f"Expected 9408, got {feats.shape[1]}"
    print("MultiRocket verification passed!")

if __name__ == "__main__":
    main()
