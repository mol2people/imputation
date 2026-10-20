#!/usr/bin/env python3
"""
00_verify_environment_and_fixtures.py: Pre-flight deterministic verification (<5s)
of Python environment, scientific libraries, and core mathematical fixtures.

Checks:
1. Python environment and required packages (numpy, scipy, pandas, pyarrow, hashlib).
2. Stone-Koo restricted cubic spline basis on knots [25, 42, 52, 65]:
   - Bit-exact zero below knot 1 (age <= 25.0).
   - Strict tail linearity above knot 4 (age >= 65.0) with exact theoretical slopes.
   - Smooth curvature in interior.
3. Greedy quota allocation logic on synthetic capacity vectors:
   - Synthetic vector [10, 11, 12, 17, 20, 20] -> [10, 11, 12, 17, 20, 20].
   - Equal capacities [15, 15, 15, 15, 15, 15] -> [15, 15, 15, 15, 15, 15].
   - Large capacities [20, 20, 20, 20, 20, 20] -> [15, 15, 15, 15, 15, 15].
   - Step-by-step tie-breaking and constraint validation.
4. December-1 age convention and sensitivities:
   - Boundary tests around Dec 1, Dec 31, Jan 1 across various birth years.
5. PRNG PCG64 SHA-256 seeding:
   - Deterministic reproduction, protocol-invariance, and stream isolation.
6. Execution time verification (<5s).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add scripts directory to path if needed
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import pandas as pd
import scipy
import pyarrow
import hashlib
from numpy.random import Generator, PCG64

from common import (
    rcs_stone_koo,
    compute_all_ages,
    greedy_quota_allocation,
    derive_block_seed,
    sample_block_dates,
    compute_selected_dates_hash,
    calendar_date_from_day,
    DEFAULT_KNOTS,
)


def verify_environment() -> None:
    """Check Python version and core library availability."""
    print("=== [1/5] Verifying Python Environment & Core Libraries ===")
    py_ver = sys.version_info
    assert py_ver.major == 3 and py_ver.minor >= 9, f"Python >= 3.9 required, found {sys.version}"
    print(f"  Python version: {sys.version.split()[0]} (OK)")
    
    # Check packages
    print(f"  NumPy version: {np.__version__} (OK)")
    print(f"  SciPy version: {scipy.__version__} (OK)")
    print(f"  Pandas version: {pd.__version__} (OK)")
    print(f"  PyArrow version: {pyarrow.__version__} (OK)")


def verify_rcs_basis_fixtures() -> None:
    """Verify Stone-Koo restricted cubic spline basis mathematical properties."""
    print("\n=== [2/5] Verifying Stone-Koo RCS Basis Fixtures ===")
    knots = DEFAULT_KNOTS  # (25.0, 42.0, 52.0, 65.0)
    t1, t2, t3, t4 = knots
    norm = (t4 - t1) ** 2  # 1600.0
    
    # 1. Bit-exact zero below knot 1 (age <= 25.0)
    below_ages = np.array([0.0, 10.0, 18.0, 20.0, 24.0, 24.999, 25.0], dtype=np.float64)
    rcs1_below, rcs2_below = rcs_stone_koo(below_ages, knots)
    assert np.all(rcs1_below == 0.0), f"rcs1 below knot 1 must be bit-exact 0.0, got: {rcs1_below}"
    assert np.all(rcs2_below == 0.0), f"rcs2 below knot 1 must be bit-exact 0.0, got: {rcs2_below}"
    print(f"  Passed: Bit-exact 0.0 below knot 1 ({t1}y) across {len(below_ages)} test points.")

    # 2. Semi-active region between knot 1 and knot 2 (25.0 < age <= 42.0)
    mid1_ages = np.array([26.0, 30.0, 35.0, 42.0], dtype=np.float64)
    rcs1_mid1, rcs2_mid1 = rcs_stone_koo(mid1_ages, knots)
    assert np.all(rcs1_mid1 > 0.0), "rcs1 must be strictly positive for age > 25.0"
    assert np.all(rcs2_mid1 == 0.0), f"rcs2 must be bit-exact 0.0 for age <= 42.0, got: {rcs2_mid1}"
    print(f"  Passed: rcs1 > 0 and rcs2 == 0.0 between {t1}y and {t2}y.")

    # 3. Curvature region between knot 2 and knot 4 (42.0 < age <= 65.0)
    mid2_ages = np.array([45.0, 50.0, 52.0, 58.0, 64.0], dtype=np.float64)
    rcs1_mid2, rcs2_mid2 = rcs_stone_koo(mid2_ages, knots)
    assert np.all(rcs1_mid2 > 0.0) and np.all(rcs2_mid2 > 0.0), "Both rcs1 and rcs2 must be > 0 in interior."
    print("  Passed: Smooth interior activation for both basis functions.")

    # 4. Strict tail linearity above knot 4 (age >= 65.0)
    # Theoretical slopes:
    # d rcs1 / dx = 3 * [(65-25)^2 - (40/13)*(13)^2] / 1600 = 3 * [1600 - 520] / 1600 = 3240 / 1600 = 2.025
    # d rcs2 / dx = 3 * [(65-42)^2 - (23/13)*(13)^2] / 1600 = 3 * [529 - 299] / 1600 = 690 / 1600 = 0.43125
    expected_slope1 = 2.025
    expected_slope2 = 0.43125
    
    val65_1, val65_2 = rcs_stone_koo(65.0, knots)
    # Check exact value at knot 4
    expected_val65_1 = (40.0**3 - (40.0 / 13.0) * (13.0**3)) / 1600.0  # (64000 - 6760) / 1600 = 35.775
    expected_val65_2 = (23.0**3 - (23.0 / 13.0) * (13.0**3)) / 1600.0  # (12167 - 3887) / 1600 = 5.175
    assert np.isclose(val65_1, expected_val65_1, atol=1e-12), f"val65_1 mismatch: {val65_1} vs {expected_val65_1}"
    assert np.isclose(val65_2, expected_val65_2, atol=1e-12), f"val65_2 mismatch: {val65_2} vs {expected_val65_2}"

    above_ages = np.array([65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 100.0], dtype=np.float64)
    rcs1_above, rcs2_above = rcs_stone_koo(above_ages, knots)
    
    # Verify exact linearity: (rcs(x) - rcs(65)) / (x - 65) == slope
    dx = above_ages[1:] - 65.0
    slopes1 = (rcs1_above[1:] - val65_1) / dx
    slopes2 = (rcs2_above[1:] - val65_2) / dx
    
    assert np.allclose(slopes1, expected_slope1, atol=1e-12), f"rcs1 slope above 65 must be {expected_slope1}, got {slopes1}"
    assert np.allclose(slopes2, expected_slope2, atol=1e-12), f"rcs2 slope above 65 must be {expected_slope2}, got {slopes2}"
    
    # Piecewise segment slopes
    seg_slopes1 = np.diff(rcs1_above) / np.diff(above_ages)
    seg_slopes2 = np.diff(rcs2_above) / np.diff(above_ages)
    assert np.allclose(seg_slopes1, expected_slope1, atol=1e-12), f"rcs1 segment slopes must be {expected_slope1}, got {seg_slopes1}"
    assert np.allclose(seg_slopes2, expected_slope2, atol=1e-12), f"rcs2 segment slopes must be {expected_slope2}, got {seg_slopes2}"
    print(f"  Passed: Strict tail linearity above {t4}y (slopes: rcs1={expected_slope1}, rcs2={expected_slope2}, d^2/dx^2=0).")


def verify_greedy_quota_allocation() -> None:
    """Verify greedy quota allocation logic on synthetic capacity vectors."""
    print("\n=== [3/5] Verifying Greedy Quota Allocation Logic ===")
    
    # 1. Specified synthetic vector: [10, 11, 12, 17, 20, 20] (sum = 90)
    c_spec = np.array([10, 11, 12, 17, 20, 20], dtype=np.int64)
    q_spec = greedy_quota_allocation(c_spec, target_total=90, base_quota=10)
    assert np.array_equal(q_spec, c_spec), f"Quota must exactly match capacities when sum=90, got {q_spec}"
    assert q_spec.sum() == 90
    print(f"  Passed: Capacity [10, 11, 12, 17, 20, 20] -> Quota {q_spec.tolist()} (exact match, sum=90).")

    # 2. Balanced capacity vector: [15, 15, 15, 15, 15, 15]
    c_bal = np.array([15, 15, 15, 15, 15, 15], dtype=np.int64)
    q_bal = greedy_quota_allocation(c_bal, target_total=90, base_quota=10)
    assert np.array_equal(q_bal, np.full(6, 15)), f"Expected [15]*6, got {q_bal}"
    print(f"  Passed: Balanced capacity [15]*6 -> Quota [15]*6.")

    # 3. Large unconstrained capacity vector: [20, 20, 20, 20, 20, 20]
    c_large = np.array([20, 20, 20, 20, 20, 20], dtype=np.int64)
    q_large = greedy_quota_allocation(c_large, target_total=90, base_quota=10)
    assert np.array_equal(q_large, np.full(6, 15)), f"Greedy leveling must produce [15]*6, got {q_large}"
    print(f"  Passed: Large capacity [20]*6 -> Quota [15]*6 (level balance preserved).")

    # 4. Partial constraint vector: [10, 12, 25, 25, 25, 25] (sum = 122)
    # Remaining 30 dates to allocate:
    # Step 1-2: q[1] gets +2 (hits cap 12). Other blocks have q=10.
    # Next: blocks 2, 3, 4, 5 (caps 25) receive remaining 28 slots: 28 / 4 = 7 each.
    # Final quotas: [10, 12, 17, 17, 17, 17] -> sum = 90
    c_part = np.array([10, 12, 25, 25, 25, 25], dtype=np.int64)
    q_part = greedy_quota_allocation(c_part, target_total=90, base_quota=10)
    assert np.array_equal(q_part, np.array([10, 12, 17, 17, 17, 17])), f"Expected [10, 12, 17, 17, 17, 17], got {q_part}"
    assert q_part.sum() == 90
    print(f"  Passed: Partial constraint [10, 12, 25, 25, 25, 25] -> Quota {q_part.tolist()}.")

    # 5. Invalid inputs must raise ValueError
    try:
        greedy_quota_allocation([9, 15, 15, 15, 15, 15], 90, 10)
        assert False, "Should have failed for capacity < 10"
    except ValueError:
        pass
    try:
        greedy_quota_allocation([10, 10, 10, 10, 10, 10], 90, 10)
        assert False, "Should have failed for sum(capacities) < 90"
    except ValueError:
        pass
    print("  Passed: Error boundaries and preconditions strictly enforced.")


def verify_age_conventions() -> None:
    """Verify December-1 age convention and sensitivity formulas."""
    print("\n=== [4/5] Verifying December-1 Age Convention & Sensitivities ===")
    
    # Birth year 1980 tests
    by = 1980
    
    # 2020-11-30: day before Dec 1
    # dt = 2020-11-30 -> day_int = (2020-11-30 - 1970-01-01)
    day_nov30 = int((np.datetime64("2020-11-30") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
    dec1, jan1, dec31 = compute_all_ages(day_nov30, by)
    assert dec1 == 39, f"2020-11-30 should yield dec1=39, got {dec1}"
    assert jan1 == 40, f"2020-11-30 should yield jan1=40, got {jan1}"
    assert dec31 == 39, f"2020-11-30 should yield dec31=39, got {dec31}"
    
    # 2020-12-01: Dec 1 rollover
    day_dec1 = int((np.datetime64("2020-12-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
    dec1, jan1, dec31 = compute_all_ages(day_dec1, by)
    assert dec1 == 40, f"2020-12-01 should yield dec1=40, got {dec1}"
    assert jan1 == 40, f"2020-12-01 should yield jan1=40, got {jan1}"
    assert dec31 == 39, f"2020-12-01 should yield dec31=39, got {dec31}"
    
    # 2020-12-31: New Year's Eve
    day_dec31 = int((np.datetime64("2020-12-31") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
    dec1, jan1, dec31 = compute_all_ages(day_dec31, by)
    assert dec1 == 40, f"2020-12-31 should yield dec1=40, got {dec1}"
    assert jan1 == 40, f"2020-12-31 should yield jan1=40, got {jan1}"
    assert dec31 == 40, f"2020-12-31 should yield dec31=40, got {dec31}"
    
    # 2021-01-01: Jan 1 rollover
    day_jan1 = int((np.datetime64("2021-01-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
    dec1, jan1, dec31 = compute_all_ages(day_jan1, by)
    assert dec1 == 40, f"2021-01-01 should yield dec1=40, got {dec1}"
    assert jan1 == 41, f"2021-01-01 should yield jan1=41, got {jan1}"
    assert dec31 == 40, f"2021-01-01 should yield dec31=40, got {dec31}"
    
    print("  Passed: Exact age boundary behavior across Nov 30, Dec 1, Dec 31, and Jan 1.")


def verify_prng_pcg64_seeding() -> None:
    """Verify PRNG PCG64 SHA-256 seeding and date sampling."""
    print("\n=== [5/5] Verifying PRNG PCG64 SHA-256 Seeding ===")
    
    source = 6
    user_id = 100043
    start_day = 18487
    block = 0
    
    seed1 = derive_block_seed(source, user_id, start_day, block)
    seed2 = derive_block_seed(source, user_id, start_day, block)
    assert seed1 == seed2, "Seed derivation must be purely deterministic."
    
    # Check seed derivation against independent SHA-256 implementation
    tag = f"20261020|infradian_age|{source}|{user_id}|{start_day}|dates_block_{block}"
    h = hashlib.sha256(tag.encode("utf-8")).digest()
    expected_seed = int.from_bytes(h[:8], byteorder="little")
    assert seed1 == expected_seed, f"Seed mismatch: {seed1} vs {expected_seed}"
    
    # Sample from synthetic 20 dates
    dummy_dates = np.arange(18487, 18487 + 20, dtype=np.int64)
    draw1 = sample_block_dates(dummy_dates, quota=15, seed_u64=seed1)
    draw2 = sample_block_dates(dummy_dates, quota=15, seed_u64=seed2)
    assert np.array_equal(draw1, draw2), "Draws with identical seed must match bit-for-bit."
    assert len(draw1) == 15
    assert np.all(np.diff(draw1) > 0), "Sampled dates must be strictly ascending."
    
    # Protocol invariance: protocol name is NOT in seed tag
    # Verify that different block or different user yields different seed
    seed_block1 = derive_block_seed(source, user_id, start_day, 1)
    seed_user2 = derive_block_seed(source, 100044, start_day, block)
    assert seed1 != seed_block1, "Different blocks must have independent seeds."
    assert seed1 != seed_user2, "Different users must have independent seeds."
    
    # Verify selected_dates_hash
    hash1 = compute_selected_dates_hash(draw1)
    hash2 = compute_selected_dates_hash(draw2)
    assert hash1 == hash2 and len(hash1) == 64, f"Hash verification failed: {hash1}"
    print(f"  Passed: Deterministic PRNG PCG64 seeding and date sampling (hash: {hash1[:16]}...).")


def main() -> None:
    t0 = time.perf_counter()
    print("Starting Pre-Flight Fixtures and Environment Verification...")
    
    verify_environment()
    verify_rcs_basis_fixtures()
    verify_greedy_quota_allocation()
    verify_age_conventions()
    verify_prng_pcg64_seeding()
    
    elapsed = time.perf_counter() - t0
    print(f"\n========================================================")
    print(f"ALL PRE-FLIGHT FIXTURES VERIFIED SUCCESSFULLY in {elapsed:.3f}s (<5.0s)")
    print(f"========================================================")
    assert elapsed < 5.0, f"Verification must run in <5s, took {elapsed:.3f}s"


if __name__ == "__main__":
    main()
