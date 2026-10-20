#!/usr/bin/env python3
"""
test_m1_windows_and_cohorts.py: Comprehensive test suite for Milestone 1.

Scope:
  - Cohort extraction, Samsung cache construction, and demographic validation.
  - Standardized 180-day windowing and block partitioning.
  - Greedy quota allocation and uniform date sampling protocols.
  - Chronological age derivation (December-1 convention and sensitivities).
  - Mathematical properties of Stone-Koo restricted cubic spline basis.
  - Direct parity testing against `scripts/common.py`.
  - Artifact verification and upstream cache immutability.

Authoritative specifications:
  - ORIGINAL_REQUEST.md
  - experiments/infradian_age_dependency/PLAN.md (§§1-6, 11)
  - experiments/infradian_age_dependency/PROJECT.md (Interface Contracts)
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Path Resolution and Environment
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments/infradian_age_dependency"
SCRIPTS_DIR = EXP_DIR / "scripts"
ARTIFACTS_DIR = EXP_DIR / "artifacts"
APPLE_CACHE_DIR = REPO_ROOT / "experiments/apple_cohort_scaling/artifacts/cache_5min"
GARMIN_CACHE_DIR = REPO_ROOT / "experiments/garmin_infradian_replication/artifacts/cache_5min"

# Add scripts directory to path to enable direct imports
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import common
    HAS_COMMON = True
except ImportError:
    HAS_COMMON = False


# -----------------------------------------------------------------------------
# Reference Implementations Derived Strictly from PLAN.md §§5-6
# -----------------------------------------------------------------------------
def reference_window_blocks(start_day: int) -> list[tuple[int, int]]:
    """Return the 6 contiguous 30-day block intervals [start, end) for window start_day."""
    return [(start_day + 30 * b, start_day + 30 * (b + 1)) for b in range(6)]


def reference_greedy_quota_allocation(
    capacities: list[int] | tuple[int, ...] | np.ndarray,
    total_quota: int = 90,
    base_quota: int = 10,
) -> list[int]:
    """
    Allocate total_quota across 6 blocks with capacity vector c_b.
    
    Rule from PLAN.md §5:
      Initialize six quotas q_b = 10.
      Allocate remaining dates one at a time to candidate block with smallest current
      quota among those with q_b < c_b; break ties by ascending block index.
    """
    c = list(capacities)
    if len(c) != 6:
        raise ValueError(f"Expected 6 block capacities, got {len(c)}")
    if any(cap < base_quota for cap in c):
        raise ValueError(f"Each block capacity must be >= {base_quota}, got {c}")
    if sum(c) < total_quota:
        raise ValueError(f"Sum of capacities ({sum(c)}) must be >= total_quota ({total_quota})")

    quotas = [base_quota] * 6
    remaining = total_quota - base_quota * 6

    for _ in range(remaining):
        candidates = [b for b in range(6) if quotas[b] < c[b]]
        if not candidates:
            raise ValueError("Greedy allocation failed: no candidate block has remaining capacity")
        min_q = min(quotas[b] for b in candidates)
        best_b = min(b for b in candidates if quotas[b] == min_q)
        quotas[best_b] += 1

    return quotas


def reference_seed_derivation(source: int, user_id: int, window_start_day: int, block_idx: int) -> int:
    """
    Derive unsigned 64-bit little-endian integer seed from SHA-256 digest.
    
    Rule from PLAN.md §5:
      20261020|infradian_age|{source}|{user_id}|{window_start_day}|dates_block_{b}
    """
    seed_str = f"20261020|infradian_age|{source}|{user_id}|{window_start_day}|dates_block_{block_idx}"
    h = hashlib.sha256(seed_str.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little")


def reference_sample_block_dates(qualified_dates: list[int] | np.ndarray, quota: int, seed: int) -> list[int]:
    """Sample quota dates without replacement using PCG64 PRNG permutation, returning sorted dates."""
    dates_list = list(qualified_dates)
    if len(dates_list) < quota:
        raise ValueError(f"Capacity {len(dates_list)} is less than quota {quota}")
    rng = np.random.Generator(np.random.PCG64(seed))
    perm = rng.permutation(len(dates_list))
    selected_indices = sorted(perm[:quota])
    return [dates_list[i] for i in selected_indices]


def reference_selected_dates_hash(sorted_dates: list[int] | np.ndarray) -> str:
    """Compute SHA-256 of sorted dates serialized as little-endian signed int64 calendar days."""
    arr = np.array(sorted_dates, dtype="<i8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def reference_december_1_age(window_start: datetime.date | int, birth_year: int | float) -> int:
    """
    Calculate chronological age using December-1 convention.
    
    Formula from PLAN.md §6:
      age = start.year - birth_year - 1 + I(start.month == 12)
    """
    if isinstance(window_start, (int, np.integer)):
        d = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(window_start))
    elif isinstance(window_start, datetime.date):
        d = window_start
    else:
        raise TypeError(f"Unsupported window_start type: {type(window_start)}")

    if pd.isna(birth_year) or birth_year is None:
        raise ValueError("Missing birth year")

    by = int(birth_year)
    if by != birth_year:
        raise ValueError(f"Non-integral birth year: {birth_year}")
    if by > d.year:
        raise ValueError(f"Birth year {by} later than window start year {d.year}")
    if by < 1900 or by > 2010:
        raise ValueError(f"Birth year {by} outside plausible range [1900, 2010]")

    age = d.year - by - 1 + (1 if d.month == 12 else 0)
    if age < 0 or age > 120:
        raise ValueError(f"Derived age {age} outside plausible range [0, 120]")
    return age


def reference_january_1_age(window_start: datetime.date | int, birth_year: int | float) -> int:
    """Sensitivity: age assuming January 1 birthday."""
    if isinstance(window_start, (int, np.integer)):
        d = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(window_start))
    else:
        d = window_start
    by = int(birth_year)
    return d.year - by


def reference_december_31_age(window_start: datetime.date | int, birth_year: int | float) -> int:
    """Sensitivity: age assuming December 31 birthday."""
    if isinstance(window_start, (int, np.integer)):
        d = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(window_start))
    else:
        d = window_start
    by = int(birth_year)
    return d.year - by - 1 + (1 if (d.month == 12 and d.day == 31) else 0)


def reference_stone_koo_rcs(age: float | np.ndarray, knots: list[float] = [25.0, 42.0, 52.0, 65.0]) -> tuple[np.ndarray, np.ndarray]:
    """Stone-Koo restricted cubic spline basis variables rcs1 and rcs2."""
    a = np.asarray(age, dtype=np.float64)
    t1, t2, t3, t4 = knots
    norm = (t4 - t1) ** 2  # 1600.0

    def pos_cube(u: np.ndarray) -> np.ndarray:
        return np.maximum(u, 0.0) ** 3

    # rcs1
    rcs1 = (
        pos_cube(a - t1)
        - ((t4 - t1) / (t4 - t3)) * pos_cube(a - t3)
        + ((t3 - t1) / (t4 - t3)) * pos_cube(a - t4)
    ) / norm

    # rcs2
    rcs2 = (
        pos_cube(a - t2)
        - ((t4 - t2) / (t4 - t3)) * pos_cube(a - t3)
        + ((t3 - t2) / (t4 - t3)) * pos_cube(a - t4)
    ) / norm

    return rcs1, rcs2


# -----------------------------------------------------------------------------
# Tier 1: Feature Coverage Tests
# -----------------------------------------------------------------------------
class TestTier1FeatureCoverage(unittest.TestCase):
    """Tier 1: Feature coverage for windowing, quotas, age derivation, and demographics."""

    def test_window_exact_180_day_span(self):
        """Window [s, s+180) must span exactly 180 calendar days."""
        for s in [0, 18000, 19000, 20000]:
            blocks = reference_window_blocks(s)
            self.assertEqual(len(blocks), 6)
            start_day = blocks[0][0]
            end_day = blocks[5][1]
            self.assertEqual(start_day, s)
            self.assertEqual(end_day, s + 180)
            self.assertEqual(end_day - start_day, 180)

    def test_six_contiguous_30_day_blocks(self):
        """Window must partition into exactly 6 contiguous blocks of 30 days each."""
        s = 18500
        blocks = reference_window_blocks(s)
        for b, (b_start, b_end) in enumerate(blocks):
            self.assertEqual(b_start, s + 30 * b)
            self.assertEqual(b_end, s + 30 * (b + 1))
            self.assertEqual(b_end - b_start, 30)
            if b > 0:
                self.assertEqual(b_start, blocks[b - 1][1], f"Gap between block {b-1} and {b}")

    def test_block_membership_boundary_behavior(self):
        """Test boundary days map unambiguously to the correct block."""
        s = 18000
        # s falls in block 0
        self.assertEqual((s - s) // 30, 0)
        # s + 29 falls in block 0
        self.assertEqual((s + 29 - s) // 30, 0)
        # s + 30 falls in block 1 (half-open [s, s+30))
        self.assertEqual((s + 30 - s) // 30, 1)
        # s + 179 falls in block 5
        self.assertEqual((s + 179 - s) // 30, 5)
        # s + 180 is outside the window
        self.assertGreaterEqual((s + 180 - s) // 30, 6)

    def test_block_quotas_expanded90_constraints(self):
        """For expanded90, every block quota q_b must be >= 10, q_b <= c_b, and sum(q_b) == 90."""
        rng = np.random.default_rng(20261020)
        for _ in range(50):
            # Generate random capacities with each >= 10 and sum >= 90
            c = [int(rng.integers(10, 31)) for _ in range(6)]
            if sum(c) < 90:
                c[5] += 90 - sum(c)
            quotas = reference_greedy_quota_allocation(c)
            self.assertEqual(len(quotas), 6)
            self.assertEqual(sum(quotas), 90)
            for b in range(6):
                self.assertGreaterEqual(quotas[b], 10)
                self.assertLessEqual(quotas[b], c[b])

    def test_block_quotas_balanced90_constraints(self):
        """For balanced90, each block quota must be exactly 15."""
        c = [15, 16, 20, 22, 18, 15]
        quotas = reference_greedy_quota_allocation(c)
        self.assertEqual(quotas, [15, 15, 15, 15, 15, 15])
        self.assertEqual(sum(quotas), 90)

    def test_unique_selected_dates_count(self):
        """Exactly 90 unique dates must be selected per eligible window."""
        s = 18500
        blocks = reference_window_blocks(s)
        c = [16, 17, 18, 15, 20, 15]
        quotas = reference_greedy_quota_allocation(c)

        all_selected = []
        for b in range(6):
            b_start, _ = blocks[b]
            # fake sorted qualified dates in this block
            q_dates = list(range(b_start, b_start + c[b]))
            seed = reference_seed_derivation(source=6, user_id=42, window_start_day=s, block_idx=b)
            sampled = reference_sample_block_dates(q_dates, quotas[b], seed)
            self.assertEqual(len(sampled), quotas[b])
            # verify uniqueness within block
            self.assertEqual(len(sampled), len(set(sampled)))
            all_selected.extend(sampled)

        self.assertEqual(len(all_selected), 90)
        # verify uniqueness across entire window
        self.assertEqual(len(set(all_selected)), 90)

    def test_december_1_age_formula_correctness(self):
        """
        Verify December-1 age formula:
          age = start.year - birth_year - 1 + I(start.month == 12)
        """
        # Case 1: Start in January 2021, birth_year 1990 -> turns 31 in Dec 2021, current age 30
        d_jan = datetime.date(2021, 1, 15)
        self.assertEqual(reference_december_1_age(d_jan, 1990), 30)

        # Case 2: Start in June 2021, birth_year 1990 -> age 30
        d_jun = datetime.date(2021, 6, 20)
        self.assertEqual(reference_december_1_age(d_jun, 1990), 30)

        # Case 3: Start in November 2021, birth_year 1990 -> age 30
        d_nov = datetime.date(2021, 11, 30)
        self.assertEqual(reference_december_1_age(d_nov, 1990), 30)

        # Case 4: Start on December 1, 2021, birth_year 1990 -> rollover to age 31
        d_dec1 = datetime.date(2021, 12, 1)
        self.assertEqual(reference_december_1_age(d_dec1, 1990), 31)

        # Case 5: Start on December 31, 2021, birth_year 1990 -> age 31
        d_dec31 = datetime.date(2021, 12, 31)
        self.assertEqual(reference_december_1_age(d_dec31, 1990), 31)

    def test_demographic_mapping_ranges(self):
        """Verify salutation in {10, 20}, birth_year in [1900, 2010], age in [0, 120]."""
        valid_salutations = {10, 20}
        self.assertIn(10, valid_salutations)  # Female
        self.assertIn(20, valid_salutations)  # Male
        self.assertNotIn(30, valid_salutations)  # Other (excluded)

        # Plausibility checks
        d = datetime.date(2021, 5, 1)
        self.assertEqual(reference_december_1_age(d, 1901), 119)
        self.assertEqual(reference_december_1_age(d, 2010), 10)


# -----------------------------------------------------------------------------
# Tier 2: Boundary & Corner Cases Tests
# -----------------------------------------------------------------------------
class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """Tier 2: Boundary conditions, corner cases, and adversarial scenarios."""

    def test_capacity_vector_10_10_10_10_10_40(self):
        """
        Capacity vector [10, 10, 10, 10, 10, 40]:
        Sum is exactly 90. Greedy allocation must assign [10, 10, 10, 10, 10, 40].
        """
        c = [10, 10, 10, 10, 10, 40]
        self.assertEqual(sum(c), 90)
        q = reference_greedy_quota_allocation(c)
        self.assertEqual(q, [10, 10, 10, 10, 10, 40])
        self.assertEqual(sum(q), 90)

    def test_capacity_vector_15_15_15_15_15_15(self):
        """
        Capacity vector [15, 15, 15, 15, 15, 15]:
        Greedy allocation must produce balanced [15, 15, 15, 15, 15, 15].
        """
        c = [15, 15, 15, 15, 15, 15]
        q = reference_greedy_quota_allocation(c)
        self.assertEqual(q, [15, 15, 15, 15, 15, 15])
        self.assertEqual(sum(q), 90)

    def test_capacity_vector_10_11_12_17_20_20(self):
        """
        Capacity vector [10, 11, 12, 17, 20, 20]:
        Qualifies for expanded90 (sum 90, each >= 10), fails balanced90 (first 3 < 15).
        Greedy allocation must produce [10, 11, 12, 17, 20, 20].
        """
        c = [10, 11, 12, 17, 20, 20]
        self.assertEqual(sum(c), 90)
        # Check protocol eligibility
        is_expanded_eligible = all(cap >= 10 for cap in c) and sum(c) >= 90
        is_balanced_eligible = all(cap >= 15 for cap in c)
        self.assertTrue(is_expanded_eligible)
        self.assertFalse(is_balanced_eligible)

        q = reference_greedy_quota_allocation(c)
        self.assertEqual(q, [10, 11, 12, 17, 20, 20])
        self.assertEqual(sum(q), 90)

    def test_capacity_symmetric_surplus(self):
        """Surplus capacity [20, 20, 20, 20, 20, 20] must allocate balanced [15, 15, 15, 15, 15, 15]."""
        c = [20, 20, 20, 20, 20, 20]
        q = reference_greedy_quota_allocation(c)
        self.assertEqual(q, [15, 15, 15, 15, 15, 15])

    def test_capacity_asymmetric_bottleneck(self):
        """
        Capacity [10, 15, 15, 15, 15, 25]:
        Block 0 is capped at 10. Blocks 1-4 take +5 to reach 15. Block 5 takes remainder (+10 to reach 20).
        Expected output: [10, 15, 15, 15, 15, 20].
        """
        c = [10, 15, 15, 15, 15, 25]
        q = reference_greedy_quota_allocation(c)
        self.assertEqual(q, [10, 15, 15, 15, 15, 20])
        self.assertEqual(sum(q), 90)

    def test_capacity_stepped_capacities(self):
        """
        Capacity [12, 12, 12, 18, 18, 18]:
        Blocks 0-2 cap at 12 (+2 each = 6 dates). Remaining 24 dates split across blocks 3-5 (+8 each).
        Expected output: [12, 12, 12, 18, 18, 18].
        """
        c = [12, 12, 12, 18, 18, 18]
        q = reference_greedy_quota_allocation(c)
        self.assertEqual(q, [12, 12, 12, 18, 18, 18])
        self.assertEqual(sum(q), 90)

    def test_allocation_insufficient_capacity_raises_error(self):
        """Sum of capacities < 90 or block capacity < 10 must raise ValueError."""
        # Sum is 89
        with self.assertRaises(ValueError):
            reference_greedy_quota_allocation([10, 10, 10, 10, 10, 39])
        # A block has capacity 9
        with self.assertRaises(ValueError):
            reference_greedy_quota_allocation([9, 20, 20, 20, 20, 20])
        # Incorrect number of blocks
        with self.assertRaises(ValueError):
            reference_greedy_quota_allocation([15, 15, 15, 15, 15])

    def test_missing_and_invalid_birth_date_handling(self):
        """Missing, future, or non-integer birth dates must raise ValueError."""
        d = datetime.date(2020, 6, 1)
        # None
        with self.assertRaises(ValueError):
            reference_december_1_age(d, None)
        # NaN
        with self.assertRaises(ValueError):
            reference_december_1_age(d, np.nan)
        # Future birth year > window start year
        with self.assertRaises(ValueError):
            reference_december_1_age(d, 2021)
        # Non-integer birth year
        with self.assertRaises(ValueError):
            reference_december_1_age(d, 1985.5)
        # Implausible birth year < 1900
        with self.assertRaises(ValueError):
            reference_december_1_age(d, 1850)
        # Implausible birth year > 2010
        with self.assertRaises(ValueError):
            reference_december_1_age(d, 2015)

    def test_age_sensitivities_ordering(self):
        """
        Verify relationship across age conventions:
          age_jan1 >= age_dec1 >= age_dec31
        """
        by = 1985
        # Test across every day of year 2020
        d = datetime.date(2020, 1, 1)
        while d.year == 2020:
            a_jan1 = reference_january_1_age(d, by)
            a_dec1 = reference_december_1_age(d, by)
            a_dec31 = reference_december_31_age(d, by)
            self.assertGreaterEqual(a_jan1, a_dec1, f"Failed at {d}")
            self.assertGreaterEqual(a_dec1, a_dec31, f"Failed at {d}")
            d += datetime.timedelta(days=1)

    def test_prng_seeding_determinism(self):
        """Identical inputs must yield bit-exact identical seeds and sampled dates."""
        s1 = reference_seed_derivation(source=6, user_id=101, window_start_day=18500, block_idx=2)
        s2 = reference_seed_derivation(source=6, user_id=101, window_start_day=18500, block_idx=2)
        self.assertEqual(s1, s2)

        q_dates = list(range(18560, 18580))
        sample1 = reference_sample_block_dates(q_dates, quota=15, seed=s1)
        sample2 = reference_sample_block_dates(q_dates, quota=15, seed=s2)
        self.assertEqual(sample1, sample2)

    def test_prng_seeding_protocol_invariance(self):
        """
        Protocol name is NOT in seed string.
        Identical source, user_id, window_start, block, and quota must yield identical selections.
        """
        seed = reference_seed_derivation(source=3, user_id=999, window_start_day=19000, block_idx=0)
        q_dates = list(range(19000, 19020))
        sample_expanded = reference_sample_block_dates(q_dates, quota=15, seed=seed)
        sample_balanced = reference_sample_block_dates(q_dates, quota=15, seed=seed)
        self.assertEqual(sample_expanded, sample_balanced)

    def test_prng_seeding_parameter_sensitivity(self):
        """Changing block, window start, user_id, or source must produce distinct seeds."""
        base = reference_seed_derivation(source=6, user_id=1000, window_start_day=18500, block_idx=0)
        diff_block = reference_seed_derivation(source=6, user_id=1000, window_start_day=18500, block_idx=1)
        diff_start = reference_seed_derivation(source=6, user_id=1000, window_start_day=18501, block_idx=0)
        diff_user = reference_seed_derivation(source=6, user_id=1001, window_start_day=18500, block_idx=0)
        diff_src = reference_seed_derivation(source=3, user_id=1000, window_start_day=18500, block_idx=0)

        seeds = {base, diff_block, diff_start, diff_user, diff_src}
        self.assertEqual(len(seeds), 5, "Collision detected among parameter variations!")

    def test_date_serialization_hash_properties(self):
        """Verify selected_dates_hash matches SHA-256 of little-endian signed int64 array."""
        dates = [18500, 18505, 18510, 18520]
        h1 = reference_selected_dates_hash(dates)
        self.assertEqual(len(h1), 64)

        # Hash is sensitive to single-day change
        dates_mod = [18500, 18505, 18511, 18520]
        h2 = reference_selected_dates_hash(dates_mod)
        self.assertNotEqual(h1, h2)

        # Hash is sensitive to ordering
        dates_rev = [18520, 18510, 18505, 18500]
        h3 = reference_selected_dates_hash(dates_rev)
        self.assertNotEqual(h1, h3)

    def test_stone_koo_spline_properties(self):
        """
        Verify Stone-Koo basis properties on knots [25, 42, 52, 65]:
          - rcs1 = 0.0 and rcs2 = 0.0 for age <= 25.
          - Strictly linear tails for age >= 65 (constant first derivative, zero second derivative).
        """
        # 1. Below knot 1: exactly zero
        ages_below = np.array([15.0, 18.0, 20.0, 24.99, 25.0])
        r1, r2 = reference_stone_koo_rcs(ages_below)
        np.testing.assert_allclose(r1, 0.0, atol=1e-12)
        np.testing.assert_allclose(r2, 0.0, atol=1e-12)

        # 2. Above knot 4: constant slope (linear)
        # Check finite differences at 70, 75, 80
        test_ages = np.array([65.0, 70.0, 75.0, 80.0])
        r1_above, r2_above = reference_stone_koo_rcs(test_ages)
        slopes_r1 = np.diff(r1_above) / np.diff(test_ages)
        slopes_r2 = np.diff(r2_above) / np.diff(test_ages)

        np.testing.assert_allclose(slopes_r1, slopes_r1[0], rtol=1e-9)
        np.testing.assert_allclose(slopes_r2, slopes_r2[0], rtol=1e-9)


# -----------------------------------------------------------------------------
# Tier 2.5: Implementation Parity with scripts/common.py
# -----------------------------------------------------------------------------
class TestCommonModuleParity(unittest.TestCase):
    """Direct verification of scripts/common.py against independent reference ground truth."""

    def setUp(self):
        if not HAS_COMMON:
            self.skipTest("common.py is not yet available in scripts/")

    def test_common_rcs_stone_koo_parity(self):
        """Verify common.rcs_stone_koo matches reference_stone_koo_rcs across full lifespan."""
        eval_ages = np.linspace(15.0, 95.0, 801)
        ref_r1, ref_r2 = reference_stone_koo_rcs(eval_ages)
        impl_r1, impl_r2 = common.rcs_stone_koo(eval_ages)

        np.testing.assert_allclose(impl_r1, ref_r1, atol=1e-12, err_msg="rcs1 parity mismatch")
        np.testing.assert_allclose(impl_r2, ref_r2, atol=1e-12, err_msg="rcs2 parity mismatch")

    def test_common_greedy_quota_allocation_parity(self):
        """Verify common.greedy_quota_allocation produces identical quotas for benchmark vectors."""
        test_cases = [
            [10, 10, 10, 10, 10, 40],
            [15, 15, 15, 15, 15, 15],
            [10, 11, 12, 17, 20, 20],
            [20, 20, 20, 20, 20, 20],
            [10, 15, 15, 15, 15, 25],
            [12, 12, 12, 18, 18, 18],
        ]
        for c in test_cases:
            ref_q = reference_greedy_quota_allocation(c)
            impl_q = common.greedy_quota_allocation(c)
            self.assertEqual(list(impl_q), ref_q, f"Mismatch for capacity {c}")

    def test_common_age_derivation_parity(self):
        """Verify common.compute_all_ages matches reference age functions."""
        # 18500 days is 2020-08-26
        for day in [18000, 18500, 18600, 18700]:
            for by in [1950, 1975, 1990, 2005]:
                ref_dec1 = reference_december_1_age(day, by)
                ref_jan1 = reference_january_1_age(day, by)
                ref_dec31 = reference_december_31_age(day, by)
                impl_dec1, impl_jan1, impl_dec31 = common.compute_all_ages(day, by)

                self.assertEqual(impl_dec1, ref_dec1, f"Mismatch dec1 at day {day}, by {by}")
                self.assertEqual(impl_jan1, ref_jan1, f"Mismatch jan1 at day {day}, by {by}")
                self.assertEqual(impl_dec31, ref_dec31, f"Mismatch dec31 at day {day}, by {by}")

    def test_common_seed_and_sampling_parity(self):
        """Verify common.derive_block_seed, sample_block_dates, and hash match reference."""
        seed_ref = reference_seed_derivation(source=6, user_id=42, window_start_day=18500, block_idx=3)
        seed_impl = common.derive_block_seed(source=6, user_id=42, window_start_day=18500, block=3)
        self.assertEqual(seed_impl, seed_ref)

        q_dates = np.array(range(18590, 18615), dtype=np.int64)
        sampled_ref = reference_sample_block_dates(q_dates, quota=15, seed=seed_ref)
        sampled_impl = common.sample_block_dates(q_dates, quota=15, seed_u64=seed_impl)
        self.assertEqual(list(sampled_impl), sampled_ref)

        hash_ref = reference_selected_dates_hash(sampled_ref)
        hash_impl = common.compute_selected_dates_hash(sampled_impl)
        self.assertEqual(hash_impl, hash_ref)

    def test_common_window_selection_search(self):
        """Verify common.find_qualifying_windows on synthetic dates."""
        # Construct synthetic dates qualifying only for expanded90
        # Block 0: 10, Block 1: 11, Block 2: 12, Block 3: 17, Block 4: 20, Block 5: 20
        start = 18000
        dates_b0 = list(range(start, start + 10))
        dates_b1 = list(range(start + 30, start + 30 + 11))
        dates_b2 = list(range(start + 60, start + 60 + 12))
        dates_b3 = list(range(start + 90, start + 90 + 17))
        dates_b4 = list(range(start + 120, start + 120 + 20))
        dates_b5 = list(range(start + 150, start + 150 + 20))
        all_dates = np.array(dates_b0 + dates_b1 + dates_b2 + dates_b3 + dates_b4 + dates_b5, dtype=np.int64)

        s1_exp, _ = common.find_qualifying_windows(all_dates, protocol="expanded90")
        s1_bal, _ = common.find_qualifying_windows(all_dates, protocol="balanced90")
        self.assertEqual(s1_exp, start)
        self.assertIsNone(s1_bal)


# -----------------------------------------------------------------------------
# Tier 3: Artifact Verification and Cache Immutability Tests
# -----------------------------------------------------------------------------
class TestTier3ArtifactVerification(unittest.TestCase):
    """Tier 3: Artifact schema, data consistency, and upstream cache protection."""

    def test_upstream_apple_cache_immutability(self):
        """Apple Watch upstream 5m cache must exist and contain exactly 11,139 files."""
        self.assertTrue(APPLE_CACHE_DIR.exists(), f"Apple cache missing at {APPLE_CACHE_DIR}")
        apple_files = list(APPLE_CACHE_DIR.glob("customer_*.npz"))
        self.assertEqual(
            len(apple_files),
            11139,
            f"Apple cache file count modified! Expected 11,139, found {len(apple_files)}",
        )

    def test_upstream_garmin_cache_immutability(self):
        """Garmin Tier 1c upstream 5m cache must exist and contain exactly 4,410 files."""
        self.assertTrue(GARMIN_CACHE_DIR.exists(), f"Garmin cache missing at {GARMIN_CACHE_DIR}")
        garmin_files = list(GARMIN_CACHE_DIR.glob("customer_*.npz"))
        self.assertEqual(
            len(garmin_files),
            4410,
            f"Garmin cache file count modified! Expected 4,410, found {len(garmin_files)}",
        )

    def test_generated_participants_parquet_if_present(self):
        """Validate participants.parquet schema, non-null constraints, and plausibility bounds."""
        candidate_paths = [
            ARTIFACTS_DIR / "manifests/participants.parquet",
            ARTIFACTS_DIR / "participants.parquet",
        ]
        p_path = next((p for p in candidate_paths if p.exists()), None)
        if p_path is None:
            self.skipTest("participants.parquet not yet generated by worker")

        df = pd.read_parquet(p_path)
        # Primary demographic keys must be non-null for all candidate participants
        base_required_cols = ["user_id", "source", "salutation", "birth_year"]
        for col in base_required_cols:
            self.assertIn(col, df.columns, f"Missing column {col} in participants.parquet")
            self.assertEqual(df[col].isna().sum(), 0, f"Null values in primary column {col}")

        # Salutation domain: 10 (female) or 20 (male)
        self.assertTrue(df["salutation"].isin([10, 20]).all(), "Invalid salutations present")

        # Plausible birth year: 1900 to 2010
        self.assertTrue((df["birth_year"] >= 1900).all() and (df["birth_year"] <= 2010).all())

        # For participants with derived age (window-eligible), verify bounds and sensitivity ordering
        age_df = df[df["age_dec1"].notna()]
        self.assertGreater(len(age_df), 0, "No participants have non-null age_dec1")
        self.assertTrue((age_df["age_dec1"] >= 0).all() and (age_df["age_dec1"] <= 120).all())
        self.assertTrue((age_df["age_jan1"] >= age_df["age_dec1"]).all())
        self.assertTrue((age_df["age_dec1"] >= age_df["age_dec31"]).all())

    def test_generated_windows_parquet_if_present(self):
        """Validate windows.parquet schema and window boundary constraints."""
        candidate_paths = [
            ARTIFACTS_DIR / "windows/windows.parquet",
            ARTIFACTS_DIR / "windows.parquet",
        ]
        w_path = next((p for p in candidate_paths if p.exists()), None)
        if w_path is None:
            self.skipTest("windows.parquet not yet generated by worker")

        df = pd.read_parquet(w_path)
        required_cols = [
            "user_id", "source", "protocol", "window_index",
            "start_day", "end_day", "num_qualified_dates", "is_eligible"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column {col} in windows.parquet")

        # Check window duration: exactly 180 days for all rows
        diffs = df["end_day"] - df["start_day"]
        self.assertTrue((diffs == 180).all(), "Window duration not exactly 180 days!")

        # Check protocols
        valid_protocols = {"expanded90", "balanced90", "balanced60"}
        self.assertTrue(df["protocol"].isin(valid_protocols).all())

        # For eligible windows: qualified dates count
        eligible = df[df["is_eligible"] == True]
        if not eligible.empty:
            for proto, min_dates in [("expanded90", 90), ("balanced90", 90), ("balanced60", 60)]:
                proto_el = eligible[eligible["protocol"] == proto]
                if not proto_el.empty:
                    self.assertTrue(
                        (proto_el["num_qualified_dates"] >= min_dates).all(),
                        f"Eligible {proto} window has < {min_dates} qualified dates",
                    )

    def test_generated_selected_dates_parquet_if_present(self):
        """Validate selected_dates.parquet: 90 dates per window, block quotas, and hashes."""
        candidate_paths = [
            ARTIFACTS_DIR / "windows/selected_dates.parquet",
            ARTIFACTS_DIR / "selected_dates.parquet",
        ]
        d_path = next((p for p in candidate_paths if p.exists()), None)
        if d_path is None:
            self.skipTest("selected_dates.parquet not yet generated by worker")

        df = pd.read_parquet(d_path)
        required_cols = [
            "user_id", "source", "protocol", "window_index",
            "selected_dates_hash", "block_index", "date", "occupied_bins"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column {col} in selected_dates.parquet")

        # Every selected date must have occupied_bins >= 120
        self.assertTrue((df["occupied_bins"] >= 120).all())

        # Block indices must be in {0, 1, 2, 3, 4, 5}
        self.assertTrue(df["block_index"].isin(range(6)).all())

        # Group by window and check date counts
        group_cols = ["user_id", "source", "protocol", "window_index"]
        grouped = df.groupby(group_cols)

        # Check a sample of windows for exact counts and block quotas
        sample_keys = list(grouped.groups.keys())[:50]
        for key in sample_keys:
            sub = grouped.get_group(key)
            proto = key[2]
            if proto in ("expanded90", "balanced90"):
                self.assertEqual(len(sub), 90, f"Window {key} has {len(sub)} dates, expected 90")
                self.assertEqual(sub["date"].nunique(), 90, f"Duplicate dates in window {key}")
                # Check block quotas
                b_counts = sub["block_index"].value_counts()
                self.assertEqual(len(b_counts), 6, f"Window {key} missing blocks")
                if proto == "balanced90":
                    self.assertTrue((b_counts == 15).all(), f"balanced90 block quota != 15: {b_counts}")
                elif proto == "expanded90":
                    self.assertTrue((b_counts >= 10).all(), f"expanded90 block quota < 10: {b_counts}")
                    self.assertEqual(b_counts.sum(), 90)

            # Check selected_dates_hash parity
            expected_hash = reference_selected_dates_hash(sorted(sub["date"].tolist()))
            actual_hash = sub["selected_dates_hash"].iloc[0]
            self.assertEqual(actual_hash, expected_hash, f"Hash mismatch in window {key}")


# -----------------------------------------------------------------------------
# Main Test Runner
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
