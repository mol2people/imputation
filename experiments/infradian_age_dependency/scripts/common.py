#!/usr/bin/env python3
"""
common.py: Shared mathematical definitions, protocols, and data structures
for the Continuous Age RCS Infradian Study.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence, Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
from numpy.random import Generator, PCG64

# Canonical repository and experiment paths
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]

ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
CACHE_5MIN_DIR = ARTIFACTS_DIR / "cache_5min"
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"
WINDOWS_DIR = ARTIFACTS_DIR / "windows"

UPSTREAM_APPLE_CACHE = REPO_ROOT / "experiments/apple_cohort_scaling/artifacts/cache_5min"
UPSTREAM_GARMIN_CACHE = REPO_ROOT / "experiments/garmin_infradian_replication/artifacts/cache_5min"
DEMOGRAPHICS_CSV = REPO_ROOT / "data/13Aug_1222.csv"
EPOCH_SOURCES_PARQUET = REPO_ROOT / "experiments/artifacts_v2/epoch_sources.parquet"
RAW_OUT_DIR = REPO_ROOT / "out"

# Landmark biological knots for Stone-Koo RCS basis (Frank Harrell RMS)
DEFAULT_KNOTS = (25.0, 42.0, 52.0, 65.0)


# =============================================================================
# 1. Restricted Cubic Spline Basis (Stone-Koo / Frank Harrell RMS)
# =============================================================================
def rcs_stone_koo(
    ages: np.ndarray | Sequence[float] | float,
    knots: Sequence[float] = DEFAULT_KNOTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs the Stone-Koo restricted cubic spline basis variables rcs1 and rcs2
    following Frank Harrell's RMS formulation:
    
      rcs_j(x) = [ (x - t_j)_+^3 - ((t4 - t_j)/(t4 - t3))*(x - t3)_+^3 + ((t3 - t_j)/(t4 - t3))*(x - t4)_+^3 ] / (t4 - t1)^2
      
    where (u)_+ = max(u, 0) and knots = [t1, t2, t3, t4].
    
    Properties:
      - For x <= t1 (25.0): rcs1 = 0.0, rcs2 = 0.0 (bit-exact zero).
      - For x >= t4 (65.0): rcs1 and rcs2 are strictly linear with constant derivatives:
          d rcs1 / dx = 2.025
          d rcs2 / dx = 0.43125
      - C^2 continuous everywhere.
    """
    t1, t2, t3, t4 = knots
    norm = float((t4 - t1) ** 2)  # (65 - 25)^2 = 1600.0
    
    x = np.asarray(ages, dtype=np.float64)
    scalar_input = x.ndim == 0
    if scalar_input:
        x = x[np.newaxis]
        
    def pos_cube(u: np.ndarray) -> np.ndarray:
        return np.maximum(u, 0.0) ** 3
    
    # rcs1 (j=1, knot t1)
    c3_1 = (t4 - t1) / (t4 - t3)  # (65 - 25)/(65 - 52) = 40/13
    c4_1 = (t3 - t1) / (t4 - t3)  # (52 - 25)/(65 - 52) = 27/13
    rcs1 = (pos_cube(x - t1) - c3_1 * pos_cube(x - t3) + c4_1 * pos_cube(x - t4)) / norm
    
    # rcs2 (j=2, knot t2)
    c3_2 = (t4 - t2) / (t4 - t3)  # (65 - 42)/(65 - 52) = 23/13
    c4_2 = (t3 - t2) / (t4 - t3)  # (52 - 42)/(65 - 52) = 10/13
    rcs2 = (pos_cube(x - t2) - c3_2 * pos_cube(x - t3) + c4_2 * pos_cube(x - t4)) / norm
    
    if scalar_input:
        return float(rcs1[0]), float(rcs2[0])
    return rcs1, rcs2


# =============================================================================
# 2. Chronological Age Derivation (December-1 Convention & Sensitivities)
# =============================================================================
def calendar_date_from_day(day_int: int) -> Tuple[int, int, int]:
    """
    Converts integer calendar days since 1970-01-01 to (year, month, day).
    """
    dt = np.datetime64("1970-01-01", "D") + np.timedelta64(int(day_int), "D")
    ts = pd.to_datetime(str(dt))
    return int(ts.year), int(ts.month), int(ts.day)


def compute_all_ages(start_day: int, birth_year: int) -> Tuple[int, int, int]:
    """
    Derives chronological age at window start day under:
      1. Primary December-1 convention:
           age_dec1 = start.year - birth_year - 1 + int(start.month == 12)
      2. Sensitivity January-1 convention:
           age_jan1 = start.year - birth_year
      3. Sensitivity December-31 convention:
           age_dec31 = start.year - birth_year - 1 + int(start.month == 12 and start.day == 31)
    """
    year, month, day = calendar_date_from_day(start_day)
    age_dec1 = year - int(birth_year) - 1 + int(month == 12)
    age_jan1 = year - int(birth_year)
    age_dec31 = year - int(birth_year) - 1 + int(month == 12 and day == 31)
    return age_dec1, age_jan1, age_dec31


# =============================================================================
# 3. Greedy Quota Allocation (expanded90 Protocol)
# =============================================================================
def greedy_quota_allocation(
    capacities: np.ndarray | Sequence[int],
    target_total: int = 90,
    base_quota: int = 10,
) -> np.ndarray:
    """
    Greedy quota allocation algorithm for expanded90:
      1. Initialize 6 block quotas: q_b = base_quota (10).
      2. Allocate remaining (target_total - 6 * base_quota = 30) dates one at a time:
         - Find eligible blocks: {b : q_b < c_b}
         - Identify minimum quota among eligible blocks: q_min = min_{b eligible} q_b
         - Select candidate with minimum quota, breaking ties by ascending block index.
         - Increment q_b.
      3. Guarantees 10 <= q_b <= c_b and sum(q_b) == target_total.
      4. If c_b >= 15 for all b, yields exactly q_b = 15 for all b (matching balanced90).
    """
    caps = np.asarray(capacities, dtype=np.int64)
    if len(caps) != 6:
        raise ValueError(f"Expected exactly 6 block capacities, got {len(caps)}")
    if np.any(caps < base_quota):
        raise ValueError(f"All block capacities must be >= {base_quota}, got {caps}")
    if int(caps.sum()) < target_total:
        raise ValueError(f"Sum of capacities ({caps.sum()}) < target_total ({target_total})")
        
    quotas = np.full(6, base_quota, dtype=np.int64)
    remaining = target_total - 6 * base_quota
    
    for _ in range(remaining):
        eligible = [b for b in range(6) if quotas[b] < caps[b]]
        min_q = min(quotas[b] for b in eligible)
        # Tie break by ascending block index
        chosen_b = next(b for b in eligible if quotas[b] == min_q)
        quotas[chosen_b] += 1
        
    return quotas


# =============================================================================
# 4. PRNG PCG64 SHA-256 Seeding and Block Sampling
# =============================================================================
def derive_block_seed(source: int, user_id: int, window_start_day: int, block: int) -> int:
    """
    Derives deterministic 64-bit unsigned integer seed from SHA-256 digest:
      tag = "20261020|infradian_age|{source}|{user_id}|{window_start_day}|dates_block_{block}"
      seed = int.from_bytes(SHA256(tag)[0:8], byteorder="little")
    """
    tag = f"20261020|infradian_age|{source}|{user_id}|{window_start_day}|dates_block_{block}"
    h = hashlib.sha256(tag.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little")


def sample_block_dates(
    block_qual_dates: np.ndarray,
    quota: int,
    seed_u64: int,
) -> np.ndarray:
    """
    Uniformly samples quota dates without replacement from sorted block qualified dates.
    Uses numpy.random.Generator(PCG64(seed_u64)) to draw a permutation, selects the
    first quota elements, and sorts the sampled dates in ascending calendar order.
    """
    c_b = len(block_qual_dates)
    if quota > c_b:
        raise ValueError(f"Quota ({quota}) cannot exceed block capacity ({c_b})")
    if quota == c_b:
        return np.sort(block_qual_dates)
        
    rng = Generator(PCG64(seed_u64))
    perm = rng.permutation(c_b)
    selected_idx = perm[:quota]
    return np.sort(block_qual_dates[selected_idx])


def compute_selected_dates_hash(sorted_selected_dates: np.ndarray) -> str:
    """
    Computes SHA-256 hash of the 90 sorted selected dates serialized as
    little-endian signed int64 calendar days.
    """
    dates_int64 = np.asarray(sorted_selected_dates, dtype="<i8")
    return hashlib.sha256(dates_int64.tobytes()).hexdigest()


# =============================================================================
# 5. Window Selection Search
# =============================================================================
def find_qualifying_windows(
    sorted_qual_dates: np.ndarray,
    protocol: str = "expanded90",
) -> Tuple[Optional[int], Optional[int]]:
    """
    Searches candidate starts s in sorted_qual_dates to find:
      s1: earliest qualifying window start in [s, s+180)
      s2: earliest qualifying window start with s2 >= s1 + 180 (repeatability)
      
    Protocols:
      - expanded90: each block c_b >= 10, total window sum(c_b) >= 90
      - balanced90: each block c_b >= 15
      - balanced60: each block c_b >= 10
    """
    if len(sorted_qual_dates) == 0:
        return None, None
        
    min_dates_needed = 90 if protocol in ("expanded90", "balanced90") else 60
    if len(sorted_qual_dates) < min_dates_needed:
        return None, None
        
    first_start = None
    second_start = None
    max_date = sorted_qual_dates[-1]
    
    for s in sorted_qual_dates:
        # If remaining date span is less than 150 days from s, block 5 cannot have dates
        if max_date < s + 150:
            break
            
        capacities = np.empty(6, dtype=np.int64)
        for b in range(6):
            b_s = s + 30 * b
            b_e = s + 30 * (b + 1)
            c_b = np.searchsorted(sorted_qual_dates, b_e) - np.searchsorted(sorted_qual_dates, b_s)
            capacities[b] = c_b
            
        qualifies = False
        if protocol == "expanded90":
            if (capacities >= 10).all() and capacities.sum() >= 90:
                qualifies = True
        elif protocol == "balanced90":
            if (capacities >= 15).all():
                qualifies = True
        elif protocol == "balanced60":
            if (capacities >= 10).all():
                qualifies = True
                
        if qualifies:
            if first_start is None:
                first_start = int(s)
            elif s >= first_start + 180:
                second_start = int(s)
                break
                
    return first_start, second_start


def get_window_block_capacities(sorted_qual_dates: np.ndarray, start_day: int) -> np.ndarray:
    """
    Returns array of 6 block capacities for window [start_day, start_day + 180).
    """
    capacities = np.empty(6, dtype=np.int64)
    for b in range(6):
        b_s = start_day + 30 * b
        b_e = start_day + 30 * (b + 1)
        c_b = np.searchsorted(sorted_qual_dates, b_e) - np.searchsorted(sorted_qual_dates, b_s)
        capacities[b] = c_b
    return capacities
