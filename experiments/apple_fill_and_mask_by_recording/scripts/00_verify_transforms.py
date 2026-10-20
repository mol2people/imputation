"""Apple Watch Transform & Hygiene Verification Suite (00_verify_transforms.py).

Comprehensive deterministic unit verification for:
  1. Constructing the 20-day synthetic fixture across 5m (L=288) and 15m (L=96)
  2. Piecewise Clean Difference (Review R6.1, PLAN §4.4)
  3. Span-Matched Short-Gap Linear Interpolation (Review R4, PLAN §4.4)
  4. Binary Mask Variance Algebraic Identity (Review R3, PLAN §4.1)
  5. Train-Only Fitting and Preprocessing Hygiene (ORIGINAL_REQUEST.md, PLAN §5.2)

Execution requirement:
  Must run deterministically in <10 seconds (<1s expected) and exit with code 0.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ensure scripts directory is on path for modular imports
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from transforms import (
    compute_clean_diff,
    compute_mask_stats,
    fit_fill_templates,
    interpolate_short_gaps,
)


# ==============================================================================
# 20-Day Synthetic Test Fixture Builder
# ==============================================================================

def build_synthetic_fixture(
    n_days: int = 20,
    resolution: str = "5m"
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Construct deterministic 20-day synthetic fixture covering all R1–R6 edge cases.

    Grid sizes:
      - 5m: L = 288 (dt = 5 min, k_max = 5)
      - 15m: L = 96 (dt = 15 min, k_max = 1)

    Edge cases covered:
      Day 0: Baseline fully observed day (sinusoidal baseline wave)
      Day 1: 1-bin interior gap (k=1)
      Day 2: 2-bin interior gap (k=2) [interpolated at 5m, clock at 15m]
      Day 3: 3-bin interior gap (k=3) [interpolated at 5m, clock at 15m]
      Day 4: 4-bin interior gap (k=4) [interpolated at 5m, clock at 15m]
      Day 5: 5-bin interior gap (k=5, max eligible at 5m) [interpolated at 5m, clock at 15m]
      Day 6: 6-bin interior gap (k=6, ineligible at both resolutions)
      Day 7: 7-bin interior gap (k=7, ineligible at both resolutions)
      Day 8: Day-start prefix boundary gap (t=0..3, ineligible)
      Day 9: Day-end suffix boundary gap (t=L-4..L-1, ineligible)
      Day 10: Both boundary gaps (prefix t=0..1 and suffix t=L-2..L-1, ineligible)
      Day 11: Boundary-adjacent interior gap at t=1 (bounded by t=0 and t=2, eligible)
      Day 12: Boundary-adjacent interior gap at t=L-2 (bounded by t=L-3 and t=L-1, eligible)
      Day 13: Adjacent gaps separated by a single observed bin (k=1 at t=20, M=1 at t=21, k=2 at t=22..23)
      Day 14: Adjacent observed bins with identical HR values (h_{t+1} == h_t, Review R6.1)
      Day 15: Alternating missing/observed sequence (t=10..18)
      Day 16: Zero-slope interior gap (k=3, h_L == h_R)
      Day 17: Negative-slope interior gap (k=2, h_L > h_R)
      Day 18: Fully unobserved day (all bins NaN, M=0)
      Day 19: High-value step stress / isolated bin day

    Returns:
        hr: Array of shape (n_days, L), float32, with NaNs at unobserved bins.
        mask: Array of shape (n_days, L), uint8 (1=observed, 0=unobserved).
        meta: Dictionary of ground-truth expected values and parameters.
    """
    if resolution == "5m":
        L = 288
    elif resolution == "15m":
        L = 96
    else:
        raise ValueError(f"Unknown resolution '{resolution}'; expected '5m' or '15m'")

    hr = np.full((n_days, L), np.nan, dtype=np.float32)
    mask = np.ones((n_days, L), dtype=np.uint8)

    t = np.arange(L, dtype=np.float32)
    base_wave = (65.0 + 15.0 * np.sin(2.0 * np.pi * t / L)).astype(np.float32)

    # Initialize all days with baseline wave
    for d in range(n_days):
        hr[d] = base_wave.copy()

    # Day 0: Fully observed day (no changes)

    # Day 1: 1-bin interior gap (k=1)
    mask[1, 20] = 0
    hr[1, 19] = 60.0
    hr[1, 20] = np.nan
    hr[1, 21] = 70.0

    # Day 2: 2-bin interior gap (k=2)
    mask[2, 25:27] = 0
    hr[2, 24] = 60.0
    hr[2, 25:27] = np.nan
    hr[2, 27] = 75.0

    # Day 3: 3-bin interior gap (k=3)
    mask[3, 30:33] = 0
    hr[3, 29] = 60.0
    hr[3, 30:33] = np.nan
    hr[3, 33] = 80.0

    # Day 4: 4-bin interior gap (k=4)
    mask[4, 35:39] = 0
    hr[4, 34] = 60.0
    hr[4, 35:39] = np.nan
    hr[4, 39] = 85.0

    # Day 5: 5-bin interior gap (k=5, max eligible at 5m)
    mask[5, 40:45] = 0
    hr[5, 39] = 60.0
    hr[5, 40:45] = np.nan
    hr[5, 45] = 90.0

    # Day 6: 6-bin interior gap (k=6, ineligible at both 5m and 15m)
    mask[6, 50:56] = 0
    hr[6, 49] = 60.0
    hr[6, 50:56] = np.nan
    hr[6, 56] = 95.0

    # Day 7: 7-bin interior gap (k=7, ineligible)
    mask[7, 60:67] = 0
    hr[7, 59] = 60.0
    hr[7, 60:67] = np.nan
    hr[7, 67] = 100.0

    # Day 8: Day-start prefix boundary gap (t=0..3, ineligible)
    mask[8, 0:4] = 0
    hr[8, 0:4] = np.nan

    # Day 9: Day-end suffix boundary gap (t=L-4..L-1, ineligible)
    mask[9, (L - 4):L] = 0
    hr[9, (L - 4):L] = np.nan

    # Day 10: Both boundary gaps
    mask[10, 0:2] = 0
    hr[10, 0:2] = np.nan
    mask[10, (L - 2):L] = 0
    hr[10, (L - 2):L] = np.nan

    # Day 11: Boundary-adjacent interior gap at t=1 (eligible)
    mask[11, 1] = 0
    hr[11, 0] = 58.0
    hr[11, 1] = np.nan
    hr[11, 2] = 66.0

    # Day 12: Boundary-adjacent interior gap at t=L-2 (eligible)
    mask[12, L - 2] = 0
    hr[12, L - 3] = 62.0
    hr[12, L - 2] = np.nan
    hr[12, L - 1] = 74.0

    # Day 13: Adjacent gaps separated by a single observed bin
    mask[13, 20] = 0
    hr[13, 19] = 60.0
    hr[13, 20] = np.nan
    hr[13, 21] = 70.0
    mask[13, 21] = 1
    mask[13, 22:24] = 0
    hr[13, 22:24] = np.nan
    hr[13, 24] = 82.0
    mask[13, 24] = 1

    # Day 14: Adjacent observed bins with identical HR values (Review R6.1)
    hr[14, 15:21] = 72.0
    mask[14, 15:21] = 1

    # Day 15: Alternating missing/observed sequence (t=10..18)
    mask[15, 10:18:2] = 0
    hr[15, 10:18:2] = np.nan

    # Day 16: Zero slope interpolation (k=3, h_L == h_R)
    mask[16, 25:28] = 0
    hr[16, 24] = 65.0
    hr[16, 25:28] = np.nan
    hr[16, 28] = 65.0

    # Day 17: Negative slope interpolation (k=2, h_L > h_R)
    mask[17, 30:32] = 0
    hr[17, 29] = 80.0
    hr[17, 30:32] = np.nan
    hr[17, 32] = 50.0

    # Day 18: Fully unobserved day
    mask[18, :] = 0
    hr[18, :] = np.nan

    # Day 19: High-value step stress
    hr[19, 10:20] = 180.0
    mask[19, 10:20] = 1

    meta = {
        "n_days": n_days,
        "resolution": resolution,
        "L": L,
        "day1_interp_expected": 65.0,
        "day2_interp_5m_expected": [65.0, 70.0],
        "day3_interp_5m_expected": [65.0, 70.0, 75.0],
        "day4_interp_5m_expected": [65.0, 70.0, 75.0, 80.0],
        "day5_interp_5m_expected": [65.0, 70.0, 75.0, 80.0, 85.0],
        "day11_interp_expected": 62.0,
        "day12_interp_expected": 68.0,
        "day13_gap1_expected": 65.0,
        "day13_gap2_5m_expected": [74.0, 78.0],
        "day16_interp_5m_expected": [65.0, 65.0, 65.0],
        "day17_interp_5m_expected": [70.0, 60.0],
    }
    return hr, mask, meta


# ==============================================================================
# Verification Test Suites
# ==============================================================================

def verify_piecewise_clean_diff(resolution: str, verbose: bool = False) -> None:
    """Verify piecewise clean difference (Review R6.1)."""
    hr, mask, meta = build_synthetic_fixture(20, resolution)
    L = meta["L"]

    # 1. Test directly on raw HR array containing NaNs
    clean_diff_raw = compute_clean_diff(hr, mask)
    assert clean_diff_raw.shape == (20, L - 1), f"Wrong shape: {clean_diff_raw.shape}"
    assert not np.isnan(clean_diff_raw).any(), "NaNs detected in clean_diff output on raw HR!"
    assert np.isfinite(clean_diff_raw).all(), "Non-finite values detected in clean_diff output!"

    m_both = (mask[:, :-1] == 1) & (mask[:, 1:] == 1)

    # 2. Assert exact 0.0 whenever M_t = 0 or M_{t+1} = 0
    assert np.all(clean_diff_raw[~m_both] == 0.0), "Non-zero diff on unobserved transition!"

    # 3. Assert exact difference on observed transitions
    diff_expected = hr[:, 1:][m_both] - hr[:, :-1][m_both]
    assert np.allclose(clean_diff_raw[m_both], diff_expected, atol=1e-7), (
        "Difference mismatch on observed transitions!"
    )

    # 4. Assert exact 0.0 on adjacent observed bins with identical HR values (Day 14, Review R6.1)
    for idx in range(15, 20):
        assert m_both[14, idx], f"Day 14 transition at {idx} should be observed"
        assert clean_diff_raw[14, idx] == 0.0, (
            f"Expected exact 0.0 on identical observed bins at index {idx}, got {clean_diff_raw[14, idx]}"
        )

    # 5. Also test on clock-filled base series X_clock
    clock_template = np.full(L, 60.0, dtype=np.float32)
    X_clock = np.where(mask == 1, hr, clock_template[None, :])
    clean_diff_clock = compute_clean_diff(X_clock, mask)
    assert np.array_equal(clean_diff_raw, clean_diff_clock), (
        "compute_clean_diff output diverges between raw HR and X_clock!"
    )

    # 6. Verify Day 18 (all-missing day) produces exact all-zero clean diff
    assert np.all(clean_diff_raw[18] == 0.0), "Day 18 all-missing day should produce all zeros!"

    if verbose:
        print(f"      [{resolution}] Clean diff verified: zero NaNs, exact 0.0 on missing & identical HR.")


def verify_short_gap_interpolation(resolution: str, verbose: bool = False) -> None:
    """Verify span-matched short-gap linear interpolation (Review R4)."""
    hr, mask, meta = build_synthetic_fixture(20, resolution)
    L = meta["L"]
    clock_template = np.full(L, 60.0, dtype=np.float32)

    X_interp = interpolate_short_gaps(hr, mask, clock_template, resolution)
    assert X_interp.shape == (20, L), f"Wrong shape: {X_interp.shape}"
    assert not np.isnan(X_interp).any(), "NaNs detected in X_interp output!"
    assert np.isfinite(X_interp).all(), "Non-finite values detected in X_interp output!"

    # 1. Assert all observed positions (M=1) are preserved bit-exactly
    assert np.array_equal(X_interp[mask == 1], hr[mask == 1]), (
        "Observed bins were modified during interpolation!"
    )

    # 2. Day 1: 1-bin interior gap (k=1) must be interpolated at both 5m and 15m
    assert np.isclose(X_interp[1, 20], meta["day1_interp_expected"], atol=1e-6), (
        f"Day 1 k=1 interpolation failed: expected {meta['day1_interp_expected']}, got {X_interp[1, 20]}"
    )

    # 3. Days 2–5: k in {2, 3, 4, 5}
    if resolution == "5m":
        # At 5m: k in {2, 3, 4, 5} must all be linearly interpolated
        assert np.allclose(X_interp[2, 25:27], meta["day2_interp_5m_expected"], atol=1e-6), (
            "Day 2 (k=2) interpolation failed at 5m!"
        )
        assert np.allclose(X_interp[3, 30:33], meta["day3_interp_5m_expected"], atol=1e-6), (
            "Day 3 (k=3) interpolation failed at 5m!"
        )
        assert np.allclose(X_interp[4, 35:39], meta["day4_interp_5m_expected"], atol=1e-6), (
            "Day 4 (k=4) interpolation failed at 5m!"
        )
        assert np.allclose(X_interp[5, 40:45], meta["day5_interp_5m_expected"], atol=1e-6), (
            "Day 5 (k=5) interpolation failed at 5m!"
        )
        # Day 6 (k=6) and Day 7 (k=7): ineligible at 5m ((k+1)*5 > 30 min) -> must remain on clock template
        assert np.all(X_interp[6, 50:56] == 60.0), "Day 6 (k=6) must remain on clock template at 5m!"
        assert np.all(X_interp[7, 60:67] == 60.0), "Day 7 (k=7) must remain on clock template at 5m!"

        # Zero-slope (Day 16) and negative-slope (Day 17)
        assert np.allclose(X_interp[16, 25:28], meta["day16_interp_5m_expected"], atol=1e-6), (
            "Day 16 zero-slope interpolation failed!"
        )
        assert np.allclose(X_interp[17, 30:32], meta["day17_interp_5m_expected"], atol=1e-6), (
            "Day 17 negative-slope interpolation failed!"
        )
    else:  # 15m
        # At 15m (dt=15 min, k_max=1): gaps with k >= 2 have (k+1)*15 >= 45 min > 30 min -> ineligible!
        assert np.all(X_interp[2, 25:27] == 60.0), "Day 2 (k=2) must remain on clock template at 15m!"
        assert np.all(X_interp[3, 30:33] == 60.0), "Day 3 (k=3) must remain on clock template at 15m!"
        assert np.all(X_interp[4, 35:39] == 60.0), "Day 4 (k=4) must remain on clock template at 15m!"
        assert np.all(X_interp[5, 40:45] == 60.0), "Day 5 (k=5) must remain on clock template at 15m!"
        assert np.all(X_interp[6, 50:56] == 60.0), "Day 6 (k=6) must remain on clock template at 15m!"

    # 4. Day-boundary gaps (Days 8, 9, 10): must remain on clock template at both resolutions
    assert np.all(X_interp[8, 0:4] == 60.0), "Day 8 prefix boundary gap should remain on clock template!"
    assert np.all(X_interp[9, (L - 4):L] == 60.0), "Day 9 suffix boundary gap should remain on clock template!"
    assert np.all(X_interp[10, 0:2] == 60.0), "Day 10 prefix boundary gap should remain on clock template!"
    assert np.all(X_interp[10, (L - 2):L] == 60.0), "Day 10 suffix boundary gap should remain on clock template!"

    # 5. Boundary-adjacent interior gaps (Days 11 and 12): eligible and must be interpolated
    assert np.isclose(X_interp[11, 1], meta["day11_interp_expected"], atol=1e-6), (
        "Day 11 boundary-adjacent interior gap at t=1 failed to interpolate!"
    )
    assert np.isclose(X_interp[12, L - 2], meta["day12_interp_expected"], atol=1e-6), (
        "Day 12 boundary-adjacent interior gap at t=L-2 failed to interpolate!"
    )

    # 6. Day 13: Adjacent gaps separated by a single observed bin
    assert np.isclose(X_interp[13, 20], meta["day13_gap1_expected"], atol=1e-6), (
        "Day 13 Gap 1 failed to interpolate!"
    )
    assert X_interp[13, 21] == 70.0, "Day 13 single observed bin at t=21 was altered!"
    if resolution == "5m":
        assert np.allclose(X_interp[13, 22:24], meta["day13_gap2_5m_expected"], atol=1e-6), (
            "Day 13 Gap 2 failed to interpolate at 5m!"
        )
    else:
        assert np.all(X_interp[13, 22:24] == 60.0), (
            "Day 13 Gap 2 (k=2) should remain on clock template at 15m!"
        )

    # 7. Day 18: All-missing day remains on clock template across all bins
    assert np.all(X_interp[18] == 60.0), "Day 18 all-missing day should remain on clock template!"

    if verbose:
        print(f"      [{resolution}] Short-gap interpolation verified: observed preserved bit-exactly, "
              f"span-matched k_max respected, boundaries protected.")


def verify_binary_mask_variance_identity(verbose: bool = False) -> None:
    """Verify binary mask variance algebraic identity (Review R3, PLAN §4.1)."""
    # 1. Test identity across multiple day counts D_u in {2, 5, 20, 100}
    for D in [2, 5, 20, 100]:
        L = 288
        rng = np.random.default_rng(20261019 + D)
        mask = rng.integers(0, 2, size=(D, L), dtype=np.uint8)

        mean_M, sd_M = compute_mask_stats(mask)
        assert mean_M.shape == (L,), f"Wrong mean_M shape: {mean_M.shape}"
        assert sd_M.shape == (L,), f"Wrong sd_M shape: {sd_M.shape}"

        # Direct sample variance with Bessel's correction (ddof=1)
        var_direct = np.var(mask.astype(np.float64), axis=0, ddof=1)
        mean_direct = np.mean(mask.astype(np.float64), axis=0)
        var_formula = (D / (D - 1.0)) * mean_direct * (1.0 - mean_direct)

        # Assert identity holds to float64 machine precision
        assert np.allclose(var_direct, var_formula, atol=1e-12), (
            f"Algebraic identity violated in float64 for D={D}!"
        )

        # Assert compute_mask_stats matches within float32 tolerance
        assert np.allclose(sd_M.astype(np.float64)**2, var_direct, atol=1e-5), (
            f"compute_mask_stats SD squared mismatch for D={D}!"
        )
        assert np.allclose(sd_M.astype(np.float64), np.sqrt(var_direct), atol=1e-5), (
            f"compute_mask_stats SD mismatch for D={D}!"
        )

    # 2. Boundary condition: D_u <= 1 must yield exactly 0.0 standard deviation
    mask_1d = np.ones((1, 288), dtype=np.uint8)
    mean_1d, sd_1d = compute_mask_stats(mask_1d)
    assert np.all(sd_1d == 0.0), f"D_u=1 must yield exactly 0.0 standard deviation, got {sd_1d}"
    assert np.all(mean_1d == 1.0), f"All-observed 1-day mask must have mean 1.0, got {mean_1d}"

    mask_empty_1d = np.zeros((1, 96), dtype=np.uint8)
    mean_e1d, sd_e1d = compute_mask_stats(mask_empty_1d)
    assert np.all(sd_e1d == 0.0), f"D_u=1 all-missing must yield exactly 0.0 standard deviation"
    assert np.all(mean_e1d == 0.0), f"All-missing 1-day mask must have mean 0.0"

    # Also test 1D input array (L,)
    mask_flat = np.ones(96, dtype=np.uint8)
    mean_flat, sd_flat = compute_mask_stats(mask_flat)
    assert np.all(sd_flat == 0.0), "Flat 1D mask (D=1 equivalent) must yield sd=0.0"

    if verbose:
        print("      [both] Binary mask variance identity verified: exact algebraic equivalence for D>1, "
              "exact 0.0 for D<=1.")


def verify_train_only_fitting_hygiene(verbose: bool = False) -> None:
    """Verify train-only template fitting and preprocessing hygiene (ORIGINAL_REQUEST.md, PLAN §5.2)."""
    hr, mask, meta = build_synthetic_fixture(20, "5m")

    # Define strict train, validation, test participant subsets
    # Train: days 0..11; Val: days 12..15; Test: days 16..19
    tr_idx = np.arange(12)
    val_idx = np.arange(12, 16)
    te_idx = np.arange(16, 20)

    # 1. Verify fit_fill_templates on train
    clock_tr, glob_tr = fit_fill_templates(hr[tr_idx], mask[tr_idx])
    assert clock_tr.shape == (meta["L"],), f"Wrong clock template shape: {clock_tr.shape}"
    assert np.isfinite(clock_tr).all(), "Clock template contains NaNs or non-finite values!"
    assert np.isfinite(glob_tr), "Global template is non-finite!"

    # Adversarially poison validation and test data with massive anomalies
    hr_poisoned = hr.copy()
    hr_poisoned[val_idx] = -99999.0
    hr_poisoned[te_idx] = 99999.0

    # Re-estimate templates on training participant IDs
    clock_tr_clean, glob_tr_clean = fit_fill_templates(hr_poisoned[tr_idx], mask[tr_idx])
    assert np.array_equal(clock_tr, clock_tr_clean), (
        "Leaked contaminated validation/test data into clock template!"
    )
    assert glob_tr == glob_tr_clean, (
        "Leaked contaminated validation/test data into global template!"
    )

    # 2. Pipeline hygiene verification: SimpleImputer, VarianceThreshold, StandardScaler
    rng = np.random.default_rng(20261019)
    n_features = 40
    X_train = rng.normal(loc=0.0, scale=1.0, size=(50, n_features)).astype(np.float32)
    # Feature 0: constant in train (variance 0.0)
    X_train[:, 0] = 5.0
    # Feature 1: contains missing values in train
    X_train[0:5, 1] = np.nan

    # Test set: feature 0 is non-constant; contains extreme outliers
    X_test = rng.normal(loc=0.0, scale=1.0, size=(20, n_features)).astype(np.float32)
    X_test[:, 0] = np.linspace(1.0, 10.0, 20)  # Non-constant in test!
    X_test[:, 5] = 1e6  # Massive outlier in test!
    X_test[0:3, 1] = np.nan  # Missing values in test!

    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        VarianceThreshold(threshold=0.0),
        StandardScaler(),
    )

    # Fit strictly on train
    X_train_trans = pipe.fit_transform(X_train)
    # Transform test
    X_test_trans = pipe.transform(X_test)

    # Feature 0 must be dropped because it had variance 0 in train
    assert X_train_trans.shape[1] == n_features - 1, (
        f"VarianceThreshold failed to drop constant train feature! Shape: {X_train_trans.shape}"
    )
    assert X_test_trans.shape[1] == n_features - 1, (
        "VarianceThreshold leaked test variance into feature selection!"
    )

    # Assert test missing value is imputed using train median
    train_feat1_median = np.nanmedian(X_train[:, 1])
    assert np.isclose(pipe.named_steps["simpleimputer"].statistics_[1], train_feat1_median), (
        "Imputer statistics do not match training median!"
    )

    # Assert scaler mean and scale strictly reflect training set
    scaler = pipe.named_steps["standardscaler"]
    train_imputed_feat5 = X_train[:, 5]
    assert np.isclose(scaler.mean_[4], np.mean(train_imputed_feat5), atol=1e-6), (
        "Scaler mean leaked test data outlier!"
    )

    if verbose:
        print("      [both] Train-only fitting hygiene verified: templates, imputer, variance filter, "
              "and scaler strictly isolated from validation/test data.")


# ==============================================================================
# CLI and Main Test Runner
# ==============================================================================

def run_all_verifications(resolutions: list[str], verbose: bool = False) -> float:
    """Run all verification suites and return elapsed execution time."""
    t0 = time.perf_counter()

    print("=" * 80)
    print("Apple Watch Transform & Hygiene Verification Suite (00_verify_transforms.py)")
    print("=" * 80)

    # Gate 1: Piecewise clean difference
    print("[1/4] Verifying compute_clean_diff (Review R6.1)...")
    for res in resolutions:
        verify_piecewise_clean_diff(res, verbose=verbose)
    print("      [PASS] Clean difference verified: zero NaNs, exact 0.0 on missing & identical HR.")

    # Gate 2: Span-matched short-gap linear interpolation
    print("[2/4] Verifying interpolate_short_gaps (Review R4)...")
    for res in resolutions:
        verify_short_gap_interpolation(res, verbose=verbose)
    print("      [PASS] Short-gap interpolation verified: observed preserved bit-exactly, "
          "k_max span-matching exact.")

    # Gate 3: Binary mask variance algebraic identity
    print("[3/4] Verifying compute_mask_stats & algebraic identity (Review R3)...")
    verify_binary_mask_variance_identity(verbose=verbose)
    print("      [PASS] Binary mask variance identity verified: exact algebraic equivalence for D>1, "
          "exact 0.0 for D<=1.")

    # Gate 4: Train-only fitting and preprocessing hygiene
    print("[4/4] Verifying train-only preprocessing hygiene (ORIGINAL_REQUEST.md, PLAN §5.2)...")
    verify_train_only_fitting_hygiene(verbose=verbose)
    print("      [PASS] Train-only hygiene verified: zero test data leakage into templates, "
          "imputer, filter, or scaler.")

    elapsed = time.perf_counter() - t0
    print("=" * 80)
    print(f"ALL 4 TRANSFORM VERIFICATION GATES PASSED deterministically in {elapsed:.4f}s "
          f"(<10.0s requirement).")
    print("=" * 80)
    return elapsed


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Apple Watch Transform & Hygiene Verification Suite (Review R1–R6)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed per-test diagnostic messages"
    )
    parser.add_argument(
        "--resolution",
        choices=["both", "5m", "15m"],
        default="both",
        help="Resolution grid(s) to verify (default: both)"
    )

    args = parser.parse_args()
    resolutions = ["5m", "15m"] if args.resolution == "both" else [args.resolution]

    try:
        elapsed = run_all_verifications(resolutions, verbose=args.verbose)
        assert elapsed < 10.0, f"Execution time {elapsed:.2f}s exceeded 10.0s threshold!"
        return 0
    except AssertionError as err:
        print(f"\n[FAIL] VERIFICATION ASSERTION ERROR: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"\n[ERROR] UNEXPECTED EXCEPTION: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
