"""Fast unit verification fixture for circular diurnal template regularization.

Tests:
  1. Midnight wrap-around boundary conditions:
     - Circular symmetry around bin 0 (midnight)
     - Exact translation invariance: roll(smooth(delta_0), 144) == smooth(delta_144)
     - Straddled midnight pulse parity with midday pulse
     - Zero boundary flattening, reflection, or truncation distortion
  2. Mean and total energy conservation:
     - Exact sum conservation (|sum(smoothed) - sum(raw)| <= 1e-10 in float64)
     - Exact mean conservation (|mean(smoothed) - mean(raw)| <= 1e-12)
     - Tested on sinusoids, realistic circadian multi-harmonics, and random profiles
  3. Smoothing bandwidth behavior (sigma in {1, 2, 4} bins):
     - Monotonic variance reduction: Var(x) > Var(s1) > Var(s2) > Var(s4)
     - Monotonic peak attenuation: max(x) > max(s1) > max(s2) > max(s4)
     - Monotonic total variation (roughness) reduction
     - Identity mapping at sigma = 0
  4. Missing bin / NaN handling:
     - Normalized circular convolution smoothly interpolates across missing runs
     - Seamless interpolation across midnight gap (bins 285..287, 0..2)
     - All-NaN array handled gracefully without exception
     - 2D batch vectorized consistency with 1D
  5. Overall runtime check: execution completes in < 2.0 seconds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Add scripts directory to path for clean import
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from circular_smoothing import (
    circular_gaussian_smooth,
    discrete_circular_gaussian_kernel,
    smooth_diurnal_template,
)


def test_midnight_wraparound_symmetry() -> None:
    """Verify circular boundary conditions across midnight (bins 287 -> 0)."""
    L = 288

    # 1. Delta pulse at midnight (bin 0)
    delta_0 = np.zeros(L, dtype=np.float64)
    delta_0[0] = 100.0

    for sigma in [1.0, 2.0, 4.0]:
        s_0 = circular_gaussian_smooth(delta_0, sigma=sigma)

        # Symmetry test: s_0[k] must equal s_0[(L - k) % L]
        for k in range(1, L // 2):
            left_val = s_0[L - k]   # 287, 286, ... (pre-midnight)
            right_val = s_0[k]       # 1, 2, ... (post-midnight)
            assert np.isclose(left_val, right_val, atol=1e-14), (
                f"Asymmetry at offset k={k} for sigma={sigma}: {left_val} != {right_val}"
            )

    # 2. Translation invariance: delta at midnight vs delta at midday (bin 144)
    delta_144 = np.zeros(L, dtype=np.float64)
    delta_144[144] = 100.0

    for sigma in [1.0, 2.0, 4.0]:
        s_0 = circular_gaussian_smooth(delta_0, sigma=sigma)
        s_144 = circular_gaussian_smooth(delta_144, sigma=sigma)

        # Rolling s_0 by 144 bins must match s_144 bit-for-bit
        s_0_rolled = np.roll(s_0, 144)
        max_diff = np.max(np.abs(s_0_rolled - s_144))
        assert max_diff <= 1e-13, (
            f"Translation invariance violated for sigma={sigma}: max_diff={max_diff:.2e}"
        )

    # 3. Straddled midnight pulse vs midday pulse
    # Midnight pulse: bins 287 and 0
    straddle_mid = np.zeros(L, dtype=np.float64)
    straddle_mid[287] = 50.0
    straddle_mid[0] = 50.0

    # Midday pulse: bins 143 and 144
    straddle_day = np.zeros(L, dtype=np.float64)
    straddle_day[143] = 50.0
    straddle_day[144] = 50.0

    for sigma in [1.0, 2.0, 4.0]:
        s_mid = circular_gaussian_smooth(straddle_mid, sigma=sigma)
        s_day = circular_gaussian_smooth(straddle_day, sigma=sigma)

        # s_day rolled by 144 should equal s_mid
        s_day_rolled = np.roll(s_day, -144)
        max_diff = np.max(np.abs(s_day_rolled - s_mid))
        assert max_diff <= 1e-13, (
            f"Straddled pulse invariance violated for sigma={sigma}: max_diff={max_diff:.2e}"
        )

    # 4. Zero boundary distortion check (distinguish wrap from reflect/nearest)
    # A step straddling midnight: bins 284..287 and 0..3 have value 80.0, rest 60.0
    step_profile = np.full(L, 60.0, dtype=np.float64)
    step_profile[284:] = 80.0
    step_profile[:4] = 80.0

    step_midday = np.roll(step_profile, 144)

    for sigma in [1.0, 2.0, 4.0]:
        s_step_mid = circular_gaussian_smooth(step_profile, sigma=sigma)
        s_step_day = circular_gaussian_smooth(step_midday, sigma=sigma)

        diff = np.max(np.abs(np.roll(s_step_day, -144) - s_step_mid))
        assert diff <= 1e-13, f"Step wrap-around distortion detected: diff={diff:.2e}"


def test_mean_and_energy_conservation() -> None:
    """Verify strict preservation of profile mean and sum (|error| <= 1e-10 in float64)."""
    L = 288
    t = np.arange(L, dtype=np.float64)

    test_profiles = []

    # 1. Pure sinusoid
    test_profiles.append(("pure_sine", 70.0 + 15.0 * np.sin(2.0 * np.pi * t / L)))

    # 2. Multi-harmonic circadian curve (fundamental + 12h + 8h harmonics + sharp dip)
    circadian = (
        65.0
        + 12.0 * np.cos(2.0 * np.pi * (t - 180.0) / L)
        - 4.0 * np.cos(4.0 * np.pi * (t - 60.0) / L)
        + 2.0 * np.sin(6.0 * np.pi * t / L)
    )
    test_profiles.append(("multi_harmonic", circadian))

    # 3. Realistic resting HR diurnal curve with nocturnal dip (bins 0-72) and daytime rise
    nocturnal_mask = (t < 72) | (t >= 270)
    realistic_hr = np.where(nocturnal_mask, 58.0 + 3.0 * np.sin(t / 10.0), 75.0 + 8.0 * np.cos(t / 20.0))
    test_profiles.append(("realistic_hr", realistic_hr))

    # 4. Noisy profile (circadian + random Gaussian noise)
    rng = np.random.default_rng(20261020)
    noisy_hr = circadian + rng.normal(0.0, 5.0, size=L)
    test_profiles.append(("noisy_hr", noisy_hr))

    for name, profile in test_profiles:
        orig_sum = np.sum(profile)
        orig_mean = np.mean(profile)

        for sigma in [1.0, 2.0, 4.0]:
            smoothed = circular_gaussian_smooth(profile, sigma=sigma)

            smoothed_sum = np.sum(smoothed)
            smoothed_mean = np.mean(smoothed)

            sum_err = abs(smoothed_sum - orig_sum)
            mean_err = abs(smoothed_mean - orig_mean)

            assert sum_err <= 1e-10, (
                f"[{name}, sigma={sigma}] Sum conservation violated: err={sum_err:.2e} (orig={orig_sum}, smooth={smoothed_sum})"
            )
            assert mean_err <= 1e-12, (
                f"[{name}, sigma={sigma}] Mean conservation violated: err={mean_err:.2e} (orig={orig_mean}, smooth={smoothed_mean})"
            )

    # 5. Batch of 100 random profiles (2D array conservation)
    rng = np.random.default_rng(42)
    batch = rng.uniform(50.0, 100.0, size=(100, L))
    batch_sums_orig = np.sum(batch, axis=1)

    for sigma in [1.0, 2.0, 4.0]:
        smoothed_batch = circular_gaussian_smooth(batch, sigma=sigma, axis=-1)
        batch_sums_smooth = np.sum(smoothed_batch, axis=1)
        max_batch_err = np.max(np.abs(batch_sums_smooth - batch_sums_orig))
        assert max_batch_err <= 1e-10, (
            f"2D batch sum conservation violated: max_err={max_batch_err:.2e}"
        )


def test_bandwidth_scaling_monotonicity() -> None:
    """Verify monotonic variance reduction and peak attenuation for sigma in {1, 2, 4}."""
    L = 288
    t = np.arange(L, dtype=np.float64)

    # Synthetic realistic noisy diurnal curve
    rng = np.random.default_rng(12345)
    base_signal = 70.0 + 15.0 * np.sin(2.0 * np.pi * t / L) + rng.normal(0.0, 4.0, size=L)

    sigmas = [0.0, 1.0, 2.0, 4.0]
    smoothed_series = [circular_gaussian_smooth(base_signal, sigma=s) for s in sigmas]

    # 1. Identity at sigma = 0
    assert np.array_equal(smoothed_series[0], base_signal), "sigma=0 must be exact identity"

    # 2. Monotonic variance reduction
    variances = [float(np.var(s)) for s in smoothed_series]
    for i in range(len(variances) - 1):
        assert variances[i] > variances[i + 1], (
            f"Variance not strictly decreasing: sigma {sigmas[i]} var={variances[i]:.4f} "
            f"<= sigma {sigmas[i+1]} var={variances[i+1]:.4f}"
        )

    # 3. Monotonic peak attenuation (max decreases, min increases, peak-to-peak decreases)
    ptps = [float(np.ptp(s)) for s in smoothed_series]
    for i in range(len(ptps) - 1):
        assert ptps[i] > ptps[i + 1], (
            f"Peak-to-peak amplitude not strictly decreasing: sigma {sigmas[i]} ptp={ptps[i]:.4f} "
            f"<= sigma {sigmas[i+1]} ptp={ptps[i+1]:.4f}"
        )

    # 4. Monotonic circular total variation reduction (roughness)
    def circular_tv(arr: np.ndarray) -> float:
        diffs = arr - np.roll(arr, 1)
        return float(np.sum(np.abs(diffs)))

    tvs = [circular_tv(s) for s in smoothed_series]
    for i in range(len(tvs) - 1):
        assert tvs[i] > tvs[i + 1], (
            f"Total variation not strictly decreasing: sigma {sigmas[i]} tv={tvs[i]:.4f} "
            f"<= sigma {sigmas[i+1]} tv={tvs[i+1]:.4f}"
        )


def test_nan_and_missing_bin_handling() -> None:
    """Verify normalized circular convolution handles missing bins and gaps cleanly."""
    L = 288
    t = np.arange(L, dtype=np.float64)

    # 1. Single isolated missing bin on smooth diurnal sine
    profile_sine = 70.0 + 15.0 * np.sin(2.0 * np.pi * t / L)
    profile_with_nan = profile_sine.copy()
    profile_with_nan[100] = np.nan

    s = circular_gaussian_smooth(profile_with_nan, sigma=2.0)
    assert not np.isnan(s).any(), "Single NaN should be interpolated by normalized convolution"
    # Interpolated value should be close to true value
    assert abs(s[100] - profile_sine[100]) < 1.0, (
        f"Interpolated error too large: {s[100]} vs {profile_sine[100]}"
    )

    # 2. Symmetric gap centered at bin 0 on even profile cos(2*pi*t / L)
    # Missing bins: -2, -1, 0, 1, 2 (bins 286, 287, 0, 1, 2)
    # Circular reflection k <-> (L - k) % L must yield identical values: s[1] == s[287], s[2] == s[286]
    profile_cos = 70.0 + 15.0 * np.cos(2.0 * np.pi * t / L)
    profile_gap_at_0 = profile_cos.copy()
    profile_gap_at_0[286:] = np.nan
    profile_gap_at_0[:3] = np.nan

    for sigma in [1.0, 2.0, 4.0]:
        s_gap = circular_gaussian_smooth(profile_gap_at_0, sigma=sigma)
        assert np.isfinite(s_gap[0]), f"Bin 0 is NaN for sigma={sigma}"
        assert np.isfinite(s_gap[287]), f"Bin 287 is NaN for sigma={sigma}"
        # Test circular symmetry: s[k] == s[(L - k) % L]
        for k in range(1, 10):
            left_bin = (L - k) % L
            right_bin = k
            pair_diff = abs(s_gap[left_bin] - s_gap[right_bin])
            assert pair_diff <= 1e-13, (
                f"Symmetric gap asymmetry around bin 0 at offset {k}: {pair_diff:.2e}"
            )

    # 3. Symmetric gap straddling midnight boundary (-0.5, between bin 287 and 0)
    # Missing bins: 285..287 and 0..2 (6 bins, 3 on each side of midnight)
    # On profile symmetric around midnight boundary: cos(2*pi*(t + 0.5) / L)
    profile_straddle = 70.0 + 15.0 * np.cos(2.0 * np.pi * (t + 0.5) / L)
    profile_straddle_gap = profile_straddle.copy()
    profile_straddle_gap[285:] = np.nan
    profile_straddle_gap[:3] = np.nan

    for sigma in [1.0, 2.0, 4.0]:
        s_straddle = circular_gaussian_smooth(profile_straddle_gap, sigma=sigma)
        # Symmetrical pairs across midnight boundary: s[0] == s[287], s[1] == s[286], s[2] == s[285]
        for offset in range(3):
            left_bin = 287 - offset
            right_bin = offset
            pair_diff = abs(s_straddle[left_bin] - s_straddle[right_bin])
            assert pair_diff <= 1e-13, (
                f"Straddled midnight gap asymmetry at offset {offset}: {pair_diff:.2e}"
            )

    # 4. All-NaN array returns all-NaN without exception or unhandled warnings
    all_nans = np.full(L, np.nan, dtype=np.float64)
    res_all_nan = circular_gaussian_smooth(all_nans, sigma=2.0)
    assert np.isnan(res_all_nan).all(), "All-NaN array should produce all-NaN result"

    # 5. 2D batch consistency: smoothing 2D array equals smoothing row-by-row
    batch = np.array([profile_sine, profile_with_nan, profile_gap_at_0, profile_straddle_gap])
    for sigma in [1.0, 2.0, 4.0]:
        smoothed_batch = circular_gaussian_smooth(batch, sigma=sigma, axis=-1)
        for row_idx in range(len(batch)):
            smoothed_single = circular_gaussian_smooth(batch[row_idx], sigma=sigma)
            max_diff = np.nanmax(np.abs(smoothed_batch[row_idx] - smoothed_single))
            assert max_diff <= 1e-12, f"2D batch differs from 1D at row {row_idx}: diff={max_diff:.2e}"


def test_discrete_kernel_equivalence() -> None:
    """Verify SciPy wrap-mode filter matches discrete circular convolution within truncation bound."""
    L = 288
    sigma = 2.0
    kernel = discrete_circular_gaussian_kernel(sigma=sigma, n_bins=L)

    # Delta at bin 0
    delta = np.zeros(L, dtype=np.float64)
    delta[0] = 1.0

    # With truncate=8.0 (radius 16 bins for sigma=2), truncation weight is exp(-32) ~ 1e-14
    s_scipy = circular_gaussian_smooth(delta, sigma=sigma, truncate=8.0)

    # Exact circular convolution with kernel
    # Since delta is at 0, conv(delta, kernel) is exactly kernel
    max_err = np.max(np.abs(s_scipy - kernel))
    assert max_err <= 1e-10, (
        f"SciPy wrap filter differs from discrete periodic kernel: max_err={max_err:.2e}"
    )


def run_unit_tests() -> dict[str, float]:
    """Execute all unit tests and record per-test execution times."""
    tests = [
        ("test_midnight_wraparound_symmetry", test_midnight_wraparound_symmetry),
        ("test_mean_and_energy_conservation", test_mean_and_energy_conservation),
        ("test_bandwidth_scaling_monotonicity", test_bandwidth_scaling_monotonicity),
        ("test_nan_and_missing_bin_handling", test_nan_and_missing_bin_handling),
        ("test_discrete_kernel_equivalence", test_discrete_kernel_equivalence),
    ]

    timings = {}
    total_start = time.perf_counter()

    for name, test_fn in tests:
        t0 = time.perf_counter()
        test_fn()
        elapsed = time.perf_counter() - t0
        timings[name] = elapsed
        print(f"  [PASS] {name:<40} ({elapsed * 1000.0:6.2f} ms)")

    total_elapsed = time.perf_counter() - total_start
    timings["total_suite"] = total_elapsed

    assert total_elapsed < 2.0, f"Suite execution took {total_elapsed:.3f}s (exceeded 2.0s limit)"
    return timings


if __name__ == "__main__":
    print("=" * 70)
    print("Running Circular Diurnal Regularization Unit Verification Fixture")
    print("Target: 288-bin diurnal profiles (bins 0-287, 5-minute resolution)")
    print("Bandwidths: sigma in {1, 2, 4} bins (5, 10, 20 minutes)")
    print("=" * 70)

    timings = run_unit_tests()

    print("=" * 70)
    print(f"ALL TESTS PASSED SUCCESSFULLY! Total runtime: {timings['total_suite'] * 1000.0:.2f} ms")
    print("=" * 70)
