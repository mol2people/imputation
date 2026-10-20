"""Adversarial stress-test suite for Apple Watch Milestone 0 transforms.

Focuses on:
  - Extreme edge cases (D_u = 0, D_u = 1, D_u > 10,000)
  - Floating point boundary conditions and precision
  - IEEE-754 special values (NaN, Inf, -Inf)
  - 1D vs 2D inputs
  - Gap patterns: all-missing, all-observed, single-bin arrays, odd shapes
  - Empty participant arrays
  - Mask dtype variations (bool, int32, int64, float32, uint8)
  - Integrity of function returns (no hardcoded outputs)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add scripts directory
scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from transforms import (
    compute_clean_diff,
    compute_mask_stats,
    fit_fill_templates,
    interpolate_short_gaps,
)


def test_compute_clean_diff_adversarial():
    print("Testing compute_clean_diff under adversarial conditions...")

    # 1. Shape preservation and empty arrays
    for shape in [(0, 288), (1, 288), (5, 96), (2, 3, 288)]:
        hr = np.random.randn(*shape).astype(np.float32)
        mask = (np.random.rand(*shape) > 0.5).astype(np.uint8)
        diff = compute_clean_diff(hr, mask)
        expected_shape = list(shape)
        expected_shape[-1] -= 1
        assert diff.shape == tuple(expected_shape), f"Shape mismatch: {diff.shape} vs {expected_shape}"
        assert diff.dtype == np.float32

    # 2. Extreme and non-finite values in hr
    # When mask=0, hr having NaN, Inf, -Inf, 1e38, -1e38 must NOT cause NaN or Inf in diff
    L = 10
    hr_extreme = np.array([[np.nan, np.inf, -np.inf, 1e35, -1e35, np.nan, 70.0, 75.0, np.nan, 80.0]])
    mask_extreme = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 0, 1]], dtype=np.uint8)
    diff = compute_clean_diff(hr_extreme, mask_extreme)
    assert not np.isnan(diff).any(), "NaN found in clean_diff with non-finite unobserved HR!"
    assert not np.isinf(diff).any(), "Inf found in clean_diff with non-finite unobserved HR!"
    # Only transition between index 6 and 7 (which is diff index 6) should be non-zero
    assert diff[0, 6] == 5.0
    diff_zero_mask = np.ones(L - 1, dtype=bool)
    diff_zero_mask[6] = False
    assert np.all(diff[0, diff_zero_mask] == 0.0)

    # 3. Mask dtypes
    for dtype in [np.uint8, np.int32, np.int64, np.float32, bool]:
        m = np.array([[1, 1, 0, 1, 1]], dtype=dtype)
        h = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=np.float32)
        d = compute_clean_diff(h, m)
        assert np.array_equal(d, np.array([[10.0, 0.0, 0.0, 10.0]], dtype=np.float32))

    # 4. Identity when h_{t+1} == h_t on observed bins
    h_same = np.array([[70.0, 70.0, 70.0]], dtype=np.float32)
    m_same = np.array([[1, 1, 1]], dtype=np.uint8)
    d_same = compute_clean_diff(h_same, m_same)
    assert np.all(d_same == 0.0)
    assert d_same.shape == (1, 2)

    # 5. Length 1 time series (edge case L=1 -> L-1=0)
    h_1 = np.array([[70.0]], dtype=np.float32)
    m_1 = np.array([[1]], dtype=np.uint8)
    d_1 = compute_clean_diff(h_1, m_1)
    assert d_1.shape == (1, 0)

    print("  -> compute_clean_diff passed all adversarial tests.")


def test_interpolate_short_gaps_adversarial():
    print("Testing interpolate_short_gaps under adversarial conditions...")

    # 1. 1D input vs 2D input
    hr_1d = np.array([60.0, np.nan, 70.0, np.nan, np.nan, 60.0], dtype=np.float32)
    mask_1d = np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8)
    tmpl_1d = np.full(6, 50.0, dtype=np.float32)
    out_1d = interpolate_short_gaps(hr_1d, mask_1d, tmpl_1d, resolution="5m", k_max=1)
    assert out_1d.ndim == 1, "1D input did not return 1D output"
    assert out_1d.shape == (6,)
    assert out_1d[1] == 65.0  # k=1 interpolated
    assert np.all(out_1d[3:5] == 50.0)  # k=2 not interpolated because k_max=1

    # 2. Empty participant array (D=0, L=288)
    hr_empty = np.zeros((0, 288), dtype=np.float32)
    out_empty = interpolate_short_gaps(hr_empty, resolution="5m")
    assert out_empty.shape == (0, 288)

    # 3. All observed day
    hr_all_obs = np.linspace(60.0, 80.0, 288, dtype=np.float32)[None, :]
    out_all_obs = interpolate_short_gaps(hr_all_obs, resolution="5m")
    assert np.array_equal(out_all_obs, hr_all_obs)

    # 4. All missing day
    hr_all_nan = np.full((1, 288), np.nan, dtype=np.float32)
    tmpl = np.full(288, 62.5, dtype=np.float32)
    out_all_nan = interpolate_short_gaps(hr_all_nan, clock_template=tmpl, resolution="5m")
    assert np.all(out_all_nan == 62.5)

    # 5. Sub-bin boundaries: gap at t=0 or t=L-1
    hr_bounds = np.array([[np.nan, 60.0, np.nan, 70.0, np.nan]], dtype=np.float32)
    mask_bounds = np.array([[0, 1, 0, 1, 0]], dtype=np.uint8)
    tmpl_bounds = np.full(5, 55.0, dtype=np.float32)
    out_bounds = interpolate_short_gaps(hr_bounds, mask_bounds, tmpl_bounds, k_max=1)
    assert out_bounds[0, 0] == 55.0  # prefix boundary remains on template
    assert out_bounds[0, 1] == 60.0  # observed preserved
    assert out_bounds[0, 2] == 65.0  # interior k=1 interpolated
    assert out_bounds[0, 3] == 70.0  # observed preserved
    assert out_bounds[0, 4] == 55.0  # suffix boundary remains on template

    # 6. k_max <= 0 edge case
    out_k0 = interpolate_short_gaps(hr_bounds, mask_bounds, tmpl_bounds, k_max=0)
    assert np.all(out_k0[0, [0, 2, 4]] == 55.0)  # all gaps remain on template

    # 7. Invalid resolution
    try:
        interpolate_short_gaps(hr_1d, resolution="30m")
        assert False, "Should raise ValueError on invalid resolution"
    except ValueError:
        pass

    # 8. Non-finite values in template or hr
    # If clock_template is None and all bins are NaN, fallback to 60.0
    hr_all_nan_no_tmpl = np.full((2, 96), np.nan, dtype=np.float32)
    out_fallback = interpolate_short_gaps(hr_all_nan_no_tmpl, resolution="15m")
    assert not np.isnan(out_fallback).any()
    assert np.all(out_fallback == 60.0)

    print("  -> interpolate_short_gaps passed all adversarial tests.")


def test_compute_mask_stats_adversarial():
    print("Testing compute_mask_stats under adversarial conditions...")

    # 1. D_u = 1: Bessel correction undefined (division by 0), must return sd=0.0
    mask_d1 = np.array([[0, 1, 1, 0, 1]], dtype=np.uint8)
    mean_1, sd_1 = compute_mask_stats(mask_d1)
    assert np.all(sd_1 == 0.0)
    assert np.array_equal(mean_1, np.array([0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32))

    # 2. 1D mask input
    mask_1d = np.array([1, 0, 1, 1], dtype=np.uint8)
    mean_1d, sd_1d = compute_mask_stats(mask_1d)
    assert mean_1d.ndim == 1 and sd_1d.ndim == 1
    assert np.all(sd_1d == 0.0)

    # 3. D_u = 2: minimal D for sample SD
    mask_d2 = np.array([[0, 1, 1, 0],
                        [1, 1, 0, 0]], dtype=np.uint8)
    mean_2, sd_2 = compute_mask_stats(mask_d2)
    # Direct var:
    # col 0: [0, 1] -> mean 0.5, var = 2/1 * 0.5 * 0.5 = 0.5 -> sd = sqrt(0.5)
    # col 1: [1, 1] -> mean 1.0, var = 0.0 -> sd = 0.0
    # col 2: [1, 0] -> mean 0.5, var = 0.5 -> sd = sqrt(0.5)
    # col 3: [0, 0] -> mean 0.0, var = 0.0 -> sd = 0.0
    assert np.isclose(sd_2[0], np.sqrt(0.5))
    assert sd_2[1] == 0.0
    assert np.isclose(sd_2[2], np.sqrt(0.5))
    assert sd_2[3] == 0.0

    # 4. Large D_u (D_u = 10,000) for floating-point stability
    D = 10000
    rng = np.random.default_rng(42)
    mask_large = rng.integers(0, 2, size=(D, 96), dtype=np.uint8)
    mean_lg, sd_lg = compute_mask_stats(mask_large)
    direct_var = np.var(mask_large.astype(np.float64), axis=0, ddof=1)
    direct_sd = np.sqrt(direct_var)
    assert np.allclose(sd_lg, direct_sd, atol=1e-5)

    # 5. Mask with all 0s or all 1s
    m_all_0 = np.zeros((10, 20), dtype=np.uint8)
    m_all_1 = np.ones((10, 20), dtype=np.uint8)
    mean_0, sd_0 = compute_mask_stats(m_all_0)
    mean_1, sd_1 = compute_mask_stats(m_all_1)
    assert np.all(mean_0 == 0.0) and np.all(sd_0 == 0.0)
    assert np.all(mean_1 == 1.0) and np.all(sd_1 == 0.0)

    # 6. D_u = 0 (empty mask)
    mask_d0 = np.zeros((0, 288), dtype=np.uint8)
    mean_0, sd_0 = compute_mask_stats(mask_d0)
    # sd_0 is guarded: D > 1 is False -> sd_0 is zeros
    assert np.all(sd_0 == 0.0)
    # Note: mean of empty slice yields NaN in numpy
    assert np.isnan(mean_0).all()

    print("  -> compute_mask_stats passed all adversarial tests.")


def test_fit_fill_templates_adversarial():
    print("Testing fit_fill_templates under adversarial conditions...")

    # 1. List of participant arrays with varying lengths D_u
    p1 = np.array([[60.0, 65.0], [62.0, 67.0]], dtype=np.float32)  # D=2, L=2
    p2 = np.array([[70.0, 75.0], [72.0, 77.0], [74.0, 79.0]], dtype=np.float32)  # D=3, L=2
    m1 = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    m2 = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)

    clock, glob = fit_fill_templates([p1, p2], [m1, m2])
    assert clock.shape == (2,)
    # Col 0 observed values: p1[0,0]=60, p1[1,0]=62, p2[0,0]=70, p2[1,0]=72 -> median of [60, 62, 70, 72] = 66.0
    assert clock[0] == 66.0
    # Col 1 observed values: p1[0,1]=65, p1[1,1]=67, p2[1,1]=77, p2[2,1]=79 -> median of [65, 67, 77, 79] = 72.0
    assert clock[1] == 72.0
    # Global median of all 8 observed values: [60, 62, 65, 67, 70, 72, 77, 79] = 68.5
    assert glob == 68.5

    # 2. Entire bin unobserved across all participants -> falls back to global median
    p_unobs = np.array([[np.nan, 70.0], [np.nan, 80.0]], dtype=np.float32)
    m_unobs = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    clock_u, glob_u = fit_fill_templates(p_unobs, m_unobs)
    assert clock_u[0] == 75.0  # falls back to global median (75.0)
    assert clock_u[1] == 75.0
    assert glob_u == 75.0

    # 3. Everything unobserved -> falls back to 60.0
    p_all_nan = np.full((5, 10), np.nan, dtype=np.float32)
    clock_all, glob_all = fit_fill_templates(p_all_nan)
    assert np.all(clock_all == 60.0)
    assert glob_all == 60.0

    print("  -> fit_fill_templates passed all adversarial tests.")


def test_integrity_and_facade():
    print("Testing for hardcoded results and facade implementations...")
    # Verify that different inputs produce different, mathematically expected outputs
    # (i.e. not returning static constants or bypassed logic)

    hr1 = np.array([[50.0, np.nan, 70.0]], dtype=np.float32)
    hr2 = np.array([[100.0, np.nan, 120.0]], dtype=np.float32)
    out1 = interpolate_short_gaps(hr1, resolution="5m")
    out2 = interpolate_short_gaps(hr2, resolution="5m")
    assert out1[0, 1] == 60.0
    assert out2[0, 1] == 110.0
    assert not np.array_equal(out1, out2), "interpolate_short_gaps returned identical outputs for different inputs!"

    diff1 = compute_clean_diff(np.array([[10.0, 20.0]]), np.array([[1, 1]]))
    diff2 = compute_clean_diff(np.array([[10.0, 30.0]]), np.array([[1, 1]]))
    assert diff1[0, 0] == 10.0
    assert diff2[0, 0] == 20.0
    assert not np.array_equal(diff1, diff2), "compute_clean_diff returned identical outputs for different inputs!"

    print("  -> Integrity check passed: functions exhibit genuine dynamic computation.")


def main():
    test_compute_clean_diff_adversarial()
    test_interpolate_short_gaps_adversarial()
    test_compute_mask_stats_adversarial()
    test_fit_fill_templates_adversarial()
    test_integrity_and_facade()
    print("\nALL ADVERSARIAL STRESS TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
