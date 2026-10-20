"""Unit tests for Milestone 2 masked imputation evaluator components."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from circular_smoothing import circular_gaussian_smooth

# Load find_eligible_gaps_for_participant from 01_evaluate_imputation_error.py
spec = importlib.util.spec_from_file_location(
    "evaluate_imputation_error",
    SCRIPT_DIR / "01_evaluate_imputation_error.py"
)
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)
find_eligible_gaps_for_participant = eval_module.find_eligible_gaps_for_participant


def test_gap_reconstruction_math():
    """Verify exact mathematical properties of gap filling and metrics."""
    # Synthetic ground truth
    x_true = np.array([70.0, 75.0, 80.0], dtype=np.float64)
    x_left = 65.0
    x_right = 85.0
    k = 3

    # Linear interpolation
    # j in 0..2 -> step = (j+1)/(k+1) = [1/4, 2/4, 3/4] = [0.25, 0.50, 0.75]
    # x_linear = 65 + [0.25, 0.5, 0.75] * 20 = [70, 75, 80]
    steps = np.arange(1, k + 1, dtype=np.float64) / (k + 1.0)
    x_linear = x_left + steps * (x_right - x_left)

    np.testing.assert_allclose(x_linear, x_true)

    # MAE and RMSE should be exactly 0
    diff = x_linear - x_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    assert mae == 0.0
    assert rmse == 0.0

    # Boundary jump for linear: |x_linear[0] - x_left| = |70 - 65| = 5.0
    boundary_jump = float(np.abs(x_linear[0] - x_left))
    assert boundary_jump == 5.0

    # Template filling
    tmpl = np.array([72.0, 72.0, 72.0], dtype=np.float64)
    diff_tmpl = tmpl - x_true  # [2.0, -3.0, -8.0]
    expected_mae = (2.0 + 3.0 + 8.0) / 3.0  # 4.333333333333333
    expected_rmse = np.sqrt((4.0 + 9.0 + 64.0) / 3.0)  # sqrt(77/3) = 5.06622805
    expected_bj = abs(72.0 - 65.0)  # 7.0

    assert np.isclose(np.mean(np.abs(diff_tmpl)), expected_mae)
    assert np.isclose(np.sqrt(np.mean(diff_tmpl ** 2)), expected_rmse)
    assert np.isclose(abs(tmpl[0] - x_left), expected_bj)


def test_gap_finder_properties():
    """Verify segment finding and gap extraction on synthetic HR recording."""
    D, L = 5, 288
    hr = np.full((D, L), np.nan, dtype=np.float64)
    # Day 0: contiguous segment from bin 10 to bin 80 (71 bins)
    hr[0, 10:81] = 70.0
    # Day 1: contiguous segment from bin 50 to bin 65 (16 bins)
    hr[1, 50:66] = 68.0

    rng = np.random.default_rng(42)
    gaps = find_eligible_gaps_for_participant(hr, rng, gaps_per_tier=5)

    assert len(gaps) > 0
    for gap_tier, k, d, t_start, t_end in gaps:
        # Gap length must match end - start + 1
        assert t_end - t_start + 1 == k
        # Must have valid boundary bins in hr[d]
        assert np.isfinite(hr[d, t_start - 1])
        assert np.isfinite(hr[d, t_end + 1])
        # Ground truth within gap must be valid
        assert np.all(np.isfinite(hr[d, t_start : t_end + 1]))


def test_normalized_convolution_on_sparse_diurnal():
    """Verify that normalized circular convolution smoothly bridges missing runs."""
    L = 288
    t = np.arange(L, dtype=np.float64)
    profile_true = 70.0 + 10.0 * np.sin(2.0 * np.pi * t / L)
    profile_with_nan = profile_true.copy()
    # Inject 4-bin gap
    profile_with_nan[100:104] = np.nan

    smoothed = circular_gaussian_smooth(profile_with_nan, sigma=2.0, nan_policy="normalize")
    assert not np.isnan(smoothed).any()
    assert abs(smoothed[101] - profile_true[101]) < 0.5


if __name__ == "__main__":
    test_gap_reconstruction_math()
    test_gap_finder_properties()
    test_normalized_convolution_on_sparse_diurnal()
    print("All unit evaluator component tests passed!")
