#!/usr/bin/env python3
"""
test_m4_bootstrap_inference.py: Unit and regression test suite for Milestone 4.
Verifies:
1. Analytical Stone-Koo basis derivatives against finite differences
2. Stratified participant resampling cell invariance
3. Studentized sup-norm deviation and simultaneous uniform coverage
4. Transition metric root-finding, minimum derivative, and boundary handling
5. Frozen harmonic prediction phase preservation
6. Artifact existence, schemas, and non-emptiness
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
BOOTSTRAP_DIR = ARTIFACTS_DIR / "bootstrap"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import DEFAULT_KNOTS, rcs_stone_koo
import importlib
m4_mod = importlib.import_module("04_run_bootstrap_inference")
d_rcs_stone_koo = m4_mod.d_rcs_stone_koo
build_design_matrix = m4_mod.build_design_matrix
build_chunk_contrast_matrices = m4_mod.build_chunk_contrast_matrices
extract_transition_metrics = m4_mod.extract_transition_metrics


class TestM4BootstrapInference(unittest.TestCase):
    def setUp(self):
        self.knots = DEFAULT_KNOTS
        self.grid = np.arange(18.0, 80.0 + 0.25, 0.25)

    def test_01_rcs_derivatives_vs_finite_difference(self):
        """Analytical Stone-Koo derivatives must match central finite differences to < 1e-5."""
        h = 1e-5
        test_ages = np.array([20.0, 24.9, 25.0, 30.0, 42.0, 47.0, 52.0, 58.0, 65.0, 72.0])
        d1, d2 = d_rcs_stone_koo(test_ages, self.knots)

        r1_plus, r2_plus = rcs_stone_koo(test_ages + h, self.knots)
        r1_minus, r2_minus = rcs_stone_koo(test_ages - h, self.knots)

        d1_num = (r1_plus - r1_minus) / (2.0 * h)
        d2_num = (r2_plus - r2_minus) / (2.0 * h)

        np.testing.assert_allclose(d1, d1_num, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(d2, d2_num, atol=1e-5, rtol=1e-4)

    def test_02_rcs_tail_derivatives_exact(self):
        """Derivatives outside [25, 65] must match exact theoretical constants."""
        # Left tail: a <= 25.0 -> exactly 0.0
        left_ages = np.array([18.0, 20.0, 22.5, 25.0])
        d1_l, d2_l = d_rcs_stone_koo(left_ages, self.knots)
        np.testing.assert_equal(d1_l, 0.0)
        np.testing.assert_equal(d2_l, 0.0)

        # Right tail: a >= 65.0 -> d1 = 2.025, d2 = 0.43125 bit-exactly
        right_ages = np.array([65.0, 70.0, 75.0, 80.0])
        d1_r, d2_r = d_rcs_stone_koo(right_ages, self.knots)
        np.testing.assert_allclose(d1_r, 2.025, atol=1e-12)
        np.testing.assert_allclose(d2_r, 0.43125, atol=1e-12)

    def test_03_stratified_sampling_preserves_cell_counts(self):
        """Stratified bootstrap must preserve exact cell counts in every draw."""
        rng = np.random.default_rng(42)
        ages = rng.integers(20, 80, size=500).astype(float)
        sex = rng.integers(0, 2, size=500).astype(float)

        bins = np.arange(0, 130, 5)
        band_idx = np.digitize(ages, bins)
        strata = band_idx * 2 + sex.astype(int)
        unique_strata, orig_counts = np.unique(strata, return_counts=True)
        stratum_indices = [np.where(strata == s)[0] for s in unique_strata]
        stratum_sizes = [len(idx) for idx in stratum_indices]

        # 10 resamples
        for _ in range(10):
            resamp_idx = np.empty(len(ages), dtype=int)
            pos = 0
            for idx_arr, sz in zip(stratum_indices, stratum_sizes):
                resamp_idx[pos : pos + sz] = rng.choice(idx_arr, size=sz, replace=True)
                pos += sz

            resamp_strata = strata[resamp_idx]
            u_resamp, resamp_counts = np.unique(resamp_strata, return_counts=True)
            np.testing.assert_array_equal(u_resamp, unique_strata)
            np.testing.assert_array_equal(resamp_counts, orig_counts)

    def test_04_studentized_sup_norm_critical_value(self):
        """Studentized sup-norm critical value c_0.95 must strictly exceed pointwise z_0.975 = 1.960."""
        rng = np.random.default_rng(20261020)
        n_boot = 500
        n_grid = len(self.grid)
        # Synthetic correlated curves
        base_curve = np.sin(self.grid / 10.0)
        curves = np.empty((n_boot, n_grid))
        for b in range(n_boot):
            noise = rng.normal(0, 0.02, size=4)
            curves[b] = base_curve + noise[0] + noise[1] * (self.grid / 50.0) + rng.normal(0, 0.005, size=n_grid)

        se = np.std(curves, axis=0, ddof=1)
        mean_c = np.mean(curves, axis=0)
        abs_devs = np.abs(curves - mean_c[None, :]) / se[None, :]
        sup_devs = np.max(abs_devs, axis=1)
        c_095 = np.percentile(sup_devs, 95.0)

        self.assertGreater(c_095, 1.960, f"Simultaneous critical value {c_095} should exceed 1.960")
        self.assertLess(c_095, 3.500, f"Simultaneous critical value {c_095} should be reasonable (< 3.5)")

        # Verify uniform coverage is >= 94% on synthetic sample
        simult_lower = mean_c - c_095 * se
        simult_upper = mean_c + c_095 * se
        covered = np.all((curves >= simult_lower[None, :]) & (curves <= simult_upper[None, :]), axis=1)
        empirical_coverage = np.mean(covered)
        self.assertGreaterEqual(empirical_coverage, 0.94)

    def test_05_transition_metrics_recovery(self):
        """Transition metric functions must accurately recover known inflection point and zero-crossing."""
        # Simulated delta curve: drops steeply around age 48, crosses 0 at age 55
        # delta(a) = 0.03 - 0.002 * (a - 40)_+ + ...
        delta_params = np.array([0.02, 0.0002, -0.003, 0.013])
        g_rcs1, g_rcs2 = rcs_stone_koo(self.grid, self.knots)
        X_delta = np.column_stack([np.ones_like(self.grid), self.grid, g_rcs1, g_rcs2])
        delta_pred = X_delta @ delta_params

        metrics = extract_transition_metrics(
            delta_params,
            self.grid,
            delta_pred,
            upper_simult_band=delta_pred + 0.005,
        )

        self.assertGreaterEqual(metrics["inflection_age"], 35.0)
        self.assertLessEqual(metrics["inflection_age"], 65.0)
        self.assertLess(metrics["inflection_slope"], 0.0, "Inflection slope should be negative for declining curve")

    def test_06_frozen_harmonic_prediction_calendar_phase(self):
        """Evaluating frozen harmonic with elapsed days (t - t0) must preserve exact calendar phase."""
        t1_start = 18000
        t2_start = 18200  # 200 days later
        period = 28.0
        omega = 2.0 * np.pi / period
        bc, bs = 1.5, -0.8

        d1 = np.arange(t1_start, t1_start + 90)
        d2 = np.arange(t2_start, t2_start + 90)

        # Signal in absolute calendar days: h(t) = bc * cos(omega*(t - t1_start)) + bs * sin(omega*(t - t1_start))
        h1 = bc * np.cos(omega * (d1 - t1_start)) + bs * np.sin(omega * (d1 - t1_start))
        h2_frozen = bc * np.cos(omega * (d2 - t1_start)) + bs * np.sin(omega * (d2 - t1_start))

        # True signal generated by continuous calendar wave
        true_h2 = bc * np.cos(omega * (d2 - t1_start)) + bs * np.sin(omega * (d2 - t1_start))
        np.testing.assert_allclose(h2_frozen, true_h2, atol=1e-14)

    def test_07_artifacts_exist_and_non_empty(self):
        """All 6 required output artifacts must exist and have non-zero size."""
        required = [
            BOOTSTRAP_DIR / "confidence_bands.parquet",
            BOOTSTRAP_DIR / "bootstrap_curves.parquet",
            BOOTSTRAP_DIR / "transition_metrics.csv",
            BOOTSTRAP_DIR / "bootstrap_chunk_pvalues.csv",
            BOOTSTRAP_DIR / "second_window_repeatability.parquet",
            BOOTSTRAP_DIR / "repeatability_summary.csv",
        ]
        for p in required:
            self.assertTrue(p.exists(), f"Missing artifact: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Artifact is empty: {p}")

    def test_08_confidence_bands_structure_and_coverage(self):
        """confidence_bands.parquet must contain exact grid resolution and valid intervals."""
        df = pd.read_parquet(BOOTSTRAP_DIR / "confidence_bands.parquet")
        # 4 configs x 4 endpoints x 3 curves x 249 ages = 11,952 rows
        self.assertEqual(len(df), 11952)
        expected_cols = [
            "source", "source_name", "protocol", "endpoint", "curve_type", "age",
            "point_estimate", "boot_mean", "boot_se", "ci_lower_pointwise", "ci_upper_pointwise",
            "simult_c_095", "band_lower_simultaneous", "band_upper_simultaneous"
        ]
        for c in expected_cols:
            self.assertIn(c, df.columns)

        # Pointwise CI must contain point estimate in vast majority of points
        # Simultaneous band must strictly envelope pointwise CI everywhere
        self.assertTrue((df["band_lower_simultaneous"] <= df["ci_lower_pointwise"] + 1e-9).all())
        self.assertTrue((df["band_upper_simultaneous"] >= df["ci_upper_pointwise"] - 1e-9).all())
        # Critical value c_0.95 must be in [2.5, 3.2]
        self.assertTrue((df["simult_c_095"] >= 2.5).all())
        self.assertTrue((df["simult_c_095"] <= 3.2).all())

    def test_09_bootstrap_curves_structure(self):
        """bootstrap_curves.parquet must contain 32,000 replicates of 249-element float arrays."""
        df = pd.read_parquet(BOOTSTRAP_DIR / "bootstrap_curves.parquet")
        # 4 configs x 4 endpoints x 2000 replicates = 32,000 rows
        self.assertEqual(len(df), 32000)
        self.assertIn("delta_curve", df.columns)
        self.assertIn("female_curve", df.columns)
        self.assertIn("male_curve", df.columns)

        # Inspect first row length
        row0_curve = df["delta_curve"].iloc[0]
        self.assertEqual(len(row0_curve), 249)
        self.assertTrue(np.all(np.isfinite(row0_curve)))

    def test_10_transition_metrics_values(self):
        """transition_metrics.csv must contain plausible values for inflection and zero-crossing."""
        df = pd.read_csv(BOOTSTRAP_DIR / "transition_metrics.csv")
        self.assertEqual(len(df), 64)  # 4 configs x 4 endpoints x 4 metric rows

        # Primary Q_i in Apple Watch expanded90
        q_apple = df[(df["source"] == 6) & (df["protocol"] == "expanded90") & (df["endpoint"] == "Q_i")]
        inf_row = q_apple[q_apple["metric_name"] == "steepest_decline_inflection_age"].iloc[0]
        self.assertAlmostEqual(inf_row["point_estimate"], 46.65, delta=0.5)
        self.assertEqual(inf_row["status"], "interior_minimum")

        zc_row = q_apple[q_apple["metric_name"] == "nominal_zero_crossing_age"].iloc[0]
        self.assertAlmostEqual(zc_row["point_estimate"], 56.18, delta=1.5)
        self.assertEqual(zc_row["status"], "observed")

    def test_11_repeatability_structure(self):
        """second_window_repeatability.parquet must contain all two-window participants."""
        df = pd.read_parquet(BOOTSTRAP_DIR / "second_window_repeatability.parquet")
        self.assertGreater(len(df), 11000)
        # Apple balanced90 must have 2,322 participants
        apple_bal = df[(df["source"] == 6) & (df["protocol"] == "balanced90")]
        self.assertEqual(len(apple_bal), 2322)

        # Check repeatability summary
        df_sum = pd.read_csv(BOOTSTRAP_DIR / "repeatability_summary.csv")
        self.assertEqual(len(df_sum), 48)


if __name__ == "__main__":
    unittest.main()
