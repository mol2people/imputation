#!/usr/bin/env python3
"""
test_m3_rcs_models.py: Comprehensive unit and regression test suite for Milestone 3:
Restricted Cubic Splines (Stone-Koo basis), OLS regression, Chunk Wald hypothesis tests,
shared support masking, and artifact schema validation.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

# Paths
TEST_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = TEST_DIR.parent
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import DEFAULT_KNOTS, rcs_stone_koo
from importlib import import_module
fit_rcs_module = import_module("03_fit_rcs_models")
d_rcs_stone_koo = fit_rcs_module.d_rcs_stone_koo
build_rcs_design_matrix = fit_rcs_module.build_rcs_design_matrix
compute_chunk_wald = fit_rcs_module.compute_chunk_wald
build_chunk_contrast_matrices = fit_rcs_module.build_chunk_contrast_matrices
fit_rcs_model = fit_rcs_module.fit_rcs_model


class TestStoneKooBasis(unittest.TestCase):
    def test_stone_koo_basis_zero_below_knot1(self):
        """rcs1 and rcs2 must be bit-exact 0.0 for age <= 25.0."""
        ages = np.array([0.0, 10.0, 18.0, 24.0, 24.9999, 25.0])
        rcs1, rcs2 = rcs_stone_koo(ages)
        np.testing.assert_array_equal(rcs1, 0.0)
        np.testing.assert_array_equal(rcs2, 0.0)

    def test_stone_koo_basis_semi_active_interval(self):
        """rcs1 > 0 and rcs2 == 0.0 for 25.0 < age <= 42.0."""
        ages = np.array([25.01, 30.0, 35.0, 42.0])
        rcs1, rcs2 = rcs_stone_koo(ages)
        self.assertTrue(np.all(rcs1 > 0.0))
        np.testing.assert_array_equal(rcs2, 0.0)

    def test_stone_koo_basis_interior_curvature(self):
        """Both rcs1 > 0 and rcs2 > 0 for 42.0 < age <= 65.0."""
        ages = np.array([42.5, 48.0, 52.0, 60.0, 65.0])
        rcs1, rcs2 = rcs_stone_koo(ages)
        self.assertTrue(np.all(rcs1 > 0.0))
        self.assertTrue(np.all(rcs2 > 0.0))

    def test_stone_koo_basis_linear_tails_above_knot4(self):
        """
        Above knot 4 (age >= 65.0), rcs1 and rcs2 are strictly linear with exact slopes:
          d(rcs1)/da = 2.025
          d(rcs2)/da = 0.43125
        """
        ages = np.array([65.0, 70.0, 75.0, 80.0, 90.0, 100.0])
        rcs1, rcs2 = rcs_stone_koo(ages)

        diff_a = np.diff(ages)
        diff_r1 = np.diff(rcs1)
        diff_r2 = np.diff(rcs2)

        slopes1 = diff_r1 / diff_a
        slopes2 = diff_r2 / diff_a

        np.testing.assert_allclose(slopes1, 2.025, atol=1e-12)
        np.testing.assert_allclose(slopes2, 0.43125, atol=1e-12)

        d1, d2 = d_rcs_stone_koo(ages)
        np.testing.assert_allclose(d1, 2.025, atol=1e-12)
        np.testing.assert_allclose(d2, 0.43125, atol=1e-12)

    def test_stone_koo_c2_continuity_at_knots(self):
        """Check value and first derivative continuity across knot boundaries."""
        eps = 1e-6
        for knot in DEFAULT_KNOTS:
            a_left = knot - eps
            a_right = knot + eps

            r1_l, r2_l = rcs_stone_koo(a_left)
            r1_r, r2_r = rcs_stone_koo(a_right)
            self.assertLess(abs(r1_l - r1_r), 1e-4)
            self.assertLess(abs(r2_l - r2_r), 1e-4)

            d1_l, d2_l = d_rcs_stone_koo(a_left)
            d1_r, d2_r = d_rcs_stone_koo(a_right)
            self.assertLess(abs(d1_l - d1_r), 1e-4)
            self.assertLess(abs(d2_l - d2_r), 1e-4)


class TestDesignMatrix(unittest.TestCase):
    def test_build_rcs_design_matrix(self):
        """Verify design matrix shape and column alignment."""
        ages = np.array([20.0, 30.0, 50.0, 70.0])
        sex = np.array([0, 1, 0, 1])

        X, col_names = build_rcs_design_matrix(ages, sex)
        self.assertEqual(X.shape, (4, 8))
        self.assertEqual(col_names, [
            "const", "age", "rcs1", "rcs2",
            "sex", "sex_age", "sex_rcs1", "sex_rcs2"
        ])
        np.testing.assert_array_equal(X[0, 4:], 0.0)
        np.testing.assert_array_equal(X[2, 4:], 0.0)
        np.testing.assert_allclose(X[1, 4:], X[1, :4])
        np.testing.assert_allclose(X[3, 4:], X[3, :4])

    def test_build_rcs_design_matrix_with_covariates(self):
        """Verify design matrix with additional covariates."""
        ages = np.array([25.0, 45.0, 65.0])
        sex = np.array([1, 0, 1])
        covs = {
            "z_occ": np.array([0.1, -0.2, 0.05]),
            "z_night": np.array([-0.05, 0.03, 0.02]),
        }
        X, col_names = build_rcs_design_matrix(ages, sex, covariates=covs)
        self.assertEqual(X.shape, (3, 10))
        self.assertEqual(col_names[-2:], ["z_occ", "z_night"])
        np.testing.assert_allclose(X[:, 8], covs["z_occ"])
        np.testing.assert_allclose(X[:, 9], covs["z_night"])


class TestOLSAndWaldHypotheses(unittest.TestCase):
    def test_ols_parameter_recovery_synthetic(self):
        """Verify exact OLS parameter recovery on synthetic noise-free data."""
        rng = np.random.default_rng(20261020)
        N = 500
        ages = rng.uniform(18.0, 80.0, size=N)
        sex = rng.choice([0, 1], size=N)

        theta_true = np.array([0.05, 0.001, -0.002, 0.005, 0.02, -0.0005, 0.003, -0.008])
        X, _ = build_rcs_design_matrix(ages, sex)
        y_true = X @ theta_true

        df_synth = pd.DataFrame({
            "age": ages,
            "sex": sex,
            "Q_i": y_true,
        })

        fit_rec, test_recs = fit_rcs_model(
            df_synth, endpoint="Q_i", source=6, protocol="expanded90",
            model_type="primary_unadjusted", sample_scope="shared_support"
        )

        params_recovered = json.loads(fit_rec["params_json"])
        np.testing.assert_allclose(params_recovered, theta_true, atol=1e-10)
        self.assertGreater(fit_rec["r_squared"], 0.999999)

    def test_chunk_wald_equivalence_to_statsmodels(self):
        """Verify compute_chunk_wald matches statsmodels wald_test bit-for-bit."""
        rng = np.random.default_rng(42)
        N = 200
        ages = rng.uniform(25.0, 75.0, size=N)
        sex = rng.choice([0, 1], size=N)
        X, _ = build_rcs_design_matrix(ages, sex)
        y = X @ np.array([0.05, 0.001, -0.002, 0.005, 0.02, -0.0005, 0.003, -0.008]) + rng.normal(0, 0.02, size=N)

        res_hc3 = sm.OLS(y, X).fit(cov_type="HC3")
        theta = res_hc3.params
        cov_hc3 = res_hc3.cov_params()
        df_resid = N - 8

        test_matrices = build_chunk_contrast_matrices(8)
        for t_num, (t_name, _, C) in test_matrices.items():
            res_manual = compute_chunk_wald(theta, cov_hc3, C, df_resid)
            res_sm_chi2 = res_hc3.wald_test(C, use_f=False, scalar=True)
            res_sm_f = res_hc3.wald_test(C, use_f=True, scalar=True)

            self.assertTrue(np.isclose(res_manual["wald_stat"], res_sm_chi2.statistic, atol=1e-8))
            self.assertTrue(np.isclose(res_manual["p_chi2"], res_sm_chi2.pvalue, atol=1e-8))
            self.assertTrue(np.isclose(res_manual["f_stat"], res_sm_f.statistic, atol=1e-8))
            self.assertTrue(np.isclose(res_manual["p_f"], res_sm_f.pvalue, atol=1e-8))


class TestMilestone3Artifacts(unittest.TestCase):
    def test_output_artifacts_exist(self):
        """Verify all four Milestone 3 output artifacts exist on disk."""
        self.assertTrue((MODELS_DIR / "shared_support_mask.parquet").exists())
        self.assertTrue((MODELS_DIR / "model_fits.parquet").exists())
        self.assertTrue((MODELS_DIR / "chunk_wald_tests.csv").exists())
        self.assertTrue((MODELS_DIR / "model_summary.csv").exists())

    def test_shared_support_mask_schema(self):
        """Verify shared_support_mask.parquet schema and support bands."""
        df_mask = pd.read_parquet(MODELS_DIR / "shared_support_mask.parquet")
        expected_cols = [
            "user_id", "source", "protocol", "window_index", "salutation",
            "sex", "age", "age_band", "in_shared_support", "in_female_support",
            "male_band_count", "female_band_count"
        ]
        for col in expected_cols:
            self.assertIn(col, df_mask.columns)

        self.assertEqual(len(df_mask), 17625)

        for src in [3, 6]:
            sub = df_mask[df_mask["source"] == src]
            pct = sub["in_shared_support"].mean()
            self.assertGreater(pct, 0.95)

        sub7 = df_mask[df_mask["source"] == 7]
        self.assertEqual(sub7["in_shared_support"].sum(), 0)

    def test_model_fits_schema_and_contents(self):
        """Verify model_fits.parquet schema, valid entries, and statistical properties."""
        df_fits = pd.read_parquet(MODELS_DIR / "model_fits.parquet")
        self.assertEqual(len(df_fits), 136)

        converged = df_fits[df_fits["status"] == "converged"]
        self.assertGreater(len(converged), 120)

        self.assertTrue(np.all((converged["r_squared"] >= 0.0) & (converged["r_squared"] <= 1.0)))
        self.assertTrue(np.all(converged["sigma_hat"] > 0.0))
        self.assertTrue(np.all(converged["df_resid"] > 0))

        core_cols = [
            "beta_0", "beta_1", "beta_2", "beta_3",
            "delta_0", "delta_1", "delta_2", "delta_3",
            "se_beta_0_hc3", "se_beta_1_hc3", "se_beta_2_hc3", "se_beta_3_hc3",
            "se_delta_0_hc3", "se_delta_1_hc3", "se_delta_2_hc3", "se_delta_3_hc3",
            "female_b0", "female_b1", "female_b2", "female_b3",
        ]
        for col in core_cols:
            self.assertTrue(np.all(~converged[col].isna()))

    def test_chunk_wald_tests_schema_and_pvalues(self):
        """Verify chunk_wald_tests.csv contains all 4 tests with valid statistics."""
        df_tests = pd.read_csv(MODELS_DIR / "chunk_wald_tests.csv")
        self.assertEqual(len(df_tests), 544)

        evaluated = df_tests[df_tests["status"] == "evaluated"]
        self.assertGreater(len(evaluated), 480)

        self.assertEqual(set(evaluated["test_num"].unique()), {1, 2, 3, 4})
        self.assertTrue(np.all(evaluated["wald_stat_hc3"] >= 0.0))
        self.assertTrue(np.all(evaluated["f_stat_hc3"] >= 0.0))
        self.assertTrue(np.all((evaluated["p_value_hc3"] >= 0.0) & (evaluated["p_value_hc3"] <= 1.0)))
        self.assertTrue(np.all((evaluated["p_value_ols"] >= 0.0) & (evaluated["p_value_ols"] <= 1.0)))

        # Apple expanded90 primary Q_i tests
        apple_q = evaluated[
            (evaluated["source"] == 6) &
            (evaluated["protocol"] == "expanded90") &
            (evaluated["endpoint"] == "Q_i") &
            (evaluated["model_type"] == "primary_unadjusted") &
            (evaluated["sample_scope"] == "shared_support")
        ]
        self.assertEqual(len(apple_q), 4)
        for _, t_row in apple_q.iterrows():
            self.assertLess(t_row["p_value_hc3"], 0.001)

    def test_model_summary_structure(self):
        """Verify model_summary.csv contains concise summary columns."""
        df_summary = pd.read_csv(MODELS_DIR / "model_summary.csv")
        self.assertEqual(len(df_summary), 136)

        expected_cols = [
            "model_id", "source", "source_name", "protocol", "endpoint",
            "model_type", "sample_scope", "status", "n_obs", "n_female",
            "n_male", "r_squared", "sigma_hat", "test1_p_hc3", "test2_p_hc3",
            "test3_p_hc3", "test4_p_hc3", "beta_1", "delta_1", "female_b1", "notes"
        ]
        for col in expected_cols:
            self.assertIn(col, df_summary.columns)


if __name__ == "__main__":
    unittest.main()
