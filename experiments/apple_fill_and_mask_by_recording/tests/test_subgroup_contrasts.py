"""Unit and integration test suite for Apple Watch Subgroup Contrasts (03_summarize_subgroup_contrasts.py).

Verifies:
  1. Allocation 0 cutpoints and 31 subgroup memberships.
  2. Mock prediction generator schema and calibration.
  3. Paired contrast mathematical logic and sign agreement strings.
  4. Spearman rank correlation calculations.
  5. Scientific report generator formatting and calibrated language compliance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure scripts directory is on sys.path
_TEST_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _TEST_DIR.parent
_SCRIPTS_DIR = _EXPERIMENT_DIR / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from transforms import compute_clean_diff, interpolate_short_gaps
import importlib.util

spec = importlib.util.spec_from_file_location("subgroup_contrasts", str(_SCRIPTS_DIR / "03_summarize_subgroup_contrasts.py"))
m3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3)


def test_subgroup_cutpoints_and_memberships():
    print("Testing subgroup cutpoints and memberships...", flush=True)

    # 1. Verify Allocation 0 cutpoints against reference artifacts
    alloc_path = m3.ALLOC_PATH
    assert alloc_path.exists(), f"Allocations file missing at {alloc_path}"
    alloc_df = pd.read_parquet(alloc_path).set_index("customer_id")

    # Synthetic descriptor table
    cids = alloc_df.index[:100].tolist()
    rng = np.random.default_rng(42)
    fake_info = pd.DataFrame({
        "customer_id": cids,
        "n_days": rng.integers(14, 800, size=len(cids)),
        "occupancy_rate_5m": rng.uniform(0.15, 0.95, size=len(cids)),
        "adequate_date_span": rng.integers(30, 900, size=len(cids)),
        "adequate_day_density": rng.uniform(0.1, 1.0, size=len(cids)),
        "daily_occupancy_sd": rng.uniform(0.02, 0.25, size=len(cids)),
        "occupancy_rate_15m": rng.uniform(0.2, 0.98, size=len(cids)),
    }).set_index("customer_id")

    cutpoints = m3.get_allocation_0_cutpoints(fake_info, alloc_df)
    assert len(cutpoints) == 4
    for dim in m3.DESCRIPTORS:
        assert dim in cutpoints
        lo, hi = cutpoints[dim]
        assert lo < hi, f"Lower cut {lo} not less than upper cut {hi} for {dim}"

    # 2. Verify 31 subgroup partitions
    groups, labels = m3.build_subgroup_memberships(fake_info, cutpoints)
    # 1 overall + 4*3 marginal (12) + 2*9 crossed (18) = 31 groups
    assert len(groups) == 31, f"Expected 31 groups, got {len(groups)}"
    assert ("overall", "all") in groups
    assert len(groups[("overall", "all")]) == len(fake_info)

    # Check 4 focal profiles
    for dim, sub, _ in m3.FOCAL_PROFILES:
        assert (dim, sub) in groups, f"Focal profile ({dim}, {sub}) missing from groups"

    print("  Subgroup cutpoints and memberships PASSED.")


def test_mock_prediction_generator():
    print("Testing mock prediction generator...", flush=True)

    alloc_df = pd.read_parquet(m3.ALLOC_PATH).set_index("customer_id")
    fake_info = pd.DataFrame(index=alloc_df.index)

    mock_df = m3.generate_mock_predictions(
        alloc_df=alloc_df.reset_index(),
        info=fake_info,
        allocations=[0],
        resolutions=["5m"],
        budgets=["all_days"],
        penalty_types=["ridge_selected_ext"],
        seed=123,
    )

    expected_cols = ["user_id", "y", "allocation", "resolution", "budget", "arm", "penalty_type", "score"]
    for col in expected_cols:
        assert col in mock_df.columns, f"Missing column {col} in mock predictions"

    # Test count = 1671 per allocation
    te_cids = alloc_df[alloc_df["alloc_0"] == "test"].index
    assert len(te_cids) == 1671

    # Check all 8 arms present
    assert set(mock_df["arm"].unique()) == set(m3.ARMS)

    # Check AUROC reasonable for H_clock
    h_clock_sub = mock_df[mock_df["arm"] == "H_clock"]
    auc_clock = roc_auc_score(h_clock_sub["y"], h_clock_sub["score"])
    assert 0.80 <= auc_clock <= 0.90, f"H_clock mock AUROC {auc_clock:.4f} outside expected [0.80, 0.90]"

    # Check M_static AUROC lower than H_clock
    m_stat_sub = mock_df[mock_df["arm"] == "M_static"]
    auc_static = roc_auc_score(m_stat_sub["y"], m_stat_sub["score"])
    assert 0.60 <= auc_static <= 0.70, f"M_static mock AUROC {auc_static:.4f} outside expected [0.60, 0.70]"

    print("  Mock prediction generator PASSED.")


def test_compute_subgroup_aurocs_and_contrasts():
    print("Testing compute_subgroup_aurocs and compute_paired_contrasts...", flush=True)

    # Create small synthetic test set
    n_pts = 60
    cids = [f"cid_{i}" for i in range(n_pts)]
    y = np.array([0] * (n_pts // 2) + [1] * (n_pts // 2))

    records = []
    for alloc in [0, 1]:
        for res in ["5m", "15m"]:
            for arm in m3.ARMS:
                base_shift = 0.5 if arm == "H_clock" else (0.4 if arm == "H_global" else 0.6)
                res_shift = 0.1 if res == "5m" else 0.0
                scores = (y - 0.5) * (base_shift + res_shift) + np.linspace(-0.2, 0.2, n_pts)
                for cid, true_y, sc in zip(cids, y, scores):
                    records.append({
                        "user_id": cid,
                        "y": true_y,
                        "allocation": alloc,
                        "resolution": res,
                        "budget": "all_days",
                        "arm": arm,
                        "penalty_type": "ridge_selected_ext",
                        "score": sc,
                    })

    preds_df = pd.DataFrame(records)
    groups = {
        ("overall", "all"): pd.Index(cids),
        ("continuity_x_occupancy", "low/low"): pd.Index(cids[:30]),
    }

    metrics = m3.compute_subgroup_aurocs(preds_df, groups)
    assert len(metrics) > 0

    contrasts = m3.compute_paired_contrasts(metrics)
    assert len(contrasts) > 0

    # Verify sign agreement string format
    for sign_str in contrasts["sign_agreement"]:
        assert "/" in sign_str or "tie" in sign_str, f"Unexpected sign agreement format: {sign_str}"

    # Verify H_global vs H_clock delta is negative (since H_global had base_shift 0.4 vs H_clock 0.5)
    g_vs_c = contrasts[contrasts["contrast_name"] == "H_global_minus_H_clock"]
    assert len(g_vs_c) > 0
    assert (g_vs_c["delta_mean"] < 0).all()

    # Verify resolution contrast 5m - 15m is positive (since 5m had res_shift 0.1)
    res_contrasts = contrasts[contrasts["contrast_category"] == "resolution_contrast"]
    assert len(res_contrasts) > 0
    assert (res_contrasts["delta_mean"] > 0).all()

    print("  compute_subgroup_aurocs and compute_paired_contrasts PASSED.")


def test_spearman_rank_correlations():
    print("Testing compute_spearman_rank_correlations...", flush=True)

    n_pts = 50
    cids = [f"cid_{i}" for i in range(n_pts)]
    records = []
    for arm in m3.ARMS:
        scores = np.arange(n_pts, dtype=float) if arm == "H_clock" else np.arange(n_pts, dtype=float) + np.random.randn(n_pts) * 0.1
        for cid, sc in zip(cids, scores):
            records.append({
                "customer_id": cid,
                "y": 1,
                "allocation": 0,
                "resolution": "5m",
                "budget": "all_days",
                "penalty_type": "ridge_selected_ext",
                "arm": arm,
                "score": sc,
            })

    preds_df = pd.DataFrame(records)
    groups = {("overall", "all"): pd.Index(cids)}

    spear_summary = m3.compute_spearman_rank_correlations(preds_df, groups)
    assert len(spear_summary) == len(m3.ARMS)

    # H_clock vs H_clock must be exactly 1.0
    clock_row = spear_summary[spear_summary["arm"] == "H_clock"].iloc[0]
    assert np.isclose(clock_row["rho_mean"], 1.0, atol=1e-8)

    print("  compute_spearman_rank_correlations PASSED.")


def test_scientific_report_generation():
    print("Testing generate_report_markdown...", flush=True)

    # Construct minimal summary and contrast dfs
    subgroup_summary = pd.DataFrame([{
        "budget": "all_days",
        "penalty_type": "ridge_selected_ext",
        "resolution": res,
        "arm": arm,
        "dimension": "overall",
        "subgroup": "all",
        "n_allocations": 3,
        "n_test_mean": 1671.0,
        "n_class_0_mean": 580.0,
        "n_class_1_mean": 1091.0,
        "auroc_mean": 0.8678 if res == "5m" else 0.8454,
        "auroc_sd": 0.0080,
        "auroc_alloc_0": 0.8678,
        "auroc_alloc_1": 0.8650,
        "auroc_alloc_2": 0.8706,
        "ap_mean": 0.9100,
        "ap_sd": 0.0050,
    } for res in ["5m", "15m"] for arm in m3.ARMS])

    paired_contrasts = pd.DataFrame([{
        "budget": "all_days",
        "penalty_type": "ridge_selected_ext",
        "resolution": "5m",
        "contrast_category": "clock_vs_global",
        "contrast_name": "H_global_minus_H_clock",
        "comparison_arm": "H_global",
        "baseline_arm": "H_clock",
        "dimension": "overall",
        "subgroup": "all",
        "n_allocations": 3,
        "delta_mean": -0.0092,
        "delta_sd": 0.0015,
        "delta_alloc_0": -0.0092,
        "delta_alloc_1": -0.0085,
        "delta_alloc_2": -0.0099,
        "sign_agreement": "3/3 -",
    }])

    spearman_summary = pd.DataFrame([{
        "budget": "all_days",
        "penalty_type": "ridge_selected_ext",
        "resolution": "5m",
        "arm": arm,
        "reference_arm": "H_clock",
        "dimension": "overall",
        "subgroup": "all",
        "n_allocations": 3,
        "rho_mean": 1.0 if arm == "H_clock" else 0.95,
        "rho_sd": 0.01,
    } for arm in m3.ARMS])

    cutpoints = {
        "history": (153.0, 356.0),
        "occupancy": (0.4531, 0.6320),
        "continuity": (0.5351, 0.9617),
        "occupancy_variability": (0.0882, 0.1191),
    }

    report = m3.generate_report_markdown(subgroup_summary, paired_contrasts, spearman_summary, cutpoints, n_allocs=3)
    assert len(report) > 500

    # Calibrated language checks: no p-values or population CIs
    assert "p-value" not in report.lower()
    assert "p <" not in report
    assert "95% ci" not in report.lower()
    assert "confidence interval" not in report.lower()

    # R1-R5 methodological points present
    assert "R1" in report
    assert "R2" in report
    assert "R3" in report
    assert "R4" in report
    assert "R5" in report
    assert "sign agreement" in report.lower()

    print("  generate_report_markdown PASSED.")


def main():
    print("=" * 70)
    print("RUNNING SUBGROUP CONTRASTS UNIT TEST SUITE")
    print("=" * 70)
    test_subgroup_cutpoints_and_memberships()
    test_mock_prediction_generator()
    test_compute_subgroup_aurocs_and_contrasts()
    test_spearman_rank_correlations()
    test_scientific_report_generation()
    print("=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
