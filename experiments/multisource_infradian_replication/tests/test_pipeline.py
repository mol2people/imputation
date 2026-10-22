"""Unit tests for the multisource implementation primitives."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisource import bootstrap, channel3000, common, daily, endpoints, longitudinal, matched, models, provenance, reml


def test_source_namespace_is_pg_daily():
    mapping = common.load_source_mapping()
    ids = dict(zip(mapping["id"], mapping["code"]))
    assert ids[2] == "Fitbit"
    assert ids[3] == "Garmin"
    assert ids[6] == "Apple"
    assert ids[7] == "Samsung"
    assert ids[9] == "Nokia"
    assert ids[13] == "GoogleFit"
    assert ids[19] == "Oura"
    assert ids[4] == "Polar"


def test_value_type_namespace():
    mapping = common.load_value_type_mapping()
    ids = dict(zip(mapping["id"], mapping["code"]))
    assert ids[65] == "RHR"
    assert ids[66] == "HR during sleep"


def test_duplicate_policy_exact_and_conflict():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "source": [65, 65, 65, 65],
            "endpoint": ["RHR"] * 4,
            "date": [10, 10, 11, 11],
            "value": [60, 60, 61, 70],
        }
    )
    collapsed, meta = provenance.collapse_duplicates(df)
    assert meta["exact_duplicate_rows"] == 1
    assert meta["conflicting_keys"] == 1
    assert len(collapsed) == 1
    assert collapsed.iloc[0]["user_id"] == 1


def test_daily_qc_range():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "source": [65, 65, 65],
            "endpoint": ["RHR"] * 3,
            "date": [1, 2, 3],
            "value": [60.0, 24.0, np.nan],
        }
    )
    out, meta = provenance.apply_daily_qc(df)
    assert len(out) == 1
    assert meta["out_of_range_rows"] == 1
    assert meta["nonfinite_rows"] == 1


def test_expanded90_window_selection():
    # 16 qualifying dates in each of six consecutive 30-day blocks.
    dates = []
    for b in range(6):
        dates.extend(range(b * 30, b * 30 + 16))
    arr = np.asarray(sorted(dates), dtype=np.int64)
    windows, selected = daily.build_windows_for_arm(
        pd.DataFrame(
            {
                "user_id": 7,
                "source": 65,
                "endpoint": "RHR",
                "date": arr,
                "value": np.full(len(arr), 60.0),
            }
        ),
        source=65,
        endpoint="RHR",
        protocols=("expanded90",),
    )
    row = windows.loc[windows["is_eligible"]].iloc[0]
    assert row["start_day"] == 0
    assert row["n_selected"] == 90
    assert selected["date"].nunique() == 90
    assert selected.groupby("block_index").size().eq(15).all()


def test_balanced90_requires_15_per_block():
    dates = []
    for b in range(6):
        dates.extend(range(b * 30, b * 30 + 12))
    arr = np.asarray(sorted(dates), dtype=np.int64)
    windows, _ = daily.build_windows_for_arm(
        pd.DataFrame(
            {
                "user_id": 7,
                "source": 65,
                "endpoint": "RHR",
                "date": arr,
                "value": np.full(len(arr), 60.0),
            }
        ),
        source=65,
        endpoint="RHR",
        protocols=("balanced90",),
    )
    assert not windows["is_eligible"].any()


def test_daily_nuisance_has_no_coverage_columns():
    dates = np.arange(90, dtype=np.int64)
    P0, Q, p = endpoints.build_daily_nuisance(dates)
    # intercept + elapsed + 6 weekday dummies = 8 raw columns; rank <= 8.
    assert p <= 8
    assert P0.shape == (90, 90)


def test_daily_endpoint_deterministic():
    rng = np.random.default_rng(0)
    dates = np.arange(0, 180, 2, dtype=np.int64)
    values = 60 + 3 * np.sin(2 * np.pi * dates / 28.0) + rng.normal(0, 1, len(dates))
    task = {
        "user_id": 1,
        "source": 2,
        "endpoint": "RHR",
        "arm": "RHR",
        "demographics": {
            "salutation": 10, "birth_year": 1980, "age_dec1": 40,
            "age_jan1": 40, "age_dec31": 40, "weight": np.nan, "height": np.nan, "bmi": np.nan,
        },
        "windows": [{
            "protocol": "expanded90", "window_index": 1, "start_day": 0,
            "selected_dates_hash": "x", "dates": dates.tolist(),
            "daily_values": dict(zip(dates.tolist(), values.tolist())),
        }],
        "null_replicates": 8,
    }
    a = endpoints.process_daily_task(task)[0]
    b = endpoints.process_daily_task(task)[0]
    assert a["status"] == "ok"
    assert np.isclose(a["Q_i"], b["Q_i"])
    assert np.isclose(a["E_i2"], b["E_i2"])


def test_reml_gradient_matches_finite_difference():
    rng = np.random.default_rng(42)
    dates = np.arange(90, dtype=np.int64)
    P0, _, _ = endpoints.build_daily_nuisance(dates)
    eigvals, U = np.linalg.eigh(P0)
    Q_sub = U[:, eigvals > 0.5]
    z = rng.normal(size=Q_sub.shape[1])
    r_rank = int(len(z))
    phi2 = 0.98
    dt_abs = np.abs(dates[:, None] - dates[None, :]).astype(np.int64)
    powers_idx = np.arange(int(dt_abs.max()) + 1, dtype=np.float64)
    K2 = (phi2 ** powers_idx)[dt_abs]
    S2 = 0.5 * (Q_sub.T @ (K2 @ Q_sub) + (Q_sub.T @ (K2 @ Q_sub)).T)
    s_init = float(z @ z) / r_rank
    theta = np.array([np.log(0.5 * s_init), 1.0, np.log(0.5 * s_init)])

    _, grad = reml._objective_and_gradient(
        theta, z, Q_sub, dt_abs, powers_idx, S2, r_rank
    )
    eps = 1e-6
    numerical = np.zeros(3)
    for k in range(3):
        plus = theta.copy()
        minus = theta.copy()
        plus[k] += eps
        minus[k] -= eps
        f_plus, _ = reml._objective_and_gradient(
            plus, z, Q_sub, dt_abs, powers_idx, S2, r_rank
        )
        f_minus, _ = reml._objective_and_gradient(
            minus, z, Q_sub, dt_abs, powers_idx, S2, r_rank
        )
        numerical[k] = (f_plus - f_minus) / (2.0 * eps)
    assert np.allclose(numerical, grad, rtol=1e-5, atol=1e-6)


def test_channel_values_prefers_cached_hrp10():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "customer_1_source_3.npz"
        np.savez(
            path,
            date=np.arange(3, dtype=np.int64),
            hr=np.full((3, common.DENSE_BINS_PER_DAY), 60.0, dtype=np.float32),
            hrp10=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        out = matched._channel_values(str(path), np.array([0, 2]))
        assert out == {0: 1.0, 2: 3.0}


def test_channel_cache_hrp10_ignores_missing_bins():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw"
        cache = root / "cache"
        raw.mkdir()
        start_ms = int(np.datetime64("2020-01-01T00:00:00", "ms").astype(np.int64))
        values = np.linspace(50.0, 70.0, common.DENSE_MIN_BINS)
        rows = ["customer,startTimestamp,timezoneOffset,longValue,source,type"]
        rows.extend(
            f"1,{start_ms + i * 300_000},0,{value},3,3000"
            for i, value in enumerate(values)
        )
        (raw / "1.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        index = channel3000.build_caches(
            pd.DataFrame({"user_id": [1], "source": [3], "dense_rows": [len(values)]}),
            cache,
            raw_dir=raw,
            workers=1,
        )
        assert len(index) == 1
        with np.load(cache / "customer_1_source_3.npz") as d:
            assert d["hrp10"].shape == (1,)
            assert np.isfinite(d["hrp10"][0])
            expected = np.percentile(values.astype(np.float32), common.HRP10_PERCENTILE)
            assert np.isclose(d["hrp10"][0], expected, rtol=0.0, atol=1e-5)


def test_chunk_contrast_shapes():
    contrasts = models.chunk_contrasts(8)
    assert contrasts[1][1].shape == (3, 8)
    assert contrasts[2][1].shape == (2, 8)
    assert contrasts[3][1].shape == (3, 8)
    assert contrasts[4][1].shape == (2, 8)


def test_duplicate_prefers_finite_value():
    df = pd.DataFrame(
        {
            "user_id": [1, 1],
            "source": [2, 2],
            "endpoint": ["RHR", "RHR"],
            "date": [10, 10],
            "value": [np.nan, 60.0],
        }
    )
    collapsed, meta = provenance.collapse_duplicates(df)
    assert meta["conflicting_keys"] == 0
    assert len(collapsed) == 1
    assert collapsed.iloc[0]["value"] == 60.0


def test_support_mask_and_female_bands():
    ages = np.array([22.0, 40.0, 70.0])
    mask = models.support_mask(ages, ["[20,25)", "[40,45)"])
    assert mask.tolist() == [True, True, False]
    ages = [22.0] * 30 + [32.0] * 30 + [42.0] * 30
    df = pd.DataFrame({"sex": [1] * 90 + [0] * 90, "age": ages + ages})
    assert "[20,25)" in models.female_support_bands(df)
    assert models.shared_support(df)[1] == ["[20,25)", "[30,35)", "[40,45)"]


def test_transition_metrics_boundary_is_undefined():
    theta = np.zeros(8)
    met = bootstrap.transition_metrics(theta, ["[30,35)", "[35,40)", "[40,45)"])
    assert np.isnan(met["inflection_age"])
    assert np.isnan(met["inflection_slope"])
    assert np.isnan(met["zero_crossing_age"])


def test_paired_bootstrap_zero_difference_with_identical_draws():
    rng = np.random.default_rng(8)
    n = 1600
    age = rng.integers(22, 72, n).astype(float)
    sex = rng.integers(0, 2, n)
    q = np.exp(0.02 + 0.0004 * (age - 40) + 0.03 * sex + rng.normal(0, 0.02, n))
    frame = pd.DataFrame(
        {
            "user_id": np.arange(n) + 1,
            "source": 6,
            "endpoint": "HRp10",
            "age": age,
            "sex": sex,
            "Q_i": q,
            "window_index": 1,
        }
    ).sort_values(["user_id", "window_index"]).reset_index(drop=True)
    bands = ["[30,35)", "[35,40)", "[40,45)", "[45,50)"]
    draws = bootstrap.draw_indices(frame, 5, 99)
    a = bootstrap.bootstrap_frame(
        frame, "Q_i", "HRp10", 6, "expanded90", 1, 5, 99,
        shared_bands=bands, draws=draws, scope="sex",
    )
    b = bootstrap.bootstrap_frame(
        frame, "Q_i", "HRp10", 6, "expanded90", 1, 5, 99,
        shared_bands=bands, draws=draws, scope="sex",
    )
    from multisource import matched
    contrast = matched._paired_difference(
        a["curves"], b["curves"], 6, "RHR", "expanded90"
    )
    direct = matched._paired_difference_arrays(a, b, 6, "RHR", "expanded90")
    assert not contrast.empty
    pd.testing.assert_frame_equal(contrast, direct, check_dtype=False, check_exact=False,
                                  rtol=1e-6, atol=1e-7)
    assert np.nanmax(np.abs(contrast["difference"])) == 0.0
    assert np.nanmax(np.abs(contrast["hi"] - contrast["lo"])) == 0.0
    failed = []
    for result in (a, b):
        copy = {**result, "curves": result["curves"].copy(),
                "arrays": {k: v.copy() for k, v in result["arrays"].items()}}
        for values in copy["arrays"].values():
            values[0, :] = np.nan
        copy["curves"].loc[copy["curves"]["draw"].eq(0), "prediction"] = np.nan
        failed.append(copy)
    table_reference = matched._paired_difference(
        failed[0]["curves"], failed[1]["curves"], 6, "RHR", "expanded90"
    )
    array_result = matched._paired_difference_arrays(
        failed[0], failed[1], 6, "RHR", "expanded90"
    )
    pd.testing.assert_frame_equal(table_reference, array_result, check_dtype=False,
                                  check_exact=False, rtol=1e-6, atol=1e-7)


def test_curve_only_bootstrap_matches_full_predictions():
    rng = np.random.default_rng(19)
    n = 900
    age = rng.integers(22, 72, n).astype(float)
    sex = rng.integers(0, 2, n)
    frame = pd.DataFrame({
        "user_id": np.arange(n) + 1,
        "age": age,
        "sex": sex,
        "Q_i": np.exp(0.02 + 0.0004 * (age - 40) + 0.03 * sex + rng.normal(0, 0.02, n)),
        "window_index": 1,
    })
    bands = ["[30,35)", "[35,40)", "[40,45)", "[45,50)"]
    draws = bootstrap.draw_indices(frame, 4, 29)
    full = bootstrap.bootstrap_frame(
        frame, "Q_i", "HRp10", 6, "expanded90", 1, 4, 29,
        shared_bands=bands, draws=draws, scope="sex",
    )
    curve = bootstrap.bootstrap_frame(
        frame, "Q_i", "HRp10", 6, "expanded90", 1, 4, 29,
        shared_bands=bands, draws=draws, scope="sex",
        return_curves=False, curve_only=True,
    )
    no_curves = bootstrap.bootstrap_frame(
        frame, "Q_i", "HRp10", 6, "expanded90", 1, 4, 29,
        shared_bands=bands, draws=draws, scope="sex", return_curves=False,
    )
    assert full["curves"].shape[0] > 0
    assert curve["curves"].empty
    assert curve["test_stats"] == {}
    assert curve["metrics"] == {}
    assert no_curves["curves"].empty
    assert no_curves["test_stats"]
    assert full["arrays"].keys() == curve["arrays"].keys()
    for scope in full["arrays"]:
        np.testing.assert_array_equal(full["arrays"][scope], curve["arrays"][scope])
        np.testing.assert_array_equal(full["point"][scope], curve["point"][scope])


def test_simultaneous_band_uses_bootstrap_se():
    ages = np.array([30.0, 30.25, 30.5])
    point = np.array([1.0, 2.0, 3.0])
    se = np.array([0.01, 0.01, 0.01])
    rows = []
    for d in range(20):
        offset = np.array([0.05, -0.05, 0.02]) * (1.0 if d % 2 == 0 else -1.0)
        rows.append(pd.DataFrame({
            "source": 3, "endpoint": "RHR", "metric": "Q_i", "protocol": "expanded90",
            "window_index": 1, "scope": "female", "draw": d, "age": ages,
            "prediction": point + offset, "se": se, "supported": True,
        }))
    rows.append(pd.DataFrame({
        "source": 3, "endpoint": "RHR", "metric": "Q_i", "protocol": "expanded90",
        "window_index": 1, "scope": "female", "draw": -1, "age": ages,
        "prediction": point, "se": se, "supported": True,
    }))
    bands = bootstrap.summarize_bands(pd.concat(rows, ignore_index=True))
    assert len(bands) == len(ages)
    # Studentization must use the bootstrap SE, not the model HC3 SE.
    expected_boot_se = np.std(
        np.vstack([
            point + np.array([0.05, -0.05, 0.02]) * (1.0 if d % 2 == 0 else -1.0)
            for d in range(20)
        ]), axis=0, ddof=1
    )
    assert np.allclose(bands["boot_se"].to_numpy(), expected_boot_se)
    assert np.allclose(bands["hc3_se"].to_numpy(), se)
    assert np.allclose(
        bands["sim_lo"].to_numpy(),
        point - bands["sim_critical_value"].to_numpy() * expected_boot_se,
    )


def test_channel_tasks_filter_requested_source():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        for uid, src in ((101, 2), (202, 6)):
            np.savez(
                cache_dir / f"customer_{uid}_source_{src}.npz",
                date=np.arange(90, dtype=np.int64),
                hr=np.full((90, common.DENSE_BINS_PER_DAY), 60.0, dtype=np.float32),
            )
        rows = []
        for uid, src, digest in ((101, 2, "a"), (202, 6, "b")):
            for d in range(90):
                rows.append({
                    "user_id": uid, "source": src, "protocol": "expanded90",
                    "window_index": 1, "selected_dates_hash": digest, "date": d,
                })
        selected = pd.DataFrame(rows)
        windows = pd.DataFrame({
            "user_id": [101, 202], "source": [2, 6], "protocol": ["expanded90"] * 2,
            "window_index": [1, 1], "is_eligible": [True, True], "start_day": [0, 0],
        })
        demographics = pd.DataFrame({
            "user_id": [101, 202], "salutation": [10, 20], "birth_year": [1980, 1970],
            "weight": [70.0, 80.0], "height": [170.0, 180.0], "bmi": [24.2, 24.7],
        })
        tasks = channel3000.build_channel_tasks(
            selected, windows, demographics, cache_dir, "expanded90", source=2,
        )
        assert len(tasks) == 1
        assert tasks[0]["source"] == 2
        assert tasks[0]["user_id"] == 101


def test_raw_scan_all_finds_source_tagged_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "1000001.csv"
        path.write_text(
            "customer,type,source,longValue\n"
            "1000001,3000,2,60\n"
            "1000001,3000,2,61\n"
            "1000001,65,2,58\n"
            "1000002,3000,19,55\n",
            encoding="utf-8",
        )
        index = channel3000.scan_raw_index_all(Path(tmp), sources=(2, 19), workers=1)
        got = {(int(r.user_id), int(r.source)): int(r.dense_rows) for r in index.itertuples()}
        assert got == {(1000001, 2): 2, (1000002, 19): 1}


def test_common_support_table_counts_shared_bands():
    df = pd.DataFrame({
        "sex": [1, 0, 1, 0, 0],
        "age": [26.0, 27.0, 31.0, 32.0, 71.0],
    })
    table = matched._common_support_table(
        df, ["[25,30)", "[30,35)"], 3, "HRp10", "expanded90", 1
    )
    by_band = table.set_index("age_band")
    assert by_band.loc["[25,30)", "female_count"] == 1
    assert by_band.loc["[25,30)", "male_count"] == 1
    assert by_band.loc["[30,35)", "female_count"] == 1
    assert by_band.loc["[30,35)", "male_count"] == 1
    assert by_band.loc["[70,75)", "female_count"] == 0
    assert by_band.loc["[70,75)", "male_count"] == 1
    assert by_band.loc["[25,30)", "is_shared_support"]
    assert not by_band.loc["[70,75)", "is_shared_support"]


def test_reml_continuity_fallback_calls_historical_fitter():
    """Ambiguous low-phi1 versus canonical basins must use the reference fitter.

    The production guard is what keeps the faster L-BFGS-B path numerically
    continuous with the historical Nelder-Mead endpoints when both optimizers
    expose genuinely different local optima.
    """
    from unittest.mock import patch

    from scipy.optimize import OptimizeResult

    z = np.array([1.0, 0.5, -0.2], dtype=np.float64)
    Q_sub = np.eye(3, dtype=np.float64)
    dates = np.array([0, 1, 2], dtype=np.int64)

    def fake_run_start(theta0, *args, **kwargs):
        # High-phi1 canonical starts are assigned a worse objective than the
        # low-phi1 anchor, making the multi-basin ambiguity explicit.
        nll = 5.0 if float(theta0[1]) < -1.0 else 6.0
        return OptimizeResult(
            success=True,
            fun=nll,
            x=np.asarray(theta0, dtype=np.float64),
            jac=np.zeros(3, dtype=np.float64),
            nit=1,
        )

    calls = []

    class Engine:
        @staticmethod
        def fit_two_component_reml(*args, **kwargs):
            calls.append(args)
            return "converged", {
                "v1": 1.0,
                "phi1": 0.1,
                "v2": 1.0,
                "phi2": 0.98,
                "reml_nll": 6.0,
                "num_starts_converged": 6,
                "best_start_idx": 1,
                "nit": 10,
                "reml_elapsed_sec": 0.01,
                "s_init": 1.0 / 3.0,
                "phi_init": 0.5,
            }

    with patch.dict("os.environ", {"MULTISOURCE_REML_CONTINUITY": "1"}), \
         patch.object(reml, "_run_start", fake_run_start), \
         patch.object(common, "load_endpoint_engine", lambda: Engine()):
        status, diag = reml.fit_two_component_reml_lbfgsb(
            z, Q_sub, dates, 1.0, 3, 0.5, adaptive_screening=True
        )

    assert status == "converged"
    assert diag["optimizer"] == "nelder_continuity"
    assert diag["continuity_fallback"] is True
    assert calls


def test_matched_injects_active_fitter():
    """Stage 7 must not silently fall back to the historical solver."""
    from unittest.mock import patch

    task = {
        "user_id": 101, "source": 2, "start_day": 17000,
        "protocol": "expanded90", "window_index": 1,
        "selected_dates_hash": "abc", "dates": [0, 1],
        "hrp10": {0: 60.0, 1: 61.0},
        "daily": {0: 59.0, 1: 60.0},
    }
    demographics = pd.DataFrame({
        "user_id": [101], "salutation": [10], "birth_year": [1980],
        "weight": [70.0], "height": [170.0], "bmi": [24.2],
    })
    captured = {}

    def fake_run_daily_arm(work, tmp, null_replicates, workers):
        captured["work"] = work
        return pd.DataFrame()

    with patch.dict("os.environ", {"MULTISOURCE_REML_OPTIMIZER": "lbfgsb"}), \
         patch.object(matched.endpoints, "run_daily_arm", fake_run_daily_arm):
        matched._run_arm_on_matched(
            [task], "matched", "hrp10", demographics, null_replicates=5, workers=1
        )

    assert captured["work"][0]["reml_fitter"] is reml.fit_two_component_reml_lbfgsb


def test_cholesky_solvers_match_pinv_reference():
    rng = np.random.default_rng(7)
    X = np.column_stack([np.ones(80), rng.normal(size=(80, 5))])
    y = X @ rng.normal(size=6) + rng.normal(size=80)
    theta, cov, resid, df_resid = models.ols_hc3(X, y)

    xtx_inv = np.linalg.pinv(X.T @ X)
    theta_ref = xtx_inv @ (X.T @ y)
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    h = np.clip(h, 0.0, 1.0 - 1e-12)
    weights = (y - X @ theta_ref) ** 2 / (1.0 - h) ** 2
    cov_ref = xtx_inv @ (X.T @ (X * weights[:, None])) @ xtx_inv

    np.testing.assert_allclose(theta, theta_ref, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(cov, cov_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(resid, y - X @ theta_ref, rtol=1e-11, atol=1e-12)
    assert df_resid == 74

    C = np.zeros((2, 6)); C[0, 1] = C[1, 2] = 1.0
    got = models.chunk_wald(theta, cov, C, df_resid)["wald_stat"]
    ctheta = C @ theta_ref
    middle = C @ cov_ref @ C.T
    ref = float(ctheta.T @ np.linalg.pinv(middle) @ ctheta)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_transition_bases_are_precomputed_exactly():
    assert np.array_equal(
        bootstrap.D_RCS_GRID, bootstrap._d_rcs(bootstrap.TRANSITION_GRID)
    )
    assert np.array_equal(
        bootstrap.DELTA_DESIGN_ZGRID,
        bootstrap._delta_design(bootstrap.TRANSITION_ZGRID),
    )


def test_longitudinal_pairing_and_summary():
    n = 12
    rows = []
    for window, start, offset in ((1, 0, 0.0), (2, 200, 0.02)):
        for i in range(n):
            rows.append({
                "user_id": i + 1,
                "source": 3,
                "endpoint": "RHR",
                "protocol": "balanced90",
                "window_index": window,
                "start_day": start,
                "selected_dates_hash": f"w{window}-{i}",
                "age": 25.0 + i,
                "sex": i % 2,
                "null_status": "converged",
                "status": "ok",
                "Q_i": 0.01 + 0.0002 * i + offset,
                "Q_i_mc_se": 0.001,
                "E_i2": 0.02 + 0.0001 * i + offset,
                "A_hat": 1.0 + 0.01 * i + offset,
                "sigma_resid": 2.0 + 0.01 * i,
                "peak_period": 28.0 + 0.1 * (i % 3),
            })
    endpoints_table = pd.DataFrame(rows)
    paired = longitudinal.build_paired_windows(endpoints_table)
    assert len(paired) == n
    assert np.allclose(
        paired["Q_i_change"].to_numpy(),
        paired["Q_i_w2"].to_numpy() - paired["Q_i_w1"].to_numpy(),
    )
    assert np.allclose(
        paired["Q_i_annualized_rate"].to_numpy(),
        paired["Q_i_change"].to_numpy() / (200.0 / 365.25),
    )
    availability = longitudinal.availability_table(endpoints_table, paired)
    assert int(availability.loc[0, "paired_converged_Q"]) == n
    summary = longitudinal.summarize(paired, n_boot=20, seed=7)
    all_row = summary.loc[summary["subgroup"].eq("all")].iloc[0]
    assert int(all_row["n"]) == n
    assert np.isclose(all_row["mean_Q_change"], 0.02)
    assert np.isclose(all_row["mean_Q_annualized_rate"], 0.02 / (200.0 / 365.25))
    assert all_row["pearson_Q_i"] > 0.99


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
