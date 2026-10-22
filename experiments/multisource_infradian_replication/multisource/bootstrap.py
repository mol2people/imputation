"""Stage 5: stratified participant bootstrap for age curves and chunk tests.

Resampling is at the participant level, stratified by 5-year age band x sex.
The same draw indices can be reused across compared arms for paired contrasts.

Implemented optimizations (PLAN.md §5.2 / EFFICIENCY_AUDIT.md §4):

* **P0 invariant design.**  The restricted cubic spline design ``X`` depends
  only on the fixed participant set ``(age_i, sex_i)``.  It is built once per
  arm; a bootstrap draw is the pure row slice ``X_b = X[draws[b], :]``.  The
  per-draw ``rcs_stone_koo`` evaluations are eliminated (audit §3.1.A).
* **P0 array storage.**  Bootstrap predictions are accumulated as float32
  ``(N_boot, N_ages)`` matrices per ``(arm, metric, scope)`` and persisted as a
  compressed ``.npz`` archive.  No per-draw DataFrames and no ``pivot_table``
  are used for storage (audit §3.1.D).
* **P1 no per-draw HC3 standard errors.**  Only the observed fit's pointwise
  HC3 SE is retained.  The simultaneous bands studentize with the empirical
  across-draw bootstrap SE (audit §3.1.C).
* **P1 cheap rank gate.**  ``np.linalg.matrix_rank`` (full ``_gesdd``) is
  replaced by a Cholesky positive-definiteness test on ``X_b^T X_b``
  (audit §3.1.B).

The long-form curve table remains available for callers that need paired,
age-resolved draws (``matched._paired_difference`` and ``summarize_bands``);
it is built once, with compact dtypes, only when requested.
"""
from __future__ import annotations

import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import common, models

AGE_GRID = models.AGE_GRID
BAND_EDGES = models.BAND_EDGES
BAND_LABELS = models.BAND_LABELS

# Scope labels written to the long curve table.  The sex-interaction model
# writes ``sex_female`` / ``sex_male`` because ``matched._paired_difference``
# selects exactly those labels.
SCOPE_LABELS: dict[str, tuple[str, ...]] = {
    "both": ("female", "sex_female", "sex_male"),
    "female": ("female",),
    "sex": ("sex_female", "sex_male"),
}
SEX_MODEL_SCOPES: tuple[tuple[float, str], ...] = (
    (1.0, "sex_female"),
    (0.0, "sex_male"),
)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _strata(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Age-band labels and numeric sex codes, row-aligned with ``df``."""
    age = pd.to_numeric(df["age"], errors="coerce").to_numpy(float)
    labels = models.band_labels_for_ages(age)
    sex = pd.to_numeric(df["sex"], errors="coerce").to_numpy(float)
    return labels, sex


def draw_indices(df: pd.DataFrame, n_boot: int, seed: int) -> np.ndarray:
    """Return ``(n_boot, n)`` row indices preserving stratum counts per draw.

    ``draws[b, j]`` is the replacement row selected for row ``j`` from ``j``'s
    own ``(age_band, sex)`` stratum.  The same array can therefore be reused
    across paired arms whose rows are aligned.

    The matrix is deterministic given ``(row order, n_boot, seed)``.  The
    vectorized ``rng.choice`` call replaces the former per-draw/per-stratum
    loop, so it does not reproduce an older RNG stream bit-for-bit; it does
    reproduce identical draws across paired arms, which is the requirement.
    """
    n = len(df)
    n_boot = int(n_boot)
    if n == 0:
        return np.empty((n_boot, 0), dtype=np.int64)
    labels, sex = _strata(df)
    order = np.lexsort((sex, labels))
    lab_sorted = labels[order]
    sex_sorted = sex[order]
    if n > 1:
        cut = np.flatnonzero(
            (lab_sorted[1:] != lab_sorted[:-1])
            | (sex_sorted[1:] != sex_sorted[:-1])
        ) + 1
        groups = np.split(order, cut)
    else:
        groups = [order]
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    draws = np.empty((n_boot, n), dtype=np.int64)
    for idx in groups:
        draws[:, idx] = rng.choice(idx, size=(n_boot, len(idx)), replace=True)
    return draws


# ---------------------------------------------------------------------------
# Design / prediction helpers
# ---------------------------------------------------------------------------


def _grid_designs() -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Evaluation-grid designs, built once per call (not per draw)."""
    female_grid, _ = models.build_design(AGE_GRID)
    sex_grid = {
        value: models.build_design(
            AGE_GRID, np.full(len(AGE_GRID), value, dtype=float)
        )[0]
        for value, _ in SEX_MODEL_SCOPES
    }
    return female_grid, sex_grid


def _point_se(grid: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Pointwise SE of a linear predictor on the evaluation grid."""
    var = np.einsum("ij,jk,ik->i", grid, cov, grid)
    return np.sqrt(np.maximum(var, 0.0))


def _full_rank(X: np.ndarray) -> bool:
    """Cheap full-rank gate: ``X^T X`` must be positive definite.

    This replaces a full SVD (``np.linalg.matrix_rank``) on every resample.
    A draw that fails the gate is skipped, leaving NaNs in that row.
    """
    try:
        np.linalg.cholesky(X.T @ X)
    except np.linalg.LinAlgError:
        return False
    return True


# ---------------------------------------------------------------------------
# Transition metrics (sex-interaction model only)
# ---------------------------------------------------------------------------


def _d_rcs(ages: np.ndarray, knots=common.DEFAULT_KNOTS) -> np.ndarray:
    """Derivative of ``[1, age, rcs1, rcs2]`` with respect to age."""
    t1, t2, t3, t4 = knots
    norm = float((t4 - t1) ** 2)
    x = np.asarray(ages, dtype=float)

    def pos_sq(u):
        return 3.0 * np.maximum(u, 0.0) ** 2

    c3_1, c4_1 = (t4 - t1) / (t4 - t3), (t3 - t1) / (t4 - t3)
    c3_2, c4_2 = (t4 - t2) / (t4 - t3), (t3 - t2) / (t4 - t3)
    d1 = (pos_sq(x - t1) - c3_1 * pos_sq(x - t3) + c4_1 * pos_sq(x - t4)) / norm
    d2 = (pos_sq(x - t2) - c3_2 * pos_sq(x - t3) + c4_2 * pos_sq(x - t4)) / norm
    return np.column_stack([np.zeros_like(x), np.ones_like(x), d1, d2])


def _delta_design(ages: np.ndarray) -> np.ndarray:
    """Design of ``female - male`` at ``ages`` for the sex-interaction model."""
    z = np.asarray(ages, dtype=float)
    female, _ = models.build_design(z, np.ones_like(z))
    male, _ = models.build_design(z, np.zeros_like(z))
    return female - male


# The transition grids and their spline designs depend only on the fixed
# knots and AGE_GRID, not on the bootstrap draw.  Build them once at module
# import instead of re-evaluating 741 rows of bases for every draw.
TRANSITION_GRID = np.arange(35.0, 65.0 + 1e-9, 0.05)
TRANSITION_ZGRID = np.arange(40.0, 75.0 + 1e-9, 0.25)
D_RCS_GRID = _d_rcs(TRANSITION_GRID)
DELTA_DESIGN_ZGRID = _delta_design(TRANSITION_ZGRID)


def transition_metrics(theta: np.ndarray, bands: list[str] | None = None) -> dict[str, float]:
    """Delta = female - male derivative minimum and downward zero crossing.

    Undefined (NaN) when the extremum or crossing sits on a support boundary,
    or when no positive-to-nonpositive crossing exists.
    """
    result = {
        "inflection_age": np.nan,
        "inflection_slope": np.nan,
        "zero_crossing_age": np.nan,
    }
    theta = np.asarray(theta, dtype=float)
    if theta.size < 8:
        return result

    grid = TRANSITION_GRID
    valid = np.ones(len(grid), dtype=bool)
    if bands is not None:
        valid = models.support_mask(grid, bands)
    idx = np.flatnonzero(valid)
    if len(idx) < 3:
        return result
    delta_deriv = D_RCS_GRID @ np.array([0.0, theta[5], theta[6], theta[7]])
    i_min = int(idx[np.argmin(delta_deriv[idx])])
    if i_min not in (int(idx[0]), int(idx[-1])):
        result["inflection_age"] = float(grid[i_min])
        result["inflection_slope"] = float(delta_deriv[i_min])

    zgrid = TRANSITION_ZGRID
    zvalid = np.ones(len(zgrid), dtype=bool)
    if bands is not None:
        zvalid = models.support_mask(zgrid, bands)
    delta = DELTA_DESIGN_ZGRID @ theta
    positive = (delta > 0.0) & zvalid
    # A crossing requires the immediately preceding supported age to be
    # positive; a series that starts at or below zero is a boundary artifact.
    hits = np.flatnonzero(
        zvalid & (delta <= 0.0) & np.concatenate(([False], positive[:-1]))
    )
    if len(hits) == 0:
        return result
    first = int(hits[0])
    a0, a1 = float(zgrid[first - 1]), float(zgrid[first])
    y0, y1 = float(delta[first - 1]), float(delta[first])
    if abs(y1 - y0) > 1e-12:
        result["zero_crossing_age"] = a0 - y0 * (a1 - a0) / (y1 - y0)
    else:
        result["zero_crossing_age"] = a0
    return result


# ---------------------------------------------------------------------------
# One arm / metric
# ---------------------------------------------------------------------------


def bootstrap_frame(
    df: pd.DataFrame,
    metric: str,
    arm_endpoint: str,
    source: int,
    protocol: str,
    window_index: int,
    n_boot: int,
    seed: int,
    shared_bands: list[str] | None = None,
    draws: np.ndarray | None = None,
    scope: str = "both",
    draw_offset: int = 0,
    return_curves: bool = True,
    curve_only: bool = False,
) -> dict[str, Any]:
    """Bootstrap one arm/metric.

    ``scope`` selects the female-only model, the sex-interaction model, or both.
    ``draws`` must be computed on the same row order as ``df`` after filtering;
    this is what makes paired contrasts use identical participant draws.
    ``curve_only`` skips HC3 covariance, Wald tests, and transition metrics;
    ``return_curves`` independently controls long-form curve materialization.
    """
    if scope not in SCOPE_LABELS:
        raise ValueError(f"unknown scope {scope!r}; expected one of {sorted(SCOPE_LABELS)}")
    if metric not in df.columns:
        raise KeyError(f"metric {metric!r} absent from frame")

    metric_values = pd.to_numeric(df[metric], errors="coerce").to_numpy(float)
    age_values = pd.to_numeric(df["age"], errors="coerce").to_numpy(float)
    sex_values = pd.to_numeric(df["sex"], errors="coerce").to_numpy(float)
    keep = (
        np.isfinite(metric_values)
        & np.isfinite(age_values)
        & np.isin(sex_values, (0.0, 1.0))
    )
    work = df.loc[keep].copy()
    if work.empty:
        return {
            "curves": pd.DataFrame(), "arrays": {}, "point": {}, "point_se": {},
            "test_stats": {}, "metrics": {}, "support": [], "n_boot": 0,
        }
    work["age"] = age_values[keep]
    work["sex"] = sex_values[keep]
    work[metric] = metric_values[keep]
    sort_cols = [c for c in ("user_id", "window_index") if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, kind="mergesort")
    work = work.reset_index(drop=True)
    n = len(work)

    if draws is None:
        draws = draw_indices(work, n_boot, seed)
    draws = np.asarray(draws, dtype=np.int64)
    if draws.ndim != 2 or draws.shape[1] != n:
        raise ValueError(
            f"draws have shape {draws.shape} but the filtered frame has {n} rows; "
            "the shared draw matrix must be built on the same aligned rows"
        )
    n_boot = int(draws.shape[0])

    ages = work["age"].to_numpy(float)
    sex = work["sex"].to_numpy(float)
    y = work[metric].to_numpy(float)

    # --- P0: invariant design matrices, evaluated once per arm ---------------
    X_female, _ = models.build_design(ages)
    X_sex, _ = models.build_design(ages, sex)
    grid_female, grid_sex = _grid_designs()
    n_ages = len(AGE_GRID)

    arrays: dict[str, np.ndarray] = {}
    point: dict[str, np.ndarray] = {}
    point_se: dict[str, np.ndarray] = {}
    supported: dict[str, np.ndarray] = {}
    test_stats: dict[str, list[float]] = {}
    metrics: dict[str, list[float]] = {}

    labels = SCOPE_LABELS[scope]

    # --- Female-only model ---------------------------------------------------
    if "female" in labels:
        female_pos = np.flatnonzero(sex == 1.0)
        female_bands = models.female_support_bands(work)
        if len(female_pos) >= common.MIN_FEMALE_PER_BAND and female_bands:
            Xf = X_female[female_pos, :]
            yf = y[female_pos]
            if _full_rank(Xf):
                fmask = models.support_mask(AGE_GRID, female_bands)
                C = np.zeros((3, 4), dtype=float)
                C[0, 1] = C[1, 2] = C[2, 3] = 1.0
                contrasts = (("female_total_age", C), ("female_nonlinear", C[1:]))
                if not curve_only:
                    theta, cov, _, df_resid = models.ols_hc3(Xf, yf)
                else:
                    theta = models.ols_coefficients(Xf, yf)
                arrays["female"] = np.full((n_boot, n_ages), np.nan, dtype=np.float32)
                point["female"] = np.where(
                    fmask, grid_female @ theta, np.nan
                ).astype(np.float32)
                point_se["female"] = (
                    np.where(fmask, _point_se(grid_female, cov), np.nan).astype(np.float32)
                    if not curve_only else np.full(n_ages, np.nan, dtype=np.float32)
                )
                supported["female"] = fmask
                if not curve_only:
                    for name, Cmat in contrasts:
                        test_stats.setdefault(f"female:{name}", []).append(
                            models.chunk_wald(theta, cov, Cmat, df_resid)["wald_stat"]
                        )
                for b in range(n_boot):
                    pos = draws[b, female_pos]
                    Xb = X_female[pos, :]
                    if not _full_rank(Xb):
                        continue
                    if not curve_only:
                        theta_b, cov_b, _, df_b = models.ols_hc3(Xb, y[pos])
                    else:
                        theta_b = models.ols_coefficients(Xb, y[pos])
                    arrays["female"][b] = np.where(
                        fmask, grid_female @ theta_b, np.nan
                    ).astype(np.float32)
                    if not curve_only:
                        for name, Cmat in contrasts:
                            test_stats.setdefault(f"female:{name}", []).append(
                                models.chunk_wald(theta_b, cov_b, Cmat, df_b)["wald_stat"]
                            )
        # ``transition_metrics`` is a female-minus-male contrast and is not
        # defined for the female-only model.

    # --- Sex-interaction model on shared support -----------------------------
    bands: list[str] = []
    if "sex_female" in labels:
        if shared_bands is not None:
            bands = [b for b in shared_bands if b in BAND_LABELS]
        else:
            _, bands = models.shared_support(work)
        if bands:
            in_bands = np.isin(
                models.band_labels_for_ages(ages), np.asarray(bands, dtype=object)
            )
            support_idx = np.flatnonzero(in_bands)
        else:
            support_idx = np.empty(0, dtype=np.int64)

        if len(support_idx) >= 60 and len(bands) >= 3:
            Xs = X_sex[support_idx, :]
            ys = y[support_idx]
            if _full_rank(Xs):
                smask = models.support_mask(AGE_GRID, bands)
                contrasts = models.chunk_contrasts(Xs.shape[1])
                if not curve_only:
                    theta, cov, _, df_resid = models.ols_hc3(Xs, ys)
                else:
                    theta = models.ols_coefficients(Xs, ys)
                for sex_value, label in SEX_MODEL_SCOPES:
                    arrays[label] = np.full((n_boot, n_ages), np.nan, dtype=np.float32)
                    grid = grid_sex[sex_value]
                    point[label] = np.where(smask, grid @ theta, np.nan).astype(np.float32)
                    point_se[label] = (
                        np.where(smask, _point_se(grid, cov), np.nan).astype(np.float32)
                        if not curve_only else np.full(n_ages, np.nan, dtype=np.float32)
                    )
                    supported[label] = smask
                if not curve_only:
                    for _, (name, Cmat) in contrasts.items():
                        test_stats.setdefault(f"sex:{name}", []).append(
                            models.chunk_wald(theta, cov, Cmat, df_resid)["wald_stat"]
                        )
                    for name, value in transition_metrics(theta, bands).items():
                        metrics.setdefault(name, []).append(value)

                for b in range(n_boot):
                    pos = draws[b, support_idx]
                    Xb = X_sex[pos, :]
                    if not _full_rank(Xb):
                        continue
                    if not curve_only:
                        theta_b, cov_b, _, df_b = models.ols_hc3(Xb, y[pos])
                    else:
                        theta_b = models.ols_coefficients(Xb, y[pos])
                    for sex_value, label in SEX_MODEL_SCOPES:
                        arrays[label][b] = np.where(
                            smask, grid_sex[sex_value] @ theta_b, np.nan
                        ).astype(np.float32)
                    if not curve_only:
                        for _, (name, Cmat) in contrasts.items():
                            test_stats.setdefault(f"sex:{name}", []).append(
                                models.chunk_wald(theta_b, cov_b, Cmat, df_b)["wald_stat"]
                            )
                        for name, value in transition_metrics(theta_b, bands).items():
                            metrics.setdefault(name, []).append(value)

    curves = pd.DataFrame()
    if return_curves and point:
        frames: list[pd.DataFrame] = []
        meta = {
            "source": int(source),
            "endpoint": str(arm_endpoint),
            "metric": str(metric),
            "protocol": str(protocol),
            "window_index": int(window_index),
        }
        draw_ids = np.arange(n_boot, dtype=np.int64) + int(draw_offset)
        for label in labels:
            if label not in point:
                continue
            mask = supported[label]
            frames.append(
                pd.DataFrame(
                    {
                        **meta,
                        "scope": label,
                        "draw": -1,
                        "age": AGE_GRID,
                        "prediction": point[label],
                        "se": point_se[label],
                        "supported": mask,
                    }
                )
            )
            if n_boot:
                frames.append(
                    pd.DataFrame(
                        {
                            **meta,
                            "scope": label,
                            "draw": np.repeat(draw_ids, n_ages),
                            "age": np.tile(AGE_GRID, n_boot),
                            "prediction": arrays[label].reshape(-1),
                            "se": np.full(n_boot * n_ages, np.nan, dtype=np.float32),
                            "supported": np.tile(mask, n_boot),
                        }
                    )
                )
        curves = pd.concat(frames, ignore_index=True)
        for col in ("endpoint", "metric", "protocol", "scope"):
            curves[col] = curves[col].astype("category")
        curves["source"] = curves["source"].astype(np.int32)
        curves["window_index"] = curves["window_index"].astype(np.int16)
        curves["draw"] = curves["draw"].astype(np.int32)
        curves["age"] = curves["age"].astype(np.float32)
        curves["prediction"] = curves["prediction"].astype(np.float32)
        curves["se"] = curves["se"].astype(np.float32)

    return {
        "curves": curves,
        "arrays": arrays,
        "point": point,
        "point_se": point_se,
        "supported": supported,
        "test_stats": test_stats,
        "metrics": metrics,
        "support": list(bands),
        "n_boot": int(n_boot),
    }


# ---------------------------------------------------------------------------
# Confidence bands
# ---------------------------------------------------------------------------


def _bands_from_arrays(
    boot: np.ndarray,
    point: np.ndarray,
    hc3: np.ndarray,
    ages: np.ndarray,
    meta: dict[str, Any],
) -> pd.DataFrame:
    """Pointwise and simultaneous bands from an ``(N_boot, N_ages)`` matrix.

    The simultaneous band studentizes with the empirical across-draw SE and a
    bootstrap critical value, not with the per-draw HC3 SEs.
    """
    boot = np.asarray(boot, dtype=float)
    point = np.asarray(point, dtype=float)
    hc3 = np.asarray(hc3, dtype=float)
    ages = np.asarray(ages, dtype=float)
    finite = np.isfinite(point)
    if not finite.any() or boot.size == 0:
        return pd.DataFrame()

    boot = boot[:, finite]
    point = point[finite]
    hc3 = hc3[finite]
    ages = ages[finite]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
        boot_se = np.nanstd(boot, axis=0, ddof=1)
    boot_se = np.maximum(boot_se, 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        student = np.abs(boot - point[None, :]) / boot_se[None, :]
        maxima = np.nanmax(student, axis=1)
    maxima = maxima[np.isfinite(maxima)]
    crit = float(np.nanpercentile(maxima, 95.0)) if len(maxima) else np.nan

    return pd.DataFrame(
        {
            **meta,
            "age": ages,
            "point": point,
            "lo": lo,
            "hi": hi,
            "boot_se": boot_se,
            "hc3_se": hc3,
            "sim_lo": point - crit * boot_se,
            "sim_hi": point + crit * boot_se,
            "sim_critical_value": crit,
            "n_boot": int(boot.shape[0]),
        }
    )


def summarize_bands(curves: pd.DataFrame) -> pd.DataFrame:
    """Bands from a long curve table with ``draw == -1`` point rows.

    This is the caller-facing path used by paired comparisons and tests.  The
    stage runner uses the array path directly and never materializes the
    bootstrap draws as a long table.
    """
    if curves is None or curves.empty:
        return pd.DataFrame()
    key_cols = [
        c for c in ("source", "endpoint", "metric", "protocol", "window_index")
        if c in curves.columns
    ]
    rows: list[pd.DataFrame] = []
    for key, group in curves.groupby(key_cols + ["scope"], sort=True, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(key_cols, key[:-1]))
        meta["scope"] = key[-1]
        point = group.loc[group["draw"].eq(-1)].sort_values("age")
        boot = group.loc[group["draw"].ge(0)]
        if point.empty or boot.empty:
            continue
        ages = point["age"].to_numpy(float)
        predictions = point["prediction"].to_numpy(float)
        se = (
            point["se"].to_numpy(float)
            if "se" in point.columns
            else np.full(len(ages), np.nan)
        )
        matrix = boot.pivot_table(
            index="draw", columns="age", values="prediction", aggfunc="mean"
        ).reindex(columns=ages)
        frame = _bands_from_arrays(matrix.to_numpy(float), predictions, se, ages, meta)
        if not frame.empty:
            rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def _boot_one(args: tuple) -> tuple[tuple, dict]:
    """Worker: fit one arm/metric.  Long-form curves are not shipped back."""
    (df, metric, arm_endpoint, source, protocol, window_index,
     n_boot, seed, draw_offset) = args
    result = bootstrap_frame(
        df, metric, arm_endpoint, source, protocol, window_index,
        n_boot, seed, draw_offset=draw_offset, return_curves=False,
    )
    result.pop("curves", None)
    # Canonical key order used by every stage-5 artifact:
    # (source, endpoint, protocol, window_index, metric).
    return (
        int(source), str(arm_endpoint), str(protocol), int(window_index), str(metric)
    ), result


def _summarize_distribution(values: list[float]) -> dict[str, float]:
    valid = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if valid.size == 0:
        return {
            "bootstrap_median": np.nan, "bootstrap_q025": np.nan,
            "bootstrap_q975": np.nan, "bootstrap_min": np.nan,
            "bootstrap_max": np.nan, "n_boot_ok": 0,
        }
    return {
        "bootstrap_median": float(np.median(valid)),
        "bootstrap_q025": float(np.percentile(valid, 2.5)),
        "bootstrap_q975": float(np.percentile(valid, 97.5)),
        "bootstrap_min": float(np.min(valid)),
        "bootstrap_max": float(np.max(valid)),
        "n_boot_ok": int(valid.size),
    }


def run_stage4(
    run_dir: Path,
    endpoints_file: Path | None = None,
    n_boot: int = common.N_BOOTSTRAP,
    seed: int = 20261020,
    workers: int = 8,
    endpoint_names: list[str] | None = None,
    window_indices: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    """Run the stratified bootstrap across all estimable arm/metric pairs.

    Cohort filtering mirrors ``models.run_stage3`` so that the bootstrap curves
    and the point-estimate models describe the same participant set.
    """
    run_dir = Path(run_dir)
    boot_dir = run_dir / "bootstrap"
    boot_dir.mkdir(parents=True, exist_ok=True)

    endpoints_file = (
        Path(endpoints_file) if endpoints_file
        else run_dir / "endpoints" / "endpoints.parquet"
    )
    if not endpoints_file.exists():
        summary = {
            "n_boot": int(n_boot), "seed": int(seed), "arms": 0, "jobs": 0,
            "status": "no_endpoints_file",
        }
        (run_dir / "stage5_bootstrap_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary

    df = pd.read_parquet(endpoints_file)
    if "null_converged" in df.columns:
        df = df.loc[df["null_converged"].fillna(False)]
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
    df = df.loc[np.isfinite(df["age"]) & df["sex"].isin([0, 1])].copy()
    endpoint_names = endpoint_names or [e for e in models.ENDPOINTS if e in df.columns]

    window_set = {int(w) for w in window_indices}
    jobs: list[tuple] = []
    arm_keys: list[tuple[int, str, str, int, str]] = []
    for (source, arm_endpoint, protocol, window_index), group in df.groupby(
        ["source", "endpoint", "protocol", "window_index"], sort=True
    ):
        if int(window_index) not in window_set:
            continue
        for metric in endpoint_names:
            if metric not in group.columns:
                continue
            finite_metric = np.isfinite(
                pd.to_numeric(group[metric], errors="coerce")
            )
            subset = group.loc[finite_metric]
            if subset.empty:
                continue
            key = (
                int(source), str(arm_endpoint), str(protocol),
                int(window_index), str(metric),
            )
            arm_keys.append(key)
            # Positional layout required by ``_boot_one``:
            # (df, metric, arm_endpoint, source, protocol, window_index,
            #  n_boot, seed, draw_offset).
            jobs.append(
                (
                    subset, key[4], key[1], key[0], key[2], key[3],
                    int(n_boot), int(seed), 0,
                )
            )

    results: list[tuple[tuple, dict]] = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_boot_one, jobs, chunksize=1))
    else:
        results = [_boot_one(job) for job in jobs]

    # --- Chunk-test distributions -------------------------------------------
    test_rows: list[dict[str, Any]] = []
    for key, result in results:
        for name, values in result.get("test_stats", {}).items():
            if not values:
                continue
            model_scope, test_name = name.split(":", 1)
            row = {
                "source": key[0], "endpoint": key[1], "protocol": key[2],
                "window_index": key[3], "metric": key[4],
                "model_scope": model_scope, "test": test_name,
                "observed_wald": float(values[0]),
                "n_boot": int(len(values) - 1),
            }
            row.update(_summarize_distribution(values[1:]))
            test_rows.append(row)
    test_frame = pd.DataFrame(test_rows)
    test_frame.to_csv(boot_dir / "bootstrap_chunk_summary.csv", index=False)

    # --- Transition-metric distributions ------------------------------------
    metric_rows: list[dict[str, Any]] = []
    for key, result in results:
        for name, values in result.get("metrics", {}).items():
            if not values:
                continue
            row = {
                "source": key[0], "endpoint": key[1], "protocol": key[2],
                "window_index": key[3], "metric": key[4],
                "metric_name": name, "point": float(values[0]),
                "n_boot": int(len(values) - 1),
            }
            finite = np.asarray([v for v in values[1:] if np.isfinite(v)], dtype=float)
            row["lo"] = float(np.percentile(finite, 2.5)) if finite.size else np.nan
            row["hi"] = float(np.percentile(finite, 97.5)) if finite.size else np.nan
            row["n_boot_ok"] = int(finite.size)
            metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(boot_dir / "transition_metrics.csv", index=False)

    # --- Bootstrap curve arrays (.npz) --------------------------------------
    payload: dict[str, np.ndarray] = {"age_grid": AGE_GRID.astype(np.float32)}
    manifest_rows: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []
    band_frames: list[pd.DataFrame] = []
    for key, result in results:
        meta = {
            "source": key[0], "endpoint": key[1], "protocol": key[2],
            "window_index": key[3], "metric": key[4],
        }
        for label, array in result.get("arrays", {}).items():
            array_key = f"{key[0]}|{key[1]}|{key[2]}|{key[3]}|{key[4]}|{label}"
            payload[array_key] = np.ascontiguousarray(array, dtype=np.float32)
            manifest_rows.append(
                {
                    **meta, "scope": label, "array_key": array_key,
                    "n_boot": int(array.shape[0]), "n_ages": int(array.shape[1]),
                }
            )
            point_frames.append(
                pd.DataFrame(
                    {
                        **meta, "scope": label, "age": AGE_GRID,
                        "prediction": result["point"][label],
                        "se": result["point_se"][label],
                    }
                )
            )
            frame = _bands_from_arrays(
                array, result["point"][label], result["point_se"][label],
                AGE_GRID, {**meta, "scope": label},
            )
            if not frame.empty:
                band_frames.append(frame)

    np.savez_compressed(boot_dir / "bootstrap_curves.npz", **payload)
    pd.DataFrame(manifest_rows).to_csv(
        boot_dir / "bootstrap_curves_manifest.csv", index=False
    )
    if point_frames:
        pd.concat(point_frames, ignore_index=True).to_parquet(
            boot_dir / "curves_point.parquet", index=False
        )
    else:
        pd.DataFrame().to_parquet(boot_dir / "curves_point.parquet", index=False)
    if band_frames:
        pd.concat(band_frames, ignore_index=True).to_parquet(
            boot_dir / "confidence_bands.parquet", index=False
        )
    else:
        pd.DataFrame().to_parquet(boot_dir / "confidence_bands.parquet", index=False)

    summary = {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "arms": len({key[:4] for key in arm_keys}),
        "arm_metrics": len(arm_keys),
        "jobs": int(len(jobs)),
        "curve_arrays": int(len(manifest_rows)),
        "status": "ok" if jobs else "no_estimable_arms",
    }
    (run_dir / "stage5_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
