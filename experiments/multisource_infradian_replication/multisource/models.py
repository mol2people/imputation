"""Stage 4: source- and endpoint-specific RCS age models.

Preserves the historical RCS basis, four chunk tests, and HC3 covariance.
Each source x endpoint x protocol x window is fitted separately.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg
from scipy import stats

from . import common

ENDPOINTS = ["Q_i", "E_i2", "A_hat", "sigma_resid"]
BAND_EDGES = np.arange(0, 125, common.AGE_BAND_WIDTH)
BAND_LABELS = [
    f"[{BAND_EDGES[i]},{BAND_EDGES[i + 1]})" for i in range(len(BAND_EDGES) - 1)
]
AGE_GRID = np.arange(common.AGE_GRID_MIN, common.AGE_GRID_MAX + 0.25, 0.25)


def build_design(ages: np.ndarray, sex: np.ndarray | None = None,
                 knots=common.DEFAULT_KNOTS) -> tuple[np.ndarray, list[str]]:
    rcs1, rcs2 = common.rcs_stone_koo(ages, knots)
    cols = [np.ones_like(ages), ages, rcs1, rcs2]
    names = ["const", "age", "rcs1", "rcs2"]
    if sex is not None:
        s = sex.astype(np.float64)
        cols.extend([s, s * ages, s * rcs1, s * rcs2])
        names.extend(["sex", "sex_age", "sex_rcs1", "sex_rcs2"])
    return np.column_stack(cols), names


def chunk_contrasts(p: int) -> dict[int, tuple[str, np.ndarray]]:
    def mat(rows):
        m = np.zeros((len(rows), p), dtype=np.float64)
        for i, (a, b) in enumerate(rows):
            m[i, a] = 1.0
            if b is not None:
                m[i, b] = 1.0
        return m

    return {
        1: ("total_female_age", mat([(1, 5), (2, 6), (3, 7)])),
        2: ("nonlinear_female_curvature", mat([(2, 6), (3, 7)])),
        3: ("total_sex_x_age_interaction", mat([(5, None), (6, None), (7, None)])),
        4: ("nonlinear_sex_x_age_interaction", mat([(6, None), (7, None)])),
    }


def ols_hc3(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    xtx = X.T @ X
    try:
        # X^T X is positive definite at every call site that passes the
        # _full_rank gate.  Cholesky avoids the full SVD performed by pinv.
        chol = scipy.linalg.cholesky(xtx, lower=True, check_finite=False)
        xtx_inv = scipy.linalg.cho_solve(
            (chol, True), np.eye(xtx.shape[0]), check_finite=False
        )
    except scipy.linalg.LinAlgError:
        # Keep the historical pseudo-inverse as a defensive fallback for
        # direct callers that bypass the rank gate.
        xtx_inv = np.linalg.pinv(xtx)
    theta = xtx_inv @ (X.T @ y)
    resid = y - X @ theta
    n, p = X.shape
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    h = np.clip(h, 0.0, 1.0 - 1e-12)
    weights = (resid ** 2) / (1.0 - h) ** 2
    cov = xtx_inv @ (X.T @ (X * weights[:, None])) @ xtx_inv
    return theta, cov, resid, int(n - p)


def ols_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return OLS coefficients using the same Cholesky solve as ``ols_hc3``."""
    xtx = X.T @ X
    try:
        chol = scipy.linalg.cholesky(xtx, lower=True, check_finite=False)
        return scipy.linalg.cho_solve(
            (chol, True), X.T @ y, check_finite=False
        )
    except scipy.linalg.LinAlgError:
        return np.linalg.pinv(xtx) @ (X.T @ y)


def chunk_wald(theta: np.ndarray, cov: np.ndarray, C: np.ndarray, df_resid: int) -> dict[str, float]:
    q = C.shape[0]
    ctheta = C @ theta
    middle = C @ cov @ C.T
    try:
        # The contrast covariance is small (2–3 rows) and positive definite
        # for all rank-gated fits.  A Cholesky solve avoids a per-draw SVD.
        solution = scipy.linalg.solve(
            middle, ctheta, assume_a="pos", check_finite=False
        )
        wald = float(ctheta @ solution)
    except (scipy.linalg.LinAlgError, ValueError):
        wald = float(ctheta.T @ np.linalg.pinv(middle) @ ctheta)
    wald = max(0.0, wald)
    return {
        "df_test": int(q),
        "wald_stat": wald,
        "f_stat": wald / q,
        "p_chi2": float(stats.chi2.sf(wald, df=q)),
        "p_f": float(stats.f.sf(wald / q, dfn=q, dfd=df_resid)),
    }


def band_labels_for_ages(ages: np.ndarray) -> np.ndarray:
    """Five-year band label for each age; out-of-range ages map to ``"nan"``."""
    return (
        pd.cut(np.asarray(ages, dtype=float), bins=BAND_EDGES, right=False, labels=BAND_LABELS)
        .astype(str)
        .to_numpy()
    )


def support_mask(ages: np.ndarray, bands: list[str]) -> np.ndarray:
    """Boolean mask selecting evaluation ages inside supported bands."""
    return np.isin(band_labels_for_ages(ages), np.asarray(bands, dtype=object))


def female_support_bands(df: pd.DataFrame) -> list[str]:
    """Bands with at least the prespecified number of women."""
    female = df.loc[df["sex"].eq(1) & np.isfinite(df["age"])]
    labels = band_labels_for_ages(female["age"].to_numpy(float))
    counts = pd.Series(labels).value_counts()
    return [
        band for band in BAND_LABELS
        if int(counts.get(band, 0)) >= common.MIN_FEMALE_PER_BAND
    ]


def shared_support(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return rows in shared-support bands and the retained band labels."""
    work = df.copy()
    work["age_band"] = pd.cut(work["age"], bins=BAND_EDGES, right=False, labels=BAND_LABELS)
    ct = pd.crosstab(work["age_band"], work["sex"])
    bands = []
    for band in BAND_LABELS:
        n_f = int(ct.loc[band, 1]) if band in ct.index and 1 in ct.columns else 0
        n_m = int(ct.loc[band, 0]) if band in ct.index and 0 in ct.columns else 0
        if n_f >= common.MIN_FEMALE_PER_BAND and n_m >= common.MIN_MALE_PER_BAND:
            bands.append(band)
    if len(bands) < common.MIN_SHARED_BANDS:
        return work.iloc[0:0], []
    return work.loc[work["age_band"].isin(bands)].copy(), bands


def fit_female(df: pd.DataFrame, metric: str, arm_endpoint: str, source: int,
               protocol: str, window_index: int) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    female = df.loc[df["sex"].eq(1) & np.isfinite(df[metric]) & df["age"].notna()].copy()
    if len(female) < 30:
        return None, pd.DataFrame()
    X, names = build_design(female["age"].to_numpy(float))
    y = female[metric].to_numpy(float)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None, pd.DataFrame()
    theta, cov, resid, df_resid = ols_hc3(X, y)
    bands = female_support_bands(female)
    if not bands:
        return None, pd.DataFrame()
    mask = support_mask(AGE_GRID, bands)
    grid_X, _ = build_design(AGE_GRID)
    pred = grid_X @ theta
    se = np.sqrt(np.einsum("ij,jk,ik->i", grid_X, cov, grid_X))
    pred = np.where(mask, pred, np.nan)
    se = np.where(mask, se, np.nan)
    curves = pd.DataFrame(
        {
            "source": int(source),
            "endpoint": arm_endpoint,
            "metric": metric,
            "protocol": protocol,
            "window_index": int(window_index),
            "scope": "female",
            "age": AGE_GRID,
            "prediction": pred,
            "se": se,
            "supported": mask,
        }
    )
    # Female age tests: total (3 df) and nonlinear (2 df).
    p = X.shape[1]
    C = np.zeros((3, p))
    C[0, 1] = 1.0
    C[1, 2] = 1.0
    C[2, 3] = 1.0
    tests = {f"female_{k}": chunk_wald(theta, cov, C, df_resid) for k in ("total_age",)}
    Cnl = C[1:]
    tests["female_nonlinear"] = chunk_wald(theta, cov, Cnl, df_resid)
    record = {
        "source": int(source),
        "source_name": common.source_name(int(source)),
        "endpoint": arm_endpoint,
        "metric": metric,
        "protocol": protocol,
        "window_index": int(window_index),
        "model": "female_rcs",
        "n_obs": int(len(female)),
        "n_female": int(len(female)),
        "n_male": 0,
        "age_min": float(female["age"].min()),
        "age_max": float(female["age"].max()),
        "female_support_bands": bands,
        "coefficients": theta.tolist(),
        "covariance": cov.tolist(),
        "df_resid": df_resid,
        "chunk_tests": tests,
    }
    return record, curves


def fit_sex_interaction(df: pd.DataFrame, metric: str, arm_endpoint: str, source: int,
                        protocol: str, window_index: int) -> tuple[dict[str, Any] | None, pd.DataFrame, list[str]]:
    support, bands = shared_support(df)
    if support.empty:
        return None, pd.DataFrame(), bands
    support = support.loc[np.isfinite(support[metric])].copy()
    if len(support) < 60:
        return None, pd.DataFrame(), bands
    X, names = build_design(support["age"].to_numpy(float), support["sex"].to_numpy(float))
    y = support[metric].to_numpy(float)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None, pd.DataFrame(), bands
    theta, cov, resid, df_resid = ols_hc3(X, y)
    p = X.shape[1]
    contrasts = chunk_contrasts(p)
    tests = {
        name: chunk_wald(theta, cov, C, df_resid) for _, (name, C) in contrasts.items()
    }
    curves = []
    mask = support_mask(AGE_GRID, bands)
    for sex_value, scope in ((1, "female"), (0, "male")):
        grid_X, _ = build_design(AGE_GRID, np.full_like(AGE_GRID, sex_value, dtype=float))
        pred = grid_X @ theta
        se = np.sqrt(np.einsum("ij,jk,ik->i", grid_X, cov, grid_X))
        pred = np.where(mask, pred, np.nan)
        se = np.where(mask, se, np.nan)
        curves.append(
            pd.DataFrame(
                {
                    "source": int(source),
                    "endpoint": arm_endpoint,
                    "metric": metric,
                    "protocol": protocol,
                    "window_index": int(window_index),
                    "scope": scope,
                    "age": AGE_GRID,
                    "prediction": pred,
                    "se": se,
                    "supported": mask,
                }
            )
        )
    record = {
        "source": int(source),
        "source_name": common.source_name(int(source)),
        "endpoint": arm_endpoint,
        "metric": metric,
        "protocol": protocol,
        "window_index": int(window_index),
        "model": "sex_interaction_rcs",
        "n_obs": int(len(support)),
        "n_female": int(support["sex"].eq(1).sum()),
        "n_male": int(support["sex"].eq(0).sum()),
        "age_min": float(support["age"].min()),
        "age_max": float(support["age"].max()),
        "shared_support_bands": bands,
        "coefficients": theta.tolist(),
        "covariance": cov.tolist(),
        "df_resid": df_resid,
        "chunk_tests": tests,
    }
    return record, pd.concat(curves, ignore_index=True), bands


def _fit_arm(args: tuple[pd.DataFrame, int, str, str, int, str]
            ) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    group, source, endpoint, protocol, window_index, metric = args
    fits: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    support_rows: list[dict[str, Any]] = []
    female_record, female_curves = fit_female(
        group, metric, endpoint, source, protocol, int(window_index)
    )
    if female_record is not None:
        fits.append(female_record)
        curves.append(female_curves)
    sex_record, sex_curves, bands = fit_sex_interaction(
        group, metric, endpoint, source, protocol, int(window_index)
    )
    if sex_record is not None:
        fits.append(sex_record)
        curves.append(sex_curves)
    work = group.copy()
    work["age_band"] = pd.cut(work["age"], bins=BAND_EDGES, right=False, labels=BAND_LABELS)
    for band in BAND_LABELS:
        band_rows = work.loc[work["age_band"].eq(band)]
        support_rows.append(
            {
                "source": source,
                "endpoint": endpoint,
                "metric": metric,
                "protocol": protocol,
                "window_index": int(window_index),
                "age_band": band,
                "female_count": int(band_rows["sex"].eq(1).sum()),
                "male_count": int(band_rows["sex"].eq(0).sum()),
                "is_shared_support": band in bands,
            }
        )
    return fits, curves, support_rows


def run_stage3(
    run_dir: Path,
    endpoints_file: Path | None = None,
    endpoint_names: list[str] | None = None,
    min_female_band: int = common.MIN_FEMALE_PER_BAND,
    min_male_band: int = common.MIN_MALE_PER_BAND,
    workers: int = 8,
    window_indices: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    """Fit all estimable source x endpoint x protocol x window models."""
    run_dir = Path(run_dir)
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    endpoints_file = Path(endpoints_file) if endpoints_file else run_dir / "endpoints" / "endpoints.parquet"
    if not endpoints_file.exists():
        summary = {
            "endpoints_file": str(endpoints_file),
            "models_fitted": 0,
            "estimable_arms": [],
            "status": "no_endpoints_file",
        }
        (run_dir / "stage4_models_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    df = pd.read_parquet(endpoints_file)
    endpoint_names = endpoint_names or [e for e in ENDPOINTS if e in df.columns]

    jobs = []
    for keys, group in df.groupby(
        ["source", "endpoint", "protocol", "window_index"], sort=True
    ):
        source, endpoint, protocol, window_index = keys
        source = int(source)
        endpoint = str(endpoint)
        if int(window_index) not in set(int(w) for w in window_indices):
            continue
        group = group.loc[group["null_converged"].fillna(False)].copy()
        group = group.loc[np.isfinite(group["age"]) & group["sex"].isin([0, 1])]
        if group.empty:
            continue
        for metric in endpoint_names:
            if metric not in group.columns:
                continue
            subset = group.loc[np.isfinite(group[metric])].copy()
            if not subset.empty:
                jobs.append((subset, source, endpoint, str(protocol), int(window_index), metric))

    results = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_fit_arm, jobs, chunksize=1))
    else:
        results = [_fit_arm(job) for job in jobs]

    fits = [record for r in results for record in r[0]]
    curves = [frame for r in results for frame in r[1]]
    support_rows = [row for r in results for row in r[2]]

    fits_df = pd.DataFrame(fits)
    if not fits_df.empty:
        fits_df.to_parquet(models_dir / "model_fits.parquet", index=False)
        # Flatten chunk tests.
        test_rows = []
        for r in fits:
            for name, result in r["chunk_tests"].items():
                test_rows.append(
                    {
                        "source": r["source"],
                        "endpoint": r["endpoint"],
                        "metric": r["metric"],
                        "protocol": r["protocol"],
                        "window_index": r["window_index"],
                        "model": r["model"],
                        "test": name,
                        **result,
                    }
                )
        pd.DataFrame(test_rows).to_csv(models_dir / "chunk_tests.csv", index=False)
        pd.DataFrame(
            [
                {k: r[k] for k in (
                    "source", "source_name", "endpoint", "metric", "protocol", "window_index",
                    "model", "n_obs", "n_female", "n_male", "age_min", "age_max", "df_resid"
                )}
                for r in fits
            ]
        ).to_csv(models_dir / "model_summary.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_parquet(models_dir / "curves.parquet", index=False)
    if support_rows:
        pd.DataFrame(support_rows).to_csv(models_dir / "age_support.csv", index=False)

    summary = {
        "endpoints_file": str(endpoints_file),
        "models_fitted": int(len(fits)),
        "estimable_arms": sorted(
            {f"{r['source']}:{r['endpoint']}:{r['metric']}:{r['protocol']}" for r in fits}
        ),
    }
    (run_dir / "stage4_models_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
