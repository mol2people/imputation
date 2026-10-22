"""Stage 3: daily-arm endpoint, nuisance, and two-component null engine.

Analysis B/C use a daily nuisance design without clock-coverage fractions,
because a supplier daily value has no within-day bin mask.  The spectral
statistic and the two-component REML null are the historical definitions.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import common, reml

_ENGINE = common.load_endpoint_engine()
BROAD_FREQS = _ENGINE.BROAD_FREQS
NARROW_PERIODS = _ENGINE.NARROW_PERIODS
NARROW_FREQS = _ENGINE.NARROW_FREQS
DELTA_F = _ENGINE.DELTA_F
IN_BAND_MASK = _ENGINE.IN_BAND_MASK
F_BAND = _ENGINE.F_BAND
# Multisource default is the analytical-gradient L-BFGS-B path.  Set
# MULTISOURCE_REML_OPTIMIZER=nelder to reproduce the historical engine.
_ACTIVE_REML_FITTER = reml.active_fitter()
fit_two_component_reml = _ACTIVE_REML_FITTER or _ENGINE.fit_two_component_reml
fit_two_component_reml_lbfgsb = reml.fit_two_component_reml_lbfgsb
profile_ar1_mle = _ENGINE.profile_ar1_mle
simulate_two_component_ar1 = _ENGINE.simulate_two_component_ar1


def precompute_fwl_harmonic_operators(
    dates: np.ndarray, P_0: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized FWL harmonic projection operators.

    Algebraically identical to the historical per-frequency loop, but the
    concatenated harmonic basis is projected with a single GEMM and the 2x2
    Gram inverses are batched across frequencies (audit §3.2.A).

    Returns ``M`` of shape ``(n, 2K)`` and ``CTC`` of shape ``(K, 2, 2)``.
    """
    dates = np.asarray(dates, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    n = len(dates)
    k = len(freqs)
    t_elapsed = dates - float(dates[0])
    omega = 2.0 * np.pi * freqs
    phase = np.outer(t_elapsed, omega)
    basis = np.empty((n, 2 * k), dtype=np.float64)
    basis[:, 0::2] = np.cos(phase)
    basis[:, 1::2] = np.sin(phase)
    projected = P_0 @ basis
    c0 = projected[:, 0::2]
    c1 = projected[:, 1::2]
    a = np.einsum("ij,ij->j", c0, c0)
    b = np.einsum("ij,ij->j", c0, c1)
    d = np.einsum("ij,ij->j", c1, c1)
    det = a * d - b * b
    bad = det < 1e-12
    if np.any(bad):
        a = a.copy()
        d = d.copy()
        a[bad] += 1e-8
        d[bad] += 1e-8
        det = a * d - b * b
    M = np.empty_like(projected)
    M[:, 0::2] = (c0 * d[None, :] - c1 * b[None, :]) / det[None, :]
    M[:, 1::2] = (c1 * a[None, :] - c0 * b[None, :]) / det[None, :]
    CTC = np.empty((k, 2, 2), dtype=np.float64)
    CTC[:, 0, 0] = a
    CTC[:, 0, 1] = b
    CTC[:, 1, 0] = b
    CTC[:, 1, 1] = d
    return M, CTC


def build_daily_nuisance(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Intercept, linear elapsed time, and six weekday indicators.

    No clock-coverage columns are fabricated for daily summaries.
    """
    n = len(dates)
    cols = [np.ones(n, dtype=np.float64), (dates - dates[0]).astype(np.float64)]
    dow = (dates + 3) % 7  # 0=Mon
    for d in range(1, 7):
        cols.append((dow == d).astype(np.float64))
    X = np.column_stack(cols)

    # Retained from the historical implementation: a column-by-column
    # Gram-Schmidt sweep.  A pivoted-QR replacement was measured (0.058 ->
    # 0.048 ms) but perturbs the downstream REML Nelder-Mead path: Q_i/E_i2
    # shift by ~1e-8 and the discrete phi2 label can flip when v2 ~ 0.  The
    # plan requires numerical continuity with the historical baseline, so the
    # original basis is kept.  See EFFICIENCY_AUDIT.md §6.
    Q = np.zeros((n, 0), dtype=np.float64)
    for j in range(X.shape[1]):
        v = X[:, j]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        if Q.shape[1] > 0:
            v_perp = v - Q @ (Q.T @ v)
            perp_norm = np.linalg.norm(v_perp)
            if perp_norm / v_norm > 1e-4:
                Q = np.column_stack([Q, v_perp / perp_norm])
        else:
            Q = (v / v_norm)[:, np.newaxis]
    P0 = np.eye(n, dtype=np.float64) - Q @ Q.T
    return P0, Q, int(Q.shape[1])


def _null_seed(source: int, user_id: int, start_day: int, digest: str, arm: str, shared: bool) -> int:
    tag = (
        f"{common.SEED_TAG}|{int(source)}|{int(user_id)}|{int(start_day)}|"
        f"null_two_component|{digest}"
    )
    if not shared:
        tag = f"{tag}|{arm}"
    return int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:8], byteorder="little")


def process_daily_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute daily endpoints, nulls, and spectra for one participant task."""
    user_id = int(task["user_id"])
    source = int(task["source"])
    endpoint = str(task["endpoint"])
    arm = str(task.get("arm", endpoint))
    demo = task["demographics"]
    b_rep = int(task.get("null_replicates", common.B_NULL_REPLICATES))
    shared_null = bool(task.get("shared_null_seed", False))
    out: list[dict[str, Any]] = []

    for win in task["windows"]:
        dates = np.asarray(win["dates"], dtype=np.int64)
        protocol = str(win["protocol"])
        widx = int(win["window_index"])
        value_map = win.get("daily_values") or task.get("daily_values") or {}
        values = np.asarray([value_map.get(int(d), np.nan) for d in dates], dtype=np.float64)

        record: dict[str, Any] = {
            "status": "ok",
            "user_id": user_id,
            "source": source,
            "endpoint": endpoint,
            "arm": arm,
            "protocol": protocol,
            "window_index": widx,
            "start_day": int(win["start_day"]),
            "selected_dates_hash": str(win["selected_dates_hash"]),
            "salutation": demo.get("salutation"),
            "birth_year": demo.get("birth_year"),
            "age_dec1": demo.get("age_dec1"),
            "age_jan1": demo.get("age_jan1"),
            "age_dec31": demo.get("age_dec31"),
            "age": demo.get("age_dec1"),
            "sex": 1 if demo.get("salutation") == 10 else (0 if demo.get("salutation") == 20 else np.nan),
            "weight": demo.get("weight", np.nan),
            "height": demo.get("height", np.nan),
            "bmi": demo.get("bmi", np.nan),
            "n_selected_dates": int(len(dates)),
        }
        if len(dates) != 90:
            record["status"] = "bad_selected_date_count"
            record["null_status"] = "failed"
            out.append(record)
            continue
        if not np.isfinite(values).all():
            record["status"] = "nonfinite_daily_value"
            record["null_status"] = "failed"
            out.append(record)
            continue

        P0, Q_basis, p_retained = build_daily_nuisance(dates)
        e0 = P0 @ values
        rss0 = float(e0 @ e0)
        r_rank = int(len(dates) - p_retained)
        record["rss_0"] = rss0
        record["p_retained"] = p_retained
        if rss0 < 1e-10 or not np.isfinite(rss0) or r_rank <= 0:
            record["status"] = "rss0_degenerate"
            record["null_status"] = "failed"
            record["sigma_resid"] = np.nan
            out.append(record)
            continue

        record["sigma_resid"] = float(np.sqrt(rss0 / r_rank))
        # Retained from the historical implementation.  Replacing this with
        # null_space(Q.T) was measured at 0.238 -> 0.049 ms, but it rotates the
        # residual basis enough to perturb the REML optimizer path (Q_i ~1e-8;
        # phi2 label flips when v2 ~ 0).  The original eigh basis reproduces
        # historical endpoints to machine precision.  See EFFICIENCY_AUDIT §6.
        eigvals, U = np.linalg.eigh(P0)
        Q_sub = U[:, eigvals > 0.5]
        z = Q_sub.T @ e0
        phi_init = profile_ar1_mle(e0, dates)
        fitter = task.get("reml_fitter") or fit_two_component_reml
        null_status, diag = fitter(z, Q_sub, dates, rss0, r_rank, phi_init)
        record["null_status"] = null_status
        record["null_diagnostics"] = diag

        if null_status != "converged":
            record["status"] = "reml_failed"
            for key in ("v1", "phi1", "v2", "phi2", "Q_i", "E_i2", "A_hat",
                        "Q_i_mc_se", "peak_period", "S_obs", "mean_S_null",
                        "p_value_discrete"):
                record[key] = np.nan
            out.append(record)
            continue

        v1, phi1, v2, phi2 = diag["v1"], diag["phi1"], diag["v2"], diag["phi2"]
        record.update({"v1": v1, "phi1": phi1, "v2": v2, "phi2": phi2})

        M_broad, CTC_broad = precompute_fwl_harmonic_operators(dates, P0, BROAD_FREQS)
        M_narrow, CTC_narrow = precompute_fwl_harmonic_operators(dates, P0, NARROW_FREQS)

        seed = _null_seed(source, user_id, int(win["start_day"]),
                          str(win["selected_dates_hash"]), arm, shared_null)
        rng = np.random.Generator(np.random.PCG64(seed))
        sim_noise = simulate_two_component_ar1(dates, v1, phi1, v2, phi2, b_rep, rng)
        sim_e = sim_noise @ P0
        sim_rss0 = np.maximum(np.sum(sim_e ** 2, axis=1), 1e-12)

        def spectra(e: np.ndarray, M: np.ndarray, CTC: np.ndarray, denom) -> np.ndarray:
            # P1: vectorized FWL spectra.  ``CTC`` is (k, 2, 2); the harmonic
            # coefficients are the even/odd columns of ``e @ M``.
            beta = e @ M
            bc = beta[:, 0::2]
            bs = beta[:, 1::2]
            drss = (
                (bc ** 2) * CTC[None, :, 0, 0]
                + 2.0 * (bc * bs) * CTC[None, :, 0, 1]
                + (bs ** 2) * CTC[None, :, 1, 1]
            )
            return np.clip(drss / denom, 0.0, 1.0)

        P_broad_null = spectra(sim_e, M_broad, CTC_broad, sim_rss0[:, None])
        P_narrow_null = spectra(sim_e, M_narrow, CTC_narrow, sim_rss0[:, None])
        P_broad_obs = spectra(e0[None, :], M_broad, CTC_broad, rss0)[0]
        P_narrow_obs = spectra(e0[None, :], M_narrow, CTC_narrow, rss0)[0]

        P_null_mean = np.mean(P_broad_null, axis=0)
        P_null_var = np.var(P_broad_null, axis=0, ddof=1)
        D_obs = P_broad_obs - P_null_mean
        record["Q_i"] = float(np.trapezoid(D_obs[IN_BAND_MASK], F_BAND) / DELTA_F)
        # P1: batched quadrature replaces the Python loop over replicates.
        Q_null = np.trapezoid(
            P_broad_null[:, IN_BAND_MASK], F_BAND, axis=-1
        ) / DELTA_F
        record["Q_i_mc_se"] = float(np.std(Q_null, ddof=1) / np.sqrt(b_rep))

        S_obs = float(np.max(P_narrow_obs))
        S_null = np.max(P_narrow_null, axis=1)
        record["S_obs"] = S_obs
        record["mean_S_null"] = float(np.mean(S_null))
        record["E_i2"] = float(S_obs - np.mean(S_null))
        record["p_value_discrete"] = float((1.0 + np.sum(S_null >= S_obs)) / (b_rep + 1.0))

        peak = int(np.argmax(P_narrow_obs))
        record["peak_period"] = float(NARROW_PERIODS[peak])
        beta_peak = e0 @ M_narrow[:, 2 * peak: 2 * peak + 2]
        record["A_hat"] = float(np.sqrt(beta_peak[0] ** 2 + beta_peak[1] ** 2))

        record["spectra"] = {
            "P_obs": P_broad_obs.astype(np.float32),
            "P_null_mean": P_null_mean.astype(np.float32),
            "P_null_var": P_null_var.astype(np.float32),
            "D_excess": D_obs.astype(np.float32),
        }
        out.append(record)
    return out


def build_daily_tasks(
    selected: pd.DataFrame,
    windows: pd.DataFrame,
    demographics: pd.DataFrame,
    source: int,
    endpoint: str,
    protocol: str,
    limit: int | None = None,
    window_indices: tuple[int, ...] = (1,),
) -> list[dict[str, Any]]:
    """Build endpoint tasks from the daily manifests."""
    sel = selected.loc[
        selected["source"].eq(int(source))
        & selected["endpoint"].eq(endpoint)
        & selected["protocol"].eq(protocol)
    ].copy()
    if sel.empty:
        return []
    demos = demographics.set_index("user_id").to_dict("index")
    starts = windows.loc[
        windows["is_eligible"]
        & windows["source"].eq(int(source))
        & windows["endpoint"].eq(endpoint)
        & windows["protocol"].eq(protocol)
    ].set_index(["user_id", "window_index"])["start_day"].to_dict()
    tasks: list[dict[str, Any]] = []
    for uid, group in sel.groupby("user_id", sort=True):
        uid = int(uid)
        row = demos.get(uid)
        if row is None:
            continue
        birth_year = int(row["birth_year"]) if pd.notna(row["birth_year"]) else None
        for (proto, widx, digest), win in group.groupby(
            ["protocol", "window_index", "selected_dates_hash"], sort=True
        ):
            widx = int(widx)
            if widx not in window_indices or (uid, widx) not in starts:
                continue
            start_day = int(starts[(uid, widx)])
            if birth_year is not None:
                age_dec1, age_jan1, age_dec31 = common.compute_all_ages(start_day, birth_year)
            else:
                age_dec1 = age_jan1 = age_dec31 = None
            demo = {
                "salutation": int(row["salutation"]) if pd.notna(row["salutation"]) else None,
                "birth_year": birth_year,
                "age_dec1": int(age_dec1) if age_dec1 is not None else None,
                "age_jan1": int(age_jan1) if age_jan1 is not None else None,
                "age_dec31": int(age_dec31) if age_dec31 is not None else None,
                "weight": float(row["weight"]) if pd.notna(row.get("weight")) else np.nan,
                "height": float(row["height"]) if pd.notna(row.get("height")) else np.nan,
                "bmi": float(row["bmi"]) if pd.notna(row.get("bmi")) else np.nan,
            }
            dates = np.sort(win["date"].to_numpy(dtype=np.int64))
            value_map = dict(zip(dates.tolist(), win["value"].astype(float).tolist()))
            tasks.append({
                "user_id": uid, "source": int(source), "endpoint": endpoint,
                "arm": endpoint, "demographics": demo,
                "windows": [{
                    "protocol": str(proto), "window_index": widx,
                    "start_day": start_day,
                    "selected_dates_hash": str(digest), "dates": dates.tolist(),
                    "daily_values": value_map,
                }],
            })
    return tasks[: int(limit)] if limit is not None else tasks


def write_daily_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """Persist one daily arm's endpoints, diagnostics, and spectra."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra = [r.pop("spectra") for r in rows if "spectra" in r]
    diagnostics = [r.pop("null_diagnostics") for r in rows if "null_diagnostics" in r]
    endpoints = pd.DataFrame(rows)
    if not endpoints.empty:
        endpoints["null_converged"] = endpoints["null_status"].eq("converged")
        endpoints["null_failure_reason"] = np.where(
            endpoints["null_converged"], "", endpoints["status"]
        )
    endpoints.to_parquet(output_dir / "endpoints.parquet", index=False)

    if spectra:
        np.savez_compressed(
            output_dir / "spectra_broad.npz",
            broad_freqs=BROAD_FREQS,
            P_obs=np.stack([s["P_obs"] for s in spectra]),
            P_null_mean=np.stack([s["P_null_mean"] for s in spectra]),
            P_null_var=np.stack([s["P_null_var"] for s in spectra]),
            D_excess=np.stack([s["D_excess"] for s in spectra]),
        )
    if diagnostics:
        pd.DataFrame(diagnostics).to_parquet(output_dir / "null_diagnostics.parquet", index=False)
    return endpoints


def run_daily_arm(
    tasks: list[dict[str, Any]],
    output_dir: Path,
    null_replicates: int = common.B_NULL_REPLICATES,
    workers: int = 8,
) -> pd.DataFrame:
    """Run a daily arm and persist endpoints, diagnostics, and spectra."""
    for task in tasks:
        task["null_replicates"] = int(null_replicates)
    rows: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        chunk = max(1, len(tasks) // (workers * 4) + 1)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for result in ex.map(process_daily_task, tasks, chunksize=chunk):
                rows.extend(result)
    else:
        for task in tasks:
            rows.extend(process_daily_task(task))
    return write_daily_outputs(rows, output_dir)
