"""Analytical-gradient L-BFGS-B fitter for the two-component REML null.

This is a drop-in alternative to the historical six-start Nelder-Mead fitter
defined in ``infradian_age_dependency/scripts/02_compute_endpoints_and_nulls.py``.
It optimizes the same objective on the residual subspace and returns the same
diagnostic keys, so existing endpoint writers do not need a schema change.

The historical engine remains the default for every other experiment.  The
multisource pipeline opts in through :func:`active_fitter`; set
``MULTISOURCE_REML_OPTIMIZER=nelder`` to reproduce the historical path.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.special import expit

PHI2_CANDIDATES = (0.98, 0.995)
FAST_SHARES = (0.3, 0.5, 0.7)
# Extra initial autocorrelations for the boundary-rescue starts.  The
# historical Nelder-Mead simplex can migrate from a high phi1 start into the
# low-phi1 basin; L-BFGS-B cannot, so these starts preserve that basin.
PHI1_FALLBACK_STARTS = (0.3,)
# Low-phi1 anchors are evaluated for every screened fit.  The historical
# Nelder-Mead simplex can reach the near-zero-phi1/high-slow-variance basin,
# while L-BFGS-B started from the profile-AR(1) phi1 can remain in a different
# local optimum.  Two anchors (one per phi2 candidate) recover that basin.
LOW_PHI1_STARTS = (0.01,)

_LOGIT_BOUND = 30.0
_NLL_FAIL = 1e12
_GRAD_SCREEN_TOL = 1e-5
_VAR_FLOOR = 1e-4
_BOUNDARY_FRAC = 1e-3
_AMBIGUITY_NLL_TOL = 1e-4


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _objective_and_gradient(
    theta: np.ndarray,
    z: np.ndarray,
    Q_sub: np.ndarray,
    dt_abs: np.ndarray,
    powers_idx: np.ndarray,
    S2: np.ndarray,
    r_rank: int,
) -> tuple[float, np.ndarray]:
    """Negative restricted log-likelihood and its exact gradient.

    ``S(theta) = s1 * Q_sub.T C1(phi1) Q_sub + s2 * S2`` with
    ``s1 = exp(theta[0])``, ``phi1 = 0.95 * expit(theta[1])``, and
    ``s2 = exp(theta[2])``.
    """
    s1 = float(np.exp(theta[0]))
    p1 = float(0.95 * expit(theta[1]))
    s2 = float(np.exp(theta[2]))

    p1_powers = p1 ** powers_idx
    C1 = p1_powers[dt_abs]
    A = _symmetrize(Q_sub.T @ (C1 @ Q_sub))
    S = s1 * A + s2 * S2
    try:
        L = scipy.linalg.cholesky(S, lower=True)
    except scipy.linalg.LinAlgError:
        return _NLL_FAIL, np.zeros(3, dtype=np.float64)

    a = scipy.linalg.solve_triangular(L, z, lower=True)
    nll = float(np.sum(np.log(np.diag(L))) + 0.5 * (a @ a))
    if not np.isfinite(nll) or nll >= _NLL_FAIL:
        return _NLL_FAIL, np.zeros(3, dtype=np.float64)

    # ``a = L^{-1} z`` is already available; reuse it instead of solving
    # both triangular systems again inside cho_solve.
    w = scipy.linalg.solve_triangular(L.T, a, lower=False)
    r = len(z)
    lower_inv, info = scipy.linalg.lapack.dpotri(L, lower=1)
    if info == 0:
        lower = np.tril(lower_inv)
        S_inv = lower + np.tril(lower, -1).T
    else:
        S_inv = _symmetrize(scipy.linalg.cho_solve((L, True), np.eye(r)))

    # theta_2 = log(s2): dS/dtheta_2 = s2 * S2.
    tr_inv_S2 = float(np.sum(S_inv * S2))
    w_S2_w = float(w @ (S2 @ w))
    g2 = 0.5 * s2 * (tr_inv_S2 - w_S2_w)

    # theta_0 = log(s1): dS/dtheta_0 = S - s2 * S2.  The trace and quadratic
    # terms telescope, leaving an O(1) expression once g2 is known.
    g0 = 0.5 * float(r_rank) - 0.5 * float(z @ w) - g2

    # theta_1: dC1/dphi1 * dphi1/dtheta_1.
    dp1 = p1 * (1.0 - p1 / 0.95)
    d_powers = np.zeros_like(powers_idx, dtype=np.float64)
    d_powers[1:] = powers_idx[1:] * (p1 ** (powers_idx[1:] - 1.0))
    dC1 = d_powers[dt_abs] * dp1
    dS1 = _symmetrize(s1 * (Q_sub.T @ (dC1 @ Q_sub)))
    tr_inv_dS1 = float(np.sum(S_inv * dS1))
    w_dS1_w = float(w @ (dS1 @ w))
    g1 = 0.5 * (tr_inv_dS1 - w_dS1_w)

    return nll, np.array([g0, g1, g2], dtype=np.float64)


def _run_start(
    theta0: np.ndarray,
    z: np.ndarray,
    Q_sub: np.ndarray,
    dt_abs: np.ndarray,
    powers_idx: np.ndarray,
    S2: np.ndarray,
    r_rank: int,
) -> scipy.optimize.OptimizeResult:
    def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return _objective_and_gradient(
            theta, z, Q_sub, dt_abs, powers_idx, S2, r_rank
        )

    return scipy.optimize.minimize(
        fun,
        theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(-_LOGIT_BOUND, _LOGIT_BOUND)] * 3,
        options={
            "maxiter": 200,
            "maxfun": 2000,
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxls": 50,
        },
    )


def _start_ok(
    opt: scipy.optimize.OptimizeResult,
    s_init: float,
    grad_tol: float,
) -> bool:
    """Screening predicate for the two central starts."""
    if not bool(opt.success) or not np.isfinite(opt.fun) or float(opt.fun) >= 1e11:
        return False
    jac = np.asarray(getattr(opt, "jac", np.full(3, np.nan)), dtype=np.float64)
    if jac.shape != (3,) or not np.all(np.isfinite(jac)):
        return False
    if float(np.max(np.abs(jac))) > float(grad_tol):
        return False
    v1 = float(np.exp(opt.x[0]))
    v2 = float(np.exp(opt.x[2]))
    if not (np.isfinite(v1) and np.isfinite(v2)):
        return False
    return v1 > _VAR_FLOOR * s_init and v2 > _VAR_FLOOR * s_init


def fit_two_component_reml_lbfgsb(
    z: np.ndarray,
    Q_sub: np.ndarray,
    dates: np.ndarray,
    rss_0: float,
    r_rank: int,
    phi_init: float,
    *,
    adaptive_screening: bool | None = None,
    grad_tol: float = _GRAD_SCREEN_TOL,
) -> tuple[str, dict[str, Any]]:
    """Fit the two-component calendar covariance with L-BFGS-B.

    Parameters match the historical ``fit_two_component_reml``.  With
    ``adaptive_screening`` enabled (default, unless
    ``MULTISOURCE_REML_SCREENING=0``), the central variance share
    ``alpha = 0.5`` is fitted for both ``phi2`` candidates first.  The four
    remaining starts are evaluated only when either central start fails the
    gradient, convergence, or variance-floor screen.
    """
    t_start = time.perf_counter()
    z = np.asarray(z, dtype=np.float64)
    Q_sub = np.asarray(Q_sub, dtype=np.float64)
    dates = np.asarray(dates, dtype=np.int64)

    dt_abs = np.abs(dates[:, None] - dates[None, :]).astype(np.int64)
    max_dt = int(np.max(dt_abs))
    powers_idx = np.arange(max_dt + 1, dtype=np.float64)
    s_init = float(rss_0) / float(r_rank)
    th1_init = float(
        np.clip(
            np.log(phi_init / max(0.95 - phi_init, 1e-12)),
            -_LOGIT_BOUND,
            _LOGIT_BOUND,
        )
    )
    if adaptive_screening is None:
        adaptive_screening = (
            os.environ.get("MULTISOURCE_REML_SCREENING", "1").strip() != "0"
        )

    S2_cache: dict[float, np.ndarray] = {}
    for phi2 in PHI2_CANDIDATES:
        p2_powers = phi2 ** np.arange(max_dt + 1, dtype=np.float64)
        K2 = p2_powers[dt_abs]
        S2_cache[float(phi2)] = _symmetrize(Q_sub.T @ (K2 @ Q_sub))

    results: list[dict[str, Any]] = []

    def run_start(
        phi2_idx: int,
        alpha_idx: int,
        phi1_start: float | None = None,
        canonical_start: int | None = None,
    ) -> dict[str, Any]:
        phi2 = float(PHI2_CANDIDATES[phi2_idx])
        alpha = float(FAST_SHARES[alpha_idx])
        th1 = th1_init
        if phi1_start is not None:
            phi1_start = float(np.clip(phi1_start, 1e-8, 0.95 - 1e-8))
            th1 = float(
                np.clip(
                    np.log(phi1_start / (0.95 - phi1_start)),
                    -_LOGIT_BOUND,
                    _LOGIT_BOUND,
                )
            )
        theta0 = np.array(
            [
                np.log(alpha * s_init),
                th1,
                np.log((1.0 - alpha) * s_init),
            ],
            dtype=np.float64,
        )
        opt = _run_start(
            theta0, z, Q_sub, dt_abs, powers_idx, S2_cache[phi2], r_rank
        )
        result = {
            "canonical_start": (
                int(canonical_start)
                if canonical_start is not None
                else 1 + 3 * phi2_idx + alpha_idx
            ),
            "phi2_idx": phi2_idx,
            "alpha_idx": alpha_idx,
            "opt": opt,
        }
        results.append(result)
        return result

    central = [run_start(i, 1) for i in range(2)]
    central_ok = [_start_ok(item["opt"], s_init, grad_tol) for item in central]
    screening_used = bool(adaptive_screening)
    if not (adaptive_screening and all(central_ok)):
        for i in range(2):
            for j in (0, 2):
                run_start(i, j)

    low_phi1_used = False
    if adaptive_screening:
        low_phi1_used = True
        for i in range(2):
            for k, phi1 in enumerate(LOW_PHI1_STARTS):
                run_start(
                    i, 0, phi1_start=phi1, canonical_start=20 + 2 * i + k
                )

    def _valid_results() -> list[dict[str, Any]]:
        return [
            item
            for item in results
            if bool(item["opt"].success)
            and np.isfinite(item["opt"].fun)
            and float(item["opt"].fun) < 1e11
        ]

    base_valid = _valid_results()
    fallback_phi1_used = False
    if adaptive_screening:
        suspicious = True
        if base_valid:
            best_base = min(base_valid, key=lambda item: float(item["opt"].fun))
            best_phi1 = float(0.95 * expit(best_base["opt"].x[1]))
            best_v1 = float(np.exp(best_base["opt"].x[0]))
            best_v2 = float(np.exp(best_base["opt"].x[2]))
            suspicious = (
                best_phi1 > 0.9
                and min(best_v1, best_v2) < _BOUNDARY_FRAC * s_init
            )
        if suspicious:
            fallback_phi1_used = True
            for i in range(2):
                for k, phi1 in enumerate(PHI1_FALLBACK_STARTS):
                    run_start(
                        i, 1, phi1_start=phi1, canonical_start=7 + 3 * i + k
                    )

    valid = _valid_results()
    num_converged = len(valid)
    elapsed = float(time.perf_counter() - t_start)

    if not valid:
        return "failed", {
            "s_init": float(s_init),
            "phi_init": float(phi_init),
            "v1": np.nan,
            "phi1": np.nan,
            "v2": np.nan,
            "phi2": np.nan,
            "reml_nll": np.nan,
            "num_starts_converged": 0,
            "best_start_idx": -1,
            "nit": 0,
            "reml_elapsed_sec": elapsed,
            "optimizer": "lbfgsb",
            "screening_used": screening_used,
            "central_starts_ok": bool(all(central_ok)),
            "fallback_phi1_used": fallback_phi1_used,
            "low_phi1_used": low_phi1_used,
            "continuity_fallback": False,
            "grad_inf_norm": np.nan,
        }

    best = min(valid, key=lambda item: float(item["opt"].fun))
    best_fun = float(best["opt"].fun)

    # Multi-start L-BFGS-B can expose more than one basin.  The historical
    # Nelder-Mead solver is the reference for those genuinely ambiguous
    # tasks; running it only here keeps the fast path for the overwhelming
    # majority while preserving endpoint continuity where basins disagree.
    continuity_fallback = False
    low_valid = [
        item for item in valid if int(item["canonical_start"]) >= 20
    ]
    canonical_valid = [
        item for item in valid if int(item["canonical_start"]) < 20
    ]
    ambiguous = False
    if low_valid and canonical_valid:
        low_best = min(float(item["opt"].fun) for item in low_valid)
        canonical_best = min(float(item["opt"].fun) for item in canonical_valid)
        ambiguous = low_best < canonical_best - _AMBIGUITY_NLL_TOL
    # Near the phi1 upper bound with non-negligible slow variance, the
    # likelihood surface is flat enough that the historical simplex and
    # L-BFGS-B can select different neighboring optima even when the low-phi1
    # anchors do not improve.  Route these boundary fits through the reference
    # solver to keep endpoints stable.
    best_phi1_value = float(0.95 * expit(best["opt"].x[1]))
    best_v2_value = float(np.exp(best["opt"].x[2]))
    if adaptive_screening and best_phi1_value > 0.9 and best_v2_value > 1.0:
        ambiguous = True
    if (
        adaptive_screening
        and ambiguous
        and os.environ.get("MULTISOURCE_REML_CONTINUITY", "1").strip() != "0"
    ):
        from . import common  # local import avoids an import cycle

        try:
            historical = common.load_endpoint_engine().fit_two_component_reml(
                z, Q_sub, dates, rss_0, r_rank, phi_init
            )
        except Exception:
            historical = ("failed", {})
        hist_status, hist_diag = historical
        if (
            hist_status == "converged"
            and np.isfinite(float(hist_diag.get("reml_nll", np.nan)))
        ):
            continuity_fallback = True
            out = dict(hist_diag)
            out["optimizer"] = "nelder_continuity"
            out["screening_used"] = False
            out["central_starts_ok"] = bool(all(central_ok))
            out["fallback_phi1_used"] = fallback_phi1_used
            out["low_phi1_used"] = low_phi1_used
            out["continuity_fallback"] = True
            out["grad_inf_norm"] = np.nan
            out["lbfgsb_best_nll"] = best_fun
            out["lbfgsb_best_phi1"] = float(
                np.clip(0.95 * expit(best["opt"].x[1]), 1e-8, 0.95 - 1e-8)
            )
            out["lbfgsb_best_v1"] = float(np.exp(best["opt"].x[0]))
            out["lbfgsb_best_v2"] = float(np.exp(best["opt"].x[2]))
            out["lbfgsb_best_phi2"] = float(PHI2_CANDIDATES[best["phi2_idx"]])
            return "converged", out

    opt = best["opt"]
    jac = np.asarray(opt.jac, dtype=np.float64)
    best_result: dict[str, Any] = {
        "v1": float(np.exp(opt.x[0])),
        "phi1": float(np.clip(0.95 * expit(opt.x[1]), 1e-8, 0.95 - 1e-8)),
        "v2": float(np.exp(opt.x[2])),
        "phi2": float(PHI2_CANDIDATES[best["phi2_idx"]]),
        "reml_nll": float(opt.fun),
        "num_starts_converged": int(num_converged),
        "best_start_idx": int(best["canonical_start"]),
        "nit": int(opt.nit),
        "reml_elapsed_sec": elapsed,
        "optimizer": "lbfgsb",
        "screening_used": screening_used,
        "central_starts_ok": bool(all(central_ok)),
        "fallback_phi1_used": fallback_phi1_used,
        "low_phi1_used": low_phi1_used,
        "continuity_fallback": continuity_fallback,
        "grad_inf_norm": float(np.max(np.abs(jac))) if jac.size else np.nan,
    }
    best_result["s_init"] = float(s_init)
    best_result["phi_init"] = float(phi_init)
    return "converged", best_result


def optimizer_name() -> str:
    return os.environ.get("MULTISOURCE_REML_OPTIMIZER", "lbfgsb").strip().lower()


def use_lbfgsb() -> bool:
    return optimizer_name() not in {"nelder", "nelder-mead", "legacy", "old"}


def active_fitter():
    """Return the L-BFGS-B fitter, or ``None`` for the historical default."""
    return fit_two_component_reml_lbfgsb if use_lbfgsb() else None
