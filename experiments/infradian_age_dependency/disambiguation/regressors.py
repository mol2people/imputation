"""Lunar and seasonal regressors for the 25-35 d infradian disambiguation.

The phase models are truncated Meeus series (Astronomical Algorithms, 2nd ed.,
chs. 47-49). Accuracy: elongation and true anomaly ~0.3 deg, i.e. a residual
phase error < 0.03 d, which reduces the projected variance by < 0.1 %.

Everything here is deterministic and unit-tested by validate.py.
"""
from __future__ import annotations

import numpy as np

SYNODIC_MONTH_DAYS = 29.530588853
ANOMALISTIC_MONTH_DAYS = 27.554549878
ANNUAL_DAYS = 365.25
J2000_JD = 2451545.0
JD_UNIX_EPOCH = 2440587.5  # Julian date at 1970-01-01T00:00Z (day 0 of the manifests)

# Pre-specified candidate periods (PLAN.md). Values are days.
CANDIDATE_PERIODS = {
    "synodic": SYNODIC_MONTH_DAYS,
    "anomalistic": ANOMALISTIC_MONTH_DAYS,
    "four_week": 28.0,
    "annual_harmonic": ANNUAL_DAYS / 12.0,
    "menstrual_median": 29.3,
}


def _julian_centuries(days_since_1970: np.ndarray) -> np.ndarray:
    jd = JD_UNIX_EPOCH + np.asarray(days_since_1970, dtype=np.float64)
    return (jd - J2000_JD) / 36525.0


def lunar_elongation_deg(days_since_1970: np.ndarray) -> np.ndarray:
    """True geocentric Sun-Moon elongation in degrees; 0 = new moon, 180 = full moon.

    Mean elongation D plus the dominant periodic terms (equation of the centre,
    evection, variation). Truncated at the 0.1 deg term.
    """
    t = _julian_centuries(days_since_1970)
    d = np.deg2rad((297.8501921 + 445267.1114034 * t - 0.0018819 * t**2
                    + t**3 / 545868.0 - t**4 / 113065000.0) % 360.0)
    m = np.deg2rad((357.5291092 + 35999.0502909 * t - 0.0001536 * t**2
                    + t**3 / 24490000.0) % 360.0)
    mp = np.deg2rad((134.9633964 + 477198.8675055 * t + 0.0087414 * t**2
                     + t**3 / 69699.0 - t**4 / 14712000.0) % 360.0)
    psi = (np.rad2deg(d)
           + 6.289 * np.sin(mp)
           - 2.100 * np.sin(m)
           + 1.274 * np.sin(2.0 * d - mp)
           + 0.658 * np.sin(2.0 * d)
           + 0.214 * np.sin(2.0 * mp)
           + 0.110 * np.sin(d))
    return psi % 360.0


def lunar_elongation_wrapped_deg(days_since_1970: np.ndarray) -> np.ndarray:
    """Elongation mapped to [-180, 180)."""
    return (lunar_elongation_deg(days_since_1970) + 180.0) % 360.0 - 180.0


def lunar_true_anomaly_deg(days_since_1970: np.ndarray) -> np.ndarray:
    """Moon's true anomaly in degrees; 0 at perigee (maximum tidal forcing)."""
    t = _julian_centuries(days_since_1970)
    mp = np.deg2rad((134.9633964 + 477198.8675055 * t + 0.0087414 * t**2
                     + t**3 / 69699.0 - t**4 / 14712000.0) % 360.0)
    nu = (np.rad2deg(mp)
          + 6.289 * np.sin(mp)
          + 1.274 * np.sin(2.0 * mp)
          + 0.658 * np.sin(3.0 * mp)
          + 0.214 * np.sin(4.0 * mp))
    return nu % 360.0


def design_baseline(dates: np.ndarray, clock_fracs: np.ndarray) -> np.ndarray:
    """Engine baseline: intercept, linear time, 6 weekday indicators, 4 clock fractions."""
    dates = np.asarray(dates, dtype=np.int64)
    n = len(dates)
    cols = [np.ones(n, dtype=np.float64),
            (dates - dates[0]).astype(np.float64)]
    dow = (dates + 3) % 7  # 0 = Monday
    for d in range(1, 7):
        cols.append((dow == d).astype(np.float64))
    for c in range(4):
        cols.append(np.asarray(clock_fracs, dtype=np.float64)[:, c])
    return np.column_stack(cols)


def design_extension(dates: np.ndarray) -> np.ndarray:
    """Six disambiguation columns: lunar L (4) then seasonal S (2)."""
    dates = np.asarray(dates, dtype=np.int64)
    psi = np.deg2rad(lunar_elongation_deg(dates))
    nu = np.deg2rad(lunar_true_anomaly_deg(dates))
    span = float(dates[-1] - dates[0])
    tau = 2.0 * (dates.astype(np.float64) - dates[0]) / max(span, 1.0) - 1.0
    return np.column_stack([np.cos(psi), np.sin(psi), np.cos(nu), np.sin(nu),
                            tau**2, tau**3])


def screen_qr(X: np.ndarray) -> tuple[np.ndarray, int]:
    """Sequential QR screening, identical to the engine's nuisance baseline builder.

    Drops a column when ||v|| < 1e-10 or ||v_perp|| / ||v|| <= 1e-4.
    """
    Q = np.zeros((X.shape[0], 0), dtype=np.float64)
    for j in range(X.shape[1]):
        v = X[:, j].astype(np.float64)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            continue
        if Q.shape[1] > 0:
            v_proj = Q @ (Q.T @ v)
            v_perp = v - v_proj
            perp_norm = np.linalg.norm(v_perp)
            if perp_norm / v_norm > 1e-4:
                Q = np.column_stack([Q, v_perp / perp_norm])
        else:
            Q = (v / v_norm)[:, np.newaxis]
    return Q, Q.shape[1]


def build_and_screen_nuisance_baseline_ext(dates: np.ndarray, clock_fracs: np.ndarray):
    """Drop-in replacement for the engine's 12-column baseline builder.

    Returns (P_0, Q_basis, p_retained) with the same contract as
    scripts/02_compute_endpoints_and_nulls.py:build_and_screen_nuisance_baseline.
    """
    dates = np.asarray(dates, dtype=np.int64)
    X = np.column_stack([design_baseline(dates, clock_fracs), design_extension(dates)])
    Q, p_retained = screen_qr(X)
    n = X.shape[0]
    P_0 = np.eye(n, dtype=np.float64) - (Q @ Q.T)
    return P_0, Q, p_retained


def partial_r2(y: np.ndarray, X_base: np.ndarray, blocks: list[np.ndarray]) -> list[float]:
    """Incremental R^2 of each block, added sequentially after X_base.

    blocks are appended in order; each R^2 is measured against the residual of
    everything before it. Returns one value per block.
    """
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        return [np.nan] * len(blocks)
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0:
        return [np.nan] * len(blocks)
    Q, _ = screen_qr(X_base)
    r = y - Q @ (Q.T @ y)
    out = []
    for block in blocks:
        block = np.asarray(block, dtype=np.float64)
        # orthogonalise the block against everything retained so far
        Qr, pr = screen_qr(block - Q @ (Q.T @ block))
        if pr == 0:
            out.append(0.0)
            continue
        r_new = r - Qr @ (Qr.T @ r)
        out.append(float((np.sum(r**2) - np.sum(r_new**2)) / sst))
        Q = np.column_stack([Q, Qr])
        r = r_new
    return out


def cosinor_columns(dates: np.ndarray, period_days: float) -> np.ndarray:
    """cos/sin pair at a fixed period (days since 1970-01-01 phase reference)."""
    f = 2.0 * np.pi / float(period_days)
    d = np.asarray(dates, dtype=np.float64)
    return np.column_stack([np.cos(f * d), np.sin(f * d)])


def acrophase_days(dates: np.ndarray, period_days: float, beta: np.ndarray) -> float:
    """Acrophase time of y ~ a cos(wt) + b sin(wt) = A cos(w(t - t_star)).

    t_star = atan2(b, a) / w modulo the period, expressed on the day axis.
    """
    a, b = float(beta[0]), float(beta[1])
    w = 2.0 * np.pi / float(period_days)
    t_star = np.arctan2(b, a) / w
    return float((t_star % period_days) + 0.0)
