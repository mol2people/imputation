#!/usr/bin/env python3
"""Validation gate for the lunar/seasonal disambiguation (PLAN.md, section Validation).

Run before any full run. Writes calibration_recovery.csv (absorption of a unit
sinusoid at each candidate period by the extended projector) and exits non-zero
on failure.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import regressors as R  # noqa: E402

EXPERIMENT_DIR = HERE.parent
MANIFEST_DATES = EXPERIMENT_DIR / "artifacts/windows/selected_dates.parquet"
CALIBRATION_OUT = HERE / "calibration_recovery.csv"

# Known new moons (UTC) from total/annular solar eclipses -- the most precisely
# known new-moon epochs. Source: published eclipse canon.
NEW_MOONS = [
    "2015-03-20T09:36", "2016-03-09T01:54", "2017-08-21T18:30", "2019-07-02T19:16",
    "2020-06-21T06:41", "2021-06-10T10:52", "2023-04-20T04:12", "2024-04-08T18:21",
    "2025-03-29T10:58", "2026-08-12T17:46",
]


def days_since_1970(iso: str) -> float:
    t = dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc)
    return (t - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)).total_seconds() / 86400.0


def test_new_moons() -> None:
    d = np.array([days_since_1970(s) for s in NEW_MOONS])
    psi = R.lunar_elongation_wrapped_deg(d)
    worst = float(np.max(np.abs(psi)))
    print(f"[1] new-moon elongation: max |psi| = {worst:.2f} deg over {len(d)} eclipses")
    assert worst < 6.0, "lunar elongation model fails the new-moon test"


def test_spectral_peaks() -> None:
    d = np.arange(days_since_1970("2000-01-01"), days_since_1970("2026-01-01"))
    for name, series, target, tol in (
        ("cos(elongation)", np.cos(np.deg2rad(R.lunar_elongation_deg(d))), R.SYNODIC_MONTH_DAYS, 0.1),
        ("cos(true anomaly)", np.cos(np.deg2rad(R.lunar_true_anomaly_deg(d))), R.ANOMALISTIC_MONTH_DAYS, 0.15),
    ):
        x = series - series.mean()
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), d=1.0)
        band = (freqs > 1 / 40.0) & (freqs < 1 / 20.0)
        peak_period = 1.0 / freqs[band][np.argmax(spec[band])]
        print(f"[2] {name}: dominant period {peak_period:.3f} d (target {target:.3f} +/- {tol})")
        assert abs(peak_period - target) <= tol, f"{name} spectral peak off target"


def _weekly_redundancy(dates: np.ndarray) -> float:
    dow = (dates + 3) % 7
    X = np.column_stack([np.ones(len(dates))] + [(dow == k).astype(float) for k in range(1, 7)])
    Q, _ = R.screen_qr(X)
    worst = 0.0
    t = dates.astype(np.float64)
    for k in (1, 2, 3, 4):
        for f in (np.cos, np.sin):
            v = f(2 * np.pi * k * t / 7.0)
            v = v - Q @ (Q.T @ v)
            worst = max(worst, float(np.linalg.norm(v) / np.linalg.norm(f(2 * np.pi * k * t / 7.0))))
    return worst


def test_weekly_redundancy() -> None:
    sel = pd.read_parquet(MANIFEST_DATES)
    sample = sel.groupby(["user_id", "source", "protocol", "window_index"]).head(90)
    worst = 0.0
    for _, g in list(sample.groupby(["user_id", "source", "protocol", "window_index"]))[:40]:
        dates = np.sort(g["date"].to_numpy(dtype=np.int64))
        worst = max(worst, _weekly_redundancy(dates))
    print(f"[3] weekly harmonics after projection on weekday space: max relative norm = {worst:.2e}")
    assert worst < 1e-8, "weekly sinusoid is not contained in the weekday-dummy span at day resolution"


def _real_windows(protocol: str, n: int = 40) -> list[np.ndarray]:
    sel = pd.read_parquet(MANIFEST_DATES)
    sel = sel.loc[sel.protocol.eq(protocol)]
    out = []
    for _, g in sel.groupby(["user_id", "source", "window_index"]):
        dates = np.sort(g["date"].to_numpy(dtype=np.int64))
        if len(dates) == 90:
            out.append(dates)
        if len(out) >= n:
            break
    return out


def _recovery_fraction(dates: np.ndarray, clock: np.ndarray, period: float,
                       Q_base_only: np.ndarray, Q_ext: np.ndarray) -> float:
    """Fraction of a unit sinusoid's variance at `period` surviving each projector."""
    c = R.cosinor_columns(dates, period)

    def frac(Q: np.ndarray) -> float:
        r = c - Q @ (Q.T @ c)
        return float(np.sum(r**2) / np.sum(c**2))

    base, ext = frac(Q_base_only), frac(Q_ext)
    return ext / base if base > 0 else np.nan


def calibration_table(protocol: str = "balanced90") -> pd.DataFrame:
    """Absorption of a unit sinusoid at each candidate period by the extended design."""
    rows = []
    rng = np.random.default_rng(0)
    windows = _real_windows(protocol)
    periods = np.round(np.arange(20.0, 60.01, 0.5), 3)
    for dates in windows:
        clock = rng.uniform(0.1, 0.9, size=(90, 4))
        clock = clock / clock.sum(axis=1, keepdims=True)
        Qb, _ = R.screen_qr(R.design_baseline(dates, clock))
        Qe, _ = R.screen_qr(np.column_stack([R.design_baseline(dates, clock),
                                             R.design_extension(dates)]))
        for p in periods:
            rows.append((p, _recovery_fraction(dates, clock, p, Qb, Qe)))
    df = pd.DataFrame(rows, columns=["period", "recovery_fraction"])
    return df.groupby("period", as_index=False)["recovery_fraction"].agg(["mean", "min", "max"]).reset_index()


def test_injection(protocol: str = "balanced90") -> pd.DataFrame:
    """End-to-end injection: design-space recovery of a unit sinusoid.

    The design-space quantity is the one that transfers to the observed band
    excess, because Q_i removes the noise floor via its surrogate null. A noisy
    injection is run as well to confirm the in-band/out-of-band ordering.
    """
    rng = np.random.default_rng(42)
    windows = _real_windows(protocol, n=20)
    print(f"[4] injection test: {len(windows)} real {protocol} windows")
    rows = []
    for dates in windows:
        clock = rng.uniform(0.1, 0.9, (90, 4))
        clock = clock / clock.sum(axis=1, keepdims=True)
        Qb, _ = R.screen_qr(R.design_baseline(dates, clock))
        Qe, _ = R.screen_qr(np.column_stack([R.design_baseline(dates, clock),
                                             R.design_extension(dates)]))
        for period in (27.554550, 28.0, 28.5, 29.3, 29.530589, 30.4375, 23.0, 36.0):
            c = R.cosinor_columns(dates, period)
            clean = c[:, 0]
            noise = _ar1_noise(dates, 0.7, seed=int(rng.integers(1 << 30)))
            rows.append((period,
                         _signal_recovery(clean, c, Qb), _signal_recovery(clean, c, Qe),
                         _signal_recovery(clean + noise, c, Qb),
                         _signal_recovery(clean + noise, c, Qe)))
    df = pd.DataFrame(rows, columns=["period", "clean_base", "clean_ext",
                                     "noisy_base", "noisy_ext"])
    g = df.groupby("period").mean()
    g["clean_recovery"] = g["clean_ext"] / g["clean_base"]
    g["noisy_recovery"] = g["noisy_ext"] / g["noisy_base"]
    print(g[["clean_recovery", "noisy_recovery"]].round(3).to_string())
    in_band = g.loc[[27.554550, 28.0, 28.5, 29.3, 29.530589, 30.4375], "clean_recovery"]
    assert (in_band < 0.05).all(), "in-band injection unexpectedly survives the band-stop"
    assert g.loc[23.0, "clean_recovery"] > 0.90, "23-d injection should survive"
    assert g.loc[36.0, "clean_recovery"] > 0.85, "36-d injection should survive"
    assert g.loc[[27.554550, 28.5], "noisy_recovery"].max() < 0.5 * g.loc[[23.0, 36.0], "noisy_recovery"].min(), \
        "noisy in-band recovery is not clearly separated from out-of-band recovery"
    return g


def _signal_recovery(y: np.ndarray, c: np.ndarray, Q: np.ndarray) -> float:
    """Sum of squares of y explained by the cos/sin pair in the complement of Q.

    Absolute (not variance-normalised) so that ratios across different
    projectors are meaningful: recovery = SS(Q_ext) / SS(Q_base).
    """
    r = c - Q @ (Q.T @ c)
    Qc, p = R.screen_qr(r)
    if p == 0:
        return 0.0
    yres = y - Q @ (Q.T @ y)
    return float(np.sum((Qc.T @ yres) ** 2))


def _ar1_noise(dates: np.ndarray, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = np.empty(len(dates))
    e[0] = rng.standard_normal()
    dt_ = np.diff(dates)
    for k in range(len(dates) - 1):
        decay = phi ** dt_[k]
        e[k + 1] = decay * e[k] + np.sqrt(max(1.0 - decay**2, 0.0)) * rng.standard_normal()
    return e


def test_engine_consistency() -> None:
    import importlib.util
    engine_path = EXPERIMENT_DIR / "scripts/02_compute_endpoints_and_nulls.py"
    spec = importlib.util.spec_from_file_location("engine", engine_path)
    engine = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(engine)

    dates = _real_windows("balanced90", n=1)[0]
    rng = np.random.default_rng(1)
    clock = rng.uniform(0.1, 0.9, size=(90, 4))
    clock = clock / clock.sum(axis=1, keepdims=True)

    P_eng, Q_eng, p_eng = engine.build_and_screen_nuisance_baseline(dates, clock)
    P_ext, Q_ext, p_ext = R.build_and_screen_nuisance_baseline_ext(dates, clock)
    corr = np.abs(np.sum(Q_ext[:, :p_eng] * Q_eng, axis=0))
    print(f"[5] engine consistency: retained {p_eng} -> {p_ext} columns; "
          f"min |<q_ext,q_eng>| = {corr.min():.6f}")
    assert p_ext == p_eng + 6
    assert np.allclose(corr, 1.0, atol=1e-8)
    assert np.allclose(P_ext @ P_eng, P_ext, atol=1e-8)


def main() -> None:
    test_new_moons()
    test_spectral_peaks()
    test_weekly_redundancy()
    for protocol in ("balanced90", "expanded90"):
        table = calibration_table(protocol)
        table.to_csv(CALIBRATION_OUT.with_name(f"calibration_recovery_{protocol}.csv"), index=False)
        band = table.loc[table.period.between(27.5, 30.5), "mean"]
        print(f"[4a] {protocol}: recovery fraction in 27.5-30.5 d = "
              f"{band.min():.3f}-{band.max():.3f}; outside band "
              f"({table.loc[~table.period.between(24, 34), 'mean'].min():.2f}"
              f"-{table.loc[~table.period.between(24, 34), 'mean'].max():.2f})")
    test_injection("balanced90")
    test_engine_consistency()
    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
