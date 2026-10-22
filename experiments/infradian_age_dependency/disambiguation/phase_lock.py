#!/usr/bin/env python3
"""Lunar phase-lock tests for the 25-35 d infradian HR excess.

Two directed tests (both valid at 180/270 d spans, no frequency separation needed):

T1  lunar-direction test.  Each window is regressed on (cos psi, sin psi), where
    psi is the true geocentric Sun-Moon elongation.  The coefficient angle is the
    lunar phase at which HR peaks; under lunar forcing it is concentrated across
    participants, under an endogenous rhythm it is uniform.  Same for nu (true
    anomaly, perigee).  No tautology: psi is an external physical variable, not
    calendar time.

T2  acrophase test.  The period is estimated freely on a 0.05 d grid over 25-35 d;
    the cosinor acrophase is mapped to psi and nu.  Concentration means the
    rhythm is lunar-locked.  Attenuated by period-estimation error.

T3  circaseptan control: day-of-week of the empirical acrophase.

T2/T3 use a permutation null (shuffle psi across participants) because the
marginal psi distribution is not exactly uniform.  T1 uses the asymptotic
Rayleigh test: under isotropic noise the coefficient angle is uniform by
construction.

Usage:
  python phase_lock.py compute --protocols balanced90 expanded90 --workers 8 --run DIR
  python phase_lock.py analyze --run DIR
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT_DIR / "rhr_sensitivity"))
sys.path.insert(0, str(EXPERIMENT_DIR / "scripts"))

import regressors as R  # noqa: E402
import shared  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "infradian_endpoint_engine", EXPERIMENT_DIR / "scripts/02_compute_endpoints_and_nulls.py")
engine = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(engine)

PERIOD_GRID = np.round(np.arange(25.0, 35.0001, 0.05), 4)
COHERENCE_PERIODS = np.round(np.arange(25.0, 35.001, 0.1), 3)   # Rbar(T) spectrum
STRATA = {
    "F20_35": lambda s, a: (s == 1) & (a >= 20) & (a <= 35),
    "F36_50": lambda s, a: (s == 1) & (a >= 36) & (a <= 50),
    "F51_75": lambda s, a: (s == 1) & (a >= 51) & (a <= 75),
    "M20_35": lambda s, a: (s == 0) & (a >= 20) & (a <= 35),
    "M36_50": lambda s, a: (s == 0) & (a >= 36) & (a <= 50),
    "M51_75": lambda s, a: (s == 0) & (a >= 51) & (a <= 75),
}


def _harmonic_fit(Q0: np.ndarray, e0: np.ndarray, dates: np.ndarray,
                  freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Explained SS of a cos/sin pair at every frequency, after projector Q0."""
    w = 2.0 * np.pi * np.outer(dates.astype(np.float64), freqs)  # (n, K)
    C = np.cos(w)
    S = np.sin(w)
    Cp = C - Q0 @ (Q0.T @ C)
    Sp = S - Q0 @ (Q0.T @ S)
    a11 = np.einsum("ij,ij->j", Cp, Cp)
    a22 = np.einsum("ij,ij->j", Sp, Sp)
    a12 = np.einsum("ij,ij->j", Cp, Sp)
    b1 = Cp.T @ e0
    b2 = Sp.T @ e0
    det = a11 * a22 - a12**2
    det = np.where(np.abs(det) < 1e-14, np.nan, det)
    bc = (a22 * b1 - a12 * b2) / det
    bs = (a11 * b2 - a12 * b1) / det
    ss = bc * b1 + bs * b2
    return ss, bc, bs


def _direction_angle(Q0: np.ndarray, e0: np.ndarray, cols: np.ndarray) -> tuple[float, float, float]:
    Cp = cols - Q0 @ (Q0.T @ cols)
    beta, *_ = np.linalg.lstsq(Cp, e0, rcond=None)
    rss = float(e0 @ e0)
    amp = float(np.hypot(beta[0], beta[1]))
    ang = float(np.degrees(np.arctan2(beta[1], beta[0])) % 360.0)
    r2 = float(np.sum((Cp @ beta) ** 2) / rss) if rss > 0 else np.nan
    return ang, amp, r2


N_ROT = 32


def _window_record(dates: np.ndarray, y: np.ndarray, clock: np.ndarray) -> dict:
    Q0, _ = R.screen_qr(R.design_baseline(dates, clock))
    e0 = y - Q0 @ (Q0.T @ y)
    rss = float(e0 @ e0)

    out: dict = {}
    ext = R.design_extension(dates)
    psi = np.deg2rad(R.lunar_elongation_deg(dates))
    nu = np.deg2rad(R.lunar_true_anomaly_deg(dates))

    # T1: lunar and anomalistic direction at the exact physical phase variables
    for tag, ang, amp, r2 in (
        ("lunar", *_direction_angle(Q0, e0, ext[:, :2])),
        ("ano", *_direction_angle(Q0, e0, ext[:, 2:4])),
    ):
        out[f"{tag}_angle"], out[f"{tag}_amp"], out[f"{tag}_R2"] = ang, amp, r2

    # Rotation null: the lunar and anomalistic bases rotated by the same random
    # offset within this window.  Preserves the data, the dates, the baseline, and
    # the within-window oscillation; destroys any global phase alignment.
    rng = np.random.default_rng(abs(hash((int(dates[0]), int(dates[-1]), len(dates)))) % (2**32))
    rot_l, rot_a = [], []
    for _ in range(N_ROT):
        delta = rng.uniform(0.0, 2.0 * np.pi)
        rot_l.append(_direction_angle(Q0, e0, np.column_stack([
            np.cos(psi + delta), np.sin(psi + delta)]))[0])
        rot_a.append(_direction_angle(Q0, e0, np.column_stack([
            np.cos(nu + delta), np.sin(nu + delta)]))[0])
    out["lunar_rot"] = rot_l
    out["ano_rot"] = rot_a

    # Sham-period specificity (not physical phases).  Out-of-band periods 18/48/55/60 d
    # are >= 2 kernel widths from the 25-35 d band edges, so any leak from the band
    # edge into them is small (22/42 d were rejected for that reason).
    for tag, period in (("sham18", 18.0), ("sham27", 27.0), ("sham28", 28.0),
                        ("sham29", 29.0), ("sham34", 34.0), ("sham48", 48.0),
                        ("sham55", 55.0), ("sham60", 60.0)):
        out[f"{tag}_angle"], out[f"{tag}_amp"], out[f"{tag}_R2"] = _direction_angle(
            Q0, e0, R.cosinor_columns(dates, period))

    # Coverage coherence (phase-lock check B).  HRp10 is a within-day quantile, so the
    # four daily clock fractions bias it nonlinearly; the baseline absorbs them only
    # linearly.  The raw coverage series is an upper bound on any coverage-mediated
    # coherence: a nonlinear map creates harmonics, not subharmonics, and a lagged map
    # shifts the phase without changing the coherence magnitude.  (Residualising
    # coverage on the baseline is an identity, because the fractions are columns of it.)
    q_int = np.ones((len(dates), 1), dtype=np.float64) / np.sqrt(len(dates))
    clock = np.asarray(clock, dtype=np.float64)
    cov = {"night": clock[:, 0], "total": clock.sum(1)}
    out["angle_cov"], out["amp_cov"] = {}, {}
    for name, x in cov.items():
        e = x - x.mean()
        if not np.all(np.isfinite(e)) or float(np.ptp(e)) <= 1e-9:
            ang = np.full(len(COHERENCE_PERIODS), np.nan, dtype=np.float32)
            amp = ang.copy()
        else:
            _, bc, bs = _harmonic_fit(q_int, e, dates, 1.0 / COHERENCE_PERIODS)
            ang = (np.degrees(np.arctan2(bs, bc)) % 360.0).astype(np.float32)
            amp = np.hypot(bc, bs).astype(np.float32)
        out["angle_cov"][f"ang_cov_{name}"] = ang
        out["amp_cov"][f"amp_cov_{name}"] = amp

    # Cross-participant phase-coherence spectrum Rbar(T): the direction of the
    # cos/sin coefficient at every trial period.  A common (environmental) rhythm
    # or a systematic calendar artifact shows up as a peak; an endogenous rhythm
    # with random phases does not.
    ss_s, bc_s, bs_s = _harmonic_fit(Q0, e0, dates, 1.0 / COHERENCE_PERIODS)
    out["angle_spec"] = (np.degrees(np.arctan2(bs_s, bc_s)) % 360.0).astype(np.float32)

    # T2: free period on the fine grid
    ss, bc, bs = _harmonic_fit(Q0, e0, dates, 1.0 / PERIOD_GRID)
    if rss <= 0 or not np.all(np.isfinite(ss)):
        out.update({"T_star": np.nan, "A_star": np.nan, "R2_star": np.nan,
                    "acrophase_day": np.nan, "psi_acrophase": np.nan,
                    "nu_acrophase": np.nan, "weekday": np.nan})
        return out
    k = int(np.nanargmax(ss))
    T = float(PERIOD_GRID[k])
    w = 2.0 * np.pi / T
    phi = float(np.arctan2(bs[k], bc[k]))
    t_star = phi / w
    t_star = dates[0] + ((t_star - dates[0]) % T)  # first cycle inside the window
    out.update({
        "T_star": T,
        "A_star": float(np.hypot(bc[k], bs[k])),
        "R2_star": float(ss[k] / rss),
        "acrophase_day": float(t_star),
        "psi_acrophase": float(R.lunar_elongation_deg(np.array([t_star]))[0]),
        "nu_acrophase": float(R.lunar_true_anomaly_deg(np.array([t_star]))[0]),
        "weekday": int(round(t_star) + 3) % 7,  # 0 = Monday
    })
    return out


def _process(task: dict) -> tuple[list[dict], list[dict]]:
    rows, spec_rows = [], []
    cache = Path(task["cache_path"])
    if not cache.exists():
        return rows, spec_rows
    try:
        with np.load(cache) as d:
            hr, date_arr = d["hr"], d["date"]
        date_to_row = {int(v): i for i, v in enumerate(date_arr)}
    except Exception:
        return rows, spec_rows
    for win in task["windows"]:
        dates = np.asarray(win["dates"], dtype=np.int64)
        y, clock, *_ = engine.extract_features_for_selected_dates(
            hr, date_to_row, dates, 10.0)
        if not np.all(np.isfinite(y)):
            continue
        rec = _window_record(dates, y, clock)
        spec = rec.pop("angle_spec")
        cov_a = rec.pop("angle_cov")
        cov_m = rec.pop("amp_cov")
        spec_rows.append({"user_id": task["user_id"], "source": task["source"],
                          "protocol": win["protocol"], "window_index": win["window_index"],
                          "angles": spec,
                          **cov_a, **cov_m})
        rec.update({"user_id": task["user_id"], "source": task["source"],
                    "protocol": win["protocol"], "window_index": win["window_index"],
                    "win_start": int(dates[0]), "win_end": int(dates[-1]),
                    "age": task["demographics"]["age_dec1"],
                    "sex": 1 if task["demographics"]["salutation"] == 10 else 0})
        rows.append(rec)
    return rows, spec_rows


def compute(protocol: str, workers: int, out: Path, limit: int | None) -> None:
    tasks = shared.build_tasks("HRp10", protocol, limit)
    print(f"{protocol}: {len(tasks)} participant tasks")
    rows: list[dict] = []
    specs: list[dict] = []
    if workers > 1:
        chunk = max(1, len(tasks) // (workers * 8) + 1)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res, spec in ex.map(_process, tasks, chunksize=chunk):
                rows.extend(res)
                specs.extend(spec)
    else:
        for t in tasks:
            res, spec = _process(t)
            rows.extend(res)
            specs.extend(spec)
    df = pd.DataFrame(rows)
    path = out / f"phase_records_{protocol}.parquet"
    df.to_parquet(path, index=False)
    keys = pd.DataFrame([{k: v for k, v in r.items()
                          if k != "angles"
                          and not k.startswith(("ang_cov_", "amp_cov_"))}
                         for r in specs])
    cov_arrays = {k: np.stack([r[k] for r in specs])
                  for k in specs[0] if k.startswith(("ang_cov_", "amp_cov_"))}
    np.savez_compressed(out / f"phase_coherence_{protocol}.npz",
                        user_id=keys.user_id.to_numpy(), source=keys.source.to_numpy(),
                        protocol=keys.protocol.to_numpy(),
                        window_index=keys.window_index.to_numpy(),
                        periods=COHERENCE_PERIODS,
                        angles=np.stack([r["angles"] for r in specs]),
                        **cov_arrays)
    print(f"wrote {path} ({len(df)} windows) and coherence npz ({len(specs)})")


# ---------------------------------------------------------------- analysis


def _rayleigh(theta_deg: np.ndarray, n_boot: int = 2000,
              seed: int = 0) -> dict:
    th = np.deg2rad(theta_deg)
    n = len(th)
    cos_m, sin_m = float(np.mean(np.cos(th))), float(np.mean(np.sin(th)))
    rbar = float(np.hypot(cos_m, sin_m))
    z = n * rbar**2
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = np.hypot(np.mean(np.cos(th[idx])), np.mean(np.sin(th[idx])))
    return {"rbar": rbar, "p_rayleigh": float(np.exp(-z)),
            "mean_angle": float(np.degrees(np.arctan2(sin_m, cos_m)) % 360.0),
            "ci_lo": float(np.quantile(boot, 0.025)),
            "ci_hi": float(np.quantile(boot, 0.975)),
            "rbar_null": float(np.sqrt(1.0 / n))}


def _acrophase_null_rbar(starts: np.ndarray, ends: np.ndarray, which: str,
                         n_draw: int = 400, seed: int = 0) -> np.ndarray:
    """Null Rbar for T2: acrophase uniform inside each window, psi/nu evaluated there."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_draw)
    for b in range(n_draw):
        t = rng.uniform(starts, ends)
        ang = R.lunar_elongation_deg(t) if which == "psi" else R.lunar_true_anomaly_deg(t)
        th = np.deg2rad(ang)
        out[b] = np.hypot(np.mean(np.cos(th)), np.mean(np.sin(th)))
    return out


def analyze(run: Path, protocols: list[str]) -> None:
    frames = []
    for p in protocols:
        f = run / f"phase_records_{p}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        raise SystemExit("no phase records found")
    df = pd.concat(frames, ignore_index=True)
    df["age_band"] = pd.cut(df.age, [19, 35, 50, 75], labels=["20-35", "36-50", "51-75"])
    df["stratum"] = np.where(df.sex == 1, "F", "M") + df.age_band.astype(str)

    # one window per participant for the primary test (independence)
    prim = (df.sort_values("window_index")
              .groupby(["user_id", "source", "protocol"], as_index=False).head(1))

    rows = []
    for protocol in protocols:
        sub = prim.loc[prim.protocol.eq(protocol)]
        for stratum in ["F20-35", "F36-50", "F51-75", "M20-35", "M36-50", "M51-75"]:
            s = sub.loc[sub.stratum.eq(stratum)]
            if len(s) < 30:
                continue
            for phase, col, rotcol in (("lunar", "lunar_angle", "lunar_rot"),
                                       ("ano", "ano_angle", "ano_rot")):
                r = _rayleigh(s[col].to_numpy())
                rm = np.stack([np.asarray(v, dtype=float) for v in s[rotcol].to_numpy()])
                th = np.deg2rad(rm)
                rot = np.hypot(np.cos(th).mean(0), np.sin(th).mean(0))  # (N_ROT,)
                p_rot = float((1 + np.sum(rot >= r["rbar"])) / (1 + len(rot)))
                rows.append({"protocol": protocol, "stratum": stratum, "test": f"T1_{phase}",
                             "n": len(s), "stat": r["rbar"], "p": r["p_rayleigh"],
                             "p_rot": p_rot, "mean_angle": r["mean_angle"],
                             "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                             "rbar_null": float(rot.mean()),
                             "note": f"null95={np.quantile(rot,0.95):.3f}"})
            for tag in ("sham18", "sham27", "sham28", "sham29", "sham34",
                        "sham48", "sham55", "sham60"):
                r = _rayleigh(s[f"{tag}_angle"].to_numpy())
                rows.append({"protocol": protocol, "stratum": stratum, "test": f"S_{tag}",
                             "n": len(s), "stat": r["rbar"], "p": r["p_rayleigh"],
                             "mean_angle": r["mean_angle"], "ci_lo": r["ci_lo"],
                             "ci_hi": r["ci_hi"], "rbar_null": r["rbar_null"],
                             "p_rot": np.nan, "note": ""})
            # T2: empirical acrophase mapped to psi / nu, null from uniform acrophase
            for phase, col in (("psi", "psi_acrophase"), ("nu", "nu_acrophase")):
                r = _rayleigh(s[col].to_numpy())
                null = _acrophase_null_rbar(s["win_start"].to_numpy(float),
                                            s["win_end"].to_numpy(float), phase)
                p_sim = float((1 + np.sum(null >= r["rbar"])) / (1 + len(null)))
                rows.append({"protocol": protocol, "stratum": stratum, "test": f"T2_{phase}",
                             "n": len(s), "stat": r["rbar"], "p": p_sim,
                             "mean_angle": r["mean_angle"], "ci_lo": r["ci_lo"],
                             "ci_hi": r["ci_hi"], "rbar_null": r["rbar_null"],
                             "p_rot": np.nan, "note": f"rayleigh_p={r['p_rayleigh']:.3g}"})
            # T3: weekday of the empirical acrophase
            counts = np.bincount(s["weekday"].astype(int).to_numpy(), minlength=7)
            chi2 = float(np.sum((counts - counts.mean()) ** 2) / counts.mean())
            from scipy.stats import chi2 as chi2_dist
            rows.append({"protocol": protocol, "stratum": stratum, "test": "T3_weekday",
                         "n": len(s), "stat": chi2, "p": float(chi2_dist.sf(chi2, df=6)),
                         "mean_angle": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "rbar_null": np.nan, "p_rot": np.nan, "note": str(counts.tolist())})
            rows.append({"protocol": protocol, "stratum": stratum,
                         "test": "median_T_star", "n": len(s),
                         "stat": float(np.nanmedian(s.T_star)), "p": np.nan,
                         "mean_angle": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "rbar_null": np.nan, "p_rot": np.nan, "note": ""})
            rows.append({"protocol": protocol, "stratum": stratum,
                         "test": "median_R2_star", "n": len(s),
                         "stat": float(np.nanmedian(s.R2_star)), "p": np.nan,
                         "mean_angle": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "rbar_null": np.nan, "p_rot": np.nan, "note": ""})
    summary = pd.DataFrame(rows)
    summary.to_csv(run / "phase_lock_summary.csv", index=False)
    print(summary.to_string(index=False))
    (run / "phase_lock_provenance.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "period_grid_days": [float(PERIOD_GRID[0]), float(PERIOD_GRID[-1]), 0.05],
        "primary": "T1 lunar/ano direction, Rayleigh on coefficient angle (rotation null reported but not used)",
        "no_multiplicity_column": "p_holm_primary removed: the rotation null it was built on is invalid for specificity",
        "secondary": "T2 acrophase-to-psi/nu with uniform-acrophase null; T3 weekday chi-square",
        "one_window_per_participant": True,
    }, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--protocols", nargs="+", default=["balanced90", "expanded90"])
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--limit", type=int, default=None)
    a = sub.add_parser("analyze")
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--protocols", nargs="+", default=["balanced90", "expanded90"])
    args = ap.parse_args()
    if args.cmd == "compute":
        args.run.mkdir(parents=True, exist_ok=True)
        for p in args.protocols:
            compute(p, args.workers, args.run, args.limit)
    else:
        analyze(args.run, args.protocols)


if __name__ == "__main__":
    main()
