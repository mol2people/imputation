#!/usr/bin/env python3
"""S-only seasonal/trend arm (PLAN.md Addendum 1, consequence 2).

The pre-registered seasonal block is tau^2, tau^3 (quadratic + cubic trend over
the window); a full annual sinusoid is not identifiable in a 156-270 d arc.
Addendum 1 measured its absorption of a unit 28-d sinusoid at 2.3-2.5 %, so it
is an identified low-frequency control, unlike the L+S block (a band-stop).

This arm recomputes Q_i through the *identical* engine with the two seasonal
columns appended to the 12-column baseline, then pairs the result with the saved
reference endpoint on (user_id, source, protocol, window_index, hash).

Pre-registered decision rule: the young-female excess is robust if
Q_S / Q_ref >= 0.70; male FPR at p_discrete <= 0.05 should stay near 5 %.

Usage:
  python seasonal_arm.py compute --protocols balanced90 expanded90 --workers 9
  python seasonal_arm.py analyze --run DIR
  python seasonal_arm.py benchmark --limit 20
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO = EXPERIMENT.parents[1]
ENGINE_PATH = EXPERIMENT / "scripts" / "02_compute_endpoints_and_nulls.py"
ENDPOINTS = EXPERIMENT / "artifacts" / "endpoints" / "endpoints.parquet"
DEFAULT_RUN = EXPERIMENT / "artifacts" / "disambiguation" / "run_20261020"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT / "rhr_sensitivity"))
import regressors as R  # noqa: E402
import shared  # noqa: E402

ENGINE = None


def _load_engine():
    spec = importlib.util.spec_from_file_location("endpoints_engine", ENGINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seasonal_baseline(dates: np.ndarray, clock_fracs: np.ndarray):
    """12-column engine baseline plus tau^2, tau^3, screened by the same QR rule."""
    dates = np.asarray(dates, dtype=np.int64)
    n = len(dates)
    cols = [np.ones(n, dtype=np.float64), (dates - dates[0]).astype(np.float64)]
    dow = (dates + 3) % 7
    for d in range(1, 7):
        cols.append((dow == d).astype(np.float64))
    for c in range(4):
        cols.append(np.asarray(clock_fracs, dtype=np.float64)[:, c])
    span = max(float(dates[-1] - dates[0]), 1.0)
    tau = 2.0 * (dates.astype(np.float64) - dates[0]) / span - 1.0
    cols.extend([tau ** 2, tau ** 3])
    Q, p_retained = R.screen_qr(np.column_stack(cols))
    P_0 = np.eye(n, dtype=np.float64) - Q @ Q.T
    return P_0, Q, p_retained


def _init() -> None:
    global ENGINE
    ENGINE = _load_engine()
    ENGINE.build_and_screen_nuisance_baseline = seasonal_baseline


def _run(task):
    out = ENGINE.process_participant_task(task)
    for r in out:
        r.pop("spectra", None)
        r.pop("null_diagnostics", None)
    return out


def _tasks(protocol: str):
    return shared.build_tasks("HRp10", protocol)


def compute(args) -> None:
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    for protocol in args.protocols:
        tasks = _tasks(protocol)
        if args.limit:
            tasks = tasks[:args.limit]
        print(f"{protocol}: {len(tasks)} participant tasks", flush=True)
        rows, t0, last = [], time.time(), 0
        with Pool(args.workers, initializer=_init) as pool:
            for k, recs in enumerate(pool.imap_unordered(_run, tasks, chunksize=16), 1):
                rows.extend(recs)
                if k % 1000 == 0 or k == len(tasks):
                    pd.DataFrame(rows).to_parquet(
                        run / f"seasonal_records_{protocol}.parquet", index=False)
                    print(f"  {k}/{len(tasks)} tasks, {len(rows)} windows, "
                          f"{time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame(rows)
        df.to_parquet(run / f"seasonal_records_{protocol}.parquet", index=False)
        print(f"{protocol}: {len(df)} rows, status "
              f"{df.status.value_counts().to_dict()}", flush=True)


def _label(sex: int, age: float) -> str:
    band = "20-35" if age <= 35 else ("36-50" if age <= 50 else "51-75")
    return ("F" if sex == 1 else "M") + band


def design_recovery(protocol: str, periods=(28.0, 34.0), n_windows: int = 200) -> pd.DataFrame:
    """Fraction of a unit cosinor surviving projection on baseline+S vs baseline.

    Coverage columns are omitted: they are near-constant within a window and
    near-collinear with the intercept, so they add nothing to the own-period
    projection.  Weekday dummies are included.
    """
    _, windows, selected = shared.load_manifests()
    sel = selected.loc[selected.protocol.eq(protocol)]
    groups = list(sel.groupby(["user_id", "source", "window_index"]))
    groups = groups[::max(1, len(groups) // n_windows)][:n_windows]
    out = {"protocol": protocol}
    for period in periods:
        rec = []
        for (uid, src, widx), g in groups:
            dates = np.sort(g.date.to_numpy(dtype=np.int64))
            n = len(dates)
            dow = (dates + 3) % 7
            base = [np.ones(n), (dates - dates[0]).astype(float)]
            base += [(dow == d).astype(float) for d in range(1, 7)]
            span = max(float(dates[-1] - dates[0]), 1.0)
            tau = 2.0 * (dates - dates[0]) / span - 1.0
            c = R.cosinor_columns(dates, period)
            Q0, _ = R.screen_qr(np.column_stack(base))
            Qs, _ = R.screen_qr(np.column_stack(base + [tau ** 2, tau ** 3]))
            r0 = c - Q0 @ (Q0.T @ c)
            rs = c - Qs @ (Qs.T @ c)
            rec.append(float(np.sum(rs ** 2) / np.sum(r0 ** 2)))
        out[f"recovery_{period:g}d"] = np.array(rec)
    return out


def analyze(args) -> None:
    run = Path(args.run)
    keys = ["user_id", "source", "protocol", "window_index", "selected_dates_hash"]
    ref = pd.read_parquet(ENDPOINTS, columns=keys + ["Q_i", "p_value_discrete"])
    ref = ref.rename(columns={"Q_i": "Q_ref", "p_value_discrete": "p_ref"})
    if ref.duplicated(keys).any():
        raise SystemExit("duplicate keys in reference endpoints")

    frames = []
    for protocol in args.protocols:
        f = run / f"seasonal_records_{protocol}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    s = pd.concat(frames, ignore_index=True)
    s = s.loc[s.status.eq("ok")].rename(
        columns={"Q_i": "Q_s", "p_value_discrete": "p_s"})
    m = s.merge(ref, on=keys, how="inner", validate="one_to_one")
    m = m.loc[m.age.between(20, 75)].copy()
    m["stratum"] = [_label(int(x), float(a)) for x, a in zip(m.sex, m.age)]
    m = m.sort_values("window_index").groupby(
        ["user_id", "source", "protocol"], as_index=False).head(1)
    print(f"paired windows: {len(m)}  (S ok: {len(s)}, ref rows: {len(ref)})")

    rng = np.random.default_rng(20261020)
    rows = []
    for protocol in args.protocols:
        sub = m.loc[m.protocol.eq(protocol)]
        for stratum, g in sub.groupby("stratum"):
            if len(g) < 30:
                continue
            qs = g.Q_s.to_numpy(float)
            qr = g.Q_ref.to_numpy(float)
            idx = rng.integers(0, len(g), (2000, len(g)))
            ratio = (qs[idx].mean(1) / qr[idx].mean(1))
            rows.append({
                "protocol": protocol, "stratum": stratum, "n": len(g),
                "Q_ref": float(qr.mean()), "Q_s": float(qs.mean()),
                "ratio": float(qs.mean() / qr.mean()),
                "ratio_lo": float(np.quantile(ratio, 0.025)),
                "ratio_hi": float(np.quantile(ratio, 0.975)),
                "absorbed_share": float(1.0 - qs.mean() / qr.mean()),
                "fpr_ref": float((g.p_ref <= 0.05).mean()),
                "fpr_s": float((g.p_s <= 0.05).mean()),
            })
    res = pd.DataFrame(rows)
    res.to_csv(run / "seasonal_arm.csv", index=False)
    print(res.round(4).to_string(index=False))

    diag = []
    for protocol in args.protocols:
        if (run / f"seasonal_records_{protocol}.parquet").exists():
            d = design_recovery(protocol)
            for period in (28.0, 34.0):
                v = d[f"recovery_{period:g}d"]
                diag.append({"protocol": protocol, "period_d": period,
                             "recovery_mean": float(v.mean()),
                             "recovery_min": float(v.min()),
                             "recovery_max": float(v.max()), "n": len(v)})
    dd = pd.DataFrame(diag)
    dd.to_csv(run / "seasonal_design_recovery.csv", index=False)
    print()
    print("design recovery of a unit sinusoid (baseline+S vs baseline):")
    print(dd.round(4).to_string(index=False))


def benchmark(args) -> None:
    tasks = _tasks(args.protocols[0])[:args.limit]
    t0 = time.time()
    with Pool(args.workers, initializer=_init) as pool:
        n = sum(len(r) for r in pool.imap_unordered(_run, tasks, chunksize=4))
    dt = time.time() - t0
    print(f"{n} windows in {dt:.1f}s on {args.workers} workers "
          f"({dt / max(n, 1):.3f} s/window single-core-equivalent)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["compute", "analyze", "benchmark"])
    ap.add_argument("--protocols", nargs="+", default=["balanced90", "expanded90"])
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    {"compute": compute, "analyze": analyze, "benchmark": benchmark}[args.mode](args)


if __name__ == "__main__":
    main()
