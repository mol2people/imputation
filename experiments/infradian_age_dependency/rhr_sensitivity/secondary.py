"""Matched paired-window summaries for RHR sensitivity arms."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(EXP / "scripts") not in sys.path:
    sys.path.insert(0, str(EXP / "scripts"))
import shared

AGE_EDGES = (20, 35, 42, 47, 52, 60, 75)
AGE_LABELS = ("20-35", "35-42", "42-47", "47-52", "52-60", "60-75")
AGE_BRACKETS = ("<40", "40-55", ">=55")
CORR_ENDPOINTS = ("Q_i", "E_i2", "A_hat", "sigma_resid", "peak_period")
SPECTRAL_PERIODS = (20.0, 26.67, 45.0)


def _seed(*parts: object) -> int:
    text = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def _groups(age: np.ndarray, sex: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    labels = pd.cut(age, AGE_EDGES, labels=AGE_LABELS, right=False, include_lowest=True)
    out = [("all", "20-75", (age >= 20) & (age < 75))]
    for label in AGE_LABELS:
        out.append(("age_bin", label, np.asarray(labels == label)))
    for label, mask in (("Female", sex == 1), ("Male", sex == 0)):
        out.append(("sex", label, mask))
    return out


def _report_groups(age: np.ndarray, sex: np.ndarray) -> list[tuple[str, str, str, np.ndarray]]:
    bracket = np.where(age < 40, "<40", np.where(age < 55, "40-55", ">=55"))
    out = []
    for b in ("All", *AGE_BRACKETS):
        for s in ("All", "Female", "Male"):
            mask = np.ones(len(age), dtype=bool)
            if b != "All":
                mask &= bracket == b
            if s == "Female":
                mask &= sex == 1
            elif s == "Male":
                mask &= sex == 0
            out.append(("age_bracket", b, s, mask))
    return out


def _draws(age: np.ndarray, sex: np.ndarray, n_boot: int, seed: int) -> Iterable[np.ndarray]:
    """Bootstrap within 5-year age-by-sex cells; one draw is reused by all arms."""
    rng = np.random.Generator(np.random.PCG64(seed))
    cell = np.floor(age / 5).astype(int) * 2 + sex.astype(int)
    cells = [np.flatnonzero(cell == c) for c in np.unique(cell)]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in cells])


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _nuisance_basis(dates: np.ndarray, clock: np.ndarray) -> np.ndarray:
    """Sequential QR screen matching the endpoint engine's nuisance design."""
    n = len(dates)
    cols = [np.ones(n), (dates - dates[0]).astype(float)]
    dow = (dates + 3) % 7
    cols.extend((dow == d).astype(float) for d in range(1, 7))
    cols.extend(clock[:, c].astype(float) for c in range(4))
    x = np.column_stack(cols)
    q = np.zeros((n, 0), dtype=float)
    for j in range(x.shape[1]):
        v = x[:, j]
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            continue
        perp = v - q @ (q.T @ v) if q.shape[1] else v
        if np.linalg.norm(perp) / norm <= 1e-4:
            continue
        q = np.column_stack((q, perp / np.linalg.norm(perp)))
    return q


def _load_cache(source: int, user_id: int, cache_memo: dict) -> tuple[np.ndarray, np.ndarray] | None:
    key = (int(source), int(user_id))
    if key not in cache_memo:
        path = shared.CACHE_PATHS.get(int(source))
        file = path / f"customer_{int(user_id)}.npz" if path else None
        if file is None or not file.is_file():
            cache_memo[key] = None
        else:
            with np.load(file) as d:
                cache_memo[key] = (d["date"].astype(np.int64), d["hr"].astype(float))
    return cache_memo[key]


def _forecast_one(dates1: np.ndarray, y1: np.ndarray, dates2: np.ndarray, y2: np.ndarray,
                  cache_dates: np.ndarray, hr: np.ndarray, period: float) -> float:
    if len(dates1) != 90 or len(dates2) != 90 or not np.isfinite(y1).all() or not np.isfinite(y2).all():
        return np.nan
    date_rows = {int(d): i for i, d in enumerate(cache_dates)}
    if any(int(d) not in date_rows for d in np.r_[dates1, dates2]):
        return np.nan
    mask1 = np.isfinite(hr[[date_rows[int(d)] for d in dates1]])
    mask2 = np.isfinite(hr[[date_rows[int(d)] for d in dates2]])
    clock1 = np.column_stack([mask1[:, lo:lo + 72].mean(axis=1) for lo in (0, 72, 144, 216)])
    clock2 = np.column_stack([mask2[:, lo:lo + 72].mean(axis=1) for lo in (0, 72, 144, 216)])
    q1 = _nuisance_basis(dates1, clock1)
    q2 = _nuisance_basis(dates2, clock2)
    e1 = y1 - q1 @ (q1.T @ y1)
    e2 = y2 - q2 @ (q2.T @ y2)
    omega = 2 * np.pi / float(period)
    c1 = np.column_stack([np.cos(omega * (dates1 - dates1[0])), np.sin(omega * (dates1 - dates1[0]))])
    c1r = c1 - q1 @ (q1.T @ c1)
    try:
        beta = np.linalg.solve(c1r.T @ c1r, c1r.T @ e1)
    except np.linalg.LinAlgError:
        return np.nan
    h2 = beta[0] * np.cos(omega * (dates2 - dates1[0])) + beta[1] * np.sin(omega * (dates2 - dates1[0]))
    h2r = h2 - q2 @ (q2.T @ h2)
    return _corr(e2, h2r)


def _daily_tables(run_dir: Path, arms: tuple[str, ...], source: int, protocol: str,
                  users: list[int]) -> dict[str, pd.DataFrame]:
    out = {}
    for arm in arms:
        path = run_dir / arm / protocol / "daily_values.parquet"
        if path.is_file():
            df = pd.read_parquet(path, columns=["user_id", "source", "arm", "protocol", "window_index",
                                                "selected_dates_hash", "date", "value"])
            out[arm] = df.loc[df.source.eq(source) & df.arm.eq(arm) & df.user_id.isin(users)].copy()
        else:
            out[arm] = pd.DataFrame()
    return out


def _spectrum_tables(run_dir: Path, source: int, protocol: str, arms: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    out = {}
    for arm in arms:
        if arm == "HRp10":
            path = EXP / "artifacts/spectra/spectra_summary.parquet"
        else:
            path = run_dir / arm / protocol / "spectra/spectra_summary.parquet"
        if not path.is_file():
            out[arm] = pd.DataFrame()
            continue
        frame = pd.read_parquet(path, filters=[("source", "==", source), ("protocol", "==", protocol)])
        frame = frame.loc[(frame.source == source) & frame.protocol.eq(protocol)].copy()
        if "arm" not in frame:
            frame["arm"] = arm
        if "broad_freqs" not in frame:
            npz = (EXP / "artifacts/spectra/spectra_broad.npz" if arm == "HRp10"
                   else run_dir / arm / protocol / "spectra/spectra_broad.npz")
            if npz.is_file():
                with np.load(npz) as d:
                    freqs = d["broad_freqs"].astype(float)
                frame["broad_freqs"] = [freqs] * len(frame)
        out[arm] = frame
    return out


def _common_two_window(g: pd.DataFrame, arms: tuple[str, ...]) -> tuple[list[int], dict[str, pd.DataFrame]]:
    keys = ["user_id", "source", "arm", "protocol", "window_index"]
    if g.duplicated(keys).any():
        raise ValueError("Duplicate endpoint rows for participant/arm/window")
    by = {}
    for metric in ("Q_i", "E_i2", "A_hat", "sigma_resid", "peak_period", "null_status", "selected_dates_hash", "age", "sex", "start_day"):
        if metric in g:
            by[metric] = g.pivot(index=["user_id", "window_index"], columns="arm", values=metric).reindex(columns=list(arms))
    q = by["Q_i"]
    ok = np.isfinite(q.to_numpy(float)).all(axis=1)
    status = by["null_status"]
    ok &= status.eq("converged").all(axis=1).to_numpy()
    hashes = by["selected_dates_hash"]
    ok &= hashes.notna().all(axis=1).to_numpy() & hashes.nunique(axis=1, dropna=True).eq(1).to_numpy()
    idx = q.index[ok]
    common = sorted(set(idx.get_level_values("user_id")[idx.get_level_values("window_index") == 1]) &
                    set(idx.get_level_values("user_id")[idx.get_level_values("window_index") == 2]))
    if not common:
        return [], by
    # Metadata must also agree across arms before using the paired rows.
    for metric in ("age", "sex", "start_day"):
        wide = by[metric]
        vals = wide.reindex(q.index).to_numpy()
        same = np.all((vals == vals[:, [0]]) | (pd.isna(vals) & pd.isna(vals[:, [0]])), axis=1)
        if not same.all():
            raise ValueError(f"{metric} differs across matched arms")
    return common, by


def run_secondary(endpoints: pd.DataFrame, run_dir: str | Path, out: str | Path,
                  n_boot: int = 2000, arms: tuple[str, ...] = ("HRp10", "HRp5", "HRp1")) -> dict[str, pd.DataFrame]:
    """Write paired longitudinal, correlation, spectrum, and phase summaries.

    All comparisons use participant/source/protocol groups with both windows,
    convergence in every arm, and common date hashes. HRp10 values for the
    phase forecast are extracted from its existing channel-3000 cache; no null
    or endpoint model is replayed.
    """
    run_dir, out = Path(run_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    need = {"user_id", "source", "arm", "protocol", "window_index", "start_day", "age", "sex", "selected_dates_hash", "null_status", *CORR_ENDPOINTS, "Q_i"}
    missing = need - set(endpoints.columns)
    if missing:
        raise ValueError(f"endpoint table lacks secondary fields: {sorted(missing)}")
    frame = endpoints.loc[endpoints.arm.isin(arms) & endpoints.window_index.isin([1, 2])].copy()
    if not set(arms).issubset(set(frame.arm)):
        raise ValueError("all requested arms must be present")
    long_rows, corr_rows, spec_rows, spec_diff_rows, forecast_rows = [], [], [], [], []
    cache_memo: dict = {}

    for (source, protocol), g in frame.groupby(["source", "protocol"], sort=True):
        common, wide = _common_two_window(g, arms)
        if not common:
            continue
        g = g[g.user_id.isin(common)].copy()
        daily = _daily_tables(run_dir, arms, int(source), str(protocol), common)
        g.sort_values(["user_id", "window_index", "arm"], inplace=True)
        uid = np.asarray(common, dtype=np.int64)
        base = g[g.arm.eq(arms[0])].drop_duplicates(["user_id", "window_index"]).set_index(["user_id", "window_index"])
        age = base.loc[pd.MultiIndex.from_product([uid, [1]], names=["user_id", "window_index"]), "age"].to_numpy(float)
        sex = base.loc[pd.MultiIndex.from_product([uid, [1]], names=["user_id", "window_index"]), "sex"].to_numpy(int)
        w1 = base.loc[pd.MultiIndex.from_product([uid, [1]], names=["user_id", "window_index"])]
        w2 = base.loc[pd.MultiIndex.from_product([uid, [2]], names=["user_id", "window_index"])]
        gap = (w2.start_day.to_numpy(float) - w1.start_day.to_numpy(float)) / 365.25
        if np.any(gap <= 0):
            raise ValueError("paired windows must have a positive calendar gap")
        q_change: dict[str, np.ndarray] = {}
        q_rate: dict[str, np.ndarray] = {}
        metric_mats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for arm in arms:
            ag = g[g.arm.eq(arm)].set_index(["user_id", "window_index"])
            a = ag.loc[pd.MultiIndex.from_product([uid, [1]], names=["user_id", "window_index"])]
            b = ag.loc[pd.MultiIndex.from_product([uid, [2]], names=["user_id", "window_index"])]
            q_change[arm] = b.Q_i.to_numpy(float) - a.Q_i.to_numpy(float)
            q_rate[arm] = q_change[arm] / gap
            for metric in CORR_ENDPOINTS:
                if metric not in metric_mats:
                    metric_mats[metric] = ({}, {})
                metric_mats[metric][0][arm] = a[metric].to_numpy(float)
                metric_mats[metric][1][arm] = b[metric].to_numpy(float)

        # Annualized Q change, age bins from first-window age, by sex and pooled.
        for kind, label, mask in _groups(age, sex):
            ix = np.flatnonzero(mask & np.isfinite(age))
            if not len(ix):
                continue
            group_age, group_sex = age[ix], sex[ix]
            seed = _seed("long", source, protocol, kind, label)
            boot = {a: {m: [] for m in ("Q_change", "Q_annualized_rate")} for a in arms}
            for draw in _draws(group_age, group_sex, n_boot, seed):
                for arm in arms:
                    boot[arm]["Q_change"].append(float(q_change[arm][ix][draw].mean()))
                    boot[arm]["Q_annualized_rate"].append(float(q_rate[arm][ix][draw].mean()))
            for metric, values in (("Q_change", q_change), ("Q_annualized_rate", q_rate)):
                for arm in arms:
                    point = float(values[arm][ix].mean())
                    ci = np.percentile(boot[arm][metric], [2.5, 97.5])
                    long_rows.append(dict(source=source, protocol=protocol, subgroup=kind, age_group=label,
                                          sex="All" if kind in ("all", "age_bin") else label,
                                          arm=arm, contrast="within_arm", measure=metric, n=len(ix),
                                          estimate=point, ci_low=float(ci[0]), ci_high=float(ci[1]), n_boot=n_boot))
                for arm in arms[1:]:
                    dif = values[arm][ix] - values[arms[0]][ix]
                    bdiff = np.asarray(boot[arm][metric]) - np.asarray(boot[arms[0]][metric])
                    ci = np.percentile(bdiff, [2.5, 97.5])
                    long_rows.append(dict(source=source, protocol=protocol, subgroup=kind, age_group=label,
                                          sex="All" if kind in ("all", "age_bin") else label,
                                          arm=arm, reference=arms[0], contrast="paired_arm_minus_reference",
                                          measure=metric, n=len(ix), estimate=float(dif.mean()),
                                          ci_low=float(ci[0]), ci_high=float(ci[1]), n_boot=n_boot))

        # Window-to-window endpoint associations in report age brackets/sex strata.
        for subgroup, age_group, sex_group, mask in _report_groups(age, sex):
            for metric, (m1, m2) in metric_mats.items():
                all_finite = np.ones(len(uid), dtype=bool)
                for arm in arms:
                    all_finite &= np.isfinite(m1[arm]) & np.isfinite(m2[arm])
                ix = np.flatnonzero(mask & all_finite)
                if not len(ix):
                    continue
                boots = {arm: [] for arm in arms}
                for draw in _draws(age[ix], sex[ix], n_boot,
                                   _seed("corr", source, protocol, metric, age_group, sex_group)):
                    for arm in arms:
                        boots[arm].append(_corr(m1[arm][ix][draw], m2[arm][ix][draw]))
                points = {arm: _corr(m1[arm][ix], m2[arm][ix]) for arm in arms}
                for arm in arms:
                    valid = np.asarray(boots[arm], float); valid = valid[np.isfinite(valid)]
                    ci = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
                    corr_rows.append(dict(source=source, protocol=protocol, endpoint=metric,
                                          age_bracket=age_group, sex=sex_group, arm=arm,
                                          contrast="within_arm", n=len(ix), correlation=points[arm],
                                          ci_low=ci[0], ci_high=ci[1], n_boot_valid=len(valid), n_boot=n_boot))
                for arm in arms[1:]:
                    diffs = np.asarray(boots[arm], float) - np.asarray(boots[arms[0]], float)
                    diffs = diffs[np.isfinite(diffs)]
                    ci = np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (np.nan, np.nan)
                    corr_rows.append(dict(source=source, protocol=protocol, endpoint=metric,
                                          age_bracket=age_group, sex=sex_group, arm=arm,
                                          reference=arms[0], contrast="paired_arm_minus_reference", n=len(ix),
                                          correlation=points[arm] - points[arms[0]], ci_low=ci[0], ci_high=ci[1],
                                          n_boot_valid=len(diffs), n_boot=n_boot))

        # Paired spectra: same female 42 <= age_w1 < 48 participants in both windows/all arms.
        spec_by_arm = _spectrum_tables(run_dir, int(source), str(protocol), arms)
        keys = {(int(u), w) for u in common for w in (1, 2)}
        cohort = [u for u, a, s in zip(uid, age, sex) if s == 1 and 42 <= a < 48]
        spec_values: dict[str, dict[tuple[int, int, float], float]] = {a: {} for a in arms}
        spec_freq: dict[str, np.ndarray] = {}
        spec_actual: dict[tuple[str, float], float] = {}
        for arm in arms:
            sf = spec_by_arm[arm]
            if sf.empty:
                continue
            if sf.duplicated(["user_id", "source", "protocol", "window_index"]).any():
                raise ValueError(f"duplicate spectra rows for {arm}/{source}/{protocol}")
            for row in sf.itertuples(index=False):
                key = (int(row.user_id), int(row.window_index))
                if key not in keys:
                    continue
                if hasattr(row, "broad_freqs"):
                    freqs = np.asarray(row.broad_freqs, dtype=float)
                else:
                    continue
                spec_freq[arm] = freqs
                expected_hash = g.loc[(g.user_id == key[0]) & (g.window_index == key[1]) & (g.arm == arm), "selected_dates_hash"].iloc[0]
                if str(row.selected_dates_hash) != str(expected_hash):
                    raise ValueError(f"spectra date hash mismatch for {arm}/{source}/{protocol}/{key}")
                obs, null = np.asarray(row.P_obs, float), np.asarray(row.P_null_mean, float)
                periods = 1.0 / freqs
                exc = np.maximum(0.0, obs - null)
                for target in SPECTRAL_PERIODS:
                    j = int(np.argmin(np.abs(periods - target)))
                    spec_values[arm][(key[0], key[1], target)] = float(exc[j])
                    spec_actual[(arm, target)] = float(periods[j])
        spec_ids = [u for u in cohort if all((int(u), w, target) in spec_values[a]
                     for a in arms for w in (1, 2) for target in SPECTRAL_PERIODS)]
        if spec_ids:
            idx_global = np.asarray([np.flatnonzero(uid == u)[0] for u in spec_ids], dtype=int)
            ages_spec, sex_spec = age[idx_global], sex[idx_global]
            for target in SPECTRAL_PERIODS:
                changes = {arm: np.asarray([spec_values[arm][(int(u), 2, target)] - spec_values[arm][(int(u), 1, target)] for u in spec_ids]) for arm in arms}
                bmeans = {arm: [] for arm in arms}
                for draw in _draws(ages_spec, sex_spec, n_boot, _seed("spec", source, protocol, target)):
                    for arm in arms:
                        bmeans[arm].append(float(changes[arm][draw].mean()))
                for arm in arms:
                    x1 = np.asarray([spec_values[arm][(int(u), 1, target)] for u in spec_ids])
                    x2 = np.asarray([spec_values[arm][(int(u), 2, target)] for u in spec_ids])
                    spec_rows.append(dict(source=source, protocol=protocol, arm=arm, period_target_days=target,
                                          period_days=spec_actual[(arm, target)],
                                          n=len(spec_ids), mean_excess_w1=float(x1.mean()), mean_excess_w2=float(x2.mean()),
                                          mean_change=float(changes[arm].mean()),
                                          change_ci_low=float(np.percentile(bmeans[arm],2.5)),
                                          change_ci_high=float(np.percentile(bmeans[arm],97.5)), n_boot=n_boot))
                for arm in arms[1:]:
                    dif = changes[arm] - changes[arms[0]]
                    bd = np.asarray(bmeans[arm]) - np.asarray(bmeans[arms[0]])
                    ci = np.percentile(bd, [2.5, 97.5])
                    spec_diff_rows.append(dict(source=source, protocol=protocol, arm=arm, reference=arms[0],
                                               period_target_days=target, n=len(spec_ids), mean_change_difference=float(dif.mean()),
                                               ci_low=float(ci[0]), ci_high=float(ci[1]), n_boot=n_boot))

        # Frozen first-window harmonic forecast, retaining absolute calendar phase.
        endpoint_lookup = {(int(r.user_id), int(r.window_index), str(r.arm)): r
                           for r in g.itertuples(index=False)}
        daily_lookup: dict[str, dict[tuple[int, int], tuple[np.ndarray, np.ndarray, str]]] = {a: {} for a in arms}
        for arm in arms:
            df = daily[arm]
            if df.empty:
                continue
            df = df.loc[(df.source == source) & df.protocol.eq(protocol)]
            for (u, w), x in df.groupby(["user_id", "window_index"], sort=False):
                x = x.sort_values("date")
                hashes = x.selected_dates_hash.dropna().astype(str).unique()
                if len(hashes) != 1:
                    raise ValueError(f"daily date-hash mismatch for {arm}/{source}/{protocol}/{u}/{w}")
                daily_lookup[arm][(int(u), int(w))] = (x.date.to_numpy(np.int64), x.value.to_numpy(float), hashes[0])
        for u, ag, sx in zip(uid, age, sex):
            for arm in arms:
                r1 = endpoint_lookup[(int(u), 1, arm)]; r2 = endpoint_lookup[(int(u), 2, arm)]
                cache = _load_cache(int(source), int(u), cache_memo)
                reason, r = "ok", np.nan
                if cache is None:
                    reason = "channel3000_cache_missing"
                else:
                    cache_dates, hr = cache
                    ys = []
                    ds = []
                    try:
                        for w in (1, 2):
                            if arm == "HRp10" and (int(u), w) not in daily_lookup[arm]:
                                # Use the already matched selected dates from any derived arm.
                                ref_arm = next((a for a in arms if (int(u), w) in daily_lookup[a]), None)
                                if ref_arm is None:
                                    raise KeyError("selected_dates_unavailable")
                                dates, _, daily_hash = daily_lookup[ref_arm][(int(u), w)]
                                rowmap = {int(d): i for i, d in enumerate(cache_dates)}
                                vals = np.asarray([hr[rowmap[int(d)]] for d in dates])
                                y = np.asarray([np.percentile(row[np.isfinite(row)], 10) for row in vals])
                            else:
                                dates, y, daily_hash = daily_lookup[arm][(int(u), w)]
                            if len(dates) != 90 or daily_hash != str((r1 if w == 1 else r2).selected_dates_hash):
                                raise ValueError("daily_selected_date_mismatch")
                            ds.append(dates); ys.append(y)
                        if len(ds[0]) != 90 or len(ds[1]) != 90 or not np.isfinite(ys[0]).all() or not np.isfinite(ys[1]).all():
                            raise ValueError("incomplete_daily_series")
                        t = float(r1.peak_period)
                        if not np.isfinite(t) or t <= 0:
                            raise ValueError("missing_peak_period")
                        r = _forecast_one(ds[0], ys[0], ds[1], ys[1], cache_dates, hr, t)
                        if not np.isfinite(r):
                            reason = "forecast_not_identifiable"
                    except (KeyError, ValueError, IndexError):
                        reason = "daily_series_or_dates_unavailable"
                forecast_rows.append(dict(user_id=int(u), source=source, protocol=protocol, arm=arm,
                                          age=float(ag), sex=int(sx), age_bracket="<40" if ag < 40 else ("40-55" if ag < 55 else ">=55"),
                                          forecast_r=r, status=reason))
            cache_memo.pop((int(source), int(u)), None)

    tables = {
        "longitudinal": pd.DataFrame(long_rows),
        "window_correlations": pd.DataFrame(corr_rows),
        "spectral_excess": pd.DataFrame(spec_rows),
        "spectral_contrasts": pd.DataFrame(spec_diff_rows),
        "forecast_participants": pd.DataFrame(forecast_rows),
    }
    # Participant forecast summaries use identical finite participants and draws across arms.
    fs = tables["forecast_participants"]
    summary_rows = []
    if not fs.empty:
        for (source, protocol, age_bracket, sex_group), block in fs.groupby(
                ["source", "protocol", "age_bracket", "sex"], sort=True):
            pivot = block.pivot(index="user_id", columns="arm", values="forecast_r").reindex(columns=list(arms))
            ok = np.isfinite(pivot.to_numpy(float)).all(axis=1)
            pivot = pivot.loc[ok]
            ids = pivot.index.to_numpy()
            if len(ids) < 2:
                continue
            meta = block.drop_duplicates("user_id").set_index("user_id").loc[ids]
            bvals = {a: [] for a in arms}
            for draw in _draws(meta.age.to_numpy(float), meta.sex.to_numpy(int), n_boot,
                               _seed("forecast", source, protocol, age_bracket, sex_group)):
                for arm in arms:
                    bvals[arm].append(float(pivot[arm].to_numpy(float)[draw].mean()))
            for arm in arms:
                ci = np.percentile(bvals[arm], [2.5, 97.5])
                summary_rows.append(dict(source=source, protocol=protocol, age_bracket=age_bracket,
                                         sex="Female" if sex_group == 1 else "Male", arm=arm,
                                         contrast="within_arm", n=len(ids), mean_forecast_r=float(pivot[arm].mean()),
                                         ci_low=float(ci[0]), ci_high=float(ci[1]), n_boot=n_boot))
            for arm in arms[1:]:
                diff = pivot[arm].to_numpy(float) - pivot[arms[0]].to_numpy(float)
                bd = np.asarray(bvals[arm]) - np.asarray(bvals[arms[0]])
                ci = np.percentile(bd, [2.5, 97.5])
                summary_rows.append(dict(source=source, protocol=protocol, age_bracket=age_bracket,
                                         sex="Female" if sex_group == 1 else "Male", arm=arm, reference=arms[0],
                                         contrast="paired_arm_minus_reference", n=len(ids), mean_forecast_r=float(diff.mean()),
                                         ci_low=float(ci[0]), ci_high=float(ci[1]), n_boot=n_boot))
    tables["forecast"] = pd.DataFrame(summary_rows)
    for name, table in tables.items():
        table.to_parquet(out / f"secondary_{name}.parquet", index=False)
    return tables
