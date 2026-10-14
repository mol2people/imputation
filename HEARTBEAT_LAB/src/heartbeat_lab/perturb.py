"""Within-customer perturbations of real events, with fixed fitted models.

Randomization is within selected local dates; every other event is unchanged.
These are conditional sensitivity experiments, not population uncertainty or
a null test of model validity. No labels enter generation.
"""
import numpy as np
import pandas as pd
from .data import aggregate_days

MODES = {
    "shuffle": "Shuffle HR within each selected day",
    "bin_shuffle": "Shuffle observed five-minute means (historical control)",
    "resample": "Resample each day's empirical HR distribution",
    "thin": "Randomly remove observations",
    "shift": "Circularly shift clock time within each day",
}


def shuffle_bins(hr, seed, fraction=1.0):
    """Preserve each day's observed-bin multiset and missing-bin locations."""
    rng = np.random.default_rng(seed)
    result = hr.copy()
    for row in result:
        observed = np.flatnonzero(np.isfinite(row))
        positions = rng.choice(observed, int(round(len(observed)*fraction)), replace=False)
        row[positions] = rng.permutation(row[positions])
    return result


def perturb_events(events, selected_dates, mode, seed, fraction=1.0, shift_hours=6):
    """Shuffle fraction of values; thin fraction of events; or bounded shifts.

    Shifts are independently sampled per day in integer 5-minute bins from
    [-shift_hours,+shift_hours], wrapped inside the same date. Intervals retain
    their durations and recorded offsets. They are synthetic clock changes,
    not timezone conversion. Resampling draws n values with replacement.
    """
    if mode not in MODES or mode == "bin_shuffle" or not 0 <= fraction <= 1 or not 0 <= shift_hours <= 12:
        raise ValueError("Invalid perturbation settings.")
    rng = np.random.default_rng(seed)
    selected_dates = pd.to_datetime(list(selected_dates))
    changed = events.loc[events.date.isin(selected_dates)].copy()
    unchanged = events.loc[~events.date.isin(selected_dates)].copy()
    pieces = [unchanged]
    for _, day in changed.groupby("date", sort=True):
        day = day.copy()
        values = day.longValue.to_numpy(copy=True)
        if mode == "shuffle":
            positions = rng.choice(len(day), int(round(len(day)*fraction)), replace=False)
            values[positions] = rng.permutation(values[positions])
            day["longValue"] = values
        elif mode == "resample":
            day["longValue"] = rng.choice(values, len(values), replace=True)
        elif mode == "thin":
            day = day.loc[rng.random(len(day)) >= fraction].copy()
        else:
            offset = int(rng.integers(-round(shift_hours*12), round(shift_hours*12)+1))*300_000
            local = day.startTimestamp.to_numpy(np.int64) + (day.timezoneOffset.to_numpy()*60_000).astype(np.int64)
            clock = local % 86_400_000
            delta = (clock + offset) % 86_400_000 - clock
            day["startTimestamp"] += delta
            day["endTimestamp"] += delta
            day["bin"] = ((clock + offset) % 86_400_000) // 300_000
            day["hour"] = day.bin // 12
        pieces.append(day)
    return pd.concat(pieces).sort_index()


def repeated_predictions(events, days, hr, models, selected_dates, mode,
                         repeats=30, seed=42, fraction=1.0, shift_hours=6, progress=None):
    """Return baseline and replicate timelines on the ORIGINAL adequate-date grid.

    Days failing adequacy after thinning have NaN predictions, never imputed
    scores. The identical perturbed events are used for every selected model.
    Unmodified daily transforms are reused; modified days are transformed anew.
    """
    if not 1 <= repeats <= 200:
        raise ValueError("Use 1–200 repetitions.")
    selected = set(pd.to_datetime(selected_dates))
    if not selected <= set(days.date):
        raise ValueError("Select dates from the original adequate-day table.")
    base_features = {name: model.transform_days(hr) for name, model in models.items()}
    baseline = []
    for name, model in models.items():
        baseline.append(model.timeline(days, hr, transformed=base_features[name]).assign(model=name))
    base = pd.concat(baseline, ignore_index=True)
    subset = events.loc[events.date.isin(selected)]
    fixed = days.loc[~days.date.isin(selected)].copy()
    fixed_indices = np.flatnonzero(~days.date.isin(selected))
    date_positions = pd.Series(np.arange(len(days)), index=days.date)
    seeds = np.random.SeedSequence(seed).generate_state(repeats)
    results = []
    for run, run_seed in enumerate(seeds):
        if mode == "bin_shuffle":
            changed_days = days.loc[days.date.isin(selected)].copy()
            changed_hr = shuffle_bins(hr[days.date.isin(selected)], int(run_seed), fraction)
        else:
            modified = perturb_events(subset, selected, mode, int(run_seed), fraction, shift_hours)
            changed_days, changed_hr = aggregate_days(modified)
        # An empty aggregate has object dtypes; don't let it erase the retained
        # days' datetime/numeric schema when all selected observations are removed.
        joined = fixed.copy() if changed_days.empty else pd.concat([fixed, changed_days], ignore_index=True)
        curves = np.concatenate([hr[fixed_indices], changed_hr])
        order = joined.date.argsort().to_numpy()
        joined = joined.iloc[order].reset_index(drop=True)
        curves = curves[order]
        for name, model in models.items():
            result = days[["date"]].copy()
            if not joined.empty:
                features = None
                if base_features[name] is not None:
                    features = base_features[name][date_positions.loc[joined.date].to_numpy()].copy()
                    changed_positions = np.flatnonzero(joined.date.isin(selected))
                    if len(changed_positions):
                        features[changed_positions] = model.transform_days(curves[changed_positions])
                timeline = model.timeline(joined, curves, transformed=features)
                result = result.merge(timeline, on="date", how="left", validate="one_to_one")
            else:
                for column in ("daily_score", "cumulative_score", "observed_bins", "adequate_days_so_far"):
                    result[column] = np.nan
            result["model"], result["run"], result["run_seed"] = name, run, int(run_seed)
            result["adequate"] = result.daily_score.notna()
            result["selected_day"] = result.date.isin(selected)
            results.append(result)
        if progress:
            progress((run+1)/repeats)
    return base, pd.concat(results, ignore_index=True)
