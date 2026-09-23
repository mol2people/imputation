#!/usr/bin/env python3
"""Run the standalone informative-recording feature experiment.

This file deliberately does not import project analysis modules.  It rebuilds only the
prespecified U/G feature representations from frozen aggregate v1 artifacts, then fits
the fixed forest over seeded repeated holdouts.  All output is local to this directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import t
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder


RUN_DIR = Path(__file__).resolve().parent
PROJECT_DIR = RUN_DIR.parent
INPUT_DIR = PROJECT_DIR / "artifacts"

STUDY_SEED = 20260920
N_REPEATS = 30
SMOKE_PER_STRATUM = 200
CHANNELS = (3000, 3001, 3002)
ADEQUATE_HOURS = 8
ADEQUATE_COV_SECONDS = 8 * 3600
ADEQUATE_N = 60

RF_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_features": 0.4,
    "min_samples_leaf": 10,
    "max_depth": None,
    "class_weight": "balanced_subsample",
    "n_jobs": 5,
}

VALUE_STATS = (
    "mean_of_daily_mean",
    "sd_of_daily_mean",
    "mean_of_daily_sd",
    "median_of_daily_median",
    "iqr_of_daily_median",
    "min_of_daily_min",
    "max_of_daily_max",
    "range_of_daily_extremes",
    "weekday_mean",
    "weekend_mean",
    "weekend_minus_weekday",
)
PROCESS_STATS = (
    "observed_days",
    "adequate_days",
    "events",
    "span_days",
    "duty_cycle",
    "events_per_observed_day",
    "mean_hours",
    "mean_cov_h",
    "longest_inadequate_run",
    "todfrac_night",
    "todfrac_morning",
    "todfrac_afternoon",
    "todfrac_evening",
    "n_hours_with_events",
    "present",
)
OVERALL_PROCESS_STATS = (
    "overall_events",
    "overall_unique_days",
    "overall_span_days",
    "any_channel_adequate_days",
    "any_channel_duty",
    "n_core_channels_present",
)


def log(message: str) -> None:
    print(message, flush=True)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def as_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=as_builtin) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_from(parts: list[int]) -> int:
    return int(np.random.SeedSequence(parts).generate_state(1, dtype=np.uint32)[0])


def ordered_unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def categorical_values(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in columns:
        values = frame[column].astype(object)
        result[column] = values.where(pd.notna(values), "<missing>").astype(str)
    return result


@dataclass(frozen=True)
class ArmSpec:
    name: str
    columns: tuple[str, ...]
    encoding: str = "native"


def load_cohorts() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(INPUT_DIR / "cohort_manifest.parquet")
    ensure(manifest["user_id"].is_unique, "cohort manifest has duplicate user IDs")

    g = manifest.loc[manifest["cohort_category"].eq("eligible")].copy()
    u = manifest.loc[
        manifest["cohort_category"].isin(["eligible", "below_coverage_gate"])
    ].copy()
    for name, frame, expected_n, expected_counts in (
        ("G", g, 20_485, {0: 6_788, 1: 13_697}),
        ("U", u, 25_784, {0: 9_662, 1: 16_122}),
    ):
        frame["y"] = frame["y"].astype(int)
        ensure(len(frame) == expected_n, f"{name} cohort size drift: {len(frame)} != {expected_n}")
        actual = frame["y"].value_counts().sort_index().to_dict()
        ensure(actual == expected_counts, f"{name} class counts drift: {actual} != {expected_counts}")
        ensure(frame["salutation"].isin([10, 20]).all(), f"{name} has a nonbinary salutation")
        ensure(frame["epoch_primary_source"].notna().all(), f"{name} has missing primary source")
        ensure(frame["epoch_multisource"].notna().all(), f"{name} has missing multisource flag")

    g_ids = set(g["user_id"])
    u_ids = set(u["user_id"])
    ensure(g_ids.issubset(u_ids), "G is not a subset of U")
    below = u.loc[u["cohort_category"].eq("below_coverage_gate"), "user_id"]
    ensure(not (set(below) & g_ids), "U cohort categories are not disjoint")
    return g.sort_values("user_id").reset_index(drop=True), u.sort_values("user_id").reset_index(drop=True)


def longest_inadequate_runs(days: pd.DataFrame, aggregate: pd.DataFrame) -> pd.Series:
    """Max calendar-date run without an adequate day inside observed span."""
    adequate = days.loc[days["adequate"], ["user", "channel", "date"]].sort_values(
        ["user", "channel", "date"]
    )
    if adequate.empty:
        return aggregate["span_days"].copy()

    bounds = adequate.groupby(["user", "channel"], sort=False)["date"].agg(
        first_adequate="min", last_adequate="max"
    )
    gaps = adequate.copy()
    gaps["gap"] = gaps.groupby(["user", "channel"], sort=False)["date"].diff().sub(1)
    interior = gaps.groupby(["user", "channel"], sort=False)["gap"].max().clip(lower=0)

    joined = aggregate.join(bounds, how="left").join(interior.rename("interior"), how="left")
    edge_start = joined["first_adequate"] - joined["first_date"]
    edge_end = joined["last_date"] - joined["last_adequate"]
    with_adequate = pd.concat(
        [edge_start.rename("start"), edge_end.rename("end"), joined["interior"]], axis=1
    ).max(axis=1).fillna(0)
    return with_adequate.where(joined["adequate_days"].gt(0), joined["span_days"])


def add_channel_wide(
    features: pd.DataFrame,
    per_channel: pd.DataFrame,
    user_ids: pd.Index,
    columns: list[str],
    prefix: str,
) -> None:
    for channel in CHANNELS:
        channel_frame = per_channel.xs(channel, level="channel").reindex(user_ids)
        for column in columns:
            features[f"{prefix}ch{channel}_{column}"] = channel_frame[column].to_numpy()


def build_feature_table(u: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the declared A, R, and V blocks for all U users from frozen aggregates."""
    user_ids = pd.Index(u["user_id"].astype(int), name="user_id")
    user_set = set(user_ids)
    log("Loading frozen daily aggregates and deriving adequate-day features …")
    days = pd.read_parquet(
        INPUT_DIR / "epoch_days.parquet",
        columns=["user", "channel", "date", "n", "cov_s", "hours", "mean", "median", "sd", "vmin", "vmax"],
    )
    days = days.loc[days["user"].isin(user_set) & days["channel"].isin(CHANNELS)].copy()
    ensure(
        not days.duplicated(["user", "channel", "date"]).any(),
        "epoch_days has duplicate user/channel/date rows",
    )
    present_users = set(days["user"].unique())
    missing_users = user_set - present_users
    ensure(not missing_users, f"{len(missing_users)} U participants have no epoch-day rows")
    days["adequate"] = (
        days["hours"].ge(ADEQUATE_HOURS)
        & (days["cov_s"].ge(ADEQUATE_COV_SECONDS) | days["n"].ge(ADEQUATE_N))
    )
    days["cov_h"] = days["cov_s"] / 3600.0

    grouped = days.groupby(["user", "channel"], sort=False)
    per_channel = grouped.agg(
        observed_days=("date", "size"),
        adequate_days=("adequate", "sum"),
        events=("n", "sum"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        mean_hours=("hours", "mean"),
        mean_cov_h=("cov_h", "mean"),
    )
    per_channel["span_days"] = per_channel["last_date"] - per_channel["first_date"] + 1
    per_channel["duty_cycle"] = per_channel["adequate_days"] / per_channel["span_days"]
    per_channel["events_per_observed_day"] = (
        per_channel["events"] / per_channel["observed_days"]
    )
    per_channel["longest_inadequate_run"] = longest_inadequate_runs(days, per_channel)

    grid = pd.MultiIndex.from_product([user_ids, CHANNELS], names=["user", "channel"])
    per_channel = per_channel.reindex(grid)
    for column in ("observed_days", "adequate_days", "events"):
        per_channel[column] = per_channel[column].fillna(0).astype(int)
    per_channel["present"] = per_channel["observed_days"].gt(0).astype(int)

    features = pd.DataFrame(index=user_ids)
    add_channel_wide(
        features,
        per_channel,
        user_ids,
        list(PROCESS_STATS[:9]) + ["present"],
        "r__",
    )

    # Event-placement fractions are built from local-hour aggregate counts.  They are
    # explicitly missing, not UTC-substituted, whenever local timezone is unusable.
    meta = pd.read_parquet(INPUT_DIR / "epoch_meta.parquet", columns=["user", "valid_events", "tz_missing_events"])
    meta = meta.set_index("user").reindex(user_ids)
    tz_ok = meta["valid_events"].gt(0) & meta["tz_missing_events"].eq(0)
    hours = pd.read_parquet(INPUT_DIR / "epoch_hours.parquet", columns=["user", "channel", "hour", "n"])
    hours = hours.loc[hours["user"].isin(user_set) & hours["channel"].isin(CHANNELS)].copy()
    ensure(hours["hour"].between(0, 23).all(), "epoch_hours contains an invalid clock hour")
    hours["bucket"] = np.select(
        [hours["hour"].le(5), hours["hour"].le(11), hours["hour"].le(17)],
        ["night", "morning", "afternoon"],
        default="evening",
    )
    bucketed = hours.groupby(["user", "channel", "bucket"], sort=False)["n"].sum()
    totals = bucketed.groupby(["user", "channel"], sort=False).transform("sum")
    fractions = (bucketed / totals).rename("fraction").reset_index()
    fraction_wide = fractions.pivot(index=["user", "channel"], columns="bucket", values="fraction")
    fraction_wide = fraction_wide.reindex(grid)
    hour_count = (
        hours.loc[hours["n"].gt(0)]
        .groupby(["user", "channel"], sort=False)["hour"]
        .nunique()
        .rename("n_hours_with_events")
        .reindex(grid)
    )
    for channel in CHANNELS:
        cfrac = fraction_wide.xs(channel, level="channel").reindex(user_ids)
        for bucket in ("night", "morning", "afternoon", "evening"):
            values = cfrac[bucket] if bucket in cfrac else pd.Series(np.nan, index=user_ids)
            features[f"r__ch{channel}_todfrac_{bucket}"] = values.where(tz_ok).to_numpy()
        values = hour_count.xs(channel, level="channel").reindex(user_ids)
        features[f"r__ch{channel}_n_hours_with_events"] = values.where(tz_ok).to_numpy()

    # Overall recording-process summaries across core channels.
    overall = days.groupby("user", sort=False).agg(
        overall_events=("n", "sum"),
        overall_unique_days=("date", "nunique"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    overall["overall_span_days"] = overall["last_date"] - overall["first_date"] + 1
    any_adequate = (
        days.groupby(["user", "date"], sort=False)["adequate"]
        .max()
        .groupby("user", sort=False)
        .sum()
        .rename("any_channel_adequate_days")
    )
    overall = overall.join(any_adequate)
    overall["any_channel_duty"] = (
        overall["any_channel_adequate_days"] / overall["overall_span_days"]
    )
    overall["n_core_channels_present"] = (
        per_channel["present"].groupby("user", sort=False).sum()
    )
    overall = overall.reindex(user_ids)
    for column in OVERALL_PROCESS_STATS:
        features[f"r__{column}"] = overall[column].to_numpy()

    # Adequate-day-only daily HR summaries.  No diurnal HR means are used because
    # epoch_hours has no date and cannot support adequate-day restriction.
    adequate_days = days.loc[days["adequate"]].copy()
    values = adequate_days.groupby(["user", "channel"], sort=False).agg(
        mean_of_daily_mean=("mean", "mean"),
        sd_of_daily_mean=("mean", "std"),
        mean_of_daily_sd=("sd", "mean"),
        median_of_daily_median=("median", "median"),
        p25_of_daily_median=("median", lambda x: x.quantile(0.25)),
        p75_of_daily_median=("median", lambda x: x.quantile(0.75)),
        min_of_daily_min=("vmin", "min"),
        max_of_daily_max=("vmax", "max"),
    )
    values["iqr_of_daily_median"] = (
        values["p75_of_daily_median"] - values["p25_of_daily_median"]
    )
    values["range_of_daily_extremes"] = (
        values["max_of_daily_max"] - values["min_of_daily_min"]
    )
    adequate_days["weekday"] = ((adequate_days["date"] + 3) % 7).lt(5)
    weekday = (
        adequate_days.loc[adequate_days["weekday"]]
        .groupby(["user", "channel"], sort=False)["mean"]
        .mean()
        .rename("weekday_mean")
    )
    weekend = (
        adequate_days.loc[~adequate_days["weekday"]]
        .groupby(["user", "channel"], sort=False)["mean"]
        .mean()
        .rename("weekend_mean")
    )
    values = values.join(weekday, how="left").join(weekend, how="left")
    values["weekend_minus_weekday"] = values["weekend_mean"] - values["weekday_mean"]
    values = values.reindex(grid)
    add_channel_wide(features, values, user_ids, list(VALUE_STATS), "v__")
    features["v__ch3000_minus_ch3001_mean"] = (
        features["v__ch3000_mean_of_daily_mean"] - features["v__ch3001_mean_of_daily_mean"]
    )
    features["v__ch3000_minus_ch3002_mean"] = (
        features["v__ch3000_mean_of_daily_mean"] - features["v__ch3002_mean_of_daily_mean"]
    )

    anchors = u.set_index("user_id").reindex(user_ids)
    features["a__epoch_primary_source"] = anchors["epoch_primary_source"].astype("Int64")
    features["a__epoch_multisource"] = anchors["epoch_multisource"].astype(bool)
    features["a__gate_pass"] = anchors["gate_pass"].astype(int)
    features["y"] = anchors["y"].astype(int)
    features["cohort_category"] = anchors["cohort_category"].astype(str)
    features.index.name = "user_id"
    features = features.reset_index()

    r_columns = [column for column in features.columns if column.startswith("r__")]
    v_columns = [column for column in features.columns if column.startswith("v__")]
    ensure(len(r_columns) == 51, f"R width drift: {len(r_columns)} != 51")
    ensure(len(v_columns) == 35, f"V width drift: {len(v_columns)} != 35")

    dictionary_rows: list[dict[str, str]] = []
    for column in ("a__epoch_primary_source", "a__epoch_multisource", "a__gate_pass"):
        dictionary_rows.append(
            {
                "feature": column,
                "block": "A",
                "units": "",
                "missingness": "not expected in frozen U cohort",
                "description": column.removeprefix("a__"),
            }
        )
    for column in r_columns:
        dictionary_rows.append(
            {
                "feature": column,
                "block": "R",
                "units": "",
                "missingness": "absent-channel timing/coverage fields are missing; presence is zero",
                "description": column.removeprefix("r__"),
            }
        )
    for column in v_columns:
        dictionary_rows.append(
            {
                "feature": column,
                "block": "V",
                "units": "bpm",
                "missingness": "missing when adequate-day value summary is unavailable",
                "description": column.removeprefix("v__"),
            }
        )
    return features, pd.DataFrame(dictionary_rows)


def verify_feature_table(features: pd.DataFrame, g: pd.DataFrame, u: pd.DataFrame) -> None:
    ensure(features["user_id"].is_unique, "feature table has duplicate user IDs")
    ensure(set(features["user_id"]) == set(u["user_id"]), "feature table is not exactly U")
    r_columns = [column for column in features.columns if column.startswith("r__")]
    v_columns = [column for column in features.columns if column.startswith("v__")]
    ensure(len(r_columns) == 51 and len(v_columns) == 35, "unexpected feature-block width")
    forbidden = ("user_id", "salutation", "gender", "sex", "who", "age", "bmi", "first_date", "last_date")
    predictors = r_columns + v_columns + [column for column in features if column.startswith("a__")]
    for column in predictors:
        ensure(
            not any(token in column.lower() for token in forbidden),
            f"forbidden predictor name: {column}",
        )
    for channel in CHANNELS:
        no_adequate = features[f"r__ch{channel}_adequate_days"].eq(0)
        channel_values = [column for column in v_columns if f"v__ch{channel}_" in column]
        ensure(
            features.loc[no_adequate, channel_values].isna().all(axis=None),
            f"channel {channel} value summaries are not all NaN when adequate days equal zero",
        )
    for channel in CHANNELS:
        frac_columns = [f"r__ch{channel}_todfrac_{name}" for name in ("night", "morning", "afternoon", "evening")]
        complete = features[frac_columns].notna().all(axis=1)
        if complete.any():
            sums = features.loc[complete, frac_columns].sum(axis=1)
            ensure(np.allclose(sums, 1.0, rtol=0, atol=1e-8), f"channel {channel} TOD fractions do not sum to one")
    g_rows = features.loc[features["user_id"].isin(g["user_id"])]
    ensure(len(g_rows) == len(g), "G could not be recovered from U feature table")


def arm_specs(cohort: str, features: pd.DataFrame) -> list[ArmSpec]:
    a_source = ["a__epoch_primary_source", "a__epoch_multisource"]
    a = a_source + (["a__gate_pass"] if cohort == "U" else [])
    r = [column for column in features.columns if column.startswith("r__")]
    v = [column for column in features.columns if column.startswith("v__")]
    v0 = [column for column in v if column.startswith("v__ch3000_")]
    resting_presence = ["r__ch3001_present", "r__ch3002_present"]

    specs: list[ArmSpec] = []
    if cohort == "U":
        specs.extend(
            [
                ArmSpec("Agate", ("a__gate_pass",)),
                ArmSpec("Asource", tuple(a_source)),
            ]
        )
    # In G, A and Asource are identical and only A is fit/reported once.
    specs.extend(
        [
            ArmSpec("A", tuple(a)),
            ArmSpec("R0", tuple((["a__gate_pass"] if cohort == "U" else []) + r)),
            ArmSpec("R", tuple(ordered_unique(a + r))),
            ArmSpec("Vm", tuple(ordered_unique(a + v)), encoding="median"),
            ArmSpec("Vn", tuple(ordered_unique(a + v))),
            ArmSpec("RV", tuple(ordered_unique(a + r + v))),
            ArmSpec("V0", tuple(ordered_unique(a + v0))),
            ArmSpec("P", tuple(ordered_unique(a + v0 + resting_presence))),
        ]
    )
    ensure(all("a__gate_pass" not in spec.columns for spec in specs) if cohort == "G" else True,
           "gate pass leaked into a G arm")
    return specs


def preprocessed_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: ArmSpec,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    raw_columns = list(spec.columns)
    all_missing = [column for column in raw_columns if train[column].isna().all()]
    remaining = [column for column in raw_columns if column not in all_missing]
    # NaN counts as a state: a constant observed value plus missingness is not dropped,
    # because native missing-value splits can use that distinction.
    zero_variance = [
        column for column in remaining if train[column].nunique(dropna=False) <= 1
    ]
    kept = [column for column in remaining if column not in zero_variance]
    categorical = [
        column
        for column in kept
        if column in {"a__epoch_primary_source", "a__epoch_multisource"}
    ]
    numeric = [column for column in kept if column not in categorical]

    if numeric:
        x_train_numeric = train[numeric].astype(float).to_numpy(copy=True)
        x_test_numeric = test[numeric].astype(float).to_numpy(copy=True)
        if spec.encoding == "median":
            medians = np.nanmedian(x_train_numeric, axis=0)
            ensure(not np.isnan(medians).any(), f"median arm retained all-missing column: {spec.name}")
            x_train_numeric = np.where(np.isnan(x_train_numeric), medians, x_train_numeric)
            x_test_numeric = np.where(np.isnan(x_test_numeric), medians, x_test_numeric)
        elif spec.encoding != "native":
            raise RuntimeError(f"unknown encoding: {spec.encoding}")
    else:
        x_train_numeric = np.empty((len(train), 0), dtype=float)
        x_test_numeric = np.empty((len(test), 0), dtype=float)

    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        x_train_categorical = encoder.fit_transform(categorical_values(train, categorical))
        x_test_categorical = encoder.transform(categorical_values(test, categorical))
        categorical_names = list(encoder.get_feature_names_out(categorical))
    else:
        x_train_categorical = np.empty((len(train), 0), dtype=float)
        x_test_categorical = np.empty((len(test), 0), dtype=float)
        categorical_names = []

    x_train = np.hstack([x_train_numeric, x_train_categorical])
    x_test = np.hstack([x_test_numeric, x_test_categorical])
    names = numeric + categorical_names
    ensure(x_train.shape[1] > 0, f"no usable predictors in arm {spec.name}")

    # OHE can still expose a constant column when a categorical train set has one level.
    # Keep partial-NaN numeric columns for the reason above.
    transformed_keep: list[int] = []
    for index in range(x_train.shape[1]):
        column = x_train[:, index]
        observed = column[~np.isnan(column)]
        states = len(np.unique(observed)) + int(np.isnan(column).any())
        if states > 1:
            transformed_keep.append(index)
    dropped_transformed = [name for index, name in enumerate(names) if index not in transformed_keep]
    x_train = x_train[:, transformed_keep]
    x_test = x_test[:, transformed_keep]
    names = [names[index] for index in transformed_keep]
    ensure(x_train.shape[1] > 0, f"all transformed predictors are constant in arm {spec.name}")
    if spec.encoding == "median":
        ensure(not np.isnan(x_train).any() and not np.isnan(x_test).any(), "Vm still has NaNs")

    schema = {
        "arm": spec.name,
        "encoding": spec.encoding,
        "raw_columns": raw_columns,
        "all_missing_training": all_missing,
        "zero_variance_raw_training": zero_variance,
        "zero_variance_transformed_training": dropped_transformed,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "transformed_feature_names": names,
        "n_transformed_features": len(names),
        "native_nan_in_training_matrix": bool(np.isnan(x_train).any()),
    }
    return x_train, x_test, names, schema


def evaluate(y_true: np.ndarray, proba_1: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    return {
        "auroc": float(roc_auc_score(y_true, proba_1)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision_0": float(precision[0]),
        "precision_1": float(precision[1]),
        "recall_0": float(recall[0]),
        "recall_1": float(recall[1]),
        "f1_0": float(f1[0]),
        "f1_1": float(f1[1]),
        "support_0": int(support[0]),
        "support_1": int(support[1]),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def fit_arm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: ArmSpec,
    forest_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    x_train, x_test, _, schema = preprocessed_matrices(train, test, spec)
    model = RandomForestClassifier(random_state=forest_seed, **RF_PARAMS)
    model.fit(x_train, train["y"].to_numpy(dtype=int))
    class_index = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(x_test)[:, class_index]
    predictions = model.predict(x_test).astype(int)
    metric = evaluate(test["y"].to_numpy(dtype=int), probabilities, predictions)
    metric["n_train"] = len(train)
    metric["n_test"] = len(test)
    metric["n_transformed_features"] = schema["n_transformed_features"]
    prediction_frame = pd.DataFrame(
        {
            "user_id": test["user_id"].to_numpy(dtype=np.int64),
            "y": test["y"].to_numpy(dtype=np.int8),
            "proba_1": probabilities,
            "pred": predictions.astype(np.int8),
            "epoch_primary_source": test["a__epoch_primary_source"].to_numpy(),
            "gate_pass": test["a__gate_pass"].to_numpy(dtype=np.int8),
        }
    )
    return metric, prediction_frame, schema


def stratified_split(frame: pd.DataFrame, cohort: str, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    if cohort == "U":
        strata = frame["y"].astype(str) + "_gate_" + frame["a__gate_pass"].astype(str)
    else:
        strata = frame["y"].astype(str)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=split_seed)
    train_index, test_index = next(splitter.split(frame, strata))
    ensure(not set(train_index) & set(test_index), "split indices overlap")
    ensure(len(train_index) + len(test_index) == len(frame), "split is not exhaustive")
    return train_index, test_index


def split_composition(
    train: pd.DataFrame, test: pd.DataFrame, cohort: str, repeat: int, split_seed: int, forest_seed: int
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split_name, frame in (("train", train), ("test", test)):
        for (source, y, gate), count in frame.groupby(
            ["a__epoch_primary_source", "y", "a__gate_pass"], dropna=False
        ).size().items():
            records.append(
                {
                    "cohort": cohort,
                    "repeat": repeat,
                    "split": split_name,
                    "epoch_primary_source": source,
                    "y": y,
                    "gate_pass": gate,
                    "n": int(count),
                    "split_seed": split_seed,
                    "forest_seed": forest_seed,
                }
            )
    return pd.DataFrame(records)


def selected_slices(predictions: pd.DataFrame, cohort: str, repeat: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, arm_frame in predictions.groupby("arm", sort=False):
        for source, subset in arm_frame.groupby("epoch_primary_source", dropna=False):
            if subset["y"].nunique() == 2:
                rows.append(
                    {
                        "cohort": cohort,
                        "repeat": repeat,
                        "arm": arm,
                        "slice": "epoch_primary_source",
                        "level": source,
                        "n": len(subset),
                        "n_y0": int(subset["y"].eq(0).sum()),
                        "n_y1": int(subset["y"].eq(1).sum()),
                        "auroc": float(roc_auc_score(subset["y"], subset["proba_1"])),
                    }
                )
        if cohort == "U":
            for gate, subset in arm_frame.groupby("gate_pass", dropna=False):
                if subset["y"].nunique() == 2:
                    rows.append(
                        {
                            "cohort": cohort,
                            "repeat": repeat,
                            "arm": arm,
                            "slice": "gate_pass",
                            "level": gate,
                            "n": len(subset),
                            "n_y0": int(subset["y"].eq(0).sum()),
                            "n_y1": int(subset["y"].eq(1).sum()),
                            "auroc": float(roc_auc_score(subset["y"], subset["proba_1"])),
                        }
                    )
    return rows


def write_cell_outputs(
    output_dir: Path,
    cohort: str,
    repeat: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    schemas: dict[str, dict[str, Any]],
) -> None:
    split_dir = output_dir / "splits" / cohort
    pred_dir = output_dir / "predictions" / cohort
    schema_dir = output_dir / "schemas" / cohort
    split_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)
    membership = pd.concat(
        [
            train[["user_id", "y", "a__gate_pass", "a__epoch_primary_source"]].assign(split="train"),
            test[["user_id", "y", "a__gate_pass", "a__epoch_primary_source"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    membership.to_parquet(split_dir / f"repeat_{repeat:02d}.parquet", index=False)
    predictions.to_parquet(pred_dir / f"repeat_{repeat:02d}.parquet", index=False)
    write_json(schema_dir / f"repeat_{repeat:02d}.json", schemas)


def run_repeats(
    frame: pd.DataFrame,
    cohort: str,
    output_dir: Path,
    n_repeats: int,
    seed_prefix: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = arm_specs(cohort, frame)
    metric_rows: list[dict[str, Any]] = []
    composition_rows: list[pd.DataFrame] = []
    slice_rows: list[dict[str, Any]] = []
    for repeat in range(n_repeats):
        split_seed = seed_from([STUDY_SEED, seed_prefix, 0, 0 if cohort == "G" else 1, repeat])
        forest_seed = seed_from([STUDY_SEED, seed_prefix, 1, 0 if cohort == "G" else 1, repeat])
        train_index, test_index = stratified_split(frame, cohort, split_seed)
        train = frame.iloc[train_index].reset_index(drop=True)
        test = frame.iloc[test_index].reset_index(drop=True)
        started = time.monotonic()
        prediction_frames: list[pd.DataFrame] = []
        schemas: dict[str, dict[str, Any]] = {}
        for spec in specs:
            metric, predictions, schema = fit_arm(train, test, spec, forest_seed)
            metric_rows.append(
                {
                    "cohort": cohort,
                    "repeat": repeat,
                    "arm": spec.name,
                    "split_seed": split_seed,
                    "forest_seed": forest_seed,
                    **metric,
                }
            )
            prediction_frames.append(predictions.assign(cohort=cohort, repeat=repeat, arm=spec.name))
            schemas[spec.name] = schema
        joined_predictions = pd.concat(prediction_frames, ignore_index=True)
        write_cell_outputs(output_dir, cohort, repeat, train, test, joined_predictions, schemas)
        composition_rows.append(split_composition(train, test, cohort, repeat, split_seed, forest_seed))
        slice_rows.extend(selected_slices(joined_predictions, cohort, repeat))
        elapsed = time.monotonic() - started
        log(f"{cohort} repeat {repeat + 1:02d}/{n_repeats}: {len(specs)} arms in {elapsed:.1f}s")
    return (
        pd.DataFrame(metric_rows),
        pd.concat(composition_rows, ignore_index=True),
        pd.DataFrame(slice_rows),
    )


def sample_smoke_u(u_frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(seed_from([STUDY_SEED, 9, 0]))
    pieces: list[pd.DataFrame] = []
    for _, group in u_frame.groupby(["y", "a__gate_pass"], sort=True):
        ensure(
            len(group) >= SMOKE_PER_STRATUM,
            f"smoke stratum too small: {len(group)} < {SMOKE_PER_STRATUM}",
        )
        positions = rng.choice(len(group), size=SMOKE_PER_STRATUM, replace=False)
        pieces.append(group.iloc[np.sort(positions)])
    return pd.concat(pieces, ignore_index=True).sort_values("user_id").reset_index(drop=True)


def frame_for_cohort(features: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    result = features.loc[features["user_id"].isin(set(ids))].copy()
    ensure(len(result) == len(ids), "cohort feature join lost participants")
    return result.sort_values("user_id").reset_index(drop=True)


def assert_reproducible(first: Path, second: Path) -> dict[str, Any]:
    first_metrics = pd.read_csv(first / "metrics.csv").sort_values(["cohort", "repeat", "arm"]).reset_index(drop=True)
    second_metrics = pd.read_csv(second / "metrics.csv").sort_values(["cohort", "repeat", "arm"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(first_metrics, second_metrics, check_exact=False, rtol=1e-12, atol=1e-12)
    for cohort in ("G", "U"):
        one = pd.read_parquet(first / "predictions" / cohort / "repeat_00.parquet").sort_values(
            ["arm", "user_id"]
        ).reset_index(drop=True)
        two = pd.read_parquet(second / "predictions" / cohort / "repeat_00.parquet").sort_values(
            ["arm", "user_id"]
        ).reset_index(drop=True)
        pd.testing.assert_series_equal(one["user_id"], two["user_id"], check_exact=True)
        pd.testing.assert_series_equal(one["y"], two["y"], check_exact=True)
        pd.testing.assert_series_equal(one["pred"], two["pred"], check_exact=True)
        np.testing.assert_allclose(one["proba_1"], two["proba_1"], rtol=1e-12, atol=1e-12)
        one_split = pd.read_parquet(first / "splits" / cohort / "repeat_00.parquet").sort_values(
            ["split", "user_id"]
        ).reset_index(drop=True)
        two_split = pd.read_parquet(second / "splits" / cohort / "repeat_00.parquet").sort_values(
            ["split", "user_id"]
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(one_split, two_split, check_exact=True)
    return {
        "status": "passed",
        "comparison": "exact memberships and hard predictions; probabilities and metrics within 1e-12",
    }


def paired_summary(deltas: pd.Series, ci_level: float) -> dict[str, Any]:
    n = len(deltas)
    sd = float(deltas.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    alpha = 1.0 - ci_level
    critical = float(t.ppf(1 - alpha / 2, df=n - 1)) if n > 1 else float("nan")
    mean = float(deltas.mean())
    return {
        "n_repeats": n,
        "mean_delta_auroc": mean,
        "sd_delta_auroc": sd,
        "mc_se": se,
        "median_delta_auroc": float(deltas.median()),
        "iqr_delta_auroc": float(deltas.quantile(0.75) - deltas.quantile(0.25)),
        "share_delta_gt_zero": float(deltas.gt(0).mean()),
        "ci_level": ci_level,
        "ci_low": mean - critical * se,
        "ci_high": mean + critical * se,
    }


def summarize_repeated_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute = (
        metrics.groupby(["cohort", "arm"], sort=False)
        .agg(
            n_repeats=("repeat", "size"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            n_features_median=("n_transformed_features", "median"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    primary_u = {("R", "A"), ("RV", "R"), ("RV", "Vn"), ("Vn", "Vm")}
    selected_exploratory = {("R", "R0"), ("P", "V0"), ("Vn", "V0")}
    for cohort, cohort_metrics in metrics.groupby("cohort", sort=False):
        pivot = cohort_metrics.pivot(index="repeat", columns="arm", values="auroc")
        ordered_arms = [arm for arm in cohort_metrics["arm"].drop_duplicates() if arm in pivot.columns]
        for left, right in combinations(ordered_arms, 2):
            # Keep every reported difference in an unambiguous order.  Add the
            # prespecified primary orientations below if this pair is one of them.
            for numerator, denominator in ((left, right),):
                is_primary = cohort == "U" and (numerator, denominator) in primary_u
                summary = paired_summary(
                    pivot[numerator] - pivot[denominator], 0.9875 if is_primary else 0.95
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "contrast": f"{numerator} - {denominator}",
                        "numerator": numerator,
                        "denominator": denominator,
                        "primary_u": is_primary,
                        **summary,
                    }
                )
        # Prespecified directions need not match arm order.  Add primary U and
        # selected exploratory orientations explicitly, without removing the full table.
        existing = {(row["numerator"], row["denominator"]) for row in rows if row["cohort"] == cohort}
        requested = set(selected_exploratory)
        if cohort == "U":
            requested |= primary_u
        for numerator, denominator in sorted(requested):
            if (numerator, denominator) not in existing:
                is_primary = cohort == "U" and (numerator, denominator) in primary_u
                summary = paired_summary(
                    pivot[numerator] - pivot[denominator], 0.9875 if is_primary else 0.95
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "contrast": f"{numerator} - {denominator}",
                        "numerator": numerator,
                        "denominator": denominator,
                        "primary_u": is_primary,
                        **summary,
                    }
                )
    comparisons = pd.DataFrame(rows)
    return absolute, comparisons


def run_bridge(g_frame: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One frozen-v1 G split bridge; descriptive only, with aligned bootstrap draws."""
    log("Running frozen-v1 G bridge …")
    split_manifest = pd.read_csv(INPUT_DIR / "split_manifest.csv")
    ensure(split_manifest["user_id"].is_unique, "v1 split manifest duplicates users")
    ensure(set(split_manifest["user_id"]) == set(g_frame["user_id"]), "v1 G split membership drift")
    joined = g_frame.merge(split_manifest, on="user_id", validate="one_to_one")
    train = joined.loc[joined["split"].eq("train")].reset_index(drop=True)
    test = joined.loc[joined["split"].eq("test")].reset_index(drop=True)
    ensure(len(train) == 16_388 and len(test) == 2_048, "unexpected frozen v1 train/test sizes")
    forest_seed = seed_from([STUDY_SEED, 2, 0])
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    schemas: dict[str, dict[str, Any]] = {}
    for spec in arm_specs("G", g_frame):
        metric, predictions, schema = fit_arm(train, test, spec, forest_seed)
        rows.append({"arm": spec.name, "forest_seed": forest_seed, **metric})
        frames.append(predictions.assign(arm=spec.name))
        schemas[spec.name] = schema
    predictions = pd.concat(frames, ignore_index=True)
    bridge_dir = output_dir / "bridge"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(bridge_dir / "predictions.parquet", index=False)
    write_json(bridge_dir / "schemas.json", schemas)

    v1 = pd.read_csv(INPUT_DIR / "predictions_test.csv")
    ensure(v1["user_id"].is_unique, "v1 prediction file duplicates users")
    ensure(set(v1["user_id"]) == set(test["user_id"]), "v1 test IDs do not match frozen G test split")
    y_by_id = test.set_index("user_id")["y"]
    ensure((v1.set_index("user_id")["y"].astype(int) == y_by_id).all(), "v1 test outcomes drift")
    v1 = v1[["user_id", "y", "proba_1"]].copy()
    v1["arm"] = "v1"
    bridge_predictions = pd.concat([predictions, v1], ignore_index=True)
    bridge_metric_rows: list[dict[str, Any]] = []
    for arm, arm_predictions in bridge_predictions.groupby("arm", sort=False):
        if arm_predictions["pred"].notna().all():
            metric = evaluate(
                arm_predictions["y"].to_numpy(),
                arm_predictions["proba_1"].to_numpy(),
                arm_predictions["pred"].to_numpy(),
            )
        else:
            metric = {"auroc": float(roc_auc_score(arm_predictions["y"], arm_predictions["proba_1"]))}
        bridge_metric_rows.append({"arm": arm, **metric})
    bridge_metrics = pd.DataFrame(bridge_metric_rows)

    # Shared, within-class resampling of the aligned frozen test IDs.
    ordered_ids = np.sort(test["user_id"].to_numpy(dtype=np.int64))
    ordered_y = y_by_id.reindex(ordered_ids).to_numpy(dtype=int)
    probability_wide = bridge_predictions.pivot(index="user_id", columns="arm", values="proba_1").reindex(ordered_ids)
    ensure(not probability_wide.isna().any().any(), "bridge prediction alignment is incomplete")
    index0 = np.flatnonzero(ordered_y == 0)
    index1 = np.flatnonzero(ordered_y == 1)
    rng = np.random.default_rng(seed_from([STUDY_SEED, 2, 1]))
    draw_indices = np.empty((1000, len(ordered_ids)), dtype=np.int32)
    draw_rows: list[dict[str, Any]] = []
    for draw in range(1000):
        indices = np.concatenate(
            [
                rng.choice(index0, size=len(index0), replace=True),
                rng.choice(index1, size=len(index1), replace=True),
            ]
        )
        draw_indices[draw] = indices
        y_draw = ordered_y[indices]
        for arm in probability_wide.columns:
            draw_rows.append(
                {
                    "draw": draw,
                    "arm": arm,
                    "auroc": float(roc_auc_score(y_draw, probability_wide[arm].to_numpy()[indices])),
                }
            )
    np.savez_compressed(bridge_dir / "bootstrap_indices.npz", user_ids=ordered_ids, indices=draw_indices)
    bootstrap = pd.DataFrame(draw_rows)
    bootstrap.to_parquet(bridge_dir / "bootstrap_auroc.parquet", index=False)
    boot_wide = bootstrap.pivot(index="draw", columns="arm", values="auroc")
    delta_rows: list[dict[str, Any]] = []
    for arm in [column for column in boot_wide.columns if column != "v1"]:
        delta = boot_wide[arm] - boot_wide["v1"]
        delta_rows.append(
            {
                "contrast": f"{arm} - v1",
                "point_delta_auroc": float(
                    bridge_metrics.set_index("arm").loc[arm, "auroc"]
                    - bridge_metrics.set_index("arm").loc["v1", "auroc"]
                ),
                "bootstrap_mean": float(delta.mean()),
                "bootstrap_sd": float(delta.std(ddof=1)),
                "bootstrap_q025": float(delta.quantile(0.025)),
                "bootstrap_q975": float(delta.quantile(0.975)),
            }
        )
    deltas = pd.DataFrame(delta_rows)
    bridge_metrics.to_csv(bridge_dir / "metrics.csv", index=False)
    deltas.to_csv(bridge_dir / "v1_paired_deltas.csv", index=False)
    return bridge_metrics, deltas


def markdown_table(frame: pd.DataFrame, columns: list[str], n: int | None = None) -> str:
    subset = frame.loc[:, columns].copy()
    if n is not None:
        subset = subset.head(n)
    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(format_value(value) for value in row) + " |" for row in subset.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def write_report(
    output_dir: Path,
    cohort_summary: pd.DataFrame,
    absolute: pd.DataFrame,
    comparisons: pd.DataFrame,
    bridge_metrics: pd.DataFrame,
    bridge_deltas: pd.DataFrame,
) -> None:
    primary = comparisons.loc[comparisons["primary_u"]].copy()
    secondary = comparisons.loc[
        (~comparisons["primary_u"])
        & comparisons["contrast"].isin(["R - R0", "P - V0", "Vn - V0"])
    ].copy()
    lines = [
        "# Informative recording-feature experiment",
        "",
        "## Design",
        "",
        "This is the standalone implementation of the Codex 2026-09-20 protocol. "
        "It uses frozen v1 aggregate artifacts only, recorded salutation 10/20, participant-level "
        "seeded 80/20 repeated splits, and a fixed 100-tree regularized random forest. "
        "It does not identify an MNAR mechanism, causal device effect, physiological effect, biological sex, or gender identity.",
        "",
        markdown_table(cohort_summary, ["cohort", "n", "n_y0", "n_y1", "role"]),
        "",
        "U is primary. G is the coverage-gated bridge/exploratory population. The 30 repeated splits "
        "use `SeedSequence([20260920, 0, cohort_code, repeat])`-derived seeds; each arm within a cell shares its forest seed.",
        "",
        "## Absolute repeated-split metrics",
        "",
        "Values are mean and SD over prescribed repeats; they are descriptive repeated-split summaries, not transport intervals.",
        "",
        markdown_table(
            absolute,
            ["cohort", "arm", "n_repeats", "auroc_mean", "auroc_sd", "balanced_accuracy_mean", "balanced_accuracy_sd", "n_features_median"],
        ),
        "",
        "## Primary U contrasts",
        "",
        "Intervals are two-sided 98.75% t intervals: Bonferroni-simultaneous 95% family coverage across the four prespecified U contrasts.",
        "",
        markdown_table(
            primary,
            ["contrast", "mean_delta_auroc", "sd_delta_auroc", "mc_se", "median_delta_auroc", "iqr_delta_auroc", "share_delta_gt_zero", "ci_low", "ci_high"],
        ),
        "",
        "`Vn - Vm` compares native-NaN and training-median value encodings under this forest; it is not a pure missingness effect because median imputation can itself create a point mass and changes candidate splits.",
        "",
        "## Selected exploratory contrasts",
        "",
        markdown_table(
            secondary,
            ["cohort", "contrast", "mean_delta_auroc", "sd_delta_auroc", "ci_low", "ci_high"],
        ),
        "",
        "## Frozen-v1 bridge",
        "",
        "The bridge has one already-used v1 test set. Its deltas are descriptive continuity checks, not feature-only effects, because representation and forest configuration differ from v1.",
        "",
        markdown_table(bridge_metrics, ["arm", "auroc"], n=20),
        "",
        markdown_table(bridge_deltas, ["contrast", "point_delta_auroc", "bootstrap_mean", "bootstrap_sd", "bootstrap_q025", "bootstrap_q975"], n=20),
        "",
        "## Limits",
        "",
        "Recording-process and source features may reflect device choice, vendor processing, wearing behavior, or selection into the export. Results are predictive associations with recorded salutation in these frozen cohorts. No post-hoc feature selection, tuning, or further representation search was performed.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def cohort_summary(g: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cohort": "G", "n": len(g), "n_y0": int(g["y"].eq(0).sum()), "n_y1": int(g["y"].eq(1).sum()), "role": "exploratory gated bridge"},
            {"cohort": "U", "n": len(u), "n_y0": int(u["y"].eq(0).sum()), "n_y1": int(u["y"].eq(1).sum()), "role": "primary ungated cohort"},
        ]
    )


def write_base_outputs(output_dir: Path, features: pd.DataFrame, dictionary: pd.DataFrame, g: pd.DataFrame, u: pd.DataFrame) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(data_dir / "features_u.parquet", index=False)
    dictionary.to_csv(data_dir / "feature_dictionary.csv", index=False)
    cohort_summary(g, u).to_csv(data_dir / "cohort_summary.csv", index=False)


def run_smoke(features: pd.DataFrame, g: pd.DataFrame, u: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    log("Running seeded smoke test twice …")
    u_full = frame_for_cohort(features, u["user_id"])
    u_smoke_one = sample_smoke_u(u_full)
    u_smoke_two = sample_smoke_u(u_full)
    pd.testing.assert_frame_equal(u_smoke_one, u_smoke_two, check_exact=True)
    g_smoke = u_smoke_one.loc[u_smoke_one["a__gate_pass"].eq(1)].copy().reset_index(drop=True)
    ensure(g_smoke["y"].nunique() == 2, "G smoke sample lost a class")
    for label in ("a", "b"):
        target = output_dir / "smoke" / label
        g_metrics, g_composition, g_slices = run_repeats(g_smoke, "G", target, 1, seed_prefix=9)
        u_metrics, u_composition, u_slices = run_repeats(u_smoke_one, "U", target, 1, seed_prefix=9)
        pd.concat([g_metrics, u_metrics], ignore_index=True).to_csv(target / "metrics.csv", index=False)
        pd.concat([g_composition, u_composition], ignore_index=True).to_csv(target / "split_composition.csv", index=False)
        pd.DataFrame(g_slices + u_slices).to_csv(target / "slices.csv", index=False)
    reproducibility = assert_reproducible(output_dir / "smoke" / "a", output_dir / "smoke" / "b")
    smoke_manifest = u_smoke_one[["user_id", "y", "a__gate_pass"]].copy()
    smoke_manifest.to_parquet(output_dir / "smoke" / "sampled_u.parquet", index=False)
    write_json(output_dir / "smoke" / "reproducibility.json", reproducibility)
    log("Seeded smoke test passed.")
    return {
        "smoke_sample_u": len(u_smoke_one),
        "smoke_sample_g": len(g_smoke),
        "smoke_per_y_gate_stratum": SMOKE_PER_STRATUM,
        **reproducibility,
    }


def run_full(features: pd.DataFrame, g: pd.DataFrame, u: pd.DataFrame, output_dir: Path) -> None:
    g_frame = frame_for_cohort(features, g["user_id"])
    u_frame = frame_for_cohort(features, u["user_id"])
    log("Running full G repeated splits …")
    g_metrics, g_composition, g_slices = run_repeats(g_frame, "G", output_dir, N_REPEATS)
    log("Running full U repeated splits …")
    u_metrics, u_composition, u_slices = run_repeats(u_frame, "U", output_dir, N_REPEATS)
    metrics = pd.concat([g_metrics, u_metrics], ignore_index=True)
    composition = pd.concat([g_composition, u_composition], ignore_index=True)
    slices = pd.DataFrame(g_slices + u_slices)
    metrics.to_csv(output_dir / "repeated_metrics.csv", index=False)
    composition.to_csv(output_dir / "split_source_composition.csv", index=False)
    slices.to_csv(output_dir / "descriptive_source_gate_slices.csv", index=False)
    finalize_existing_full(features, g, u, output_dir)


def finalize_existing_full(features: pd.DataFrame, g: pd.DataFrame, u: pd.DataFrame, output_dir: Path) -> None:
    """Write summaries and bridge from completed repeated-split artifacts without refitting them."""
    metrics = pd.read_csv(output_dir / "repeated_metrics.csv")
    expected = {"G": 8 * N_REPEATS, "U": 10 * N_REPEATS}
    observed = metrics.groupby("cohort").size().to_dict()
    ensure(observed == expected, f"incomplete repeated metrics: {observed} != {expected}")
    absolute, comparisons = summarize_repeated_metrics(metrics)
    absolute.to_csv(output_dir / "absolute_metric_summary.csv", index=False)
    comparisons.to_csv(output_dir / "paired_auroc_comparisons.csv", index=False)
    g_frame = frame_for_cohort(features, g["user_id"])
    bridge_metrics, bridge_deltas = run_bridge(g_frame, output_dir)
    write_report(output_dir, cohort_summary(g, u), absolute, comparisons, bridge_metrics, bridge_deltas)


def provenance(output_dir: Path, mode: str, smoke: dict[str, Any] | None) -> None:
    files = {
        str(path.relative_to(PROJECT_DIR)): sha256(path)
        for path in (
            INPUT_DIR / "cohort_manifest.parquet",
            INPUT_DIR / "epoch_days.parquet",
            INPUT_DIR / "epoch_hours.parquet",
            INPUT_DIR / "epoch_meta.parquet",
            INPUT_DIR / "split_manifest.csv",
            INPUT_DIR / "predictions_test.csv",
            PROJECT_DIR / "CODEX_INFORMATIVE_RECORDING_FEATURE_PLAN_2026-09-20.md",
            RUN_DIR / "run_experiment.py",
        )
    }
    write_json(
        output_dir / "reproducibility.json",
        {
            "mode": mode,
            "study_seed": STUDY_SEED,
            "n_repeats": N_REPEATS,
            "random_split": "StratifiedShuffleSplit(train_size=0.8), participant-disjoint and exhaustive",
            "rf_params": RF_PARAMS,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "input_hashes": files,
            "smoke": smoke,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full", "all", "finalize"), default="all")
    parser.add_argument("--output", type=Path, default=RUN_DIR / "output")
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    g, u = load_cohorts()
    if args.mode == "finalize":
        features = pd.read_parquet(output_dir / "data" / "features_u.parquet")
        verify_feature_table(features, g, u)
        finalize_existing_full(features, g, u, output_dir)
        smoke_path = output_dir / "smoke" / "reproducibility.json"
        smoke_result = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
        provenance(output_dir, args.mode, smoke_result)
        elapsed = time.monotonic() - started
        log(f"Completed mode={args.mode} in {elapsed / 60:.1f} minutes. Outputs: {output_dir}")
        return
    features, dictionary = build_feature_table(u)
    verify_feature_table(features, g, u)
    write_base_outputs(output_dir, features, dictionary, g, u)
    log(f"Built U feature table: {len(features):,} participants, 51 R and 35 V numeric features.")

    smoke_result: dict[str, Any] | None = None
    if args.mode in {"smoke", "all"}:
        smoke_result = run_smoke(features, g, u, output_dir)
    if args.mode in {"full", "all"}:
        run_full(features, g, u, output_dir)
    provenance(output_dir, args.mode, smoke_result)
    elapsed = time.monotonic() - started
    log(f"Completed mode={args.mode} in {elapsed / 60:.1f} minutes. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
