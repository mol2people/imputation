#!/usr/bin/env python
"""Standalone OR-Tools configurable source-specific salutation experiment.

This runner deliberately does not import or modify the repository's existing
analysis modules. It scans the raw epoch files once, creates new source-specific
participant features and allocations, then fits the two complementary 100-tree
forests per source. All runtime output stays below this experiment directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

# A raw scan can have five worker processes and an RF can have five joblib
# workers. Prevent numeric libraries inside those workers from multiplying
# that budget with their own thread pools.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import joblib
import numpy as np
import pandas as pd
import sklearn
from ortools import __version__ as ORTOOLS_VERSION
from ortools.sat.python import cp_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_OUTPUT = HERE / "output"
OUTPUT = DEFAULT_OUTPUT
RAW_DIR = REPO / "out"
SALUTATION_PATH = REPO / "13Aug_1222.csv"
WHO_PATH = REPO / "df_whoOneAverage.csv"

STUDY_SEED = 20260921
MAX_WORKERS = 5
# Raw parsing uses up to MAX_WORKERS. The six small final forests are fitted
# serially: an actual repeatability check of this local sklearn build found
# non-identical probabilities with five joblib workers despite fixed seeds.
RF_WORKERS = 1
N_BOOTSTRAP = 10
TARGET_SOURCES = {3: "Garmin", 6: "Apple", 7: "Samsung"}
SOURCE_ORDER = tuple(TARGET_SOURCES)
CHANNELS = (3000, 3001, 3002)
CORE_CHANNEL_SET = set(CHANNELS)
HR_MIN, HR_MAX = 25.0, 230.0
DAY_MS = 86_400_000
HOUR_MS = 3_600_000
TZ_MS = 60_000
RF_PARAMS = {
    "n_estimators": 100,
    "max_features": 0.4,
    "min_samples_leaf": 10,
    "max_depth": 10,
    "class_weight": "balanced_subsample",
}

ALLOCATION_VARIABLE_ALIASES = {
    "salutation": "y",
    "y": "y",
    "age_band_5y": "age_band_5y",
    "bmi_group": "bmi_group",
}


@dataclass(frozen=True)
class AllocationConfig:
    """Two-fold allocation choices, expressed as train and test fractions."""

    balance_columns: tuple[str, ...]
    train_fraction: float
    test_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "balance_variables": [allocation_variable_label(column) for column in self.balance_columns],
            "balance_columns": list(self.balance_columns),
            "fold_sizes": {"train": self.train_fraction, "test": self.test_fraction},
            "hard_salutation_balance": "y" in self.balance_columns,
        }


DEFAULT_ALLOCATION_CONFIG = AllocationConfig(
    balance_columns=("y", "age_band_5y", "bmi_group"),
    train_fraction=0.5,
    test_fraction=0.5,
)


def log(message: str) -> None:
    print(message, flush=True)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def seed_from(parts: Iterable[int]) -> int:
    return int(np.random.SeedSequence(list(parts)).generate_state(1, dtype=np.uint32)[0])


def allocation_variable_label(column: str) -> str:
    return "salutation" if column == "y" else column


def parse_balance_columns(raw: str) -> tuple[str, ...]:
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    ensure(requested, "--balance-vars needs at least one variable")
    unknown = sorted(set(requested) - set(ALLOCATION_VARIABLE_ALIASES))
    ensure(
        not unknown,
        "unknown balance variable(s): " + ", ".join(unknown)
        + "; choose from salutation, age_band_5y, bmi_group",
    )
    columns = tuple(ALLOCATION_VARIABLE_ALIASES[item] for item in requested)
    ensure(len(set(columns)) == len(columns), "--balance-vars contains a duplicate variable")
    return columns


def parse_fold_sizes(raw: str) -> tuple[float, float]:
    try:
        sizes = np.asarray([float(item.strip()) for item in raw.split(",")], dtype=float)
    except ValueError as exc:
        raise RuntimeError("--fold-sizes must be two positive numbers, e.g. 0.5,0.5 or 3,1") from exc
    ensure(len(sizes) == 2, "--fold-sizes currently requires train,test (exactly two values)")
    ensure(np.isfinite(sizes).all() and (sizes > 0).all(), "--fold-sizes must be finite and positive")
    fractions = sizes / sizes.sum()
    return float(fractions[0]), float(fractions[1])


def parse_allocation_config(balance_vars: str, fold_sizes: str) -> AllocationConfig:
    train_fraction, test_fraction = parse_fold_sizes(fold_sizes)
    return AllocationConfig(
        balance_columns=parse_balance_columns(balance_vars),
        train_fraction=train_fraction,
        test_fraction=test_fraction,
    )


def configure_output(raw: str) -> Path:
    candidate = (HERE / raw).resolve()
    ensure(not Path(raw).is_absolute(), "--output-dir must be relative to this experiment directory")
    ensure(candidate != HERE and HERE in candidate.parents, "--output-dir must stay below this experiment directory")
    return candidate


def cohort_path_for_read() -> Path:
    own = OUTPUT / "cohort_manifest.parquet"
    if own.exists():
        return own
    shared = DEFAULT_OUTPUT / "cohort_manifest.parquet"
    ensure(shared.exists(), f"no cohort manifest in {OUTPUT} or {DEFAULT_OUTPUT}; run --mode scan first")
    log(f"Reusing cohort cache: {shared}")
    return shared


def loaded_allocation_config() -> AllocationConfig:
    path = OUTPUT / "allocation_solver.json"
    ensure(path.exists(), f"missing allocation metadata: {path}")
    payload = json.loads(path.read_text())
    raw = payload.get("allocation_config")
    if raw is None:
        return DEFAULT_ALLOCATION_CONFIG
    fold_sizes = raw["fold_sizes"]
    return AllocationConfig(
        balance_columns=tuple(raw["balance_columns"]),
        train_fraction=float(fold_sizes["train"]),
        test_fraction=float(fold_sizes["test"]),
    )


def ensure_output_config(config: AllocationConfig) -> None:
    path = OUTPUT / "allocation_solver.json"
    if not path.exists():
        return
    existing = loaded_allocation_config()
    ensure(
        existing == config,
        f"{OUTPUT} already contains a different allocation; use --output-dir for this configuration",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def age_band(age: float) -> str:
    if not np.isfinite(age):
        return "MISSING"
    integer_age = int(age)
    if integer_age < 0 or integer_age > 120:
        return "INVALID"
    low = 5 * (integer_age // 5)
    return f"{low}-{low + 4}"


VALUE_SUFFIXES = ("mean", "sd", "min", "q25", "median", "q75", "max")
TIME_BUCKETS = ("night", "morning", "afternoon", "evening")


def feature_columns() -> list[str]:
    columns = [
        "f__overall_n_events",
        "f__overall_n_days",
        "f__overall_span_days",
        "f__overall_n_unique_hours",
        "f__overall_n_channels",
        *[f"f__overall_{suffix}" for suffix in VALUE_SUFFIXES],
        "f__overall_daily_mean_mean",
        "f__overall_daily_mean_sd",
        "f__overall_daily_n_mean",
        "f__overall_daily_n_sd",
        *[f"f__overall_tod_{bucket}" for bucket in TIME_BUCKETS],
    ]
    for channel in CHANNELS:
        prefix = f"f__ch{channel}"
        columns.extend(
            [
                f"{prefix}_n_events",
                f"{prefix}_n_days",
                f"{prefix}_span_days",
                f"{prefix}_n_unique_hours",
                *[f"{prefix}_{suffix}" for suffix in VALUE_SUFFIXES],
                f"{prefix}_daily_mean_mean",
                f"{prefix}_daily_mean_sd",
                f"{prefix}_daily_n_mean",
                f"{prefix}_daily_n_sd",
                *[f"{prefix}_tod_{bucket}" for bucket in TIME_BUCKETS],
            ]
        )
    return columns


FEATURE_COLUMNS = feature_columns()


def value_stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {suffix: np.nan for suffix in VALUE_SUFFIXES}
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def fill_summary(
    row: dict[str, Any], prefix: str, values: np.ndarray, dates: np.ndarray,
    hours: np.ndarray, channels: np.ndarray | None = None,
) -> None:
    """Add a fixed set of value and recording summaries under a feature prefix."""
    n = len(values)
    row[f"{prefix}_n_events"] = int(n)
    if n == 0:
        row[f"{prefix}_n_days"] = 0
        row[f"{prefix}_span_days"] = 0
        row[f"{prefix}_n_unique_hours"] = 0
        for suffix in VALUE_SUFFIXES:
            row[f"{prefix}_{suffix}"] = np.nan
        row[f"{prefix}_daily_mean_mean"] = np.nan
        row[f"{prefix}_daily_mean_sd"] = np.nan
        row[f"{prefix}_daily_n_mean"] = np.nan
        row[f"{prefix}_daily_n_sd"] = np.nan
        for bucket in TIME_BUCKETS:
            row[f"{prefix}_tod_{bucket}"] = np.nan
        return

    row[f"{prefix}_n_days"] = int(np.unique(dates).size)
    row[f"{prefix}_span_days"] = int(dates.max() - dates.min() + 1)
    row[f"{prefix}_n_unique_hours"] = int(np.unique(hours).size)
    for suffix, value in value_stats(values).items():
        row[f"{prefix}_{suffix}"] = value

    daily = pd.DataFrame({"date": dates, "value": values}).groupby("date", sort=False).agg(
        daily_mean=("value", "mean"), daily_n=("value", "size")
    )
    means = daily["daily_mean"].to_numpy(dtype=float)
    ns = daily["daily_n"].to_numpy(dtype=float)
    row[f"{prefix}_daily_mean_mean"] = float(np.mean(means))
    row[f"{prefix}_daily_mean_sd"] = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
    row[f"{prefix}_daily_n_mean"] = float(np.mean(ns))
    row[f"{prefix}_daily_n_sd"] = float(np.std(ns, ddof=1)) if len(ns) > 1 else 0.0

    bucket_index = np.minimum(hours // 6, 3)
    for index, bucket in enumerate(TIME_BUCKETS):
        row[f"{prefix}_tod_{bucket}"] = float(np.mean(bucket_index == index))
    if channels is not None:
        row[f"{prefix}_n_channels"] = int(np.unique(channels).size)


def summarize_source(uid: int, source: int, frame: pd.DataFrame) -> dict[str, Any]:
    """Make one source-restricted feature row from already valid core events."""
    starts = frame["start"].to_numpy(dtype=np.int64)
    values = frame["value"].to_numpy(dtype=float)
    types = frame["type"].to_numpy(dtype=np.int64)
    tz = frame["timezone"].to_numpy(dtype=float)
    local_ms = starts.astype(np.float64) + tz * TZ_MS
    dates = np.floor(local_ms / DAY_MS).astype(np.int64)
    hours = (np.floor(local_ms / HOUR_MS).astype(np.int64)) % 24

    row: dict[str, Any] = {
        "user_id": int(uid),
        "source_id": int(source),
        "source_name": TARGET_SOURCES[source],
        "source_event_count": int(len(frame)),
        "source_first_epoch_ms": int(starts.min()),
        "source_last_epoch_ms": int(starts.max()),
        "source_tz_missing_events": int(frame["timezone_missing"].sum()),
    }
    fill_summary(row, "f__overall", values, dates, hours, types)
    for channel in CHANNELS:
        use = types == channel
        fill_summary(row, f"f__ch{channel}", values[use], dates[use], hours[use])
    for column in FEATURE_COLUMNS:
        row.setdefault(column, np.nan)
    return row


def process_file(task: tuple[int, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one raw file and return metadata plus source-specific summaries.

    The earliest valid core-HR measurement is source agnostic, as requested for
    the fixed age-at-first-epoch proxy. Model summaries are restricted to one
    of the three requested sources.
    """
    uid, path_string = task
    path = Path(path_string)
    metadata: dict[str, Any] = {
        "user_id": int(uid),
        "status": "unknown",
        "raw_events": 0,
        "valid_core_events": 0,
        "first_epoch_local_date": None,
        "first_epoch_timezone_missing": None,
        "target_events_3": 0,
        "target_events_6": 0,
        "target_events_7": 0,
        "error": None,
    }
    try:
        raw = pd.read_csv(
            path,
            usecols=lambda column: column in {
                "startTimestamp", "type", "longValue", "source", "timezoneOffset"
            },
        )
    except Exception as exc:  # a raw file can be malformed despite nonzero bytes
        metadata["status"] = "unreadable"
        metadata["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        return metadata, []

    metadata["raw_events"] = int(len(raw))
    required = {"startTimestamp", "type", "longValue"}
    if not required.issubset(raw.columns):
        metadata["status"] = "missing_required_columns"
        return metadata, []
    if raw.empty:
        metadata["status"] = "header_only"
        return metadata, []

    start = pd.to_numeric(raw["startTimestamp"], errors="coerce")
    value = pd.to_numeric(raw["longValue"], errors="coerce")
    event_type = pd.to_numeric(raw["type"], errors="coerce")
    source = (
        pd.to_numeric(raw["source"], errors="coerce")
        if "source" in raw.columns
        else pd.Series(np.nan, index=raw.index)
    )
    timezone_offset = (
        pd.to_numeric(raw["timezoneOffset"], errors="coerce")
        if "timezoneOffset" in raw.columns
        else pd.Series(np.nan, index=raw.index)
    )
    valid = (
        start.notna()
        & (start > 0)
        & value.notna()
        & value.between(HR_MIN, HR_MAX)
        & event_type.isin(CORE_CHANNEL_SET)
    )
    if not bool(valid.any()):
        metadata["status"] = "no_valid_core_event"
        return metadata, []

    valid_frame = pd.DataFrame(
        {
            "start": start.loc[valid].astype("int64"),
            "value": value.loc[valid].astype(float),
            "type": event_type.loc[valid].astype("int64"),
            "source": source.loc[valid],
            "timezone_raw": timezone_offset.loc[valid],
        }
    )
    metadata["valid_core_events"] = int(len(valid_frame))
    first_position = int(np.argmin(valid_frame["start"].to_numpy(dtype=np.int64)))
    first = valid_frame.iloc[first_position]
    first_tz_missing = bool(pd.isna(first["timezone_raw"]))
    first_tz = 0.0 if first_tz_missing else float(first["timezone_raw"])
    first_local_ms = int(first["start"]) + int(round(first_tz * TZ_MS))
    metadata["first_epoch_local_date"] = str(
        pd.Timestamp(first_local_ms, unit="ms", tz="UTC").date()
    )
    metadata["first_epoch_timezone_missing"] = first_tz_missing

    valid_frame["timezone_missing"] = valid_frame["timezone_raw"].isna()
    valid_frame["timezone"] = valid_frame["timezone_raw"].fillna(0.0).astype(float)
    valid_frame["source"] = pd.to_numeric(valid_frame["source"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for source_id in SOURCE_ORDER:
        subset = valid_frame.loc[valid_frame["source"].eq(source_id)].copy()
        metadata[f"target_events_{source_id}"] = int(len(subset))
        if len(subset):
            rows.append(summarize_source(uid, source_id, subset))
    metadata["status"] = "ok" if rows else "no_target_source_event"
    return metadata, rows


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(SALUTATION_PATH, usecols=["user_id", "salutation", "birth_date"])
    labels["salutation"] = pd.to_numeric(labels["salutation"], errors="coerce")
    labels = labels.loc[labels["salutation"].isin([10, 20])].copy()
    ensure(not labels.empty, "no 10/20 salutation labels")
    conflicts = labels.groupby("user_id")["salutation"].nunique()
    ensure(not conflicts.gt(1).any(), "one or more users have conflicting 10/20 labels")
    labels = labels.sort_values("user_id").drop_duplicates("user_id", keep="last")
    labels["birth_year"] = pd.to_numeric(labels["birth_date"], errors="coerce")
    labels["y"] = (labels["salutation"] == 20).astype(int)
    return labels[["user_id", "salutation", "birth_year", "y"]].reset_index(drop=True)


def load_bmi() -> pd.DataFrame:
    who = pd.read_csv(WHO_PATH, usecols=["user", "bmi_grp"])
    who = who.rename(columns={"user": "user_id"})
    who["bmi_group"] = who["bmi_grp"].astype("string").fillna("MISSING").astype(str)
    conflicts = who.groupby("user_id")["bmi_group"].nunique(dropna=False)
    ensure(not conflicts.gt(1).any(), "linked BMI table has conflicting values for a user")
    return who[["user_id", "bmi_group"]].drop_duplicates("user_id")


def raw_file_inventory(labels: pd.DataFrame) -> tuple[list[tuple[int, str]], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    tasks: list[tuple[int, str]] = []
    for uid in labels["user_id"].astype(int):
        path = RAW_DIR / f"{uid}.csv"
        if not path.exists():
            rows.append({"user_id": uid, "file_status": "missing", "bytes": 0})
            continue
        try:
            size = path.stat().st_size
        except OSError:
            rows.append({"user_id": uid, "file_status": "unstatable", "bytes": np.nan})
            continue
        if size <= 0:
            rows.append({"user_id": uid, "file_status": "zero_byte", "bytes": int(size)})
            continue
        rows.append({"user_id": uid, "file_status": "nonempty", "bytes": int(size)})
        tasks.append((uid, str(path)))
    return tasks, pd.DataFrame(rows)


def derive_age_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out["first_epoch_local_date"], errors="coerce")
    birth = pd.to_numeric(out["birth_year"], errors="coerce")
    proxy = dates.dt.year - birth - 1 + (dates.dt.month == 12).astype(float)
    proxy = proxy.where(proxy.between(0, 120))
    out["age_at_first_epoch_proxy"] = proxy.astype(float)
    out["age_band_5y"] = [age_band(value) for value in out["age_at_first_epoch_proxy"].to_numpy()]
    return out


def scan_raw_cohort(workers: int, rescan: bool) -> pd.DataFrame:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cohort_path = OUTPUT / "cohort_manifest.parquet"
    if cohort_path.exists() and not rescan:
        log(f"Reusing existing cohort: {cohort_path}")
        return pd.read_parquet(cohort_path)

    labels = load_labels()
    bmis = load_bmi()
    tasks, files = raw_file_inventory(labels)
    log(
        f"Raw candidate inventory: {len(labels):,} binary-labelled users; "
        f"{len(tasks):,} nonempty files. Scanning with {workers} workers."
    )
    metadata_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, task): task[0] for task in tasks}
        for future in as_completed(futures):
            uid = futures[future]
            try:
                metadata, rows = future.result()
            except Exception as exc:  # defensive parent-level worker handling
                metadata = {
                    "user_id": int(uid),
                    "status": "worker_failure",
                    "raw_events": 0,
                    "valid_core_events": 0,
                    "first_epoch_local_date": None,
                    "first_epoch_timezone_missing": None,
                    "target_events_3": 0,
                    "target_events_6": 0,
                    "target_events_7": 0,
                    "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                }
                rows = []
            metadata_rows.append(metadata)
            source_rows.extend(rows)
            done += 1
            if done % 250 == 0 or done == len(tasks):
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = done / elapsed
                eta = (len(tasks) - done) / rate / 60
                log(f"  scanned {done:,}/{len(tasks):,} files ({rate:.1f}/s; ETA {eta:.1f} min)")

    metadata = pd.DataFrame(metadata_rows)
    candidates = pd.DataFrame(source_rows)
    ensure(not candidates.empty, "no valid events from Garmin, Apple, or Samsung")
    priority = {source: index for index, source in enumerate(SOURCE_ORDER)}
    candidates["source_priority"] = candidates["source_id"].map(priority)
    winner = (
        candidates.sort_values(["user_id", "source_event_count", "source_priority"], ascending=[True, False, True])
        .drop_duplicates("user_id", keep="first")
        .drop(columns="source_priority")
        .reset_index(drop=True)
    )
    cohort = winner.merge(labels, on="user_id", how="inner", validate="one_to_one")
    cohort = cohort.merge(metadata, on="user_id", how="left", validate="one_to_one")
    cohort = cohort.merge(bmis, on="user_id", how="left", validate="one_to_one")
    cohort["bmi_group"] = cohort["bmi_group"].astype("string").fillna("MISSING").astype(str)
    cohort = derive_age_proxy(cohort)
    cohort = cohort.sort_values(["source_id", "user_id"]).reset_index(drop=True)
    ensure(cohort["user_id"].is_unique, "source assignment duplicated a participant")
    ensure(cohort["y"].isin([0, 1]).all(), "non-binary label entered cohort")
    ensure(cohort["source_id"].isin(SOURCE_ORDER).all(), "unexpected selected source")

    files.to_csv(OUTPUT / "raw_file_inventory.csv", index=False)
    metadata.sort_values("user_id").to_csv(OUTPUT / "scan_status.csv", index=False)
    cohort.to_parquet(cohort_path, index=False)
    dictionary = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "representation": ["source-restricted epoch summary"] * len(FEATURE_COLUMNS),
            "predictor": [True] * len(FEATURE_COLUMNS),
        }
    )
    dictionary.to_csv(OUTPUT / "feature_dictionary.csv", index=False)

    file_counts = files["file_status"].value_counts(dropna=False).to_dict()
    scan_counts = metadata["status"].value_counts(dropna=False).to_dict()
    flow = pd.DataFrame(
        [
            {"stage": "binary_salutation_labels", "n": int(len(labels))},
            {"stage": "nonempty_raw_files", "n": int(file_counts.get("nonempty", 0))},
            {"stage": "valid_core_epoch_events", "n": int(scan_counts.get("ok", 0) + scan_counts.get("no_target_source_event", 0))},
            {"stage": "valid_target_source_event", "n": int(scan_counts.get("ok", 0))},
            {"stage": "assigned_source_specific_cohort", "n": int(len(cohort))},
        ]
    )
    flow.to_csv(OUTPUT / "cohort_flow.csv", index=False)
    json_dump(
        OUTPUT / "scan_reproducibility.json",
        {
            "study_seed": STUDY_SEED,
            "workers": workers,
            "raw_dir": str(RAW_DIR.resolve()),
            "sources": TARGET_SOURCES,
            "valid_core_rule": {"channels": list(CHANNELS), "hr_range": [HR_MIN, HR_MAX]},
            "source_assignment": "largest valid target-source event count; ties 3,6,7",
            "age_proxy": "birth year plus earliest valid core-HR local date; Dec-1 birthday convention",
            "labels_sha256": sha256(SALUTATION_PATH),
            "bmi_sha256": sha256(WHO_PATH),
            "runner_sha256": sha256(Path(__file__)),
            "versions": {"python": sys.version, "pandas": pd.__version__, "sklearn": sklearn.__version__, "ortools": ORTOOLS_VERSION},
            "file_counts": file_counts,
            "scan_counts": scan_counts,
        },
    )
    log(f"Cohort complete: {len(cohort):,} users across {cohort['source_id'].value_counts().to_dict()}")
    return cohort


def apportion_counts(total: int, weights: Iterable[float]) -> list[int]:
    """Integer targets that sum exactly to total, using a stable largest remainder rule."""
    values = np.asarray(list(weights), dtype=float)
    ensure(len(values) > 0 and np.isfinite(values).all() and (values > 0).all(), "invalid allocation weights")
    fractions = values / values.sum()
    ideal = total * fractions
    counts = np.floor(ideal).astype(int)
    remaining = total - int(counts.sum())
    order = sorted(range(len(counts)), key=lambda index: (-(ideal[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1
    return counts.tolist()


def fold_targets(n: int, config: AllocationConfig) -> tuple[int, int]:
    train, test = apportion_counts(n, (config.train_fraction, config.test_fraction))
    return train, test


def test_fraction_ratio(config: AllocationConfig) -> tuple[int, int]:
    """A compact exact ratio used by CP-SAT's integer objective."""
    ratio = Fraction(config.test_fraction).limit_denominator(100)
    ensure(
        abs(float(ratio) - config.test_fraction) < 1e-12,
        "--fold-sizes must reduce to a test fraction with denominator at most 100; use simple ratios such as 3,1 or 80,20",
    )
    return ratio.numerator, ratio.denominator


def proportional_targets(values: pd.Series, target: int) -> dict[int, int]:
    """Apportion one exact fold target across categorical levels."""
    counts = values.value_counts()
    ensure(len(counts) > 0, "cannot apportion an empty categorical variable")
    ideal = counts.to_numpy(dtype=float) * target / len(values)
    assigned = np.floor(ideal).astype(int)
    remaining = target - int(assigned.sum())
    labels = counts.index.tolist()
    order = sorted(range(len(labels)), key=lambda index: (-(ideal[index] - assigned[index]), labels[index]))
    for index in order[:remaining]:
        assigned[index] += 1
    return {int(label): int(count) for label, count in zip(labels, assigned)}


def add_proportional_deviations(
    model: cp_model.CpModel, variables: list[Any], frame: pd.DataFrame,
    group_columns: list[str], test_numerator: int, test_denominator: int, prefix: str,
) -> list[Any]:
    """Integer distance from the requested test-fold proportion for each cell."""
    deviations: list[Any] = []
    groups = frame.groupby(group_columns, dropna=False, sort=True).indices
    for number, positions in enumerate(groups.values()):
        positions = list(map(int, positions))
        expression = sum(variables[position] for position in positions)
        deviation = model.new_int_var(0, test_denominator * len(positions), f"{prefix}_{number}")
        model.add_abs_equality(
            deviation,
            test_denominator * expression - test_numerator * len(positions),
        )
        deviations.append(deviation)
    return deviations


def objective_weights(
    n: int, test_denominator: int, n_marginal_sets: int, preserve_legacy_default: bool,
) -> tuple[int, int]:
    """Strict lexicographic weights that remain inside CP-SAT's int64 objective."""
    if preserve_legacy_default:
        return 10_000_000_000_000, 1_000_000
    joint_bound = test_denominator * n
    marginal_bound = test_denominator * n * n_marginal_sets
    tie_bound = 9 * n
    marginal_weight = tie_bound + 1
    joint_weight = marginal_weight * marginal_bound + tie_bound + 1
    maximum = joint_weight * joint_bound + marginal_weight * marginal_bound + tie_bound
    ensure(maximum < 2**62, "requested allocation ratio makes the CP-SAT objective too large")
    return joint_weight, marginal_weight


def allocate_source(
    frame: pd.DataFrame, source_id: int, config: AllocationConfig = DEFAULT_ALLOCATION_CONFIG,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use CP-SAT for a deterministic, configurable two-fold source split."""
    data = frame.sort_values("user_id").reset_index(drop=True).copy()
    n = len(data)
    ensure(n >= 4, f"source {source_id}: too few participants")
    ensure(not set(config.balance_columns) - set(data.columns), "requested balance variable is unavailable")
    train_target, test_target = fold_targets(n, config)
    ensure(train_target > 0 and test_target > 0, f"source {source_id}: requested fold sizes leave a fold empty")
    test_numerator, test_denominator = test_fraction_ratio(config)
    model = cp_model.CpModel()
    test = [model.new_bool_var(f"test_{index}") for index in range(n)]
    model.add(sum(test) == test_target)
    class_targets: dict[int, int] | None = None
    if "y" in config.balance_columns:
        class_targets = proportional_targets(data["y"], test_target)
        ensure(set(class_targets) == {0, 1}, "each source cohort needs both salutation classes")
        for label, target in class_targets.items():
            positions = np.flatnonzero(data["y"].to_numpy(dtype=int) == label).tolist()
            model.add(sum(test[position] for position in positions) == target)

    joint = add_proportional_deviations(
        model, test, data, list(config.balance_columns), test_numerator, test_denominator, "joint"
    )
    marginal = [
        deviation
        for column in config.balance_columns
        if column != "y"
        for deviation in add_proportional_deviations(
            model,
            test,
            data,
            [column],
            test_numerator,
            test_denominator,
            {"age_band_5y": "age", "bmi_group": "bmi"}.get(column, column),
        )
    ]
    random = np.random.default_rng(seed_from([STUDY_SEED, source_id, 17]))
    tie_weights = random.integers(1, 10, size=n, endpoint=False).astype(int)
    # The weights make the hierarchy strict: one joint-cell unit dominates every
    # margin term; one margin unit dominates all random tie-break terms.
    joint_weight, marginal_weight = objective_weights(
        n,
        test_denominator,
        len(config.balance_columns) - ("y" in config.balance_columns),
        config == DEFAULT_ALLOCATION_CONFIG,
    )
    objective = (
        joint_weight * sum(joint)
        + marginal_weight * sum(marginal)
        + sum(int(weight) * variable for weight, variable in zip(tie_weights, test))
    )
    model.minimize(objective)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    # CP-SAT 9.15 exposes this protobuf field as a signed 32-bit integer.
    solver.parameters.random_seed = int(seed_from([STUDY_SEED, source_id, 18]) % (2**31 - 1))
    solver.parameters.max_time_in_seconds = 300.0
    solver.parameters.log_search_progress = False
    status = solver.solve(model)
    ensure(status in (cp_model.OPTIMAL, cp_model.FEASIBLE), f"CP-SAT failed for source {source_id}: {solver.status_name(status)}")
    data["split"] = ["test" if solver.value(variable) else "train" for variable in test]
    achieved = {
        int(label): int(((data["y"] == label) & (data["split"] == "test")).sum())
        for label in (0, 1)
    }
    ensure(int((data["split"] == "test").sum()) == test_target, "total allocation target drift")
    if class_targets is not None:
        ensure(achieved == class_targets, "class allocation target drift")
    information = {
        "source_id": source_id,
        "source_name": TARGET_SOURCES[source_id],
        "n": n,
        "train_target": train_target,
        "test_target": test_target,
        "test_class_targets": class_targets,
        "test_class_achieved": achieved,
        "solver_status": solver.status_name(status),
        "objective_value": float(solver.objective_value),
        "best_objective_bound": float(solver.best_objective_bound),
        "wall_time_seconds": float(solver.wall_time),
        "num_conflicts": int(solver.num_conflicts),
        "num_branches": int(solver.num_branches),
    }
    return data, information


def allocation_balance(frame: pd.DataFrame, config: AllocationConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_id = int(frame["source_id"].iloc[0])
    families: list[tuple[list[str], str]] = [
        ([column], allocation_variable_label(column)) for column in config.balance_columns
    ]
    if len(config.balance_columns) > 1:
        families.append(
            (list(config.balance_columns), "_x_".join(allocation_variable_label(column) for column in config.balance_columns))
        )
    for columns, name in families:
        grouped = frame.groupby(columns + ["split"], dropna=False).size().rename("n").reset_index()
        totals = frame.groupby(columns, dropna=False).size().rename("total").reset_index()
        merged = grouped.merge(totals, on=columns, validate="many_to_one")
        merged["proportion"] = merged["n"] / merged["total"]
        for _, row in merged.iterrows():
            rows.append(
                {
                    "source_id": source_id,
                    "balance_family": name,
                    "cell": " | ".join(str(row[column]) for column in columns),
                    "split": str(row["split"]),
                    "n": int(row["n"]),
                    "total": int(row["total"]),
                    "proportion": float(row["proportion"]),
                }
            )
    return pd.DataFrame(rows)


def make_allocations(cohort: pd.DataFrame, config: AllocationConfig) -> pd.DataFrame:
    ensure_output_config(config)
    manifests: list[pd.DataFrame] = []
    balances: list[pd.DataFrame] = []
    information: list[dict[str, Any]] = []
    for source_id in SOURCE_ORDER:
        source = cohort.loc[cohort["source_id"].eq(source_id)].copy()
        ensure(not source.empty, f"no participants assigned to source {source_id}")
        allocated, summary = allocate_source(source, source_id, config)
        manifests.append(allocated)
        balances.append(allocation_balance(allocated, config))
        information.append(summary)
        log(
            f"Allocated {TARGET_SOURCES[source_id]}: n={summary['n']:,}, "
            f"train/test={summary['train_target']:,}/{summary['test_target']:,}, "
            f"CP-SAT={summary['solver_status']} in {summary['wall_time_seconds']:.2f}s"
        )
    manifest = pd.concat(manifests, ignore_index=True).sort_values(["source_id", "user_id"]).reset_index(drop=True)
    ensure(manifest["user_id"].is_unique, "allocation manifest duplicated a participant")
    keep = [
        "user_id", "source_id", "source_name", "salutation", "y", "birth_year",
        "first_epoch_local_date", "age_at_first_epoch_proxy", "age_band_5y", "bmi_group",
        "source_event_count", "split",
    ]
    manifest[keep].to_csv(OUTPUT / "split_manifest.csv", index=False)
    manifest[keep].to_parquet(OUTPUT / "split_manifest.parquet", index=False)
    pd.concat(balances, ignore_index=True).to_csv(OUTPUT / "split_balance.csv", index=False)
    json_dump(
        OUTPUT / "allocation_solver.json",
        {"allocation_config": config.as_dict(), "sources": information, "ortools": ORTOOLS_VERSION},
    )
    return manifest


def metric_dict(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
    }


def bootstrap_predictions(
    prediction: pd.DataFrame, source_id: int, direction_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = prediction["y_true"].to_numpy(dtype=int)
    p = prediction["probability_1"].to_numpy(dtype=float)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    ensure(len(idx0) and len(idx1), "bootstrap test set lacks a class")
    rng = np.random.default_rng(seed_from([STUDY_SEED, source_id, direction_index, 91]))
    draw_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for draw in range(N_BOOTSTRAP):
        take = np.concatenate(
            [
                rng.choice(idx0, size=len(idx0), replace=True),
                rng.choice(idx1, size=len(idx1), replace=True),
            ]
        )
        metrics = metric_dict(y[take], p[take])
        draw_rows.append({"source_id": source_id, "direction_index": direction_index, "draw": draw, **metrics})
        for position, index in enumerate(take):
            index_rows.append(
                {
                    "source_id": source_id,
                    "direction_index": direction_index,
                    "draw": draw,
                    "position": position,
                    "user_id": int(prediction["user_id"].iloc[index]),
                    "y_true": int(y[index]),
                }
            )
    return pd.DataFrame(draw_rows), pd.DataFrame(index_rows)


def fit_direction(
    source: pd.DataFrame, direction: str, direction_index: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train_label, test_label = ("train", "test") if direction_index == 0 else ("test", "train")
    train = source.loc[source["split"].eq(train_label)].sort_values("user_id").reset_index(drop=True)
    test = source.loc[source["split"].eq(test_label)].sort_values("user_id").reset_index(drop=True)
    source_id = int(source["source_id"].iloc[0])
    ensure(train["y"].nunique() == 2, f"source {source_id} {direction}: train lacks a class")
    ensure(test["y"].nunique() == 2, f"source {source_id} {direction}: test lacks a class")
    features = FEATURE_COLUMNS
    ensure(not set(features) - set(source.columns), "feature matrix is incomplete")
    keep = [column for column in features if train[column].notna().any()]
    ensure(keep, f"source {source_id} {direction}: all feature columns are missing in train")
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train = imputer.fit_transform(train[keep])
    x_test = imputer.transform(test[keep])
    seed = seed_from([STUDY_SEED, source_id, direction_index, 31])
    forest = RandomForestClassifier(random_state=seed, n_jobs=RF_WORKERS, **RF_PARAMS)
    forest.fit(x_train, train["y"].to_numpy(dtype=int))
    class_one = int(np.flatnonzero(forest.classes_ == 1)[0])
    probability = forest.predict_proba(x_test)[:, class_one]
    prediction = pd.DataFrame(
        {
            "user_id": test["user_id"].to_numpy(dtype=np.int64),
            "source_id": source_id,
            "source_name": TARGET_SOURCES[source_id],
            "direction": direction,
            "train_split": train_label,
            "test_split": test_label,
            "y_true": test["y"].to_numpy(dtype=int),
            "probability_1": probability,
            "predicted": (probability >= 0.5).astype(int),
        }
    )
    metrics = metric_dict(prediction["y_true"].to_numpy(dtype=int), probability)
    metrics.update(
        {
            "source_id": source_id,
            "source_name": TARGET_SOURCES[source_id],
            "direction": direction,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_y0": int((train["y"] == 0).sum()),
            "train_y1": int((train["y"] == 1).sum()),
            "test_y0": int((test["y"] == 0).sum()),
            "test_y1": int((test["y"] == 1).sum()),
            "rf_seed": seed,
            "n_raw_features": len(FEATURE_COLUMNS),
            "n_kept_features": len(keep),
            "n_transformed_features": int(x_train.shape[1]),
        }
    )
    draws, indices = bootstrap_predictions(prediction, source_id, direction_index)
    direction_dir = OUTPUT / "models" / f"source_{source_id}" / direction
    direction_dir.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(direction_dir / "predictions.csv", index=False)
    draws.to_csv(direction_dir / "bootstrap_metrics.csv", index=False)
    indices.to_csv(direction_dir / "bootstrap_indices.csv", index=False)
    json_dump(
        direction_dir / "feature_schema.json",
        {
            "raw_feature_columns": FEATURE_COLUMNS,
            "kept_feature_columns": keep,
            "dropped_all_missing_in_train": sorted(set(FEATURE_COLUMNS) - set(keep)),
            "n_transformed_features": int(x_train.shape[1]),
            "rf_params": forest.get_params(),
        },
    )
    joblib.dump({"imputer": imputer, "forest": forest, "source_id": source_id, "direction": direction}, direction_dir / "model.joblib")
    return prediction, metrics, draws.assign(direction=direction), indices.assign(direction=direction)


def run_models(cohort: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined = cohort.merge(manifest[["user_id", "split"]], on="user_id", how="inner", validate="one_to_one")
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    draws: list[pd.DataFrame] = []
    indices: list[pd.DataFrame] = []
    for source_id in SOURCE_ORDER:
        source = combined.loc[combined["source_id"].eq(source_id)].copy()
        for direction_index, direction in enumerate(("allocated", "flipped")):
            log(f"Fitting {TARGET_SOURCES[source_id]} {direction} direction with {RF_WORKERS} RF worker")
            prediction, result, bootstrap, resample_indices = fit_direction(source, direction, direction_index)
            predictions.append(prediction)
            metrics.append(result)
            draws.append(bootstrap)
            indices.append(resample_indices)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame = pd.DataFrame(metrics).sort_values(["source_id", "direction"]).reset_index(drop=True)
    draw_frame = pd.concat(draws, ignore_index=True)
    index_frame = pd.concat(indices, ignore_index=True)
    prediction_frame.to_csv(OUTPUT / "out_of_fold_predictions.csv", index=False)
    metric_frame.to_csv(OUTPUT / "model_metrics.csv", index=False)
    draw_frame.to_csv(OUTPUT / "bootstrap_metrics.csv", index=False)
    index_frame.to_csv(OUTPUT / "bootstrap_indices.csv", index=False)
    return prediction_frame, metric_frame, draw_frame


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def write_report() -> None:
    cohort = pd.read_parquet(cohort_path_for_read())
    allocation = pd.read_csv(OUTPUT / "split_manifest.csv")
    metrics = pd.read_csv(OUTPUT / "model_metrics.csv")
    bootstrap = pd.read_csv(OUTPUT / "bootstrap_metrics.csv")
    solver = json.loads((OUTPUT / "allocation_solver.json").read_text())
    config = loaded_allocation_config()
    fold_text = f"{config.train_fraction:.1%}/{config.test_fraction:.1%} train/test"
    balance_text = ", ".join(allocation_variable_label(column) for column in config.balance_columns)
    salutation_constraint = (
        "Salutation counts were hard-constrained proportionally within each source."
        if "y" in config.balance_columns
        else "Salutation was not hard-constrained in this allocation."
    )
    lines = [
        "# OR-Tools source-specific salutation RF",
        "",
        "## Design",
        "",
        "Three separate source-specific cohorts were built from every nonzero raw epoch file with a recorded salutation 10 or 20. Each participant was assigned once to the target source with the most valid Garmin/Apple/Samsung core-HR events; only selected-source events formed predictors. The target is recorded salutation, not biological sex or gender identity.",
        "",
        f"The fixed RF configuration was 100 trees, `max_features=0.4`, `min_samples_leaf=10`, `max_depth=10`, and `class_weight=balanced_subsample`. Hyperparameters were not retuned. OR-Tools CP-SAT made an exact source-specific {fold_text} allocation. Its balance objective used: {balance_text}. {salutation_constraint} Each allocation was evaluated in both directions; each test prediction was bootstrapped ten times within salutation class without refitting.",
        "",
        "The raw scan used five processes; CP-SAT and final RF fits used one worker each. Numeric-library thread pools were capped at one thread, so the experiment never exceeded the five-core limit.",
        "",
        "Age is a fixed age-at-first-epoch proxy: the source table stores a birth year rather than a full date, so a Dec-1 birthday convention was applied to the earliest valid core-HR measurement's local date. Five-year bands and linked BMI categories were frozen before allocation.",
        "",
        "## Cohort",
        "",
    ]
    cohort_rows = []
    for source_id in SOURCE_ORDER:
        subset = cohort.loc[cohort["source_id"].eq(source_id)]
        cohort_rows.append(
            [
                TARGET_SOURCES[source_id],
                f"{len(subset):,}",
                f"{int((subset['y'] == 0).sum()):,}",
                f"{int((subset['y'] == 1).sum()):,}",
                f"{int(subset['age_at_first_epoch_proxy'].notna().sum()):,}",
                f"{int(subset['bmi_group'].ne('MISSING').sum()):,}",
            ]
        )
    lines.extend(markdown_table(["source", "n", "sal10", "sal20", "age proxy present", "BMI present"], cohort_rows))
    lines.extend(["", "## Allocation", ""])
    allocation_rows = []
    for source_id in SOURCE_ORDER:
        source = allocation.loc[allocation["source_id"].eq(source_id)]
        for split in ("train", "test"):
            subset = source.loc[source["split"].eq(split)]
            allocation_rows.append(
                [
                    TARGET_SOURCES[source_id],
                    split,
                    f"{len(subset):,}",
                    f"{int((subset['y'] == 0).sum()):,}",
                    f"{int((subset['y'] == 1).sum()):,}",
                ]
            )
    lines.extend(markdown_table(["source", "allocation", "n", "sal10", "sal20"], allocation_rows))
    lines.extend(["", "CP-SAT diagnostics:", ""])
    diagnostic_rows = []
    for item in solver["sources"]:
        diagnostic_rows.append(
            [
                item["source_name"],
                item["solver_status"],
                f"{item['wall_time_seconds']:.3f}",
                str(item["num_conflicts"]),
                str(item["num_branches"]),
            ]
        )
    lines.extend(markdown_table(["source", "status", "seconds", "conflicts", "branches"], diagnostic_rows))
    lines.extend(["", "## Held-out results", ""])
    result_rows = []
    for _, row in metrics.iterrows():
        result_rows.append(
            [
                str(row["source_name"]),
                str(row["direction"]),
                f"{int(row['n_train']):,}",
                f"{int(row['n_test']):,}",
                f"{row['auroc']:.4f}",
                f"{row['accuracy']:.4f}",
                f"{row['balanced_accuracy']:.4f}",
            ]
        )
    lines.extend(markdown_table(["source", "direction", "train", "test", "AUROC", "accuracy", "balanced accuracy"], result_rows))
    lines.extend(["", "## Ten-draw bootstrap summaries", ""])
    bootstrap_rows = []
    for (source_id, direction), group in bootstrap.groupby(["source_id", "direction"], sort=True):
        for metric in ("auroc", "accuracy", "balanced_accuracy"):
            value = group[metric].to_numpy(dtype=float)
            bootstrap_rows.append(
                [
                    TARGET_SOURCES[int(source_id)],
                    str(direction),
                    metric,
                    f"{np.mean(value):.4f}",
                    f"{np.std(value, ddof=1):.4f}",
                    f"[{np.quantile(value, .025):.4f}, {np.quantile(value, .975):.4f}]",
                ]
            )
    lines.extend(markdown_table(["source", "direction", "metric", "mean", "SD", "2.5--97.5%"], bootstrap_rows))
    lines.extend(
        [
            "",
            "## Limits",
            "",
            f"These results condition on this raw export, the source assignment rule, the frozen {fold_text} allocation, and the chosen RF configuration. The ten bootstrap draws quantify resampling variation of saved test predictions, not tuning, allocation, cohort-selection, or transport uncertainty. Source-specific results are descriptive and should not be interpreted as vendor rankings or physiological effects.",
            "",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines))


def verify_outputs() -> None:
    cohort = pd.read_parquet(cohort_path_for_read())
    allocation = pd.read_csv(OUTPUT / "split_manifest.csv")
    predictions = pd.read_csv(OUTPUT / "out_of_fold_predictions.csv")
    bootstrap = pd.read_csv(OUTPUT / "bootstrap_metrics.csv")
    indices = pd.read_csv(OUTPUT / "bootstrap_indices.csv")
    ensure(cohort["user_id"].is_unique, "cohort has duplicate users")
    ensure(allocation["user_id"].is_unique, "allocation has duplicate users")
    ensure(set(cohort["user_id"]) == set(allocation["user_id"]), "cohort/allocation IDs differ")
    ensure(set(allocation["split"]) == {"train", "test"}, "illegal allocation split")
    config = loaded_allocation_config()
    for source_id in SOURCE_ORDER:
        allocated = allocation.loc[allocation["source_id"].eq(source_id)]
        n_train = int((allocated["split"] == "train").sum())
        n_test = int((allocated["split"] == "test").sum())
        expected_train, expected_test = fold_targets(len(allocated), config)
        ensure((n_train, n_test) == (expected_train, expected_test), f"source {source_id}: fold-size drift")
        if "y" in config.balance_columns:
            expected = proportional_targets(allocated["y"], n_test)
            got = {label: int(((allocated["split"] == "test") & (allocated["y"] == label)).sum()) for label in (0, 1)}
            ensure(got == expected, f"source {source_id}: class balance drift")
        source_prediction = predictions.loc[predictions["source_id"].eq(source_id)]
        ensure(source_prediction["user_id"].value_counts().eq(1).all(), f"source {source_id}: OOF duplication")
        ensure(set(source_prediction["user_id"]) == set(allocated["user_id"]), f"source {source_id}: OOF coverage drift")
        for direction in ("allocated", "flipped"):
            pred = source_prediction.loc[source_prediction["direction"].eq(direction)]
            ensure(len(pred) == (n_test if direction == "allocated" else n_train), f"source {source_id}: direction size drift")
            b = bootstrap.loc[(bootstrap["source_id"] == source_id) & (bootstrap["direction"] == direction)]
            ensure(len(b) == N_BOOTSTRAP, f"source {source_id}: bootstrap draw count drift")
            i = indices.loc[(indices["source_id"] == source_id) & (indices["direction"] == direction)]
            ensure(i["draw"].nunique() == N_BOOTSTRAP, f"source {source_id}: bootstrap indices drift")
            class_sizes = pred["y_true"].value_counts().to_dict()
            for draw, group in i.groupby("draw"):
                got_sizes = group["y_true"].value_counts().to_dict()
                ensure(got_sizes == class_sizes, f"source {source_id}: bootstrap class size drift draw={draw}")
    json_dump(OUTPUT / "verification.json", {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat()})
    log("Verification passed: allocations, flip coverage, predictions, and bootstrap draws.")


def self_test(config: AllocationConfig) -> None:
    """Small deterministic CP-SAT check without raw data access."""
    rng = np.random.default_rng(STUDY_SEED)
    rows = []
    for index in range(101):
        rows.append(
            {
                "user_id": index,
                "source_id": 3,
                "y": int(index % 3 != 0),
                "age_band_5y": f"{20 + 5 * (index % 8)}-{24 + 5 * (index % 8)}",
                "bmi_group": ("normal", "overweight", "obese")[index % 3],
                "noise": float(rng.normal()),
            }
        )
    data = pd.DataFrame(rows)
    first, first_info = allocate_source(data, 3, config)
    second, second_info = allocate_source(data, 3, config)
    ensure(first["split"].equals(second["split"]), "CP-SAT allocation is not reproducible")
    expected_train, expected_test = fold_targets(len(data), config)
    got_train = int((first["split"] == "train").sum())
    got_test = int((first["split"] == "test").sum())
    ensure((got_train, got_test) == (expected_train, expected_test), "fold-size target drift")
    if "y" in config.balance_columns:
        expected = proportional_targets(data["y"], expected_test)
        ensure(first_info["test_class_achieved"] == expected, "class target drift")
        ensure(second_info["test_class_achieved"] == expected, "repeated class target drift")
    log("Self-test passed: configurable OR-Tools allocation is deterministic and target-balanced.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("all", "scan", "allocate", "model", "report", "verify", "self-test"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="raw-scan workers (1--5)")
    parser.add_argument(
        "--balance-vars",
        default="salutation,age_band_5y,bmi_group",
        help="comma-separated allocation variables: salutation, age_band_5y, bmi_group",
    )
    parser.add_argument(
        "--fold-sizes",
        default="0.5,0.5",
        help="relative train,test sizes; e.g. 0.5,0.5 or 3,1",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="relative result directory; use a new one for a non-default allocation",
    )
    parser.add_argument("--rescan", action="store_true", help="rebuild raw cohort even if saved")
    args = parser.parse_args()
    ensure(1 <= args.workers <= MAX_WORKERS, f"workers must be 1--{MAX_WORKERS}")
    config = parse_allocation_config(args.balance_vars, args.fold_sizes)
    if args.mode == "self-test":
        self_test(config)
        return
    global OUTPUT
    OUTPUT = configure_output(args.output_dir)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.mode in ("all", "scan"):
        cohort = scan_raw_cohort(args.workers, args.rescan)
    else:
        cohort = pd.read_parquet(cohort_path_for_read())
    if args.mode == "scan":
        return
    if args.mode in ("all", "allocate"):
        manifest = make_allocations(cohort, config)
    else:
        ensure_output_config(config)
        manifest = pd.read_parquet(OUTPUT / "split_manifest.parquet")
    if args.mode == "allocate":
        return
    if args.mode in ("all", "model"):
        run_models(cohort, manifest)
    if args.mode == "model":
        return
    if args.mode in ("all", "report"):
        write_report()
    if args.mode == "report":
        return
    if args.mode in ("all", "verify"):
        verify_outputs()


if __name__ == "__main__":
    main()
