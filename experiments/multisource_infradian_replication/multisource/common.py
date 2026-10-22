"""Shared constants, paths, and protocol primitives.

The window/seed/age primitives are imported from the historical
``infradian_age_dependency`` implementation so that the two experiments use one
definition of ``expanded90``, ``balanced90``, the block sampler, and the age
convention.  Nothing in this module writes to the historical experiment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
UPSTREAM_EXP = REPO_ROOT / "experiments/infradian_age_dependency"
UPSTREAM_SCRIPTS = UPSTREAM_EXP / "scripts"
UPSTREAM_PLAN = UPSTREAM_EXP / "PLAN.md"

ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
DEFAULT_RUN_ROOT = ARTIFACTS_DIR / "runs"

DEMOGRAPHICS_CSV = REPO_ROOT / "data/13Aug_1222.csv"
DAILY_PARQUET = UPSTREAM_EXP / "rhr_sensitivity/supplier_daily_normalized.parquet"
SOURCE_MAPPING_CSV = (
    REPO_ROOT.parent / "datenspende_epoch2daily/mapping/pg_daily_sources.csv"
)
VALUE_TYPE_MAPPING_CSV = (
    REPO_ROOT.parent / "datenspende_epoch2daily/mapping/pg_daily_value_types.csv"
)
EPOCH_SOURCES_PARQUET = REPO_ROOT / "experiments/artifacts_v2/epoch_sources.parquet"
RAW_OUT_DIR = REPO_ROOT / "out"

# Sources explicitly in scope for the expansion (PLAN.md, objective).
SCOPE_SOURCES: tuple[int, ...] = (2, 3, 6, 7, 9, 13, 19)
# Polar is excluded by scope.  Unknown IDs stay inventory-only.
EXCLUDED_SOURCES: tuple[int, ...] = (4,)
UNRESOLVED_SOURCES: tuple[int, ...] = (38, 46, 48)

SOURCE_NAMES: dict[int, str] = {
    2: "Fitbit",
    3: "Garmin",
    4: "Polar",
    6: "Apple",
    7: "Samsung",
    9: "Nokia",
    13: "Google Fit",
    19: "Oura",
    38: "Unknown-38",
    46: "Unknown-46",
    48: "Unknown-48",
}

# Upstream pg_daily value-type namespace.  Analysis labels are implementation
# names; the exported type IDs are the authoritative mapping.
DAILY_ENDPOINTS: dict[str, int] = {"RHR": 65, "HR_during_sleep": 66}
ENDPOINT_TYPE_IDS: dict[int, str] = {v: k for k, v in DAILY_ENDPOINTS.items()}

PROTOCOLS: tuple[str, ...] = ("expanded90", "balanced90")
PROTOCOL_QUOTAS: dict[str, int] = {"expanded90": 90, "balanced90": 90}

# Daily-arm value plausibility convention.  This is a QC convention, not a
# vendor-validated rule (PLAN.md, provenance gate).
DAILY_VALUE_MIN = 25.0
DAILY_VALUE_MAX = 230.0

# Dense channel-3000 conventions, identical to the historical experiment.
DENSE_CHANNEL = 3000
DENSE_MIN_BINS = 120
DENSE_BINS_PER_DAY = 288
DENSE_VALUE_MIN = 25.0
DENSE_VALUE_MAX = 230.0
HRP10_PERCENTILE = 10.0

# Age conventions.
AGE_PLAUSIBLE_MIN = 0
AGE_PLAUSIBLE_MAX = 120
AGE_GRID_MIN = 18
AGE_GRID_MAX = 80
AGE_BAND_WIDTH = 5
MIN_FEMALE_PER_BAND = 30
MIN_MALE_PER_BAND = 30
MIN_SHARED_BANDS = 3

# Inference.
B_NULL_REPLICATES = 30
N_BOOTSTRAP = 2000
SEED_TAG = "20261020|infradian_age"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_upstream_common = _load_module("_ms_upstream_common", UPSTREAM_SCRIPTS / "common.py")

# Re-exported protocol primitives (single source of truth).
compute_all_ages = _upstream_common.compute_all_ages
calendar_date_from_day = _upstream_common.calendar_date_from_day
rcs_stone_koo = _upstream_common.rcs_stone_koo
DEFAULT_KNOTS = _upstream_common.DEFAULT_KNOTS
greedy_quota_allocation = _upstream_common.greedy_quota_allocation
derive_block_seed = _upstream_common.derive_block_seed
sample_block_dates = _upstream_common.sample_block_dates
compute_selected_dates_hash = _upstream_common.compute_selected_dates_hash
find_qualifying_windows = _upstream_common.find_qualifying_windows
get_window_block_capacities = _upstream_common.get_window_block_capacities


_ENDPOINT_ENGINE = None


def load_endpoint_engine():
    """Load the historical endpoint engine once per process."""
    global _ENDPOINT_ENGINE
    if _ENDPOINT_ENGINE is None:
        _ENDPOINT_ENGINE = _load_module(
            "_ms_upstream_endpoint_engine",
            UPSTREAM_SCRIPTS / "02_compute_endpoints_and_nulls.py",
        )
    return _ENDPOINT_ENGINE


def source_name(source: int) -> str:
    return SOURCE_NAMES.get(int(source), f"Source {int(source)}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(values: Iterable[int]) -> str:
    arr = np.asarray(list(values), dtype="<i8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def utc_now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def ensure_run_dir(run_dir: Path, overwrite: bool = False) -> Path:
    """Create a run namespace; overwrite permits reuse but does not clear it.

    A later ``--skip-channel`` run can therefore reuse an old
    ``cache_index.parquet`` through manifest building. Use a fresh namespace
    when the requested run scope changes.
    """
    run_dir = Path(run_dir)
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose a new namespace."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_source_mapping(path: Path = SOURCE_MAPPING_CSV) -> pd.DataFrame:
    """Read the upstream pg_daily source namespace.

    The file is semicolon separated with a UTF-8 BOM.  Numeric IDs are only
    meaningful inside this namespace; do not substitute the Thryve mapping.
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype(int)
    df["code"] = df["code"].astype(str).str.strip()
    return df.sort_values("id").reset_index(drop=True)


def load_value_type_mapping(path: Path = VALUE_TYPE_MAPPING_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype(int)
    df["code"] = df["code"].astype(str).str.strip()
    return df.sort_values("id").reset_index(drop=True)


def audit_demographics(path: Path = DEMOGRAPHICS_CSV) -> pd.DataFrame:
    """Count administrative demographic exclusions without inferring values."""
    raw = pd.read_csv(path)
    total = len(raw)
    uid = pd.to_numeric(raw["user_id"], errors="coerce")
    dup_rows = int(uid.duplicated(keep=False).sum())
    sal = pd.to_numeric(raw["salutation"], errors="coerce")
    birth = raw["birth_date"].astype("string").str.strip()
    year = pd.to_numeric(birth.str.extract(r"^(\d{4})", expand=False), errors="coerce")
    current_year = int(pd.Timestamp.now().year)
    valid_birth = year.between(1900, current_year, inclusive="both")
    valid_sal = sal.isin([10, 20])
    # Duplicate user IDs are excluded entirely by load_demographics.
    unique_ok = ~uid.duplicated(keep=False)
    eligible = unique_ok & valid_sal & valid_birth
    rows = [
        {"item": "raw_rows", "count": total},
        {"item": "duplicate_user_id_rows", "count": dup_rows},
        {"item": "missing_or_nonfinite_user_id", "count": int(uid.isna().sum())},
        {"item": "missing_salutation", "count": int(sal.isna().sum())},
        {"item": "salutation_10", "count": int(sal.eq(10).sum())},
        {"item": "salutation_20", "count": int(sal.eq(20).sum())},
        {"item": "other_salutation", "count": int((sal.notna() & ~valid_sal).sum())},
        {"item": "missing_birth_year", "count": int(year.isna().sum())},
        {"item": "birth_year_outside_1900_current", "count": int((year.notna() & ~valid_birth).sum())},
        {"item": "eligible_unique_users", "count": int(eligible.sum())},
    ]
    return pd.DataFrame(rows)


def load_demographics(path: Path = DEMOGRAPHICS_CSV) -> pd.DataFrame:
    """Load participant demographics and derive birth year/age columns.

    ``birth_date`` stores a birth year in the current export.  The parser also
    accepts a full date string and takes its year, but never infers missing
    values.
    """
    raw = pd.read_csv(path)
    required = {"user_id", "salutation", "birth_date"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Demographics table lacks columns: {sorted(missing)}")
    out = pd.DataFrame({"user_id": pd.to_numeric(raw["user_id"], errors="raise").astype("int64")})
    out["salutation"] = pd.to_numeric(raw["salutation"], errors="coerce")
    birth = raw["birth_date"].astype("string").str.strip()
    year = pd.to_numeric(birth.str.extract(r"^(\d{4})", expand=False), errors="coerce")
    out["birth_year"] = year
    out["weight"] = pd.to_numeric(raw.get("weight"), errors="coerce") if "weight" in raw else np.nan
    out["height"] = pd.to_numeric(raw.get("height"), errors="coerce") if "height" in raw else np.nan
    out["bmi"] = np.where(
        out["height"].gt(0) & out["weight"].notna(),
        out["weight"] / (out["height"] / 100.0) ** 2,
        np.nan,
    )
    out = out.drop_duplicates("user_id", keep=False)
    current_year = int(pd.Timestamp.now().year)
    out = out.loc[
        out["birth_year"].between(1900, current_year, inclusive="both")
        & out["salutation"].isin([10, 20])
    ].copy()
    out["birth_year"] = out["birth_year"].astype("int64")
    out["salutation"] = out["salutation"].astype("int64")
    return out.reset_index(drop=True)


def attach_ages(records: pd.DataFrame, demographics: pd.DataFrame, start_col: str = "start_day") -> pd.DataFrame:
    """Join demographics and derive the three age conventions at window start."""
    out = records.merge(demographics, on="user_id", how="left", validate="many_to_one")
    ages = [
        compute_all_ages(int(day), int(year))
        if pd.notna(day) and pd.notna(year)
        else (np.nan, np.nan, np.nan)
        for day, year in zip(out[start_col], out["birth_year"])
    ]
    out[["age_dec1", "age_jan1", "age_dec31"]] = pd.DataFrame(ages, index=out.index)
    out["age"] = out["age_dec1"]
    out["sex"] = np.where(out["salutation"].eq(10), 1, np.where(out["salutation"].eq(20), 0, np.nan))
    out.loc[~out["age"].between(AGE_PLAUSIBLE_MIN, AGE_PLAUSIBLE_MAX), "age"] = np.nan
    return out
