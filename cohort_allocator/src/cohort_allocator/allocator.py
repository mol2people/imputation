"""Generic participant-level fold allocation.

The allocator expects one row per participant. It preserves every input column
and adds one fold-label column. Selected split variables are handled as
categorical values, including an explicit missing-value category.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from ortools import __version__ as ORTOOLS_VERSION
from ortools.sat.python import cp_model


PACKAGE_VERSION = "0.1.0"
_PRIVATE_PREFIX = "__cohort_allocator_"
_FOLD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class AllocationError(ValueError):
    """Raised when an allocation request or its input table is invalid."""


@dataclass(frozen=True)
class AllocationConfig:
    """All user-controlled allocation choices."""

    id_column: str
    stratify_columns: tuple[str, ...]
    fold_names: tuple[str, ...]
    fold_weights: tuple[float, ...]
    partition_columns: tuple[str, ...] = ()
    fold_column: str = "fold"
    seed: int = 20260921
    time_limit_seconds: float = 300.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id_column": self.id_column,
            "stratify_columns": list(self.stratify_columns),
            "partition_columns": list(self.partition_columns),
            "fold_column": self.fold_column,
            "fold_sizes": dict(zip(self.fold_names, self.fold_weights, strict=True)),
            "seed": self.seed,
            "time_limit_seconds": self.time_limit_seconds,
            "allocation_method": (
                "exact partition-level fold counts; CP-SAT minimizes selected "
                "joint-cell imbalance, then selected marginal imbalance, then "
                "a seeded tie-break"
            ),
        }


@dataclass
class AllocationResult:
    """Allocated participant table and its auditable solver diagnostics."""

    assignments: pd.DataFrame
    balance: pd.DataFrame
    summary: dict[str, Any]
    config: AllocationConfig


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AllocationError(message)


def _normalise_columns(values: Sequence[str], argument_name: str) -> tuple[str, ...]:
    columns = tuple(str(value).strip() for value in values if str(value).strip())
    _ensure(columns, f"{argument_name} needs at least one column")
    _ensure(
        len(set(columns)) == len(columns),
        f"{argument_name} contains duplicate columns",
    )
    return columns


def _normalise_fold_sizes(
    fold_sizes: Mapping[str, float] | Sequence[tuple[str, float]],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    pairs = list(fold_sizes.items()) if isinstance(fold_sizes, Mapping) else list(fold_sizes)
    _ensure(len(pairs) >= 2, "at least two folds are required")
    names = tuple(str(name).strip() for name, _ in pairs)
    _ensure(all(_FOLD_NAME.fullmatch(name) for name in names), "fold names may contain only letters, digits, '.', '_', and '-'")
    _ensure(len(set(names)) == len(names), "fold names must be unique")
    try:
        weights = tuple(float(weight) for _, weight in pairs)
    except (TypeError, ValueError) as exc:
        raise AllocationError("fold sizes must be numeric") from exc
    _ensure(
        all(math.isfinite(weight) and weight > 0 for weight in weights),
        "fold sizes must be finite and positive",
    )
    total = sum(weights)
    return names, tuple(weight / total for weight in weights)


def make_config(
    *,
    id_column: str,
    stratify_columns: Sequence[str],
    fold_sizes: Mapping[str, float] | Sequence[tuple[str, float]],
    partition_columns: Sequence[str] = (),
    fold_column: str = "fold",
    seed: int = 20260921,
    time_limit_seconds: float = 300.0,
) -> AllocationConfig:
    """Construct and validate a reusable allocation configuration."""
    names, weights = _normalise_fold_sizes(fold_sizes)
    stratify = _normalise_columns(stratify_columns, "stratify_columns")
    partitions = tuple(str(value).strip() for value in partition_columns if str(value).strip())
    _ensure(
        len(set(partitions)) == len(partitions),
        "partition_columns contains duplicate columns",
    )
    _ensure(str(id_column).strip(), "id_column is required")
    _ensure(str(fold_column).strip(), "fold_column is required")
    _ensure(math.isfinite(float(time_limit_seconds)) and float(time_limit_seconds) > 0, "time_limit_seconds must be positive")
    return AllocationConfig(
        id_column=str(id_column).strip(),
        stratify_columns=stratify,
        fold_names=names,
        fold_weights=weights,
        partition_columns=partitions,
        fold_column=str(fold_column).strip(),
        seed=int(seed),
        time_limit_seconds=float(time_limit_seconds),
    )


def fold_target_counts(total: int, weights: Sequence[float]) -> list[int]:
    """Return exact integer fold targets using stable largest-remainder rounding."""
    _ensure(total >= 0, "total must be non-negative")
    values = np.asarray(weights, dtype=float)
    _ensure(len(values) >= 2 and np.isfinite(values).all() and (values > 0).all(), "invalid fold sizes")
    values = values / values.sum()
    ideal = total * values
    counts = np.floor(ideal).astype(int)
    remaining = total - int(counts.sum())
    order = sorted(range(len(counts)), key=lambda index: (-(ideal[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1
    return counts.tolist()


def _stable_identifier_key(value: Any) -> str:
    """A reproducible sort key that does not coerce integer and string IDs together."""
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _display_value(value: Any) -> str:
    try:
        if bool(pd.isna(value)):
            return "<MISSING>"
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _encode_column(frame: pd.DataFrame, column: str, private_column: str) -> dict[int, str]:
    codes, levels = pd.factorize(frame[column], sort=False, use_na_sentinel=True)
    missing_code = len(levels)
    codes = np.asarray(codes, dtype=int)
    codes[codes < 0] = missing_code
    frame[private_column] = codes
    labels = {int(index): _display_value(value) for index, value in enumerate(levels)}
    if (codes == missing_code).any():
        labels[missing_code] = "<MISSING>"
    return labels


def _groups_from_codes(frame: pd.DataFrame, columns: Sequence[str]) -> dict[tuple[int, ...], list[int]]:
    if not columns:
        return {(): list(range(len(frame)))}
    groups: dict[tuple[int, ...], list[int]] = {}
    for position, values in enumerate(frame.loc[:, list(columns)].itertuples(index=False, name=None)):
        key = tuple(int(value) for value in values)
        groups.setdefault(key, []).append(position)
    return groups


def _derived_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return max(1, value % (2**31 - 1))


def _add_proportional_deviations(
    model: cp_model.CpModel,
    assignment: list[list[Any]],
    groups: Mapping[tuple[int, ...], Sequence[int]],
    targets: Sequence[int],
    total: int,
    prefix: str,
) -> list[Any]:
    """Add integer absolute deviations from each cell's requested fold share."""
    deviations: list[Any] = []
    for cell_number, positions in enumerate(groups.values()):
        cell_size = len(positions)
        for fold_number, target in enumerate(targets):
            assigned = sum(assignment[position][fold_number] for position in positions)
            deviation = model.new_int_var(
                0,
                total * cell_size,
                f"{prefix}_{cell_number}_{fold_number}",
            )
            model.add_abs_equality(
                deviation,
                total * assigned - target * cell_size,
            )
            deviations.append(deviation)
    return deviations


def _solve_phase(
    model: cp_model.CpModel,
    objective: Any,
    *,
    seed: int,
    time_limit_seconds: float,
    phase: str,
) -> tuple[cp_model.CpSolver, dict[str, Any]]:
    model.minimize(objective)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = False
    status = solver.solve(model)
    status_name = solver.status_name(status)
    _ensure(
        status in (cp_model.OPTIMAL, cp_model.FEASIBLE),
        f"CP-SAT did not find a {phase} allocation: {status_name}",
    )
    return solver, {
        "phase": phase,
        "status": status_name,
        "objective_value": int(round(solver.objective_value)),
        "best_objective_bound": float(solver.best_objective_bound),
        "wall_time_seconds": float(solver.wall_time),
        "num_conflicts": int(solver.num_conflicts),
        "num_branches": int(solver.num_branches),
    }


def _allocate_partition(
    frame: pd.DataFrame,
    *,
    config: AllocationConfig,
    stratify_codes: Sequence[str],
    partition_number: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    total = len(frame)
    _ensure(total > 0, "cannot allocate an empty partition")
    targets = fold_target_counts(total, config.fold_weights)
    n_folds = len(config.fold_names)
    model = cp_model.CpModel()
    assignment = [
        [model.new_bool_var(f"assign_{row}_{fold}") for fold in range(n_folds)]
        for row in range(total)
    ]
    for row in range(total):
        model.add_exactly_one(assignment[row])
    for fold, target in enumerate(targets):
        model.add(sum(assignment[row][fold] for row in range(total)) == target)

    joint_groups = _groups_from_codes(frame, stratify_codes)
    joint_deviations = _add_proportional_deviations(
        model,
        assignment,
        joint_groups,
        targets,
        total,
        "joint",
    )
    marginal_deviations = (
        [
            deviation
            for column in stratify_codes
            for deviation in _add_proportional_deviations(
                model,
                assignment,
                _groups_from_codes(frame, [column]),
                targets,
                total,
                f"margin_{column}",
            )
        ]
        if len(stratify_codes) > 1
        else []
    )

    # Each phase fixes the preceding objective at its achieved integer value.
    # This is lexicographic without unsafe, very large objective weights.
    phase_count = 3 if marginal_deviations else 2
    phase_limit = config.time_limit_seconds / phase_count
    phase_records: list[dict[str, Any]] = []

    joint_total = sum(joint_deviations)
    solver, record = _solve_phase(
        model,
        joint_total,
        seed=_derived_seed(config.seed, partition_number, "joint"),
        time_limit_seconds=phase_limit,
        phase="joint_balance",
    )
    phase_records.append(record)
    model.add(joint_total == record["objective_value"])

    if marginal_deviations:
        marginal_total = sum(marginal_deviations)
        solver, record = _solve_phase(
            model,
            marginal_total,
            seed=_derived_seed(config.seed, partition_number, "marginal"),
            time_limit_seconds=phase_limit,
            phase="marginal_balance",
        )
        phase_records.append(record)
        model.add(marginal_total == record["objective_value"])

    random = np.random.default_rng(_derived_seed(config.seed, partition_number, "tie"))
    tie_weights = random.integers(1, 10, size=(total, n_folds), endpoint=False)
    tie_total = sum(
        int(tie_weights[row, fold]) * assignment[row][fold]
        for row in range(total)
        for fold in range(n_folds)
    )
    solver, record = _solve_phase(
        model,
        tie_total,
        seed=_derived_seed(config.seed, partition_number, "tie"),
        time_limit_seconds=phase_limit,
        phase="seeded_tie_break",
    )
    phase_records.append(record)

    fold_indices = np.empty(total, dtype=int)
    for row in range(total):
        selected = [fold for fold in range(n_folds) if solver.value(assignment[row][fold])]
        _ensure(len(selected) == 1, "solver returned an invalid participant assignment")
        fold_indices[row] = selected[0]
    achieved = np.bincount(fold_indices, minlength=n_folds).tolist()
    _ensure(achieved == targets, "fold-size target drift")
    return (
        np.asarray([config.fold_names[index] for index in fold_indices], dtype=object),
        {
            "n": total,
            "fold_targets": dict(zip(config.fold_names, targets, strict=True)),
            "fold_achieved": dict(zip(config.fold_names, achieved, strict=True)),
            "joint_cell_count": len(joint_groups),
            "phase_results": phase_records,
            "all_phases_optimal": all(item["status"] == "OPTIMAL" for item in phase_records),
        },
    )


def _partition_values(
    frame: pd.DataFrame,
    positions: Sequence[int],
    partition_columns: Sequence[str],
) -> dict[str, str]:
    if not partition_columns:
        return {}
    first = frame.iloc[positions[0]]
    return {column: _display_value(first[column]) for column in partition_columns}


def _balance_table(
    frame: pd.DataFrame,
    *,
    config: AllocationConfig,
    stratify_codes: Sequence[str],
    partition_codes: Sequence[str],
    labels: Mapping[str, Mapping[int, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    families: list[tuple[str, Sequence[str]]] = [
        ("joint", stratify_codes),
    ]
    families.extend(
        (column, [code])
        for column, code in zip(config.stratify_columns, stratify_codes, strict=True)
    )
    partition_groups = _groups_from_codes(frame, partition_codes)
    for _, positions in partition_groups.items():
        partition_frame = frame.iloc[positions].reset_index(drop=True)
        partition = _partition_values(frame, positions, config.partition_columns)
        for family, code_columns in families:
            for cell_key, cell_positions in _groups_from_codes(partition_frame, code_columns).items():
                raw_columns = (
                    config.stratify_columns
                    if family == "joint"
                    else (family,)
                )
                cell = {
                    raw: labels[code][int(value)]
                    for raw, code, value in zip(raw_columns, code_columns, cell_key, strict=True)
                }
                total = len(cell_positions)
                fold_counts = partition_frame.iloc[cell_positions][config.fold_column].value_counts()
                for fold in config.fold_names:
                    n = int(fold_counts.get(fold, 0))
                    rows.append(
                        {
                            "partition": json.dumps(partition, sort_keys=True),
                            "balance_family": family,
                            "cell": json.dumps(cell, sort_keys=True),
                            "fold": fold,
                            "n": n,
                            "total": total,
                            "proportion": n / total,
                        }
                    )
    return pd.DataFrame(rows)


def allocate_cohort(
    demographics: pd.DataFrame,
    config: AllocationConfig,
) -> AllocationResult:
    """Allocate all participants and return fold-labelled cohort tables.

    Allocation occurs independently within every unique partition when
    partition_columns is nonempty. The selected stratify columns must already
    be appropriate categorical variables; pre-bin continuous measures first.
    """
    _ensure(not demographics.empty, "demographic summary is empty")
    _ensure(
        not any(str(column).startswith(_PRIVATE_PREFIX) for column in demographics.columns),
        f"input columns may not begin with {_PRIVATE_PREFIX!r}",
    )
    required = {
        config.id_column,
        *config.stratify_columns,
        *config.partition_columns,
    }
    missing = sorted(required - set(demographics.columns))
    _ensure(not missing, "demographic summary lacks required column(s): " + ", ".join(missing))
    _ensure(
        config.fold_column not in demographics.columns,
        f"input already has {config.fold_column!r}; choose another fold_column",
    )
    _ensure(not demographics[config.id_column].isna().any(), f"{config.id_column!r} contains missing participant IDs")
    _ensure(
        not demographics[config.id_column].duplicated().any(),
        f"{config.id_column!r} must identify exactly one row per participant",
    )

    original_columns = list(demographics.columns)
    work = demographics.copy()
    id_key = work[config.id_column].map(_stable_identifier_key)
    _ensure(not id_key.duplicated().any(), f"{config.id_column!r} has ambiguous textual/type identifiers")
    work[_PRIVATE_PREFIX + "id_key"] = id_key
    work = work.sort_values(_PRIVATE_PREFIX + "id_key", kind="stable").reset_index(drop=True)

    stratify_codes: list[str] = []
    partition_codes: list[str] = []
    labels: dict[str, Mapping[int, str]] = {}
    for number, column in enumerate(config.stratify_columns):
        code = f"{_PRIVATE_PREFIX}stratify_{number}"
        labels[code] = _encode_column(work, column, code)
        stratify_codes.append(code)
    for number, column in enumerate(config.partition_columns):
        code = f"{_PRIVATE_PREFIX}partition_{number}"
        _encode_column(work, column, code)
        partition_codes.append(code)

    work[config.fold_column] = pd.Series(pd.NA, index=work.index, dtype="string")
    partition_summaries: list[dict[str, Any]] = []
    partition_groups = _groups_from_codes(work, partition_codes)
    for partition_number, (_, positions) in enumerate(partition_groups.items()):
        partition_frame = work.iloc[positions].reset_index(drop=True)
        folds, information = _allocate_partition(
            partition_frame,
            config=config,
            stratify_codes=stratify_codes,
            partition_number=partition_number,
        )
        work.loc[positions, config.fold_column] = folds
        information["partition"] = _partition_values(work, positions, config.partition_columns)
        partition_summaries.append(information)

    _ensure(work[config.fold_column].notna().all(), "some participants were not allocated")
    balance = _balance_table(
        work,
        config=config,
        stratify_codes=stratify_codes,
        partition_codes=partition_codes,
        labels=labels,
    )
    assignments = work.loc[:, [*original_columns, config.fold_column]].copy()
    assignments = assignments.reset_index(drop=True)
    summary = {
        "package": {"name": "cohort-allocator", "version": PACKAGE_VERSION},
        "ortools_version": ORTOOLS_VERSION,
        "n_participants": int(len(assignments)),
        "config": config.as_dict(),
        "partitions": partition_summaries,
        "allocation_complete": True,
    }
    return AllocationResult(
        assignments=assignments,
        balance=balance,
        summary=summary,
        config=config,
    )


def read_demographic_summary(path: str | Path) -> pd.DataFrame:
    """Read a participant-level CSV or Parquet demographic summary."""
    source = Path(path)
    _ensure(source.is_file(), f"input file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(source)
        except ImportError as exc:
            raise AllocationError(
                "Parquet input requires pyarrow; install cohort-allocator[parquet]"
            ) from exc
    raise AllocationError("input must be a .csv, .parquet, or .pq file")


def _write_table(frame: pd.DataFrame, path: Path, output_format: str) -> Path:
    if output_format == "csv":
        output = path.with_suffix(".csv")
        frame.to_csv(output, index=False)
        return output
    if output_format == "parquet":
        output = path.with_suffix(".parquet")
        try:
            frame.to_parquet(output, index=False)
        except ImportError as exc:
            raise AllocationError(
                "Parquet output requires pyarrow; install cohort-allocator[parquet] "
                "or use --format csv"
            ) from exc
        return output
    raise AllocationError("output_format must be 'csv' or 'parquet'")


def write_prepared_cohorts(
    result: AllocationResult,
    output_dir: str | Path,
    *,
    output_format: str = "parquet",
) -> dict[str, Path]:
    """Write the manifest, one complete cohort table per fold, and audit files.

    A nonempty output directory is rejected so a rerun cannot silently replace
    participant-level assignments.
    """
    if output_format == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise AllocationError(
                "Parquet output requires pyarrow; install cohort-allocator[parquet] "
                "or use --format csv"
            ) from exc
    output = Path(output_dir)
    if output.exists():
        _ensure(output.is_dir(), f"output path is not a directory: {output}")
        _ensure(not any(output.iterdir()), f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cohorts = output / "cohorts"
    cohorts.mkdir()

    artifacts: dict[str, Path] = {}
    artifacts["assignments"] = _write_table(result.assignments, output / "assignments", output_format)
    artifacts["balance"] = _write_table(result.balance, output / "balance", "csv")
    for fold in result.config.fold_names:
        artifacts[f"cohort:{fold}"] = _write_table(
            result.assignments.loc[result.assignments[result.config.fold_column].eq(fold)].copy(),
            cohorts / fold,
            output_format,
        )

    summary = dict(result.summary)
    summary["artifacts"] = {
        name: str(path.relative_to(output))
        for name, path in artifacts.items()
    }
    summary_path = output / "allocation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    artifacts["summary"] = summary_path
    return artifacts
