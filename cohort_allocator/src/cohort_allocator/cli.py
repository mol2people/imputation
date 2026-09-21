"""Command-line interface for cohort-allocator."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .allocator import (
    AllocationError,
    allocate_cohort,
    make_config,
    read_demographic_summary,
    write_prepared_cohorts,
)


def _columns(raw: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _fold_sizes(raw: str) -> list[tuple[str, float]]:
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if len(pieces) < 2:
        raise AllocationError("--fold-sizes needs at least two folds")
    named = ["=" in piece for piece in pieces]
    if any(named) and not all(named):
        raise AllocationError("--fold-sizes must use either all NAME=SIZE entries or all numeric entries")
    pairs: list[tuple[str, float]] = []
    try:
        if all(named):
            for piece in pieces:
                name, size = piece.split("=", maxsplit=1)
                pairs.append((name.strip(), float(size.strip())))
        else:
            for number, piece in enumerate(pieces, start=1):
                pairs.append((f"fold_{number}", float(piece)))
    except ValueError as exc:
        raise AllocationError("--fold-sizes entries must be numeric, e.g. train=8,test=2") from exc
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Allocate a participant-level demographic summary into balanced, "
            "disjoint folds with OR-Tools."
        )
    )
    parser.add_argument("summary", type=Path, help="participant-level .csv or .parquet demographic summary")
    parser.add_argument("--id-column", required=True, help="unique participant identifier column")
    parser.add_argument(
        "--stratify",
        required=True,
        help="comma-separated categorical columns to balance jointly and marginally",
    )
    parser.add_argument(
        "--fold-sizes",
        required=True,
        help="relative fold sizes, preferably named: train=8,validation=1,test=1",
    )
    parser.add_argument(
        "--partition-by",
        default="",
        help="optional comma-separated columns to allocate independently, e.g. source_id",
    )
    parser.add_argument("--fold-column", default="fold", help="name of the appended fold-label column")
    parser.add_argument("--seed", type=int, default=20260921, help="deterministic allocation seed")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="total CP-SAT time limit in seconds per partition (default: 300)",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for prepared cohorts")
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet", help="cohort table format")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = make_config(
            id_column=args.id_column,
            stratify_columns=_columns(args.stratify),
            fold_sizes=_fold_sizes(args.fold_sizes),
            partition_columns=_columns(args.partition_by),
            fold_column=args.fold_column,
            seed=args.seed,
            time_limit_seconds=args.time_limit,
        )
        result = allocate_cohort(read_demographic_summary(args.summary), config)
        artifacts = write_prepared_cohorts(
            result,
            args.output_dir,
            output_format=args.format,
        )
    except AllocationError as exc:
        parser.error(str(exc))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Prepared {len(result.assignments):,} participants in {len(result.summary['partitions'])} partition(s).")
    print(f"Manifest: {artifacts['assignments']}")
    print(f"Audit:    {artifacts['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
