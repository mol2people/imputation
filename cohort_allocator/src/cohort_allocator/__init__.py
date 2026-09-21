"""Participant-level, balanced cohort allocation with OR-Tools."""

from .allocator import (
    AllocationConfig,
    AllocationError,
    AllocationResult,
    allocate_cohort,
    fold_target_counts,
    make_config,
    read_demographic_summary,
    write_prepared_cohorts,
)

__all__ = [
    "AllocationConfig",
    "AllocationError",
    "AllocationResult",
    "allocate_cohort",
    "fold_target_counts",
    "make_config",
    "read_demographic_summary",
    "write_prepared_cohorts",
]
