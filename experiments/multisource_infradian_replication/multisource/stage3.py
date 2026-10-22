"""Persistent worker queue for Stage 3 endpoint and null computation.

The historical runner created one ``ProcessPoolExecutor`` per arm.  That left
all but the last worker idle at every arm boundary and paid repeated pool
startup costs.  This module submits every daily and channel task to one pool,
while writing each arm as soon as its ordered results are complete.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import channel3000, endpoints, reml


def _dispatch(payload: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    kind, task = payload
    if kind == "daily":
        return endpoints.process_daily_task(task)
    if kind == "channel":
        return channel3000.process_channel_task(task)
    raise ValueError(f"Unknown Stage 3 task kind: {kind!r}")


def _prepare_task(kind: str, task: dict[str, Any], null_replicates: int) -> None:
    task["null_replicates"] = int(null_replicates)
    if kind == "channel":
        task["save_daily_values"] = True
    fitter = reml.active_fitter()
    if fitter is not None:
        task["reml_fitter"] = fitter


def _write_job(job: dict[str, Any], rows: list[dict[str, Any]]) -> pd.DataFrame:
    kind = str(job["kind"])
    output_dir = Path(job["output_dir"])
    if kind == "daily":
        frame = endpoints.write_daily_outputs(rows, output_dir)
    elif kind == "channel":
        frame = channel3000.write_channel_outputs(rows, output_dir)
    else:
        raise ValueError(f"Unknown Stage 3 job kind: {kind!r}")
    print(f"{job.get('label', kind)}: {len(frame)} rows -> {output_dir}")
    return frame


def _consume(
    flat: list[tuple[int, int, str, dict[str, Any]]],
    results: Iterable[list[dict[str, Any]]],
    jobs: list[dict[str, Any]],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    current_job: int | None = None
    rows: list[dict[str, Any]] = []

    for (job_idx, _task_idx, _kind, _task), result in zip(flat, results):
        if current_job is None:
            current_job = job_idx
        elif job_idx != current_job:
            frames.append(_write_job(jobs[current_job], rows))
            rows = []
            current_job = job_idx
        if result:
            rows.extend(result)

    if current_job is not None:
        frames.append(_write_job(jobs[current_job], rows))
    return frames


def run_arms(
    jobs: list[dict[str, Any]],
    workers: int = 8,
    null_replicates: int = 30,
) -> list[pd.DataFrame]:
    """Execute all arm jobs on one persistent pool.

    ``jobs`` is an ordered list of dicts with ``kind`` (``daily``/``channel``),
    ``tasks``, ``output_dir``, and an optional ``label``.  Results are consumed
    in submission order, so each arm is written with the same row order as the
    per-arm runner.
    """
    flat: list[tuple[int, int, str, dict[str, Any]]] = []
    for job_idx, job in enumerate(jobs):
        kind = str(job["kind"])
        for task_idx, task in enumerate(job["tasks"]):
            _prepare_task(kind, task, null_replicates)
            flat.append((job_idx, task_idx, kind, task))

    if not flat:
        return []

    payloads = ((kind, task) for _job, _idx, kind, task in flat)
    if workers > 1 and len(flat) > 1:
        chunk = max(1, min(4, len(flat) // (int(workers) * 32) + 1))
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            return _consume(
                flat,
                ex.map(_dispatch, payloads, chunksize=chunk),
                jobs,
            )
    return _consume(flat, (_dispatch(payload) for payload in payloads), jobs)
