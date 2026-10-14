"""One person's CSV -> adequate days, 288 bins/day, eight daily summaries.

Native timestamps are UTC milliseconds; timezoneOffset is minutes EAST of UTC.
Simple CSV timestamps must include their UTC offset (preserved per row).
"""
import numpy as np
import pandas as pd

STATS = ["mean", "median", "sd", "vmin", "vmax", "n", "cov_h", "hours"]


def union_hours(start, end):
    """Interval union assigned to START date, matching the original analysis."""
    order = np.argsort(start, kind="stable")
    start, end = start[order], end[order]
    cumulative = np.maximum.accumulate(end)
    first = np.r_[0, np.flatnonzero(start[1:] > cumulative[:-1]) + 1]
    last = np.r_[first[1:] - 1, len(start) - 1]
    return float(np.maximum(0, cumulative[last] - start[first]).sum() / 3_600_000)


def normalize_epochs(frame):
    """Validate one person's CSV and retain native events with local day/bin keys."""
    frame = frame.copy()
    for col in ("user", "user_id", "customer_id"):
        if col in frame and frame[col].nunique(dropna=False) != 1:
            raise ValueError("Upload one person's data at a time.")
    if {"timestamp", "hr"}.issubset(frame):
        starts, ends, offsets = [], [], []
        for row in frame.itertuples(index=False):
            stamp = pd.Timestamp(row.timestamp)
            if pd.isna(stamp) or stamp.tzinfo is None:
                raise ValueError("Every timestamp needs an explicit offset, e.g. +02:00.")
            starts.append(stamp.value // 1_000_000)
            offsets.append(stamp.utcoffset().total_seconds() / 60)
            end = getattr(row, "end_timestamp", None)
            if end is None or pd.isna(end):
                ends.append(starts[-1])
            else:
                end = pd.Timestamp(end)
                if end.tzinfo is None:
                    raise ValueError("Every end_timestamp needs an explicit offset.")
                ends.append(end.value // 1_000_000)
        frame = pd.DataFrame(dict(startTimestamp=starts, endTimestamp=ends,
                                  timezoneOffset=offsets, longValue=frame.hr.to_numpy(),
                                  source=3, type=3000))
    required = ["startTimestamp", "endTimestamp", "timezoneOffset", "longValue", "source", "type"]
    if missing := set(required) - set(frame):
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    valid = (numeric.source.eq(3) & numeric.type.eq(3000)
             & numeric.longValue.between(25, 230) & numeric.startTimestamp.gt(0)
             & np.isfinite(numeric[["startTimestamp", "timezoneOffset", "longValue"]]).all(axis=1))
    frame = numeric.loc[valid].copy()
    if frame.empty:
        raise ValueError("No valid type-3000/source-3 HR rows (25–230 bpm, known timezone).")
    start = frame.startTimestamp.to_numpy(np.int64)
    local_ms = start + (frame.timezoneOffset.to_numpy() * 60_000).astype(np.int64)
    # Also catches common seconds-instead-of-milliseconds input errors.
    local_dates = pd.to_datetime(local_ms, unit="ms")
    if (local_dates.year < 2000).any() or (local_dates.year > 2100).any():
        raise ValueError("Expected contemporary UTC timestamps in milliseconds, not seconds.")
    frame["date"] = local_dates.normalize()
    frame["bin"] = (local_ms % 86_400_000) // 300_000
    frame["hour"] = frame.bin // 12
    end = frame.endTimestamp.to_numpy()
    frame["endTimestamp"] = np.where(np.isfinite(end) & (end >= start), end, start)
    return frame, dict(input_rows=len(numeric), retained_rows=len(frame),
                       rejected_rows=len(numeric) - len(frame))


def aggregate_days(frame):
    """Aggregate validated events; empty output is allowed after perturbation."""
    rows, bins = [], []
    for date, day in frame.groupby("date", sort=True):
        values = day.longValue.to_numpy()
        coverage = union_hours(day.startTimestamp.to_numpy(), day.endTimestamp.to_numpy())
        hours = day.hour.nunique()
        if hours < 8 or (coverage < 8 and len(day) < 60):
            continue
        curve = day.groupby("bin").longValue.mean().reindex(range(288)).to_numpy(np.float32)
        bins.append(curve)
        rows.append([date, values.mean(), np.median(values), values.std(ddof=1),
                     values.min(), values.max(), len(day), coverage, hours])
    table = pd.DataFrame(rows, columns=["date", *STATS])
    return table, np.stack(bins) if bins else np.empty((0, 288), dtype=np.float32)


def prepare_days(frame):
    """Return (adequate days, HR with NaN gaps, quality); no interpolation."""
    events, quality = normalize_epochs(frame)
    table, hr = aggregate_days(events)
    quality.update(observed_days=int(events.date.nunique()), adequate_days=len(table))
    if table.empty:
        raise ValueError("No adequate days: need ≥8 clock-hours AND (≥60 readings OR ≥8 hours interval coverage).")
    return table, hr, quality
