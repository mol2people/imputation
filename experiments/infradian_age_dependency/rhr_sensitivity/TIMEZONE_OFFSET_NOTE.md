# Timezone Offset Distribution Note

## Context

The supplier daily table (`data/15Sep_2230_PG_type65-66.csv`) contains a
`timezone_offset` column (in minutes) that records the timezone context for
each row. This column is needed to interpret the `date` column correctly.

## Distribution (n = 9,507,720 non-null; 3,359,066 null = 26.1% of total)

| Offset (min) | Timezone | Rows | % of non-null |
|---|---|---|---|
| 120 | UTC+2 (CEST) | 5,007,670 | 52.6% |
| 60 | UTC+1 (CET) | 3,647,628 | 38.4% |
| 0 | UTC | 794,545 | 8.4% |
| 180 | UTC+3 | 17,345 | 0.18% |
| -240 | UTC-4 | 6,105 | 0.06% |
| ... | ... | ... | ... |
| -1425 | UTC-23.75 | 285 | 0.003% |
| 1440 | UTC+24 | 56 | 0.0006% |

**Three dominant offsets (0, 60, 120)** cover 9,449,843 rows = **99.4%** of
non-null data.

## Issues

1. **Extreme/invalid values**: 892 rows have timezone_offset outside the
   valid range [-720, +840] (UTC-12 to UTC+14). These include values like
   -1425, -1365, 1440, 1425, 1410. No timezone has such extreme offsets;
   these are data quality issues.

2. **Null values**: 3,359,066 rows (26.1%) have null timezone_offset.
   Per user instruction, these are treated as **Berlin time (CET/CEST)**
   and kept in the analysis. Since CET/CEST offsets (60/120 minutes) do
   not change the calendar date when applied to midnight, the date in
   these rows is used as-is.

3. **Rare but valid values**: Values like 180, -240, -300, 480, etc. are
   valid timezone offsets but appear rarely (<0.2% each). Per user
   instruction, these are excluded as "rare."

## Decision

Per user instruction: "note the offset issue somewhere, and for now just
ignore offsets that are rare."

**Filtering criterion**: Keep rows with `timezone_offset` in {0, 60, 120}
or null (treated as Berlin time / CET/CEST).

This retains:
- The three dominant, standard timezone offsets: UTC (0), CET (60), CEST (120)
- Null timezone_offset rows (26.1% of data), treated as Berlin time (CET/CEST)

**Excluded**:
- Rows with extreme timezone_offset values (892 rows with values outside
  [-720, +840])
- Rows with rare but valid timezone_offset values (180, -240, -300, etc.)

## Impact on Analysis

- **Data retention**: ~12.8M rows retained out of 12.87M total (>99%)
- **Null timezone_offset rows**: kept, treated as Berlin time (CET/CEST)
- **Extreme/rare offsets dropped**: ~59K rows (0.5% of total)
- **Conversion**: For offsets {0, 60, 120}, the timezone conversion
  (date + offset applied to midnight) does not change the calendar date
  (midnight + 0/1/2 hours stays on the same day). The date in these rows
  is used as-is.
- **Null timezone_offset rows**: treated as Berlin time (CET/CEST);
  same as offsets 60/120 — date used as-is, no calendar-date change.

## Notes

- The `timezone_offset` column is in minutes.
- The most common offsets are 60 (CET, UTC+1) and 120 (CEST, UTC+2),
  which correspond to Berlin time (CET/CEST).
- Rows with timezone_offset = 0 (UTC) are from users in the UTC+0
  timezone (e.g., UK/GMT). For these rows, the date is also used as-is
  (midnight + 0 hours stays on the same day).
- The analysis does not apply any timezone conversion to the `date`
  column because for the dominant offsets ({0, 60, 120}) and null rows
  (treated as Berlin time/CET/CEST), the conversion does not change the
  calendar date.
