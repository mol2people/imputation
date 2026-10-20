# Next-stage source and cache inventory

Date: 2026-10-20. This records completed, read-only inventory checks supporting the [v2 plan](PLAN.md). All proposed imputation, source replication and multisource analyses remain planned. No raw query, cache construction, detector, model or simulation was run for this inventory.

## Available sources and observation metadata

Vendor identities are verified for Garmin/source 3, Apple/source 6 and Samsung/source 7 in the project [README](../../README.md:9). Sources 9 and 13 occur in the retained source metadata, but their vendor names have not been verified. Do not assign names by inference. This is an inventory of the five sources retained in the v2 epoch metadata, not necessarily all providers in the original export.

The following counts come from [epoch_sources.parquet](../artifacts_v2/epoch_sources.parquet), selecting channel 3000 and grouping unique participant IDs by source. This table contains source-specific event and recorded-date counts, not qualified v2 windows. Participants can occur in multiple source rows.

| Source | HR3000 participants | Participants with ≥60 recorded HR3000 dates | Maximum recorded HR3000 dates per participant | Ready full-history 5-minute cache |
|---|---:|---:|---:|---|
| Garmin / 3 | 5,812 | 5,397 | 980 | 1,000-person binning pilot |
| Apple / 6 | 13,912 | 11,483 | 905 | Full temporal cohort of 11,139 |
| Samsung / 7 | 2,089 | 1,642 | 980 | None found |
| Unmapped / 9 | 1,486 | 1,214 | 701 | None found |
| Unmapped / 13 | 1,193 | 717 | 591 | None found |

Recorded-date counts do not establish the number of dates with ≥120 occupied bins, block-level occupancy, a qualifying 180-day window or two non-overlapping windows. No such source-specific feasibility count was computed here. A maximum history length does not imply dense or uninterrupted recording.

The [base cohort manifest](../artifacts_v2/cohort_manifest.parquet) contains 27,256 participant rows, categorical `age_group`, categorical `bmi_grp`, salutation and source-membership indicators. Exact age and clinical reproductive status are not supplied. Age/BMI fields are populated in that manifest; salutation is missing for 13 rows. Eligible source cohorts use recorded salutation 10/20, coded y=0/1. Availability of a field is not evidence of clinical validity.

## Ready caches versus metadata-only possibilities

| Artifact | Available content | Limitation for the next stage |
|---|---|---|
| [Apple full-history 5m index](../apple_cohort_scaling/artifacts/cache_5min/customers.csv) | 11,139 participants; per-user `hr[D,288]`, integer `date[D]`, `stats[D,8]` | V2 window selection still defines the analysis population; this is the already explored source. |
| [Garmin full-history 5m index](../garmin_temporal_binning/artifacts/cache_5min/customers.csv) | 1,000 participants; 395,486 adequate participant-days; per-user history 14–979 days; same array schema | This relaxed Garmin pilot is not the strict 3,848-person cohort. Its v2-qualified window counts are unknown. |
| [Garmin strict day-bin cache](../temporal_garmin/cache/day_bins.npz) and [day40 table](../r1b_dateaware_garmin_2026-09-21/cache/day40_table.parquet) | 3,848 participants × exactly 40 selected days; 153,920 rows | Forty selected days do not supply full-history support for the primary 180-day/90-observation protocol. Do not infer raw-history absence from this capped cache. |
| [Pooled daily epoch summaries](../artifacts_v2/epoch_days.parquet) | 17,789,582 rows: `user, channel, date, n, cov_s, hours, mean, median, sd, vmin, vmax` | No source column, no 5-minute mask or HRp10. Multisource participants cannot be separated by device from these summaries alone. |
| Local [raw participant directory](../../out) | Event CSVs include customer, timestamps, value type, value, source and timezone offset | A source-specific adapter/cache build and validity audit would be new work; raw presence does not establish usable windows. |
| [Daily-record input](../../data/15Sep_2230_PG_type65-66.csv) | Existing local input with `user_id,date,type,value,source,timezone_offset` | Daily types 65/66 are different endpoints and do not reconstruct the channel-3000 observation mask. The path is a symlink to a local export. |

Both Apple and Garmin pilot caches span 2020-04-12 to 2022-12-17 across their participants; individual and selected-window eras vary. The strict Garmin 40-day cache spans 2020-04-12 to 2022-11-01 across participants. These cohort-wide ranges are not matched calendar exposure. Source-era differences, within-source recording changes and device switching need explicit auditing before comparing effects.

## Age-category and salutation support

Garmin counts below refer exactly to the existing 1,000-person full-history binning pilot in [garmin_binning_5_allocations.parquet](../garmin_temporal_binning/artifacts/garmin_binning_5_allocations.parquet). They precede v2 eligibility. In particular, **26 elderly female-coded participants is the maximum available in this pilot before window filtering**; any age-specific replication may have substantially less support.

| Age category | Garmin pilot salutation 10 / y0 | Garmin pilot salutation 20 / y1 |
|---|---:|---:|
| Young Adults | 84 | 124 |
| Middle Ager | 252 | 429 |
| Elderly | 26 | 85 |
| Total | 362 | 638 |

Samsung counts below join all source-7/channel-3000 participant IDs in the saved source table to the base manifest. They include multisource and otherwise ineligible participants; they are not a ready single-source or qualified-window cohort.

| Age category | Samsung raw-HR3000 metadata salutation 10 / y0 | Samsung raw-HR3000 metadata salutation 20 / y1 |
|---|---:|---:|
| Young Adults | 158 | 312 |
| Middle Ager | 524 | 857 |
| Elderly | 68 | 170 |
| Total | 750 | 1,339 |

For comparison, the existing eligible single-source base manifest contains Garmin 4,410, Apple 11,143, Samsung 1,498, source 9: 396 and source 13: 543. These base gates can be passed through a core channel other than 3000. They therefore do not determine HR3000 replication eligibility.

## Channel and strict-cohort distinctions

The [configuration](../../src/config.py:29) identifies channel 3000 as `HeartRate`, 3001 as `HeartRateResting`, and 3002 as `HeartRateRestingHourly`; the [results reference](../../reports/results_reference.md:45) describes the same mapping. These exported quantities are distinct. The metadata do not establish vendor derivation algorithms or equivalence between hourly resting HR and HRp10 calculated from observed HR3000 bins.

The [strict source-cohort table](../artifacts_sq/source_cohorts.csv) reports Garmin 3,848, Apple 4,446, Samsung 541, source 9: 9 and source 13: 41 any-core-channel strict passes. Critically, Samsung has **one channel-3000 strict pass and 540 channel-3002 passes**. Its historical 541-person RF cohort is therefore not evidence for 541 dense channel-3000 trajectories. Source 9/13 strict cohorts are also small and were audit-only. A replication using 3002 would be a separately specified endpoint, not a substitute silently pooled with 3000.

Source-specific strict support and window metadata are available in [eligibility_by_channel.parquet](../artifacts_sq/eligibility_by_channel.parquet) and [cohort_manifest_all_sources.parquet](../artifacts_sq/cohort_manifest_all_sources.parquet). They concern the historical strict gate, not the current v2 window rules.

## Multisource participants and possible paired designs

The saved source table gives the following same-ID overlaps for channel 3000:

| Source pair | Shared participant IDs |
|---|---:|
| Garmin–Apple | 1,102 |
| Garmin–Samsung | 72 |
| Apple–Samsung | 5 |

By contrast, the existing single-source Apple full cache and Garmin pilot/strict cohorts have zero participant-ID overlap. These statements refer to different populations and are consistent. The base manifest flags 2,509 epoch-multisource participants, but multisource presence alone does not establish switching chronology or concurrent measurements.

A paired-device design is a possibility, not demonstrated feasibility. It requires source-specific dates and observation support, separation of simultaneous use from switching, label/metadata consistency, matched calendar windows and a check of source-era changes. The pooled daily summaries omit source; returning to source-tagged raw events would be necessary to construct separate device sequences for these participants. No paired windows or switching effects were computed here.

## Timezone, bin and statistic semantics

The [Garmin pilot cache builder](../garmin_temporal_binning/scripts/02_build_garmin_binning_caches.py:47) requires finite timezone offsets and builds local dates from `startTimestamp + timezoneOffset × 60000`. Channel-3000 values are restricted to 25–230 bpm. A cached adequate day needs eight occupied clock hours and either eight union-covered hours or 60 readings; see [the implemented gate](../garmin_temporal_binning/scripts/02_build_garmin_binning_caches.py:88). Bins are means of readings grouped by their local start-time bin; this should not be assumed to be a duration-weighted physiological mean.

The [pooled configuration](../../src/config.py:52) instead permits missing timezone offsets as flagged UTC. Source comparisons must freeze one policy and report affected observations, local-date boundaries and offset changes; these different historical policies should not be silently combined. Apple point events and Garmin interval-style records also give different meanings to event counts and union coverage, as documented in [README](../../README.md:13).

The shared per-user daily-statistic order is `mean, median, sd, vmin, vmax, n, cov_h, hours`. Matching column order does not make recording statistics measurement-equivalent. The Apple cache has no native sub-bin counts, so it cannot support count-weighted harmonization from cache arrays alone. Original masks must remain distinct from any future filled values; imputation details belong to the [next-stage plan](PLAN.md), not to this inventory.

## Planning implication and verification boundary

The smallest ready-cache replication candidate is the Garmin full-history 1,000-person pilot, conditional on a new outcome-independent v2 window-feasibility audit and sufficient age×salutation support. Samsung/source 7 is the next verified provider to audit from raw channel-3000 metadata, with no inference from its 3002-dominated strict cohort. Sources 9/13 require vendor provenance and source-specific support checks first. Multisource data offer a possible later within-person design, not a currently verified paired sample.

Counts above were obtained by reading saved tables and cache indexes; earlier inventory checks also read cache date arrays. No full-history cache for Samsung/9/13 was found in the inspected experiment cache indexes. This does not prove no other raw data exist. No new files other than this document were created for the present inventory task, and no source data or existing result table was modified.
