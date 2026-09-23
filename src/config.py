"""Frozen configuration for the salutation-prediction experiment.

All values here are predeclared before allocation and before any model fitting.
Nothing in this file may be tuned using salutation labels, model performance, or
epoch-value/label associations.  See EXPERIMENT_PLAN.md.
"""
from pathlib import Path
import os

# ---------------------------------------------------------------- variant ---
# v1 = frozen protocol.  v2 = requested variant: additionally use age_group,
# bmi_grp and epoch source-membership indicators as predictors, and exclude the
# very small sources 2, 4 and 19 at the record level.
VARIANT = os.environ.get("EXP_VARIANT", "v1")
EXCLUDE_SMALL_SOURCES = VARIANT != "v1"
INCLUDE_DEMOGRAPHIC_PREDICTORS = VARIANT != "v1"

# ---------------------------------------------------------------- paths -----
PROJECT = Path("/Users/bulat/Documents/datenspende_IMPUTE_sex")
DAILY_CSV = PROJECT / "data" / "15Sep_2230_PG_type65-66.csv"
SALUTATION_CSV = PROJECT / "data" / "13Aug_1222.csv"
WHO_CSV = PROJECT / "data" / "df_whoOneAverage.csv"
EPOCH_DIR = PROJECT / "out"
ARTIFACTS = (PROJECT / "experiments" / "artifacts" if VARIANT == "v1"
             else PROJECT / "experiments" / f"artifacts_{VARIANT}")
ARTIFACTS.mkdir(exist_ok=True)

# ------------------------------------------------------------- provenance ---
# Epoch value-type codes present in the export and their mapped meanings
# (mapping/epoch_value_types.csv).  The export establishes the codes; it does
# not establish physiological interpretation or vendor derivation.
CORE_CHANNELS = {3000: "HeartRate", 3001: "HeartRateResting", 3002: "HeartRateRestingHourly"}
# Daily type 65 = RHR, 66 = HR during sleep.  Only 65/66 define the candidate set.
DAILY_TYPES = [65, 66]

# ------------------------------------------------------------- exclusions ---
EXCLUDED_SOURCES = [38, 46, 48] + ([2, 4, 19] if EXCLUDE_SMALL_SOURCES else [])
KEEP_SALUTATIONS = [10, 20]              # outcome 0 = 10, outcome 1 = 20
CODE_30_EXCLUDED = True                  # code 30 dropped entirely (8 records)

# Extra categorical predictors added in the v2 variant.
EXTRA_CATEGORICAL = ["age_group", "bmi_grp"] if INCLUDE_DEMOGRAPHIC_PREDICTORS else []
INCLUDE_SOURCE_INDICATORS = INCLUDE_DEMOGRAPHIC_PREDICTORS

# ------------------------------------------------------- measurement rules ---
# Plausible resting/active heart-rate range in bpm.  Values outside are treated
# as invalid measurements (technical validity rule, not a label-driven filter).
HR_MIN, HR_MAX = 25.0, 230.0

# ------------------------------------------------------- coverage / gate -----
# Local calendar day = floor((startTimestamp + timezoneOffset*60000) / 86400000)
# with missing timezoneOffset treated as 0 (UTC) and flagged.
LOCAL_DAY_MS = 86_400_000
TZ_MS = 60_000

# A local day is "adequately observed" for a core channel when events occupy
# >= ADEQUATE_HOURS distinct local clock hours AND the recording is either
# interval-dense (union coverage >= ADEQUATE_COV_H hours) or sample-dense
# (>= ADEQUATE_N valid measurements).  The disjunction covers interval devices
# (e.g. Garmin 1-min buckets) and point-event devices (e.g. Apple HR samples)
# without conflating them.
ADEQUATE_HOURS = 8
ADEQUATE_COV_H = 8
ADEQUATE_N = 60

# Primary eligibility gate: >= MIN_ADEQUATE_DAYS adequate days in at least one
# core channel.  Days are never pooled across channels.
MIN_ADEQUATE_DAYS = 14

# ------------------------------------------------------------- allocation ---
SEED = 20260918
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10
BALANCE_TOLERANCE = 0.01                 # 1 percentage point design tolerance
RARE_STRATUM_CUTS = (3, 10)              # <3, 3-9, >=10 joint-stratum sizes

# ---------------------------------------------------------------- model ------
RF_SEED = 20260918
BOOTSTRAP_N = 10
BOOTSTRAP_SEED = 20260918
