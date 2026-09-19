"""Frozen configuration for the side-quest (sq) protocol.

Independent of ``src/config.py``: that module is variant-dependent via the
``EXP_VARIANT`` environment variable and creates a variant artifacts directory
on import.  The side quest must explicitly use the **v2** artifact directory and
v2 exclusions, so the frozen constants are restated here instead of imported.
See SIDEQUEST_PLAN.md (sections referenced in each module).
"""
from pathlib import Path

PROJECT = Path("/Users/bulat/Documents/datenspende_IMPUTE_sex")
ARTIFACTS_V2 = PROJECT / "artifacts_v2"       # side quest reads v2 only
ARTIFACTS_SQ = PROJECT / "artifacts_sq"       # side quest writes here only
EPOCH_DIR = PROJECT / "out"                   # raw epoch export (targeted mini-scan)

# ---------------------------------------------------------------- universe ---
# Sole-source modelling loop is driven by this tuple (plan section 7).
MODEL_SOURCE_IDS = (3, 6, 7)
# Epoch sources retained by the v2 record-level filters (38/46/48 dropped; the
# small sources 2/4/19 were additionally dropped for v2).
ALLOWED_SOURCES = (3, 6, 7, 9, 13)
EXCLUDED_SOURCES = (38, 46, 48, 2, 4, 19)

CORE_CHANNELS = (3000, 3001, 3002)
CHANNEL_PRIORITY = (3000, 3001, 3002)         # fixed tie-break, ascending

# ------------------------------------------------------- measurement rules ---
HR_MIN, HR_MAX = 25.0, 230.0
LOCAL_DAY_MS = 86_400_000
TZ_MS = 60_000

# v2 feature-day rule (deliberately distinct from the 12 h strict rule)
ADEQUATE_HOURS = 8
ADEQUATE_COV_H = 8
ADEQUATE_N = 60

# ------------------------------------------------- longitudinal coverage gate
STRICT_COV_S = 43_200.0                       # 12 h union coverage -> strict day
WINDOW_WEEKS = 13
MIN_ADEQUATE_WEEKS = 11                       # K = 2 slack weeks
MIN_STRICT_DAYS = 65
WINDOW_INCLUSIVE_DAYS = 91                    # [m, m + 90]
WINDOW_THIRTEENTH_MONDAY_OFFSET = 84

# ------------------------------------------------------------ modelling rule --
MIN_CLASS_FOR_MODEL = 200                     # >= 200 per outcome class

# ------------------------------------------------------ allocation / modelling
SEED = 20260918
RF_SEED = 20260918
SQ_BOOTSTRAP_N = 100                          # exactly 100 resamples (plan section 5)
SQ_BOOTSTRAP_SEED = 20260918
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10

VARIANTS = ("demo", "rec", "win", "roll_rec", "roll_win", "all")

# frozen raw widths (plan section 3); deviations must be explained in the
# feature dictionary, never absorbed silently.
WIDTH_D, WIDTH_A, WIDTH_B, WIDTH_C_REC, WIDTH_C_WIN, WIDTH_ALL = 2, 78, 78, 120, 120, 398

BUCKETS = {"night": (0, 5), "morning": (6, 11), "afternoon": (12, 17), "evening": (18, 23)}
ROLL_SERIES = ("daily_mean", "daily_median", "daily_sd", "daily_hours")
ROLL_WIDTHS = (7, 30)
ROLL_SUMMARIES = ("mean", "sd", "min", "max", "slope")

# ------------------------------------------------------------- frozen anchors --
# Reproduced exactly or the implementation stops and reports drift.
# NOTE (2026-09-19, user decision): the frozen table in SIDEQUEST_PLAN.md section 2
# drifted for two cells (Apple ch3002 4,363 and Samsung ch3002 539).  The section-1
# gate semantics were declared authoritative; every anchor below follows from them
# (any-core, class counts, model union, and the 80/10/10 largest-remainder splits,
# whose rule reproduces the untouched Garmin row exactly).
FROZEN_BASE_SINGLE_SOURCE = {3: 4410, 6: 11143, 7: 1498, 9: 396, 13: 543}
FROZEN_ANY_CORE = {3: 3848, 6: 4446, 7: 541, 9: 9, 13: 41}          # union = 8,885
# (sal10, sal20) per source among any-core passers
FROZEN_ANY_CORE_CLASS = {3: (1384, 2464), 6: (1373, 3073), 7: (209, 332),
                         9: (4, 5), 13: (5, 36)}
FROZEN_CHANNEL_PASS = {3: {3000: 3847, 3001: 2148, 3002: 3181},
                       6: {3000: 28, 3001: 105, 3002: 4367},
                       7: {3000: 1, 3001: 0, 3002: 540},
                       9: {3000: 0, 3001: 0, 3002: 9},
                       13: {3000: 19, 3001: 1, 3002: 26}}
FROZEN_ALL_SOURCE_PASS = 8885
FROZEN_MODEL_UNION = 8835
# frozen 80/10/10 tables: source -> split -> (total, sal10, sal20)
# Target rule (reproduces the untouched Garmin row and the recorded Apple row
# exactly): T = largest_remainder(FRACS, N) with v2's tie-break (ties -> lowest
# split index), then class-1 (sal20) apportioned by the same rule and sal10
# derived as T - sal20.  Source 7's train sal10 is 167, not 166: 166+266 != 433
# and 166+21+21 != 209.
FROZEN_SPLITS = {
    3: {"train": (3078, 1107, 1971), "val": (385, 138, 247), "test": (385, 139, 246)},
    6: {"train": (3557, 1098, 2459), "val": (445, 138, 307), "test": (444, 137, 307)},
    7: {"train": (433, 167, 266), "val": (54, 21, 33), "test": (54, 21, 33)},
}

REASON_ORDER = ("no_channel_rows", "span_lt_13_weeks", "max_adequate_weeks_lt_11",
                "max_strict_days_among_week_qualifying_lt_65", "pass")

CONTEXT_ANCHORS = {
    "v2_pooled_test_auroc": 0.68333,
    "v2_pooled_bootstrap_auroc_sd_10draws": 0.00873,
    "v1_model_source3_subgroup_test_auroc_n533": 0.74038,
}
