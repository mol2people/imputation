"""Deterministic fixtures for the side-quest rolling blocks (plan section 8).

Hand-computed expectations for calendar gaps, the endpoint-adequacy gate,
minimum-count support, pre-window lookback, and the calendar-day OLS slope.

Run:  python src/sidequest/test_rolling_fixture.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import _endpoint_summaries, _rolling_endpoints  # noqa: E402

# Grid days 100..109; day 102 fails the feature-day rule (calendar gap).
# Values on adequate days: 10,20,_,40,50,60,70,80,90,100.
FIRST = 100
adequate = np.ones(10, dtype=bool)
adequate[2] = False
v = np.array([10., 20., np.nan, 40., 50., 60., 70., 80., 90., 100.])

# --- trailing 7-day mean, minimum support 4 ---------------------------------
t_idx, vals = _rolling_endpoints(v, adequate, 7, 4)
expect_t = np.array([4, 5, 6, 7, 8, 9])            # t=103 support 3 -> undefined
expect_v = np.array([30., 36., 250 / 6, 320 / 6, 65., 70.])
assert np.array_equal(t_idx, expect_t), (t_idx, expect_t)
assert np.allclose(vals, expect_v, atol=1e-12), (vals, expect_v)

# --- endpoint day must itself be adequate ------------------------------------
adeq2 = adequate.copy()
adeq2[9] = False                    # day 109 inadequate: endpoint t=9 drops out
t2, v2 = _rolling_endpoints(v, adeq2, 7, 4)
assert np.array_equal(t2, np.array([4, 5, 6, 7, 8])), t2

# --- 30-day window on a 10-day grid: no endpoints ----------------------------
t30, v30 = _rolling_endpoints(v, adequate, 30, 15)
assert t30.size == 0 and v30.size == 0

# --- summaries over the defined endpoints ------------------------------------
s = _endpoint_summaries(t_idx, vals, FIRST)
assert np.isclose(s["mean"], np.mean(expect_v))
assert np.isclose(s["min"], 30.0) and np.isclose(s["max"], 70.0)
assert np.isclose(s["sd"], np.std(expect_v, ddof=1))
# OLS slope against actual calendar days 104..109
x = FIRST + expect_t
slope_ref = np.polyfit(x, expect_v, 1)[0]
assert np.isclose(s["slope"], slope_ref), (s["slope"], slope_ref)

# --- pre-window lookback (C_win scoping) --------------------------------------
# Window [106, 109]: only endpoints t=6..9 lie inside; their trailing windows
# reach back to day 100 (pre-window input by construction).
inwin = (FIRST + t_idx) >= 106
sw = _endpoint_summaries(t_idx[inwin], vals[inwin], FIRST)
assert np.isclose(sw["mean"], np.mean(expect_v[inwin]))
assert np.isclose(sw["min"], expect_v[inwin].min())
assert np.isclose(sw["max"], 70.0)
assert np.isfinite(sw["slope"])     # 4 endpoints -> slope defined

# --- degenerate cases ---------------------------------------------------------
s1 = _endpoint_summaries(np.array([3]), np.array([42.0]), FIRST)
assert np.isclose(s1["mean"], 42.0) and np.isnan(s1["sd"]) and np.isnan(s1["slope"])
s0 = _endpoint_summaries(np.empty(0, dtype=np.int64), np.empty(0), FIRST)
assert all(np.isnan(x_) for x_ in s0.values())

# constant series: slope exactly 0, sd 0
tc, vc = _rolling_endpoints(np.full(10, 7.0), adequate, 7, 4)
sc = _endpoint_summaries(tc, vc, FIRST)
assert np.isclose(sc["slope"], 0.0) and np.isclose(sc["sd"], 0.0)

print("rolling fixture tests passed")
