#!/Users/bulat/micromamba/envs/datenspende/bin/python
import os
import numpy as np

cid = 1222449
c_path = f"experiments/apple_cohort_scaling/artifacts/cache_5min/customer_{cid}.npz"
with np.load(c_path) as npz:
    hr = npz["hr"]
    dates = npz["date"]

valid_bins = np.isfinite(hr).sum(axis=1)
qual_dates = np.sort(dates[valid_bins >= 120])
print(f"Customer {cid}: total dates={len(dates)}, qual dates={len(qual_dates)}")

import importlib.util
script_path = "experiments/female_infradian_cycles_v2/scripts/01_extract_and_calibrate_infradian.py"
spec = importlib.util.spec_from_file_location("m01", script_path)
m01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m01)

w1_start, w2_start = m01.find_qualifying_windows(qual_dates, 15)
print("w1_start:", w1_start, "w2_start:", w2_start)

w1_dates = m01.select_window_dates(qual_dates, w1_start, cid, 15)
w1_hr_p10, w1_clock, w1_occ, w1_night = m01.extract_features_on_dates(hr, dates, w1_dates)
print("w1_hr_p10 range:", np.min(w1_hr_p10), np.max(w1_hr_p10), "std:", np.std(w1_hr_p10))

X_base, names, proj = m01.build_nonperiodic_baseline(w1_dates, w1_clock)
print("Retained names:", names)
e_0 = proj @ w1_hr_p10
print("e_0 std:", np.std(e_0), "rss_0:", np.sum(e_0**2))

try:
    phi, s2 = m01.estimate_calendar_ar1(e_0, w1_dates)
    print("phi:", phi, "s2:", s2)
except Exception as e:
    print("Error in estimate_calendar_ar1:", type(e), e)
