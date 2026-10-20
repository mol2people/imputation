# Actual-date feasibility of 180-day windows

Date: 2026-10-20. Descriptive EDA only; no HR spectral analysis, detector, surrogate, group test or predictive model was run.

Population: the 8,425 participants with `screened_pass=True` in the [v1 participant table](../female_infradian_cycles/results/infradian_metrics_by_participant.csv). Read each participant's [Apple 5-minute cache](../apple_cohort_scaling/artifacts/cache_5min/). A qualified date has at least 120 finite bins, matching v1's additional screening rule. All resulting qualified-date counts matched the saved `n_adequate_days`. Only dates and finite-bin indicators were used; HR magnitudes and fitted periodicity metrics did not determine windows.

For each participant, candidate starts are their qualified dates. A window is the half-open calendar interval `[start, start + 180)`, divided into six half-open 30-day blocks `[start + 30j, start + 30(j+1))`, j=0,…,5. Every block must contain at least k qualified dates, for k=10,15,20. These thresholds guarantee at least 60/90/120 qualified dates per accepted window. The complete qualified date set determines support; no truncation to the first 60/90/120 dates is applied.

For each k, choose the earliest qualifying start as the first window. A second window is the earliest qualifying start at least 180 calendar days after that first start. Thus the pair has no overlapping calendar dates. This is the requested greedy rule, not an optimization for other matching criteria. Candidate starts are restricted to observed qualified dates; arbitrary unobserved calendar starts are not searched.

[WINDOW_FEASIBILITY.csv](WINDOW_FEASIBILITY.csv) reports age-category × salutation-code-cell counts and fractions. `y=0`/`y=1` are recorded salutation labels, not verified biology. These windows ensure daily lower-tail-HR (`hr_p10`) support. They do not ensure sufficient nighttime observations, stable clock occupancy, seasonal matching, low autocorrelation, or a coherent oscillation. Results concern the v1-screened cohort; no newly eligible participants were sought.

```
   age_group  y  k_per_30d_block  n_v1_screened  n_at_least_one_window  n_two_nonoverlapping_windows  fraction_one  fraction_two
     Elderly  0               10            289                    197                           102      0.681661      0.352941
     Elderly  0               15            289                    172                            91      0.595156      0.314879
     Elderly  0               20            289                    152                            70      0.525952      0.242215
     Elderly  1               10            703                    460                           244      0.654339      0.347084
     Elderly  1               15            703                    401                           195      0.570413      0.277383
     Elderly  1               20            703                    340                           160      0.483642      0.227596
 Middle Ager  0               10           2088                   1331                           668      0.637452      0.319923
 Middle Ager  0               15           2088                   1082                           503      0.518199      0.240900
 Middle Ager  0               20           2088                    838                           368      0.401341      0.176245
 Middle Ager  1               10           3634                   2449                          1339      0.673913      0.368465
 Middle Ager  1               15           3634                   2086                          1081      0.574023      0.297468
 Middle Ager  1               20           3634                   1703                           848      0.468630      0.233352
Young Adults  0               10            502                    292                           137      0.581673      0.272908
Young Adults  0               15            502                    224                           106      0.446215      0.211155
Young Adults  0               20            502                    174                            79      0.346614      0.157371
Young Adults  1               10           1209                    789                           438      0.652605      0.362283
Young Adults  1               15           1209                    666                           348      0.550868      0.287841
Young Adults  1               20           1209                    515                           250      0.425972      0.206782
```

Cache reads and aggregation took 7.8 seconds. Source participant-CSV SHA-256: `be416c0a2b85cdd6a63183e23ecff1c1f7b731e80ce318b512df60549d27ef22`. Caches were read in place and not modified; no cache content hashes were collected. The current CSV and explicit rules above describe the feasibility computation; no executable detector or experiment implementation was added.
