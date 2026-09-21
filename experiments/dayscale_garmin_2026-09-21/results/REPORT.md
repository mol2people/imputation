# Day-level salutation classification (Garmin) — results

Implements [`../PLAN.md`](../PLAN.md). 30 FS-phase participant splits (paired); 5 workers, n_jobs=1 per fit.

Reference lines: participant-level AUROC 0.7445 (frozen A3_k50), 0.7470 (AGE_all, age experiment).

## 1. Day-level AUROC (mean ± SD over 30 repeats)

| arm | test AUROC | val AUROC | ECE (val) | n features | train days | val days | test days |
|---|---|---|---|---|---|---|---|
| `A` | 0.6669 ± 0.0127 | 0.6649 | 0.0810 | 28 | 726,998 | 212,081 | 212,122 |
| `B` | 0.7073 ± 0.0163 | 0.7076 | 0.0684 | 52 | 726,998 | 212,081 | 212,122 |
| `A_covs` | 0.6717 ± 0.0169 | 0.6706 | 0.0767 | 32 | 726,998 | 212,081 | 212,122 |
| `A_K100` (r=0) | 0.6521 | 0.6342 | 0.0759 | 28 | 267,998 | 213,242 | 211,959 |
| `A_uncapped` (r=0) | 0.6540 | 0.6378 | 0.0745 | 28 | 987,629 | 213,242 | 211,959 |

## 2. Participant-cluster bootstrap 95% CI (day AUROC)

| arm | exact AUROC (mean) | boot mean | mean q2.5 | mean q97.5 |
|---|---|---|---|---|
| `A` | 0.6669 | 0.6669 | 0.6338 | 0.6987 |
| `B` | 0.7073 | 0.7074 | 0.6669 | 0.7464 |
| `A_covs` | 0.6717 | 0.6717 | 0.6376 | 0.7054 |

> **Erratum (2026-09-21).** `run_dayscale.py` zipped the bootstrap tuple
> `(arm, r, boot_mean, q025, q975, exact)` against the names
> `(arm, r, exact, boot_mean, q025, q975)` — every CSV column carried the
> previous tuple element's value, and the table above was originally printed
> from those mislabeled columns. All values were correct; only labels were
> shifted. `cluster_bootstrap.csv` has been relabeled in place (data
> unchanged, verified: boot mean ∈ [q2.5, q97.5] for all 90 rows; exact ≈
> boot mean, |Δ| ≤ 0.002). The script's zip order is fixed for future runs;
> `dayscale_repro.json` records the sha of the pre-fix script (see git
> history of this commit's parent).

## 3. k-day scaling curve (participant AUROC, mean over 30 repeats)

| k (days) | A | B | A+covs | n users (A) |
|---|---|---|---|---|
| 1 | 0.6653 ± 0.0122 | 0.7045 ± 0.0156 | 0.6740 ± 0.0143 | 578 |
| 2 | 0.6902 ± 0.0147 | 0.7168 ± 0.0175 | 0.6965 ± 0.0167 | 578 |
| 4 | 0.7074 ± 0.0158 | 0.7236 ± 0.0182 | 0.7120 ± 0.0183 | 578 |
| 8 | 0.7181 ± 0.0168 | 0.7274 ± 0.0186 | 0.7214 ± 0.0193 | 578 |
| 16 | 0.7237 ± 0.0173 | 0.7292 ± 0.0190 | 0.7262 ± 0.0199 | 578 |
| 32 | 0.7269 ± 0.0176 | 0.7302 ± 0.0190 | 0.7290 ± 0.0201 | 578 |
| 64 | 0.7279 ± 0.0174 | 0.7302 ± 0.0189 | 0.7298 ± 0.0202 | 577 |
| 128 | 0.7273 ± 0.0180 | 0.7305 ± 0.0184 | 0.7255 ± 0.0219 | 546 |
| all | 0.7302 ± 0.0179 | 0.7312 ± 0.0191 | 0.7317 ± 0.0204 | 578 |

## 4. Covariates decomposition (A+covs − A, paired by repeat)

- k=1: Δ=+0.0087 [+0.0046, +0.0128]
- k=2: Δ=+0.0063 [+0.0017, +0.0110]
- k=4: Δ=+0.0045 [-0.0006, +0.0097]
- k=8: Δ=+0.0033 [-0.0020, +0.0087]
- k=16: Δ=+0.0025 [-0.0030, +0.0080]
- k=32: Δ=+0.0021 [-0.0035, +0.0077]
- k=64: Δ=+0.0019 [-0.0037, +0.0075]
- k=128: Δ=-0.0018 [-0.0074, +0.0038]
- k=all: Δ=+0.0016 [-0.0041, +0.0072]

## 5. Top day-level features (arm A; Gini and PI averaged over 30 repeats)

| feature | Gini (mean) | PI mean (mean) |
|---|---|---|
| `d_ch3000_mean` | 0.1024 | 0.06576 |
| `d_ch3000_sd` | 0.0629 | 0.00699 |
| `d_ch3000_vmin` | 0.0576 | 0.00563 |
| `d_ch3001_sd` | 0.0521 | 0.00657 |
| `d_gap_3000_3001_sd` | 0.0510 | 0.00571 |
| `d_gap_3000_3002_mean` | 0.0497 | 0.00969 |
| `d_ch3000_vmax` | 0.0489 | 0.00487 |
| `d_ch3000_median` | 0.0468 | 0.00317 |
| `d_gap_3000_3001_mean` | 0.0446 | 0.00189 |
| `d_gap_3000_3002_sd` | 0.0422 | 0.00073 |
| `d_ch3000_n` | 0.0410 | 0.00048 |
| `d_ch3002_sd` | 0.0407 | 0.00087 |
| `d_ch3001_mean` | 0.0391 | 0.00265 |
| `d_ch3000_cov_h` | 0.0382 | -0.00003 |
| `d_ch3002_mean` | 0.0357 | -0.00032 |

## 6. Interpretation

**(a) A single adequate day carries real but bounded signal.** One day's
summary stats as a user-score reach participant-unit AUROC 0.6653 (A); the
day-unit pooled AUROC (§1) is 0.6669 — the two are equivalent by construction
(one day = one score). The strongest single-day representation is
deviation-from-own-baseline (arm B): 0.7045, +0.039 over raw day stats.

**(b) Per-user integration is the dominant effect; it saturates fast.**
Averaging k days' logits lifts A from 0.6653 (k=1) to 0.7181 (k=8), 0.7269
(k=32), 0.7302 (all) — a +0.065 integration gain, 4–20× larger than any arm
or covariate contrast. Gains concentrate in k=1→8; beyond ~32 days the curve
is flat (k=64 0.7279, k=128 0.7273, all 0.7302 — within repeat noise of each
other). The marginal day's information decays quickly: after a month, more
of the same daily summaries add ~nothing.

**(c) Day pooling does not reach the participant-level matrix.** All-arms
ceiling ≈ 0.730–0.732 vs participant-level references 0.7445 (frozen A3_k50)
/ 0.7470 (AGE_all). The ~0.014 gap is cross-day structure that mean-logit
pooling cannot extract — participant features like sd-of-daily-mean, rolling
w7 extremes and tod means encode *how days vary*, not just their average
level. Within-day time-of-day shape is absent from all arms here (PLAN §3
limitation); part of the gap may live there.

**(d) Arm B's advantage is front-loaded and self-eliminating.** B − A:
+0.0392 (k=1) → +0.0138 (k=4) → +0.0032 (k=16) → +0.0010 (all). Knowing how
a day deviates from the person's own baseline is the best short-history
representation, but accumulated raw days recover the same user-level
structure. Deployment read: use baseline-deviation features when <1 week of
history exists; beyond a month they add nothing.

**(e) Demographics are front-loaded and subsumed the same way.** A+covs − A
decays from +0.0087 (k=1, CI clear of zero) to ≈0 by k≥16. Age/bmi carry
information that per-user day accumulation replaces — the participant-level
saturation story (demographics implicit in behavior features) reproduced at
the day level.

**(f) Effective sample size is participants, not days.** Cluster-bootstrap
95% CIs at 212k test days are ±0.033 wide (A [0.634, 0.699]) — driven by 578
test participants. The 30-repeat SD (§1, ~0.013–0.016) quantifies split noise
only and understates day-pooling uncertainty. Boot mean ≈ exact (|Δ| ≤ 0.002)
confirms the pooled-day AUROC is cluster-bootstrap-unbiased.

**(g) The day-level signal is mostly level, then variability.**
`d_ch3000_mean` dominates (Gini 0.102, 4× the runner-up), followed by
`d_ch3000_sd`, `vmin/vmax`, and vendor-gap features. Within the constraints
of this cache, "how high and how variable was HR today" — no circadian shape
available.

**(h) The train-day cap is inert.** r=0 sensitivity: A(K=300) = 0.6525 ≈
A(K=100) = 0.6521 ≈ A_uncapped = 0.6540 (all within repeat noise) — the
K=300/participant cap neither helps nor hurts; day-rich users do not skew
training.

**Bottom line:** salutation information in heart-rate days is real from day
one (+0.04 over chance-referenced baselines is absent — AUROC 0.665 vs prior
0.647 is modest), integrates to ~0.73 within a month, and never reaches the
0.744–0.747 participant ceiling — the residual is cross-day and possibly
within-day structure, not more days.

## 7. Caveats

- Clustered units (days within participants): the 30-repeat SD is split noise; §2 bootstrap is day-pooling uncertainty.
- No within-day time-of-day decomposition (not cached); deferred to a targeted rescan.
- Adequate-day selection may interact with sex via compliance (medians M 467 / F 482 adequate days — near-null, recorded).
- ch3001/ch3002 vendor-processing caveat unchanged; ch3001 is literally a daily vendor value.
- Label = recorded salutation, not biological sex/gender.
- Exploratory; arm-B baselines are per-user, label-blind, own-days-only (transductive).
- Run note: 4 loky workers died (macOS memory pressure, 17:11–19:52 window);
  the tail completed on a single surviving worker; wall 9,699 s vs ~2.5 h at
  full 5-worker strength. No results affected (joblib re-queued the lost
  tasks); recorded for compute-planning.