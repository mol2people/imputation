# Days scaling — full-history pool (all adequate days)

`day_scaling_all.py` · 2026-09-24 · upstream `cache/bins_all/` from `scan_bins_all.py`
(1,635,698 adequate day-rows, 3,848 users, mean 425/user, min 80, p50 476, p90 539,
max 979; obs bins/day p50 283/288, zero-mask days 0).

## Design

- **Phase A — fixed-k curve** (k ∈ {4, 7, 14, 21, 30, 40, 60, 80}; 80 = min pool ⇒
  no per-k cohort selection), 10 allocations. Days drawn from **each user's entire
  adequate-day history** (uniform over their pool), via ONE nested random permutation
  per user per allocation (comp seed 5); k-points are prefixes, kernels shared across
  k within an allocation. MR kernels fit per allocation on 4 seeded days from train
  users' 80-day windows (comp seeds 0/1); per-allocation train-only mask-aware fill
  from those windows. Arms: MR α=1e3 / α=3e3 (B40Z ⊕ MRZ, 19,121 cols), Profile24,
  Summary_RF — identical hygiene/folds/fold-test gate to v2.
- **Phase B — all-days endpoint** ("up to mean"): pooled features over each user's
  **full** pool (variable 80–979 days, mean 425). SINGLE transform pass, one kernel
  draw (comp seed 6, fit on 4 seeded days from alloc-0 train users' full pools,
  alloc-0 fill), then hygiene/ridge/RF refit under **all 10 fold-splits**.
  Endpoint SD therefore reflects fold-split variation only (see caveats).

Wall: scan 183 s (4 workers, 0 errors) + experiment 1,736 s (~145 s/alloc, endpoint
460 s). Peak RSS **10.9 GiB — exceeds the 8 GiB v2 gate** (dayall table + pooled
slabs in RAM; recorded as a deviation, no failure/swapping).

## Results (mean over 10 allocs; endpoint = 10 fold-splits)

Test AUROC by days budget k:

| k | MR α=1e3 | MR α=3e3 | Profile24 | Summary_RF |
|---:|---:|---:|---:|---:|
| 4 | 0.713 | 0.735 | 0.686 | 0.669 |
| 7 | 0.760 | 0.780 | 0.695 | 0.676 |
| 14 | 0.799 | 0.816 | 0.714 | 0.686 |
| 21 | 0.824 | 0.839 | 0.722 | 0.696 |
| 30 | 0.845 | 0.857 | 0.727 | 0.698 |
| 40 | 0.858 | 0.868 | 0.731 | 0.702 |
| 60 | 0.875 | 0.881 | 0.736 | 0.709 |
| 80 | 0.885 | 0.889 | 0.740 | 0.710 |
| **all** (mean 425) | **0.926** | 0.918 | 0.744 | 0.711 |

Val AUROC (MR α=1e3): 0.719 → 0.879 (k=80) → 0.924 (all). MR-vs-P24 gap **widens**
with budget: +0.127 (k=40) → +0.146 (k=80) → **+0.182** (all days).

### Marginal gains (test, MR α=1e3)

| budget step | AUROC gain / day |
|---|---:|
| 4→7 | 0.0158 |
| 7→14 | 0.0055 |
| 14→21 | 0.0035 |
| 21→30 | 0.0024 |
| 30→40 | 0.0013 |
| 40→60 | 0.0009 |
| 60→80 | 0.0005 |
| 80→all (+345 d) | 0.0001 |

Concave in raw days, roughly log-like through k=80 (~+0.02 per doubling), still
strictly positive at the endpoint: +0.041 from k=80 to full history.

### Pool effect at matched k (full history vs first-40-day window, paired, 10 allocs, test)

| arm | k=4 | k=7 | k=21 | k=30 | k=40 |
|---|---|---|---|---|---|
| MR α=1e3 | −0.001 | +0.016* | +0.019** | +0.020** | +0.022** |
| MR α=3e3 | −0.000 | +0.014* | +0.016** | +0.019** | +0.021** |
| Profile24 | −0.005 | −0.010* | −0.008 | −0.002 | −0.006 |
| Summary_RF | −0.007 | −0.012 | +0.001 | −0.002 | −0.001 |

\* CI excludes 0 at p<0.05; \** p<0.01. **The position effect is MR-specific**:
sampling across the full history beats the early window by ~+0.02 at matched k for
the transform, while Profile24/RF are flat-to-negative (cheap arms don't care which
days they get; the transform exploits the more varied later days).

### Shrinkage × budget (α flip)

At k ≤ 21, α=3e3 > α=1e3 (+0.02 at k=4); at k=80 they converge (+0.004); at the
endpoint **α=1e3 > α=3e3 (+0.009)** — more pooling needs less shrinkage. With α
frozen at 1e3, the reported curve is a **floor**: the endpoint optimum is likely
below 1e3, so the full-history ceiling is probably higher than 0.926.

### Reference points

- v2 frozen (day40 window): MR 0.836, Combined (MR⊕HYDRA) **0.857** test.
- Full-history MR α=1e3 alone: **0.926** test — beats the entire v2 representation
  ladder by +0.070 without HYDRA or α tuning, using data already on disk.
- Endpoint control for pool-size leakage: from k=80 → all-days, Profile24 gains
  +0.005 and Summary_RF +0.001 (they use the same full-pool B40/hour statics),
  while MR gains +0.041 — the endpoint gain is transform pooling, not a "how much
  data does this user have" artifact carried by the static features.

## Caveats (recorded)

1. **Endpoint single kernel draw**: SD 0.0055 (fold-only); a fold+kernel-comparable
   SD is ~0.011–0.015 (cf. k=80: 0.0115). The k=80→endpoint gain (+0.041) is far
   beyond both.
2. **Endpoint budget heterogeneity**: "all days" = each user's own pool (80–979);
   pool size correlates with engagement. The P24/RF control above argues the
   confound is small, but the endpoint is "use everything", not a fixed 425-day
   budget.
3. **α frozen** (1e3/3e3 only): true per-k optima would shift the curve up at both
   ends (less shrinkage at large k, more at small k).
4. **Cohort**: long-history Garmin export (≥80 adequate days, mean 425) — generalizes
   to long-history users, not the export median (~40 days).
5. **Peak RSS 10.9 GiB > 8 GiB v2 gate** (exploratory run; deviation recorded).
6. Endpoint kernels fit on alloc-0 train users' full-pool days (comp seed 6) and
   evaluated under all 10 fold-splits — mild unsupervised cross-split exposure of
   val/test days to bias quantiles (same class of exposure as the v2 fill/fit
   protocol; documented here).

## Files

- `results/days_scaling_all_metrics.csv` — per (alloc, k, arm) rows; k = −1 ⇒ all-days endpoint.
- `cache/bins_all/*.npy` — full-history 288-bin hr/mask (memmap), user/date/days_per_user/starts.
- `cache/scan_all_stdout.log`, `cache/day_scaling_all_stdout.log` — run logs.
