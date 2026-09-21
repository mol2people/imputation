# Age counter vs age_group (Garmin, source 3) — results

Run 2026-09-20T18:07:12 | implements [`../PLAN.md`](../PLAN.md) | 30 FS-phase repeats (same splits/seeds), paired; 5 workers, n_jobs=1 per fit.

**REF50 replica validation:** max |delta AUROC| vs frozen A3_k50/A1 = 0.00e+00; top-50 sets identical on all 30/30 repeats = True

## 1. Absolute AUROC (mean ± SD over 30 repeats)

| arm | test AUROC | val AUROC (top-50) | val AUROC (hygiene) | n cols (hygiene) |
|---|---|---|---|---|
| `REF50` | 0.7445 ± 0.0165 | 0.7647 | 0.7466 | 400 |
| `AGE50` | 0.7475 ± 0.0175 | 0.7683 | 0.7514 | 398 |
| `AGE50_force` | 0.7475 ± 0.0175 | 0.7683 | 0.7514 ‡ | 398 |
| `AGE50_nodemo` | 0.7466 ± 0.0178 | 0.7677 | 0.7489 | 395 |
| `AGE_all` | 0.7470 ± 0.0162 | 0.7514 ‡ | 0.7514 | 398 |

‡ `AGE50_force` shares `AGE50`'s hygiene fit by construction (same matrix,
same seeds); `AGE_all` has no top-50 fit — its "top-50" column is the
hygiene fit (the only fit it ran).

## 2. Paired test-AUROC deltas (same repeats)

| comparison | mean Δ | 95% t-CI | SD | share > 0 |
|---|---|---|---|---|
| AGE50 − REF50 | +0.0029 | [-0.0008, +0.0067] | 0.0101 | 0.70 |
| AGE50_force − REF50 | +0.0029 | [-0.0008, +0.0067] | 0.0101 | 0.70 |
| AGE50_nodemo − REF50 | +0.0021 | [-0.0022, +0.0064] | 0.0114 | 0.70 |
| AGE_all − REF50 | +0.0025 | [-0.0012, +0.0061] | 0.0098 | 0.60 |
| AGE50_force − AGE50 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | 0.00 |

## 3. Age feature inside the AGE50 ranking

- selected into the top-50 in **30/30** repeats; overall PI rank median 1 (min 1, max 28).
- mean PI value of `age` = 0.00637 (AUROC decay per shuffled column, val split).
- **Direct comparison with the replaced feature:** in the frozen REF50 (`A3_k50`) top-50 lists, `demo__age_group=Elderly` was present in 29/30 repeats at median position 13 (mean 17.3, range 3–50). The continuous counter strictly dominates the categorical one-hot: 30/30 vs 29/30 membership, median rank 1 vs 13. In REF50 the only other demo-block members were the reference-level one-hots (`Middle Ager` 2/30, `Young Adults` 3/30) and near-absent bmi one-hots (`overweight` 2/30, `normal` 1/30).

## 4. AGE50 top-50 membership frequency (top 20)

| x/30 | column | REF50 x/30 |
|---|---|---|
| 30 | `rec__ch3000_weekend_minus_weekday` | 27 |
| 30 | `age` | 0 |
| 28 | `rec__ch3000_minus_ch3002_mean` | 26 |
| 25 | `win__ch3000_minus_ch3002_mean` | 25 |
| 24 | `win__ch3000_weekend_minus_weekday` | 24 |
| 23 | `rec__ch3000_minus_ch3001_mean` | 23 |
| 22 | `win__ch3000_minus_ch3001_mean` | 20 |
| 21 | `rec__ch3000_weekday_mean` | 19 |
| 20 | `rec__ch3002_sd_of_daily_mean` | 22 |
| 20 | `rec__ch3000_tod_morning` | 23 |
| 19 | `rec__ch3000_sd_of_daily_mean` | 21 |
| 18 | `rec__ch3002_mean_of_daily_sd` | 17 |
| 18 | `win__ch3002_sd_of_daily_mean` | 21 |
| 17 | `win__ch3000_sd_of_daily_mean` | 18 |
| 16 | `roll_rec__ch3001_daily_sd_w7_max` | 18 |
| 16 | `win__ch3001_sd_of_daily_mean` | 11 |
| 16 | `win__ch3000_tod_afternoon` | 13 |
| 16 | `rec__ch3000_tod_afternoon` | 15 |
| 16 | `roll_win__ch3001_daily_sd_w30_mean` | 12 |
| 15 | `win__ch3000_min_of_daily_min` | 10 |

## 5. Selection stability

- mean pairwise Jaccard across repeats within `AGE50`: 0.193 (FS-phase `all` A3_k50 was ≈0.19).
- mean same-repeat Jaccard `AGE50` vs `REF50`: 0.241.
- more often selected under `AGE50`: `rec__ch3000_weekend_minus_weekday`, `age`, `rec__ch3000_minus_ch3002_mean`, `win__ch3000_minus_ch3001_mean`, `rec__ch3000_weekday_mean`, `rec__ch3002_mean_of_daily_sd`, `win__ch3001_sd_of_daily_mean`, `win__ch3000_tod_afternoon`
- less often selected under `AGE50`: `demo__age_group=Elderly`, `rec__ch3000_tod_morning`, `rec__ch3002_sd_of_daily_mean`, `rec__ch3000_sd_of_daily_mean`, `win__ch3002_sd_of_daily_mean`, `win__ch3000_sd_of_daily_mean`, `roll_rec__ch3001_daily_sd_w7_max`, `roll_rec__ch3002_daily_sd_w7_max`

## 6. Univariate context and anchor sensitivity

- AUROC of `age` alone (window_start anchor): 0.5546; fixed-cutoff anchor (2022−B): 0.5579; `Elderly` one-hot: 0.5331.
- Spearman ρ between the two age definitions: 0.9944 (anchor choice is immaterial at this correlation).
- age at window_start: 20–86 y, median 50 (birth years 1935–2000, complete for all 3,848).

## 7. Caveats

- Exploratory; reuses the FS-phase participants and split seeds — same-partition reuse is never independent validation.
- Age is a linked profile covariate (like the demo block it replaces), not epoch-derived; its association with salutation is cohort structure, not physiology.
- Label is recorded salutation, not biological sex/gender; the vendor-processing caveat for ch3001/ch3002 is unchanged.
- 95% t-CIs quantify split/model randomization conditional on this cohort, not sampling or transportability.

## 8. Interpretation

**Headline: the counter beats the categorical, but the ceiling is participant-level structure, not age.** Swapping the five-level `demo__age_group` one-hot block for the single DOB-derived counter `age` moves paired test AUROC by +0.0029 (95% t-CI [−0.0008, +0.0067], share>0 0.70). This is a consistent, directionally stable improvement but its CI still touches zero; the honest reading is "equal-to-slightly-better, at 19% of the columns."

**Why it helps: resolution, not new information.** Both encodings draw on the same birth-year field, yet the counter (a) resolves within-band heterogeneity (a 62- vs 75-year-old are both "Elderly"), (b) is monotone in the underlying variable, so a tree needs one split where the one-hot block needs several, and (c) is selected *first* (median PI rank 1, membership 30/30) versus Elderly's median position 13 in REF50 — the forest spends its first split budget on it. Notably, `AGE50_force − AGE50 = 0.0000` exactly: `age` already made the top-50 in every repeat, so forcing it in changes nothing — selection is not the bottleneck.

**The gain is not merely a selection artifact.** `AGE_all` (no selection, all ~398 hygiene columns) lands at 0.7470, inside the AGE50 CI band, and `AGE50_nodemo` (age only, bmi dropped) at 0.7466 with Δ +0.0021 [−0.0022, +0.0064] — bmi one-hots were in REF50's top-50 only 1–2/30 repeats, i.e., noise-level, and their removal costs nothing. So the demo block's entire useful content was the Elderly contrast, and the counter subsumes it.

**What the swap does to the rest of the ranking is coherent, not churn.** Freed columns under AGE50 go to within-source contrast features (`rec__ch3000_minus_ch3002_mean`, `win__ch3000_minus_ch3001_mean`), weekday/weekend structure, and `rec__ch3002_mean_of_daily_sd` — i.e., exactly the cross-channel and temporal placement signals the participant-level reports flagged as the strongest HR-derived family. Displaced are diffuse sd-of-daily summaries and rolling maxima. Jaccard stability (0.193 within AGE50 ≈ FS-phase 0.19) says the selection instability is protocol-inherent, not age-specific.

**Magnitude context.** Univariate `age` AUROC 0.5546 vs `Elderly` 0.5331 — age is a weak single predictor that the forest integrates efficiently; the +0.003 net effect on a 0.745 base is small because the model already extracted most demographic signal through correlated availability/behavior proxies. Do not read this as "age explains salutation" — the association is cohort recruitment structure (who reports which salutation code), and the anchor check (ρ = 0.9944) confirms the definition, not the choice of anchor, drives anything.

**Practical recommendation.** Use the continuous counter in place of the age-group block going forward (strictly better or equal, cheaper, cleaner semantics); keep bmi out of small-selection pipelines (no measurable contribution here). The remaining ~0.75 ceiling on this cohort is set by participant-level chronic HR structure plus placement proxies, not by demographics — which is precisely what the day-level experiment (next) interrogates.
