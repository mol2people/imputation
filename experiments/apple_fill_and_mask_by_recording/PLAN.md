# Apple ch3000 filling, difference-masking, and explicit mask sequences by recording pattern

2026-10-19 (amended 2026-10-19 per review R1–R6). Plan only; no model fitting or execution in this change.

## 1. Question and scope

In the full Apple Watch benchmark ($N = 11,139$, [`experiments/apple_cohort_scaling/REPORT.md`](../apple_cohort_scaling/REPORT.md)), 5-minute MultiRocket plus summaries exceeds 15-minute MultiRocket plus summaries by $+0.0175 \pm 0.0082$ test AUROC overall under validation-selected $\alpha$ ($+0.0213 \pm 0.0068$ at fixed $\alpha = 1000$). Stratifying the saved predictions by training-derived recording-pattern terciles ([`BINNING_BY_RECORDING.md`](../apple_cohort_scaling/BINNING_BY_RECORDING.md)) showed that low within-day occupancy alone does not remove the 5-minute advantage:

- **High continuity + low occupancy** (mean 5m occupancy $40.7\%$, median adequate-day continuity $\approx 99\%$): 5m leads 15m by **$+0.0246 \pm 0.0291$** ($0.8700$ vs. $0.8454$, all-days selected $\alpha$).
- **Low continuity + low occupancy**: near tie, **$+0.0058 \pm 0.0211$** ($0.8228$ vs. $0.8170$ all days) and **$-0.0042 \pm 0.0206$** at $\le 40$ adequate days.
- **Low history + high occupancy** (median $\approx 65$ adequate days, $71.2\%$ 5m occupancy, $97.7\%$ median continuity): near tie, **$+0.0056 \pm 0.0209$** all days ($0.8130$ vs. $0.8073$) and **$+0.0001 \pm 0.0276$** at $\le 40$ adequate days ($0.8001$ vs. $0.8000$).
- **Middle history + high occupancy**: 5m leads by **$+0.0354 \pm 0.0161$** ($0.8950$ vs. $0.8596$, positive in 10/10 allocations).

Because coarsening from 5 minutes ($L = 288$) to 15 minutes ($L = 96$) simultaneously alters (i) missing-bin fraction ($46.0\% \to 31.7\%$), (ii) step-jumps at observed$\leftrightarrow$imputed boundaries in the first-difference series $\Delta X_t = X_{t+1} - X_t$, (iii) within-gap levels and slopes from clock-bin median filling, (iv) physical kernel duration ($9 \times d \times \Delta t$), and (v) static mask-feature dimensionality ($288 \to 96$), the existing saved predictions cannot isolate why 5m leads in some sparse profiles and ties in others.

This experiment holds the cohort, allocations, and cached daily grids fixed and evaluates the sensitivity of fitted predictors across the saved `continuity × occupancy × history` subgroups to four input and representation choices:

1. **Clock-versus-global filling contrast (`H_clock` vs. `H_global`)**: Replacing the time-varying training clock-bin median $\mu^{\text{clock}}_t$ with a constant training global median $\mu^{\text{global}}$ changes imputed levels in $X$, boundary-jump amplitudes $(h_t - \mu)$ at gap edges, and within-gap slopes ($\Delta \mu^{\text{global}} = 0$ vs. $\Delta \mu^{\text{clock}}_t$) together. This contrast measures net pipeline sensitivity to clock-specific median imputation.
2. **Withholding gap-related difference values (`H_clean_diff`)**: In `MultiRocket`, half of the $18,816$ sequence features come from the first difference $\Delta X_t = X_{t+1} - X_t$. Under `H_clock`, transitions with $M_t \neq M_{t+1}$ take values $\mu^{\text{clock}}_{t+1} - h_t$ or $h_{t+1} - \mu^{\text{clock}}_t$ (encoding both missingness timing and the participant's HR deviation from the population median at entry/exit), while interior gap transitions $(M_t = M_{t+1} = 0)$ take the template slope $\mu^{\text{clock}}_{t+1} - \mu^{\text{clock}}_t$. Setting $\Delta^{\text{clean}} X_t = 0$ whenever $M_t = 0$ or $M_{t+1} = 0$ withholds both boundary jumps and interior template slopes from the $\Delta X$ branch while leaving the base branch $X^{\text{clock}}$ and the zero-location pattern in $\Delta^{\text{clean}} X$ intact.
3. **Span-matched short-gap linear interpolation (`H_short_interp`)**: Tests whether linearly bridging interior gaps with bounding bin centers separated by at most $30\text{ minutes}$—i.e., $(k + 1)\Delta t \le 30\text{ min}$, corresponding to $1 \le k \le 5$ missing bins at 5m ($10\text{–}30\text{ min}$ center separation) and $k = 1$ missing bin at 15m ($30\text{ min}$ center separation)—alters 5m and 15m discrimination while leaving wider or boundary gaps on $\mu^{\text{clock}}_t$. Because cached bins do not retain sub-bin event timestamps or sensor state, short vs. long gaps are operational run-length categories rather than verified passive-sampling or off-wrist states.
4. **Explicit binary observation-mask representations (`M_static`, `M_temporal`, `H_plus_M_concat`, `H_plus_M_stack`)**: Evaluates how much discrimination and $5\text{m} - 15\text{m}$ subgroup variation is achieved by static and MultiRocket representations of the binary mask $M_{u,d,t} \in \{0,1\}$ without HR values (`M_static`, `M_temporal`), and tests whether combining HR and mask representations via joint feature concatenation (`H_plus_M_concat`) or validation-tuned two-score blending (`H_plus_M_stack`) improves over `H_clock`.

Target: recorded salutation (`10 → 0/F`, `20 → 1/M`), a profile field rather than biological sex or gender identity. Source 6 (Apple Watch), channel `3000` HR only.

---

## 2. Read-only inputs and workspace isolation

All code, diagnostics, and outputs for this study live strictly under `experiments/apple_fill_and_mask_by_recording/`. Existing files outside this directory are never modified. No packages are installed; use `/Users/bulat/micromamba/envs/datenspende/bin/python`.

Read-only inputs (relative to repository root):

- `experiments/apple_cohort_scaling/artifacts/cache_5min/customers.csv` and `customer_{cid}.npz` ($N = 11,139$; keys `date`, `stats`, `hr` with shape $(D_u, 288)$ in `float32`, `NaN` where unobserved).
- `experiments/apple_cohort_scaling/artifacts/cache_15min/customers.csv` and `customer_{cid}.npz` (same participants and dates; `hr` with shape $(D_u, 96)$ in `float32`).
- `experiments/apple_cohort_scaling/artifacts/apple_10_allocations.parquet` (columns `customer_id`, `y`, `age_group`, `bmi_group`, `alloc_0` … `alloc_9` with $7,797$ train / $1,671$ val / $1,671$ test per allocation).
- `experiments/apple_cohort_scaling/artifacts/scaling_predictions_test.parquet` and `apple_benchmark_metrics.csv` (reference scores and metrics for `H_clock` parity verification).
- `experiments/apple_cohort_scaling/artifacts/binning_by_recording/training_cutpoints.csv` and `summary.csv` (reference subgroup cutpoints and baseline 5m vs. 15m contrasts).

No raw `out/{user_id}.csv` files are read. Because the Apple `.npz` caches store `hr` (`NaN` when empty) and daily `stats` without per-bin event counts `cnt`, the binary observation mask is defined as:
$$M_{u,d,t} = \mathbb{I}\!\left(\text{isfinite}(h_{u,d,t})\right) \in \{0, 1\}.$$
Count-sequence arms (`C_temporal`) are out of scope unless a new cache scan is commissioned later.

---

## 3. Stage 0: Cache exposure and mask-degeneracy audit (before model fitting)

Before running MultiRocket across allocations, execute a fast read-only diagnostic script (`scripts/01_audit_gap_and_mask_exposure.py`) over the cached 5m and 15m arrays and save aggregate tables to `results/exposure_overall.csv` and `results/exposure_by_subgroup.csv`:

1. **Adjacent-bin transition and gap-span anatomy ($t \to t+1$, $L - 1$ pairs per day)**:
   For each participant $u$, resolution $L \in \{288, 96\}$, and history budget (`all_days` vs. `up_to_40d`), compute the fraction of daily transitions in mutually exclusive sets:
   - **Observed-to-observed** $\mathcal{T}_{11} = \{t : M_t = 1 \land M_{t+1} = 1\}$.
   - **Span-matched short-gap transitions** $\mathcal{T}_{\text{short},30\text{m}}$: transitions touching or inside an interior missing run of length $1 \le k \le k_{\max}$ bounded on both sides by observed bins within the same day with center separation $(k+1)\Delta t \le 30\text{ min}$ ($k_{\max} = 5$ at 5m; $k_{\max} = 1$ at 15m). At 5m, also separately record the sub-share with $1 \le k \le 2$ ($10\text{–}15\text{ min}$ center separation) vs. $3 \le k \le 5$ ($20\text{–}30\text{ min}$ center separation).
   - **Long/boundary gap edges** $\mathcal{T}_{\text{edge}}$: remaining transitions with $M_t \neq M_{t+1}$ belonging to interior missing runs of length $k > k_{\max}$ or day-boundary missing runs.
   - **Long/boundary gap interiors** $\mathcal{T}_{00,\text{long}}$: remaining transitions with $M_t = 0 \land M_{t+1} = 0$.
   Also report per-participant missing-bin fractions and the fraction of missing bins eligible for `H_short_interp` at 5m and 15m (mirroring the exposure audit in [`filling_sensitivity_garmin/results/REPORT.md`](../filling_sensitivity_garmin/results/REPORT.md#L68-L94)).
2. **Difference-series magnitude across transition regimes**:
   Using allocation-0 training clock-bin medians $\mu^{\text{clock}}_t$ (and global median $\mu^{\text{global}}$), compute for each participant the RMS and mean absolute value of $\Delta X_t = X_{t+1} - X_t$ separately over:
   - Observed-to-observed transitions $t \in \mathcal{T}_{11}$: $|h_{t+1} - h_t|$.
   - Gap-boundary transitions $t : M_t \neq M_{t+1}$: $|\mu^{\text{clock}}_{t+1} - h_t|$ or $|h_{t+1} - \mu^{\text{clock}}_t|$ (and under $\mu^{\text{global}}$).
   - Gap-interior transitions $t : M_t = 0 \land M_{t+1} = 0$: $|\mu^{\text{clock}}_{t+1} - \mu^{\text{clock}}_t|$.
   Summarize medians and IQRs overall and across the training-derived `continuity × occupancy` and `history × occupancy` tercile cells.
3. **Binary-mask MultiRocket bias degeneracy check**:
   On allocation-0 sampled training days ($n = 4,096$ days), fit `mr_fit` on $M \in \{0.0, 1.0\}$ and $\Delta M \in \{-1.0, 0.0, 1.0\}$ at 5m ($L = 288$) and 15m ($L = 96$). Record:
   - The number of unique bias thresholds per kernel/dilation slot.
   - The number of post-transform daily features (out of $9,408$) and pooled participant features (out of $18,816$) dropped by `VarianceThreshold(0.0)` on training participants.

---

## 4. Stage 1: Experimental arms and mathematical definitions

Follow the repository's pilot discipline ([`recording_structure_garmin/PLAN.md`](../recording_structure_garmin/PLAN.md), [`filling_sensitivity_garmin/PLAN.md`](../filling_sensitivity_garmin/PLAN.md)): evaluate **allocations 0, 1, 2** first in the primary pass across both resolutions (**5m** [$L = 288$] and **15m** [$L = 96$]) and both history budgets (**`all_days`** and **`up_to_40d`**). Design the runner (`scripts/02_run_fill_and_mask_benchmark.py`) with `--allocations` so extending to allocations 3–9 requires no code changes if warranted by the 3-allocation readout.

### 4.1 Static feature blocks

- **$B_L(u)$** ($305$ cols at 5m, $113$ cols at 15m): Existing static summary block from [`04_run_apple_benchmark.py`](../apple_cohort_scaling/scripts/04_run_apple_benchmark.py) (mean and sample SD of 8 daily stats over the selected days, weekday-minus-weekend mean-HR contrast, and $L$ per-bin mean observation frequencies $\bar{M}_{u,\cdot,t}$). Computed once from raw cached `days` and `hr` before any filling; never recomputed from filled HR.
- **$S(M)_L(u)$** ($2L$ cols: $576$ at 5m, $192$ at 15m): Per-bin across-day mean $\bar{M}_{u,\cdot,t}$ and sample SD $s_{M,u,t}$ (with `ddof=1`, or $0$ if $D_u \le 1$) of the binary mask $M_{u,d,t}$ across the participant's selected days. Note that for binary $M_{u,d,t} \in \{0,1\}$ and $D_u > 1$,
  $$s_{M,u,t}^2 = \frac{D_u}{D_u - 1}\,\bar{M}_{u,\cdot,t}\!\left(1 - \bar{M}_{u,\cdot,t}\right),$$
  so conditional on $D_u$, $s_{M,u,t}$ carries no statistical information beyond $\bar{M}_{u,\cdot,t}$, though it supplies a nonlinear basis function to the linear Ridge classifier. It is retained for schema comparability with [`recording_structure_garmin`](../recording_structure_garmin/PLAN.md).
- **$S(M)_{\text{SD},L}(u)$** ($L$ cols: $288$ at 5m, $96$ at 15m): Only the across-day sample SD half $s_{M,u,t}$, used when appending to $B_L(u)$ because $\bar{M}_{u,\cdot,t}$ is already present in $B_L(u)$.

### 4.2 Sequence transforms $T_L(X, \Delta X)$

Vendored `MultiRocket` (`num_features=1250`, `max_dilations_per_kernel=32`, `n_features_per_kernel=4`), producing $4,704$ daily features from base series $X \in \mathbb{R}^{D_u \times L}$ and $4,704$ daily features from difference series $\Delta X \in \mathbb{R}^{D_u \times (L-1)}$ ($9,408$ daily features total). Pooled across each participant's $D_u$ selected days by arithmetic mean and sample SD (`ddof=1`) into $18,816$ sequence features. All days in the selected budget are pooled (all cached days are adequate by construction).

### 4.3 Arm table (evaluated at $L \in \{288, 96\}$ and budget $\in \{\text{all\_days}, \text{up\_to\_40d}\}$)

| Arm | Base input $X_{u,d,t}$ | Difference input $\Delta X_{u,d,t}$ | Participant features | Width (5m / 15m) | Role |
|---|---|---|---|---:|---|
| `H_clock` | $h_{u,d,t}$ if $M=1$, else $\mu^{\text{clock}}_t$ | $X^{\text{clock}}_{u,d,t+1} - X^{\text{clock}}_{u,d,t}$ | $B_L + T(X^{\text{clock}}, \Delta X^{\text{clock}})$ | $19,121$ / $18,929$ | Reference arm; must match saved `apple_cohort_scaling` scores on allocs 0–2. |
| `H_global` | $h_{u,d,t}$ if $M=1$, else $\mu^{\text{global}}$ | $X^{\text{global}}_{u,d,t+1} - X^{\text{global}}_{u,d,t}$ | $B_L + T(X^{\text{global}}, \Delta X^{\text{global}})$ | $19,121$ / $18,929$ | Clock-versus-global filling contrast: replaces $\mu^{\text{clock}}_t$ with constant $\mu^{\text{global}}$ (changing gap levels, boundary-jump sizes, and within-gap slopes together). |
| `H_clean_diff` | $h_{u,d,t}$ if $M=1$, else $\mu^{\text{clock}}_t$ | $\Delta^{\text{clean}} X_{u,d,t}$ (defined piecewise below) | $B_L + T(X^{\text{clock}}, \Delta^{\text{clean}} X)$ | $19,121$ / $18,929$ | Withholds gap-related difference values (both $M_t \neq M_{t+1}$ boundary jumps and $M_t=M_{t+1}=0$ template slopes) by setting $\Delta^{\text{clean}} X_t = 0$ whenever either endpoint is missing, holding base $X^{\text{clock}}$ fixed. |
| `H_short_interp` | Linear interpolation across interior missing runs of $1 \le k \le k_{\max}$ ($k_{\max}=5$ at 5m, $1$ at 15m; center span $\le 30\text{ min}$) bounded by $M=1$ on the same day; $\mu^{\text{clock}}_t$ elsewhere | $X^{\text{interp}}_{u,d,t+1} - X^{\text{interp}}_{u,d,t}$ | $B_L + T(X^{\text{interp}}, \Delta X^{\text{interp}})$ | $19,121$ / $18,929$ | Span-matched short-gap interpolation ($\le 30\text{ min}$ between bounding observed bin centers at both resolutions); leaves longer and boundary gaps on $\mu^{\text{clock}}_t$. |
| `M_static` | None | None | $S(M)_L$ | $576$ / $192$ | Static observation-mask mean + SD baseline (no HR values). |
| `M_temporal` | $M_{u,d,t} \in \{0.0, 1.0\}$ | $M_{u,d,t+1} - M_{u,d,t} \in \{-1.0, 0.0, 1.0\}$ | $S(M)_L + T(M, \Delta M)$ | $19,392$ / $19,008$ | Richer mask sequence representation (nonlinear occupancy/gap/arrangement features; no HR values). |
| `H_plus_M_concat` | `H_clock` and `M_temporal` blocks | Same | $B_L + S(M)_{\text{SD},L} + T_{\text{HR}} + T_M$ | $38,225$ / $37,841$ | Joint single-penalty Ridge fit over concatenated HR and mask blocks (analogous to Garmin `H_recording`). |
| `H_plus_M_stack` | Separate Ridge scores from `H_clock` and `M_temporal` | Same | Convex combination $(1 - w)\,\tilde{s}_{\text{HR}} + w\,\tilde{s}_M$ of validation-standardized Ridge scores | $2$ (meta) | Two-score blending: selects $\alpha_{\text{HR}}$ and $\alpha_M$ separately on validation, then selects $w \in \{0.00, 0.05, \dots, 1.00\}$ on validation AUROC. |

### 4.4 Piecewise definition of `H_clean_diff` and interpolation rule for `H_short_interp`

1. **Piecewise difference (`H_clean_diff`)**:
   Because unobserved entries in cached `hr` are `NaN` (where IEEE-754 multiplication `NaN * 0` yields `NaN`), $\Delta^{\text{clean}} X_{u,d,t}$ for $t \in \{0, \dots, L-2\}$ must be constructed piecewise (e.g., via `np.where(M[:, :-1] & M[:, 1:], np.diff(X_clock, axis=1), 0.0)`):
   $$\Delta^{\text{clean}} X_{u,d,t} = \begin{cases} h_{u,d,t+1} - h_{u,d,t}, & \text{if } M_{u,d,t} = 1 \text{ and } M_{u,d,t+1} = 1, \\ 0.0, & \text{otherwise.} \end{cases}$$
2. **Span-matched short-gap linear interpolation (`H_short_interp`)**:
   Following [`filling_sensitivity_garmin/PLAN.md`](../filling_sensitivity_garmin/PLAN.md), an interior missing run of length $k \in \{1, \dots, k_{\max}\}$ at indices $t_L + 1, \dots, t_R - 1$ (where $t_R - t_L = k + 1$, $0 \le t_L < t_R < L$, $M_{u,d,t_L} = 1$, $M_{u,d,t_R} = 1$, and $M_{u,d,t_L + j} = 0$ for $1 \le j \le k$) has bounding bin centers separated by $(k + 1)\Delta t$. To match the maximum bridged physical separation at $\le 30\text{ minutes}$ across both resolutions (and match the $6$-bin / $30$-minute threshold in `filling_sensitivity_garmin`), set $k_{\max} = 5$ at 5m ($\Delta t = 5\text{ min}$, $(k+1)\Delta t \in [10, 30]\text{ min}$) and $k_{\max} = 1$ at 15m ($\Delta t = 15\text{ min}$, $(k+1)\Delta t = 30\text{ min}$). Fill each eligible run within the same participant-day as:
   $$X^{\text{interp}}_{u,d,t_L + j} = h_{u,d,t_L} + \frac{j}{k + 1}\left(h_{u,d,t_R} - h_{u,d,t_L}\right), \quad j = 1, \dots, k.$$
   Runs with $k > k_{\max}$ or touching day boundaries ($t = 0$ or $t = L - 1$) remain on $\mu^{\text{clock}}_t$. Never interpolate across midnight or use filled values as endpoints.

---

## 5. Matched fitting, seeds, and regularization

1. **Exact seed and day-sampling alignment with `apple_cohort_scaling`**:
   For allocation $a \in \{0, \dots, 9\}$, budget $b \in \{\text{all\_days}, \text{up\_to\_40d}\}$, and resolution $r \in \{\text{5m}, \text{15m}\}$, define:
   $$\text{seed\_base}(a, b, r) = 20261014 + 100a + 10\,\mathbb{I}(b = \text{up\_to\_40d}) + \mathbb{I}(r = \text{15m}).$$
   - Sample the exact same $\le 4,096$ training participant-days `(cid, pick)` using `np.random.default_rng(seed_base)` as in [`04_run_apple_benchmark.py:219-233`](../apple_cohort_scaling/scripts/04_run_apple_benchmark.py#L219-L233).
   - Reuse those exact sampled training participant-days across all HR arms (`H_clock`, `H_global`, `H_clean_diff`, `H_short_interp`) and `M_temporal` within $(a, b, r)$.
   - For HR arms, call `seed_numba(seed_base)` before `mr_fit` on $X$ and `seed_numba(seed_base + 1)` before `mr_fit` on $\Delta X$ (so `H_clock` is bit-aligned with `04_run_apple_benchmark.py`).
   - For `M_temporal`, call `seed_numba(seed_base + 50)` before `mr_fit` on $M$ and `seed_numba(seed_base + 51)` before `mr_fit` on $\Delta M$.
   - Reuse the fitted `H_clock` and `M_temporal` feature blocks to form `H_plus_M_concat` and `H_plus_M_stack` without re-running `mr_transform`.
2. **Train-only preprocessing**:
   For each block (`H_clock`, `H_global`, `H_clean_diff`, `H_short_interp`, `M_static`, `M_temporal`, `H_plus_M_concat`), fit `SimpleImputer(strategy="median") -> VarianceThreshold(0.0) -> StandardScaler()` strictly on training participants ($N_{\text{train}} = 7,797$) and transform validation ($1,671$) and test ($1,671$) participants.
3. **Regularization grid and selection**:
   - Candidate grid: $\alpha \in \{100, 500, 1000, 5000, 10000, 100000\}$.
   - Record results under three penalty rules:
     - `ridge_selected_ext` (**primary readout**): $\alpha \in \{100, 500, 1000, 5000, 10000, 100000\}$, accommodating stronger shrinkage needed by wide concatenated or mask-only blocks (as in [`recording_structure_garmin`](../recording_structure_garmin/results/REPORT.md)).
     - `ridge_selected` (secondary parity readout): $\alpha \in \{100, 500, 1000, 5000, 10000\}$, matching [`04_run_apple_benchmark.py`](../apple_cohort_scaling/scripts/04_run_apple_benchmark.py#L46).
     - `ridge_fixed` (secondary sensitivity readout): $\alpha = 1000.0$.
   - Match the exact `np.isclose` tie-breaking rule in [`04_run_apple_benchmark.py:267-268`](../apple_cohort_scaling/scripts/04_run_apple_benchmark.py#L267-L268):
     ```python
     max_auc = max(val_aurocs.values())
     best_alpha = max([alp for alp, auc in val_aurocs.items() if np.isclose(auc, max_auc, atol=1e-12)])
     ```
4. **Two-score blending (`H_plus_M_stack`)**:
   Within each allocation $(a, b, r)$ and penalty rule (`ridge_selected_ext`, `ridge_selected`, `ridge_fixed`):
   - Take validation and test scores $s_{\text{HR}}$ from `H_clock` and $s_M$ from `M_temporal`.
   - Standardize each score stream using its validation mean and validation sample SD (`ddof=1`; if $\hat{\sigma}_{\text{val}} == 0$, set standardized stream to $0$):
     $$\tilde{s}_{\text{HR}} = \frac{s_{\text{HR}} - \hat{\mu}_{\text{val,HR}}}{\hat{\sigma}_{\text{val,HR}}}, \quad \tilde{s}_M = \frac{s_M - \hat{\mu}_{\text{val},M}}{\hat{\sigma}_{\text{val},M}}.$$
   - Evaluate $s(w) = (1 - w)\,\tilde{s}_{\text{HR}} + w\,\tilde{s}_M$ for $w \in \{0.00, 0.05, 0.10, \dots, 1.00\}$ on validation AUROC; break near-ties (`np.isclose(auc, max_auc, atol=1e-12)`) by choosing the smallest $w$ (favoring `H_clock`). Apply the validation-selected $w^*$ unchanged to test scores. Note that including $w = 0$ guarantees validation AUROC at least equal to `H_clock`, but does not guarantee test-set improvement and adds one validation-selected hyperparameter over restricted scalar score streams.

---

## 6. Subgroup readout and primary contrasts

**Primary readout**: **`all_days` budget under `ridge_selected_ext`** across allocations 0, 1, 2. Retain `ridge_selected`, `ridge_fixed`, and `up_to_40d` in saved CSVs as secondary parity and history-capping sensitivity checks.

Using the exact subgroup logic from [`06_summarize_binning_by_recording.py`](../apple_cohort_scaling/scripts/06_summarize_binning_by_recording.py) (training-participant terciles of full-history `n_days`, `occupancy_rate_5m`, `adequate_day_density` [continuity], and `daily_occupancy_sd`), compute within-allocation test AUROCs and paired deltas across:

- **Overall cohort** ($N_{\text{test}} = 1,671$ per allocation)
- **Marginal terciles** of `history`, `occupancy`, `continuity`, and `occupancy_variability` (12 groups)
- **Crossed terciles** `continuity_x_occupancy` (9 groups) and `history_x_occupancy` (9 groups), including the four focal profiles previously identified in [`BINNING_BY_RECORDING.md`](../apple_cohort_scaling/BINNING_BY_RECORDING.md):
  1. `continuity_x_occupancy = low/low` (sparse within and across days)
  2. `continuity_x_occupancy = high/low` (sparse within days, consistent across dates)
  3. `history_x_occupancy = low/high` (short dense histories)
  4. `history_x_occupancy = middle/high` (medium-length dense histories)

### Key paired contrasts (computed within allocation, budget, resolution, and subgroup)

1. **Clock-versus-global filling contrast**: $\text{AUROC}(\text{H\_global}) - \text{AUROC}(\text{H\_clock})$.
2. **Withholding gap-related differences**: $\text{AUROC}(\text{H\_clean\_diff}) - \text{AUROC}(\text{H\_clock})$ and $\text{AUROC}(\text{H\_clean\_diff}) - \text{AUROC}(\text{H\_global})$.
3. **Span-matched short-gap interpolation contrast**: $\text{AUROC}(\text{H\_short\_interp}) - \text{AUROC}(\text{H\_clock})$.
4. **Mask representation increment**: $\text{AUROC}(\text{M\_temporal}) - \text{AUROC}(\text{M\_static})$ (measures the gain from adding pooled MultiRocket features of $M$ and $\Delta M$ over static per-bin means/SDs; does not isolate temporal order from occupancy/gap nonlinearities).
5. **Joint concatenation and two-score blending contrasts**:
   - $\text{AUROC}(\text{H\_plus\_M\_concat}) - \text{AUROC}(\text{H\_clock})$.
   - $\text{AUROC}(\text{H\_plus\_M\_stack}) - \text{AUROC}(\text{H\_clock})$.
   - $\text{AUROC}(\text{H\_plus\_M\_stack}) - \text{AUROC}(\text{H\_plus\_M\_concat})$.
6. **Resolution contrast ($5\text{m} - 15\text{m}$) within each arm**:
   $$\Delta_{5\text{m}-15\text{m}}(\text{arm}) = \text{AUROC}_{5\text{m}}(\text{arm}) - \text{AUROC}_{15\text{m}}(\text{arm}).$$
   Compare $\Delta_{5\text{m}-15\text{m}}(\text{H\_clock})$ against $\Delta_{5\text{m}-15\text{m}}(\text{H\_global})$, $\Delta_{5\text{m}-15\text{m}}(\text{H\_clean\_diff})$, $\Delta_{5\text{m}-15\text{m}}(\text{H\_short\_interp})$, and $\Delta_{5\text{m}-15\text{m}}(\text{M\_temporal})$.

Also report paired test-score Spearman rank correlations $\rho(\text{arm}, \text{H\_clock})$ per allocation to detect rank-order sensitivity even when aggregate AUROC is stable.

---

## 7. Verification gates and deliverables

1. **Synthetic unit fixture (`scripts/00_verify_transforms.py`)**:
   Before touching full-cohort fits, test on a small deterministic fixture ($20$ synthetic days with $1$-bin, $2$-bin, $5$-bin, $6$-bin, and boundary gaps, plus adjacent observed bins with identical HR values):
   - `H_clean_diff` produces zero `NaN` values, equals $0.0$ whenever $M_t = 0$ or $M_{t+1} = 0$, and equals $h_{t+1} - h_t$ whenever $M_t = M_{t+1} = 1$ (including $0.0$ when $h_{t+1} = h_t$).
   - `H_short_interp` linearly interpolates interior runs of length $1 \le k \le k_{\max}$ ($k_{\max}=5$ at 5m, $1$ at 15m), leaves $k > k_{\max}$ and boundary runs untouched on $\mu^{\text{clock}}_t$, and preserves all observed bins ($M_t = 1$) bit-exactly.
   - Train-only assertions: `fill_clock`, `fill_global`, imputer, variance filter, and scaler depend strictly on `train` participant IDs.
2. **Exact reference parity gate on Allocation 0 (`H_clock`)**:
   In allocation 0 (and every executed allocation), compare `H_clock` test predictions against `experiments/apple_cohort_scaling/artifacts/scaling_predictions_test.parquet`:
   - Assert $\max |\Delta \text{score}| \le 10^{-5}$ and $|\Delta \text{AUROC}| \le 10^{-10}$ for both `all_days` and `up_to_40d` at 5m and 15m under `ridge_fixed` and `ridge_selected`. Halt and diagnose any discrepancy before running or interpreting other arms.
3. **Deliverable files under `experiments/apple_fill_and_mask_by_recording/`**:
   - `PLAN.md` (this protocol)
   - `scripts/00_verify_transforms.py` (unit fixtures for piecewise difference masking, span-matched short-gap interpolation, and train-only hygiene)
   - `scripts/01_audit_gap_and_mask_exposure.py` (Stage 0 cache transition/jump anatomy & `T(M)` bias-degeneracy audit)
   - `scripts/02_run_fill_and_mask_benchmark.py` (Stage 1 batched MultiRocket benchmark runner with `--allocations` and `--smoke` flags)
   - `scripts/03_summarize_subgroup_contrasts.py` (Subgroup AUROC, paired deltas, $5\text{m}-15\text{m}$ arm contrasts, and Spearman $\rho$ tables)
   - `results/` (populated when Stage 0 / Stage 1 are executed: `exposure_overall.csv`, `exposure_by_subgroup.csv`, `benchmark_metrics.csv`, `test_predictions.parquet`, `subgroup_summary.csv`, `REPORT.md`)

---

## 8. Interpretation limits and stopping rule

- **Exploratory status and reused splits**: Allocations 0, 1, and 2 reuse the existing OR-Tools splits from `apple_cohort_scaling`, and the four focal recording-pattern subgroups were selected after inspecting 10-allocation test results in [`BINNING_BY_RECORDING.md`](../apple_cohort_scaling/BINNING_BY_RECORDING.md). Do not report p-values or population confidence intervals; report per-allocation values, mean, SD/range, and sign agreement (e.g., `3/3 +`).
- **Operational advancement heuristic**: Stop after Stage 0 and the 3-allocation Stage 1 readout (`results/REPORT.md`). Treat a shift of roughly $\ge 0.01$ AUROC in the `all_days` / `ridge_selected_ext` primary readout with consistent direction across all 3 allocations as an operational prioritization heuristic for extending specific arms to allocations 3–9 (which assesses stability across the remaining reused splits, not independent confirmation).
- **Scope of inference**: Even when holding participants, dates, and kernels fixed, changing the fill rule (`H_clock`, `H_global`, `H_short_interp`) or $\Delta X$ rule (`H_clean_diff`) refits the MultiRocket bias quantiles and Ridge weights on the modified training sequences. Resulting AUROC differences measure the sensitivity of the fitted pipeline to these input/representation changes within this cohort; they do not partition physiological versus behavioral signal or establish the physical cause of recording gaps.

---


## Separate review comments — 2026-10-19

These comments are recommendations, not amendments to the protocol above. The review inspected the plan, relevant reports, and source code; no analyses or tests were run.

**Overall assessment:** Proceed after tightening interpretation and correcting the implementation specifications below. Matched participants and histories, training-only preprocessing, reference parity, reusable feature blocks, and the three-allocation pilot are appropriate. Keep the experiment focused on sensitivity of fitted predictors to filling and mask representations; most recommendations do not require a larger benchmark.

### R1. Filling contrasts do not isolate the named mechanisms (§§1, 4.3, 6)

`H_global − H_clock` changes imputed levels, boundary-jump amplitudes, and within-gap slopes simultaneously. Rename “template slope effect” to a clock-versus-global filling contrast.

`H_clean_diff` is the most informative intervention, but removes both boundary jumps and interior template slopes. Its zero pattern still conveys missingness, and the unchanged base branch retains imputation effects. Boundary jumps also encode actual HR deviations from the population median: losing discrimination after removing them would not establish reliance on purely behavioral artifacts. Describe this contrast as withholding gap-related difference values, consistent with the scope limitation in §8.

**Optional, only if boundary-specific attribution is essential:** Add one arm that zeros transitions with $M_t \ne M_{t+1}$ but retains clock-template slopes at $(0,0)$ transitions. Comparing this arm with `H_clock` and `H_clean_diff` separates the two input interventions, although refitted-model AUROC differences still do not form a causal partition.

### R2. Stacking is useful, but does not guarantee test-performance protection (§§1, 4.3, 5.4, 6)

Including $w=0$ guarantees that the selected blend cannot underperform `H_clock` on the validation AUROC criterion, assuming nonzero score SD. It can still underperform on test. Selecting component alphas and then the weight on the same validation set is reasonable for this pilot, but adds selection flexibility.

The Garmin [report](../recording_structure_garmin/results/REPORT.md) explicitly treats redundancy and estimation burden as possible explanations for the concatenation loss. It does not establish shared-penalty feature dilution as its cause. Revise that attribution and the claim that stacking prevents dilution.

Blending separately trained scores is also a restricted alternative to joint feature fitting. A null stacking result means these two score streams did not combine usefully under the tested procedure; it does not establish that mask features contain no complementary information.

### R3. The mask contrast is a representation increment (§§4.1, 4.3, 6)

`M_temporal − M_static` adds nonlinear functions of occupancy and gap structure as well as sensitivity to arrangement. Better AUROC establishes that the richer mask representation helps, but does not isolate temporal order.

For binary masks and $D_u>1$,
$$s_{M,u,t}^{2}=\frac{D_u}{D_u-1}\,\bar M_{u,t}(1-\bar M_{u,t}).$$
Conditional on history length, the SD block contains no information beyond the means. It can still help a linear ridge model by supplying a nonlinear basis, so retaining it for comparability is defensible; it should not be interpreted as an independent measure of variability.

**Optional:** Add a shuffle control only if identifying arrangement is a primary objective. Otherwise, narrower interpretation is sufficient for the current predictive question.

### R4. Interpolation spans differ across resolutions (§§1, 3, 4.4)

With $k$ missing bins, endpoint bin centers are $(k+1)\Delta t$ apart. The proposed rule therefore bridges 10–15-minute endpoint separations at 5m, versus 30 minutes at 15m. A change in the resolution contrast consequently also reflects different interpolation spans.

Either explicitly describe this as a resolution-specific filling policy, or match endpoint spans if the intended comparison requires the same physical intervention. The caches do not retain exact event times within bins and cannot establish that short gaps arise from passive sampling or that long gaps are off-wrist periods. Those are hypotheses about gap origin.

### R5. Specify the primary readout and retain exploratory status (§§6, 8)

Three allocations are reasonable for prioritization. Multiple arms, penalty rules, budgets, and subgroup contrasts nevertheless create many opportunities for a favorable result. Recommendation: nominate `all_days` with `ridge_selected_ext` as the primary readout; retain the original grid, fixed alpha, and capped histories as secondary readouts for parity and sensitivity.

The focal subgroups emerged from inspected test results across all ten allocations. Extending promising arms to allocations 3–9 would assess stability across reused splits, not provide independent confirmation. Treat the roughly $0.01$ AUROC and consistent-direction advancement rule as an operational prioritization heuristic, not an inferential criterion.

### R6. Correct two implementation specifications before coding (§§4.3, 5.3, 7)

1. **Define `H_clean_diff` piecewise.** Cached gaps are `NaN`, so literal multiplication by the observation indicator leaves `NaN × 0 = NaN`. The intended definition is
   $$\Delta^{\text{clean}}X_{u,d,t}=\begin{cases}h_{u,d,t+1}-h_{u,d,t}, & M_{u,d,t}=M_{u,d,t+1}=1,\\0, & \text{otherwise}.\end{cases}$$
   The fixture's “zero if and only if missing” condition is also false: two equal observed values legitimately have zero difference. Require zero whenever either endpoint is missing, and the observed difference otherwise.
2. **Match the reference alpha-selection behavior.** The [reference selector](../apple_cohort_scaling/scripts/04_run_apple_benchmark.py#L268) uses `np.isclose`, rather than exact AUROC equality. Baseline parity requires reproducing that actual selection rule and documenting it accurately.
