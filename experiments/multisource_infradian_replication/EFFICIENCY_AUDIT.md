# Multisource Infradian Replication: Computational Efficiency and Scaling Audit

**Date**: 2026-10-22
**Scope**: [`experiments/multisource_infradian_replication/`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/)
**Environment**: Python 3.14 on macOS (`staff`/ARM64), 8–9 worker cores.

---

## 1. Executive Summary

The multisource infradian replication implementation is functionally correct and incorporates coarse-grained parallel execution via [`concurrent.futures.ProcessPoolExecutor`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/endpoints.py#L296) across independent participants and arms. Intermediate stage results are persisted as Parquet and compressed NPZ files, which prevents redundant data reprocessing across pipeline re-runs.

However, the pre-optimization codebase contained critical algorithmic inefficiencies, unvectorized inner loops, and high-overhead memory patterns:
1. **Stratified Participant Bootstrap ([`multisource/bootstrap.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py))**: Invariant restricted cubic spline (RCS) design matrices are repeatedly recalculated in Python for all $N_{\text{boot}} = 2{,}000$ resamples; singular value decompositions (SVDs) are computed on every draw for rank verification; unneeded pointwise standard errors are evaluated and discarded; and millions of small Pandas DataFrame objects are instantiated inside the bootstrap loop, generating a 600 MB Parquet artifact and multi-gigabyte memory consumption during pivot aggregation.
2. **Spectral & REML Estimation ([`multisource/endpoints.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/endpoints.py))**: Frisch-Waugh-Lovell harmonic projection operators and partial $R^2$ spectra are calculated through non-vectorized Python loops over 281 discrete frequencies per participant; trapezoidal quadrature runs as a scalar list comprehension; and residual subspace projection uses an eigensystem decomposition (`eigh`) rather than direct QR decomposition (evaluated and rejected for continuity reasons; see §3.2.D and §6).
3. **Matched Endpoint Selection ([`multisource/matched.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/matched.py))**: Disk reads of raw 5-minute `.npz` cache files and quantile computations run serially on the main thread prior to worker dispatch.

**Empirical Scaling**: On an 8–9 core machine, a bounded run with $B=30$ null replicates and a 2,000-task cap per arm completed in **~13.5 minutes**. The implementations in §6 remove avoidable bootstrap design rebuilds, long-form draw storage, Python frequency loops, and serial cache decompression. The QR/`eigh` replacement was measured and **rejected** because it perturbs the REML optimizer path. On 2026-10-22 the multisource default REML optimizer was changed to analytical-gradient L-BFGS-B with adaptive screening; the historical Nelder–Mead path remains selectable with `MULTISOURCE_REML_OPTIMIZER=nelder`. On 120 real daily tasks the new path is 2.3× faster, is never worse in NLL by more than `4.4e-7`, and keeps every $Q_i$ within `2.5e-6`; 13/120 tasks fall back to the historical solver when the multi-start basins disagree. On the final bounded $B=50$ run, Stage 3 falls from `553.65 s` to `210.33 s` (**2.63×**) and the complete run takes 6m39s, but the global endpoint target is not met: 73/18,696 rows (0.39%) change $Q_i$ by more than `1e-5` (max `1.55e-2`). See §7.4 and §8.7. The full $B=199$, $N_{\text{boot}}=1{,}000$ primary-window run completed on 2026-10-22 in **28m23s** with **65,449** rows and **65,443** converged; six `rss0_degenerate` rows remain flagged for user inspection. The earlier 46,430-task estimate used the bounded channel count and is superseded by 65,449 (43,855 daily + 21,594 channel).

---

## 2. Empirical Benchmark Profile (Mid-Size Validation Run, Pre-Optimization)

The run directory [`artifacts/runs/mid_size_20261021/`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/artifacts/runs/mid_size_20261021/) evaluated 16 arms with $B=30$ null replicates, $N_{\text{boot}}=2{,}000$, and a maximum limit of 2,000 tasks per arm (totaling 18,696 converged participant tasks).

| Stage | Script / Module | Wall Time | Observed Output Size | Computational Characteristics |
|---|---|---:|---:|---|
| **1. Inventory & QC** | [`multisource/provenance.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/provenance.py) | **1m 09s** | 19.2 MB Parquet | Linear scan of 12.8M raw rows; duplicate removal and range gating. |
| **1b. Channel Cache** | [`multisource/channel3000.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/channel3000.py) | **1m 09s** | 1,545 NPZ files | Fast-filtered candidate scan based on minimum 120 dense bins. |
| **2. Manifests** | [`multisource/daily.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/daily.py) | **32s** | 4.8 MB Parquet | Deterministic greedy quota window selection. |
| **3. Endpoints & Nulls** | [`multisource/endpoints.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/endpoints.py) | **8m 09s** | 6.2 MB Parquet | 18,696 tasks; 6 Nelder-Mead starts + AR(1) null simulations per task. |
| **4. Age Models** | [`multisource/models.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/models.py) | **2s** | 120 KB CSV/Parquet | Closed-form OLS-HC3 and Wald chunk tests. |
| **5. Bootstrap** | [`multisource/bootstrap.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py) | **1m 15s** | **600.1 MB** Parquet | 2,000 resamples $\times$ 16 arms; extreme memory and serialization overhead. |
| **6. Matched Contrasts** | [`multisource/matched.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/matched.py) | **2m 25s** | 2.1 MB Parquet | Exact-date matching, re-fitting endpoints, paired resampling. |

---

## 3. Detailed Technical Analysis of Inefficiencies

Line references in this section describe the **pre-optimization** code audited on 2026-10-21. The implemented replacements and their measured effects are in §6.

### 3.1. Stage 5: Participant Bootstrap Bottlenecks

**Location**: [`multisource/bootstrap.py:161-279`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L161-L279)

#### A. Invariant Basis Recalculation inside Bootstrap Loop
* **Mechanism**: In [`bootstrap_frame`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L196) and [`line 253`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L253), each bootstrap draw $b \in \{0, \dots, N_{\text{boot}}-1\}$ extracts a resampled DataFrame `sub = work.iloc[pos]` and calls [`_female_design(sub)`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L59) or [`_sex_design(sub)`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L63). These functions invoke [`common.rcs_stone_koo`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/common.py#L111) to re-evaluate the cubic spline transformations on resampled ages.
* **Mathematical Inefficiency**: The sample participant set $\mathcal{P} = \{(a_i, s_i, y_i)\}_{i=1}^N$ is fixed. A bootstrap resample draw $b$ is an index map $\pi_b: \{1, \dots, N\} \to \{1, \dots, N\}$. Because:
  $$X_b = X[\pi_b, :]$$
  the full model design matrix $X \in \mathbb{R}^{N \times p}$ can be formed once prior to resampling. Evaluating spline functions 2,000 times per metric per arm introduces large CPU interpreter overhead.

#### B. Full SVD for Rank Verification on Every Draw
* **Mechanism**: Lines [`198`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L198) and [`229`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L229) execute `np.linalg.matrix_rank(Xb)`.
* **Mathematical Inefficiency**: `np.linalg.matrix_rank` executes a full LAPACK `_gesdd` singular value decomposition. For 16 arms $\times$ 4 metrics $\times$ 2 model scopes $\times$ 2,000 draws, this executes over **250,000 SVDs**. Because resampling is stratified across 5-year age bands and sex strata, rank deficiency is rare and can be detected during normal OLS computation via positive definiteness of $X_b^T X_b$.

#### C. Pointwise Standard Errors Calculated and Discarded
* **Mechanism**: In each bootstrap draw, [`_predict`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L48-L56) computes:
  $$\text{se}(a) = \sqrt{\text{diag}\left(X_{\text{grid}} \, \text{Cov}_{\text{HC3}}(\hat{\theta}_b) \, X_{\text{grid}}^T\right)}$$
  across 249 evaluation grid points via `np.einsum`.
* **Logical Inefficiency**: [`summarize_bands`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L308) computes simultaneous confidence bands using the empirical sample standard deviation across bootstrap curves:
  $$\hat{\sigma}_{\text{boot}}(a) = \sqrt{\frac{1}{B-1} \sum_{b=1}^B \left(\hat{f}_b(a) - \bar{f}(a)\right)^2}$$
  The individual draw-level standard error array `sb` is never read and is discarded.

#### D. Pandas Micro-DataFrame Allocations and Disk Saturation
* **Mechanism**: Lines [`204-208`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L204-L208) instantiate a 249-row `pd.DataFrame` containing string and float columns inside the inner resample loop.
* **Storage Inefficiency**: A full run produces millions of tiny DataFrames that are joined via [`pd.concat`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L276). In the mid-size run, [`bootstrap_curves.parquet`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/artifacts/runs/mid_size_20261021/bootstrap) reached **600.1 MB** for a restricted subset. Storing predictions as a dense float32 array $\hat{Y} \in \mathbb{R}^{N_{\text{boot}} \times 249}$ requires $2{,}000 \times 249 \times 4 \approx 2.0\text{ MB}$ per arm, which eliminates the multi-gigabyte [`pivot_table`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L292) operation.

---

### 3.2. Stage 3: Spectral and REML Engine Bottlenecks

**Location**: [`multisource/endpoints.py:122-211`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/endpoints.py#L122-L211) and [`02_compute_endpoints_and_nulls.py:213-348`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/infradian_age_dependency/scripts/02_compute_endpoints_and_nulls.py#L213-L348)

#### A. Scalar Discrete Frequency Projections
In [`precompute_fwl_harmonic_operators`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/infradian_age_dependency/scripts/02_compute_endpoints_and_nulls.py#L320-L348), the harmonic basis $C(f) = [\cos(2\pi f t), \sin(2\pi f t)]$ is projected onto the residual subspace:
$$\tilde{C}_i = P_0 C(f_i), \quad M_i = \tilde{C}_i \left(\tilde{C}_i^T \tilde{C}_i\right)^{-1}$$
via a Python loop over 281 broad frequencies and 21 narrow frequencies (302 iterations). Constructing the concatenated basis matrix $C \in \mathbb{R}^{N \times 604}$ once allows computing $\tilde{C} = P_0 C$ in a single BLAS-3 `dgemm` call.

#### B. Non-Vectorized Partial $R^2$ Evaluation
In [`process_daily_task`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/endpoints.py#L166-L175):
```python
def spectra(e: np.ndarray, M: np.ndarray, CTC: np.ndarray, denom) -> np.ndarray:
    beta = e @ M
    out_p = np.empty((beta.shape[0], len(CTC)), dtype=np.float64)
    for i in range(len(CTC)):
        bc = beta[:, 2 * i]
        bs = beta[:, 2 * i + 1]
        c = CTC[i]
        drss = bc ** 2 * c[0, 0] + 2.0 * bc * bs * c[0, 1] + bs ** 2 * c[1, 1]
        out_p[:, i] = drss
    return np.clip(out_p / denom, 0.0, 1.0)
```
This loop runs over 281 frequencies in Python for both observed and simulated series ($4 \times 281 = 1{,}124$ Python loop iterations per participant task). Vectorizing this calculation across frequencies:
```python
bc = beta[:, 0::2]
bs = beta[:, 1::2]
drss = (bc**2) * CTC[:, 0, 0] + 2.0 * (bc * bs) * CTC[:, 0, 1] + (bs**2) * CTC[:, 1, 1]
```
executes across all frequencies simultaneously in vectorized C/NumPy code.

#### C. Quadrature Loop
Lines [`186-189`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/endpoints.py#L186-L189) integrate null replicates using a Python list comprehension:
```python
Q_null = np.array([
    np.trapezoid(P_broad_null[b, IN_BAND_MASK], F_BAND) / DELTA_F
    for b in range(b_rep)
])
```
Passing `axis=-1` to `np.trapezoid` computes all $B$ integrals in one operation:
```python
Q_null = np.trapezoid(P_broad_null[:, IN_BAND_MASK], F_BAND, axis=-1) / DELTA_F
```

#### D. Nuisance Subspace and Eigensystem Decomposition
[`build_daily_nuisance`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/endpoints.py#L32-L58) performs an iterative Gram-Schmidt loop in Python over columns of $X$. In addition, [`process_daily_task:136`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/endpoints.py#L136) diagonalizes $P_0$ via `np.linalg.eigh(P0)` to find the orthogonal complement $Q_{\text{sub}}$.

**Evaluation outcome**: a pivoted-QR basis and `null_space(Q.T)` were implemented and benchmarked (Gram–Schmidt 0.058 → 0.048 ms; `eigh` 0.238 → 0.049 ms). Both rotate the nuisance/residual basis by $O(1)$, which changes the floating-point Nelder–Mead path. On 120 real tasks, $Q_i$ and $E_{i2}$ shifted by up to $1.1\times10^{-8}$ and the discrete `phi2` label flipped in 16–26 tasks when $v_2\approx0$. The original Gram–Schmidt + `eigh` path reproduces the historical endpoints to machine precision ($Q_i$ max $9.7\times10^{-17}$, `phi2` unchanged). Because the time saving is ~0.14% of a REML-dominated task, the QR/`eigh` replacement was **reverted** to satisfy the plan's numerical-continuity requirement.

---

### 3.3. Stage 6: Matched Cohort Single-Threaded I/O

**Location**: [`multisource/matched.py:40-107`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/matched.py#L40-L107)

In [`exact_date_cohort`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/matched.py#L71-L84), candidate participant cache files (`.npz`) are opened and read sequentially on the main thread:
```python
for (uid, widx, digest), group in sel.groupby(...):
    cache_path = channel3000.resolve_cache_path(uid, int(source), Path(cache_dir))
    hrp10 = _channel_values(str(cache_path), dates) if cache_path else {}
```
For large arms (such as Garmin, with >1,300 candidate participants), the main process executes over 100,000 percentile evaluations and file decompressions on a single CPU core before worker subprocesses are initialized.

---

## 4. Concrete Optimization Action Plan

The table below outlines the necessary fixes, exact file references, and estimated speedups.

| Priority | Target Module | Optimization | Measured / expected effect |
|---|---|---|---|
| **P0** | [`multisource/bootstrap.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py) | **Precompute design matrix $X$**: slice rows $X[\pi_b, :]$ instead of calling `rcs_stone_koo`. | Implemented. Removes 2,000 spline builds per arm (0.23 s for $n=2{,}000$ versus a 2.08 s $B=2{,}000$ fit); OLS/Wald now dominates. |
| **P0** | [`multisource/bootstrap.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py) | **Vectorize bootstrap curve storage**: store $\mathbb{R}^{N_{\text{boot}} \times 249}$ NumPy array; eliminate micro-DataFrames and `pivot_table`. | Implemented. Per arm (3 scopes, $B=2{,}000$, 249 ages): 40.4 MB long DataFrame → 6.0 MB float32 arrays → 4.0 MB compressed `.npz` (85% in-memory reduction). |
| **P1** | [`multisource/endpoints.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/endpoints.py) | **Vectorize `spectra`, `np.trapezoid`, and FWL operators**: eliminate Python frequency loops in worker processes. | Implemented. Observed spectra 0.487 → 0.008 ms (59×); quadrature 0.096 → 0.004 ms (22×); broad FWL operator build 2.14 → 0.27 ms (7.8×). End-to-end task remains REML-bound. |
| **P1** | [`multisource/endpoints.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/endpoints.py) | **Replace Gram-Schmidt and `eigh` with pivoted QR** (proposed). | **Evaluated and rejected.** Setup gain is 0.01–0.19 ms/task (~0.14% of a REML-dominated task), but the basis rotation shifts $Q_i$/$E_{i2}$ by ~$10^{-8}$ and flips the discrete `phi2` label when $v_2\approx0$. Original Gram–Schmidt + `eigh` reproduced 120 historical tasks to machine precision and was retained. |
| **P2** | [`multisource/matched.py`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/matched.py) | **Parallelize cache decompression**: move `_channel_values` extraction into worker pool tasks. | Implemented. Cache reads distributed over the pool; `workers=1` and `workers=4` produce identical tasks, retention, and matched values. |

The implemented rewrites are algebraically equivalent, not approximations. Observed Wald statistics in stage 5 reproduce stage 4 exactly; point predictions and pointwise HC3 SEs agree to float32 storage precision (max $2.4\times10^{-7}$ and $2.9\times10^{-8}$ across 64 arm–metric jobs).

---

## 5. Architectural Correctness and Safeguards

The pipeline implements the core methodological safeguards specified in [`PLAN.md`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/PLAN.md):
- Estimands between dense observed-bin HR (`HRp10`), daily RHR (type 65), and daily sleep HR (type 66) remain strictly segregated and are not pooled.
- Exact duplicate rows are collapsed, and conflicting daily records are excluded without averaging.
- Stratified participant resampling preserves empirical age-band $\times$ sex proportions across all resample draws.
- Paired comparisons in Stage 6 enforce common converged sets, joint age support recalculation, and identical bootstrap draw indices across compared arms.

The retained optimizations preserve numerical fidelity and protocol compliance. They reduce bootstrap, storage, and matched-stage I/O overhead. They do not change the REML-bound stage-3 cost; the measured full-run result is in §7.5.

---

## 6. Implementation Status and Verification (2026-10-21)

Four of the five §4 items are implemented in `multisource/`; the QR/`eigh` item was evaluated and reverted (§6.1). This section records the actual code paths, artifact schema, and measured checks; it supersedes the pre-implementation estimates above.

### 6.1 Code changes

| Item | Implementation |
|---|---|
| **P0 design reuse** | [`bootstrap.bootstrap_frame`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py) builds `X_female` (`n × 4`) and `X_sex` (`n × 8`) once from the fixed participant set, plus the evaluation-grid designs. Each draw is a row slice `X[draws[b], :]`; no per-draw `build_design` or `rcs_stone_koo` call remains. |
| **P0 array storage** | Bootstrap predictions are accumulated as float32 `(N_boot, 249)` arrays per `(source, endpoint, protocol, window_index, metric, scope)`. Stage 5 writes `bootstrap_curves.npz` (`age_grid` plus one array per key), `bootstrap_curves_manifest.csv`, `curves_point.parquet`, `confidence_bands.parquet`, `bootstrap_chunk_summary.csv`, and `transition_metrics.csv`. The long-form curve table is built only when a caller requests it (`return_curves=True`), with categorical metadata and float32 payload; stage 5 never materializes per-draw rows. |
| **P1 no per-draw pointwise SEs** | Draw-level HC3 *pointwise* standard errors are no longer computed. The observed-fit pointwise HC3 SE is retained for `confidence_bands.hc3_se`. Draw-level HC3 *covariances* are still computed because the four bootstrap chunk-Wald distributions require them; the audit's original wording conflated the two. |
| **P1 rank gate** | `np.linalg.matrix_rank` (full `_gesdd`) is replaced by a Cholesky positive-definiteness test on `X_b^T X_b`. A failed draw is skipped and leaves NaNs in that array row. |
| **P1 vectorized endpoint algebra** | `precompute_fwl_harmonic_operators` is now a local vectorized implementation (one `P0 @ C` GEMM, batched 2×2 Gram inverses). `spectra` is a single array expression over frequencies. `Q_null` uses `np.trapezoid(..., axis=-1)`. |
| **P1 QR / null space (rejected)** | Pivoted QR and `null_space(Q.T)` were implemented and measured, then reverted. They shift $Q_i$/$E_{i2}$ by ~$10^{-8}$ and flip the discrete `phi2` label when $v_2\approx0$. The original Gram–Schmidt + `eigh` basis reproduces 120 historical tasks to machine precision and is retained. |
| **P2 parallel cache reads** | `exact_date_cohort` separates planning, parallel `np.load` decompression (`_load_channel_values` over `ProcessPoolExecutor`), and assembly. `run_matched_source` forwards `workers`. |

### 6.2 Verification performed

- **Test suite**: all 25 tests in `tests/test_pipeline.py` pass (run directly; `pytest` is not installed in the environment).
- **Stage-5 smoke run**: `scripts/05_bootstrap.py --n-boot 8 --workers 2` on `mid_size_20261021/endpoints/endpoints.parquet`, written to the new namespace `artifacts/runs/efficiency_smoke_20261021/`. Result: 64 arm–metric jobs, 128 curve arrays, `status=ok`.
- **Numerical fidelity**: across all 64 jobs, stage-5 point predictions and pointwise HC3 SEs match `models.fit_female` / `models.fit_sex_interaction` to max $2.4\times10^{-7}$ and $2.9\times10^{-8}$ (float32 storage). Observed chunk-Wald statistics match `models/chunk_tests.csv` exactly (e.g. `182.144715`, `34.914579`, `168.950151`, `19.721253`).
- **Algebraic equivalence**: vectorized FWL operators agree with the upstream engine to $2.2\times10^{-16}$ (`M`) and $4.3\times10^{-14}$ (`CTC`); vectorized spectra and quadrature agree to machine precision.
- **Real-task reproducibility**: on 120 historical daily tasks (source 2, RHR, balanced90, $B=30$), the retained endpoint path reproduces `Q_i`, `E_i2`, `A_hat`, `sigma_resid`, `p_value_discrete`, `peak_period`, and the fitted `phi2` label to machine precision: max $|ΔQ_i| = 9.7\times10^{-17}$, max $|ΔE_{i2}| = 4.4\times10^{-16}$, and zero changes in `phi2` or `p_value_discrete`.
- **P2 determinism**: synthetic 5-participant, 90-date cohort produced byte-identical tasks, retention, and matched values with `workers=1` and `workers=4`.
- **Full `B=199` run**: completed 2026-10-22; 65,449 rows, 65,443 converged, 1703.04 s wall (28m23s). Six rows are `rss0_degenerate`; the user will inspect the source data. See §7.5.

### 6.3 Measured costs and remaining bottleneck

| Measurement | Value |
|---|---:|
| `bootstrap_frame`, $n=2{,}000$, $B=2{,}000$, 3 scopes | 2.08 s |
| Per-draw spline rebuild avoided (2 designs × 2,000 draws) | 0.23 s |
| Long curves DataFrame vs float32 arrays vs compressed `.npz` (3 scopes) | 40.4 MB → 6.0 MB → 4.0 MB |
| `spectra`, observed row (231 frequencies) | 0.487 ms → 0.008 ms |
| Quadrature, $B=30$ replicates | 0.096 ms → 0.004 ms |
| Broad FWL operator build | 2.14 ms → 0.27 ms |
| `eigh(P0)` / `null_space` alternative (measured, rejected) | 0.238 ms → 0.049 ms; reverted |
| `process_daily_task`, $B=30$, median | ~148 ms |
| Share of task time in `fit_two_component_reml` (cProfile, 10 tasks) | 99.4% |

The last two rows delimit the speedup available from the original non-REML optimizations. Stage-3 wall time was dominated by the historical REML fit, so the 2026-10-22 L-BFGS-B replacement is the decisive change: 2.3× on the 120-task benchmark, with the multi-basin continuity fallback described in §7. The earlier changes still reduce the non-REML kernel cost by 5–60×, the bootstrap fit to OLS/Wald cost, the curve footprint by 85% in memory, and the matched-stage serial I/O.

---

## 7. REML Optimizer, Threading, and Persistent Queue (2026-10-22)

This section supersedes the statement in §6.3 that changing the REML optimizer requires a separate decision. The decision was made, implemented, and measured.

### 7.1 Implementation

| Component | Code path | Behavior |
|---|---|---|
| Analytical-gradient fitter | `multisource/reml.py` | L-BFGS-B on `theta=(log s1, logit(phi1/0.95), log s2)`, exact gradient, bounds, six canonical starts. |
| Fitter injection | `endpoints.py`, `channel3000.py`, upstream `02_compute_endpoints_and_nulls.py` | Tasks accept `task["reml_fitter"]`; upstream experiments keep the historical fitter unless injected. |
| Default selection | `reml.active_fitter()` | L-BFGS-B by default in Stage 3; `MULTISOURCE_REML_OPTIMIZER=nelder` restores Nelder–Mead. `MULTISOURCE_REML_CONTINUITY=0` disables the multi-basin historical fallback. |
| Adaptive screening | `reml.fit_two_component_reml_lbfgsb` | Central `alpha=0.5` starts for both `phi2`; standard four-start fallback on boundary/gradient failure; two low-`phi1` anchors; a boundary-rescue start; and a historical Nelder–Mead continuity fallback when the L-BFGS-B starts expose different basins or a high-`phi1`/large-`v2` boundary fit is selected. `MULTISOURCE_REML_SCREENING=0` disables screening. |
| Single-thread BLAS | `multisource/runtime.py` | Imported before NumPy/Pandas in `run_all.py` and all seven stage entrypoints. macOS/Accelerate uses `VECLIB_MAXIMUM_THREADS=1`; OpenBLAS/MKL variables are non-Darwin only. |
| Persistent Stage 3 queue | `multisource/stage3.py` | One `ProcessPoolExecutor` over all daily/channel tasks; ordered result consumption and per-arm writes preserve historical row order. |
| Cached `HRp10` | `channel3000.build_caches`, `matched._channel_values` | Float64 `hrp10` stored per participant; historical caches without the key use the finite-bin percentile fallback. |

### 7.2 Gradient and optimizer checks

- Central finite differences on a synthetic 90-day task: max absolute error `9.2e-8`, relative error `5e-10`.
- 120 source-2 RHR `balanced90` tasks at \(B=30\), same seeds:
  - wall time 24.6 s (Nelder–Mead) vs 10.7 s (L-BFGS-B), **2.3×** in a paired run under concurrent load;
  - no task has NLL worse by more than `4.4e-7`; no task improves by more than `1e-6`;
  - every task has `|ΔQ_i| < 1e-5` (max `2.50e-6`); max `|ΔE_i2| = 1.06e-5`;
  - 29 `phi2` labels change, all at \(v_2\) near the numerical floor; p-values are unchanged.
- The low-`phi1` anchors are required: without them, six canonical starts are worse by up to `1.05` NLL on historical tasks whose Nelder–Mead solution lies in the near-zero-`phi1`/high-`v_2` basin. When the low anchors improve on the canonical fit, or when a high-`phi1`/large-`v_2` boundary fit is selected, the fitter runs the historical Nelder–Mead solver and returns its result. This continuity fallback occurred on 13/120 benchmark tasks.
- Adaptive screening used 4 starts for 23/120 tasks, 6 for 13/120, 8 for 66/120, and 10 for 18/120. The plan's projected 2.8× reduction in starts was not observed because many valid fits sit on the variance boundary; the net speedup comes from fewer L-BFGS-B iterations.

### 7.3 Cache correctness finding

The first implementation of cached `HRp10` used `np.percentile(hr[dense], 10, axis=1)`. Because `hr` contains NaNs for unobserved five-minute bins, this returned NaN for any day with fewer than 288 observed bins. It was found before the shadow run completed, fixed to `np.nanpercentile`, and covered by `test_channel_cache_hrp10_ignores_missing_bins`. On 80 source-3 caches (33,158 finite daily values), the cached `hrp10` equals the historical row-wise finite-bin percentile exactly (maximum absolute difference `0.0`).

### 7.4 Full-run comparison

The clean \(B=50\), `limit=2000` shadow run in `artifacts/runs/b50_shadow_lowphi_20261022` completed in **6m47s**, compared with a Stage-3 time of `553.65 s` in `artifacts/runs/b50_limit2000_20261021` versus `212.39 s` for the shadow run (**2.61×**). All 18,696 endpoints converged in both runs; keys, columns, per-arm order, and global row order are identical.

The 120-task benchmark target does not generalize to the full bounded run:

| Metric | Full \(B=50\) result |
|---|---:|
| `Q_i` difference < `1e-5` | 18,623/18,696 (99.61%) |
| Max `Q_i` difference | `1.55e-2` |
| `Q_i` changes above `1e-5` | 73 (0.39%) |
| Max `E_i2` difference | `3.28e-2` |
| `p_value_discrete` changes | 43 |
| `phi2` flips | 2,917 (15.6%) |
| NLL: old better by >`1e-6` | 16 (max improvement `3.94e-5`) |
| NLL: new better by >`1e-6` | 72 (max improvement `2.13`) |
| Continuity fallback | 407 (2.2%) |
| Model coefficients / chunk Wald | max `3.99e-3` / `0.633` |
| Bootstrap point / SE | max `1.08e-3` / `1.23e-4` |
| Matched daily values | exact |
| Matched endpoint `Q_i` / paired difference | max `7.47e-3` / `1.67e-4` |

The changes are concentrated in non-source-2 arms and are mostly lower-NLL low-`phi1` basins found by L-BFGS-B. The continuity guard covers detectable alternative basins but cannot detect a better basin found by the canonical L-BFGS-B starts themselves; strict historical parity requires the historical fitter. The default remains the faster L-BFGS-B path, with the limitation recorded in `OPTIMIZATION_PLAN.md` §7.2. The full \(B=199\) run has now been executed; see §7.5.

### 7.5 Full `B=199` primary-window run

The unbounded primary-window run `full_b199_boot1000_20261022` used
`--sources 2 3 6 7 9 13 19`, both protocols, nine workers, `B=199`,
`n_boot=1000`, and no `--limit`. It completed in **1703.04 s (28m23s)**.

| Stage | Result |
|---|---:|
| Stage 1b cache construction | about 10m48s; 22,345 caches |
| Stage 3 endpoints | 65,449 rows (43,855 daily + 21,594 channel); 65,443 converged; about 11m32s |
| Stage 4 models | 104 models |
| Stage 5 bootstrap | 96 jobs; `status=ok`; about 18s |
| Stage 6 matched | 16 comparisons; about 4m40s |
| Run directory | 727 MB |

The 18,696 rows shared with the bounded `B=199` run reproduce exactly for
`Q_i`, `Q_i_mc_se`, `E_i2`, `A_hat`, `sigma_resid`, and `p_value_discrete`.
On that bounded cohort, `B=199` versus `B=50` has median
\(|\Delta Q_i| = 1.88\times10^{-3}\), median `Q_i_mc_se` ratio
`B=50/B=199 = 1.98`, and 150 versus 151 significant group chunk tests (3
disagreements). Endpoint-level p-values are less stable: 1,445 versus 1,268 at
\(p<0.05\), with 545 disagreements. The full cohort's larger model set is not
directly comparable to the bounded cohort because it adds participants.

Six full-run rows are `rss0_degenerate` (`null_status=failed`): source 2 user
`1124280` and source 6 user `1222449` across the affected endpoint/protocol
arms. They are not counted as converged. The user will inspect the source
data; no remediation or rerun is planned. The full run used
\(N_{\text{boot}}=1{,}000\), below the prespecified 2,000; bootstrap precision
is correspondingly lower. Exact-date matched comparisons are non-empty only
for sources 3 and 6.

---

## 8. Second-Wave Optimization Opportunities and Findings (2026-10-22)

Code inspection following the clean 6m47s shadow run identifies six concrete leverage points to accelerate the pipeline further without sacrificing numerical rigor:

### 8.1. Stage 7 Matched Execution Fallback to Nelder–Mead
* **Location**: [`multisource/matched.py:160-207`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/matched.py#L160-L207) and [`multisource/endpoints.py:376-394`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/endpoints.py#L376-L394)
* **Finding**: `stage3.py` explicitly injects `task["reml_fitter"] = reml.active_fitter()`, but `endpoints.run_daily_arm` does not set it when invoked directly. In `matched._run_arm_on_matched`, tasks are constructed and passed to `endpoints.run_daily_arm` without `reml_fitter`. As a result, the ~5,200 matched REML tasks silently fall back to historical Nelder–Mead rather than running under L-BFGS-B.
* **Remedy**: Default `reml_fitter` in `endpoints.process_daily_task` or inject `reml.active_fitter()` in `_run_arm_on_matched`. This yields an immediate ~2.5× speedup on Stage 7 wall time (~75 seconds saved).

### 8.2. LAPACK SVD `_gesdd` (`pinv`) in Bootstrap Inner Loop
* **Location**: [`multisource/models.py:56-82`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/models.py#L56-L82)
* **Finding**: In `ols_hc3`, the design covariance is inverted using `np.linalg.pinv(X.T @ X)`. In `chunk_wald`, test contrasts are evaluated using `np.linalg.pinv(C @ cov @ C.T)`. Across 16 arms $\times$ 4 metrics $\times$ 2,000 bootstrap draws, this invokes full LAPACK `_gesdd` SVDs over 250,000 times. Because `_full_rank(X)` has already verified positive-definiteness via Cholesky, these systems are strictly non-singular.
* **Remedy**: Replace `pinv` with Cholesky/triangular solve or `scipy.linalg.inv` in `ols_hc3` and small analytic/LU inversion in `chunk_wald`.

### 8.3. Static Spline Knot Rebuilding in `transition_metrics`
* **Location**: [`multisource/bootstrap.py:186-218`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/bootstrap.py#L186-L218)
* **Finding**: In `transition_metrics(theta_b, bands)`, `_d_rcs(grid)` (601 evaluation points) and `_delta_design(zgrid)` (141 evaluation points) are recomputed from scratch on every bootstrap draw ($128{,}000$ times per full run). The coordinate grids and spline knots are constant.
* **Remedy**: Precompute `D_RCS_GRID` ($601 \times 4$) and `DELTA_DESIGN_ZGRID` ($141 \times 8$) at module load time.

### 8.4. L-BFGS-B Inversion and Triangular Solve Redundancies
* **Location**: [`multisource/reml.py:80-101`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/reml.py#L80-L101)
* **Finding**:
  1. `w = scipy.linalg.cho_solve((L, True), z)` re-solves $L y = z$ and $L^T w = y$, ignoring the fact that $a = L^{-1} z$ was already computed on line 75 (`scipy.linalg.solve_triangular(L, z, lower=True)`). Calling `solve_triangular(L.T, a, lower=False)` cuts triangular solve work in half.
  2. `S_inv = _symmetrize(scipy.linalg.cho_solve((L, True), np.eye(r)))` allocates an $r \times r$ identity matrix and computes the full dense inverse on every gradient step. Replacing this with LAPACK `dpotri` (in-place triangular inversion) or solving $L V = S_2$ avoids matrix allocations and reduces flop counts.

### 8.5. Redundant Low-$\phi_1$ Starts on Low Autocorrelation Profiles
* **Location**: [`multisource/reml.py:257-263`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/reml.py#L257-L263)
* **Finding**: `LOW_PHI1_STARTS` ($\phi_1 = 0.01$) are evaluated unconditionally on every task when adaptive screening is active. For tasks where `phi_init` is already $< 0.05$, this start duplicates the central start trajectory, adding 2 redundant L-BFGS-B runs.
* **Remedy**: Condition the low-$\phi_1$ start on `phi_init > 0.10`.
* **Status**: Tested and rejected; see §8.7.

### 8.6. Overtight L-BFGS-B Stopping Tolerances
* **Location**: [`multisource/reml.py:126-133`](file:///Users/bulat/Documents/datenspende_IMPUTE_sex/experiments/multisource_infradian_replication/multisource/reml.py#L126-L133)
* **Finding**: `ftol = 1e-10` forces extra line-search iterations at the bottom of the basin where objective changes are near double-precision machine noise. Setting `ftol = 1e-8` and `gtol = 1e-5` reduces function evaluations by ~20% with zero change to endpoint precision.
* **Status**: Tested and rejected; the zero-change claim did not hold. See §8.7.


### 8.7. Implementation and Validation Status (2026-10-22)

The second-wave findings were checked against the implementation and the 120-task historical benchmark. The pipeline test suite now has 25 tests, including new tests for Stage 7 fitter injection, Cholesky/pseudo-inverse equivalence, and precomputed transition bases.

| Finding | Status | Evidence |
|---|---|---|
| 8.1 Stage 7 fitter injection | Implemented | `matched._run_arm_on_matched` sets `task["reml_fitter"] = reml.active_fitter()` before calling `endpoints.run_daily_arm`; `test_matched_injects_active_fitter` passes. The measured Stage-7 wall time is `61.02 s` versus `61.50 s` in the first-wave run, so the projected ~75 s saving was not reproduced on this bounded matched subset. |
| 8.2 Cholesky instead of SVD `pinv` | Implemented | `models.ols_hc3` uses Cholesky plus `cho_solve`; `models.chunk_wald` uses a positive-definite solve. The pseudo-inverse remains only as a defensive fallback; `test_cholesky_solvers_match_pinv_reference` agrees with the reference to `1e-11`. The combined 8.2/8.3 bootstrap path runs in `19.40 s` versus `20.77 s` in the first-wave run. |
| 8.3 Precomputed transition bases | Implemented | `bootstrap.D_RCS_GRID` and `bootstrap.DELTA_DESIGN_ZGRID` are built once at import; `test_transition_bases_are_precomputed_exactly` confirms bitwise equality with recomputation. First-wave-to-final bootstrap predictions differ by at most `2.98e-8`. |
| 8.4 Triangular/inversion cleanup | Implemented | The gradient reuses `a=L^{-1}z`, solves `w=L^{-T}a`, and uses LAPACK `dpotri` for `S^{-1}`. On a synthetic rank-8 problem the new and old objective/gradient agree to `0.0` and `6.9e-18`; the finite-difference test still passes. |
| 8.5 Conditional low-`phi_1` starts | Tested and rejected | The rejected variant (`artifacts/runs/b50_shadow_secondwave_20261022`) skipped the anchors on 3,005/18,696 tasks. Relative to the first-wave run with unconditional anchors, 481 of the 499 NLL regressions above `1e-6` occurred on skipped tasks (max `+0.0618` NLL). The low anchor is not redundant when `phi_init` is near zero: it can find a better high-`v_2` basin that the central trajectory misses. The anchors remain unconditional. |
| 8.6 Looser stopping tolerances | Tested and rejected | The recommended `ftol=1e-8`, `gtol=1e-5` failed the 120-task target: 8/120 `Q_i` differences exceeded `1e-5` (max `2.39e-5`), and 3 tasks were worse in NLL by more than `1e-6` (max `6.14e-6`). A conservative `ftol=1e-9`, `gtol=2e-6` passed the 120-task screen but added 18 NLL regressions (max `4.46e-6`) and extra endpoint changes in the full run without a material wall-time gain. The historical `ftol=1e-10`, `gtol=1e-6` is retained. |

The final default keeps 8.1–8.4 and the historical stopping tolerances. The low-`phi_1` anchors are unconditional, as in the first-wave implementation.

Final bounded run: `artifacts/runs/b50_shadow_secondwave_final_20261022` completed in **6m39s**; Stage 3 took `210.33 s` versus `553.65 s` for the baseline (**2.63×**) and `212.39 s` for the first-wave run. All 18,696 endpoints converged; per-arm and global row order match; sampled caches reproduce `hrp10` exactly (33,158 values, max `0.0`). Final versus first wave: `Q_i` max `1.89e-6` (0/18,696 above `1e-5`), `p_value_discrete` changes `0`, NLL difference `-3.86e-8` to `+1.68e-7`, model coefficients max `7.94e-8`, chunk Wald max `8.58e-5`, bootstrap prediction max `2.98e-8`, matched endpoint `Q_i` max `1.39e-7`, paired difference max `2.10e-9`. Final versus baseline retains the first-wave endpoint changes: 73/18,696 `Q_i` above `1e-5` (max `1.55e-2`), 43 p-value changes, 16/72 NLL regressions/improvements, and 407 continuity fallbacks. The full validation is detailed in `OPTIMIZATION_PLAN.md` §7.3.
