# Recorded-salutation prediction from wearable epoch data

Experiment overview, updated 24 September 2026 after the days-scaling extension: 17 experiment stages. The latest review checked saved metrics, reports, provenance, selected logs and implementation; the days-scaling rows cover the 23–24 September runs. The linked meta-report (22 September) predates them and does not yet cover this extension.

The prediction target is recorded salutation, codes 10 versus 20. Results do not establish biological sex, gender identity, or causal physiological effects. Cohorts and evaluation designs differ across experiments, so their absolute AUROCs are not directly interchangeable. The current [meta-report](reports/recorded_salutation_meta_report_2026-09-22.md), also available as a [PDF lab report](reports/recorded_salutation_meta_report_2026-09-22.pdf), covers the evidence, cosinor preprocessing, filling sensitivity and revised research priorities; it supersedes the earlier syntheses.

## Experiment results

| Experiment | Main result |
|---|---|
| [Pooled v1](reports/REPORT.md) | 20,485 participants; default RF test AUROC **0.7132**, balanced accuracy **0.5896**. Strongly asymmetric class recall. |
| [Pooled v2](experiments/artifacts_v2/model_report.md) | Added demographics/source indicators and changed source exclusions; AUROC **0.6833**. Different cohort and split prevent attributing the difference to features alone. |
| [Strict source-specific models](experiments/artifacts_sq/sidequest_report.md) | Garmin **3,848**, Apple **4,446**, Samsung **541** participants. Compared demographics, full-recording, 91-day and rolling representations. Epoch summaries outperform demographics; useful representations differ by source. |
| [RF tuning](experiments/artifacts_sq/tuned_500_comparison.md) | Regularization improved performance, particularly for wide matrices. Established the shared configuration: `max_features=0.4`, `min_samples_leaf=10`, `class_weight="balanced_subsample"`, unrestricted depth. Later repeated-split studies used 100 trees. |
| [Feature selection](experiments/artifacts_sq/fsplit/feat_sel_report.md) | Across 30 repeated splits, hygiene reduced matrix width without a detected performance loss. Correlation pruning and top-k selection offered no consistent improvement; selected subsets were unstable. |
| [Feature engineering](experiments/artifacts_sq/feng/feat_eng_report.md) | Tested interactions, channel ratios and rolling-window differences. Primary gains were small—at most **+0.0034 AUROC**—with intervals crossing zero. |
| [Informative recording / missingness](experiments/2026-09-20_informative_recording_features/README.md) | Ungated cohort **25,784**. Recording model **0.6826**, values with source/gate context **0.7072**, combined **0.7239**. Explicit recording features add **+0.0167** beyond the value model. Native NaNs offer no detectable advantage over median imputation. |
| [Continuous age versus age groups](experiments/age_vs_agegroup_garmin_2026-09-20/results/REPORT.md) | Garmin AUROC **0.7445 → 0.7475**. Paired gain **+0.0029**, interval **[−0.0008, +0.0067]**: promising but not established superiority. |
| [OR-Tools 50/50, flipped](experiments/2026-09-21_ortools_source_50_50_salutation_rf/PLAN.md) | Broader ungated source cohorts; age/BMI/salutation-balanced allocation. AUROCs in the two directions: Garmin **0.7290 / 0.7094**, Apple **0.6875 / 0.6787**, Samsung **0.6289 / 0.6493**. This experiment fixes 100 trees and `max_depth=10`, alongside the shared feature-fraction, leaf-size and class-weight settings. |
| [R1a circadian features](experiments/r1a_circadian_garmin_2026-09-21/results/REPORT.md) | Added 35 hourly-profile/cosinor features. Garmin baseline **0.7434**, combined **0.7463**, coverage-residualized combination **0.7470**. Small positive increments. |
| [Day-scale modelling](experiments/dayscale_garmin_2026-09-21/results/REPORT.md) | Raw-day predictions aggregated per participant rise from **0.6653** with one sampled day to **0.7269** with 32 and **0.7302** with all days. Personal-baseline features help most at small k. |
| [R1a with balanced allocator folds](experiments/r1a_allocfolds_garmin_2026-09-21/results/REPORT.md) | Ten allocations: baseline **0.7415**, combined **0.7509**, residualized combination **0.7475**. Both increments were positive in all ten allocations and their Bonferroni intervals exclude zero. |
| [R1b date-aware circadian](experiments/r1b_dateaware_garmin_2026-09-21/results/REPORT.md) | Per-day curves over each user's first 40 adequate days, on the R1a-AF folds. Day-level AUROC **0.6589 → 0.6782** (**+0.0193**, coverage-residualized **+0.0181**); participant baseline **0.7415**, combined **0.7513** (**+0.0098**). The gain survives the tested linear coverage adjustment; this does not establish coverage independence. |
| [Temporal representations](experiments/temporal_garmin/results/REPORT.md) | Same 3,848 participants and first-40-day window, 10 allocations. Combined MultiRocket/HYDRA/profile model: **0.8489 validation / 0.8565 test**, versus summary ridge **0.7077 / 0.7157** and summary RF **0.6962 / 0.7031**. Within-day value shuffling reduces MultiRocket test AUROC **0.8361 → 0.6643**, implicating temporal arrangement without identifying its mechanism. Corrected v2 is primary. An extended-alpha addendum improves Combined test **0.8625 → 0.8784 on allocation 0 only**; this is not a new ten-allocation benchmark. Extended to larger day budgets in the days-scaling rows below. |
| [Recording structure](experiments/recording_structure_garmin/results/REPORT.md) | Eight arms × three allocations. Mask/count temporal models reach test **0.6166 / 0.6332**, versus static **0.5465 / 0.5498**. Gains over position-shuffled temporal controls are **+0.0307 / +0.0357**. Appending recording blocks to the HR model gives **−0.0170 validation / −0.0107 test**; no benefit in this configuration, not proof that recording information is redundant. |
| [Filling sensitivity](experiments/filling_sensitivity_garmin/results/REPORT.md) | Four fills × three allocations, matched B40 + MultiRocket + ridge: test **0.8464 clock median**, **0.8485 global median**, **0.8470 linear**, **0.8473 PCHIP with linear fallback**. PCHIP − linear: **+0.0002 test / −0.0004 validation**. Interpolation reaches only **3.87% of missing bins**; global median replaces the fill rule at all missing bins. No material aggregate AUROC sensitivity to these tested choices, but no general imputation-robustness conclusion. |
| [Days scaling](experiments/temporal_garmin/results/DAYS_SCALING_REPORT.md) | Same 3,848 participants and folds. Within the first-40-day window, random-k beats chronological first-k chiefly for MultiRocket (**+0.021** test at k=4, [+0.005, +0.037], MR-specific at k=4; the cheap arms also trend positive at larger k but ~3–6× smaller); the k=40 identity draw replays v2 bit-exactly (Δ=0.00e+00); an MR-arm bug that omitted B40Z in the first window runs was corrected and the buggy outputs archived. Sampling k adequate days from each user's **full history** (mean 425, min 80) instead of the first 40 lifts MultiRocket test AUROC **0.8361 → 0.8583** (k=40) → **0.8854** (k=80) → **0.9264** with all days — +0.070 over the best combined 40-day model, single arm, frozen α. The budget curve is concave: the endpoint sits between a 1/√k moment-precision extrapolation (asymptote ≈0.93, a lower bound) and a log-linear extrapolation (≈0.99); no fixed-k point above 80 was measured, so the ceiling is unobserved. Profile24 gains little (+0.06 across the same range) and matches the fsplit RF references within 0.001; summary RF gains +0.04. Endpoint CI is fold-split-only (single kernel draw) and α is untuned there, so 0.9264 is a floor; results are conditional on the ≥80-day cohort gate. |
| [Tier 1c Garmin MultiRocket Relaxation](HEARTBEAT_LAB/README.md) | Expanded cohort to **4,410** single-source Garmin participants (+562 wear-fail delta users). 10-allocation OR-Tools balanced cross-validation: overall MultiRocket test AUROC **0.9094 ± 0.0058** (strict stratum **0.9235 ± 0.0069**, delta wear-fail stratum **0.8156 ± 0.0448**), versus summary baseline **0.7196 ± 0.0179** (+0.1898 paired gain). Demonstrates that relaxing longitudinal continuity gates does not degrade core discrimination. |
| [Apple Watch MultiRocket Pilot](experiments/apple_pilot_multirocket/README.md) | Single-source Apple Watch cohort ($N=1,000$), 5-allocation OR-Tools CV. Despite episodic point-event sampling (~54% bin occupancy), MultiRocket 5-min test AUROC reaches **0.7554 ± 0.0594** (up to **0.8181**), outperforming 15-min downsampling (**0.7431**) and delivering a **+0.1526 (+15.3 pt)** paired gain over static diurnal summaries (**0.6029**). |

The informative-recording and OR-Tools result summaries above were read from each experiment's local `output/report.md`. Those output directories are excluded from Git because they also contain participant-level artifacts; the links point to the published experiment documentation.

## Experiment Worktree & Lineage

The visual worktree below maps the lineage of all completed, active, and planned investigations, illustrating where experiments diverged across hardware sampling regimes and cohort inclusion gates:

```mermaid
flowchart TD
    classDef completed fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef milestone fill:#cce5ff,stroke:#004085,stroke-width:3px,color:#004085;
    classDef divergence fill:#fff3cd,stroke:#856404,stroke-width:2px,stroke-dasharray: 5 5,color:#856404;
    classDef planned fill:#f8f9fa,stroke:#6c757d,stroke-width:2px,stroke-dasharray: 4 4,color:#495057;

    ROOT["<b>Datenspende Cohort Universe</b><br/>out/{user_id}.csv & cohort_manifest.parquet"]

    %% First Divergence: Device Physics
    D1{"<b>DIVERGENCE 1: Hardware Sampling Physics</b><br/>Continuous PPG vs. Episodic Point Events"}:::divergence
    ROOT --> D1

    %% Left Branch: Garmin
    subgraph GARMIN_BRANCH ["Garmin Cohort (Source 3) — Continuous 1-Min Averages"]
        G_RAW["Garmin Single-Source Universe<br/>N = 4,410 eligible (34.6% F / 65.4% M)"]
        D1 -->|Continuous ~95% bin occupancy| G_RAW

        G_RF["<b>Historical RF Baseline (artifacts_sq)</b><br/>Tabular 91-day window summaries<br/>AUROC = 0.6918"]:::completed
        G_RAW --> G_RF

        D2{"<b>DIVERGENCE 2: Longitudinal Gating</b><br/>Strict Continuous Window vs. Relaxed Wear"}:::divergence
        G_RAW --> D2

        G_T0["<b>Tier 0 Strict MultiRocket (HEARTBEAT_LAB)</b><br/>N = 3,848 (>=11 adequate wks / 13 wks)<br/>Clock-fill Ridge (alpha=1000)<br/>AUROC = 0.9100"]:::completed
        D2 -->|Strict Window Pass (87.3%)| G_T0

        G_DELTA["<b>Stage 1: Delta Wear-Fail Zero-Shot</b><br/>N = 562 wear-fail users<br/>Frozen Tier 0 MR on Delta<br/>AUROC = 0.7850<br/>(Short span: 0.8559 | Intermittent: 0.7599)"]:::completed
        D2 -->|Wear-Fail Delta (12.7%)| G_DELTA

        G_T1C["<b>Stage 2: Tier 1c 10-Alloc CV (Milestone)</b><br/>N = 4,410 full cohort (OR-Tools 70/15/15)<br/><b>Overall AUROC = 0.9094 +/- 0.0058</b><br/>↳ Strict Stratum: 0.9235 +/- 0.0069<br/>↳ Delta Stratum: 0.8156 +/- 0.0448<br/>↳ Summary Baseline: 0.7196 +/- 0.0179"]:::milestone
        G_T0 --> G_T1C
        G_DELTA --> G_T1C

        %% Garmin Planned
        P_BIN["<b>PLANNED: Garmin Temporal Granularity Ablation</b><br/>1-min (L=1440) vs 2.5-min (L=576) vs 5-min (L=288)<br/>vs 10-min (L=144) vs 15-min (L=96)"]:::planned
        G_T1C -.-> P_BIN

        P_T2["<b>PLANNED: Tier 2 Primary Garmin Expansion</b><br/>N = 5,636 (includes multi-source switchers)<br/>Evaluate device transition artifacts"]:::planned
        G_T1C -.-> P_T2
    end

    %% Right Branch: Apple Watch
    subgraph APPLE_BRANCH ["Apple Watch Cohort (Source 6) — Episodic Point Events"]
        A_RAW["Apple Single-Source Universe<br/>N = 11,143 eligible (34.6% F / 65.4% M)"]
        D1 -->|Episodic ~54% bin occupancy| A_RAW

        D3{"<b>DIVERGENCE 3: Gate Definition Conflict</b><br/>Interval Union (cov_s) vs. Clock-Hour Count"}:::divergence
        A_RAW --> D3

        A_HIST["<b>Historical Sidelining (artifacts_sq)</b><br/>12h union rule (cov_s >= 43.2ks) failed for dt=0<br/>99.7% dropped on Ch3000 (only 28 passed)<br/>Only Ch3002 hourly resting modeled (RF=0.6814)"]:::completed
        D3 -->|Interval Union cov_s >= 12h| A_HIST

        A_ADAPT["<b>Day Adequacy Rediscovery</b><br/>Replaced interval rule with >=8h & >=60 events<br/>100% of users qualify (mean 265 days/user)"]:::completed
        D3 -->|Clock-Hour & Event Rule| A_ADAPT

        A_PILOT["<b>Apple Pilot Benchmark (Milestone)</b><br/>N = 1,000 users, 5 OR-Tools allocations<br/><b>MR 5-min (L=288): 0.7554 +/- 0.0594</b><br/>MR 15-min (L=96): 0.7431 +/- 0.0437<br/>Summary Baseline: 0.6029 +/- 0.0631<br/>Paired MR Gain: +0.1526"]:::milestone
        A_ADAPT --> A_PILOT

        %% Apple Planned
        P_TRANS["<b>PLANNED: Cross-Device Transfer</b><br/>Evaluate frozen Garmin MR zero-shot on Apple<br/>Quantify vendor domain shift"]:::planned
        A_PILOT -.-> P_TRANS

        P_3002["<b>PLANNED: Apple Channel 3002 Modeling</b><br/>MultiRocket on hourly resting HR (L=24)<br/>N = 11,056 users"]:::planned
        A_PILOT -.-> P_3002

        P_FULL["<b>PLANNED: Full Apple Scaling (N=11,143)</b><br/>Full-scale 10-allocation cross-validation<br/>2.5x larger than Garmin"]:::planned
        A_PILOT -.-> P_FULL
    end

    %% Cross-device transfer connection
    G_T0 -.->|Zero-shot weights transfer| P_TRANS
```

Detailed summaries and documentation:
- [`learnings/experiment_lineage_worktree.md`](learnings/experiment_lineage_worktree.md): complete inventory and divergence analysis.
- [`learnings/future_experiments_backlog.md`](learnings/future_experiments_backlog.md): prioritized follow-up designs (binning ablation, cross-device transfer).
- [`learnings/multirocket_garmin_synthesis.md`](learnings/multirocket_garmin_synthesis.md): audit of the >0.90 AUROC MultiRocket model architecture.

## What the results support

The largest advance is **representing within-day HR sequences rather than only their summaries**. On matched data, the combined temporal model reaches 0.8565 test AUROC versus 0.7157 for summary ridge. Hourly/circadian features provide smaller useful increments. These results establish predictability in the selected cohort, not the physiological origin of the signal.

**Extending the day budget is the largest single lever measured on this cohort.** Sampling days from each user's full adequate history instead of the first 40 lifts single-arm MultiRocket from 0.8361 to 0.9264 test — **+0.070 over the best combined 40-day model** — and the budget curve is concave, bracketed at the endpoint between a 1/√k moment-precision extrapolation (asymptote ≈0.93, a lower bound) and a log-linear one (≈0.99); no fixed-k point above 80 was measured, so the ceiling is unobserved. The gain is overwhelmingly transform-specific: Profile24 gains +0.06 and summary RF +0.04 across the same range; Profile24 (0.7442) matches the fsplit RF references (0.7435 default, 0.7445 feature-selected) within 0.001. Which days are sampled matters chiefly for the transform: random-k beats chronological first-k within the window (**+0.021** test at k=4, MR-specific at k=4), and full-history sampling beats the first-40 window by ≈+0.02 at matched k; the cheap arms trend positive on the same axes but ~3–6× smaller.

**Recording structure is predictive in its own right.** Mask/count sequences achieve approximately 0.62–0.63 test AUROC. Their shuffle controls implicate temporal arrangement, but jointly disrupt local order and clock alignment. The HR-value shuffle preserves the original mask while disrupting value order and value-to-time alignment; it does not eliminate interactions between HR, sampling and recording behavior. None of these AUROC differences measures a fraction of physiological information.

**The tested filling conventions have little effect on aggregate AUROC after refitting.** PCHIP and linear interpolation produce nearly identical test rankings (Spearman 0.9993–0.9995). However, they alter the fill rule at only 197,148 of 5,089,133 missing bins, approximately 0.44% of the entire input grid. Global median is the broader intervention, yet its small test advantage reverses on validation. Retain clock-bin medians as the reference; there is no convincing reason to switch. Stable AUROC does not establish accurate reconstruction of missing HR, robustness to sampling density, or robustness in a less selected population.

Feature hygiene remains useful for reducing matrix width. The tested feature-selection and feature-engineering methods do not support a general performance improvement. Continuous age is a compact alternative to age groups, but its observed gain does not establish superiority. The pooled informative-recording study finds complementarity between recording and value features; its different population and representation prevent transferring that conclusion directly to the strict Garmin temporal model.

The informative-recording experiment supports using observation patterns as predictors. It does not identify an MNAR mechanism or establish performance among participants whose salutation is missing. Its executed protocol is documented in [the informative-recording plan](plans/CODEX_INFORMATIVE_RECORDING_FEATURE_PLAN_2026-09-20.md); [MNAR_PLAN.md](plans/MNAR_PLAN.md) is the earlier proposal and should not be treated as the exact executed design.

## Interpretation qualifications

- **Day counts are not calendar follow-up.** The original day-scale script samples k days with replacement across the record and may use 60 additional personal-baseline days. R1b instead evaluates chronological first-k adequate days (first-day curve-model AUROC approximately 0.669), conditional on eventual cohort eligibility. Forty adequate days need not mean forty calendar days; neither design is external prospective validation.
- **The full-history scaling endpoint is conditional on its design.** "All days" uses each user's own pool (80–979, mean 425), and pool size correlates with engagement. A same-statics control (Profile24 +0.005, RF +0.001 versus MultiRocket +0.041 over the 80→all transition) indicates that the static-feature coverage confound is small but does not eliminate it. The endpoint uses one kernel draw across all ten fold-splits, so its CI (0.0055) is fold-split-only, and α was frozen at 1e3 although the α=3e3 sensitivity arm already underperforms there — the 0.9264 endpoint is a floor, not an optimum. The scan requires ≥80 adequate days, so the curve speaks for the long-history stratum, not the export median (~40 adequate days).
- **Coverage residualization is a predictive adjustment, not proof of deconfounding.** The R1a-AF claim that approximately 37% of the increment is mediated by composition/activity is not identified by these AUROC differences. Likewise, the larger gain under balanced folds does not establish why it increased. The [R1a implementation](experiments/r1a_circadian_garmin_2026-09-21/run_r1a.py) fits a linear adjustment for hourly coverage shares and total recording volume on training participants.
- **The cosinor is a descriptive weighted harmonic regression.** It uses hourly HR means and event-count weights, with 24-, 12- and 8-hour components. No detrending, smoothing or interpolation precedes the fit; missing hours have zero fitting weight. The new meta-report defines the model, its preprocessing and a targeted trend sensitivity. A known R1b residual-RMS feature bug produces NaNs on partial-coverage days; it should be corrected in a future feature revision, not silently reinterpreted as valid fit quality.
- **Filling sensitivity is conditional on the pipeline.** All arms retain B40 recording features and refit their temporal transform and ridge head. Allocation 1 selects different ridge penalties across fills, so ranking differences cannot be assigned to the fill values alone. The pilot establishes neither equivalence nor an upper bound on future imputation effects.
- **Some earlier interpretations are superseded.** Failure of recording-feature concatenation does not prove redundancy. MultiRocket biases are quantiles of convolution responses, not raw binary mask values; binary inputs alone do not establish the claimed transform degeneracy. The filling report's explanation of validation/test sign differences as solely an alpha-selection artifact is also unproven.
- **The comparison designs differ.** V1/v2 and the original sidequest used optimized allocations; feature selection, engineering, age and original R1a used repeated label-stratified random splits. R1a-AF uses demographic balancing. These remain evaluations on reused cohorts, not independent validation. Repeated-split intervals describe split/model randomization conditional on the observed cohort.
- **Use the current synthesis for interpretation.** The [Markdown meta-report](reports/recorded_salutation_meta_report_2026-09-22.md) and its [PDF export](reports/recorded_salutation_meta_report_2026-09-22.pdf) supersede earlier summaries and the older temporal roadmap. Historical experiment outputs remain the execution record; the current synthesis distinguishes their numerical results from unsupported explanations.

## Research direction

The work's strengths are fast controlled ablations, matched comparisons and explicit implementation records. Its main improvement opportunity is to keep explanations narrower than the interventions identify. Prioritize experiments that change a research decision over more precision on tiny deltas from reused participants.

- **Main scientific direction: broader imputation sensitivity, awaiting data access.** Test dependence on gap length, missingness burden and sampling density in a larger, more representative dataset, including the population where salutation is missing. Separate wearable-value reconstruction from the effects of target-label imputation on downstream analyses. The required larger dataset is not currently available to this workspace. This pilot narrows one filling question; it does not close that research direction.

- **Cheapest demonstrated lever: extend the day budget on the existing cohort.** The full-history curve is concave with an unobserved ceiling above k=80 (bracketed ≈0.93–0.99 by per-user extrapolation); per-budget α tuning on the saved endpoint features is fast, and a fixed-k point beyond 80 with an explicit cohort-selection rule (e.g., restricting to users with ≥ k days) can extend the curve further. All of this stays on the gated long-history cohort; the export median (~40 adequate days) is not covered by this evidence, and no physiological origin is claimed for the increment. The MR-arm erratum from the days-scaling runs (B40Z omitted in the first window scripts) is corrected and the buggy outputs are archived; future representation work should treat 19,121-column MR arms as the matched width.

- **Next feasible research direction: multichannel representations.** Compare ch3000 alone with HR plus activity/steps and, where supported by coverage and semantics, sleep/resting-HR channels. Hold participants, days, folds and learners fixed; include channel-availability controls. Treat channel selection and source definitions explicitly, especially for vendor-derived measures. Multichannel work is now in scope; no new experiment is executed by this documentation update. Evaluation should match the days-scaling budget, not only the 40-day window — otherwise the reference temporal representation is understated and representation work is confounded with budget.

- **Engineering candidates for filling:** evaluate a TabPFN-TS-style imputer and, secondarily, a TiRex adaptation against simple fills on masked observed blocks before scaling. Forecasting support is not automatically interior-gap imputation support. Keep these as candidates, with local CPU feasibility unmeasured; see the meta-report's primary sources and evaluation setup.

- **Cheap secondary checks:** Gaussian naive Bayes is reasonable as a small summary-feature baseline, but existing ridge/RF baselines already establish that temporal representation matters. A joint trend-plus-cosinor sensitivity is more interpretable than automatic detrending. Neither check should displace the budget extension, multichannel work, or main data-access imputation study.

- **Keep evaluation matched and exploratory until fresh data arrive.** Use training participants to fit shared transforms, validation to select configurations, and the existing test folds for explicitly exploratory comparisons. Freeze the eventual pipeline before evaluating new participants. There is no evidence-based numerical ceiling or promised AUROC gain.

## Cohort allocation package

[`cohort_allocator`](cohort_allocator/README.md) is the separate reusable OR-Tools package. It accepts a participant-level demographic summary, variables to balance, optional independent partitions, and relative fold sizes, then exports assignments, prepared cohorts and balance diagnostics. R1a-AF is the completed experiment demonstrating its use for the later feature comparisons.
