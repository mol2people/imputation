# Experiment Lineage & Worktree Diagram

This visual worktree maps all historical, recently completed, and planned experiments in the repository, highlighting key **methodological divergence points**, **sampling regimes**, and **model performance benchmarks**.

---

## 1. High-Level Experiment DAG

```mermaid
flowchart TD
    %% Styling classes
    classDef completed fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef milestone fill:#cce5ff,stroke:#004085,stroke-width:3px,color:#004085;
    classDef divergence fill:#fff3cd,stroke:#856404,stroke-width:2px,stroke-dasharray: 5 5,color:#856404;
    classDef planned fill:#f8f9fa,stroke:#6c757d,stroke-width:2px,stroke-dasharray: 4 4,color:#495057;

    ROOT["<b>Datenspende Cohort Universe</b><br/>out/{user_id}.csv & cohort_manifest.parquet"]

    %% First Divergence: Device Physics
    D1{"<b>DIVERGENCE 1: Hardware Sampling Physics</b><br/>Continuous PPG vs. Episodic Point Events"}
    ROOT --> D1

    %% Left Branch: Garmin
    subgraph GARMIN_BRANCH ["Garmin Cohort (Source 3) — Continuous 1-Min Averages"]
        G_RAW["Garmin Single-Source Universe<br/>N = 4,410 eligible (34.6% F / 65.4% M)"]
        D1 -->|Continuous ~95% bin occupancy| G_RAW

        G_RF["<b>Historical RF Baseline (artifacts_sq)</b><br/>Tabular 91-day window summaries<br/>AUROC = 0.6918"]:::completed
        G_RAW --> G_RF

        G_T0["<b>Tier 0 Strict MultiRocket (HEARTBEAT_LAB)</b><br/>N = 3,848 (>=11 adequate wks / 13 wks)<br/>Clock-fill Ridge (alpha=1000)<br/>AUROC = 0.9100"]:::completed
        G_RAW --> G_T0

        D2{"<b>DIVERGENCE 2: Longitudinal Gating</b><br/>Strict Continuous Window vs. Relaxed Wear"}
        G_RAW --> D2
        D2 -->|Strict Window Pass (87.3%)| G_T0

        G_DELTA["<b>Stage 1: Delta Wear-Fail Zero-Shot</b><br/>N = 562 wear-fail users<br/>Frozen Tier 0 MR on Delta<br/>AUROC = 0.7850<br/>(Short span: 0.8559 | Intermittent: 0.7599)"]:::completed
        D2 -->|Wear-Fail Delta (12.7%)| G_DELTA

        G_T1C["<b>Stage 2: Tier 1c 10-Alloc CV (Milestone)</b><br/>N = 4,410 full cohort (OR-Tools 70/15/15)<br/><b>Overall AUROC = 0.9094 +/- 0.0058</b><br/>↳ Strict Stratum: 0.9235 +/- 0.0069<br/>↳ Delta Stratum: 0.8156 +/- 0.0448<br/>↳ Summary Baseline: 0.7196 +/- 0.0179"]:::milestone
        G_T0 --> G_T1C
        G_DELTA --> G_T1C

        %% Garmin Granularity Benchmark
        G_BIN["<b>Garmin Granularity Benchmark (Completed)</b><br/>N = 1,000 users, 5 OR-Tools allocations<br/>5-min: 0.8410 | 1-min: 0.8350 | 15-min: 0.8228<br/>↳ Strict Core: 5-min wins (0.8647 vs 0.8320)<br/>↳ Delta Stratum: 15-min rescues (+16.7 pt lift to 0.7687)"]:::completed
        G_T1C --> G_BIN

        P_T2["<b>PLANNED: Tier 2 Primary Garmin Expansion</b><br/>N = 5,636 (includes multi-source switchers)<br/>Evaluate device transition artifacts"]:::planned
        G_T1C -.-> P_T2
    end

    %% Right Branch: Apple Watch
    subgraph APPLE_BRANCH ["Apple Watch Cohort (Source 6) — Episodic Point Events"]
        A_RAW["Apple Single-Source Universe<br/>N = 11,143 eligible (34.6% F / 65.4% M)"]
        D1 -->|Episodic ~54% bin occupancy| A_RAW

        D3{"<b>DIVERGENCE 3: Gate Definition Conflict</b><br/>Interval Union (cov_s) vs. Clock-Hour Count"}
        A_RAW --> D3

        A_HIST["<b>Historical Sidelining (artifacts_sq)</b><br/>12h union rule (cov_s >= 43.2ks) failed for dt=0<br/>99.7% dropped on Ch3000 (only 28 passed)<br/>Only Ch3002 hourly resting modeled (RF=0.6814)"]:::completed
        D3 -->|Interval Union cov_s >= 12h| A_HIST

        A_ADAPT["<b>Day Adequacy Rediscovery</b><br/>Replaced interval rule with >=8h & >=60 events<br/>100% of users qualify (mean 265 days/user)"]:::completed
        D3 -->|Clock-Hour & Event Rule| A_ADAPT

        A_PILOT["<b>Apple Pilot Benchmark (Milestone)</b><br/>N = 1,000 users, 5 OR-Tools allocations<br/><b>MR 5-min (L=288): 0.7554 +/- 0.0594</b><br/>MR 15-min (L=96): 0.7431 +/- 0.0437<br/>Summary Baseline: 0.6029 +/- 0.0631<br/>Paired MR Gain: +0.1526"]:::milestone
        A_ADAPT --> A_PILOT

        %% Apple Planned
        P_WEAR["<b>PLANNED: Apple Wear Phenotypes & Imputation Ablation</b><br/>Decompose wear regimes & test boxcar pooling<br/>vs. flat-median poisoning"]:::planned
        A_PILOT -.-> P_WEAR

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

---

## 2. Inventory of Branches & Experiments

### Branch A: Garmin Cohort (Source 3)
| Status | Experiment / Node | Scope | Method / Model | Primary Result | Key Finding / Artifact |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **Ran** | Historical Baseline | $N = 3,848$ | Random Forest on 91-day aggregates | AUROC $0.6918$ | [`artifacts_sq/sidequest_report.md`](../experiments/artifacts_sq/sidequest_report.md) |
| **Ran** | Strict Tier 0 Baseline | $N = 3,848$ | MultiRocket (5m, clock-fill) + Ridge | AUROC $0.9100$ | [`learnings/multirocket_garmin_synthesis.md`](multirocket_garmin_synthesis.md) |
| **Ran** | Delta Zero-Shot | $N = 562$ | Frozen Tier 0 MultiRocket on wear-fail | AUROC $0.7850$ | Short span retains 0.8559; intermittent wear drops to 0.7599. [`HEARTBEAT_LAB/artifacts/eval_delta_562/`](../HEARTBEAT_LAB/artifacts/eval_delta_562) |
| **Ran** | **Tier 1c 10-Alloc CV (Milestone)** | $N = 4,410$ | MultiRocket + Ridge on 10 OR-Tools splits | **AUROC $0.9094 \pm 0.0058$** | Overall cohort firmly $>0.90$; strict stratum $= 0.9235$; retrained delta stratum $= 0.8156$. [`tier1c_10_alloc_results.csv`](../HEARTBEAT_LAB/artifacts/tier1c_cv_results/tier1c_10_alloc_results.csv) |
| **Ran** | **Garmin Granularity Benchmark** | $N = 1,000$ | MultiRocket 1m vs 5m vs 15m | **5m: $0.8410$ \| 1m: $0.8350$ \| 15m: $0.8228$** | Strict: 5m ($0.8647$) beats 15m ($0.8320$). Delta: 15m ($0.7687$) rescues 1m ($0.6022$) by **+0.1665 lift**. [`experiments/garmin_temporal_binning/README.md`](../experiments/garmin_temporal_binning/README.md) |
| **Will Run** | Tier 2 Multi-Source Expansion | $N = 5,636$ | MultiRocket on primary Garmin + switchers | TBD | Tests device transition artifacts. |

---

### Branch B: Apple Watch Cohort (Source 6)
| Status | Experiment / Node | Scope | Method / Model | Primary Result | Key Finding / Artifact |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **Ran** | Early Sidequest | $N = 4,367$ | Random Forest on Ch3002 resting hourly | AUROC $0.6814$ | Ch3000 was discarded because point-events failed the continuous interval coverage rule. |
| **Ran** | **Apple Pilot Benchmark (Milestone)** | $N = 1,000$ | MultiRocket 5m vs 15m vs Summary | **MR 5m: $0.7554 \pm 0.0594$**<br/>MR 15m: $0.7431 \pm 0.0437$<br/>Summary: $0.6029 \pm 0.0631$ | MultiRocket yields a **+15.3 point paired gain** over static metrics. 5-minute resolution beats 15-minute. [`experiments/apple_pilot_multirocket/README.md`](../experiments/apple_pilot_multirocket/README.md) |
| **Will Run** | Apple Wear Phenotypes & Imputation | $N = 1,000$ / $11,143$ | MultiRocket with boxcar pooling vs splines vs masks | TBD | Tests if flat median imputation poisons convolutions on episodic Apple data. |
| **Will Run** | Cross-Device Transfer | $N = 1,000$ | Frozen Garmin MultiRocket zero-shot on Apple | TBD | Measures cross-hardware domain shift. |
| **Will Run** | Channel 3002 Modeling | $N = 11,056$ | MultiRocket on 24-bin hourly resting HR | TBD | Tests watchOS resting algorithm vs. ambient HR. |
| **Will Run** | Full Cohort Scaling | $N = 11,143$ | 10-allocation CV on complete Apple dataset | TBD | Evaluates asymptotic sample size scaling. |

---

## 3. The 3 Crucial Divergence Points

1. **Divergence 1 (Sensor Sampling Physics):**
   - **Garmin:** Continuous 1-minute averages ($\approx 95\%$ daily bin occupancy). Low imputation reliance.
   - **Apple Watch:** Episodic point events ($\Delta t = 0\text{ ms}$, $\approx 54\%$ daily bin occupancy). Requires clock-median diurnal imputation.
2. **Divergence 2 (Gating Definition Conflict):**
   - In `experiments/artifacts_sq`, the coverage gate required $\ge 12\text{h}$ of interval union coverage ($\text{cov\_s} \ge 43,200\text{ s}$). Because Apple readings had $\Delta t = 0$, union duration evaluated to $\approx 0\text{ s}$, discarding Apple Ch3000.
   - Correcting the gate to **clock-hour coverage ($\ge 8\text{h}$) and count ($\ge 60$ readings)** restored 100% eligibility for Apple participants.
3. **Divergence 3 (Longitudinal Persistence vs. Missingness):**
   - In Garmin, relaxing the strict 13-week window added 562 wear-fail participants.
   - Retraining inside folds proved that MultiRocket can absorb wear intermittency, recovering delta test performance to **$0.8156$** while preserving the strict core at **$0.9235$**.
