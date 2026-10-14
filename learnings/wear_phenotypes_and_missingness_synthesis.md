# Wear Phenotypes, Missingness Architectures, and Convolutional Dynamics: Cross-Vendor Synthesis (Garmin vs. Apple Watch)

This document synthesizes key discoveries regarding **behavioral wear phenotypes**, **sensor-level sampling physics**, and their interaction with **MultiRocket temporal convolutions and imputation strategies** across the German COVID-19 Data Donation (*Datenspende*) cohort.

---

## 1. Executive Summary: The Missingness-Convolution Interaction

A central finding of this investigation is that missing wearable data is not missing at random (MCAR), nor does it behave as uniform white noise. Instead:
1. **Behavioral Phenotypes Drive Missingness Structure:** Cohorts naturally partition into distinct physiological-use clusters: continuous wearers, chronic intermittent wearers (weeks off-wrist), and diurnal chargers (nightly bedside charging).
2. **Convolutional Poisoning by Static Imputation:** In high-frequency grids (1-minute, 5-minute), filling multi-hour missingness gaps with a flat population clock-median poisons MultiRocket convolutional kernels. Kernels compute differences ($\Delta x_t = x_{t+d} - x_t$) that evaluate to exact zeros ($\Delta x_t = 0$) across imputed spans, bounded by artificial step-discontinuities at observed boundaries.
3. **Temporal Boxcar Pooling as Physical Recovery:** Downsampling to 15-minute bins ($L=96$) acts as physical boxcar pooling. Any single valid sensor reading within a 900-second window populates the bin with true physiological signal, dramatically shrinking imputation reliance and lifting classification AUROC by up to **+16.7 to +23.4 points** on fragmented wearers.

---

## 2. Garmin Wear Phenotypes & Temporal Resolution Dynamics

### 2.1 Cohort Phenotype Decomposition ($N = 4,410$, Tier 1c)
When relaxing the strict longitudinal continuity gate ($\ge 11$ adequate weeks out of 13), $N = 562$ "wear-fail" delta participants were admitted. Decomposing this intake revealed that failure to meet the continuous window is dominated by chronic intermittency:

| Phenotype | Sample Size ($n$) | Share of Delta | Median Tenure | Median Wear Density | Median Max Gap | Active Hours / Day |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strict Core (Tier 0)** | 3,848 | — | 435 days | $>0.95$ | $\le 7$ days | $23.2$ h/day |
| **Phenotype A (Intermittent Wearers)** | **432** | **$76.9\%$** | **460 days** | **$0.41$** | **82.5 days** ($\approx 2.7$ mo) | **$14.3$ h/day** |
| **Phenotype B (Short Tenure / Dropouts)** | **128** | **$22.8\%$** | **53.5 days** | **$1.00$** | **1.0 day** | **23.5 h/day** |
| **Empty Non-Wearers** | 2 | $0.4\%$ | — | $0.00$ | — | $0.0$ h/day |

- **Phenotype B participants** are continuous wearers who simply stopped participating early. Because their daily recordings are complete, MultiRocket zero-shot performance remains high (**0.8559 AUROC**).
- **Phenotype A participants** represent the primary challenge: users with long enrollment spans who wear the device intermittently for a few days, abandon it for 1–3 months, and wear it for only $\approx 14$ hours/day when active.

---

### 2.2 Garmin Temporal Granularity Benchmark Results ($N = 1,000$, 5-Allocation CV)

We evaluated 1-minute ($L=1,440$), 5-minute ($L=288$), and 15-minute ($L=96$) binning across 5 demographic/label-balanced allocations on 8 CPU cores ([`garmin_binning_benchmark_results.csv`](../experiments/garmin_temporal_binning/artifacts/garmin_binning_benchmark_results.csv)):

| Model / Resolution | Grid Length ($L$) | Overall Test AUROC | Strict Stratum AUROC | Delta Stratum AUROC | Mean CPU Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MultiRocket 1-min** | 1,440 | $0.8350 \pm 0.0369$ | **0.8645** | $0.6022$ | 356.4 s |
| **MultiRocket 5-min (Baseline)** | 288 | **$0.8410 \pm 0.0380$** | **0.8647** | $0.6753$ | 84.7 s |
| **MultiRocket 15-min** | 96 | $0.8228 \pm 0.0411$ | 0.8320 | **0.7687** | **39.3 s** |
| **Summary Ridge Baseline** | 8 statics | $0.7143 \pm 0.0360$ | 0.7289 | 0.6225 | 0.01 s |

#### The Fundamental Resolution Tradeoff:
1. **On Continuous Wearers (Strict Stratum):** Fine binning wins (**0.8647 at 5m vs. 0.8320 at 15m**, $+0.0327$). When real observations exist, 5-minute resolution captures autonomic transient slopes and post-exertional cardiac recovery dynamics that are smoothed out by 15-minute pooling.
2. **On Intermittent Wearers (Delta Stratum):** Coarse binning rescues performance (**0.7687 at 15m vs. 0.6022 at 1m**, **+0.1665 lift**; and **+0.0934 over 5m**). At 1-minute resolution, flat median filling across 10-hour daily absences collapses AUROC to near chance. 15-minute pooling ensures that intermittent wearers are represented by genuine physiological measurements rather than synthetic flatlines.

---

## 3. Apple Watch Missingness Patterns & Behavioral Phenotypes

### 3.1 Sensor-Level Sampling Physics (watchOS Adaptive Triggering)
Apple Watch (Source 6, Channel 3000) does **not** log periodic averages. It emits discrete instantaneous point events ($\Delta t = 0$) governed by battery-optimization algorithms:
- **Sedentary / Rest:** Background checks occur every **$5\text{ to }8\text{ minutes}$** ($p_{50} = 32\text{ s}, p_{90} = 5.7\text{ min}, p_{95} = 7.9\text{ min}$).
- **Active / Exertion:** High-frequency bursts occur every **$1\text{ to }3\text{ seconds}$** ($p_{10} = 1.0\text{ s}, p_{25} = 3.0\text{ s}$).
- **Daily Reading Volume:** Median **280 events/day** (IQR: $[196, 605]$). Apple yields $\approx 5\times$ fewer discrete points per day than Garmin (1,440/day).

---

### 3.2 Diurnal Missingness Curve Across 39,208 Participant-Days

Analysis of the 1,000 Apple pilot participants revealed that missingness is heavily structured by time-of-day:

| Clock Window | Hourly Presence ($\ge 1$ event/hr) | 5-min Bin Occupancy ($L=288$) | Dominant Missingness Driver |
| :--- | :---: | :---: | :--- |
| **Night (00:00 – 06:00)** | **$50.2\% \text{ -- } 56.5\%$** | **$43.0\% \text{ -- } 44.7\%$** | **Off-wrist bedside charging** + sleeping |
| **Morning (07:00 – 09:00)** | $76.0\% \text{ -- } 88.6\%$ | $49.8\% \text{ -- } 60.3\%$ | Waking, putting watch on wrist |
| **Daytime (10:00 – 17:00)** | **$93.2\% \text{ -- } 95.0\%$** | **$63.5\% \text{ -- } 66.8\%$** | **Episodic sedentary sampling** (1/3 bins empty) |
| **Evening (18:00 – 23:00)** | $62.3\% \text{ -- } 92.7\%$ | $45.0\% \text{ -- } 61.7\%$ | Taking watch off before sleep |

- **Wakefulness Paradox:** Even when compliant wearers wear the watch throughout the daytime ($95\%$ hourly presence), **$34\text{--}36\%$ of 5-minute bins remain empty** because sedentary background checks are spaced 6–8 minutes apart.
- **Night Absence:** Across all user-days, **$35.0\%$ of days have zero readings between 00:00 and 06:00**.

---

### 3.3 Apple Behavioral Phenotypes: Chargers vs. Sleep Trackers
The Apple population splits cleanly along night-wear habits:
1. **"Night-Chargers" ($\approx 25.3\%$ of users):**
   - Wear watch $<10\%$ of nights (median nocturnal hours = 0).
   - The entire 6-hour nocturnal trajectory is 100% missing every single day, forcing flat-median imputation across 72 consecutive 5-minute bins.
2. **"Sleep Trackers / Continuous Wearers" ($\approx 58.7\%$ of users):**
   - Wear watch $>50\%$ of nights, averaging 4.8 active hours between 00:00 and 06:00.
3. **Irregular Switchers ($\approx 16.0\%$):**
   - Charge watch overnight 1–3 times per week.

---

## 4. Cross-Vendor Comparative Matrix

| Dimension | Garmin (Source 3) | Apple Watch (Source 6) |
| :--- | :--- | :--- |
| **Raw Sampling Architecture** | Continuous 1-minute periodic averages | Episodic instantaneous point events ($\Delta t = 0$) |
| **Daily Data Volume** | Deterministic 1,440 readings/day | Variable: median 280 events/day ($p_{25}=196, p_{75}=605$) |
| **5-min Bin Occupancy** | $\approx 95\%$ (in compliant wearers) | **$55.3\%$** across cohort ($65\%$ daytime, $43\%$ night) |
| **Dominant Wear Failure Mode** | **Phenotype A:** Multi-month vacation/abandonment gaps (median 82.5d max gap) | **Night-Charging:** 25.3% of users charge bedside nightly (6h daily nocturnal gap) |
| **MultiRocket Baseline AUROC** | **$0.9094 \pm 0.0058$** (Tier 1c, $N=4,410$) | **$0.7554 \pm 0.0594$** (Pilot, $N=1,000$) |
| **Paired Gain over Statics** | **+0.1898** (vs. Summary 0.7196) | **+0.1526** (vs. Summary 0.6029) |
| **Resolution Sensitivity** | Strict: 5m ($0.865$) $>$ 15m ($0.832$)<br/>Delta: 15m ($0.769$) $\gg$ 5m ($0.675$) $\gg$ 1m ($0.602$) | 5m ($0.755$) $\approx$ 15m ($0.743$)<br/>Ablation with splines/masks pending |

---

## 5. Methodological & Architectural Recommendations

1. **Mandatory Wear-Stratified Reporting:**
   - Any wearable benchmark must report performance separately for **continuous wearers** vs. **intermittent / charging wearers**. Reporting a single pooled AUROC conceals large disparities (e.g., Garmin's 0.865 strict vs. 0.602 delta at 1-min).
2. **Avoid High-Resolution Flat-Median Imputation on Fragmented Data:**
   - When missingness exceeds 30%, flat median imputation degrades temporal convolution. For fragmented cohorts, **temporal boxcar binning (15m or 30m)** or **smooth diurnal splines** should be preferred to preserve kernel validity.
3. **Dual-Channel MultiRocket for Episodic Wear:**
   - Future Apple and relaxed Garmin models should feed a 2-channel input: Channel 0 as normalized heart rate, and Channel 1 as a binary observation indicator ($m_t \in \{0, 1\}$). This enables linear ridge heads to learn distinct weights for measured dynamics versus imputed baselines.
