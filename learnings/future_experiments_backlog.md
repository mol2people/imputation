# Future Experiments Backlog

This document captures prioritized follow-up experiments and architectural ablations identified during the Garmin relaxed cohort evaluation and the Apple Watch pilot benchmark.

---

## 1. Garmin Temporal Bin Size & Granularity Ablation (In Progress / Benchmark Running)

**Motivation:**
Discretizing continuous Garmin 1-minute HR epochs into 5-minute intervals ($L = 288$ bins/day) is standard, but does finer (1-min) or coarser (15-min) binning alter discriminative power across wear phenotypes?

### Initial Findings (from 1,000-user 5-allocation benchmark):
- **Continuous Wearers (Strict Stratum):** Fine binning wins (5-min AUROC **0.8620** vs. 15-min **0.8254**).
- **Intermittent Wearers (Delta Stratum):** Coarse binning prevents missingness collapse. Moving from 1-min ($0.5099$) or 5-min ($0.6054$) to 15-min ($0.7441$) delivers a **+0.1387 to +0.2342 AUROC lift**, proving that physical boxcar aggregation vastly outperforms flat median imputation across gaps.

---

## 2. Apple Watch: Wear Phenotypes, Missingness Artifacts, and Imputation/Resolution Ablation

**Motivation:**
In Garmin, we discovered that 76.9% of relaxed-cohort users are chronic intermittent wearers (Phenotype A), and that flat population-median imputation poisons MultiRocket convolutions across multi-hour gaps (collapsing AUROC to $\approx 0.51$ at 1-min resolution), which 15-minute boxcar pooling rescues (+14 to +23 points).

Apple Watch (Source 6) features **intrinsically episodic point-event sampling** (only $\approx 54\%$ bin occupancy at 5-minute resolution even for compliant wearers, with large missing nocturnal stretches when watches are charged). It is highly probable that the Apple pilot benchmark ($0.7554$ AUROC) is partially suppressed by flat-median imputation artifacts over empty intervals.

### Hypotheses & Dynamical Mechanisms
1. **Wear Phenotype Heterogeneity in Apple:**
   - Apple users likely exhibit distinct wear patterns: 24/7 continuous wearers, daytime-only wearers (nightly charging, $\approx 14\text{--}16\text{h}$ coverage), and episodic/intermittent wearers.
   - Does MultiRocket achieve $>0.85$ AUROC on 24/7 Apple wearers while collapsing on daytime-only/intermittent wearers?
2. **Convolutional Poisoning by Median Imputation:**
   - MultiRocket differences ($\Delta x_t = x_{t+d} - x_t$) over flat median fills evaluate to exact zeros ($\Delta x_t = 0$), bounded by artificial step-discontinuities where real data resumes.
   - For daytime-only wearers, 8 hours of imputed nocturnal flatlines force kernels to fit synthetic boundary steps rather than circadian dynamics.
3. **Imputation & Resolution Remedies:**
   - **Coarser Boxcar Pooling:** Test 15-min ($L=96$), 30-min ($L=48$), and 60-min ($L=24$) aggregations.
   - **Smooth Inter-Day Diurnal Splines:** Replace flat clock medians with diurnal cubic spline interpolation or forward-fill with decay to eliminate step discontinuities.
   - **Dual-Channel MultiRocket (Trajectory + Mask):** Input Channel 0 as HR trajectory and Channel 1 as a binary observation mask ($m_t \in \{0, 1\}$), allowing kernels to condition on whether a reading is observed or imputed.

### Experimental Design
- **Cohort:** 1,000-user balanced pilot allocation (or full $N = 11,143$ Apple cohort).
- **Phenotyping:** Quantify longitudinal wear density, max gap length, active hours/day, and night-time wear fraction.
- **Ablation Grid:**
  - Imputation: Population Clock Median vs. Linear/Spline Interpolation vs. Masked Dual-Channel.
  - Temporal Granularity: 5-min ($L=288$) vs. 15-min ($L=96$) vs. 30-min ($L=48$) vs. 60-min ($L=24$).
- **Metrics:** Test AUROC overall and stratified by wear density tiers ($>80\%$, $50\text{--}80\%$, $<50\%$) and night wear presence.

---

## 3. Cross-Device Transfer (Garmin $\to$ Apple Watch)

**Motivation:**
Does a MultiRocket model trained purely on Garmin continuous 1-minute data transfer zero-shot to Apple Watch episodic point-event data, or does device-specific sampling impose a domain shift penalty?

### Experimental Design
- Take the frozen Garmin MultiRocket model (fitted on 3,848 Garmin participants in `HEARTBEAT_LAB/artifacts/models/`).
- Evaluate zero-shot on the 1,000 Apple Watch pilot participants (`experiments/apple_pilot_multirocket/`).
- Compare zero-shot AUROC against the within-Apple trained benchmark ($0.7554$).

---

## 4. Apple Channel 3002 (Hourly Resting Heart Rate)

**Motivation:**
In Apple Watch exports, Channel 3000 represents ambient instantaneous HR point events, while Channel 3002 represents hourly resting heart rate computed by watchOS.

### Experimental Design
- Evaluate 24-bin daily curves ($L = 24$, 1 reading/hour) using MultiRocket and Ridge regression across the 11,056 eligible Apple users with Channel 3002.
- Test whether vendor-filtered resting HR carries higher signal-to-noise than unconstrained ambient daytime readings.

---

## 5. Full Apple Cohort Scaling ($N = 11,143$)

**Motivation:**
The pilot benchmark demonstrated that MultiRocket achieves $0.7554 \pm 0.0594$ AUROC on $N = 1,000$ Apple users. Scaling to the full single-source cohort ($N = 11,143$, 2.5x larger than Garmin) will determine asymptotic sample-efficiency and whether a larger training set compensates for episodic sampling.

### Execution Plan
- Batch process all 11,143 Apple CSVs into 5-minute `.npz` caches using multi-process pool.
- Generate 10-allocation OR-Tools balanced splits.
- Run 10-allocation cross-validation with 7 CPU threads.

---

## 6. Garmin Tier 2 Multi-Source Expansion ($N = 5,636$)

**Motivation:**
Evaluate the 1,226 participants who switched between Garmin and other wearable brands to test whether device transitions induce systematic artifact leakage or feature distortion.
