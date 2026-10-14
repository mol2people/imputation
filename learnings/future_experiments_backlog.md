# Future Experiments Backlog

This document captures prioritized follow-up experiments and architectural ablations identified during the Garmin relaxed cohort evaluation and the Apple Watch pilot benchmark.

---

## 1. Garmin Temporal Bin Size & Granularity Ablation

**Motivation:**
Currently, all Garmin MultiRocket models discretize raw heart rate epochs into 5-minute intervals ($L = 288$ bins/day). However, Garmin raw exports emit continuous 1-minute averaged heart rate readings. It is an open question whether finer or coarser temporal discretization materially alters MultiRocket's discriminative power or merely shifts computational overhead.

### Hypotheses & Dynamical Systems Tradeoffs
- **1-minute bins ($L = 1,440$ bins/day):**
  - *Mechanism:* Preserves high-frequency cardiac autonomic dynamics, transient heart rate variability (HRV) proxies, and rapid post-exertional recovery slopes.
  - *Tradeoffs:* Time series length increases $5\times$. MultiRocket's fixed kernel length of 9 at dilation 1 spans only 9 minutes (vs. 45 minutes at 5-min bins). To capture multi-hour circadian rhythms, maximum dilation would need adjustment, and feature extraction compute increases $\approx 4\text{--}5\times$.
- **2.5-minute bins ($L = 576$ bins/day):**
  - Intermediate resolution balancing ultradian transient fidelity with tractable kernel spans.
- **5-minute bins ($L = 288$ bins/day, Baseline):**
  - Current gold standard ($0.910$ AUROC). Low-pass filters high-frequency sensor noise while preserving sharp diurnal transitions (morning surge, nocturnal dip, postprandial changes).
- **10-minute ($L = 144$) and 15-minute ($L = 96$) bins:**
  - *Mechanism:* Tests whether fine-grained inflection dynamics are necessary, or if coarse circadian envelope/amplitude alone explains the high AUROC.
  - *Advantage:* Speeds up transform compute by $2\text{--}3\times$.

### Experimental Design
- **Cohort:** Matched sample of 1,000 Garmin participants (or full Tier 0 $N = 3,848$).
- **Ablation Matrix:** $L \in \{1440, 576, 288, 144, 96\}$.
- **Metrics:** Test AUROC (paired difference vs. 288-bin baseline), Brier score, feature transform runtime, and memory footprint.

---

## 2. Cross-Device Transfer (Garmin $\to$ Apple Watch)

**Motivation:**
Does a MultiRocket model trained purely on Garmin continuous 1-minute data transfer zero-shot to Apple Watch episodic point-event data, or does device-specific sampling impose a domain shift penalty?

### Experimental Design
- Take the frozen Garmin MultiRocket model (fitted on 3,848 Garmin participants in `HEARTBEAT_LAB/artifacts/models/`).
- Evaluate zero-shot on the 1,000 Apple Watch pilot participants (`experiments/apple_pilot_multirocket/`).
- Compare zero-shot AUROC against the within-Apple trained benchmark ($0.7554$).

---

## 3. Apple Channel 3002 (Hourly Resting Heart Rate)

**Motivation:**
In Apple Watch exports, Channel 3000 represents ambient instantaneous HR point events, while Channel 3002 represents hourly resting heart rate computed by watchOS.

### Experimental Design
- Evaluate 24-bin daily curves ($L = 24$, 1 reading/hour) using MultiRocket and Ridge regression across the 11,056 eligible Apple users with Channel 3002.
- Test whether vendor-filtered resting HR carries higher signal-to-noise than unconstrained ambient daytime readings.

---

## 4. Full Apple Cohort Scaling ($N = 11,143$)

**Motivation:**
The pilot benchmark demonstrated that MultiRocket achieves $0.7554 \pm 0.0594$ AUROC on $N = 1,000$ Apple users. Scaling to the full single-source cohort ($N = 11,143$, 2.5x larger than Garmin) will determine asymptotic sample-efficiency and whether a larger training set compensates for episodic sampling.

### Execution Plan
- Batch process all 11,143 Apple CSVs into 5-minute `.npz` caches using multi-process pool.
- Generate 10-allocation OR-Tools balanced splits.
- Run 10-allocation cross-validation with 7 CPU threads.

---

## 5. Garmin Tier 2 Multi-Source Expansion ($N = 5,636$)

**Motivation:**
Evaluate the 1,226 participants who switched between Garmin and other wearable brands to test whether device transitions induce systematic artifact leakage or feature distortion.
