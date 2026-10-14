# Apple Watch MultiRocket Pilot Benchmark ($N = 1,000$)

## Executive Summary

This pilot evaluated whether MultiRocket time-series convolutions can predict recorded salutation (female `10` vs. male `20`) on Apple Watch participants (Source 6) under sparse, event-driven heart rate sampling.

Historically (`experiments/artifacts_sq`), Apple Watch was largely sidelined because the strict longitudinal coverage gate required $\ge 12.0$ hours of continuous interval union coverage ($\text{STRICT\_COV\_S} = 43,200\text{ s}$). Because Apple Watch emits **instantaneous point events** ($\Delta t = 0\text{ ms}$) rather than 1-minute continuous interval buckets (like Garmin), union coverage evaluated near zero, disqualifying 99.7% of Apple participants on Channel 3000.

When evaluated using **clock-hour distribution and event count** ($\ge 8\text{ clock hours} \land \ge 60\text{ readings}$), **100% of the 1,000 sampled single-source Apple users qualified** with an average of **264.9 adequate days** per participant (54.1% daily bin occupancy at 5-minute resolution; 68.3% at 15-minute resolution).

---

## Benchmark Results (5 OR-Tools Balanced 70/15/15 Allocations)

All imputations, MultiRocket bias quantiles, scalers, and Ridge regressions were fitted strictly on training participants ($N_{\text{train}} = 700$, $N_{\text{val}} = 150$, $N_{\text{test}} = 150$) within each allocation.

| Allocation | MultiRocket 5-min (288 bins) | MultiRocket 15-min (96 bins) | Static HR Summary | Paired MR Gain (5m vs Summary) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | **0.8022** | 0.7987 | 0.6847 | **+0.1175** |
| **1** | **0.7704** | 0.7102 | 0.5599 | **+0.2105** |
| **2** | **0.8181** | 0.7824 | 0.6489 | **+0.1692** |
| **3** | 0.6949 | **0.7123** | 0.5895 | **+0.1054** |
| **4** | 0.6915 | **0.7121** | 0.5314 | **+0.1601** |
| **Mean $\pm$ SD** | **0.7554 $\pm$ 0.0594** | **0.7431 $\pm$ 0.0437** | **0.6029 $\pm$ 0.0631** | **+0.1526 $\pm$ 0.0412** |

---

## Key Scientific Findings

1. **MultiRocket Delivers Substantial Value on Apple Watch:**
   Even with ~46% missing 5-minute bins filled with population clock-time medians, MultiRocket achieves a **mean Test AUROC of 0.7554** (ranging up to **0.8181**), representing a massive **+0.1526 (+15.3 points) paired gain** over static heart rate summaries ($0.6029$).
2. **Temporal Resolution Tradeoff (5-min vs 15-min):**
   - **5-minute bins (288 bins, 54.1% occupancy):** $\text{AUROC} = 0.7554 \pm 0.0594$.
   - **15-minute bins (96 bins, 68.3% occupancy):** $\text{AUROC} = 0.7431 \pm 0.0437$.
   - **Insight:** Finer temporal resolution with clock-median imputation outperforms coarser 15-minute averaging by **+0.0123 AUROC**, though the 15-minute representation exhibits slightly lower cross-fold variance ($\pm 0.0437$ vs $\pm 0.0594$) and runs $2.5\times$ faster.
3. **Comparison with Garmin:**
   - On Garmin ($N=1,000$ to $4,410$), MultiRocket reaches **~0.908 AUROC** due to continuous 1-minute PPG tracking (>90% bin occupancy).
   - On Apple Watch ($N=1,000$), MultiRocket reaches **~0.755 AUROC** due to sparse, intermittent event sampling (~54% occupancy).
   - Crucially, the **marginal contribution of time-series convolution** is large on both devices: **+0.192 on Garmin**, and **+0.153 on Apple Watch**.
