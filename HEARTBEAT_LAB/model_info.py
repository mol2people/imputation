"""Human-readable names and explanations for the saved experiments."""
from collections import Counter

import pandas as pd
import streamlit as st

MODEL_NAMES = {"summary": "HR summaries", "multirocket": "MultiRocket + HR summaries"}
FILL_NAMES = {"clock": "Median for each clock time", "global": "One overall median"}


def model_labels(metadata):
    labels = {}
    for name, info in metadata.items():
        title = "HR summaries" if info["model"] == "summary" else "MultiRocket"
        if info["model"] == "multirocket":
            title += " · " + {"clock": "clock-time fill", "global": "global fill"}.get(info["filling"], info["filling"])
        elif info["model"] != "summary":
            title = info["model"]
        if info.get("protocol") == "temporal":
            title += " · time split"
        days = info.get("day_budget")
        if days:
            title += f" · ≤{days} days"
        elif days == 0:
            title += " · all days"
        labels[name] = title
    counts = Counter(labels.values())
    runs = Counter()
    for name, title in labels.items():
        if counts[title] > 1:
            runs[title] += 1
            labels[name] = f"{title} · run {runs[title]}"
    return labels


def describe_model(info):
    if info["model"] == "summary":
        return ("Baseline using HR level, variability, weekday/weekend differences and recording coverage. "
                "A ridge regression combines 305 summary features; it does not use the ordered HR values within a day.")
    if info["model"] != "multirocket":
        return "See this bundle's saved configuration for its model specification."
    fill = ("Missing five-minute HR bins use the training median for that clock time (with an overall-median fallback)."
            if info["filling"] == "clock" else
            "Missing five-minute HR bins all use the same overall training median.")
    return ("Adds MultiRocket features of within-day HR patterns and changes to the same 305 summary features. "
            "Daily pattern features are pooled across days and combined by ridge regression. " + fill)


def with_model_labels(table, labels):
    return table.assign(model_label=table.model.map(labels))


def show_model_details(name, info, label, calibration):
    with st.expander(f"Model details — {label}"):
        protocol = info.get("protocol")
        if protocol == "temporal":
            st.write(f"**Time split:** training days precede {info['train_before']}; validation customers first appear "
                     f"from {info['train_before']} to before {info['test_from']}; test customers first appear on/after "
                     f"{info['test_from']}. Customers are disjoint and histories are truncated to their assigned window.")
        elif protocol == "random":
            st.write("**Random customer split:** stratified 70% training / 15% validation / 15% test. "
                     "All days from one customer stay in that customer's split.")
        else:
            st.write("Split protocol was not recorded in this bundle's metadata.")
        budget = info.get("day_budget")
        st.write(f"**History used for fitting/evaluation:** {'all adequate days in each assigned window' if budget == 0 else f'first {budget} adequate days per customer/window' if budget else 'budget not recorded'}. "
                 f"Actual training histories contain {info['training_days']['min']}–{info['training_days']['max']} days.")
        st.write(f"**Ridge penalty α:** {info['alpha']:g}. **Seed:** {info['seed']}. "
                 f"**Probability calibration:** sigmoid fitted to {calibration['n']} validation customers, "
                 f"including {calibration['n_F']} F and {calibration['n_M']} M.")
        if info.get("max_customers"):
            st.caption(f"Initial customer budget: {info['max_customers']}; temporal eligibility may reduce the retained cohort.")
        rows = [{"Split": "Training", "Customers": info["training_customers"], "Customer AUROC": None}]
        for split, title in [("val", "Validation / calibration"), ("test", "Test")]:
            metric = info["metrics"][split]
            rows.append({"Split": title, "Customers": metric["n"], "Customer AUROC": metric["auroc"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("AUROC evaluates one whole/budgeted history per customer, not individual days. "
                   "Bundles with different splits or history budgets are different experiments.")
        st.caption(f"Saved bundle ID: {name}. CSVs retain this ID alongside the readable model label.")


def show_methods():
    with st.expander("Methods — from uploaded HR to predictions"):
        st.markdown("""
**1. Read and filter HR.** One person's epoch CSV is converted to local clock time
using each observation's timezone offset. Native study files use Garmin source 3,
HR type 3000. Valid HR is 25–230 bpm. Duplicate observations are retained.

**2. Build adequate days.** A day needs at least 8 distinct observed clock-hours
and either 60 readings or 8 hours of interval-union coverage. Intervals are assigned
to their start date, including those crossing midnight. Each day becomes 288
five-minute bins, averaging readings that start in each bin. Gaps stay missing
until the chosen model fills them; there is no interpolation or per-person HR normalization.

**3. Construct model features.** The **HR summaries** baseline uses eight daily
statistics: mean, median, SD, minimum, maximum, reading count, interval coverage
and observed clock-hours. Their means and sample SDs across days, a weekday-minus-weekend
HR contrast, and 288 observation frequencies give **305 features**. Coverage therefore
contributes to predictions as well as HR values.

**MultiRocket** includes all those summaries, then applies convolution filters
at multiple temporal scales to each day's HR curve and its successive differences.
At the default setting this gives 9,408 features per day. Their means and sample SDs
across days add 18,816 features, for **19,121 input features** before constant columns
are removed. Clock-time fill uses a different training median for each five-minute
clock position; global fill uses one training median for every gap. Neither fill
learns from the uploaded customer. Pooling summarizes days; it does not model
the order of days in a customer's history.

**4. Fit and evaluate by customer.** The base model's gap filling, MultiRocket
parameters, feature-median imputation, constant-column removal, standardization
and ridge regression are fitted using training customers only. Random splitting
uses disjoint customers in 70/15/15 proportions. Time splitting uses first adequate
recorded dates to assign earlier/later customers and truncates histories to the
appropriate calendar window. The selected model's details show its actual dates,
sample sizes, day budget and ridge penalty. Small temporal pilots are different
experiments from the full-cohort models.

**5. Convert scores to percentages.** Ridge gives an unbounded score. A sigmoid
fitted to validation customers maps it to estimated P(M); P(F) = 1 − P(M). Test
customers do not fit that mapping. The displayed label is M at ≥50%, otherwise F.
Reliability plots compare predicted P(M) with observed M frequency; Brier score
and log loss measure overall probabilistic accuracy (lower is better). This is
still the previously explored study cohort, not an external validation cohort.

**6. Read the timelines and stress tests.** Individual-day predictions apply the
history-trained model to one day. Cumulative predictions use days up to each date.
For a single day, undefined summary features use training medians and the between-day
SD of MultiRocket features is zero. Both timelines transfer the customer-history
calibration, so their percentages are **pseudo-probabilities**, not validated daily
probabilities. The saved training day budget is not a cap on the exploratory timeline.

Shuffling raw readings preserves each selected day's HR values and timestamps but
can change the distribution of five-minute averages. Shuffling already-binned HR
preserves the observed bin values and gap mask. Resampling draws from each day's
empirical HR distribution; thinning removes readings; clock shifts move readings
and gaps together within that date. Each run keeps the model and calibration fixed.
The plotted 5th–95th percentile band shows sensitivity across randomizations,
not a confidence interval. Days that fail adequacy after perturbation lose their prediction.
""")
