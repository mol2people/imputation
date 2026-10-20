#!/usr/bin/env python3
"""Build publication-grade scientific reports as PDF categorized by scientific provenance.

Uses pandoc (pypandoc_binary) + typst in pure Python (zero headless chrome).
Translates Markdown, tables, figures, and mathematical typography into lab-report PDFs.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pypandoc
from pypdf import PdfReader
import typst

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pdf_builder")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_BASE = REPO_ROOT / "reports" / "scientific_provenance_pdfs"
TEMP_TYP_DIR = Path("/tmp/typst_build")
TEMP_TYP_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ReportConfig:
    report_id: str
    provenance_category: str
    category_label: str
    source_md: Path
    output_pdf_name: str
    date: str
    title: str
    cohort: str
    target_estimand: str
    epistemic_classification: str
    validation_protocol: str
    key_findings_summary: str
    figures: list[tuple[Path, str]]  # (path, caption)


REPORTS_CONFIG: list[ReportConfig] = [
    ReportConfig(
        report_id="EXP-01-EXPLORATORY-DISCOVERY",
        provenance_category="01_exploratory_discovery",
        category_label="01 Exploratory Discovery",
        source_md=REPO_ROOT / "experiments" / "female_infradian_cycles" / "results" / "REPORT.md",
        output_pdf_name="2026-10-20_female-infradian-cycles_exploratory-spectral-discovery.pdf",
        date="2026-10-20",
        title="Cohort-Scale Infradian Rhythmicity in Continuous Heart Rate: Exploratory Spectral Discovery",
        cohort="German Corona Data Donation — Apple Watch Single-Source (N = 11,139)",
        target_estimand="Observational screening for 25–35 day infradian rhythmicity across sex and age strata",
        epistemic_classification="Exploratory Discovery: Unstratified cohort screening; hypothesis generation",
        validation_protocol="Lomb-Scargle Periodogram & Single-Harmonic Cosinor Regression; wear-time artifact contrast",
        key_findings_summary=(
            "Observed distinct spectral peak at ~28.5 days in young reproductive-age females (18–35y), "
            "largely attenuated in older females and males. Physiology amplitude ~1.67 bpm markedly distinct from wear artifacts."
        ),
        figures=[
            (
                REPO_ROOT / "experiments" / "female_infradian_cycles" / "figures" / "fig1_cohort_spectral_density.png",
                "Cohort-wide Lomb-Scargle spectral power density across sex and age groups.",
            ),
            (
                REPO_ROOT / "experiments" / "female_infradian_cycles" / "figures" / "fig2_individual_periodograms_and_series.png",
                "Individual longitudinal resting heart rate series and single-subject periodograms.",
            ),
            (
                REPO_ROOT / "experiments" / "female_infradian_cycles" / "figures" / "fig3_physiology_vs_wear_periodograms.png",
                "Physiological infradian periodicity contrasted against sensor wear-time compliance periodograms.",
            ),
        ],
    ),
    ReportConfig(
        report_id="EXP-02-CALIBRATED-HYPOTHESIS-TESTING",
        provenance_category="02_calibrated_hypothesis_testing",
        category_label="02 Calibrated Hypothesis Testing",
        source_md=REPO_ROOT / "experiments" / "female_infradian_cycles_v2" / "results" / "REPORT.md",
        output_pdf_name="2026-10-20_female-infradian-cycles-v2_calibrated-hypothesis-testing.pdf",
        date="2026-10-20",
        title="Standardized Exposure & Continuous AR(1) Bootstrap Validation of Infradian Heart-Rate Cycles",
        cohort="Qualified Apple Watch Cohort with >= 120-Day Standardized Continuous Window (N = 4,631)",
        target_estimand="Statistical excess spectral power in 25–35 day window calibrated against AR(1) surrogate noise",
        epistemic_classification="Calibrated Hypothesis Testing: Controlled exposure null testing against serially correlated noise",
        validation_protocol="Per-participant Continuous AR(1) bootstrap (B = 199), HC3 robust Wald tests, Holm multiplicity adjustment",
        key_findings_summary=(
            "Young females display significant 25–35 day excess power over subject-specific red-noise nulls "
            "(Wald chi^2 = 14.82, p < 0.001). Wear-compliance metric showed zero significant excess power, confirming biological origin."
        ),
        figures=[
            (
                REPO_ROOT / "experiments" / "female_infradian_cycles_v2" / "figures" / "fig1_excess_power_contrasts.png",
                "Calibrated excess infradian power (E_i) and cosinor amplitude (A_i) contrasts across age strata with HC3 95% CIs.",
            ),
            (
                REPO_ROOT / "experiments" / "female_infradian_cycles_v2" / "figures" / "fig2_wear_compliance_dissociation.png",
                "Double dissociation between physiological infradian signal and sensor wear-time sampling behavior.",
            ),
            (
                REPO_ROOT / "experiments" / "female_infradian_cycles_v2" / "figures" / "fig3_raw_periodograms_3_to_60d.png",
                "Full continuous spectral periodograms (3 to 60 days) across age and sex cells with continuous AR(1) surrogate envelopes.",
            ),
            (
                REPO_ROOT / "experiments" / "female_infradian_cycles_v2" / "figures" / "fig4_excess_power_and_edge_argmax.png",
                "Argmax period distributions across the 25–35 day grid contrasted against continuous AR(1) and white noise nulls.",
            ),
            (
                REPO_ROOT / "experiments" / "female_infradian_cycles_v2" / "figures" / "fig5_two_component_null_spectrum.png",
                "Two-component continuous AR(1)+slow null model recalibration matching empirical red-noise power up to 60 days.",
            ),
            (
                REPO_ROOT / "experiments" / "female_infradian_cycles_v2" / "figures" / "fig6_individual_case_trajectories.png",
                "Representative individual longitudinal resting heart rate trajectories and fitted harmonic waveforms.",
            ),
        ],
    ),
    ReportConfig(
        report_id="EXP-03-CONFIRMATORY-MECHANISMS",
        provenance_category="03_confirmatory_mechanisms",
        category_label="03 Confirmatory Mechanisms",
        source_md=REPO_ROOT / "experiments" / "female_infradian_extensions" / "results" / "REPORT.md",
        output_pdf_name="2026-10-20_female-infradian-extensions_confirmatory-mechanisms.pdf",
        date="2026-10-20",
        title="Longitudinal Period Stability, Acrophase Phase-Locking, and Metabolic Invariance of Infradian Rhythms",
        cohort="Long-History Cohort (N = 502 with >= 365 days) and WHO BMI Strata (Underweight, Normal, Overweight, Obese)",
        target_estimand="Sliding 90-day period drift, calendar acrophase Rayleigh uniformity, and BMI ANCOVA contrasts",
        epistemic_classification="Confirmatory Mechanisms: Longitudinal persistence, circular uniformity, and biological covariate invariance",
        validation_protocol="Sliding 90-day window tracking, Rayleigh test of circular uniformity, multivariable OLS ANCOVA with HC3 errors",
        key_findings_summary=(
            "Period stability confirmed over >= 365 days (mean drift < 1.2 days). Acrophase distribution is circular uniform "
            "(Rayleigh p = 0.870), ruling out calendar/lunar artifact (wear showed p = 2.25e-11). Amplitude invariant to BMI (p = 0.609)."
        ),
        figures=[
            (
                REPO_ROOT / "experiments" / "female_infradian_extensions" / "figures" / "fig1_spectrogram_longitudinal_cases.png",
                "Sliding 90-day spectrograms demonstrating longitudinal infradian cycle stability across calendar days.",
            ),
            (
                REPO_ROOT / "experiments" / "female_infradian_extensions" / "figures" / "fig2_acrophase_circular_polar_plots.png",
                "Circular polar rose plots and Rayleigh uniformity test distributions for physiology vs. sensor wear.",
            ),
            (
                REPO_ROOT / "experiments" / "female_infradian_extensions" / "figures" / "fig3_bmi_amplitude_and_period_violins.png",
                "Infradian amplitude and period distributions across WHO BMI strata (Underweight, Normal, Overweight, Obese).",
            ),
        ],
    ),
    ReportConfig(
        report_id="EXP-04-PREDICTIVE-UTILITY-BENCHMARKS",
        provenance_category="04_predictive_utility_benchmarks",
        category_label="04 Predictive Utility Benchmarks",
        source_md=REPO_ROOT / "experiments" / "monthly_periodicity_downstream" / "results" / "REPORT.md",
        output_pdf_name="2026-10-20_monthly-periodicity-downstream_predictive-utility-benchmark.pdf",
        date="2026-10-20",
        title="Downstream Predictive Utility of Infradian Rhythm Representations for Recorded Sex Classification",
        cohort="Apple Watch Allocation 0 Held-out Test Set (N_test = 1,671; N_train = 7,797)",
        target_estimand="Test AUROC gain from augmenting diurnal/circadian baselines with 11 infradian features",
        epistemic_classification="Predictive Utility: Translation of biological rhythmicity into machine learning performance on unseen data",
        validation_protocol="Strict Allocation 0 test evaluation; 2,000 paired stratified bootstrap replicates for CIs and p-values",
        key_findings_summary=(
            "Infradian features yield statistically significant AUROC gains: +0.0109 on clock baseline (p < 0.001) "
            "and +0.0061 on linear baseline (p < 0.001). Subgroup AUROC in young females reaches 0.9071."
        ),
        figures=[
            (
                REPO_ROOT / "experiments" / "monthly_periodicity_downstream" / "figures" / "fig1_downstream_auroc_forest.png",
                "Forest plot of test AUROC and 95% bootstrap confidence intervals across model arms.",
            ),
            (
                REPO_ROOT / "experiments" / "monthly_periodicity_downstream" / "figures" / "fig2_young_female_auroc_lift.png",
                "Age-stratified downstream classification AUROC highlighting the predictive lift in young females.",
            ),
        ],
    ),
    ReportConfig(
        report_id="EXP-05A-IMPUTATION-ARCHITECTURE",
        provenance_category="05_imputation_and_regularization",
        category_label="05 Imputation and Regularization",
        source_md=REPO_ROOT / "experiments" / "apple_fill_and_mask_by_recording" / "results" / "REPORT.md",
        output_pdf_name="2026-10-20_apple-fill-and-mask-by-recording_imputation-architecture.pdf",
        date="2026-10-20",
        title="Missing-Data Imputation and Observation-Mask Architecture Sensitivity Across Apple Watch Allocations",
        cohort="Apple Watch Allocations 0, 1, 2 (N = 11,139 total across 10-fold allocations)",
        target_estimand="8-arm fill and mask ablation: H_clock vs H_global, short-gap interpolation, clean diff, mask concat/stack",
        epistemic_classification="Architecture Sensitivity: Systematic sensitivity analysis of missing-data handling across recording patterns",
        validation_protocol="MultiRocket feature extraction + Ridge regression across 3 independent holdout allocations; Continuity x Occupancy x History subgroups",
        key_findings_summary=(
            "User-specific clock median filling (H_clock) dominates global median across all allocations and subgroups. "
            "Short-gap interpolation provides marginal benefits in dense records; explicit mask concatenation improves low-occupancy stability."
        ),
        figures=[
            (
                REPO_ROOT / "experiments" / "apple_fill_and_mask_by_recording" / "figures" / "fig1_imputation_arms_comparison.png",
                "Comparison of held-out test AUROC across missing-data imputation and mask representation arms.",
            ),
        ],
    ),
    ReportConfig(
        report_id="EXP-05B-CIRCULAR-DIURNAL-REGULARIZATION",
        provenance_category="05_imputation_and_regularization",
        category_label="05 Imputation and Regularization",
        source_md=REPO_ROOT / "experiments" / "apple_circular_diurnal_regularization" / "results" / "REPORT.md",
        output_pdf_name="2026-10-20_apple-circular-diurnal-regularization_template-regularization.pdf",
        date="2026-10-20",
        title="Circular Diurnal Template Regularization: Reference Specification & Empirical Benchmark",
        cohort="200 Stratified Test Participants (11,900 Masked Gaps) & Allocation 0 Test Cohort (N = 1,671)",
        target_estimand="1D Circular Gaussian smoothing across midnight (sigma in {1,2,4}); Masked MAE, RMSE, Boundary Jump, and Downstream AUROC",
        epistemic_classification="Template Regularization: Mathematical derivation, unit verification, and empirical gap-filling optimization",
        validation_protocol="Masked imputation benchmark (59,500 evaluations); Allocation 0 Ridge regression benchmark; 1,000 paired bootstrap replicates",
        key_findings_summary=(
            "Optimal regularizer is sigma* = 1 bin (5 min), minimizing RMSE (9.628 bpm) and boundary jump (8.637 bpm). "
            "Downstream AUROC improves in low-occupancy participants (+0.0028 template-only, +0.0018 combined), buffering against sparse diurnal noise."
        ),
        figures=[
            (
                REPO_ROOT / "experiments" / "apple_circular_diurnal_regularization" / "figures" / "fig1_diurnal_template_smoothing.png",
                "Raw median diurnal profile vs. circular Gaussian smoothed templates with midnight wrap-around inset.",
            ),
            (
                REPO_ROOT / "experiments" / "apple_circular_diurnal_regularization" / "figures" / "fig2_imputation_error_curves.png",
                "Imputation reconstruction error (MAE, RMSE) and boundary step discontinuity curves across gap lengths.",
            ),
        ],
    ),
    ReportConfig(
        report_id="EXP-06A-CROSS-COHORT-TRANSFER",
        provenance_category="06_cross_cohort_and_scaling",
        category_label="06 Cross-Cohort and Scaling",
        source_md=REPO_ROOT / "experiments" / "apple_garmin_transfer_pilot" / "REPORT.md",
        output_pdf_name="2026-10-19_apple-garmin-transfer-pilot_device-generalization.pdf",
        date="2026-10-19",
        title="Apple Watch to Garmin Cross-Cohort Transferability Pilot Benchmark",
        cohort="Garmin Pilot (N = 1,000) and Apple Watch Pilot (N = 1,000)",
        target_estimand="Cross-device domain transfer performance of diurnal and circadian representation features",
        epistemic_classification="Cross-Cohort Generalization: External validity assessment across heterogeneous sensor ecosystems",
        validation_protocol="Train-on-Apple, test-on-Garmin (and reciprocal); out-of-domain AUROC and domain adaptation diagnostics",
        key_findings_summary=(
            "Diurnal median templates exhibit robust cross-device domain transfer (AUROC > 0.65 across devices), "
            "whereas high-frequency temporal autocorrelation features show sensor-specific recording calibration bias."
        ),
        figures=[],
    ),
    ReportConfig(
        report_id="EXP-06B-COHORT-SCALING",
        provenance_category="06_cross_cohort_and_scaling",
        category_label="06 Cross-Cohort and Scaling",
        source_md=REPO_ROOT / "experiments" / "apple_cohort_scaling" / "REPORT.md",
        output_pdf_name="2026-10-14_apple-cohort-scaling_full-cohort-benchmark.pdf",
        date="2026-10-14",
        title="Full Single-Source Apple Watch Cohort Scaling Benchmark Report",
        cohort="Qualified Apple Watch Participants with >= 14 Adequate Days (N = 11,139 across 10 repeated holdouts)",
        target_estimand="MultiRocket sequence classification scaling, history budget allocation (40-day vs unconstrained), and 5-min vs 15-min resolution",
        epistemic_classification="Cohort Scaling: Large-sample benchmark scaling and computational feasibility reference",
        validation_protocol="10 repeated OR-Tools balanced 70/15/15 holdouts (N_train = 7,797, N_val = 1,671, N_test = 1,671)",
        key_findings_summary=(
            "5-minute resolution consistently outperforms 15-minute resolution across all allocations. "
            "Unconstrained history achieves test AUROC 0.732 +/- 0.008. MultiRocket sequence kernels demonstrate stable asymptotic scaling."
        ),
        figures=[
            (
                REPO_ROOT / "experiments" / "apple_cohort_scaling" / "figures" / "fig1_resolution_and_history_scaling.png",
                "Apple Watch cohort classification AUROC scaling across resolution (5m vs 15m) and history budgets.",
            ),
        ],
    ),
    ReportConfig(
        report_id="EXP-06C-GARMIN-INFRADIAN-REPLICATION",
        provenance_category="06_cross_cohort_and_scaling",
        category_label="06 Cross-Cohort and Scaling",
        source_md=REPO_ROOT / "experiments" / "garmin_infradian_replication" / "REPORT.md",
        output_pdf_name="2026-10-20_garmin-infradian-replication_cross-device-and-spectral.pdf",
        date="2026-10-20",
        title="Cross-Device Garmin Replication, Sample Scaling, and Spectral Diagnostics of Infradian Heart-Rate Cycles",
        cohort="Garmin Tier 1c Single-Source (N = 3,760 qualified) & Apple Watch Extended (N = 5,516 qualified)",
        target_estimand="Cross-device external validity of 25–35d excess infradian power, period persistence, and 3–120d spectral diagnostics",
        epistemic_classification="Cross-Cohort Generalization & Spectral Diagnostics: Independent device replication and long-period red-noise resolution",
        validation_protocol="Independent Garmin cohort evaluation; B=199 two-component null bootstrap; 3–120d harmonic sweeps; nonoverlapping second-window persistence",
        key_findings_summary=(
            "Garmin cohort independently confirms young female infradian excess (contrast +0.0474, Wald chi^2 = 81.1, p < 1e-17). "
            "Spectral diagnostics (3–120d) resolve long-period male rise as red-noise continuum and polynomial drift rather than harmonic resonance."
        ),
        figures=[
            (
                REPO_ROOT / "experiments" / "garmin_infradian_replication" / "figures" / "R1_forest_contrasts.png",
                "Forest plot of HC3 Wald excess score contrasts (E_i) across Apple k=15, Apple k=10, and Garmin k=15.",
            ),
            (
                REPO_ROOT / "experiments" / "garmin_infradian_replication" / "figures" / "R2_rejection_rates.png",
                "Empirical false positive and rejection rates at alpha = 0.05 across age and sex cells under the two-component null.",
            ),
            (
                REPO_ROOT / "experiments" / "garmin_infradian_replication" / "figures" / "R3_pvalue_qq.png",
                "-log10 p Quantile-Quantile plots against uniform expectation confirming male calibration and female deflection.",
            ),
            (
                REPO_ROOT / "experiments" / "garmin_infradian_replication" / "figures" / "R4_excess_score_distributions.png",
                "Violin distributions of individual participant excess infradian score E_i across cohorts and age brackets.",
            ),
            (
                REPO_ROOT / "experiments" / "garmin_infradian_replication" / "figures" / "R5_persistence.png",
                "Second-window period agreement |T_1 - T_2| ECDFs against permuted nulls, and frozen out-of-sample phase correlation.",
            ),
            (
                REPO_ROOT / "experiments" / "garmin_infradian_replication" / "figures" / "R6_garmin_mean_periodogram.png",
                "Full continuous spectral periodograms (3 to 60 days) on Garmin cohort with participant-level AR(1) surrogate envelopes.",
            ),
            (
                REPO_ROOT / "experiments" / "garmin_infradian_replication" / "figures" / "R7_long_period_male_rise.png",
                "Diagnostic 3 to 120 day spectral decomposition resolving male long-period power as red noise and polynomial drift.",
            ),
        ],
    ),
    ReportConfig(
        report_id="EXP-03B-INFRADIAN-AGE-DEPENDENCY",
        provenance_category="03_confirmatory_mechanisms",
        category_label="03 Confirmatory Mechanisms",
        source_md=REPO_ROOT / "experiments" / "infradian_age_dependency" / "REPORT.md",
        output_pdf_name="2026-10-20_infradian-age-dependency_rcs-age-trajectory.pdf",
        date="2026-10-20",
        title="Continuous-Age Dependence of Monthly Resting Heart-Rate Rhythmicity: Restricted Cubic Spline Sex-by-Age Analysis",
        cohort="Apple Watch (expanded90 N = 5,162) and Garmin Tier 1c (expanded90 N = 3,754); Samsung non-estimable (N = 8)",
        target_estimand="Female-minus-male contrast in band-integrated 25–35d excess power as a continuous function of age (Harrell RCS, knots 25/42/52/65)",
        epistemic_classification="Confirmatory Mechanism: Pre-specified RCS model with stratified bootstrap inference; cross-sectional, associational",
        validation_protocol="Stone-Koo RCS, chunk Wald tests (HC3), 2,000 stratified participant bootstraps, simultaneous bands, two-component REML nulls, second-window repeatability",
        key_findings_summary=(
            "Steepest decline of the female-male contrast at 46.65y (Apple, CI 45.40-47.40) and 47.25y (Garmin, CI 44.45-48.50). "
            "Nominal zero-crossing 56.18y (Apple) and 70.20y (Garmin); simultaneous-band cessation not reached for Q_i. "
            "Second-window repeatability present in females <55y, absent in females >=55y and males."
        ),
        figures=[
            (
                REPO_ROOT / "experiments" / "infradian_age_dependency" / "figures" / "fig1_rcs_continuous_partial_effects.png",
                "Continuous RCS curves by sex with pointwise 95% CIs and simultaneous bands.",
            ),
            (
                REPO_ROOT / "experiments" / "infradian_age_dependency" / "figures" / "fig2_excess_heatmaps_and_fwhm.png",
                "Female age-by-period excess heatmaps, spectral slices and FWHM.",
            ),
            (
                REPO_ROOT / "experiments" / "infradian_age_dependency" / "figures" / "fig3_delta_curve_transitions.png",
                "Female-minus-male delta curves with transition landmarks.",
            ),
            (
                REPO_ROOT / "experiments" / "infradian_age_dependency" / "figures" / "fig4_source_and_protocol_comparisons.png",
                "Cross-device, protocol and second-window repeatability comparisons.",
            ),
        ],
    ),
]


def render_report_to_pdf(config: ReportConfig) -> dict[str, Any]:
    """Compile a single report from Markdown to Typst and then to PDF."""
    log.info(f"Rendering {config.report_id} -> {config.output_pdf_name}...")
    if not config.source_md.exists():
        raise FileNotFoundError(f"Source markdown not found: {config.source_md}")

    # Read markdown source
    md_content = config.source_md.read_text(encoding="utf-8")

    # If first line is a title (# ...), remove it so we don't duplicate the title banner
    lines = md_content.splitlines()
    if lines and lines[0].startswith("# "):
        md_content = "\n".join(lines[1:])

    # Convert Markdown to Typst syntax using pandoc
    typst_body = pypandoc.convert_text(md_content, to="typst", format="markdown")

    # Build Figures Typst Section
    figures_typst = ""
    if config.figures:
        fig_blocks = []
        for i, (fig_path, caption) in enumerate(config.figures, start=1):
            if fig_path.exists():
                fig_blocks.append(f"""
#pagebreak()
#align(center)[
  #block(stroke: 0.5pt + rgb("d0d7de"), inset: 2pt, radius: 2pt)[
    #image("{fig_path.resolve()}", width: 95%)
  ]
  #v(5pt)
  #text(size: 8.5pt, fill: rgb("333333"))[*Figure {i}:* {caption}]
]
""")
        if fig_blocks:
            figures_typst = "\n#pagebreak()\n= Publication Figures & Empirical Visualizations\n" + "".join(fig_blocks)

    # Escape json strings for typst string literals
    def to_typst_str(val: str) -> str:
        # JSON dump returns a valid quoted string with escaped quotes and backslashes
        return json.dumps(val)

    # Wrap in complete Typst document with lab report styles
    typst_full_doc = f"""
#set page(
  paper: "a4",
  margin: (top: 2.2cm, bottom: 2.0cm, left: 1.8cm, right: 1.8cm),
  header: [
    #grid(
      columns: (1fr, 1fr),
      align(left)[#text(size: 7.5pt, weight: "bold", fill: rgb("0b3c5d"))[{config.category_label.upper()}]],
      align(right)[#text(size: 7.5pt, fill: rgb("555555"))[{config.report_id} | {config.date}]]
    )
    #v(-4pt)
    #line(length: 100%, stroke: 0.5pt + rgb("d0d7de"))
  ],
  footer: [
    #line(length: 100%, stroke: 0.5pt + rgb("d0d7de"))
    #v(2pt)
    #grid(
      columns: (1fr, 1fr),
      align(left)[#text(size: 7.5pt, fill: rgb("666666"))[Robert Koch Institute Corona Data Donation — Reference Laboratory Report]],
      align(right)[#text(size: 7.5pt, fill: rgb("666666"))[Page #context counter(page).display("1 of 1", both: true)]]
    )
  ]
)
#set text(font: "Helvetica", size: 8.5pt, lang: "en")
#set par(justify: true, leading: 0.65em)

#let horizontalrule = line(length: 100%, stroke: 0.5pt + rgb("d0d7de"))

#show heading.where(level: 1): it => block(below: 10pt, above: 14pt)[
  #text(fill: rgb("0b3c5d"), weight: "bold", size: 13pt)[#it.body]
  #v(2pt)
  #line(length: 100%, stroke: 1pt + rgb("0b3c5d"))
]
#show heading.where(level: 2): it => block(below: 8pt, above: 11pt)[
  #text(fill: rgb("1d2731"), weight: "bold", size: 10.5pt)[#it.body]
  #v(2pt)
  #line(length: 100%, stroke: 0.5pt + rgb("328cc1"))
]
#show heading.where(level: 3): it => block(below: 6pt, above: 9pt)[
  #text(fill: rgb("328cc1"), weight: "bold", size: 9.5pt)[#it.body]
]

#let doc_title = {to_typst_str(config.title)}
#let doc_date = {to_typst_str(config.date)}
#let doc_cohort = {to_typst_str(config.cohort)}
#let doc_estimand = {to_typst_str(config.target_estimand)}
#let doc_role = {to_typst_str(config.epistemic_classification)}
#let doc_val = {to_typst_str(config.validation_protocol)}
#let doc_findings = {to_typst_str(config.key_findings_summary)}

#rect(
  fill: rgb("f6f8fa"),
  stroke: (left: 4pt + rgb("0b3c5d")),
  inset: 10pt,
  radius: (right: 4pt),
  width: 100%,
)[
  #text(size: 8pt, weight: "bold", fill: rgb("0b3c5d"))[SCIENTIFIC PROVENANCE: {config.category_label.upper()}] \
  #v(2pt)
  #text(size: 14pt, weight: "bold", fill: rgb("0b3c5d"))[#doc_title]
]

#v(8pt)

#table(
  columns: (22%, 28%, 22%, 28%),
  stroke: 0.5pt + rgb("d0d7de"),
  fill: (col, row) => if calc.even(col) {{ rgb("f8f9fa") }} else {{ rgb("ffffff") }},
  inset: 5pt,
  [*Study Date:*], doc_date, [*Target Estimand:*], doc_estimand,
  [*Cohort:*], doc_cohort, [*Epistemic Role:*], doc_role,
  [*Validation:*], table.cell(colspan: 3)[#doc_val],
  [*Key Findings:*], table.cell(colspan: 3)[#doc_findings]
)

#v(8pt)

{typst_body}

{figures_typst}
"""

    # Write temporary Typst file
    temp_typ_path = TEMP_TYP_DIR / f"{config.report_id}.typ"
    temp_typ_path.write_text(typst_full_doc, encoding="utf-8")

    # Output path
    category_dir = OUTPUT_BASE / config.provenance_category
    category_dir.mkdir(parents=True, exist_ok=True)
    out_pdf_path = category_dir / config.output_pdf_name

    # Compile with Typst
    typst.compile(str(temp_typ_path), output=str(out_pdf_path), root=Path("/"))

    # Inspect generated PDF with pypdf
    file_size_bytes = out_pdf_path.stat().st_size
    pdf_reader = PdfReader(str(out_pdf_path))
    num_pages = len(pdf_reader.pages)

    # Count images
    total_imgs = sum(len(p.images) for p in pdf_reader.pages)

    log.info(f"Successfully generated {out_pdf_path.name} ({num_pages} pages, {total_imgs} figures, {file_size_bytes / 1024:.1f} KB)")

    return {
        "report_id": config.report_id,
        "provenance_category": config.provenance_category,
        "category_label": config.category_label,
        "date": config.date,
        "title": config.title,
        "source_md": str(config.source_md.relative_to(REPO_ROOT)),
        "output_pdf": str(out_pdf_path.relative_to(REPO_ROOT)),
        "absolute_path": str(out_pdf_path),
        "file_size_bytes": file_size_bytes,
        "page_count": num_pages,
        "cohort": config.cohort,
        "target_estimand": config.target_estimand,
        "epistemic_classification": config.epistemic_classification,
        "validation_protocol": config.validation_protocol,
        "key_findings": config.key_findings_summary,
        "figures_count": total_imgs,
    }


def generate_catalog_markdown(manifest: list[dict[str, Any]]) -> str:
    """Generate Diátaxis technical reference catalog of all rendered PDFs."""
    lines = [
        "# Scientific Reports Catalog: Experimental Provenance & PDF Archive",
        "",
        "## 1. Overview & Classification Schema",
        "",
        "This catalog indexes all recent experimental investigations in the Robert Koch Institute Corona Data Donation (`datenspende_IMPUTE_sex`) project. All reports are rendered into publication-grade, self-contained PDF laboratory reference documents without headless Chrome.",
        "",
        "### Epistemic Provenance Categories",
        "",
        "1. **`01_exploratory_discovery`**: Unstratified population screening, observational hypothesis generation, and signal discovery.",
        "2. **`02_calibrated_hypothesis_testing`**: Rigorous hypothesis testing under standardized exposure with per-subject serially correlated null models (continuous AR(1) surrogate bootstrap) and multiplicity control.",
        "3. **`03_confirmatory_mechanisms`**: Mechanistic validation, longitudinal temporal stability across extended observation (≥365 days), phase-locking uniformity tests (Rayleigh), and biological covariate invariance (WHO BMI strata).",
        "4. **`04_predictive_utility_benchmarks`**: Translation of biological rhythmicity into machine learning performance on strictly held-out test partitions.",
        "5. **`05_imputation_and_regularization`**: Mathematical formulation, unit verification, and sensitivity benchmarking of missing-data reconstruction architectures.",
        "6. **`06_cross_cohort_and_scaling`**: Cross-device transferability (Apple Watch vs. Garmin) and asymptotic cohort scaling benchmarks.",
        "",
        "---",
        "",
        "## 2. Complete PDF Catalog & Master Manifest",
        "",
        "| Category | Date | Report Title | Target Estimand | Pages | Figures | Size | PDF Document |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for item in manifest:
        pdf_rel = item["output_pdf"]
        pdf_name = Path(pdf_rel).name
        size_kb = item["file_size_bytes"] / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
        lines.append(
            f"| **{item['category_label']}** | `{item['date']}` | {item['title']} | {item['target_estimand']} | "
            f"{item['page_count']} | {item['figures_count']} | {size_str} | [{pdf_name}](file://{item['absolute_path']}) |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Detailed Experimental Summaries by Category",
        "",
    ])

    # Group by category
    categories = sorted(list({item["provenance_category"] for item in manifest}))
    for cat in categories:
        cat_items = [item for item in manifest if item["provenance_category"] == cat]
        cat_label = cat_items[0]["category_label"]
        lines.append(f"### Category: {cat_label}")
        lines.append("")
        for item in cat_items:
            lines.extend([
                f"#### [{item['report_id']}] {item['title']}",
                "",
                f"- **PDF Artifact**: [`{Path(item['output_pdf']).name}`](file://{item['absolute_path']}) ({item['page_count']} pages, {item['figures_count']} figures, {item['file_size_bytes'] / 1024:.1f} KB)",
                f"- **Source Markdown**: [`{item['source_md']}`](file://{REPO_ROOT / item['source_md']})",
                f"- **Date**: `{item['date']}`",
                f"- **Cohort**: {item['cohort']}",
                f"- **Target Estimand**: {item['target_estimand']}",
                f"- **Epistemic Classification**: {item['epistemic_classification']}",
                f"- **Validation Protocol**: {item['validation_protocol']}",
                f"- **Key Empirical Findings**: {item['key_findings']}",
                "",
            ])

    lines.extend([
        "---",
        "",
        "## 4. Technical Specifications & Reproducibility",
        "",
        "- **PDF Engine**: Pure Python `pypandoc_binary` (Pandoc AST math/table converter) + `typst` (Rust-native typesetting engine). Zero headless Chrome, zero Chromium/Puppeteer.",
        "- **Mathematical Typography**: Native vector typesetting of all LaTeX math (fractions, roots, sums, integrals, matrices, piecewise cases, Greek letters, accents).",
        "- **Visualizations**: High-resolution 300 DPI publication figures embedded with dedicated page allocation and structured captions.",
        "- **Geometry**: A4 portrait, 1.8cm margins, running headers with epistemic classification tag, running footers with exact `Page X of Y` counters.",
        "- **Builder Script**: [`reports/build_scientific_reports_pdf.py`](file://" + str(REPO_ROOT / "reports" / "build_scientific_reports_pdf.py") + ")",
        "- **Manifest Output**: [`reports/scientific_provenance_pdfs/manifest.json`](file://" + str(OUTPUT_BASE / "manifest.json") + ")",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    log.info(f"Starting compilation of {len(REPORTS_CONFIG)} scientific reports to PDF via Pandoc + Typst...")

    manifest: list[dict[str, Any]] = []

    for cfg in REPORTS_CONFIG:
        res = render_report_to_pdf(cfg)
        manifest.append(res)

    # Save manifest.json
    manifest_path = OUTPUT_BASE / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "reports_count": len(manifest),
                "manifest": manifest,
            },
            f,
            indent=2,
        )
    log.info(f"Saved manifest to {manifest_path}")

    # Save CATALOG.md
    catalog_md = generate_catalog_markdown(manifest)
    catalog_path = OUTPUT_BASE / "CATALOG.md"
    catalog_path.write_text(catalog_md, encoding="utf-8")
    log.info(f"Saved catalog to {catalog_path}")

    print("\n" + "=" * 80)
    print("SCIENTIFIC REPORTS PDF COMPILATION COMPLETE")
    print("=" * 80)
    for m in manifest:
        print(f"[{m['provenance_category']}] {m['output_pdf']} ({m['page_count']} pages, {m['figures_count']} figs, {m['file_size_bytes'] / 1024:.1f} KB)")
    print("=" * 80)
    print(f"Catalog: {catalog_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
