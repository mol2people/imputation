"""Verification script for Milestone 4 deliverables."""
from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"

# 1. Verify Figure 1
fig1_path = FIGURES_DIR / "fig1_diurnal_template_smoothing.png"
assert fig1_path.exists(), f"Missing Figure 1: {fig1_path}"
assert fig1_path.stat().st_size > 100_000, f"Figure 1 file size too small: {fig1_path.stat().st_size} bytes"
print(f"[PASS] Figure 1 verified: {fig1_path} ({fig1_path.stat().st_size / 1024:.1f} KB)")

# 2. Verify Figure 2
fig2_path = FIGURES_DIR / "fig2_imputation_error_curves.png"
assert fig2_path.exists(), f"Missing Figure 2: {fig2_path}"
assert fig2_path.stat().st_size > 100_000, f"Figure 2 file size too small: {fig2_path.stat().st_size} bytes"
print(f"[PASS] Figure 2 verified: {fig2_path} ({fig2_path.stat().st_size / 1024:.1f} KB)")

# 3. Verify REPORT.md
report_path = RESULTS_DIR / "REPORT.md"
assert report_path.exists(), f"Missing REPORT.md: {report_path}"
report_text = report_path.read_text(encoding="utf-8")
assert len(report_text) > 5000, f"REPORT.md too short: {len(report_text)} chars"

required_sections = [
    "1. Executive Summary & Lead Findings",
    "2. Mathematical Specification & Operators",
    "3. R1 Unit Verification Fixture Summary",
    "4. R2 Masked Imputation Fidelity Benchmark",
    "5. R3 Downstream Classification Benchmark",
    "6. Practical Recommendations & Clinical Integration",
    "7. Validity Conditions, Caveats, and Assumptions",
]

for sec in required_sections:
    assert sec in report_text, f"Missing required section in REPORT.md: {sec}"
    print(f"  [PASS] Found section: {sec}")

# Verify key empirical metrics present in text
key_strings = [
    "sigma^* = 1.0",
    "8.637",  # boundary jump for s1
    "9.628",  # rmse for s1
    "0.6787", # baseline test auroc
    "0.6791", # s1 test auroc
    "+0.000395", # delta auroc
    "5.14",   # test suite runtime
]
for ks in key_strings:
    assert ks in report_text, f"Missing key metric string in REPORT.md: {ks}"
    print(f"  [PASS] Verified metric presence: {ks}")

print("\nALL MILESTONE 4 DELIVERABLES VERIFIED SUCCESSFULLY!")
