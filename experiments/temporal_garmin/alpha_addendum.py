#!/usr/bin/env python
"""TEMPORAL Step 0.7 — α extended-grid addendum.

Predeclared in NEXT_STEPS.md §Step 0.7 (2026-09-22) BEFORE this script ran:
rebuild alloc-0 arm matrices exactly as the v2 run (fit_alloc(0): same seeds,
same mask-aware fill, same train-only hygiene), sweep α over the predeclared
extended grid on VALIDATION only, report val curves + alloc-0 test AUROC at
α* and at the frozen α.

Frozen decision rule: Combined val gain ≥ 0.01 over the frozen-α val AUROC
→ user decision on a v3 full rerun; below → record only (α boundary stays a
caveat, not a defect — same protocol across arms kept the ladder fair).

Usage:  python alpha_addendum.py
Output: results/alpha_addendum.json + results/alpha_addendum.md
"""
from __future__ import annotations

import json
import resource
import sys
import time

import run_temporal as rt  # sets thread env before numpy/numba/torch import

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

RESULTS = rt.RESULTS
# predeclared extended grid (NEXT_STEPS §Step 0.7)
GRID = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
FAMS = ("Summary_linear", "Profile24", "Profile288",
        "MultiRocket", "HYDRA", "Combined")


def log(msg):
    print(f"[addendum] {msg}", flush=True)


def main():
    t0 = time.perf_counter()
    log(f"start {time.strftime('%Y-%m-%dT%H:%M:%S')}")

    # frozen alpha per family, exactly as run (from the v2 metrics csv)
    mm = pd.read_csv(RESULTS / "temporal_metrics.csv")
    a0 = mm[mm.alloc == 0].set_index("arm")
    frozen = {fam: float(a0.alpha[fam]) for fam in FAMS}
    csv_val = {fam: float(a0.auroc_val[fam]) for fam in FAMS}
    csv_test = {fam: float(a0.auroc_test[fam]) for fam in FAMS}
    log(f"frozen alpha from metrics csv: {frozen}")

    # rebuild alloc-0 exactly as the v2 run
    Hydra, SparseScaler, mr_fit, mr_transform = rt.vendor_imports()
    users, y, d40, hr, mask, cnt, pu, pd_, folds, r1b = rt.load_data()
    static = rt.build_static(users, d40, mask, hr)
    hr_raw = hr.reshape(len(users), rt.DAYS, rt.N_BINS)
    mask_pu = mask.reshape(len(users), rt.DAYS, rt.N_BINS)
    log(f"loaded {len(users)} users; rebuilding alloc-0 arm matrices")

    result = rt.fit_alloc(0, users, y, folds, static, hr_raw, mask_pu,
                          mr_transform, SparseScaler)
    y_tr, y_va, y_te = result["y_tr"], result["y_va"], result["y_te"]
    log(f"alloc-0 rebuilt (train={len(y_tr)} val={len(y_va)} "
        f"test={len(y_te)}); sweeping {len(GRID)} alphas x {len(FAMS)} families")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "grid": GRID, "frozen_alpha": frozen,
           "decision_rule": ("Combined val gain >= 0.01 over frozen-alpha val "
                             "-> user decision on v3 full rerun; else record only"),
           "families": {}, "sanity": {}}

    for fam in FAMS:
        Ztr, Zva, Zte = result["arm_mats"][fam]
        trace = []
        best_a, best_auc = None, -1.0
        for a in GRID:
            m = Ridge(alpha=float(a), fit_intercept=True).fit(Ztr, y_tr)
            auc_v = float(roc_auc_score(y_va, m.predict(Zva)))
            auc_t = float(roc_auc_score(y_te, m.predict(Zte)))
            trace.append({"alpha": float(a), "auroc_val": auc_v,
                          "auroc_test": auc_t})
            if auc_v > best_auc:            # strict: smallest α wins ties
                best_auc, best_a = auc_v, float(a)
        fa = frozen[fam]
        row_fa = next(t for t in trace if t["alpha"] == fa)
        gain = best_auc - row_fa["auroc_val"]
        out["families"][fam] = {
            "alpha_star": best_a, "auroc_val_star": best_auc,
            "auroc_val_frozen": row_fa["auroc_val"],
            "auroc_test_frozen": row_fa["auroc_test"],
            "auroc_test_star": next(t for t in trace
                                     if t["alpha"] == best_a)["auroc_test"],
            "gain_val": gain, "trace": trace,
        }
        # sanity vs v2 metrics csv (same matrices, same frozen alpha)
        out["sanity"][fam] = {
            "csv_auroc_val": csv_val[fam], "rebuild_auroc_val": row_fa["auroc_val"],
            "csv_auroc_test": csv_test[fam],
            "rebuild_auroc_test": row_fa["auroc_test"],
            "absdiff_val": abs(csv_val[fam] - row_fa["auroc_val"]),
            "absdiff_test": abs(csv_test[fam] - row_fa["auroc_test"]),
        }
        log(f"  {fam:15s} alpha*={best_a:>8.0f}  val {row_fa['auroc_val']:.4f}"
            f" -> {best_auc:.4f} ({gain:+.4f})  "
            f"test {row_fa['auroc_test']:.4f} -> "
            f"{out['families'][fam]['auroc_test_star']:.4f}")

    combined_gain = out["families"]["Combined"]["gain_val"]
    out["combined_gain_val"] = combined_gain
    out["verdict"] = ("USER DECISION: v3 full rerun (Combined val gain "
                      f"{combined_gain:+.4f} >= 0.01)"
                      if combined_gain >= 0.01 else
                      f"RECORD ONLY (Combined val gain {combined_gain:+.4f} < 0.01)")
    # darwin getrusage ru_maxrss is in BYTES (linux: KiB) — convert to MiB
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024
    out["peak_rss_mb"] = round(peak_mib, 1)
    out["wall_s"] = round(time.perf_counter() - t0, 1)

    json.dump(out, open(RESULTS / "alpha_addendum.json", "w"), indent=2)

    # md table
    L = ["### α extended-grid addendum (alloc 0; predeclared NEXT_STEPS §Step 0.7)", "",
         "Grid `[1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]`, selection on validation "
         "only (strict max, smallest α wins ties, as in precalibration); test at α* "
         "reported for completeness (exploratory).", "",
         "| family | frozen α | val @ frozen | α* | val @ α* | gain | test @ frozen | test @ α* |",
         "|---|---|---|---|---|---|---|---|"]
    for fam in FAMS:
        f = out["families"][fam]
        L.append(f"| {fam} | {frozen[fam]:.0f} | {f['auroc_val_frozen']:.4f} | "
                 f"{f['alpha_star']:.0f} | {f['auroc_val_star']:.4f} | "
                 f"{f['gain_val']:+.4f} | {f['auroc_test_frozen']:.4f} | "
                 f"{f['auroc_test_star']:.4f} |")
    L += ["", f"**Verdict: {out['verdict']}**", "",
          f"Rebuild sanity vs v2 metrics csv (alloc 0, frozen α): max |Δval| = "
          f"{max(s['absdiff_val'] for s in out['sanity'].values()):.1e}, max |Δtest| = "
          f"{max(s['absdiff_test'] for s in out['sanity'].values()):.1e} "
          "(≤1e-6 gate; expected 1–2 ULP from threaded-BLAS ridge).",
          f"Wall {out['wall_s']} s; peak RSS {out['peak_rss_mb']} MiB."]
    open(RESULTS / "alpha_addendum.md", "w").write("\n".join(L) + "\n")

    log(f"verdict: {out['verdict']}")
    log(f"done in {out['wall_s']} s; peak RSS {out['peak_rss_mb']} MiB — "
        f"results/alpha_addendum.{{json,md}}")


if __name__ == "__main__":
    main()
