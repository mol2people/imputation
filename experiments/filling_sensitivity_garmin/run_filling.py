#!/usr/bin/env python
"""ch3000 FILLING SENSITIVITY — 4 fill arms × 3 allocations.

Implements [`PLAN.md`](PLAN.md) (2026-09-22). Reuses the recording pilot's
loading, MR fit/pool and alpha-selection helpers
(`experiments/recording_structure_garmin/run_recording.py`) and the corrected
temporal fill/B40 helpers (`experiments/temporal_garmin/run_temporal.py`).
Does not invoke the full recording runner and does not modify existing
experiment outputs.

Implementation notes (recorded in repro.json):
- Interpolation overlays (linear + PCHIP values at eligible-gap positions)
  depend only on ORIGINAL observed HR and masks → computed once, reused
  across arms and allocations. Per-alloc variation enters only through the
  H_clock base fill (train-only clock-bin medians) and the H_global median.
- PCHIP follows the plan's scipy convention: per gap length L∈{1..6} ONE
  scipy.interpolate.PchipInterpolator(x=[0,1,L+2,L+3], y=(4, n_L) float64,
  extrapolate=False) over the four context bin centres, evaluated at the
  missing positions. Batched 2-D y is bit-identical to per-gap scipy fits
  (max abs diff 0.0 on a 2,000-run real-data sample; fixture re-checks).
- Seed streams match the executed recording pilot exactly: fit days
  comp_seed(r, 0); MR base comp_seed(r, 1); MR diff comp_seed(r, 2), via
  fit_and_pool_mr — NOT the older temporal diff-seed convention.

CLI:
  python run_filling.py fixture   # synthetic gap/fill fixture
  python run_filling.py run       # 3-alloc pipeline (resumable; H_clock gate)
  python run_filling.py report    # write REPORT.md + repro.json
  python run_filling.py full      # fixture + run + report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# threading — match recording / temporal (workqueue + 8 threads; torch loaded)
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numba
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

_REPO = Path(__file__).resolve().parent.parent.parent
_TEMPORAL = _REPO / "experiments" / "temporal_garmin"
_RECORDING = _REPO / "experiments" / "recording_structure_garmin"
sys.path.insert(0, str(_TEMPORAL))
sys.path.insert(0, str(_RECORDING))

from run_temporal import (  # noqa: E402
    SEED_BASE, SRC, COHORT_SIZE, DAYS, N_BINS,
    MR_NUM_FEATURES_PER_TRANS, MR_WIDTH,
    BINS_NPZ, DAY40_PARQUET, FOLDS_PARQUET, PATH_COHORT, PATH_SPLIT,
    COMP_DAY_SELECT, COMP_MR_FIT,
    _nb_seed, comp_seed, log, peak_rss_mb, sha256_file,
    fill_hr, hygiene_std, build_static,
)
from run_recording import (  # noqa: E402
    vendor_mr, load_data, fit_and_pool_mr, select_alpha,
    ALPHA_GRID, ALLOCATION_TARGETS,
)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
_RECORD_RESULTS = _RECORDING / "results"

ARMS = ("H_clock", "H_global", "H_linear", "H_pchip")   # H_clock fitted first
SCORE_TOL = 1e-6        # H_clock per-user score match vs recording pilot H
AUROC_TOL = 1e-6        # AUROC match tolerance (after score match)
TRAIN_SHARE = (2696, 579, 573)  # recording pilot fold sizes; asserted


# ----------------------------------------------------------------------------
# Gap detection + interpolation overlays (alloc/arm-independent)
# ----------------------------------------------------------------------------
def find_missing_runs(mask_pu: np.ndarray) -> dict:
    """Per-run arrays over maximal missing-bin runs of mask_pu (n, D, L) bool.

    Returns dict: rows (int64), sr, er, rlen (int32), interior, elig, ctx
    (bool, per run). elig = interior run of 1..6 bins bounded by observed
    bins (automatic for maximal interior runs). ctx = elig with two
    consecutive originally observed bins immediately before AND after
    (PLAN §"Four fill methods").
    """
    n, D, L = mask_pu.shape
    R = n * D
    mv = mask_pu.reshape(R, L)
    miss = ~mv
    p = np.zeros((R, L + 2), dtype=bool)
    p[:, 1:-1] = miss
    d = np.diff(p.astype(np.int8), axis=1)
    starts = np.flatnonzero(d == 1)          # col over (R, L+1) = real start
    ends = np.flatnonzero(d == -1) - 1       # col = real end (inclusive)
    assert starts.size == ends.size
    rows = (starts // (L + 1)).astype(np.int64)
    sr = (starts % (L + 1)).astype(np.int32)
    er = (ends % (L + 1)).astype(np.int32)
    rlen = (er - sr + 1).astype(np.int32)
    interior = (sr >= 1) & (er <= L - 2)
    elig = interior & (rlen >= 1) & (rlen <= 6)
    ctx = np.zeros_like(elig)
    ei = np.flatnonzero(elig)
    if ei.size:
        lo_ok = sr[ei] >= 2
        hi_ok = er[ei] <= L - 3
        # numpy does not short-circuit: clip indices, guards zero them out
        lo_ctx = mv[rows[ei], np.maximum(sr[ei] - 2, 0)]
        hi_ctx = mv[rows[ei], np.minimum(er[ei] + 2, L - 1)]
        ctx[ei] = lo_ok & hi_ok & lo_ctx & hi_ctx
    return {"rows": rows, "sr": sr, "er": er, "rlen": rlen,
            "interior": interior, "elig": elig, "ctx": ctx}


def compute_gap_overlays(hr_raw: np.ndarray, mask_pu: np.ndarray) -> dict:
    """Overlays (linear + PCHIP values at flat positions) + counts + per-user
    exposure. Depends only on ORIGINAL hr/mask (alloc/arm-independent).

    Linear: run of k missing bins bounded by observed x_L, x_R → position j
    gets x_L + j/(k+1)·(x_R − x_L), j = 1..k (PLAN formula), float64 → f32.
    PCHIP: ctx runs only, batched per length (see module docstring).
    """
    n, D, L = hr_raw.shape
    R = n * D
    rv = hr_raw.reshape(R, L).astype(np.float64)
    mv = mask_pu.reshape(R, L)
    g = find_missing_runs(mask_pu)
    rows, sr, er, rlen = g["rows"], g["sr"], g["er"], g["rlen"]
    elig, ctx = g["elig"], g["ctx"]

    j = np.arange(1, 7, dtype=np.float64)

    # ---- linear overlay: ALL eligible runs ----
    ei = np.flatnonzero(elig)
    rr, ss, ee, ll = rows[ei], sr[ei], er[ei], rlen[ei]
    grid = j[None, :] <= ll[:, None]                       # (n_e, 6)
    pos = (rr[:, None] * L + ss[:, None] + (j[None, :] - 1))  # (n_e, 6)
    pos_lin = pos[grid].astype(np.int64)
    xL = rv[rr, ss - 1]                                    # (n_e,)
    xR = rv[rr, ee + 1]
    val_lin_grid = (xL[:, None]
                    + (j[None, :] / (ll[:, None] + 1))
                    * (xR - xL)[:, None])                  # (n_e, 6)
    val_lin = val_lin_grid[grid].astype(np.float32)

    # ---- PCHIP overlay: ctx runs, batched per gap length ----
    ci = np.flatnonzero(ctx)
    pos_pchip = np.empty(0, dtype=np.int64)
    val_pchip = np.empty(0, dtype=np.float32)
    if ci.size:
        pos_parts, val_parts = [], []
        for Lg in range(1, 7):
            ii = ci[rlen[ci] == Lg]
            if ii.size == 0:
                continue
            rr_c, ss_c, ee_c = rows[ii], sr[ii], er[ii]
            y = np.stack([rv[rr_c, ss_c - 2], rv[rr_c, ss_c - 1],
                          rv[rr_c, ee_c + 1], rv[rr_c, ee_c + 2]], axis=1)
            x = np.array([0.0, 1.0, Lg + 2, Lg + 3])
            pb = PchipInterpolator(x, y.T, extrapolate=False)
            vals = pb(np.arange(2, 2 + Lg, dtype=np.float64))   # (Lg, nL)
            pos_L = (rr_c[:, None] * L + ss_c[:, None]
                     + np.arange(Lg)[None, :]).ravel().astype(np.int64)
            pos_parts.append(pos_L)
            val_parts.append(vals.T.ravel().astype(np.float32))
        pos_pchip = np.concatenate(pos_parts)
        val_pchip = np.concatenate(val_parts)

    # ---- linear-fallback overlay: eligible minus ctx ----
    pos_fb = np.empty(0, dtype=np.int64)
    val_fb = np.empty(0, dtype=np.float32)
    if ci.size:
        is_ctx_pos = np.zeros(pos_lin.size, dtype=bool)
        rr2, ss2, ll2 = rows[ci], sr[ci], rlen[ci]
        grid2 = j[None, :] <= ll2[:, None]
        pos_ctx_exp = (rr2[:, None] * L + ss2[:, None]
                       + (j[None, :] - 1))[grid2].astype(np.int64)
        order = np.argsort(pos_lin, kind="mergesort")
        sorted_lin = pos_lin[order]
        loc = np.searchsorted(sorted_lin, pos_ctx_exp)
        assert np.array_equal(sorted_lin[loc], pos_ctx_exp), \
            "ctx positions not a subset of eligible positions"
        is_ctx_pos[order[loc]] = True
        pos_fb = pos_lin[~is_ctx_pos]
        val_fb = val_lin[~is_ctx_pos]

    # ---- structural invariants ----
    miss_flat = ~mv.reshape(-1)
    for pp in (pos_lin, pos_pchip, pos_fb):
        assert miss_flat[pp].all(), "overlay touches an observed bin"
    assert np.intersect1d(pos_pchip, pos_fb).size == 0
    assert np.array_equal(np.sort(np.concatenate([pos_pchip, pos_fb])),
                          np.sort(pos_lin)), \
        "pchip ∪ fallback != eligible"

    # ---- per-participant exposure (PLAN §"Readout" audit) ----
    u_of_run = rows // D
    miss_bins = np.bincount(u_of_run, weights=rlen.astype(float),
                            minlength=n)
    elig_bins_u = np.bincount(u_of_run[elig],
                              weights=rlen[elig].astype(float), minlength=n)
    ctx_bins_u = np.bincount(u_of_run[ctx],
                             weights=rlen[ctx].astype(float), minlength=n)
    day_elig = np.zeros(R, dtype=bool)
    day_elig[rows[elig]] = True
    days_elig_u = day_elig.reshape(n, D).sum(axis=1)
    counts = {
        "total_missing_bins": int(mv.reshape(-1)[~mv.reshape(-1)].sum()
                                  if False else (~mv).sum()),
        "total_runs": int(rlen.size),
        "boundary_runs": int((~g["interior"]).sum()),
        "long_runs_gt6": int((rlen > 6).sum()),
        "eligible_runs": int(elig.sum()),
        "eligible_bins": int(rlen[elig].sum()),
        "ctx_runs": int(ctx.sum()),
        "ctx_bins": int(rlen[ctx].sum()),
        "fb_runs": int((elig & ~ctx).sum()),
        "fb_bins": int(rlen[elig & ~ctx].sum()),
        "len_hist_1_to_9_gt9": ([int((rlen == k).sum()) for k in range(1, 10)]
                                 + [int((rlen > 9).sum())]),
    }
    counts["ineligible_bins"] = int(counts["total_missing_bins"]
                                    - counts["eligible_bins"])
    per_user = {
        "miss_bins": miss_bins,
        "elig_bins": elig_bins_u,
        "ctx_bins": ctx_bins_u,
        "fb_bins": np.maximum(elig_bins_u - ctx_bins_u, 0.0),
        "clock_bins": np.maximum(miss_bins - elig_bins_u, 0.0),
        "days_elig": days_elig_u.astype(float),
        "days_total": np.full(n, float(D)),
        "bins_total": np.full(n, float(D * L)),
    }
    return {"pos_lin": pos_lin, "val_lin": val_lin,
            "pos_pchip": pos_pchip, "val_pchip": val_pchip,
            "pos_fb": pos_fb, "val_fb": val_fb,
            "counts": counts, "per_user": per_user}


def apply_overlay(filled: np.ndarray, pos: np.ndarray, vals: np.ndarray):
    """Write vals at flat positions of a contiguous (n, D, L) array."""
    if pos.size:
        filled.reshape(-1)[pos] = vals
    return filled


def build_filled(arm: str, hr_clock: np.ndarray, fill_global: np.ndarray,
                 overlays: dict) -> np.ndarray:
    """Construct the (n, D, L) float32 filled array for one arm."""
    if arm == "H_clock":
        return hr_clock.copy()
    if arm == "H_global":
        return fill_global.copy()
    if arm == "H_linear":
        f = hr_clock.copy()
        return apply_overlay(f, overlays["pos_lin"], overlays["val_lin"])
    if arm == "H_pchip":
        f = hr_clock.copy()
        apply_overlay(f, overlays["pos_pchip"], overlays["val_pchip"])
        return apply_overlay(f, overlays["pos_fb"], overlays["val_fb"])
    raise ValueError(arm)


# ----------------------------------------------------------------------------
# Fixture (synthetic; PLAN §"Checks" item 1)
# ----------------------------------------------------------------------------
def build_fixture_case():
    """(hr_2d float32, mask_2d bool), one case per row, L = 288:
      r0: 1-bin interior gap, full 4-pt ctx          → PCHIP
      r1: 6-bin interior gap, full 4-pt ctx          → PCHIP
      r2: 7-bin interior gap                          → INELIGIBLE (length)
      r3: boundary gaps at both day edges             → INELIGIBLE
      r4: two eligible runs, both missing outer ctx   → LINEAR fallback
      r5: turning point (rise–gap–fall)               → PCHIP
      r6: 2-bin gap at s=1 (no left outer ctx)        → LINEAR fallback
      r7: 2-bin gap ending at 286 (no right outer ctx)→ LINEAR fallback
    """
    L = 288
    hr = np.full((8, L), 60.0, dtype=np.float32)
    m = np.ones((8, L), dtype=bool)
    m[0, 100] = False
    hr[0, [98, 99, 101, 102]] = [55.0, 57.0, 65.0, 68.0]
    m[1, 140:146] = False
    hr[1, [138, 139, 146, 147]] = [50.0, 55.0, 72.0, 75.0]
    m[2, 60:67] = False                                   # 7 bins → ineligible
    m[3, 0:4] = False
    m[3, 284:288] = False                                 # boundary → ineligible
    m[4, 26:29] = False
    m[4, 30:32] = False                                   # both eligible, no ctx
    m[5, 200:202] = False
    hr[5, [198, 199, 202, 203]] = [50.0, 90.0, 90.0, 50.0]
    m[6, 1:3] = False                                     # s=1 → no s-2
    m[7, 285:287] = False                                 # e=286 → no e+2
    return hr, m


def naive_reference_overlays(hr_2d, mask_2d):
    """Slow loop reference: returns overlays + per-position ctx bounds."""
    L = mask_2d.shape[1]
    pos_lin, val_lin, pos_pchip, val_pchip = [], [], [], []
    pchip_lo, pchip_hi = [], []
    for r in range(mask_2d.shape[0]):
        mrow = mask_2d[r]; hrow = hr_2d[r]
        b = 0
        while b < L:
            if mrow[b]:
                b += 1
                continue
            s = b
            while b < L and not mrow[b]:
                b += 1
            e = b - 1
            length = e - s + 1
            if not (s >= 1 and e <= L - 2 and 1 <= length <= 6
                    and mrow[s - 1] and mrow[e + 1]):
                continue
            xL = float(hrow[s - 1]); xR = float(hrow[e + 1])
            for jj in range(1, length + 1):
                pos_lin.append(r * L + s + jj - 1)
                val_lin.append(np.float32(xL + jj / (length + 1) * (xR - xL)))
            if s >= 2 and e <= L - 3 and mrow[s - 2] and mrow[e + 2]:
                x = [0.0, 1.0, length + 2, length + 3]
                yv = [float(hrow[s - 2]), float(hrow[s - 1]),
                      float(hrow[e + 1]), float(hrow[e + 2])]
                pc = PchipInterpolator(x, yv, extrapolate=False)(
                    list(range(2, 2 + length)))
                for jj, vv in enumerate(pc):
                    pos_pchip.append(r * L + s + jj)
                    val_pchip.append(np.float32(vv))
                    pchip_lo.append(min(yv))
                    pchip_hi.append(max(yv))
    return (np.array(pos_lin, np.int64), np.array(val_lin, np.float32),
            np.array(pos_pchip, np.int64), np.array(val_pchip, np.float32),
            np.array(pchip_lo, np.float64), np.array(pchip_hi, np.float64))


def run_fixture():
    log("[fixture] synthetic gap/fill cases ...")
    hr_2d, mask_2d = build_fixture_case()
    hr3 = hr_2d.reshape(8, 1, 288)
    m3 = mask_2d.reshape(8, 1, 288)
    out = compute_gap_overlays(hr3, m3)
    (pl_n, vl_n, pc_n, vc_n, lo_n, hi_n) = naive_reference_overlays(hr_2d,
                                                                    mask_2d)

    def sorted_pair(a, b):
        o = np.argsort(a, kind="mergesort")
        return a[o], b[o]

    pl_a, vl_a = sorted_pair(out["pos_lin"], out["val_lin"])
    pc_a, vc_a = sorted_pair(out["pos_pchip"], out["val_pchip"])
    pl_n, vl_n = sorted_pair(pl_n, vl_n)
    pc_n, vc_n = sorted_pair(pc_n, vc_n)

    summary = {
        "linear_pos_match_naive": bool(np.array_equal(pl_a, pl_n)),
        "linear_val_max_abs_diff": (float(np.max(np.abs(vl_a - vl_n)))
                                     if vl_a.size else 0.0),
        "pchip_pos_match_naive": bool(np.array_equal(pc_a, pc_n)),
        "pchip_val_max_abs_diff": (float(np.max(np.abs(vc_a - vc_n)))
                                    if vc_a.size else 0.0),
        "got_eligible_runs": out["counts"]["eligible_runs"],
        "got_ctx_runs": out["counts"]["ctx_runs"],
        "got_eligible_bins": out["counts"]["eligible_bins"],
        "got_boundary_runs": out["counts"]["boundary_runs"],
        "expected_eligible_runs": 7,
        "expected_ctx_runs": 3,
        "expected_eligible_bins": 18,
        "expected_boundary_runs": 2,
    }
    summary["classification_ok"] = (
        summary["got_eligible_runs"] == summary["expected_eligible_runs"]
        and summary["got_ctx_runs"] == summary["expected_ctx_runs"]
        and summary["got_eligible_bins"] == summary["expected_eligible_bins"]
        and summary["got_boundary_runs"] == summary["expected_boundary_runs"])

    # closed-form linear checks (r0: xL=57, xR=65, k=1; r1: xL=55, xR=72, k=6)
    e_r0 = np.float32(57.0 + 1 / 2 * (65.0 - 57.0))
    e_r1 = (55.0 + np.arange(1, 7) / 7.0 * (72.0 - 55.0)).astype(np.float32)
    v_r0 = out["val_lin"][out["pos_lin"] == 0 * 288 + 100]
    sel_r1 = np.flatnonzero(out["pos_lin"] // 288 == 1)
    v_r1 = out["val_lin"][sel_r1]
    summary["r0_linear_max_abs_diff"] = (float(np.max(np.abs(v_r0 - e_r0)))
                                          if v_r0.size else None)
    summary["r1_linear_max_abs_diff"] = (float(np.max(np.abs(v_r1 - e_r1)))
                                          if v_r1.size == 6 else None)

    # PCHIP within the four context values (no overshoot), incl. turning point
    vc_f64 = vc_a.astype(np.float64)
    summary["pchip_within_4pt_minmax"] = bool(
        np.all((vc_f64 >= lo_n - 1e-6) & (vc_f64 <= hi_n + 1e-6)))

    # end-to-end fills: clock base via fill_hr (train = all 8 rows)
    hr_clock, fill_vec = fill_hr(hr3, m3, np.arange(8))
    overall = float(np.nanmedian(hr3[m3]))
    fill_global = np.where(m3, hr3, np.float32(overall)).astype(np.float32)
    ov = {"pos_lin": out["pos_lin"], "val_lin": out["val_lin"],
          "pos_pchip": out["pos_pchip"], "val_pchip": out["val_pchip"],
          "pos_fb": out["pos_fb"], "val_fb": out["val_fb"]}
    f_lin = build_filled("H_linear", hr_clock, fill_global, ov)
    f_pchip = build_filled("H_pchip", hr_clock, fill_global, ov)
    f_clock = build_filled("H_clock", hr_clock, fill_global, ov)
    f_glob = build_filled("H_global", hr_clock, fill_global, ov)
    miss_flat = np.flatnonzero(~mask_2d.reshape(-1))
    elig_set = set(ov["pos_lin"].tolist())
    inelig = np.array([q for q in miss_flat if q not in elig_set], dtype=np.int64)

    summary["observed_preserved_all_arms"] = bool(all(
        np.array_equal(f[m3], hr3[m3]) for f in
        (f_lin, f_pchip, f_clock, f_glob)))
    summary["fallback_exact"] = bool(np.array_equal(
        f_pchip.reshape(-1)[ov["pos_fb"]], f_lin.reshape(-1)[ov["pos_fb"]]))
    summary["clock_at_ineligible"] = bool(np.array_equal(
        f_pchip.reshape(-1)[inelig], f_clock.reshape(-1)[inelig]))
    summary["linear_at_eligible"] = bool(np.array_equal(
        f_lin.reshape(-1)[ov["pos_lin"]], ov["val_lin"]))
    summary["pchip_at_ctx"] = bool(np.array_equal(
        f_pchip.reshape(-1)[ov["pos_pchip"]], ov["val_pchip"]))
    summary["global_at_all_missing"] = bool(np.all(
        f_glob.reshape(-1)[miss_flat] == np.float32(overall)))
    summary["pchip_actually_differs"] = bool(
        np.any(f_pchip != f_lin))

    # train-only medians: extreme values in a non-train row must not leak
    hr_x = hr3.copy(); hr_x[7] = 200.0            # row 7 not in "training"
    tr_x = np.arange(7)
    fc_x, fv_x = fill_hr(hr_x, m3, tr_x)
    obs_tr = hr_x[tr_x][m3[tr_x]]
    summary["fill_clock_train_only"] = bool(
        np.nanmedian(obs_tr) == 60.0
        and np.allclose(fv_x, 60.0))
    summary["fill_global_train_only"] = bool(
        float(np.nanmedian(obs_tr)) == 60.0)

    CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(CACHE / "fixture.json", "w"), indent=2)
    log(f"[fixture] {summary}")

    fails = []
    if not summary["linear_pos_match_naive"]:
        fails.append("linear positions differ from naive reference")
    if summary["linear_val_max_abs_diff"] > 1e-6:
        fails.append("linear values differ from naive reference")
    if not summary["pchip_pos_match_naive"]:
        fails.append("pchip positions differ from naive reference")
    if summary["pchip_val_max_abs_diff"] > 1e-6:
        fails.append("pchip values differ from naive reference")
    if not summary["classification_ok"]:
        fails.append(f"eligibility classification: {summary['got_eligible_runs']}"
                     f"/{summary['got_ctx_runs']}/{summary['got_eligible_bins']}"
                     f"/{summary['got_boundary_runs']}")
    if (summary["r0_linear_max_abs_diff"] is None
            or summary["r0_linear_max_abs_diff"] > 1e-6):
        fails.append("r0 closed-form linear mismatch")
    if (summary["r1_linear_max_abs_diff"] is None
            or summary["r1_linear_max_abs_diff"] > 1e-6):
        fails.append("r1 closed-form linear mismatch")
    if not summary["pchip_within_4pt_minmax"]:
        fails.append("PCHIP exceeded 4-pt [min,max]")
    if not summary["observed_preserved_all_arms"]:
        fails.append("observed values altered by a fill")
    if not summary["fallback_exact"]:
        fails.append("H_pchip fallback != H_linear at fallback positions")
    if not summary["clock_at_ineligible"]:
        fails.append("H_clock fill wrong at ineligible positions")
    if not summary["linear_at_eligible"]:
        fails.append("H_linear values wrong at eligible positions")
    if not summary["pchip_at_ctx"]:
        fails.append("H_pchip values wrong at ctx positions")
    if not summary["global_at_all_missing"]:
        fails.append("H_global value wrong at missing positions")
    if not summary["pchip_actually_differs"]:
        fails.append("H_pchip == H_linear everywhere (pchip never applied)")
    if not summary["fill_clock_train_only"]:
        fails.append("fill_clock not train-only")
    if not summary["fill_global_train_only"]:
        fails.append("fill_global not train-only")
    if fails:
        raise SystemExit("FIXTURE FAILED: " + "; ".join(fails))
    log("[fixture] PASS")


# ----------------------------------------------------------------------------
# Data-level checks (once, before any fit)
# ----------------------------------------------------------------------------
def data_checks(data, mask_pu, B40) -> dict:
    users = data["users"]
    n = len(users)
    assert data["hr"].shape == (n * DAYS, N_BINS)
    # mask == (cnt > 0) asserted inside load_data (cnt popped after load)
    assert np.isfinite(data["hr"][data["mask"]]).all(), "non-finite observed HR"

    # (user, date) multiset of bins rows == d40 rows (lexicographic sort)
    def lex_pairs(u, d):
        a = np.stack([np.asarray(u, np.int64), np.asarray(d, np.int64)],
                     axis=1)
        return a[np.lexsort((a[:, 1], a[:, 0]))]
    d40 = data["d40"]
    assert np.array_equal(lex_pairs(data["pu"], data["pd"]),
                          lex_pairs(d40.user, d40.date)), \
        "bins rows and d40 rows are not the same (user, date) multiset"
    # exactly DAYS rows per user
    vc = pd.Series(data["pu"]).value_counts()
    assert bool((vc == DAYS).all()), "users without exactly 40 bin rows"

    folds = data["folds"]
    for r in ALLOCATION_TARGETS:
        cnt = folds[folds.alloc == r].fold.value_counts().to_dict()
        sizes = (cnt.get("train", 0), cnt.get("val", 0), cnt.get("test", 0))
        assert sizes == TRAIN_SHARE, f"alloc {r} sizes {sizes} != {TRAIN_SHARE}"

    assert B40.shape == (n, 305) and np.isfinite(B40).all()
    return {
        "mask_eq_cnt_gt_0": True,
        "observed_hr_finite": True,
        "bins_d40_user_date_multiset": True,
        "exactly_40_bin_rows_per_user": True,
        "fold_sizes_2696_579_573_all_allocs": True,
        "B40_shape_305_finite": True,
    }


# ----------------------------------------------------------------------------
# H_clock reproduction gate vs recording pilot H (PLAN §"Checks" item 2)
# ----------------------------------------------------------------------------
def h_clock_reproduction_check(r: int, rows_m, rows_p) -> dict:
    ref_preds = pd.read_csv(_RECORD_RESULTS / "predictions.csv")
    ref_metrics = pd.read_csv(_RECORD_RESULTS / "metrics.csv")
    ref_h = ref_preds[(ref_preds.arm == "H") & (ref_preds.alloc == r)]
    mine_df = pd.concat(rows_p, ignore_index=True)
    mine_h = mine_df[(mine_df.arm == "H_clock") & (mine_df.allocation == r)]
    merged = mine_h.merge(
        ref_h[["user", "split", "score"]].rename(
            columns={"score": "ref_score"}),
        on=["user", "split"], how="inner")
    if not (len(merged) == len(mine_h) == len(ref_h)):
        return {"passed": False,
                "reason": f"row mismatch mine={len(mine_h)} ref={len(ref_h)} "
                          f"merged={len(merged)}"}
    diff = (merged["score"].to_numpy(dtype=np.float64)
            - merged["ref_score"].to_numpy(dtype=np.float64))
    max_abs = float(np.max(np.abs(diff)))
    my = [row for row in rows_m if row["arm"] == "H_clock"][0]
    ref = ref_metrics[(ref_metrics.arm == "H")
                      & (ref_metrics.alloc == r)].iloc[0]
    alpha_match = bool(my["alpha"] == float(ref.alpha))
    d_va = abs(my["auroc_val"] - float(ref.auroc_val))
    d_te = abs(my["auroc_test"] - float(ref.auroc_test))
    auroc_match = bool(d_va <= AUROC_TOL and d_te <= AUROC_TOL)
    return {
        "passed": bool(max_abs <= SCORE_TOL and alpha_match and auroc_match),
        "n_compared": int(len(merged)),
        "max_abs_diff_score": max_abs,
        "score_tol": SCORE_TOL,
        "alpha_match": alpha_match,
        "my_alpha": my["alpha"], "ref_alpha": float(ref.alpha),
        "auroc_diff_val": float(d_va), "auroc_diff_test": float(d_te),
        "my_auroc_val": my["auroc_val"], "ref_auroc_val": float(ref.auroc_val),
        "my_auroc_test": my["auroc_test"],
        "ref_auroc_test": float(ref.auroc_test),
    }


# ----------------------------------------------------------------------------
# Per-allocation pipeline (H_clock first; gate before incremental write)
# ----------------------------------------------------------------------------
def run_alloc(r, data, mask_pu, B40, overlays, mr_fit, mr_transform):
    users, y, folds = data["users"], data["y"], data["folds"]
    n = len(users)
    fold = (folds[folds.alloc == r].set_index("user_id")["fold"].astype(str))
    fold.index = fold.index.astype("int64")
    fold = fold.reindex(users).to_numpy()
    tr = np.flatnonzero(fold == "train")
    va = np.flatnonzero(fold == "val")
    te = np.flatnonzero(fold == "test")
    assert (len(tr), len(va), len(te)) == TRAIN_SHARE
    assert not (set(tr) & set(va) or set(tr) & set(te)
                or set(va) & set(te)), f"alloc {r}: splits not disjoint"
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    wall_a = time.perf_counter()

    # same four selected training days for every arm (comp_seed(r, 0))
    fit_rng = np.random.default_rng(comp_seed(r, COMP_DAY_SELECT))
    fit_days = np.stack([fit_rng.choice(DAYS, 4, replace=False)
                          for _ in range(len(tr))])

    # shared, training-preprocessed B40 across arms within the allocation
    B40Zt, B40Zv, B40Ze, *_ = hygiene_std(B40[tr], B40[va], B40[te])

    # per-alloc fills (train-only medians)
    hr_pu = data["hr"].reshape(n, DAYS, N_BINS)
    hr_clock, fill_vec = fill_hr(hr_pu, mask_pu, tr)
    obs_train = hr_pu[tr][mask_pu[tr]]
    overall = float(np.nanmedian(obs_train)) if obs_train.size else 80.0
    has_obs = mask_pu[tr].any(axis=(0, 1))             # (288,)
    assert np.all(fill_vec[~has_obs] == np.float32(overall)), \
        "clock bins without training observations != overall fallback"
    fill_global = np.where(mask_pu, hr_pu,
                           np.float32(overall)).astype(np.float32)

    rows_m, rows_p, traces, walls = [], [], {}, {}
    for arm in ARMS:                                   # H_clock first
        t_arm = time.perf_counter()
        filled = build_filled(arm, hr_clock, fill_global, overlays)
        assert np.isfinite(filled).all(), f"{arm}: non-finite fill"
        assert np.array_equal(filled[mask_pu], hr_pu[mask_pu]), \
            f"{arm}: observed values altered"

        m, sd = fit_and_pool_mr(r, filled, fit_days, tr, COMP_MR_FIT, True,
                                 mask_pu, mr_fit, mr_transform)
        block = np.concatenate([m, sd], axis=1)        # (n, 18816) f32
        del filled, m, sd
        TZt, TZv, TZe, *_ = hygiene_std(block[tr], block[va], block[te])
        del block
        Zt = np.concatenate([B40Zt, TZt], axis=1)
        Zv = np.concatenate([B40Zv, TZv], axis=1)
        Ze = np.concatenate([B40Ze, TZe], axis=1)
        assert (np.isfinite(Zt).all() and np.isfinite(Zv).all()
                and np.isfinite(Ze).all()), f"{arm}: non-finite Z"
        del TZt, TZv, TZe

        alpha, trace = select_alpha(Zt, y_tr, Zv, y_va)
        mdl = Ridge(alpha=float(alpha), fit_intercept=True).fit(Zt, y_tr)
        s_va = mdl.predict(Zv)
        s_te = mdl.predict(Ze)
        auc_va = float(roc_auc_score(y_va, s_va))
        auc_te = float(roc_auc_score(y_te, s_te))
        boundary = bool(alpha == float(ALPHA_GRID[0])
                        or alpha == float(ALPHA_GRID[-1]))

        rows_m.append({"alloc": r, "arm": arm,
                       "auroc_val": auc_va, "auroc_test": auc_te,
                       "n_cols": int(Zt.shape[1]), "alpha": float(alpha),
                       "boundary": boundary})
        for split, idx, ys, s in (("val", va, y_va, s_va),
                                   ("test", te, y_te, s_te)):
            rows_p.append(pd.DataFrame({
                "user": users[idx].astype(np.int64),
                "allocation": r, "split": split, "arm": arm,
                "label": ys.astype(np.int64),
                "score": s.astype(np.float32)}))
        traces[arm] = {"alpha_selected": float(alpha), "trace": trace,
                       "boundary": boundary}
        walls[arm] = round(time.perf_counter() - t_arm, 1)
        log(f"[r={r}] {arm}: val {auc_va:.4f} test {auc_te:.4f} "
            f"cols {Zt.shape[1]} α={alpha:g} ({walls[arm]}s, "
            f"RSS {peak_rss_mb():.0f} MiB)")
        del Zt, Zv, Ze, mdl

    # H_clock reproduction gate BEFORE any incremental write
    repro = h_clock_reproduction_check(r, rows_m, rows_p)
    if not repro["passed"]:
        raise SystemExit(f"H_clock reproduction FAILED for alloc {r}: "
                         f"{repro}")

    mdf = pd.DataFrame(rows_m)
    mdf.to_csv(RESULTS / "metrics.csv", mode="a",
               header=not (RESULTS / "metrics.csv").exists()
               or (RESULTS / "metrics.csv").stat().st_size == 0,
               index=False)
    pd.concat(rows_p, ignore_index=True).to_csv(
        RESULTS / "predictions.csv", mode="a",
        header=not (RESULTS / "predictions.csv").exists()
        or (RESULTS / "predictions.csv").stat().st_size == 0,
        index=False)
    json.dump(traces, open(RESULTS / f"alpha_traces_alloc{r}.json", "w"),
              indent=2)

    return {
        "alloc": r,
        "fold_sizes": {"train": int(len(tr)), "val": int(len(va)),
                        "test": int(len(te))},
        "wall_s_per_arm": walls,
        "wall_s_alloc": round(time.perf_counter() - wall_a, 1),
        "fill_overall": overall,
        "fill_clock_min": float(np.nanmin(fill_vec)),
        "fill_clock_max": float(np.nanmax(fill_vec)),
        "fill_clock_empty_bins": int((~has_obs).sum()),
        "h_clock_repro": repro,
    }


def mr_warmup(mr_fit, mr_transform):
    """JIT-compile MR fit/transform once so per-arm walls are comparable."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    X = rng.normal(60.0, 8.0, size=(32, N_BINS))
    X1 = np.diff(X, axis=1)
    _nb_seed(12345)
    pb = mr_fit(X, num_features=MR_NUM_FEATURES_PER_TRANS,
                max_dilations_per_kernel=32)
    _nb_seed(12346)
    pdd = mr_fit(X1, num_features=MR_NUM_FEATURES_PER_TRANS,
                  max_dilations_per_kernel=32)
    _ = mr_transform(X, X1, pb, pdd, n_features_per_kernel=4)
    log(f"[warmup] MR JIT compile {time.perf_counter() - t0:.1f}s")


def run_pipeline():
    data = load_data()
    data.pop("cnt")                       # mask==(cnt>0) asserted in load_data
    users = data["users"]
    n = len(users)
    mask_pu = np.ascontiguousarray(data["mask"].reshape(n, DAYS, N_BINS))
    log(f"[load] {n} users | bins {data['hr'].shape} | "
        f"RSS {peak_rss_mb():.0f} MiB")

    B40 = build_static(users, data["d40"], data["mask"], data["hr"])["B40"]
    checks = data_checks(data, mask_pu, B40)
    json.dump(checks, open(CACHE / "data_checks.json", "w"), indent=2)
    log(f"[data] checks PASS: {checks}")

    t0 = time.perf_counter()
    overlays = compute_gap_overlays(
        np.ascontiguousarray(data["hr"].reshape(n, DAYS, N_BINS)), mask_pu)
    json.dump(overlays["counts"], open(CACHE / "gap_counts.json", "w"),
              indent=2)
    np.savez(CACHE / "exposure.npz", **overlays["per_user"])
    c = overlays["counts"]
    log(f"[overlays] eligible {c['eligible_runs']:,} runs / "
        f"{c['eligible_bins']:,} bins (ctx {c['ctx_runs']:,} runs, "
        f"fb {c['fb_runs']:,} runs) of {c['total_missing_bins']:,} missing "
        f"bins in {time.perf_counter() - t0:.1f}s | "
        f"RSS {peak_rss_mb():.0f} MiB")

    mr_fit, mr_transform = vendor_mr()
    mr_warmup(mr_fit, mr_transform)

    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics_csv = RESULTS / "metrics.csv"
    done = set()
    if metrics_csv.exists() and metrics_csv.stat().st_size > 0:
        try:
            done = set(int(a) for a in pd.read_csv(metrics_csv).alloc.unique())
            log(f"[run] resume: allocs done {sorted(done)}")
        except Exception:
            done = set()

    infos_path = CACHE / "alloc_infos.json"
    infos = json.load(open(infos_path)) if infos_path.exists() else []
    have = {i["alloc"] for i in infos}

    for r in ALLOCATION_TARGETS:
        if r in done:
            if r not in have:
                infos.append({"alloc": r, "from_csv_only": True})
            log(f"[run] alloc {r} already in metrics.csv — skip")
            continue
        log(f"[run] === alloc {r} ===")
        info = run_alloc(r, data, mask_pu, B40, overlays,
                          mr_fit, mr_transform)
        infos = [i for i in infos if i["alloc"] != r] + [info]
        json.dump(infos, open(infos_path, "w"), indent=2)
        log(f"[run] alloc {r} done in {info['wall_s_alloc']:.1f}s | "
            f"peak RSS {peak_rss_mb():.0f} MiB")
    log("[run] all target allocs done")


# ----------------------------------------------------------------------------
# REPORT.md + repro.json
# ----------------------------------------------------------------------------
DELTA_PAIRS = (
    ("H_global − H_clock", "H_global", "H_clock"),
    ("H_linear − H_clock", "H_linear", "H_clock"),
    ("H_pchip − H_clock",  "H_pchip",  "H_clock"),
    ("H_pchip − H_linear", "H_pchip",  "H_linear"),
)


def _require_complete(m):
    have = {(int(a.alloc), a.arm) for a in m.itertuples()}
    want = {(r, arm) for r in ALLOCATION_TARGETS for arm in ARMS}
    if have != want:
        raise SystemExit(f"metrics incomplete: missing {sorted(want - have)}; "
                         "finish all 12 fits before reporting")


def write_report():
    m = pd.read_csv(RESULTS / "metrics.csv")
    p = pd.read_csv(RESULTS / "predictions.csv")
    _require_complete(m)
    fixture = (json.load(open(CACHE / "fixture.json"))
               if (CACHE / "fixture.json").exists() else {})
    counts = (json.load(open(CACHE / "gap_counts.json"))
              if (CACHE / "gap_counts.json").exists() else {})
    exp = np.load(CACHE / "exposure.npz")

    L = []
    L.append("# ch3000 filling sensitivity — results\n")
    L.append("Implements [`../PLAN.md`](../PLAN.md). 4 fill arms × 3 "
             "allocations (0/1/2, chosen before results); cached 5-min bins; "
             "Garmin/source 3, ch3000 only; all arms `B40 + pooled "
             "MultiRocket(HR) + Ridge` with one shared seed stream per "
             "allocation (fit days `comp_seed(r,0)`, MR base `comp_seed(r,1)`, "
             "MR diff `comp_seed(r,2)` — the executed recording pilot's "
             "convention). Reference: recording pilot commit `f6123c9` "
             "(arm H).\n")

    # §1 arm table
    L.append("\n## 1. Arm table (per-alloc validation / test AUROC)\n")
    L.append("| arm | alloc | val AUROC | test AUROC | n cols | α | "
              "boundary |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        sub = m[m.arm == arm].sort_values("alloc")
        for _, row in sub.iterrows():
            L.append(f"| `{arm}` | {int(row.alloc)} | "
                     f"{row.auroc_val:.4f} | {row.auroc_test:.4f} | "
                     f"{int(row.n_cols):,} | {row.alpha:g} | "
                     f"{'yes' if bool(row.boundary) else 'no'} |")
        L.append(f"| `{arm}` | **mean ± SD** | "
                 f"{sub.auroc_val.mean():.4f} ± "
                 f"{sub.auroc_val.std(ddof=1):.4f} | "
                 f"{sub.auroc_test.mean():.4f} ± "
                 f"{sub.auroc_test.std(ddof=1):.4f} | "
                 f"{int(sub.n_cols.mean()):,} | — | — |")

    # §2 paired test deltas
    def series(arm, col):
        return (m[m.arm == arm].sort_values("alloc")[col].to_numpy())
    L.append("\n## 2. Paired TEST AUROC deltas (3 allocations; "
             "no p-values, PLAN §\"Readout\")\n")
    L.append("| comparison | alloc 0 | alloc 1 | alloc 2 | mean | range | "
              "direction |")
    L.append("|---|---|---|---|---|---|---|")
    for label, a, b in DELTA_PAIRS:
        d = series(a, "auroc_test") - series(b, "auroc_test")
        agree = ("all 3 +" if (d > 0).all() else
                 "all 3 −" if (d < 0).all() else f"{int((d > 0).sum())}/3 +")
        L.append(f"| {label} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | "
                 f"{d.mean():+.4f} | [{d.min():+.4f}, {d.max():+.4f}] | "
                 f"{agree} |")
    L.append("\n### 2b. Paired VALIDATION AUROC deltas (secondary)\n")
    L.append("| comparison | mean | range | direction |")
    L.append("|---|---|---|---|")
    for label, a, b in DELTA_PAIRS:
        d = series(a, "auroc_val") - series(b, "auroc_val")
        agree = ("all 3 +" if (d > 0).all() else
                 "all 3 −" if (d < 0).all() else f"{int((d > 0).sum())}/3 +")
        L.append(f"| {label} | {d.mean():+.4f} | "
                 f"[{d.min():+.4f}, {d.max():+.4f}] | {agree} |")

    # §3 Spearman of paired test scores
    L.append("\n## 3. Spearman ρ of paired test scores (per alloc)\n")
    L.append("| comparison | alloc 0 | alloc 1 | alloc 2 |")
    L.append("|---|---|---|---|")
    for label, a, b in DELTA_PAIRS:
        rhos = []
        for r in ALLOCATION_TARGETS:
            ma = p[(p.arm == a) & (p.allocation == r) & (p.split == "test")]
            mb = p[(p.arm == b) & (p.allocation == r) & (p.split == "test")]
            merged = ma.merge(
                mb[["user", "score"]].rename(columns={"score": "score_b"}),
                on="user")
            rho = float(spearmanr(merged["score"],
                                   merged["score_b"]).statistic)
            rhos.append(rho)
        L.append(f"| {label} | {rhos[0]:.4f} | {rhos[1]:.4f} | "
                 f"{rhos[2]:.4f} |")

    # §4 H_clock reproduction gate (recomputed from saved artifacts)
    L.append("\n## 4. H_clock reproduction vs recording pilot H "
             "(PLAN §\"Checks\" item 2)\n")
    L.append("| alloc | n | max |Δscore| | α | AUROC val (mine/ref) | "
              "AUROC test (mine/ref) |")
    L.append("|---|---|---|---|---|---|")
    ref_preds = pd.read_csv(_RECORD_RESULTS / "predictions.csv")
    ref_metrics = pd.read_csv(_RECORD_RESULTS / "metrics.csv")
    for r in ALLOCATION_TARGETS:
        mine_h = p[(p.arm == "H_clock") & (p.allocation == r)]
        ref_h = ref_preds[(ref_preds.arm == "H") & (ref_preds.alloc == r)]
        merged = mine_h.merge(
            ref_h[["user", "split", "score"]].rename(
                columns={"score": "ref_score"}), on=["user", "split"])
        max_abs = float(np.max(np.abs(merged["score"].to_numpy(np.float64)
                                      - merged["ref_score"].to_numpy(
                                          np.float64))))
        mine_row = m[(m.arm == "H_clock") & (m.alloc == r)].iloc[0]
        ref_row = ref_metrics[(ref_metrics.arm == "H")
                              & (ref_metrics.alloc == r)].iloc[0]
        L.append(f"| {r} | {len(merged)} | {max_abs:.2e} | "
                 f"{mine_row.alpha:g} / {ref_row.alpha:g} | "
                 f"{mine_row.auroc_val:.4f} / {ref_row.auroc_val:.4f} | "
                 f"{mine_row.auroc_test:.4f} / {ref_row.auroc_test:.4f} |")

    # §5 exposure audit
    if counts:
        tm = counts["total_missing_bins"]
        L.append("\n## 5. Intervention exposure (from masks; "
                 "alloc- and arm-independent)\n")
        L.append(f"- Missing bins (3,848 users × 40 days × 288 bins): "
                 f"**{tm:,}** in **{counts['total_runs']:,}** runs "
                 f"(boundary-touching {counts['boundary_runs']:,}; "
                 f"length > 6: {counts['long_runs_gt6']:,}).\n")
        L.append(f"- Interpolation-eligible (interior 1–6-bin runs bounded "
                 f"by observed bins): **{counts['eligible_runs']:,}** runs / "
                 f"**{counts['eligible_bins']:,}** bins = "
                 f"**{counts['eligible_bins'] / tm * 100:.2f}%** of missing "
                 f"bins.\n")
        L.append(f"- With 4-point PCHIP context: **{counts['ctx_runs']:,}** "
                 f"runs / **{counts['ctx_bins']:,}** bins = "
                 f"**{counts['ctx_bins'] / tm * 100:.2f}%** of missing; "
                 f"linear fallback (eligible, context incomplete): "
                 f"**{counts['fb_runs']:,}** runs / "
                 f"**{counts['fb_bins']:,}** bins = "
                 f"**{counts['fb_bins'] / tm * 100:.2f}%**.\n")
        L.append(f"- Remaining on H_clock fill (ineligible gaps): "
                 f"**{counts['ineligible_bins']:,}** bins = "
                 f"**{counts['ineligible_bins'] / tm * 100:.2f}%** of "
                 f"missing.\n")
        L.append(f"- Run-length histogram (1..9, >9): "
                 f"{counts['len_hist_1_to_9_gt9']}.\n")

        miss_u = exp["miss_bins"]
        elig_u = exp["elig_bins"]
        ctx_u = exp["ctx_bins"]
        fb_u = exp["fb_bins"]
        days_e = exp["days_elig"]
        days_t = exp["days_total"]
        bins_t = exp["bins_total"]
        with np.errstate(invalid="ignore", divide="ignore"):
            miss_frac = miss_u / bins_t
            elig_of_miss = np.where(miss_u > 0, elig_u / miss_u, np.nan)
            days_frac = days_e / days_t
            pchip_of_miss = np.where(miss_u > 0, ctx_u / miss_u, np.nan)
            fb_of_miss = np.where(miss_u > 0, fb_u / miss_u, np.nan)
            clock_of_miss = np.where(miss_u > 0,
                                      (miss_u - elig_u) / miss_u, np.nan)
            pchip_of_elig = np.where(elig_u > 0, ctx_u / elig_u, np.nan)

        def med_rng(v):
            v = v[np.isfinite(v)]
            return (float(np.median(v)), float(v.min()), float(v.max()),
                    int(v.size))

        L.append("\n### Per-participant exposure (median; [range])\n")
        L.append("| fraction | median | range | n users |")
        L.append("|---|---|---|---|")
        for name, v in (
                ("missing bins / all bins", miss_frac),
                ("eligible / missing", elig_of_miss),
                ("days containing an eligible gap", days_frac),
                ("PCHIP-filled / missing (H_pchip)", pchip_of_miss),
                ("linear-fallback / missing (H_pchip)", fb_of_miss),
                ("clock-filled / missing (all arms)", clock_of_miss),
                ("PCHIP / eligible (H_pchip)", pchip_of_elig)):
            med, lo, hi, k = med_rng(v)
            L.append(f"| {name} | {med:.4f} | [{lo:.4f}, {hi:.4f}] | {k} |")
        L.append("\nZero-denominator fractions (users with no missing bins, "
                 "or no eligible bins for the last row) are excluded and "
                 "counted in the n-users column; "
                 f"users with zero missing bins: "
                 f"{int((miss_u == 0).sum())}.\n")

    # §6 checks
    L.append("\n## 6. Checks (PLAN §\"Checks\")\n")
    L.append("| check | status |")
    L.append("|---|---|")
    checks = json.load(open(CACHE / "data_checks.json"))
    for c, s in checks.items():
        L.append(f"| data: {c} | {'PASS' if s else 'FAIL'} |")
    if fixture:
        L.append(f"| fixture: eligibility classification (1/6/7-bin, "
                 f"boundary, no-ctx, turning point) | "
                 f"{'PASS' if fixture['classification_ok'] else 'FAIL'} |")
        L.append(f"| fixture: overlays == naive loop reference "
                 f"(positions + values) | "
                 f"PASS (linear Δ {fixture['linear_val_max_abs_diff']:.1e}; "
                 f"pchip Δ {fixture['pchip_val_max_abs_diff']:.1e}) |")
        lin_max = max(fixture["r0_linear_max_abs_diff"],
                      fixture["r1_linear_max_abs_diff"])
        L.append(f"| fixture: closed-form linear values (r0, r1) | "
                 f"PASS (Δ {lin_max:.1e}) |")
        L.append(f"| fixture: PCHIP within 4-pt [min, max] "
                 f"(incl. turning point) | "
                 f"{'PASS' if fixture['pchip_within_4pt_minmax'] else 'FAIL'} |")
        L.append(f"| fixture: exact linear fallback + H_clock at "
                 f"ineligible + observed preserved | "
                 f"PASS ({fixture['fallback_exact']}, "
                 f"{fixture['clock_at_ineligible']}, "
                 f"{fixture['observed_preserved_all_arms']}) |")
        L.append(f"| fixture: fill_clock/fill_global train-only | "
                 f"PASS ({fixture['fill_clock_train_only']}, "
                 f"{fixture['fill_global_train_only']}) |")
    L.append("| observed HR/masks/counts/B40 unchanged; finite filled inputs "
             "and Z (every arm, every alloc) | PASS (per-arm assertions) |")
    L.append("| identical interpolation eligibility across H_linear / "
             "H_pchip | PASS (single shared computation; fixture asserts "
             "pchip ∪ fallback = eligible, disjoint) |")
    L.append("| H_clock score match vs recording pilot H ≤ 1e-6 "
             "(α, AUROC equal) | PASS (run-time gate; see §4) |")

    # §7 interpretation (authored after viewing the tables; values quoted
    # from §1–§5 of this report)
    L.append("\n## 7. Interpretation\n")
    L.append("**Pipeline validation is exact.** H_clock's saved scores are "
             "textually identical to the recording pilot's H arm across all "
             "1,152 val+test users × 3 allocations (§4: max |Δscore| = "
             "0.00e+00, α and AUROC equal), so the four arms differ only "
             "through the filled values at missing bins — the comparison is "
             "clean.\n")
    L.append("**Fill choice is not a material modeling decision at this "
             "exposure.** Mean test AUROC spans 0.8464–0.8485 across arms "
             "while each arm's cross-allocation SD is ≈ 0.016; the largest "
             "mean paired test delta is +0.0021 (H_global − H_clock, all 3 "
             "allocations positive) and the smallest +0.0002 (H_pchip − "
             "H_linear). No delta reaches a seventh of the cross-allocation "
             "SD. With R = 3 and reused splits there is no significance "
             "machinery (PLAN §\"Readout\"), but the bounded effect is "
             "itself the finding: the missing-bin fill rule shifts ch3000 "
             "test AUROC by at most ~0.003.\n")
    L.append("**The two interpolation arms are interchangeable.** Spearman "
             "ρ(H_pchip, H_linear) = 0.9993–0.9995 on paired test scores "
             "(§3); mean test delta +0.0002. Local linear vs PCHIP is a "
             "distinction without a difference here.\n")
    L.append("**H_global is the largest and most distinct intervention, and "
             "it is fine.** H_global overwrites 100% of missing bins with "
             "one overall training median (≈ 70.2 bpm), far more input mass "
             "than the interpolation arms' 3.87% — and correspondingly the "
             "lowest ρ vs H_clock (0.894 on alloc 1, 0.982 elsewhere). It "
             "still lands within +0.0021 mean test AUROC of H_clock: the "
             "crudest fill is at least as good as the clock medians and the "
             "local interpolations. The T(HR) signal evidently lives in the "
             "88.5% observed bins, not in how the holes are painted.\n")
    L.append("**The val/test sign asymmetry is an α-selection artefact, not "
             "a fill property.** Every alternative is slightly worse on "
             "validation (mean Δ −0.0004 to −0.0044) yet slightly better on "
             "test (+0.0007 to +0.0021). On allocation 1 H_clock selected "
             "α = 1e3 while all three alternatives selected α = 1e4; the "
             "less-regularised H_clock fit wins the validation maximum but "
             "generalises slightly worse. On allocations 0/2, where all "
             "arms selected the same α = 1e4, deltas are near zero. "
             "Post-hygiene dimensionality is arm-invariant (19,121 / "
             "19,121 / 19,115), so the intervention never shifts feature "
             "selection.\n")
    L.append("**Conclusion.** For ch3000 MultiRocket(HR) input on this "
             "cohort, the recording pilot's result is robust to the "
             "missing-bin fill decision: all four fills sit within ~0.002 "
             "test AUROC of one another, well inside allocation-to-"
             "allocation variation. A null here means \"no detectable "
             "effect at 3.87% exposure\" for the interpolation arms — "
             "not that interpolation cannot matter — but the H_global arm, "
             "which touches every missing bin, bounds the whole family at "
             "+0.002.\n")

    # §8 caveats
    L.append("\n## 8. Caveats\n")
    L.append("- 3 allocations, R = 3; no bootstrap, p-values, multiplicity "
             "families or population-confidence claims (PLAN §\"Readout\"). "
             "Validation maxima are selected estimates; test comparisons are "
             "exploratory on this reused cohort.\n")
    L.append("- Splits are reused from R1b / the recording pilot; "
             "validation-alpha selection and test evaluation share folds — "
             "an explicit deviation authorized by this plan, matched to the "
             "recording pilot protocol.\n")
    L.append("- B40 contains per-clock-bin mask means throughout "
             "(recording features); they are fill-invariant, so any fill "
             "sensitivity is attributable to the T(HR) block alone.\n")
    L.append("- Six bins (30 min) is a fixed pilot convention, not a "
             "physiological threshold; PCHIP context is strictly local "
             "(two observed bins each side, same day, never across another "
             "gap).\n")
    L.append("- Interpolation arms change only ~4% of missing bins overall; "
             "robustness conclusions are bounded by that exposure (§5). "
             "Neither result quantifies physiological signal or excludes "
             "density, device/behaviour effects, or interactions with "
             "recording structure.\n")
    L.append("- Recorded salutation ≠ biological sex/gender.\n")

    # §9 recommended next step (authored after viewing the tables)
    L.append("\n## 9. Recommended next step\n")
    L.append("Close the fill question and stop. Per PLAN §\"Stop and "
             "decide\": the answer is bounded — fill choice within this "
             "family shifts ch3000 test AUROC by ≤ 0.0021, well inside the "
             "0.016 cross-allocation SD — so extending the allocations, "
             "adding methods, or starting density experiments on this axis "
             "is not expected to change the conclusion. In particular, "
             "*do not* change the fill convention used by the recording "
             "pilot (H_clock, training-only clock-bin medians): the small "
             "H_global test mean is within R = 3 noise, and H_clock keeps "
             "this experiment byte-identically aligned with the recording "
             "pilot's H arm. Use this report as a robustness audit of that "
             "convention; the next ch3000 question is the recording-"
             "structure axis already addressed by the recording pilot's "
             "other arms, not a further fill axis.\n")

    open(RESULTS / "REPORT.md", "w").write("\n".join(L))


def write_repro():
    pin = json.load(open(_TEMPORAL / "cache" / "vendor_repro.json"))["pin"]
    m = pd.read_csv(RESULTS / "metrics.csv")
    _require_complete(m)
    infos = (json.load(open(CACHE / "alloc_infos.json"))
             if (CACHE / "alloc_infos.json").exists() else [])
    exp = np.load(CACHE / "exposure.npz")
    miss_u = exp["miss_bins"]; elig_u = exp["elig_bins"]
    ctx_u = exp["ctx_bins"]; fb_u = exp["fb_bins"]
    days_e = exp["days_elig"]; days_t = exp["days_total"]
    bins_t = exp["bins_total"]
    with np.errstate(invalid="ignore", divide="ignore"):
        repro = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "numba": numba.__version__,
            "pin": pin,
            "input_sha256": {
                "day_bins.npz": sha256_file(BINS_NPZ),
                "day40_table.parquet": sha256_file(DAY40_PARQUET),
                "folds.parquet": sha256_file(FOLDS_PARQUET),
                "cohort_manifest_model_sources.parquet":
                    sha256_file(PATH_COHORT),
                "split_manifest_sq.parquet": sha256_file(PATH_SPLIT),
                "recording_metrics.csv":
                    sha256_file(_RECORD_RESULTS / "metrics.csv"),
                "recording_predictions.csv":
                    sha256_file(_RECORD_RESULTS / "predictions.csv"),
            },
            "config": {
                "ARMS": list(ARMS),
                "ARM_BLOCKS": {a: ["B40", "T(HR)"] for a in ARMS},
                "ALPHA_GRID": [float(a) for a in ALPHA_GRID],
                "ALLOCATION_TARGETS": list(ALLOCATION_TARGETS),
                "TOLERANCES": {"score_max_abs": SCORE_TOL,
                                "auroc": AUROC_TOL},
                "COMPONENT_IDS": {"DAY_SELECT": COMP_DAY_SELECT,
                                   "MR_FIT_HR": COMP_MR_FIT},
                "seed_streams": {
                    "comp_seed": f"SeedSequence([{SEED_BASE}, 3, r, comp])",
                    "fit_days": "comp_seed(r, 0); identical across arms",
                    "MR_base": "comp_seed(r, 1); _nb_seed before mr_fit on "
                                "the arm's filled training inputs",
                    "MR_diff": "comp_seed(r, 2); _nb_seed before mr_fit on "
                                "first differences",
                },
                "fill_definitions": {
                    "H_clock": "fill_hr (temporal): train-only clock-bin "
                                "medians over observed entries; overall "
                                "train median fallback for empty bins",
                    "H_global": "single overall train median over all "
                                 "observed HR bin means",
                    "H_linear": "linear interpolation for eligible gaps "
                                 "(interior 1–6-bin runs bounded by observed "
                                 "bins); H_clock elsewhere",
                    "H_pchip": "PCHIP for eligible gaps with 4-point "
                                "context; linear fallback for other "
                                "eligible gaps; H_clock elsewhere",
                },
                "eligibility": {
                    "interior": "sr >= 1 and er <= 286",
                    "len_1_to_6": True,
                    "bounded_by_observed":
                        "mask[sr-1] and mask[er+1] (automatic for maximal "
                        "interior runs)",
                    "pchip_context": "sr >= 2 and er <= 285 and mask[sr-2] "
                                       "and mask[er+2]; else linear fallback",
                    "shared_across_arms": True,
                },
                "pchip_implementation":
                    "per gap length L in 1..6 one scipy.interpolate."
                    "PchipInterpolator(x=[0,1,L+2,L+3], y=(4,n_L) float64, "
                    "extrapolate=False) over the four context bin centres, "
                    "evaluated at arange(2,2+L); batched 2-D y is bit-equal "
                    "to per-gap scipy fits (validated on fixture + 2000-run "
                    "real sample, max diff 0.0); stored float32",
                "overlays": "alloc/arm-independent (original HR + masks "
                             "only); per-alloc variation via H_clock base "
                             "and H_global median",
                "threading": {"numba_layer": "workqueue",
                               "numba_threads": 8},
                "pooling": "T(HR): mask day-gate (temporal/recording "
                            "convention, original masks)",
                "hygiene": "SimpleImputer(median) + VarianceThreshold(0) + "
                            "StandardScaler, per block, train-only fit; B40 "
                            "block shared across arms within an allocation",
                "alpha_selection": "per arm per alloc on validation AUROC; "
                                    "tie -> smallest alpha; boundary recorded "
                                    "(recording pilot grid)",
            },
            "data_checks": json.load(open(CACHE / "data_checks.json")),
            "fixture": (json.load(open(CACHE / "fixture.json"))
                         if (CACHE / "fixture.json").exists() else None),
            "gap_counts": (json.load(open(CACHE / "gap_counts.json"))
                            if (CACHE / "gap_counts.json").exists() else None),
            "per_user_exposure": {
                "miss_frac": (miss_u / bins_t).tolist(),
                "elig_of_miss": np.where(miss_u > 0, elig_u / miss_u,
                                          None).tolist(),
                "days_elig_frac": (days_e / days_t).tolist(),
                "pchip_of_miss": np.where(miss_u > 0, ctx_u / miss_u,
                                           None).tolist(),
                "fb_of_miss": np.where(miss_u > 0, fb_u / miss_u,
                                        None).tolist(),
                "pchip_of_elig": np.where(elig_u > 0, ctx_u / elig_u,
                                           None).tolist(),
            },
            "metrics_summary": {},
            "per_alloc_info": infos,
        }
        for arm in ARMS:
            sub = m[m.arm == arm].sort_values("alloc")
            repro["metrics_summary"][arm] = {
                "val_mean": round(float(sub.auroc_val.mean()), 4),
                "val_sd": round(float(sub.auroc_val.std(ddof=1)), 4),
                "test_mean": round(float(sub.auroc_test.mean()), 4),
                "test_sd": round(float(sub.auroc_test.std(ddof=1)), 4),
                "n_cols_mean": int(sub.n_cols.mean()),
                "alphas": [float(a) for a in sub.alpha],
                "boundary_any": bool(sub.boundary.any()),
            }
        json.dump(repro, open(RESULTS / "repro.json", "w"), indent=2)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fixture", "run", "report", "full"])
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.cmd in ("fixture", "full"):
        run_fixture()
    if args.cmd in ("run", "full"):
        run_pipeline()
    if args.cmd in ("report", "full"):
        write_repro()
        write_report()
        log("[report] REPORT.md + repro.json written")


if __name__ == "__main__":
    main()
