#!/usr/bin/env python
"""TEMPORAL stage 0 — vendor pinned references + fixture.

PLAN §4: clone the two pinned commits into `cache/vendor/`, record their
commit SHA and the SHA256 of the used files, sys.path shim, then run a
fixture on 288-length series (the production input length) to confirm
import, JIT compile, expected widths, byte-identical repeat and
batch-vs-unbatched transform agreement for both MultiRocket and HYDRA.

Outputs:
  cache/vendor_repro.json   pins + file SHAs + fixture measurements
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# reproducible: cap threads BEFORE importing numba/torch.
os.environ.setdefault("NUMBA_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
# torch (pip) ships its own libomp; conda's llvmlite/numba ships another.
# Default numba threading layer "omp" links libomp → dual-load segfault on
# numba prange when torch is imported in the same process. Use numba's
# built-in "workqueue" (no libomp touch) — prange with independent
# per-iteration writes is byte-identical across 1/4/8 threads (verified).
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # belt-and-suspenders
import numba  # noqa: E402
numba.set_num_threads(8)
torch.set_num_threads(8)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
VENDOR = CACHE / "vendor"
REPRO = CACHE / "vendor_repro.json"

# frozen pins (TEMPORAL_PLAN.md §4, frozen 2026-09-21)
MR_REPO = "ChangWeiTan/MultiRocket"
MR_PIN = "3ccaa4f8ed11769b904af885e17018c39c89a2ac"
MR_DIR = VENDOR / "multirocket"
MR_FILE = MR_DIR / "multirocket" / "multirocket.py"

HYDRA_REPO = "angus924/hydra"
HYDRA_PIN = "144bb7aa3186654042f00f36b078370c620acb06"
HYDRA_DIR = VENDOR / "hydra"
HYDRA_FILE = HYDRA_DIR / "code" / "hydra.py"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_repo(url_slug: str, target: Path, pin: str, name: str) -> str:
    """Clone if missing; if present, verify HEAD == pin; else reset to pin."""
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet",
                        f"https://github.com/{url_slug}.git", str(target)],
                       check=True)
    head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if head != pin:
        subprocess.run(["git", "-C", str(target), "fetch", "--quiet", "--all"],
                       check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--quiet", pin],
                       check=True)
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
        if head != pin:
            raise SystemExit(f"{name}: HEAD {head} != pin {pin}")
    print(f"  [vendor] {name} @ {head}", flush=True)
    return head


def fixture_multirocket() -> dict:
    """Module-level fit/transform fixture on 288-length series.

    Confirms width (~9,408/day with num_features=1250 per transformation),
    byte-identical repeat (same seed → identical bits), and batch-vs-
    unbatched transform agreement (transforming one example vs N examples
    of the same data must match).
    """
    sys.path.insert(0, str(MR_DIR))  # package root: `multirocket` package lives here
    from multirocket.multirocket import fit as mr_fit, transform as mr_transform
    from numba import njit

    # ERRATUM (recorded in repro): numba's np.random.randint inside njit uses
    # numba's internal per-thread RNG state, NOT numpy's global RNG. Python-side
    # np.random.seed() does not reset it. This njit shim seeds the same state
    # that _fit_biases' np.random.randint draws from (parallel=False → main
    # thread), preserving the plan's invariant: seeded, reproducible biases.
    @njit(cache=False)
    def _nb_seed(s):
        np.random.seed(s)

    rng = np.random.default_rng(20260921)
    # fixture: 10 synthetic days, 288-length, float64
    n, L = 10, 288
    X = rng.standard_normal((n, L)).astype(np.float64)
    X1 = np.diff(X, axis=1)  # first-order diff

    # ---- width: fit two transformations (base + diff1) with num_features=1250
    seed = 314159
    _nb_seed(seed)
    params_base = mr_fit(X, num_features=1250, max_dilations_per_kernel=32)
    _nb_seed(seed + 1)
    params_diff = mr_fit(X1, num_features=1250, max_dilations_per_kernel=32)
    # Warm-up (JIT): one transform on small batch, excluded from timing.
    _ = mr_transform(X[:2], X1[:2], params_base, params_diff, n_features_per_kernel=4)

    t0 = time.perf_counter()
    F_full = mr_transform(X, X1, params_base, params_diff, n_features_per_kernel=4)
    t_full = time.perf_counter() - t0

    # byte-identical repeat (re-seed numba RNG before each fit)
    _nb_seed(seed)
    params_base2 = mr_fit(X, num_features=1250, max_dilations_per_kernel=32)
    _nb_seed(seed + 1)
    params_diff2 = mr_fit(X1, num_features=1250, max_dilations_per_kernel=32)
    F_full2 = mr_transform(X, X1, params_base2, params_diff2, n_features_per_kernel=4)
    byte_repeat = bool(np.array_equal(F_full, F_full2))

    # batch-vs-unbatched: transform 1 example in a batch of 1 vs 10
    F_single = mr_transform(X[:1], X1[:1], params_base, params_diff,
                            n_features_per_kernel=4)
    batch_vs_single = bool(np.array_equal(F_single, F_full[:1]))

    # nan_to_num safety
    F_nan_safe = np.nan_to_num(F_full)

    return {
        "n_examples": n, "input_length": L, "num_features_per_transformation": 1250,
        "width_day": int(F_full.shape[1]),
        "width_day_per_transformation_base": int(F_full.shape[1] // 4 // 2),
        "byte_identical_repeat": byte_repeat,
        "batch_vs_unbatched_agreement": batch_vs_single,
        "any_nan": bool(np.isnan(F_full).any()),
        "any_inf": bool(np.isinf(F_full).any()),
        "nan_to_num_zero_frac": float((F_nan_safe == 0).mean()),
        "transform_wall_s_warmup_excluded": round(t_full, 4),
        "max_abs": float(np.max(np.abs(F_full))),
    }


def fixture_hydra() -> dict:
    """Hydra(k=8, g=64) + SparseScaler fixture on 288-length series."""
    from hydra import Hydra, SparseScaler  # type: ignore

    rng = np.random.default_rng(20260921)
    n, L = 10, 288
    # README: input is torch.FloatTensor, shape = (num_examples, 1, length)
    X = torch.from_numpy(rng.standard_normal((n, L)).astype(np.float32)).unsqueeze(1)
    torch.manual_seed(20260921)
    h = Hydra(input_length=L, k=8, g=64, seed=20260921)
    _ = h.batch(X[:2], batch_size=2)  # JIT warm-up
    t0 = time.perf_counter()
    F = h.batch(X, batch_size=8)
    t = time.perf_counter() - t0

    # byte-identical repeat
    torch.manual_seed(20260921)
    h2 = Hydra(input_length=L, k=8, g=64, seed=20260921)
    F2 = h2.batch(X, batch_size=8)
    byte_repeat = bool(torch.equal(F, F2))

    # batch-vs-unbatched: single example
    F_one = h.batch(X[:1], batch_size=1)
    batch_vs_single = bool(torch.equal(F_one, F[:1]))

    # SparseScaler fit/transform on these features
    ss = SparseScaler(mask=True, exponent=4)
    Z = ss.fit_transform(F)  # torch tensor
    ss2 = SparseScaler(mask=True, exponent=4)
    ss2.fit(F)
    Z2 = ss2.transform(F)
    ss_byte_repeat = bool(torch.equal(Z, Z2))

    return {
        "n_examples": n, "input_length": L, "k": 8, "g": 64,
        "width_day": int(F.shape[1]),
        "width_day_expected": 6 * 2 * 32 * 8 * 2,  # = 6144
        "byte_identical_repeat": byte_repeat,
        "batch_vs_unbatched_agreement": batch_vs_single,
        "sparse_scaler_byte_repeat": ss_byte_repeat,
        "any_nan": bool(torch.isnan(F).any().item()),
        "any_inf": bool(torch.isinf(F).any().item()),
        "transform_wall_s_warmup_excluded": round(t, 4),
        "max_abs": float(F.abs().max().item()),
    }


def main() -> None:
    t0 = time.perf_counter()
    print(f"[vendor_setup] start | python {sys.version.split()[0]} | "
          f"numpy {np.__version__} | torch {torch.__version__}", flush=True)

    # torch + numba coexistence smoke: JIT-compile and run a tiny prange
    # kernel in this torch-loaded process (segfaulted before workqueue fix).
    from numba import njit, prange

    @njit(fastmath=True, parallel=True, cache=False)
    def _smoke(X):
        n, m = X.shape
        out = np.zeros(n)
        for i in prange(n):
            s = 0.0
            for j in range(m):
                s += X[i, j]
            out[i] = s
        return out

    _smoke(np.zeros((2, 2)))
    print(f"  [vendor] torch+numba prange smoke OK "
          f"(layer={numba.config.THREADING_LAYER})", flush=True)

    mr_head = ensure_repo(MR_REPO, MR_DIR, MR_PIN, "MultiRocket")
    hydra_head = ensure_repo(HYDRA_REPO, HYDRA_DIR, HYDRA_PIN, "HYDRA")

    # sys.path shims before any vendored import
    sys.path.insert(0, str(HYDRA_DIR / "code"))
    sys.path.insert(0, str(MR_DIR))

    mr_sha = sha256_file(MR_FILE)
    hydra_sha = sha256_file(HYDRA_FILE)
    print(f"  [vendor] mr_file sha256   {mr_sha[:16]}…", flush=True)
    print(f"  [vendor] hydra_file sha256 {hydra_sha[:16]}…", flush=True)

    print("  [vendor] fixture: MultiRocket module-level fit/transform", flush=True)
    mr_meas = fixture_multirocket()
    print(f"    width/day={mr_meas['width_day']} "
          f"(per-trans base={mr_meas['width_day_per_transformation_base']}) | "
          f"repeat={mr_meas['byte_identical_repeat']} "
          f"batch-vs-single={mr_meas['batch_vs_unbatched_agreement']} "
          f"any_nan={mr_meas['any_nan']} wall={mr_meas['transform_wall_s_warmup_excluded']}s",
          flush=True)

    print("  [vendor] fixture: HYDRA Hydra(k=8,g=64) + SparseScaler", flush=True)
    hydra_meas = fixture_hydra()
    print(f"    width/day={hydra_meas['width_day']} (expected {hydra_meas['width_day_expected']}) | "
          f"repeat={hydra_meas['byte_identical_repeat']} "
          f"batch-vs-single={hydra_meas['batch_vs_unbatched_agreement']} "
          f"sparse-byte-repeat={hydra_meas['sparse_scaler_byte_repeat']} "
          f"any_nan={hydra_meas['any_nan']} wall={hydra_meas['transform_wall_s_warmup_excluded']}s",
          flush=True)

    # width gates
    gates = {
        "mr_width_9408": mr_meas["width_day"] == 9408,
        "mr_byte_repeat": mr_meas["byte_identical_repeat"],
        "mr_batch_vs_single": mr_meas["batch_vs_unbatched_agreement"],
        "mr_no_nan": not mr_meas["any_nan"] and not mr_meas["any_inf"],
        "hydra_width_6144": hydra_meas["width_day"] == 6144,
        "hydra_byte_repeat": hydra_meas["byte_identical_repeat"],
        "hydra_batch_vs_single": hydra_meas["batch_vs_unbatched_agreement"],
        "hydra_no_nan": not hydra_meas["any_nan"] and not hydra_meas["any_inf"],
    }
    print(f"  [vendor] gates: {gates}", flush=True)
    if not all(gates.values()):
        raise SystemExit(f"VENDOR FIXTURE GATE FAILED: {gates}")

    repro = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "numba": numba.__version__,
        "pin": {
            "multirocket_repo": MR_REPO,
            "multirocket_commit": mr_head,
            "multirocket_file_sha256": mr_sha,
            "multirocket_file_path": str(MR_FILE.relative_to(HERE)),
            "hydra_repo": HYDRA_REPO,
            "hydra_commit": hydra_head,
            "hydra_file_sha256": hydra_sha,
            "hydra_file_path": str(HYDRA_FILE.relative_to(HERE)),
        },
        "fixture_multirocket": mr_meas,
        "fixture_hydra": hydra_meas,
        "gates": gates,
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(repro, open(REPRO, "w"), indent=2, default=str)
    print(f"[vendor_setup] wrote {REPRO.relative_to(HERE)} | "
          f"wall {repro['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
