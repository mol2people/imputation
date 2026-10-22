"""Process-level runtime configuration.

The worker pool parallelizes across participants; BLAS-level threading inside
each worker only oversubscribes the machine.  These variables must be set
before NumPy/SciPy load their native libraries, so import this module before
``numpy`` or ``pandas`` in every entrypoint.

This environment's NumPy 2.4 is built against Apple Accelerate
(``ACCELERATE_NEW_LAPACK``), not OpenBLAS.  On macOS the relevant cap is
``VECLIB_MAXIMUM_THREADS``; OpenBLAS/MKL variables are set only on other
platforms.
"""
from __future__ import annotations

import os
import sys

_COMMON_VARS = ("OMP_NUM_THREADS",)
_APPLE_VARS = ("VECLIB_MAXIMUM_THREADS",)
_OTHER_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_single_threaded_blas() -> None:
    names = _COMMON_VARS + (_APPLE_VARS if sys.platform == "darwin" else _OTHER_VARS)
    for name in names:
        os.environ.setdefault(name, "1")
