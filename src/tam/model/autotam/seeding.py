# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

r"""
Reproducibility of the AutoTAM search: ``AutoTAM(..., seed=42)``.

The search draws from the global generators of ``random``, ``numpy`` and ``torch`` (about thirty draws in the knowledge graph, the population nodes and
the evolutionary loop). With a seed, ``fit()`` seeds the three generators, runs, and puts them back as they were found: two fits with the same seed
give the same experts whatever was drawn before, and the program around the fit is not disturbed. Without a seed nothing is touched.
"""

import functools
import logging
import os
import platform
import random
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from tam.model.autotam.failures import log_failure

logger = logging.getLogger(__name__)

#: <seeding_rules>
@contextmanager
def seeded(seed: Optional[int]):
    """Seeds ``random``, ``numpy`` and ``torch`` for the duration of the block and restores their state afterwards.

    Not thread-safe: the generators are global, and the AutoTAM pipeline is single-threaded.

    Args:
        seed: The seed, or ``None`` to leave the generators alone.
    """
    if seed is None:
        yield
        return
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def seeded_method(method):
    """Decorator: runs a method of an object that has a ``seed`` attribute inside ``seeded(self.seed)``."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with seeded(getattr(self, "seed", None)):
            return method(self, *args, **kwargs)
    return wrapper


def rank_key(score: float, label: str) -> Tuple[float, str]:
    """Key that ranks candidates by their score rounded to 10 significant digits, then by their label.

    Two scores that differ only in the last digits (the processors round differently) tie, and the tie is broken by the label, so the ranking does
    not depend on the order the candidates arrived in nor on the machine.

    Args:
        score: Score of the candidate (lower is better).
        label: Stable name of the candidate, its formula or its expert name.

    Returns:
        The sort key.
    """
    return (float(f"{score:.9e}"), label)


def environment_info() -> Dict[str, Any]:
    """The environment of a run: versions, threads, linear-algebra backend and processor, as plain values for the run info file.

    The same seed gives the same result in the same environment; two environments can differ in the last bits of a sum (another BLAS kernel,
    another thread count, another version) and a search that compares candidates may then pick another champion. Recording the environment
    makes such a difference traceable. Every entry that cannot be read is ``None``.
    """
    import pandas
    import scipy

    def safe(read):
        try:
            return read()
        except (OSError, KeyError, ValueError, AttributeError, TypeError, ImportError) as exc:
            log_failure(logger, "Reading the environment for the run info", exc, level=logging.DEBUG)
            return None

    def blas() -> Optional[str]:
        config = np.show_config(mode="dicts")
        found = config.get("Build Dependencies", {}).get("blas", {})
        return " ".join(str(found.get(key)) for key in ("name", "version") if found.get(key)) or None

    def cpu_flags() -> Optional[list]:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("flags"):
                    return sorted(flag for flag in line.split(":", 1)[1].split() if flag in ("avx", "avx2", "fma", "avx512f", "sse4_2"))
        return None

    return {
        "python": platform.python_version(), "platform": platform.platform(), "processor": safe(platform.processor),
        "numpy": np.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(), "cpu_count": os.cpu_count(), "blas": safe(blas), "cpu_flags": safe(cpu_flags),
        "cuda": bool(torch.cuda.is_available()), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_coretype": os.environ.get("OPENBLAS_CORETYPE"),
    }
#: </seeding_rules>
