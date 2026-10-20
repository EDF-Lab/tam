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
import random
from contextlib import contextmanager
from typing import Optional, Tuple

import numpy as np
import torch

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
#: </seeding_rules>
