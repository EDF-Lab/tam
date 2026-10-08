# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
The strategy registry (mirror of spectrum/_factory.py).

build_strategy maps a loss name to a ReweightingStrategy; adding a loss is one file
plus one line here. build_strategy("l2") is the Gaussian identity (== ordinary least squares).
"""
from __future__ import annotations

from typing import Optional

from ._base_strategy import ReweightingStrategy
from ._glm import gaussian_family, gamma_family, binomial_family
from ._count_families import negative_binomial_family, poisson_family_with_quantiles, tweedie_family
from ._expectile import ExpectileLoss
from ._robust import HuberLoss, StudentTLoss

_STRATEGY_ALIASES = {
    "l2": gaussian_family, "gaussian": gaussian_family, "normal": gaussian_family,
    "gamma": gamma_family, "poisson": poisson_family_with_quantiles, "binomial": binomial_family,
}


#: <build_strategy>
def build_strategy(loss: str, tau: float = 0.5, nu: float = 4.0, delta: float = 1.345,
                   power: float = 1.5, dispersion: Optional[float] = None) -> ReweightingStrategy:
    """Map a loss name to a ReweightingStrategy.

    ``power`` is the Tweedie power (1 < power < 2); ``dispersion`` fixes the negative binomial alpha (estimated when None).
    """
    key = loss.lower().replace("-", "_")
    if key in _STRATEGY_ALIASES:
        return _STRATEGY_ALIASES[key]()
    if key in ("tweedie", "compound_poisson"):
        return tweedie_family(power)
    if key in ("negative_binomial", "negbin", "nb"):
        return negative_binomial_family(dispersion)
    if key == "expectile":
        return ExpectileLoss(tau)
    if key == "huber":
        return HuberLoss(delta)
    if key in ("student_t", "studentt", "student"):
        return StudentTLoss(nu)
    raise ValueError(
        f"Unknown loss '{loss}'. Supported: l2/gaussian, gamma, poisson, negative_binomial, tweedie, binomial, expectile, huber, student_t."
    )
#: </build_strategy>
