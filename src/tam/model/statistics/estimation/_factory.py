# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
The strategy registry (mirror of spectrum/_factory.py).

build_strategy maps a loss name to a ReweightingStrategy; adding a loss is one file
plus one line here. build_strategy("l2") is the Gaussian identity (== ordinary least squares).
"""
from __future__ import annotations

from ._base_strategy import ReweightingStrategy
from ._glm import gaussian_family, gamma_family, poisson_family, binomial_family
from ._expectile import ExpectileLoss
from ._robust import HuberLoss, StudentTLoss

_STRATEGY_ALIASES = {
    "l2": gaussian_family, "gaussian": gaussian_family, "normal": gaussian_family,
    "gamma": gamma_family, "poisson": poisson_family, "binomial": binomial_family,
}


#: <build_strategy>
def build_strategy(loss: str, tau: float = 0.5, nu: float = 4.0, delta: float = 1.345) -> ReweightingStrategy:
    """Map a loss name to a ReweightingStrategy."""
    key = loss.lower().replace("-", "_")
    if key in _STRATEGY_ALIASES:
        return _STRATEGY_ALIASES[key]()
    if key == "expectile":
        return ExpectileLoss(tau)
    if key == "huber":
        return HuberLoss(delta)
    if key in ("student_t", "studentt", "student"):
        return StudentTLoss(nu)
    raise ValueError(
        f"Unknown loss '{loss}'. Supported: l2/gaussian, gamma, poisson, binomial, expectile, huber, student_t."
    )
#: </build_strategy>
