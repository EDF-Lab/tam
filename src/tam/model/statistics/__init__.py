# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Public API for the Statistic Methods.
"""

from .estimation import (
    ReweightingStrategy,
    GLMFamily,
    ExpectileLoss,
    HuberLoss,
    StudentTLoss,
    build_strategy,
    empirical_expectile,
    expectile_level_for_quantile,
)
from .distributions import GaussianCopulaTAM

# The static conformal engine lives at the model root; re-exported here for convenience.
from ..safety import SafetyTAM
from .risk import (
    ConformalDistributionalTAM,
    adaptive_conformal_scores,
    adaptive_conformal_intervals,
    posterior_prediction,
    GeneralizedParetoTail,
    fit_gpd_tail,
)

__all__ = [
    "ReweightingStrategy",
    "GLMFamily",
    "ExpectileLoss",
    "HuberLoss",
    "StudentTLoss",
    "build_strategy",
    "empirical_expectile",
    "expectile_level_for_quantile",
    "GaussianCopulaTAM",
    "SafetyTAM",
    "ConformalDistributionalTAM",
    "adaptive_conformal_scores",
    "adaptive_conformal_intervals",
    "posterior_prediction",
    "GeneralizedParetoTail",
    "fit_gpd_tail",
]
