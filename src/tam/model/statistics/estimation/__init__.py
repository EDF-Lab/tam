# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Estimation: the "how" behind every non-Gaussian target.
"""

from ._base_strategy import ReweightingStrategy
from ._glm import (
    Link,
    IdentityLink,
    LogLink,
    LogitLink,
    GLMFamily,
    gaussian_family,
    gamma_family,
    poisson_family,
    binomial_family,
)
from ._expectile import ExpectileLoss, empirical_expectile, expectile_level_for_quantile
from ._robust import HuberLoss, StudentTLoss
from ._factory import build_strategy

__all__ = [
    # GLM links
    "Link",
    "IdentityLink",
    "LogLink",
    "LogitLink",
    # Reweighting strategies (GLM families + M-estimators)
    "ReweightingStrategy",
    "GLMFamily",
    "ExpectileLoss",
    "HuberLoss",
    "StudentTLoss",
    # Family constructors
    "gaussian_family",
    "gamma_family",
    "poisson_family",
    "binomial_family",
    # Factory
    "build_strategy",
    # Expectile calibration
    "empirical_expectile",
    "expectile_level_for_quantile",
]
