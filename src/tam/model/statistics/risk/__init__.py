# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Uncertainty & risk: post-fit statistical bounds that build on the static conformal engine (SafetyTAM).

Layers: streaming ACI (aci), conformalized distributional wrapper (conformal), epistemic parameter
uncertainty (uncertainty) and Extreme Value Theory tail scoring (extremes).
"""

from .aci import adaptive_conformal_scores, adaptive_conformal_intervals, update_risk_level
from .conformal import ConformalDistributionalTAM
from .uncertainty import posterior_prediction
from .extremes import GeneralizedParetoTail, fit_gpd_tail

__all__ = [
    "adaptive_conformal_scores",
    "adaptive_conformal_intervals",
    "update_risk_level",
    "ConformalDistributionalTAM",
    "posterior_prediction",
    "GeneralizedParetoTail",
    "fit_gpd_tail",
]
