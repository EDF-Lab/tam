# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Public namespace for the tam.model package - the root layer.

The root holds the core linear algebra (the P-WLS atom in _base / _math / _dispatcher), the top-level
Meta-Models (the user-facing API), and the model-agnostic static conformal engine SafetyTAM. The
model-agnostic statistical schedules and the advanced risk layers live one level down in tam.model.statistics.

This re-exports the root Meta-Models and SafetyTAM so they import directly from tam.model, and - for the flat
legacy paths after the statistics classes moved into nested packages - the relocated classes as thin aliases.
"""

# --- Root Meta-Models (the top-level user API) ---
from .additive import StaticTAM
from .adaptative import AdaptiveTAM
from .kalman import KalmanTAM
from .opera import OperaTAM
from .hierarchical import HierarchicalTAM
from .neural import NeuralTAM

# --- Root static conformal engine ---
from .safety import SafetyTAM

# --- Statistics-layer classes (aliased here for convenience / legacy flat paths) ---
from .statistics.risk.conformal import ConformalDistributionalTAM
from .statistics.risk.extremes import GeneralizedParetoTail, fit_gpd_tail
from .statistics.risk.uncertainty import posterior_prediction
from .statistics.distributions.copula import GaussianCopulaTAM
from .statistics.estimation import build_strategy

__all__ = [
    # Meta-Models
    "StaticTAM",
    "AdaptiveTAM",
    "KalmanTAM",
    "OperaTAM",
    "HierarchicalTAM",
    "NeuralTAM",
    # Static conformal engine
    "SafetyTAM",
    # Statistics-layer aliases
    "ConformalDistributionalTAM",
    "GeneralizedParetoTail",
    "fit_gpd_tail",
    "posterior_prediction",
    "GaussianCopulaTAM",
    "build_strategy",
]
