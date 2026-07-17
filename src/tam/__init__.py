# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Time series Additive Model (TAM)
The unified framework for interpretable, physics-informed, and high-performance time series forecasting.
"""

from .model.additive import StaticTAM
from .model.opera import OperaTAM
from .model.kalman import KalmanTAM
from .model.adaptative import AdaptiveTAM
from .model.hierarchical import HierarchicalTAM
from .model.neural import NeuralTAM
from .model.safety import SafetyTAM
from .model.autotam.auto_tam import AutoTAM
from .evaluation.tracker import BenchmarkTracker 

__version__ = "1.2.6"

__all__ = [
    "StaticTAM",
    "OperaTAM",
    "KalmanTAM",
    "AdaptiveTAM",
    "HierarchicalTAM",
    "NeuralTAM",
    "SafetyTAM",
    "AutoTAM",
    "BenchmarkTracker",
    "__version__"
]