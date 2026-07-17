# SPDX-FileCopyrightText: 2025-2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
TAM Evaluation Module
"""

from .tracker import BenchmarkTracker
from .metrics import calculate_regression_metrics
from .performance_analyzer import analyze_residuals, detect_temporal_degradation
from .eval_plotting import plot_benchmark_dashboard, generate_summary_table

__all__ = [
    "BenchmarkTracker",
    "calculate_regression_metrics",
    "analyze_residuals",
    "detect_temporal_degradation",
    "plot_benchmark_dashboard",
    "generate_summary_table"
]