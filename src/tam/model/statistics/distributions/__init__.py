# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-License-Identifier: LGPL-3.0-or-later
# Author : Yann Allioux

"""
Joint (multivariate) distributional models.

The univariate distributional/mixture models are folded into StaticTAM; only the multivariate copula
joiner is a standalone class here.
"""

from .copula import GaussianCopulaTAM

__all__ = ["GaussianCopulaTAM"]
