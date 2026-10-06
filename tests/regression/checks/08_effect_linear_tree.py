# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Linear tree effect (varying-coefficient leaves): number of leaves."""
EXPERIMENTAL = True
from common import BASE


def run(res):
    res.static("leaves4", "load ~ lt(temperature, max_leaves=4) + " + BASE)
    res.static("leaves8", "load ~ lt(temperature, max_leaves=8) + " + BASE)
    res.static("leaves16", "load ~ lt(temperature, max_leaves=16) + " + BASE)
    res.static("two_features", "load ~ lt(temperature, max_leaves=6) + lt(toy, max_leaves=6) + " + BASE)
    res.static("max_depth", "load ~ lt(temperature, max_leaves=8, max_depth=3) + " + BASE)
    res.static("slope_other_feature", "load ~ lt(temperature, slope='load_d1', max_leaves=6) + " + BASE)
    res.static("sp_alpha1", "load ~ lt(temperature, max_leaves=6, sp_alpha=1.0) + " + BASE)
    res.static("split_quantile", "load ~ lt(temperature, max_leaves=6, split_strategy='quantile') + " + BASE)
    res.static("seed7", "load ~ lt(temperature, max_leaves=6, seed=7) + " + BASE)
    res.static("extrapolate_constant", "load ~ lt(temperature, max_leaves=6, extrapolate='constant') + " + BASE)
    res.static("penalty_weak", "load ~ lt(temperature, max_leaves=6, ap=-12) + " + BASE)
