# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tree effect: number of trees, depth, interaction with other features, seeded."""
EXPERIMENTAL = True
from common import BASE


def run(res):
    res.static("t4_d2", "load ~ t(temperature, n_trees=4, max_depth=2, seed=42) + " + BASE)
    res.static("t8_d3", "load ~ t(temperature, n_trees=8, max_depth=3, seed=42) + " + BASE)
    res.static("t16_d4", "load ~ t(temperature, n_trees=16, max_depth=4, seed=42) + " + BASE)
    res.static("seed7", "load ~ t(temperature, n_trees=8, max_depth=3, seed=7) + " + BASE)
    res.static("others", "load ~ t(temperature, others='toy', n_trees=8, max_depth=3, seed=42) + " + BASE)
    res.static("max_leaves", "load ~ t(temperature, n_trees=8, max_depth=4, max_leaves=6, seed=42) + " + BASE)
    res.static("sp_alpha1", "load ~ t(temperature, n_trees=8, max_depth=3, seed=42, sp_alpha=1.0) + " + BASE)
    res.static("split_quantile", "load ~ t(temperature, n_trees=8, max_depth=3, seed=42, split_strategy='quantile') + " + BASE)
    res.static("split_uniform", "load ~ t(temperature, n_trees=8, max_depth=3, seed=42, split_strategy='uniform') + " + BASE)
    res.static("depth1", "load ~ t(temperature, n_trees=8, max_depth=1, seed=42) + " + BASE)
    res.static("one_tree", "load ~ t(temperature, n_trees=1, max_depth=4, seed=42) + " + BASE)
    res.static("others_two", "load ~ t(temperature, others='toy|load_d7', n_trees=8, max_depth=3, seed=42) + " + BASE)
    res.static("extrapolate_constant", "load ~ t(temperature, n_trees=8, max_depth=3, seed=42, extrapolate='constant') + " + BASE)
    res.static("penalty_weak", "load ~ t(temperature, n_trees=8, max_depth=3, seed=42, ap=-12) + " + BASE)
    res.static("sp_alpha0_penalised", "load ~ t(temperature, n_trees=8, max_depth=3, seed=42, sp_alpha=0.0, ap=-1) + " + BASE)
    res.static("sp_alpha1_penalised", "load ~ t(temperature, n_trees=8, max_depth=3, seed=42, sp_alpha=1.0, ap=-1) + " + BASE)
