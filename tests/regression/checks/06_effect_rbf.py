# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Radial basis effect: centres, bandwidth, interaction with other features, seeded centres."""
from common import BASE


def run(res):
    res.static("c10", "load ~ rbf(temperature, n_centers=10, gamma=0.1, seed=0) + " + BASE)
    res.static("c30", "load ~ rbf(temperature, n_centers=30, gamma=0.1, seed=0) + " + BASE)
    res.static("gamma1", "load ~ rbf(temperature, n_centers=20, gamma=1.0, seed=0) + " + BASE)
    res.static("seed1", "load ~ rbf(temperature, n_centers=20, gamma=0.1, seed=1) + " + BASE)
    res.static("others", "load ~ rbf(temperature, others='toy', n_centers=30, gamma=0.1, seed=0) + " + BASE)
    res.static("nu0.5", "load ~ rbf(temperature, n_centers=20, nu=0.5, seed=0) + " + BASE)
    res.static("nu1.5", "load ~ rbf(temperature, n_centers=20, nu=1.5, seed=0) + " + BASE)
    res.static("nu2.5", "load ~ rbf(temperature, n_centers=20, nu=2.5, seed=0) + " + BASE)
    res.static("gamma_default", "load ~ rbf(temperature, n_centers=20, seed=0) + " + BASE)
    res.static("others_two", "load ~ rbf(temperature, others='toy|load_d7', n_centers=20, gamma=0.5, seed=0) + " + BASE)
    res.static("extrapolate_constant", "load ~ rbf(temperature, n_centers=20, gamma=0.1, seed=0, extrapolate='constant') + " + BASE)
    res.static("extrapolate_linear", "load ~ rbf(temperature, n_centers=20, gamma=0.1, seed=0, extrapolate='linear') + " + BASE)
    res.static("extrapolate_saturation", "load ~ rbf(temperature, n_centers=20, gamma=0.1, seed=0, extrapolate='saturation') + " + BASE)
    res.static("penalty_weak", "load ~ rbf(temperature, n_centers=20, gamma=0.1, seed=0, ap=-12) + " + BASE)
