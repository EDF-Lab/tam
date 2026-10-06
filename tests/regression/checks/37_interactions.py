# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Interactions: the ``others`` parameter of the radial basis effect (one and two partner features), and tensor products of splines."""
from common import BASE

FAMILIES = {
    "rbf_others_two": "rbf(temperature, others='toy|load_d1', n_centers=20, gamma=0.5)",
    "rbf_others_one": "rbf(temperature, others='toy', n_centers=20, gamma=0.5)",
    "rbf_others_seeded": "rbf(temperature, others='toy|load_d1', n_centers=20, gamma=0.5, seed=7)",
    "rbf_without_others": "rbf(temperature, n_centers=20, gamma=0.5)",
    "tensor_spline_spline": "te(s(temperature, k=5), s(toy, k=5))",
    "tensor_with_main_effects": "te(s(temperature, k=5), s(toy, k=5)) + s(temperature, k=5) + s(toy, k=5)",
    "tensor_three_margins": "te(s(temperature, k=4), s(toy, k=4), s(load_d1, k=4))",
    "tensor_penalised": "te(s(temperature, k=5), s(toy, k=5), ap=-2)",
}


def run(res):
    for label, effect in FAMILIES.items():
        res.static(label, "load ~ " + effect + " + " + BASE)
