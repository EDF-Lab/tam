# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Interactions with a neural effect: the ``others`` parameter of ``n()`` (a categorical partner, a continuous one, two partners)."""
EXPERIMENTAL = True
from common import BASE

FAMILIES = {
    "neural_others_weekday": "n(temperature, others='day_type_week', n_neurons=10, act='relu')",
    "neural_others_toy": "n(temperature, others='toy', n_neurons=10, act='relu')",
    "neural_others_two": "n(temperature, others='day_type_week|toy', n_neurons=10, act='tanh')",
    "neural_without_others": "n(temperature, n_neurons=10, act='relu')",
}


def run(res):
    for label, effect in FAMILIES.items():
        res.static(label, "load ~ " + effect + " + " + BASE)
