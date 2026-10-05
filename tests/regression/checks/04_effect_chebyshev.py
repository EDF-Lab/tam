# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Chebyshev polynomial effect: degree and smoothness."""
from common import BASE


def run(res):
    res.static("deg3", "load ~ p(temperature, deg=3) + " + BASE)
    res.static("deg6", "load ~ p(temperature, deg=6) + " + BASE)
    res.static("deg10", "load ~ p(temperature, deg=10) + " + BASE)
    res.static("smoothness2", "load ~ p(temperature, deg=8, s=2) + " + BASE)
    res.static("two_features", "load ~ p(temperature, deg=5) + p(toy, deg=5) + " + BASE)
    res.static("smoothness0", "load ~ p(temperature, deg=8, s=0) + " + BASE)
    res.static("smoothness1", "load ~ p(temperature, deg=8, s=1) + " + BASE)
    res.static("extrapolate_continue", "load ~ p(temperature, deg=6, extrapolate='continue') + " + BASE)
    res.static("extrapolate_constant", "load ~ p(temperature, deg=6, extrapolate='constant') + " + BASE)
    res.static("extrapolate_linear", "load ~ p(temperature, deg=6, extrapolate='linear') + " + BASE)
    res.static("extrapolate_saturation", "load ~ p(temperature, deg=6, extrapolate='saturation') + " + BASE)
    res.static("penalty_weak", "load ~ p(temperature, deg=6, ap=-12) + " + BASE)
    res.static("deg1", "load ~ p(temperature, deg=1) + " + BASE)
    res.static("smoothness0_penalised", "load ~ p(temperature, deg=10, s=0, ap=-1) + " + BASE)
    res.static("smoothness1_penalised", "load ~ p(temperature, deg=10, s=1, ap=-1) + " + BASE)
    res.static("smoothness2_penalised", "load ~ p(temperature, deg=10, s=2, ap=-1) + " + BASE)
