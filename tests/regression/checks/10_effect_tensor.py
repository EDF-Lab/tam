# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tensor product effect: spline by spline, linear by Fourier, categorical by spline."""
from common import BASE


def run(res):
    res.static("spline_spline", "load ~ te(s(temperature, k=6), s(toy, k=6)) + " + BASE)
    res.static("linear_fourier", "load ~ te(l(temperature), f(toy, m=4, s=1, cyclic=True)) + " + BASE)
    res.static("categorical_spline", "load ~ te(c(day_type_week, n_cat=7, topo='nominal'), s(temperature, k=6)) + " + BASE)
    res.static("with_main_effects", "load ~ te(s(temperature, k=6), s(toy, k=6)) + s(temperature, k=6) + " + BASE)
    res.static("penalty_weak", "load ~ te(s(temperature, k=6), s(toy, k=6), ap=-10) + " + BASE)
    res.static("three_margins", "load ~ te(s(temperature, k=4), s(toy, k=4), l(load_d7)) + " + BASE)
    res.static("fourier_margin", "load ~ te(s(temperature, k=5), f(toy, m=3, s=1, cyclic=True)) + " + BASE)
    res.static("extrapolate_constant", "load ~ te(s(temperature, k=5), s(toy, k=5), extrapolate='constant') + " + BASE)
    res.static("penalty_strong", "load ~ te(s(temperature, k=5), s(toy, k=5), ap=2) + " + BASE)
    res.static("chebyshev_margin", "load ~ te(p(temperature, deg=4), s(toy, k=5)) + " + BASE)
    res.static("two_tensors", "load ~ te(s(temperature, k=4), s(toy, k=4)) + te(s(temperature, k=4), l(load_d7)) + " + BASE)
