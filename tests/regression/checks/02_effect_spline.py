# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""P-spline effect: number of knots, degree, penalty order, the three extrapolation modes."""
from common import BASE


def run(res):
    res.static("k6", "load ~ s(temperature, k=6) + " + BASE)
    res.static("k12", "load ~ s(temperature, k=12) + " + BASE)
    res.static("k20", "load ~ s(temperature, k=20) + " + BASE)
    res.static("deg2", "load ~ s(temperature, k=10, deg=2) + " + BASE)
    res.static("deg3_p1", "load ~ s(temperature, k=10, deg=3, p=1) + " + BASE)
    res.static("deg3_p3", "load ~ s(temperature, k=10, deg=3, p=3) + " + BASE)
    res.static("extrapolate_continue", "load ~ s(temperature, k=10, extrapolate='continue') + " + BASE)
    res.static("extrapolate_constant", "load ~ s(temperature, k=10, extrapolate='constant') + " + BASE)
    res.static("extrapolate_linear", "load ~ s(temperature, k=10, extrapolate='linear') + " + BASE)
    res.static("two_splines", "load ~ s(temperature, k=8) + s(toy, k=8) + " + BASE)
    res.static("penalty_weak", "load ~ s(temperature, k=10, ap=-12) + " + BASE)
    res.static("penalty_strong", "load ~ s(temperature, k=10, ap=1) + " + BASE)
    res.static("extrapolate_saturation", "load ~ s(temperature, k=10, extrapolate='saturation') + " + BASE)
    res.static("k4_deg1", "load ~ s(temperature, k=4, deg=1) + " + BASE)
    res.static("deg3_p2_k10_explicit", "load ~ s(temperature, k=10, deg=3, p=2) + " + BASE)
    res.static("penalty_order1_penalised", "load ~ s(temperature, k=12, deg=3, p=1, ap=-1) + " + BASE)
    res.static("penalty_order2_penalised", "load ~ s(temperature, k=12, deg=3, p=2, ap=-1) + " + BASE)
    res.static("penalty_order3_penalised", "load ~ s(temperature, k=12, deg=3, p=3, ap=-1) + " + BASE)
    res.static("degree2_penalised", "load ~ s(temperature, k=12, deg=2, p=1, ap=-1) + " + BASE)
