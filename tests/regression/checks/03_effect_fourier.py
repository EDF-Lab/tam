# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fourier effect: number of harmonics, smoothness, cyclic or not."""
from common import BASE


def run(res):
    res.static("m3", "load ~ f(toy, m=3, s=1, cyclic=True) + " + BASE)
    res.static("m6", "load ~ f(toy, m=6, s=1, cyclic=True) + " + BASE)
    res.static("m12", "load ~ f(toy, m=12, s=1, cyclic=True) + " + BASE)
    res.static("smoothness2", "load ~ f(toy, m=6, s=2, cyclic=True) + " + BASE)
    res.static("not_cyclic", "load ~ f(temperature, m=6, s=1, cyclic=False) + " + BASE)
    res.static("cyclic_and_not", "load ~ f(toy, m=6, s=1, cyclic=True) + f(temperature, m=4, s=1, cyclic=False) + " + BASE)
    res.static("scaled1", "load ~ f(toy, m=6, s=1, cyclic=True, scaled=1.0) + " + BASE)
    res.static("scaled_two_pi", "load ~ f(toy, m=6, s=1, cyclic=True, scaled=6.283185307179586) + " + BASE)
    res.static("not_cyclic_scaled1", "load ~ f(temperature, m=6, s=1, cyclic=False, scaled=1.0) + " + BASE)
    res.static("smoothness0", "load ~ f(toy, m=6, s=0, cyclic=True) + " + BASE)
    res.static("smoothness3", "load ~ f(toy, m=6, s=3, cyclic=True) + " + BASE)
    res.static("extrapolate_constant", "load ~ f(temperature, m=4, s=1, cyclic=False, extrapolate='constant') + " + BASE)
    res.static("extrapolate_linear", "load ~ f(temperature, m=4, s=1, cyclic=False, extrapolate='linear') + " + BASE)
    res.static("extrapolate_saturation", "load ~ f(temperature, m=4, s=1, cyclic=False, extrapolate='saturation') + " + BASE)
    res.static("penalty_weak", "load ~ f(toy, m=6, s=1, cyclic=True, ap=-12) + " + BASE)
    res.static("smoothness0_penalised", "load ~ f(toy, m=8, s=0, cyclic=True, ap=-1) + " + BASE)
    res.static("smoothness1_penalised", "load ~ f(toy, m=8, s=1, cyclic=True, ap=-1) + " + BASE)
    res.static("smoothness2_penalised", "load ~ f(toy, m=8, s=2, cyclic=True, ap=-1) + " + BASE)
    res.static("smoothness3_penalised", "load ~ f(toy, m=8, s=3, cyclic=True, ap=-1) + " + BASE)
