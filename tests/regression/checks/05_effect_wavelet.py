# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wavelet effect: scales and locations."""
from common import BASE


def run(res):
    res.static("s3_l8", "load ~ w(temperature, n_scales=3, n_locations=8) + " + BASE)
    res.static("s4_l12", "load ~ w(temperature, n_scales=4, n_locations=12) + " + BASE)
    res.static("s5_l16", "load ~ w(temperature, n_scales=5, n_locations=16) + " + BASE)
    res.static("toy", "load ~ w(toy, n_scales=4, n_locations=12) + " + BASE)
    res.static("two_features", "load ~ w(temperature, n_scales=3, n_locations=8) + w(toy, n_scales=3, n_locations=8) + " + BASE)
    res.static("extrapolate_constant", "load ~ w(temperature, n_scales=3, n_locations=8, extrapolate='constant') + " + BASE)
    res.static("extrapolate_linear", "load ~ w(temperature, n_scales=3, n_locations=8, extrapolate='linear') + " + BASE)
    res.static("extrapolate_saturation", "load ~ w(temperature, n_scales=3, n_locations=8, extrapolate='saturation') + " + BASE)
    res.static("one_scale", "load ~ w(temperature, n_scales=1, n_locations=12) + " + BASE)
    res.static("penalty_weak", "load ~ w(temperature, n_scales=3, n_locations=8, ap=-12) + " + BASE)
    res.static("penalty_strong", "load ~ w(temperature, n_scales=3, n_locations=8, ap=2) + " + BASE)
    res.static("scales_penalised", "load ~ w(temperature, n_scales=4, n_locations=10, ap=-1) + " + BASE)
