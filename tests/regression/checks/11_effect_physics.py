# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Universal physics effect: penalty by differential operators."""
EXPERIMENTAL = True
from common import BASE


def run(res):
    res.static("d1", "load ~ phys(toy, basis='spline', k=10, D1=0.5, D2=0.0) + " + BASE)
    res.static("d2", "load ~ phys(toy, basis='spline', k=10, D1=0.0, D2=1.0) + " + BASE)
    res.static("d1_d2", "load ~ phys(toy, basis='spline', k=12, D1=0.5, D2=1.0) + " + BASE)
    res.static("temperature", "load ~ phys(temperature, basis='spline', k=10, D1=0.5, D2=1.0) + " + BASE)
    res.static("d1_only_explicit", "load ~ phys(toy, basis='spline', k=10, D1=1.0) + " + BASE)
    res.static("d2_only_explicit", "load ~ phys(toy, basis='spline', k=10, D2=1.0) + " + BASE)
    res.static("d3", "load ~ phys(toy, basis='spline', k=10, D3=1.0) + " + BASE)
    res.static("d1_d2_d3", "load ~ phys(toy, basis='spline', k=10, D1=0.5, D2=1.0, D3=0.5) + " + BASE)
    res.static("d0", "load ~ phys(toy, basis='spline', k=10, D0=1.0) + " + BASE)
    res.static("default_weights", "load ~ phys(toy, basis='spline', k=10) + " + BASE)
    res.static("fourier_basis", "load ~ phys(toy, basis='fourier', n_coeffs=8, D1=0.5, D2=1.0) + " + BASE)
    res.static("fourier_basis_default", "load ~ phys(toy, basis='fourier', n_coeffs=12) + " + BASE)
    res.static("spline_k5", "load ~ phys(toy, basis='spline', k=5, D2=1.0) + " + BASE)
    res.static("spline_k20", "load ~ phys(toy, basis='spline', k=20, D2=1.0) + " + BASE)
    res.static("strong_weights", "load ~ phys(toy, basis='spline', k=10, D1=10.0, D2=10.0) + " + BASE)
    res.static("weak_weights", "load ~ phys(toy, basis='spline', k=10, D1=0.001, D2=0.001) + " + BASE)
    res.static("extrapolate_constant", "load ~ phys(toy, basis='spline', k=10, D2=1.0, extrapolate='constant') + " + BASE)
    res.static("penalty_weak", "load ~ phys(toy, basis='spline', k=10, D2=1.0, ap=-12) + " + BASE)
    res.static("d0_penalised", "load ~ phys(toy, basis='spline', k=12, D0=1.0, ap=-1) + " + BASE)
    res.static("d1_penalised", "load ~ phys(toy, basis='spline', k=12, D1=1.0, ap=-1) + " + BASE)
    res.static("d2_penalised", "load ~ phys(toy, basis='spline', k=12, D2=1.0, ap=-1) + " + BASE)
    res.static("d3_penalised", "load ~ phys(toy, basis='spline', k=12, D3=1.0, ap=-1) + " + BASE)
    res.static("d1_d2_penalised", "load ~ phys(toy, basis='spline', k=12, D1=0.5, D2=1.0, ap=-1) + " + BASE)
    res.static("strong_penalised", "load ~ phys(toy, basis='spline', k=12, D1=10.0, D2=10.0, ap=-1) + " + BASE)
    res.static("fourier_d1_penalised", "load ~ phys(toy, basis='fourier', n_coeffs=8, D1=1.0, ap=-1) + " + BASE)
