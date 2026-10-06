# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fixed penalties: the smoothing weight of an effect, the model-wide default, and the number of coefficients of the fit."""
from common import BASE


def run(res):
    for ap in (-12, -8, -4, 0, 4):
        res.static(f"spline_ap{ap}", f"load ~ s(temperature, k=10, ap={ap}) + " + BASE)
    for default in (-12.0, -9.0, -5.0):
        res.static(f"default_alpha_p{default:g}", "load ~ s(temperature, k=10) + s(toy, k=8) + " + BASE, default_alpha_p=default)

    def coefficients():
        model = res.static("shape", "load ~ s(temperature, k=10) + " + BASE)
        res.value("shape.coefficients", tuple(model.coefficients_.shape))
        res.value("shape.groups", len(model.unique_groups_))

    res.attempt("shape", coefficients)
