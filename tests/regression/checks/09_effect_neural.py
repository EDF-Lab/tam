# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Neural effect solved in closed form inside a StaticTAM: width, activation, depth, interaction."""
EXPERIMENTAL = True
from common import BASE


def run(res):
    res.static("relu", "load ~ n(temperature, n_neurons=16, act='relu', n_hidden_layers=1) + " + BASE)
    res.static("tanh", "load ~ n(temperature, n_neurons=16, act='tanh', n_hidden_layers=1) + " + BASE)
    res.static("cos", "load ~ n(temperature, n_neurons=16, act='cos', n_hidden_layers=1) + " + BASE)
    res.static("wide", "load ~ n(temperature, n_neurons=48, act='relu', n_hidden_layers=1) + " + BASE)
    res.static("deep", "load ~ n(temperature, n_neurons=16, act='relu', n_hidden_layers=2) + " + BASE)
    res.static("others", "load ~ n(temperature, others='toy', n_neurons=16, act='relu', n_hidden_layers=1) + " + BASE)
    res.static("seed1", "load ~ n(temperature, n_neurons=16, act='relu', seed=1) + " + BASE)
    res.static("seed42", "load ~ n(temperature, n_neurons=16, act='relu', seed=42) + " + BASE)
    res.static("three_layers", "load ~ n(temperature, n_neurons=16, act='relu', n_hidden_layers=3) + " + BASE)
    res.static("tanh_deep", "load ~ n(temperature, n_neurons=16, act='tanh', n_hidden_layers=2) + " + BASE)
    res.static("cos_deep", "load ~ n(temperature, n_neurons=16, act='cos', n_hidden_layers=2) + " + BASE)
    res.static("others_two", "load ~ n(temperature, others='toy|load_d7', n_neurons=16, act='relu') + " + BASE)
    res.static("extrapolate_constant", "load ~ n(temperature, n_neurons=16, act='relu', extrapolate='constant') + " + BASE)
    res.static("extrapolate_continue", "load ~ n(temperature, n_neurons=16, act='relu', extrapolate='continue') + " + BASE)
    res.static("penalty_weak", "load ~ n(temperature, n_neurons=16, act='relu', ap=-12) + " + BASE)
    res.static("default_width", "load ~ n(temperature, act='relu') + " + BASE)
