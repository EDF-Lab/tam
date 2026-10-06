# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""PID effect on the lagged load: window and derivative penalty."""
from common import BASE


def run(res):
    res.static("w6", "load ~ pid(load_d7, w=6, d_pen=10.0) + " + BASE)
    res.static("w24", "load ~ pid(load_d7, w=24, d_pen=10.0) + " + BASE)
    res.static("w24_d0", "load ~ pid(load_d7, w=24, d_pen=0.0) + " + BASE)
    res.static("w24_d100", "load ~ pid(load_d7, w=24, d_pen=100.0) + " + BASE)
    res.static("w1", "load ~ pid(load_d7, w=1, d_pen=10.0) + " + BASE)
    res.static("w2", "load ~ pid(load_d7, w=2, d_pen=10.0) + " + BASE)
    res.static("w3", "load ~ pid(load_d7, w=3, d_pen=10.0) + " + BASE)
    res.static("w48", "load ~ pid(load_d7, w=48, d_pen=10.0) + " + BASE)
    res.static("w7_default", "load ~ pid(load_d7) + " + BASE)
    res.static("d_pen_0_w6", "load ~ pid(load_d7, w=6, d_pen=0) + " + BASE)
    res.static("d_pen_negative_small", "load ~ pid(load_d7, w=6, d_pen=0.001) + " + BASE)
    res.static("d_pen_1000", "load ~ pid(load_d7, w=6, d_pen=1000.0) + " + BASE)
    res.static("extrapolate_constant", "load ~ pid(load_d7, w=6, d_pen=10.0, extrapolate='constant') + " + BASE)
    res.static("extrapolate_linear", "load ~ pid(load_d7, w=6, d_pen=10.0, extrapolate='linear') + " + BASE)
    res.static("penalty_weak", "load ~ pid(load_d7, w=6, d_pen=10.0, ap=-12) + " + BASE)
    res.static("penalty_strong", "load ~ pid(load_d7, w=6, d_pen=10.0, ap=2) + " + BASE)
    res.static("d_pen_0_penalised", "load ~ pid(load_d7, w=6, d_pen=0, ap=-2) + " + BASE)
    res.static("d_pen_1.0_penalised", "load ~ pid(load_d7, w=6, d_pen=1.0, ap=-2) + " + BASE)
    res.static("d_pen_10.0_penalised", "load ~ pid(load_d7, w=6, d_pen=10.0, ap=-2) + " + BASE)
    res.static("d_pen_100.0_penalised", "load ~ pid(load_d7, w=6, d_pen=100.0, ap=-2) + " + BASE)
    res.static("w1_penalised", "load ~ pid(load_d7, w=1, d_pen=10.0, ap=-2) + " + BASE)
    res.static("w3_penalised", "load ~ pid(load_d7, w=3, d_pen=10.0, ap=-2) + " + BASE)
    res.static("w12_penalised", "load ~ pid(load_d7, w=12, d_pen=10.0, ap=-2) + " + BASE)
