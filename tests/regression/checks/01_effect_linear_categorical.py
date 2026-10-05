# SPDX-FileCopyrightText: 2026 EDF (Electricité De France)
# SPDX-FileContributor: Yann Allioux
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Linear and categorical effects: fixed penalties, the topologies, the number of levels given or inferred. Also the difference order of the ordinal topology."""
from common import BASE


def run(res):
    res.static("linear_one", "load ~ l(temperature) + " + BASE)
    res.static("linear_three", "load ~ l(temperature) + l(load_d7) + l(toy) + " + BASE)
    res.static("linear_weak_penalty", "load ~ l(temperature, ap=-20) + " + BASE)
    res.static("linear_strong_penalty", "load ~ l(temperature, ap=3) + " + BASE)
    res.static("nominal", "load ~ c(month, n_cat=12, topo='nominal') + " + BASE)
    res.static("ordinal", "load ~ c(month, n_cat=12, topo='ordinal') + " + BASE)
    res.static("fourier_topology", "load ~ c(month, n_cat=12, topo='fourier') + " + BASE)
    res.static("levels_inferred", "load ~ c(month, topo='nominal') + " + BASE)
    res.static("weekday_only", "load ~ l(temperature) + c(day_type_jf, n_cat=2, topo='nominal') + " + BASE)
    res.static("ordinal_difference_order1", "load ~ c(month, n_cat=12, topo='ordinal', p_order=1) + " + BASE)
    res.static("ordinal_difference_order2", "load ~ c(month, n_cat=12, topo='ordinal', p_order=2) + " + BASE)
    res.static("ordinal_difference_order3", "load ~ c(month, n_cat=12, topo='ordinal', p_order=3) + " + BASE)
    res.static("fourier_topology_weekday", "load ~ c(day_type_week, n_cat=7, topo='fourier') + " + BASE)
    res.static("ordinal_weekday", "load ~ c(day_type_week, n_cat=7, topo='ordinal') + " + BASE)
    res.static("nominal_penalty_strong", "load ~ c(month, n_cat=12, topo='nominal', ap=2) + " + BASE)
    res.static("nominal_penalty_weak", "load ~ c(month, n_cat=12, topo='nominal', ap=-15) + " + BASE)
    res.static("extrapolate_constant", "load ~ c(month, n_cat=12, topo='nominal', extrapolate='constant') + " + BASE)
    res.static("extrapolate_linear", "load ~ c(month, n_cat=12, topo='nominal', extrapolate='linear') + " + BASE)
    res.static("ordinal_order1_penalised", "load ~ c(month, n_cat=12, topo='ordinal', p_order=1, ap=-1) + " + BASE)
    res.static("ordinal_order2_penalised", "load ~ c(month, n_cat=12, topo='ordinal', p_order=2, ap=-1) + " + BASE)
    res.static("ordinal_order3_penalised", "load ~ c(month, n_cat=12, topo='ordinal', p_order=3, ap=-1) + " + BASE)
    res.static("nominal_penalised", "load ~ c(month, n_cat=12, topo='nominal', ap=-1) + " + BASE)
    res.static("fourier_topology_penalised", "load ~ c(month, n_cat=12, topo='fourier', ap=-1) + " + BASE)
